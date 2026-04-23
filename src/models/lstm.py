"""
Bidirectional LSTM with soft-attention pooling — delta (log-ratio) prediction.

Core design change for MAPE / sMAPE / R²:
  Instead of predicting log1p(y[t+h]) directly, the model predicts the
  log-ratio delta:  Δ[h] = log1p(y[t+h]) - log1p(y[t])

  Recovery:  ŷ[t+h] = expm1( log1p(y[t]) + Δ̂[h] )

  Why this helps:
    • Summer troughs (y≈100) no longer produce 200% MAPE when the absolute
      prediction is off by 300 cases — the model only needs to predict a small
      positive/negative ratio change, not the raw scale.
    • The target is naturally zero-mean and bounded (±2 log-units covers
      virtually all week-over-week flu changes), making learning stable.
    • Recovery is exact — no inverse-scaler zero-fill approximation needed.
    • SEQ_LEN stays at 52 — long sequences (104) cause gradient vanishing
      through BiLSTM backprop, collapsing predictions to the training mean.

Other design choices:
  • Single StandardScaler on the full feature matrix for input (X) only.
  • Huber loss on delta targets — robust to COVID-era extreme week-over-week
    spikes while still penalising large ratio errors.
  • Dropout + L2 regularisation per Amendolara et al. (2023).
  • BatchNorm omitted from recurrent blocks (disrupts temporal dependencies).
"""

import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.config import (
    FEATURE_COLS, HORIZON, LSTM_BATCH, LSTM_EPOCHS, LSTM_LR, LSTM_UNITS, LSTM_UNITS2,
    MIN_TRAIN, SEQ_LEN,
)

# Fix all random seeds for reproducibility across runs
_SEED = 42
os.environ["PYTHONHASHSEED"] = str(_SEED)
np.random.seed(_SEED)
tf.random.set_seed(_SEED)
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

_TARGET      = "ilitotal"
_TARGET_IDX  = FEATURE_COLS.index(_TARGET)
_N_FEATURES  = len(FEATURE_COLS)
_LSTM_UNITS  = LSTM_UNITS
_LSTM_UNITS2 = LSTM_UNITS2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_lstm_model() -> keras.Model:
    """
    BiLSTM(128) → Dropout → BiLSTM(64) → Dropout → soft-attention → Dense.

    Output: HORIZON-dimensional delta vector (log-ratio changes).
    Loss  : Huber — smooth for small deltas, linear for COVID-spike outliers.

    Input  shape: (SEQ_LEN, N_FEATURES)
    Output shape: (HORIZON,)
    """
    reg    = keras.regularizers.l2(1e-4)
    inputs = keras.layers.Input(shape=(SEQ_LEN, _N_FEATURES))

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(_LSTM_UNITS, return_sequences=True,
                          kernel_regularizer=reg, recurrent_regularizer=reg)
    )(inputs)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(_LSTM_UNITS2, return_sequences=True, kernel_regularizer=reg)
    )(x)
    x = keras.layers.Dropout(0.2)(x)

    # Soft-attention over the sequence
    hidden_dim = _LSTM_UNITS2 * 2          # 128
    attn = keras.layers.Dense(1, activation="tanh")(x)
    attn = keras.layers.Flatten()(attn)
    attn = keras.layers.Activation("softmax")(attn)
    attn = keras.layers.RepeatVector(hidden_dim)(attn)
    attn = keras.layers.Permute([2, 1])(attn)
    x    = keras.layers.Multiply()([x, attn])
    x    = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(HORIZON)(x)   # no activation — regression

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss="huber",
    )
    return model


# ---------------------------------------------------------------------------
# Sequence builders
# ---------------------------------------------------------------------------

def _sequence_weights(dates: np.ndarray, seq_len: int, n_seqs: int) -> np.ndarray:
    """
    Per-sequence training weights based on epidemic regime.

    COVID-suppression sequences are heavily down-weighted: near-zero ILI during
    2020-2021 would otherwise teach the model "low ILI → stay low", which is wrong
    for the post-COVID test period where flu returned with full seasonal amplitude.

    Anchor date = the last date in the input window (position seq_len - 1 + j).
    """
    COVID_START = np.datetime64("2020-03-15")
    COVID_END   = np.datetime64("2021-09-01")
    anchor_idx = np.clip(np.arange(seq_len - 1, seq_len - 1 + n_seqs), 0, len(dates) - 1)
    ad = dates[anchor_idx]
    w = np.ones(n_seqs, dtype=float)
    w[ad < COVID_START] = 0.5   # pre-COVID: informative but less relevant
    w[(ad >= COVID_START) & (ad <= COVID_END)] = 0.1  # aberrant zero-flu period
    return w


def _make_delta_sequences(
    scaled_X: np.ndarray,
    log_y:    np.ndarray,
    seq_len:  int,
    horizon:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Δ) training pairs.

    X  : scaled feature windows of shape (seq_len, n_features).
    Δ  : log-ratio targets — log_y[i:i+horizon] - log_y[i-1], shape (horizon,).
         Δ[h] = how much does log1p(ilitotal) change in h steps from position i-1.
    """
    X, y = [], []
    limit = len(scaled_X) - horizon
    for i in range(seq_len, limit + 1):
        X.append(scaled_X[i - seq_len : i])
        last_log = log_y[i - 1]                      # anchor = last known value
        y.append(log_y[i : i + horizon] - last_log)  # deltas from that anchor
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_lstm(
    df: pd.DataFrame,
    test_weeks: int,
) -> tuple[dict, dict, dict, keras.Model | None, StandardScaler | None]:
    """
    Train two LSTM instances (eval and future) with delta prediction.

    Returns
    -------
    eval_preds   : {state: {week_starts, actual, pred}}
    future_preds : {state: {week_starts, pred}}
    metrics      : {state: compute_metrics dict}
    future_model : trained Keras model (all data)
    future_scaler: fitted StandardScaler (all data) — for external use
    """
    # -----------------------------------------------------------------------
    # Collect per-state arrays
    # -----------------------------------------------------------------------
    state_data: dict[str, tuple] = {}
    for state, sdf in df.groupby("state"):
        sdf = sdf.sort_values("week_start").reset_index(drop=True)
        if len(sdf) < SEQ_LEN + test_weeks + MIN_TRAIN + HORIZON:
            continue

        feat  = sdf[FEATURE_COLS].values.astype(float)
        log_y = np.log1p(np.maximum(feat[:, _TARGET_IDX], 0.0))  # unscaled log
        feat[:, _TARGET_IDX] = log_y                              # replace raw with log

        dates = pd.to_datetime(sdf["week_start"]).values
        state_data[state] = (dates, feat, log_y)

    if not state_data:
        logger.warning("No states had enough data for LSTM training")
        return {}, {}, {}, None, None

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=40, restore_best_weights=True,
            min_delta=5e-5, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=12, min_lr=1e-7, verbose=0
        ),
    ]

    split = -(test_weeks + HORIZON)   # e.g. -56 for test_weeks=52

    # -----------------------------------------------------------------------
    # Eval model — train split only
    # -----------------------------------------------------------------------
    eval_scaler = StandardScaler()
    eval_scaler.fit(np.vstack([feat[:split] for _, feat, _ in state_data.values()]))

    X_e, y_e, w_e = [], [], []
    for _, (dates, feat, log_y) in state_data.items():
        Xt, yt = _make_delta_sequences(
            eval_scaler.transform(feat[:split]),
            log_y[:split],
            SEQ_LEN, HORIZON,
        )
        X_e.append(Xt); y_e.append(yt)
        w_e.append(_sequence_weights(dates[:split], SEQ_LEN, len(Xt)))

    eval_model = build_lstm_model()
    eval_model.fit(
        np.vstack(X_e), np.vstack(y_e),
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
        validation_split=0.1, callbacks=callbacks, verbose=0,
        sample_weight=np.concatenate(w_e),
    )
    logger.info("LSTM eval model trained on %d sequences", sum(len(x) for x in X_e))

    # -----------------------------------------------------------------------
    # Future model — all data
    # -----------------------------------------------------------------------
    future_scaler = StandardScaler()
    future_scaler.fit(np.vstack([feat for _, feat, _ in state_data.values()]))

    X_f, y_f, w_f = [], [], []
    for _, (dates, feat, log_y) in state_data.items():
        Xt, yt = _make_delta_sequences(
            future_scaler.transform(feat),
            log_y,
            SEQ_LEN, HORIZON,
        )
        X_f.append(Xt); y_f.append(yt)
        w_f.append(_sequence_weights(dates, SEQ_LEN, len(Xt)))

    future_model = build_lstm_model()
    future_model.fit(
        np.vstack(X_f), np.vstack(y_f),
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
        validation_split=0.1, callbacks=callbacks, verbose=0,
        sample_weight=np.concatenate(w_f),
    )
    logger.info("LSTM future model trained on %d sequences", sum(len(x) for x in X_f))

    # -----------------------------------------------------------------------
    # Per-state evaluation + future forecast
    # -----------------------------------------------------------------------
    eval_preds:   dict = {}
    future_preds: dict = {}
    metrics:      dict = {}

    for state, (dates, feat, log_y) in state_data.items():
        scaled_eval = eval_scaler.transform(feat)
        test_start  = len(feat) - test_weeks - HORIZON
        actual_list, pred_list, date_list = [], [], []

        for i in range(0, test_weeks, HORIZON):
            ctx_end   = test_start + i + HORIZON
            ctx_start = ctx_end - SEQ_LEN
            if ctx_start < 0 or ctx_end >= len(scaled_eval):
                break

            # Delta prediction
            delta_hat = eval_model.predict(
                scaled_eval[ctx_start:ctx_end][np.newaxis], verbose=0
            )[0]                                    # (HORIZON,) log-ratio deltas

            last_log  = log_y[ctx_end - 1]          # anchor: last known log1p value
            actual_log = log_y[ctx_end : ctx_end + HORIZON]
            n_avail    = min(len(actual_log), HORIZON)

            # Recover ilitotal: expm1(anchor + predicted_delta)
            pred_vals   = np.maximum(np.expm1(last_log + delta_hat[:n_avail]),   0.0)
            actual_vals = np.maximum(np.expm1(actual_log[:n_avail]),              0.0)

            pred_list.extend(pred_vals)
            actual_list.extend(actual_vals)
            date_list.extend(dates[ctx_end : ctx_end + n_avail])

        eval_preds[state] = {
            "week_starts": date_list,
            "actual": np.array(actual_list),
            "pred":   np.array(pred_list),
        }
        metrics[state] = compute_metrics(np.array(actual_list), np.array(pred_list))

        # Future forecast
        sf_feat     = future_scaler.transform(feat)
        delta_future = future_model.predict(sf_feat[-SEQ_LEN:][np.newaxis], verbose=0)[0]
        last_log_fut = log_y[-1]
        pred_future  = np.maximum(np.expm1(last_log_fut + delta_future), 0.0)

        last_date = pd.Timestamp(dates[-1])
        future_preds[state] = {
            "week_starts": [last_date + pd.Timedelta(weeks=w + 1) for w in range(HORIZON)],
            "pred": pred_future,
        }

    return eval_preds, future_preds, metrics, future_model, future_scaler
