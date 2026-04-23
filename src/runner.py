"""
Orchestrator: runs all models, merges results, saves CSVs.

run_forecast() is the single entry point used by both the CLI and the API.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    ARIMAX_EXOG, HORIZON, MIN_TRAIN, OUTPUT_DIR, XGB_COLS,
)
from src.data.features import monthly_env_averages, project_future_env
from src.data.loader import load_environmental
from src.evaluation.metrics import adf_test, agg_metrics, compute_metrics
from src.models.arimax import fit_arimax
from src.models.ensemble import apply_weighted_ensemble, compute_ensemble_weights
from src.models.holt_winters import fit_holt_winters
from src.models.lstm import run_lstm
from src.models.xgboost_model import (
    fit_xgboost,
    fit_xgboost_for_future,
    get_feature_importance,
)

logger = logging.getLogger(__name__)

PRED_COLS = ["arimax_pred", "hw_pred", "xgb_pred", "lstm_pred"]
COL_ORDER = [
    "state", "week_start", "year", "week", "actual",
    "arimax_pred", "hw_pred", "xgb_pred", "lstm_pred", "ensemble_pred",
]


# ---------------------------------------------------------------------------
# Per-state ARIMAX + Holt-Winters + XGBoost (rolling HORIZON-step evaluation)
# ---------------------------------------------------------------------------

def _run_per_state_models(
    df: pd.DataFrame,
    test_weeks: int,
    env_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    monthly_avgs = monthly_env_averages(env_df)
    eval_records:   list = []
    future_records: list = []
    arimax_m: dict  = {}
    hw_m:     dict  = {}
    xgb_m:    dict  = {}
    xgb_imp:  dict  = {}

    states = sorted(df["state"].unique())
    for i, state in enumerate(states, 1):
        sdf = df[df["state"] == state].sort_values("week_start").reset_index(drop=True)
        n   = len(sdf)
        train_end_base = n - test_weeks
        if train_end_base < MIN_TRAIN:
            logger.warning("State %s has too few training weeks (%d) — skipped", state, train_end_base)
            continue

        logger.info("[%d/%d] %s", i, len(states), state)

        all_y    = sdf["ilitotal"].values.astype(float)
        all_exog = sdf[ARIMAX_EXOG].values.astype(float)
        all_xgb  = sdf[XGB_COLS].values.astype(float)
        adf      = adf_test(np.log1p(all_y[:train_end_base]))

        # Post-COVID training window: ARIMAX and HW are trained only on data
        # from 2022-01-02 onwards.  COVID suppression (2020-2021) drove ILI to
        # near-zero and distorts seasonal component estimates; post-2022 seasons
        # are structurally similar to the 2024-2026 test period.
        # Safety floor: always keep at least MIN_TRAIN weeks regardless of cutoff.
        _covid_end = pd.Timestamp("2022-01-02")
        post_covid_idx = int((pd.to_datetime(sdf["week_start"]) < _covid_end).sum())

        actual_all, arimax_all, hw_all, xgb_all = [], [], [], []

        for step in range(0, test_weeks, HORIZON):
            train_end = train_end_base + step
            if train_end >= n:
                break
            end = min(train_end + HORIZON, n)  # clip last chunk to dataset boundary
            h   = end - train_end              # actual steps in this chunk (< HORIZON at tail)

            forecast_exog = all_exog[train_end:end]
            actual        = all_y[train_end:end]

            # ARIMAX: post-COVID data, capped at 3 years — enough history for
            # stable SARIMAX parameter estimation.
            arimax_start = max(post_covid_idx, train_end - 156)
            arimax_start = min(arimax_start, train_end - MIN_TRAIN)

            # HW: exactly 2 recent flu seasons (104 weeks = MIN_TRAIN).
            # Tighter window ensures the seasonal component reflects only the
            # most recent patterns, which best resemble the test period.
            hw_start = max(post_covid_idx, train_end - 104)
            hw_start = min(hw_start, train_end - MIN_TRAIN)

            a_pred  = fit_arimax(all_y[arimax_start:train_end], all_exog[arimax_start:train_end], forecast_exog)
            h_pred  = fit_holt_winters(all_y[hw_start:train_end], h)
            xg_pred = fit_xgboost(all_xgb, all_y, train_end)[:h]

            actual_all.extend(actual)
            arimax_all.extend(a_pred)
            hw_all.extend(h_pred)
            xgb_all.extend(xg_pred)

            for k in range(len(actual)):
                row = sdf.iloc[train_end + k]
                eval_records.append({
                    "state":            state,
                    "week_start":       pd.Timestamp(row["week_start"]).strftime("%Y-%m-%d"),
                    "year":             int(row["year"]),
                    "week":             int(row["week"]),
                    "actual":           float(actual[k]),
                    "arimax_pred":      float(a_pred[k])  if not np.isnan(a_pred[k])  else None,
                    "hw_pred":          float(h_pred[k]) if not np.isnan(h_pred[k]) else None,
                    "xgb_pred":         float(xg_pred[k]) if not np.isnan(xg_pred[k]) else None,
                    "adf_stationary":   adf["stationary"],
                })

        arimax_m[state] = compute_metrics(np.array(actual_all), np.array(arimax_all))
        hw_m[state]     = compute_metrics(np.array(actual_all), np.array(hw_all))
        xgb_m[state]    = compute_metrics(np.array(actual_all), np.array(xgb_all))

        # Feature importance for plotting
        imp = get_feature_importance(all_xgb, all_y, XGB_COLS)
        if imp:
            xgb_imp[state] = imp

        # Future 4-week forecast — all models on full dataset
        last_date    = pd.Timestamp(sdf["week_start"].iloc[-1])
        future_dates = pd.DatetimeIndex([
            last_date + pd.Timedelta(weeks=w) for w in range(1, HORIZON + 1)
        ])
        future_env = project_future_env(
            env_df, state, future_dates, monthly_avgs, historical_ilitotal=all_y
        )

        fut_arimax_start = max(post_covid_idx, n - 156)
        fut_arimax_start = min(fut_arimax_start, n - MIN_TRAIN)
        fut_hw_start = max(post_covid_idx, n - 104)
        fut_hw_start = min(fut_hw_start, n - MIN_TRAIN)
        arimax_future = fit_arimax(all_y[fut_arimax_start:], all_exog[fut_arimax_start:], future_env[ARIMAX_EXOG].values.astype(float))
        hw_future     = fit_holt_winters(all_y[fut_hw_start:], HORIZON)
        xgb_future    = fit_xgboost_for_future(all_xgb, all_y)

        for j, d in enumerate(future_dates):
            iso = d.isocalendar()
            future_records.append({
                "state":       state,
                "week_start":  d.strftime("%Y-%m-%d"),
                "year":        int(iso.year),
                "week":        int(iso.week),
                "actual":      None,
                "arimax_pred": float(arimax_future[j]) if not np.isnan(arimax_future[j]) else None,
                "hw_pred":     float(hw_future[j])     if not np.isnan(hw_future[j])     else None,
                "xgb_pred":    float(xgb_future[j])    if not np.isnan(xgb_future[j])    else None,
            })

    return (
        pd.DataFrame(eval_records),
        pd.DataFrame(future_records),
        {"arimax": arimax_m, "holt_winters": hw_m, "xgboost": xgb_m},
        xgb_imp,
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def rolling_cv_metrics(df: pd.DataFrame, n_splits: int = 3) -> dict:
    """Expanding-window cross-validation for ARIMAX, Holt-Winters, and XGBoost."""
    tss          = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: dict[str, list] = {"arimax": [], "holt_winters": [], "xgboost": []}

    for state in sorted(df["state"].unique()):
        sdf = df[df["state"] == state].sort_values("week_start").reset_index(drop=True)
        if len(sdf) < MIN_TRAIN * 2:
            continue
        y    = sdf["ilitotal"].values.astype(float)
        exog = sdf[ARIMAX_EXOG].values.astype(float)
        xgbX = sdf[XGB_COLS].values.astype(float)

        for train_idx, test_idx in tss.split(y):
            if len(train_idx) < MIN_TRAIN:
                continue
            train_y    = y[train_idx]
            train_exog = exog[train_idx]

            fold_actual, fold_a, fold_h, fold_x = [], [], [], []
            for step in range(0, len(test_idx), HORIZON):
                chunk = test_idx[step : step + HORIZON]
                if not len(chunk):
                    break
                actual  = y[chunk]
                a_pred  = fit_arimax(train_y, train_exog, exog[chunk])
                h_pred  = fit_holt_winters(train_y, len(chunk))
                xg_pred = fit_xgboost(xgbX, y, int(train_idx[-1]) + 1)[:len(chunk)]

                fold_actual.extend(actual)
                fold_a.extend(a_pred)
                fold_h.extend(h_pred)
                fold_x.extend(xg_pred)

                train_y    = y[:chunk[-1] + 1]
                train_exog = exog[:chunk[-1] + 1]

            if fold_actual:
                a = np.array(fold_actual)
                fold_metrics["arimax"].append(compute_metrics(a, np.array(fold_a)))
                fold_metrics["holt_winters"].append(compute_metrics(a, np.array(fold_h)))
                fold_metrics["xgboost"].append(compute_metrics(a, np.array(fold_x)))

    return {
        name: {
            k: round(float(np.nanmean([m[k] for m in folds if m[k] is not None])), 4)
            for k in ["rmse", "mae", "mape", "smape", "r2"]
        }
        for name, folds in fold_metrics.items()
        if folds
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_forecast(
    df: pd.DataFrame,
    test_weeks: int = 52,
    run_cv: bool = True,
) -> dict:
    """
    Train ARIMAX, Holt-Winters, XGBoost, and LSTM; build weighted ensemble.

    Saves
    -----
    output/forecast_eval.csv       — test-period actuals vs all predictions
    output/forecast_next4weeks.csv — next 4 weeks beyond the dataset
    output/xgb_feature_importance.csv
    """
    env_df = load_environmental()

    logger.info("Running ARIMAX + Holt-Winters + XGBoost (%d states)…", df["state"].nunique())
    eval_df, future_df, ps_metrics, xgb_imp = _run_per_state_models(df, test_weeks, env_df)

    logger.info("Running LSTM (all states combined)…")
    lstm_eval, lstm_future, lstm_metrics, _, _ = run_lstm(df, test_weeks)

    # Merge LSTM predictions
    def _merge_lstm(frame: pd.DataFrame, rows: list) -> pd.DataFrame:
        if not rows:
            frame["lstm_pred"] = np.nan
            return frame
        return frame.merge(pd.DataFrame(rows), on=["state", "week_start"], how="left")

    eval_df = _merge_lstm(eval_df, [
        {"state": s, "week_start": pd.Timestamp(d).strftime("%Y-%m-%d"), "lstm_pred": float(p)}
        for s, info in lstm_eval.items()
        for d, p in zip(info["week_starts"], info["pred"])
    ])
    future_df = _merge_lstm(future_df, [
        {"state": s, "week_start": pd.Timestamp(d).strftime("%Y-%m-%d"), "lstm_pred": float(p)}
        for s, info in lstm_future.items()
        for d, p in zip(info["week_starts"], info["pred"])
    ])

    # Ensure all prediction columns exist
    for frame in [eval_df, future_df]:
        for col in PRED_COLS:
            if col not in frame.columns:
                frame[col] = np.nan

    # R²-optimised ensemble weights
    ensemble_weights = compute_ensemble_weights(eval_df, PRED_COLS)
    for frame in [eval_df, future_df]:
        frame["ensemble_pred"] = frame.apply(
            lambda row: apply_weighted_ensemble(
                row, PRED_COLS,
                ensemble_weights.get(row["state"], {c: 0.25 for c in PRED_COLS}),
            ),
            axis=1,
        )

    # Save CSVs
    for frame in [eval_df, future_df]:
        for c in COL_ORDER:
            if c not in frame.columns:
                frame[c] = np.nan

    eval_path   = OUTPUT_DIR / "forecast_eval.csv"
    future_path = OUTPUT_DIR / "forecast_next4weeks.csv"
    eval_df[COL_ORDER].to_csv(eval_path,   index=False)
    future_df[COL_ORDER].to_csv(future_path, index=False)
    logger.info("Saved evaluation CSV  → %s", eval_path)
    logger.info("Saved 4-week ahead    → %s", future_path)

    if xgb_imp:
        imp_df = pd.DataFrame(xgb_imp).T
        imp_df.index.name = "state"
        imp_df.to_csv(OUTPUT_DIR / "xgb_feature_importance.csv")

    # Aggregate metrics
    ensemble_metrics = {
        state: compute_metrics(
            grp["actual"].values.astype(float),
            grp["ensemble_pred"].values.astype(float),
        )
        for state, grp in eval_df.dropna(subset=["actual"]).groupby("state")
    }
    xgb_eval_metrics = {
        state: compute_metrics(
            grp["actual"].values.astype(float),
            grp["xgb_pred"].values.astype(float),
        )
        for state, grp in eval_df.dropna(subset=["actual", "xgb_pred"]).groupby("state")
    }

    cv_summary = {}
    if run_cv:
        logger.info("Running expanding-window cross-validation…")
        cv_summary = rolling_cv_metrics(df)

    return {
        "eval_csv":         str(eval_path),
        "future_csv":       str(future_path),
        "test_weeks":       test_weeks,
        "forecast_horizon": HORIZON,
        "states_evaluated": sorted(eval_df["state"].unique().tolist()),
        "n_states":         int(eval_df["state"].nunique()),
        "ensemble_weights": {
            s: {k: round(v, 4) for k, v in w.items()}
            for s, w in ensemble_weights.items()
        },
        "metrics": {
            "arimax":       agg_metrics(ps_metrics.get("arimax", {})),
            "holt_winters": agg_metrics(ps_metrics.get("holt_winters", {})),
            "xgboost":      agg_metrics(xgb_eval_metrics),
            "lstm":         agg_metrics(lstm_metrics),
            "ensemble":     agg_metrics(ensemble_metrics),
        },
        "cv_metrics": cv_summary,
    }
