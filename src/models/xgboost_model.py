"""
XGBoost direct multi-output forecasting model.

Innovation over Chen et al. (2024): uses the direct strategy — a single
MultiOutputRegressor predicts all HORIZON steps simultaneously from features
at time t, avoiding the compounding errors of recursive single-step forecasting.
"""

import logging

import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

from src.config import HORIZON, MIN_TRAIN, XGB_PARAMS

logger = logging.getLogger(__name__)


def _build_direct_pairs(
    X: np.ndarray,
    y: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (features-at-t, delta-targets-at-t+1..t+HORIZON) training pairs.

    Targets are log-ratio deltas: log1p(y[t+h]) - log1p(y[t]).
    Using deltas rather than absolute log-levels makes the targets stationary
    and lets the model focus on change prediction, with the current level
    (ilitotal_log in X) serving as the recovery anchor at inference time.
    """
    n_pairs = train_end - HORIZON
    X_tr = X[:n_pairs]
    y_tr = np.stack([
        np.log1p(np.maximum(y[i + 1 : i + 1 + HORIZON], 0.0))
        - np.log1p(max(float(y[i]), 0.0))
        for i in range(n_pairs)
    ])   # shape: (n_pairs, HORIZON) — log-ratio deltas from y[i]
    return X_tr, y_tr


def fit_xgboost(
    all_X: np.ndarray,
    all_y: np.ndarray,
    train_end: int,
) -> np.ndarray:
    """
    Fit XGBoost on history up to train_end and predict the next HORIZON steps.

    Features at row train_end-1 (last known observation) are used as the
    forecast input — consistent with the direct forecasting strategy.

    Returns np.ndarray of length HORIZON, ≥ 0.
    """
    X_tr, y_tr = _build_direct_pairs(all_X, all_y, train_end)
    if len(X_tr) < MIN_TRAIN:
        return np.full(HORIZON, np.nan)

    try:
        model = MultiOutputRegressor(xgb.XGBRegressor(**XGB_PARAMS), n_jobs=1)
        model.fit(X_tr, y_tr)
        pred_delta = model.predict(all_X[train_end - 1 : train_end])[0]
        anchor_log = np.log1p(max(float(all_y[train_end - 1]), 0.0))
        return np.maximum(np.expm1(anchor_log + pred_delta), 0.0)
    except Exception as exc:
        logger.warning("XGBoost fit failed: %s", exc)
        return np.full(HORIZON, np.nan)


def fit_xgboost_for_future(
    all_X: np.ndarray,
    all_y: np.ndarray,
) -> np.ndarray:
    """
    Fit XGBoost on the full dataset and predict beyond the last observation.

    Uses the last row of all_X as the forecast feature snapshot.
    Returns np.ndarray of length HORIZON, ≥ 0.
    """
    X_tr, y_tr = _build_direct_pairs(all_X, all_y, len(all_y))
    if len(X_tr) < MIN_TRAIN:
        return np.full(HORIZON, np.nan)

    try:
        model = MultiOutputRegressor(xgb.XGBRegressor(**XGB_PARAMS), n_jobs=1)
        model.fit(X_tr, y_tr)
        pred_delta = model.predict(all_X[-1:])[0]
        anchor_log = np.log1p(max(float(all_y[-1]), 0.0))
        return np.maximum(np.expm1(anchor_log + pred_delta), 0.0)
    except Exception as exc:
        logger.warning("XGBoost future fit failed: %s", exc)
        return np.full(HORIZON, np.nan)


def get_feature_importance(
    all_X: np.ndarray,
    all_y: np.ndarray,
    feature_names: list[str],
) -> dict[str, float] | None:
    """
    Fit XGBoost on the full dataset and return mean gain feature importances,
    averaged across the HORIZON sub-estimators.
    """
    X_tr, y_tr = _build_direct_pairs(all_X, all_y, len(all_y))
    if len(X_tr) < MIN_TRAIN:
        return None

    try:
        model = MultiOutputRegressor(xgb.XGBRegressor(**XGB_PARAMS), n_jobs=1)
        model.fit(X_tr, y_tr)
        importances = np.mean(
            [est.feature_importances_ for est in model.estimators_], axis=0
        )
        return dict(zip(feature_names, importances.tolist()))
    except Exception as exc:
        logger.warning("XGBoost importance extraction failed: %s", exc)
        return None
