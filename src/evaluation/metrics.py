"""Evaluation metrics and stationarity diagnostics."""

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, MAPE, sMAPE, and R² between actual and predicted arrays.

    MAPE excludes weeks where actual < 10 to prevent blow-up on near-zero values.
    sMAPE (symmetric MAPE) is robust to cases where actual ≈ 0.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(p)
    a, p = a[mask], p[mask]
    if len(a) < 2:
        return {"rmse": None, "mae": None, "mape": None, "smape": None, "r2": None}

    rmse  = float(np.sqrt(mean_squared_error(a, p)))
    mae   = float(mean_absolute_error(a, p))
    nz    = a >= 10
    mape  = float(np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100) if nz.any() else None
    denom = (np.abs(a) + np.abs(p)) / 2
    smape = float(np.mean(np.where(denom > 0, np.abs(a - p) / denom, 0)) * 100)
    r2    = float(r2_score(a, p))
    return {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape, "r2": r2}


def agg_metrics(state_metrics: dict) -> dict:
    """Average per-state metric dicts into a single summary dict."""
    if not state_metrics:
        return {}
    keys = ["rmse", "mae", "mape", "smape", "r2"]
    return {
        k: round(float(np.nanmean([v[k] for v in state_metrics.values() if v[k] is not None])), 4)
        for k in keys
    }


def adf_test(series: np.ndarray) -> dict:
    """
    Augmented Dickey-Fuller stationarity test.

    Returns adf_stat, p_value, and a boolean `stationary` (p < 0.05).
    Used as a pre-modeling diagnostic following Zheng et al. (2024).
    """
    try:
        result = adfuller(series[~np.isnan(series)], autolag="AIC")
        return {
            "adf_stat":   float(result[0]),
            "p_value":    float(result[1]),
            "stationary": bool(result[1] < 0.05),
        }
    except Exception as exc:
        logger.warning("ADF test failed: %s", exc)
        return {"adf_stat": None, "p_value": None, "stationary": None}
