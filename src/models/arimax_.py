"""ARIMAX model: SARIMAX with exogenous environmental + lag features."""

import logging

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

# Fallback order chain — richer AR/MA specs tried first, simpler ones as fallback.
# Seasonality is encoded via month_sin/cos and week_sin/cos in exog (plus
# explicit lag_13/26/52 features) rather than a seasonal SARIMAX order,
# avoiding the numerical instability of s=52 on weekly data.
_ORDER_CHAIN = [
    ((3, 1, 2), (0, 0, 0, 0)),
    ((2, 1, 2), (0, 0, 0, 0)),
    ((2, 1, 1), (0, 0, 0, 0)),
    ((1, 1, 2), (0, 0, 0, 0)),
    ((1, 1, 1), (0, 0, 0, 0)),
    ((1, 1, 0), (0, 0, 0, 0)),
]


def fit_arimax(
    train_y: np.ndarray,
    train_exog: np.ndarray,
    forecast_exog: np.ndarray,
) -> np.ndarray:
    """
    Fit ARIMAX on log1p(ilitotal) and return inverse-transformed forecasts.

    The exogenous matrix must contain only features that are known at ALL
    recursive forecast steps (i.e. no lag_k with k ≤ HORIZON).  The caller
    is responsible for supplying the correct exog columns.

    Parameters
    ----------
    train_y       : 1-D array of historical ILI counts.
    train_exog    : 2-D array of exogenous features aligned with train_y.
    forecast_exog : 2-D array of exogenous features for the forecast horizon.

    Returns
    -------
    np.ndarray of length len(forecast_exog), clipped to ≥ 0.
    Returns an array of NaN if all order attempts fail.
    """
    log_y = np.log1p(np.maximum(np.asarray(train_y).flatten(), 0.0))
    train_exog    = np.asarray(train_exog)
    forecast_exog = np.asarray(forecast_exog)

    for order, seasonal_order in _ORDER_CHAIN:
        try:
            fit = SARIMAX(
                log_y, exog=train_exog,
                order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=400, method="lbfgs")
            log_pred = np.asarray(
                fit.forecast(steps=len(forecast_exog), exog=forecast_exog)
            ).flatten()
            return np.maximum(np.expm1(log_pred), 0.0)
        except Exception as exc:
            logger.debug("ARIMAX order %s failed: %s", order, exc)

    logger.warning("All ARIMAX orders failed — returning NaN array")
    return np.full(len(forecast_exog), np.nan)
