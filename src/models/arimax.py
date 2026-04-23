"""ARIMAX model: SARIMAX with exogenous environmental + lag features."""

import logging
import warnings

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
    # Simpler fallbacks for small-count / high-variance series (e.g. DC)
    ((0, 1, 1), (0, 0, 0, 0)),
    ((0, 1, 0), (0, 0, 0, 0)),
    ((1, 0, 1), (0, 0, 0, 0)),
    ((1, 0, 0), (0, 0, 0, 0)),
]

# Secondary optimizers tried when lbfgs fails for a given order.
# powell is gradient-free and most robust for ill-conditioned likelihoods.
_FALLBACK_METHODS = ["bfgs", "nm", "powell"]

# No-exog fallback orders used when all exogenous attempts fail.
# These sacrifice env-feature accuracy but always converge.
_UNIVARIATE_ORDERS = [
    ((1, 1, 1), (0, 0, 0, 0)),
    ((0, 1, 1), (0, 0, 0, 0)),
    ((1, 1, 0), (0, 0, 0, 0)),
    ((0, 1, 0), (0, 0, 0, 0)),
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

    def _try_fit(sarimax_kwargs: dict, fit_kwargs: dict, forecast_exog_arr: np.ndarray) -> np.ndarray | None:
        """Return finite predictions or None."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # convergence warnings are expected in fallback
                fit = SARIMAX(**sarimax_kwargs).fit(**fit_kwargs)
            fcast_kw = {"steps": len(forecast_exog_arr)}
            if forecast_exog_arr.shape[1] > 0:
                fcast_kw["exog"] = forecast_exog_arr
            log_pred = np.asarray(fit.forecast(**fcast_kw)).flatten()
            if np.all(np.isfinite(log_pred)):
                return np.maximum(np.expm1(log_pred), 0.0)
        except Exception as exc:
            logger.debug("ARIMAX %s failed: %s", fit_kwargs, exc)
        return None

    base_sarimax = dict(enforce_stationarity=False, enforce_invertibility=False)

    # Phase 1: full exogenous model
    for order, seasonal_order in _ORDER_CHAIN:
        for method in ("lbfgs", *_FALLBACK_METHODS):
            result = _try_fit(
                {**base_sarimax, "endog": log_y, "exog": train_exog,
                 "order": order, "seasonal_order": seasonal_order},
                {"disp": False, "maxiter": 600, "method": method},
                forecast_exog,
            )
            if result is not None:
                return result

    # Phase 2: univariate fallback — drop exog entirely (e.g. DC with sparse env data)
    logger.warning("ARIMAX exog models all failed — trying univariate fallback")
    n_steps = len(forecast_exog)
    for order, seasonal_order in _UNIVARIATE_ORDERS:
        for method in ("lbfgs", "nm", "powell"):
            result = _try_fit(
                {**base_sarimax, "endog": log_y,
                 "order": order, "seasonal_order": seasonal_order},
                {"disp": False, "maxiter": 600, "method": method},
                np.empty((n_steps, 0)),   # signal: no exog
            )
            if result is not None:
                return result

    logger.warning("All ARIMAX orders/optimisers failed — returning NaN array")
    return np.full(len(forecast_exog), np.nan)
