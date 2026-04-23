"""Holt-Winters exponential smoothing model."""

import logging

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)

# Try configs in order: (trend, seasonal, damped_trend)
# Multiplicative seasonal captures the proportional peak-to-trough ratio
# of flu data; damped trend prevents runaway trend extrapolation.
_CONFIGS = [
    ("add", "add",  True),   # additive + damped — stable default
    ("add", "add",  False),  # classic Holt-Winters additive
    ("add", "mul",  True),   # multiplicative seasonal + damped (better proportional fit)
    ("add", "mul",  False),
    ("add",  None,  True),   # trend-only, damped
    ("add",  None,  False),
]


def fit_holt_winters(train_y: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Holt-Winters exponential smoothing on log1p(ilitotal).

    Tries additive and multiplicative seasonal components with and without
    damped trend.  Falls back to simpler configurations on numerical failure.

    Parameters
    ----------
    train_y : 1-D array of historical ILI counts (raw, not log-transformed).
    n_steps : number of steps to forecast.

    Returns
    -------
    np.ndarray of length n_steps, ≥ 0.
    """
    log_y = np.log1p(np.maximum(np.asarray(train_y).flatten(), 0.0))

    for trend, seasonal, damped in _CONFIGS:
        # Multiplicative seasonal requires all positive values
        if seasonal == "mul" and np.any(log_y <= 0):
            continue
        try:
            model = ExponentialSmoothing(
                log_y,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=52 if seasonal else None,
                damped_trend=damped,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, remove_bias=True)
            raw = np.asarray(fit.forecast(n_steps)).flatten()
            result = np.maximum(np.expm1(raw), 0.0)
            if np.all(np.isfinite(result)):
                return result
        except Exception as exc:
            logger.debug("Holt-Winters trend=%s seasonal=%s damped=%s failed: %s",
                         trend, seasonal, damped, exc)

    logger.warning("Holt-Winters all configs failed — returning NaN array")
    return np.full(n_steps, np.nan)
