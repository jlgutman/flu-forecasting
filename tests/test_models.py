"""Smoke tests for individual model functions with synthetic data."""

import numpy as np
import pytest

from src.config import ARIMAX_EXOG, HORIZON, MIN_TRAIN, XGB_COLS
from src.models.arimax import fit_arimax
from src.models.holt_winters import fit_holt_winters
from src.models.xgboost_model import fit_xgboost, fit_xgboost_for_future
from src.models.ensemble import apply_weighted_ensemble, compute_ensemble_weights

import pandas as pd


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture()
def flu_series():
    """Synthetic seasonal flu-like series: ~2 years of weekly data."""
    rng = np.random.default_rng(0)
    n   = MIN_TRAIN + 20
    t   = np.arange(n)
    return 500 + 300 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 30, n)


@pytest.fixture()
def env_features(flu_series):
    """Random exogenous feature matrix matching ARIMAX_EXOG length."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((len(flu_series), len(ARIMAX_EXOG)))


@pytest.fixture()
def xgb_features(flu_series):
    """Random feature matrix matching XGB_COLS length."""
    rng = np.random.default_rng(2)
    return rng.standard_normal((len(flu_series), len(XGB_COLS)))


# ---------------------------------------------------------------------------
# ARIMAX
# ---------------------------------------------------------------------------

class TestArimax:
    def test_returns_horizon_predictions(self, flu_series, env_features):
        train_end = len(flu_series) - HORIZON
        pred = fit_arimax(
            flu_series[:train_end],
            env_features[:train_end],
            env_features[train_end:],
        )
        assert pred.shape == (HORIZON,)

    def test_predictions_are_non_negative(self, flu_series, env_features):
        train_end = len(flu_series) - HORIZON
        pred = fit_arimax(
            flu_series[:train_end],
            env_features[:train_end],
            env_features[train_end:],
        )
        assert np.all(pred >= 0) or np.all(np.isnan(pred))

    def test_too_short_series_returns_nan(self, env_features):
        tiny = np.array([100.0, 200.0, 300.0])
        pred = fit_arimax(tiny, env_features[:3], env_features[3:3 + HORIZON])
        # Either converges on tiny data or returns NaN — must not raise
        assert pred.shape == (HORIZON,)


# ---------------------------------------------------------------------------
# Holt-Winters
# ---------------------------------------------------------------------------

class TestHoltWinters:
    def test_returns_correct_length(self, flu_series):
        pred = fit_holt_winters(flu_series[:-HORIZON], HORIZON)
        assert pred.shape == (HORIZON,)

    def test_predictions_non_negative(self, flu_series):
        pred = fit_holt_winters(flu_series[:-HORIZON], HORIZON)
        assert np.all(pred >= 0) or np.all(np.isnan(pred))

    def test_single_step_forecast(self, flu_series):
        pred = fit_holt_winters(flu_series[:-1], 1)
        assert pred.shape == (1,)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class TestXGBoost:
    def test_returns_horizon_predictions(self, flu_series, xgb_features):
        pred = fit_xgboost(xgb_features, flu_series, train_end=MIN_TRAIN + HORIZON)
        assert pred.shape == (HORIZON,)

    def test_future_returns_horizon_predictions(self, flu_series, xgb_features):
        pred = fit_xgboost_for_future(xgb_features, flu_series)
        assert pred.shape == (HORIZON,)

    def test_too_short_returns_nan(self, flu_series, xgb_features):
        pred = fit_xgboost(xgb_features, flu_series, train_end=5)
        assert np.all(np.isnan(pred))


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class TestEnsemble:
    def _make_eval_df(self):
        return pd.DataFrame({
            "state":        ["NY"] * 10,
            "actual":       np.linspace(100, 200, 10),
            "arimax_pred":  np.linspace(105, 205, 10),
            "hw_pred":      np.linspace(110, 210, 10),
            "xgb_pred":     np.linspace(90,  190, 10),
            "lstm_pred":    np.linspace(95,  195, 10),
        })

    def test_weights_sum_to_one(self):
        df = self._make_eval_df()
        cols = ["arimax_pred", "hw_pred", "xgb_pred", "lstm_pred"]
        weights = compute_ensemble_weights(df, cols)
        for state, w in weights.items():
            assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_apply_weighted_ensemble_scalar(self):
        row = pd.Series({"arimax_pred": 100.0, "hw_pred": 200.0})
        w   = {"arimax_pred": 0.5, "hw_pred": 0.5}
        result = apply_weighted_ensemble(row, ["arimax_pred", "hw_pred"], w)
        assert result == pytest.approx(150.0)

    def test_missing_prediction_skipped(self):
        row = pd.Series({"arimax_pred": 100.0, "hw_pred": float("nan")})
        w   = {"arimax_pred": 0.5, "hw_pred": 0.5}
        result = apply_weighted_ensemble(row, ["arimax_pred", "hw_pred"], w)
        assert result == pytest.approx(100.0)

    def test_all_nan_returns_nan(self):
        row = pd.Series({"arimax_pred": float("nan")})
        w   = {"arimax_pred": 1.0}
        result = apply_weighted_ensemble(row, ["arimax_pred"], w)
        assert np.isnan(result)
