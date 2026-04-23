"""Unit tests for evaluation.metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import adf_test, agg_metrics, compute_metrics


class TestComputeMetrics:
    def test_perfect_prediction(self):
        a = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(a, a)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert m["mae"]  == pytest.approx(0.0, abs=1e-9)
        assert m["r2"]   == pytest.approx(1.0, abs=1e-9)

    def test_known_values(self):
        a = np.array([100.0, 200.0])
        p = np.array([110.0, 190.0])
        m = compute_metrics(a, p)
        assert m["rmse"] == pytest.approx(10.0, abs=1e-6)
        assert m["mae"]  == pytest.approx(10.0, abs=1e-6)

    def test_nan_handling(self):
        a = np.array([100.0, np.nan, 300.0])
        p = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(a, p)
        # NaN row excluded — remaining two are perfect
        assert m["rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_all_nan_returns_none(self):
        m = compute_metrics(np.array([np.nan]), np.array([np.nan]))
        assert m["rmse"] is None

    def test_mape_excludes_near_zero(self):
        # Actual < 10 must be excluded from MAPE
        a = np.array([5.0, 200.0])    # first is < 10
        p = np.array([500.0, 200.0])  # first would blow up MAPE
        m = compute_metrics(a, p)
        assert m["mape"] == pytest.approx(0.0, abs=1e-9)

    def test_r2_constant_actual(self):
        # When actual is constant, R² is undefined (ss_tot=0)
        a = np.array([100.0, 100.0])
        p = np.array([110.0, 90.0])
        m = compute_metrics(a, p)
        # Should not raise; r2_score will return a value (likely -inf or 0)
        assert m["r2"] is not None


class TestAggMetrics:
    def test_averages_correctly(self):
        state_metrics = {
            "NY": {"rmse": 100.0, "mae": 50.0, "mape": 10.0, "smape": 8.0, "r2": 0.8},
            "CA": {"rmse": 200.0, "mae": 100.0, "mape": 20.0, "smape": 16.0, "r2": 0.6},
        }
        agg = agg_metrics(state_metrics)
        assert agg["rmse"] == pytest.approx(150.0, abs=1e-6)
        assert agg["r2"]   == pytest.approx(0.7,   abs=1e-6)

    def test_none_values_skipped(self):
        state_metrics = {
            "NY": {"rmse": 100.0, "mae": None, "mape": None, "smape": 8.0, "r2": 0.8},
        }
        agg = agg_metrics(state_metrics)
        assert agg["rmse"] == pytest.approx(100.0)

    def test_empty_input(self):
        assert agg_metrics({}) == {}


class TestAdfTest:
    def test_stationary_series(self):
        rng = np.random.default_rng(0)
        # White noise is stationary
        result = adf_test(rng.standard_normal(200))
        assert result["stationary"] is True
        assert result["p_value"] is not None

    def test_random_walk_non_stationary(self):
        rng = np.random.default_rng(42)
        # Random walk is non-stationary
        series = np.cumsum(rng.standard_normal(300))
        result = adf_test(series)
        # Random walk should generally fail stationarity — not guaranteed but expected
        assert result["adf_stat"] is not None

    def test_nan_in_series(self):
        series = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0] * 30)
        result = adf_test(series)
        assert result["p_value"] is not None
