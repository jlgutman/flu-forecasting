"""Feature engineering: dataset assembly, imputation, lag features, future projection."""

import logging

import numpy as np
import pandas as pd

from src.data.loader import load_flu_cases, load_environmental, load_equity

logger = logging.getLogger(__name__)

_ENV_COLS = ["RHAV", "RHMN", "RHMX", "RHRR", "TAVG", "TMIN", "TMAX", "TRR", "CRD"]

# All lag offsets computed for the dataset (short + medium + long)
_ALL_LAGS = [1, 2, 3, 4, 5, 8, 13, 26, 52, 104]

# Rolling window sizes for ilitotal statistics (all use PAST values only → no leakage)
_ROLLING_WINDOWS = [4, 8, 13, 26]


def monthly_env_averages(env_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute per-state historical monthly averages for env feature projection."""
    tmp = env_df.copy()
    tmp["month"] = tmp["year_month"].apply(lambda ym: pd.Period(ym, "M").month)
    return tmp.groupby(["state", "month"])[_ENV_COLS].mean().reset_index()


def _add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged and rolling-window ilitotal features in-place.

    Lags computed: 1, 2, 3, 4, 5, 8, 13, 26, 52 weeks.
    Rolling stats : mean and std over 4, 8, 13-week trailing windows.

    All features are computed per state from PAST observations only (shift/rolling
    never uses the current row), so there is no data leakage.
    """
    df = df.sort_values(["state", "week_start"])
    grp = df.groupby("state")["ilitotal"]

    # Lag features
    for lag in _ALL_LAGS:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Log-transform lags (log1p of non-negative clipped values)
    for lag in _ALL_LAGS:
        df[f"lag_{lag}_log"] = np.log1p(df[f"lag_{lag}"].fillna(0).clip(lower=0))

    # Rolling window statistics (trailing, exclusive of current row via shift(1))
    for w in _ROLLING_WINDOWS:
        past = grp.shift(1)  # exclude current observation
        roll = past.transform(lambda s: s.rolling(w, min_periods=max(1, w // 2)).mean())
        df[f"rolling_{w}_mean_log"] = np.log1p(roll.fillna(0).clip(lower=0))

        roll_std = past.transform(lambda s: s.rolling(w, min_periods=max(1, w // 2)).std())
        # std for 4-week (short-term volatility) and 26-week (seasonal amplitude)
        if w in (4, 26):
            df[f"rolling_{w}_std_log"] = np.log1p(roll_std.fillna(0).clip(lower=0))

    # Drop raw (unlogged) lag columns — not used by any model
    df = df.drop(columns=[f"lag_{lag}" for lag in _ALL_LAGS])

    return df


def project_future_env(
    env_df: pd.DataFrame,
    state: str,
    future_dates: pd.DatetimeIndex,
    monthly_avgs: pd.DataFrame | None = None,
    historical_ilitotal: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build a feature row for each future week beyond environmental data coverage.

    Climate features use the state's historical monthly averages.
    Lag features use actual past observations — never predicted values.

    Only features used by ARIMAX are produced here (XGBoost/LSTM use the
    precomputed dataset rows, not this function).

    Safe lags for ARIMAX (recursive, 4-step horizon): lag_k where k ≥ 5,
    because at step t+4 we need y[t+4-k] which is historical only when k > 4.
    """
    if monthly_avgs is None:
        monthly_avgs = monthly_env_averages(env_df)

    state_avgs = monthly_avgs[monthly_avgs["state"] == state].set_index("month")
    n_hist = len(historical_ilitotal) if historical_ilitotal is not None else 0

    rows = []
    for j, d in enumerate(future_dates):
        m = d.month
        w = min(int(d.isocalendar().week), 52)
        row: dict = {"week_start": d, "month": m}

        for col in _ENV_COLS:
            row[col] = state_avgs.loc[m, col] if m in state_avgs.index else np.nan

        row["month_sin"] = np.sin(2 * np.pi * m / 12)
        row["month_cos"] = np.cos(2 * np.pi * m / 12)
        row["week_sin"]  = np.sin(2 * np.pi * w / 52)
        row["week_cos"]  = np.cos(2 * np.pi * w / 52)
        row["covid"]      = 0.0
        row["post_covid"] = 1.0  # all future dates are in the post-COVID era

        if historical_ilitotal is not None:
            def _lag_log(k: int) -> float:
                idx = n_hist - k + j
                val = float(historical_ilitotal[idx]) if 0 <= idx < n_hist else 0.0
                return float(np.log1p(max(0.0, val)))

            # lag_4 is safe for ARIMAX 4-step-ahead recursive forecast:
            # at future step j, lag_4[j] = y[n_hist - 4 + j] which is always
            # historical (j=0..3, so index = n_hist-4..n_hist-1, all < n_hist).
            for lag in [4, 5, 8, 13, 26, 52, 104]:
                row[f"lag_{lag}_log"] = _lag_log(lag)
        else:
            for lag in [4, 5, 8, 13, 26, 52, 104]:
                row[f"lag_{lag}_log"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def build_dataset(states: list[str] | None = None) -> pd.DataFrame:
    """
    Join weekly flu cases with monthly environmental and seasonal equity data.

    Returns one row per state per CDC week with all model-ready features:
      state, year, week, week_start, year_month, season_label,
      month, month_sin/cos, week_sin/cos, ilitotal, num_providers, total_patients,
      RHAV/RHMN/RHMX/RHRR, TAVG/TMIN/TMAX/TRR, STPOP/STLA/POPPCT/POPDEN/CRD,
      vax_coverage, spend_metric_1, poverty_pct, covid,
      lag_{1,2,3,4,5,8,13,26,52}_log,
      rolling_{4,8,13}_mean_log, rolling_4_std_log
    """
    flu    = load_flu_cases()
    env    = load_environmental()
    equity = load_equity()

    merged = flu.merge(env, on=["state", "year_month"], how="left")
    merged = merged.merge(equity, on=["state", "season_label"], how="left")

    if states:
        merged = merged[merged["state"].isin(states)]

    # Env NaN imputation: (1) per-state ffill/bfill, (2) historical monthly avg,
    # (3) global column mean as last resort
    merged[_ENV_COLS] = merged.groupby("state")[_ENV_COLS].transform(
        lambda x: x.ffill().bfill()
    )
    monthly_avgs = monthly_env_averages(env)
    month_mean = merged.merge(
        monthly_avgs, on=["state", "month"], how="left", suffixes=("", "_avg")
    ).reset_index(drop=True)
    merged = merged.reset_index(drop=True)
    for col in _ENV_COLS:
        avg_col = col + "_avg"
        if avg_col in month_mean.columns:
            merged[col] = merged[col].combine_first(month_mean[avg_col])
    merged[_ENV_COLS] = merged[_ENV_COLS].fillna(merged[_ENV_COLS].mean())

    # COVID suppression indicator (2020-03-15 – 2021-09-01)
    merged["covid"] = (
        (merged["week_start"] >= pd.Timestamp("2020-03-15")) &
        (merged["week_start"] <= pd.Timestamp("2021-09-01"))
    ).astype(float)

    # Post-COVID recovery regime (after 2021-09-01) — flu rebounded with altered patterns
    merged["post_covid"] = (
        merged["week_start"] > pd.Timestamp("2021-09-01")
    ).astype(float)

    # Current log-ILI (anchor feature — lets XGBoost scale predictions to current level)
    merged["ilitotal_log"] = np.log1p(merged["ilitotal"].clip(lower=0))

    # Lagged and rolling features (per-state, no leakage)
    merged = _add_lag_and_rolling_features(merged)

    result = merged.sort_values(["state", "week_start"]).reset_index(drop=True)
    logger.info(
        "Dataset built: %d rows, %d states, %s – %s",
        len(result),
        result["state"].nunique(),
        result["week_start"].min().date(),
        result["week_start"].max().date(),
    )
    return result
