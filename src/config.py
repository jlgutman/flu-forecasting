"""
Central configuration — all constants, paths, and feature lists in one place.
Import from here instead of hard-coding values in individual modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).parent.parent   # flu-forecast/
DATA_DIR   = ROOT_DIR / "datasets"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Forecast settings
# ---------------------------------------------------------------------------
HORIZON     = 4     # weeks ahead to predict
SEQ_LEN     = 52   # LSTM look-back window (1 full annual cycle)
MIN_TRAIN   = 104   # minimum training weeks required (~2 years)

# ---------------------------------------------------------------------------
# LSTM hyperparameters  (Amendolara et al. 2023 methodology, tuned)
# ---------------------------------------------------------------------------
LSTM_EPOCHS  = 200   # EarlyStopping handles early exit
LSTM_BATCH   = 128   # larger batch → stable gradient estimates for long sequences
LSTM_LR      = 1e-4  # conservative LR; ReduceLROnPlateau decays further
LSTM_UNITS   = 160   # BiLSTM first layer units (×2 after bidirectional)
LSTM_UNITS2  = 96    # BiLSTM second layer units (×2 after bidirectional)

# ---------------------------------------------------------------------------
# XGBoost hyperparameters  (Chen et al. 2024, 5-fold CV tuned + extended)
# ---------------------------------------------------------------------------
XGB_PARAMS: dict = dict(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.75,
    colsample_bytree=0.75,  # 75% of features — more interactions vs. ~50% before
    min_child_weight=2,
    reg_alpha=0.05,          # light L1 to handle correlated lag features
    reg_lambda=1.0,          # L2 shrinkage
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

# ARIMAX: only features known at ALL recursive forecast steps t+1..t+HORIZON.
# At step t+h, lag_k requires y[t+h-k].  For this to be fully historical we
# need k > HORIZON (= 4), so the minimum safe lag is 5.
# Rolling-window features are excluded — they would need predicted values at
# steps t+2..t+HORIZON, which are unavailable during recursive forecasting.
ARIMAX_EXOG: list[str] = [
    "RHAV", "RHRR", "TAVG", "TRR", "CRD",
    "month_sin", "month_cos", "week_sin", "week_cos",
    # lag_4_log is the minimum safe lag for 4-step recursive ARIMAX:
    # at forecast step h, lag_4[t+h] = y[t+h-4] which is historical for h=1..4.
    "lag_4_log", "lag_5_log", "lag_8_log", "lag_13_log", "lag_26_log", "lag_52_log", "lag_104_log",
    "covid", "post_covid",
]

# XGBoost: direct multi-output — predicts all HORIZON steps at once from
# a single snapshot at t, so ALL lags and rolling stats are safe (no future
# values are needed at inference time).
XGB_COLS: list[str] = [
    "RHAV", "RHRR", "TAVG", "TRR", "CRD",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "lag_1_log", "lag_2_log", "lag_3_log", "lag_4_log",
    "lag_5_log", "lag_8_log", "lag_13_log", "lag_26_log", "lag_52_log", "lag_104_log",
    "rolling_4_mean_log", "rolling_8_mean_log", "rolling_13_mean_log",
    "rolling_26_mean_log", "rolling_4_std_log", "rolling_26_std_log",
    "covid", "post_covid",
    # Current log-ILI level: strongest single predictor; no leakage since it's
    # y[t] at the feature row and targets are y[t+1..t+H].
    "ilitotal_log",
]

# LSTM: receives SEQ_LEN-wide windows; includes raw ilitotal so the model
# can learn its own temporal structure via attention over the full sequence.
FEATURE_COLS: list[str] = [
    "RHAV", "RHRR", "TAVG", "TRR", "CRD",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "lag_1_log", "lag_2_log", "lag_3_log", "lag_4_log",
    "lag_5_log", "lag_8_log", "lag_13_log", "lag_26_log", "lag_52_log", "lag_104_log",
    "rolling_4_mean_log", "rolling_8_mean_log", "rolling_13_mean_log",
    "rolling_26_mean_log", "rolling_4_std_log", "rolling_26_std_log",
    "covid", "post_covid", "ilitotal",
]

# Backward-compatible alias
EXOG_COLS = ARIMAX_EXOG

# ---------------------------------------------------------------------------
# State abbreviation map (excludes "New York City" — overlaps with NY state)
# ---------------------------------------------------------------------------
STATE_ABBREV: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
