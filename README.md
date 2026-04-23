# flu-forecasting

Multi-model influenza-like illness (ILI) forecasting pipeline using CDC weekly state data. Produces 4-week-ahead predictions per U.S. state via ARIMAX, Holt-Winters, XGBoost, and a BiLSTM ensemble.

## Setup

Requires Python ≥ 3.10 and TensorFlow ≥ 2.15 (use a conda environment for TF on macOS/Python 3.13+).

```bash
conda create -n flu-forecasting python=3.13
conda activate flu-forecasting
pip install -r requirements.txt
pip install -e .
```

## Data

Place these files in `data/` before running:

| File | Description |
|------|-------------|
| `flucases2010forward.csv` | CDC weekly ILI counts by state (MMWR epiweeks) |
| `flu_environmental_factors_data.csv` | NOAA monthly climate + Census features (UTF-8 BOM) |
| `state_season_race_equity_20260406.csv` | Seasonal equity data (vax coverage, poverty, population) |

If `flucases2010forward.csv` has missing MMWR weeks, run `python scripts/impute_flucases.py` to fill them.

## Usage

```bash
# Full pipeline — all states, 52-week hold-out
python scripts/run_forecast.py

# Specific states
python scripts/run_forecast.py --states NY CA TX --test-weeks 52

# Quick smoke test
python scripts/run_forecast.py --states NY --test-weeks 8

# REST API (GET /health, POST /api/forecast)
python -m src.serving.api
```

Outputs are written to `output/`:
- `forecast_eval.csv` — test-period actuals vs. all model predictions
- `forecast_next4weeks.csv` — 4-week-ahead forecast
- `xgb_feature_importance.csv`

```bash
# Generate plots after running the pipeline
python scripts/plot_results.py
```

## Models

| Model | Strategy |
|-------|----------|
| ARIMAX | SARIMAX with rolling 3-year post-COVID window; exogenous climate + lag features |
| Holt-Winters | Seasonal exponential smoothing; 2-season rolling window |
| XGBoost | Direct multi-output; all lags + rolling stats as features |
| BiLSTM | All states combined; log-ratio delta targets; soft-attention pooling |

The ensemble weights models by 1/RMSE per state.

## Tests

```bash
pytest tests/ -v
pytest tests/test_models.py::TestArimax -v
```
