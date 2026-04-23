"""FastAPI REST server for the flu forecasting pipeline."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.features import build_dataset
from src.runner import run_forecast

logger = logging.getLogger(__name__)

_cache: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading and merging datasets…")
    _cache["df"] = build_dataset()
    df = _cache["df"]
    logger.info(
        "Ready — %d rows, %d states, %s → %s",
        len(df), df["state"].nunique(),
        df["year_month"].min(), df["year_month"].max(),
    )
    yield
    _cache.clear()


app = FastAPI(
    title="Flu Forecast API",
    description=(
        "Multi-model influenza forecasting (ARIMAX · Holt-Winters · XGBoost · LSTM). "
        "POST /api/forecast to run models and generate output CSVs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    states: Optional[list[str]] = Field(
        default=None,
        description="Two-letter state codes (e.g. ['CA','TX']). Omit for all states.",
        examples=[["CA", "TX", "NY"]],
    )
    test_weeks: int = Field(
        default=52, ge=8, le=260,
        description="Weeks held out for evaluation (default 52 = 1 year).",
    )
    run_cv: bool = Field(
        default=False,
        description="Run expanding-window cross-validation (slower).",
    )


class HealthResponse(BaseModel):
    status: str
    rows_loaded: int
    states: int
    date_range: str


@app.get("/health", response_model=HealthResponse)
def health():
    """Dataset load status — call before /api/forecast."""
    df = _cache.get("df")
    if df is None:
        return HealthResponse(status="loading", rows_loaded=0, states=0, date_range="")
    return HealthResponse(
        status="ok",
        rows_loaded=len(df),
        states=int(df["state"].nunique()),
        date_range=f"{df['year_month'].min()} → {df['year_month'].max()}",
    )


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    """
    Run ARIMAX, Holt-Winters, XGBoost, and LSTM on the weekly flu dataset.

    - Trains on all weeks except the last `test_weeks`.
    - Evaluates on held-out period → **output/forecast_eval.csv**
    - Forecasts next 4 weeks → **output/forecast_next4weeks.csv**
    - Returns RMSE / MAE / MAPE / R² per model.
    """
    df = _cache.get("df")
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded yet — try again shortly.")

    if req.states:
        unknown = set(req.states) - set(df["state"].unique())
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown state codes: {sorted(unknown)}")
        df = df[df["state"].isin(req.states)].copy()

    return run_forecast(df, test_weeks=req.test_weeks, run_cv=req.run_cv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
