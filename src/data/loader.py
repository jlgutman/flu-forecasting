"""Raw data loaders — one function per source CSV, no business logic."""

import logging

import numpy as np
import pandas as pd

from src.config import DATA_DIR, STATE_ABBREV

logger = logging.getLogger(__name__)


def _assign_season(ym: str) -> str:
    p = pd.Period(ym, "M")
    if p.month >= 10:
        return f"{p.year}-{str(p.year + 1)[2:]}"
    return f"{p.year - 1}-{str(p.year)[2:]}"


def _mmwr_week_start(year: int, week: int) -> pd.Timestamp:
    """
    Compute the Sunday start date for a CDC MMWR (epiweek) week.

    MMWR weeks start on Sunday; Jan 4 of each year always falls in MMWR week 1.
    This differs from ISO weeks (Monday-based) — using ISO parsing mis-places
    week 1 of years where Jan 4 falls mid-week (e.g. 2026-W01 → Dec 29 in ISO
    vs Jan 4 in MMWR).
    """
    jan4 = pd.Timestamp(year, 1, 4)
    days_to_sunday = (jan4.weekday() + 1) % 7   # Mon=0…Sun=6 → days back to Sunday
    w1_start = jan4 - pd.Timedelta(days=days_to_sunday)
    return w1_start + pd.Timedelta(weeks=week - 1)


def load_flu_cases() -> pd.DataFrame:
    """
    Load CDC weekly ILI surveillance — one row per state per week.

    Source: data/flucases2010forward.csv
    Key decisions:
      - 'New York City' is a separate CDC entry that overlaps NY state → excluded.
      - 'X' suppression codes in numeric fields are coerced to NaN.
      - Week dates computed via MMWR calendar (Sunday-based), not ISO (Monday-based).
    """
    df = pd.read_csv(DATA_DIR / "flucases2010forward.csv")
    df = df[df["REGION"] != "New York City"].copy()

    for col in ["ILITOTAL", "NUM. OF PROVIDERS", "TOTAL PATIENTS"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["state"] = df["REGION"].map(STATE_ABBREV)
    df = df.dropna(subset=["state", "ILITOTAL"])

    df["week_start"] = [
        _mmwr_week_start(int(yr), int(wk))
        for yr, wk in zip(df["YEAR"], df["WEEK"])
    ]

    df["year_month"]   = df["week_start"].dt.to_period("M").astype(str)
    df["season_label"] = df["year_month"].apply(_assign_season)
    df["month"]        = df["week_start"].dt.month
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["week_num"]     = df["WEEK"].clip(upper=52)
    df["week_sin"]     = np.sin(2 * np.pi * df["week_num"] / 52)
    df["week_cos"]     = np.cos(2 * np.pi * df["week_num"] / 52)

    logger.debug("Loaded %d flu-case rows across %d states", len(df), df["state"].nunique())

    return df.rename(columns={
        "YEAR": "year", "WEEK": "week",
        "ILITOTAL": "ilitotal",
        "NUM. OF PROVIDERS": "num_providers",
        "TOTAL PATIENTS": "total_patients",
    })[[
        "state", "year", "week", "week_start",
        "year_month", "season_label",
        "month", "month_sin", "month_cos",
        "week_sin", "week_cos",
        "ilitotal", "num_providers", "total_patients",
    ]]


def load_environmental() -> pd.DataFrame:
    """
    Load NOAA climate + 2010 Census features (monthly, per state).

    Source: data/flu_environmental_factors_data.csv  (UTF-8 BOM → encoding='utf-8-sig')
    """
    df = pd.read_csv(
        DATA_DIR / "flu_environmental_factors_data.csv",
        encoding="utf-8-sig",
    )
    df = df.rename(columns={"STATE": "state", "DATE": "year_month"})
    keep = [
        "state", "year_month",
        "RHAV", "RHMN", "RHMX", "RHRR",
        "TAVG", "TMIN", "TMAX", "TRR",
        "STPOP", "STLA", "POPPCT", "POPDEN", "CRD",
    ]
    logger.debug("Loaded %d environmental rows", len(df))
    return df[keep].copy()


def load_equity() -> pd.DataFrame:
    """
    Load state-season equity data — Overall race/ethnicity group only.

    Source: data/state_season_race_equity_20260406.csv
    """
    df = pd.read_csv(DATA_DIR / "state_season_race_equity_20260406.csv")
    result = df[df["race_eth"] == "Overall"][[
        "state", "season_label", "vax_coverage", "spend_metric_1", "poverty_pct"
    ]].copy()
    logger.debug("Loaded %d equity rows", len(result))
    return result
