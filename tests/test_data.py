"""Unit tests for data loading and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import _mmwr_week_start


class TestMmwrWeekStart:
    """MMWR (CDC epiweek) calendar correctness — the bug fixed in this codebase."""

    def test_2026_week_1_is_january_4(self):
        # Jan 4, 2026 is a Sunday → MMWR week 1 starts on Jan 4
        result = _mmwr_week_start(2026, 1)
        assert result == pd.Timestamp("2026-01-04")

    def test_2025_week_1_starts_dec_29_2024(self):
        # Jan 4, 2025 is a Saturday → Sunday before is Dec 29, 2024
        result = _mmwr_week_start(2025, 1)
        assert result == pd.Timestamp("2024-12-29")

    def test_2025_week_53_is_dec_28(self):
        # 2025 has 53 MMWR weeks; week 53 starts Dec 28, 2025
        result = _mmwr_week_start(2025, 53)
        assert result == pd.Timestamp("2025-12-28")

    def test_week_53_before_2026_week_1(self):
        # Chronological ordering: 2025-W53 (Dec 28) < 2026-W1 (Jan 4)
        w53 = _mmwr_week_start(2025, 53)
        w1  = _mmwr_week_start(2026, 1)
        assert w53 < w1

    def test_consecutive_weeks_are_7_days_apart(self):
        for year in [2020, 2021, 2022, 2023, 2024]:
            for week in range(1, 52):
                d1 = _mmwr_week_start(year, week)
                d2 = _mmwr_week_start(year, week + 1)
                assert (d2 - d1).days == 7, f"Failed for {year} week {week}"

    def test_all_starts_are_sundays(self):
        for year in [2020, 2023, 2026]:
            for week in range(1, 53):
                d = _mmwr_week_start(year, week)
                assert d.weekday() == 6, f"{year}-W{week}: {d} is not a Sunday"
