#!/usr/bin/env python3
"""
CLI entry point — train all models and generate forecast CSVs.

Usage
-----
# All 50 states, 52-week hold-out, no CV:
python scripts/run_forecast.py

# Specific states with CV:
python scripts/run_forecast.py --states NY CA TX --test-weeks 52 --cv

# Quick smoke test (8 weeks, no CV):
python scripts/run_forecast.py --states NY --test-weeks 8
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running without `pip install -e .` by adding src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.features import build_dataset
from src.runner import run_forecast


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run the flu forecasting pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--states", nargs="*", metavar="STATE",
        help="Two-letter state abbreviations (e.g. NY CA TX). Omit for all 50 states.",
    )
    parser.add_argument(
        "--test-weeks", type=int, default=52, metavar="N",
        help="Number of weeks to hold out for evaluation.",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run expanding-window cross-validation (adds significant run time).",
    )
    args = parser.parse_args()

    df      = build_dataset(states=args.states or None)
    results = run_forecast(df, test_weeks=args.test_weeks, run_cv=args.cv)

    print("\n=== Metrics ===")
    print(json.dumps(results["metrics"], indent=2))

    if results.get("cv_metrics"):
        print("\n=== CV Metrics ===")
        print(json.dumps(results["cv_metrics"], indent=2))

    print(f"\nEvaluation CSV : {results['eval_csv']}")
    print(f"4-week forecast: {results['future_csv']}")


if __name__ == "__main__":
    main()
