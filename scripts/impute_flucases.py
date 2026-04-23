"""
Seasonal weighted mean imputation for missing ILINet flu surveillance weeks.

Adds rows for 2026 W14 and W15 for every region, using a weighted mean
of the same week from years 2019–2025.

Weights: 2025=30%, 2024=25%, 2023=20%, 2022=10%, 2021=7%, 2020=5%, 2019=3%
Missing (X) values for a given year/region/week are skipped; weights are
re-normalized among available years.
"""

import pandas as pd
from src.config import DATA_DIR

INPUT_CSV = DATA_DIR / "flucases2010forward_original.csv"
OUTPUT_CSV = DATA_DIR / "flucases2010forward.csv"

IMPUTE_WEEKS = [15, 16]
IMPUTE_YEAR = 2026
NUMERIC_COLS = ["ILITOTAL", "NUM. OF PROVIDERS", "TOTAL PATIENTS"]

WEIGHTS = {
    2025: 0.30,
    2024: 0.25,
    2023: 0.20,
    2022: 0.10,
    2021: 0.07,
    2020: 0.05,
    2019: 0.03,
}


def weighted_mean(values_by_year: dict[int, float]) -> float | None:
    """
    Compute weighted mean from {year: value} dict, using only years with
    valid (non-X) values. Weights are re-normalized to sum to 1.
    Returns None if no valid years exist.
    """
    available = {yr: val for yr, val in values_by_year.items() if val is not None}
    if not available:
        return None
    total_weight = sum(WEIGHTS[yr] for yr in available)
    result = sum(WEIGHTS[yr] * val for yr, val in available.items()) / total_weight
    return result


def parse_numeric(value) -> float | None:
    """Return float for numeric strings, None for 'X' or missing."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)

    # Convert YEAR and WEEK to int for filtering
    df["YEAR"] = df["YEAR"].astype(int)
    df["WEEK"] = df["WEEK"].astype(int)

    regions = df["REGION"].unique()
    region_type_map = df.drop_duplicates("REGION").set_index("REGION")["REGION TYPE"]

    imputed_rows = []

    for week in IMPUTE_WEEKS:
        for region in regions:
            region_type = region_type_map[region]
            row_values = {
                "REGION TYPE": region_type,
                "REGION": region,
                "YEAR": IMPUTE_YEAR,
                "WEEK": week,
                "IS_IMPUTED": True,
            }

            for col in NUMERIC_COLS:
                values_by_year = {}
                for yr in WEIGHTS:
                    mask = (
                        (df["REGION"] == region)
                        & (df["YEAR"] == yr)
                        & (df["WEEK"] == week)
                    )
                    matches = df[mask]
                    if matches.empty:
                        values_by_year[yr] = None
                    else:
                        values_by_year[yr] = parse_numeric(matches.iloc[0][col])

                result = weighted_mean(values_by_year)
                # Round to nearest integer and store as string to match source format
                row_values[col] = str(round(result)) if result is not None else "X"

            imputed_rows.append(row_values)

    imputed_df = pd.DataFrame(imputed_rows, columns=list(df.columns) + ["IS_IMPUTED"])

    # Add IS_IMPUTED=False to original data
    df["IS_IMPUTED"] = False

    output_df = pd.concat([df, imputed_df], ignore_index=True)

    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Written {len(output_df)} rows to {OUTPUT_CSV}")
    print(f"  Original rows : {len(df)}")
    print(f"  Imputed rows  : {len(imputed_df)}")

    # Print a sample of the imputed rows
    print("\nSample imputed rows:")
    sample = imputed_df[imputed_df["REGION"].isin(["Alabama", "California", "Texas"])]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
