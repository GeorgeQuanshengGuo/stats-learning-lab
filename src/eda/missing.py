"""Read-only missing value summaries.

Version 0.1 does not clean or impute missing values.
"""

import pandas as pd


def build_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value counts and percentages for each column."""
    missing = df.isna().sum()

    return pd.DataFrame(
        {
            "variable": df.columns,
            "missing_count": missing,
            "missing_pct": missing / len(df) * 100 if len(df) else 0,
        }
    ).reset_index(drop=True)


def build_row_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per observation with row-level missing value counts."""
    missing_count = df.isna().sum(axis=1)
    column_count = len(df.columns)

    return pd.DataFrame(
        {
            "row_number": range(1, len(df) + 1),
            "missing_count": missing_count,
            "missing_pct": missing_count / column_count * 100 if column_count else 0,
        }
    )


def missing_value_table(data: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for older page code."""
    return build_missing_table(data)
