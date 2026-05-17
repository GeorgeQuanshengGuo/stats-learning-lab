"""Missing value cleaning functions.

Each function returns a cleaned copy of the data plus a human-readable log
entry. The input DataFrame is never modified in place.
"""

from typing import Any

import pandas as pd


def drop_missing_rows_for_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop rows where one selected column is missing."""
    rows_before = len(df)
    missing_before = _missing_count(df, column)

    new_df = df.dropna(subset=[column]).copy()

    return new_df, _build_log_entry(
        operation="drop_missing_rows_for_column",
        column=column,
        rows_before=rows_before,
        rows_after=len(new_df),
        missing_before=missing_before,
        missing_after=_missing_count(new_df, column),
        details=f"Dropped rows where {column} was missing.",
    )


def drop_all_rows_with_any_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop every row that contains at least one missing value."""
    rows_before = len(df)
    missing_before = int(df.isna().sum().sum())

    new_df = df.dropna(axis=0, how="any").copy()

    return new_df, _build_log_entry(
        operation="drop_all_rows_with_any_missing",
        column=None,
        rows_before=rows_before,
        rows_after=len(new_df),
        missing_before=missing_before,
        missing_after=int(new_df.isna().sum().sum()),
        details="Dropped rows that had one or more missing values.",
    )


def fill_numeric_mean(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing values in a numeric column with the column mean."""
    fill_value = pd.to_numeric(df[column], errors="coerce").mean()
    return _fill_column(
        df=df,
        column=column,
        fill_value=fill_value,
        operation="fill_numeric_mean",
        details=f"Filled missing values in {column} with the mean: {fill_value}.",
    )


def fill_numeric_median(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing values in a numeric column with the column median."""
    fill_value = pd.to_numeric(df[column], errors="coerce").median()
    return _fill_column(
        df=df,
        column=column,
        fill_value=fill_value,
        operation="fill_numeric_median",
        details=f"Filled missing values in {column} with the median: {fill_value}.",
    )


def fill_categorical_mode(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing values in a categorical column with its most common value."""
    modes = df[column].dropna().mode()
    fill_value = None if modes.empty else modes.iloc[0]

    return _fill_column(
        df=df,
        column=column,
        fill_value=fill_value,
        operation="fill_categorical_mode",
        details=f"Filled missing values in {column} with the mode: {fill_value}.",
    )


def fill_categorical_missing_label(
    df: pd.DataFrame,
    column: str,
    label: str = "Missing",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing values in a categorical column with a visible label."""
    return _fill_column(
        df=df,
        column=column,
        fill_value=label,
        operation="fill_categorical_missing_label",
        details=f"Filled missing values in {column} with the label: {label}.",
    )


def _fill_column(
    df: pd.DataFrame,
    column: str,
    fill_value: Any,
    operation: str,
    details: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill one column and build the standard cleaning log entry."""
    rows_before = len(df)
    missing_before = _missing_count(df, column)
    new_df = df.copy(deep=True)

    if fill_value is not None and not pd.isna(fill_value):
        new_df[column] = new_df[column].fillna(fill_value)

    return new_df, _build_log_entry(
        operation=operation,
        column=column,
        rows_before=rows_before,
        rows_after=len(new_df),
        missing_before=missing_before,
        missing_after=_missing_count(new_df, column),
        details=details,
    )


def _missing_count(df: pd.DataFrame, column: str) -> int:
    """Count missing values in one column."""
    return int(df[column].isna().sum())


def _build_log_entry(
    operation: str,
    column: str | None,
    rows_before: int,
    rows_after: int,
    missing_before: int,
    missing_after: int,
    details: str,
) -> dict[str, Any]:
    """Create a consistent log entry for one cleaning operation."""
    return {
        "operation": operation,
        "column": column,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "details": details,
    }
