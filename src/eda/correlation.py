"""Correlation helpers for exploratory data analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd


SUPPORTED_CORRELATION_METHODS = {"pearson", "spearman", "kendall"}


def build_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Return a numeric correlation matrix using the selected method."""
    method = _validate_method(method)
    numeric_data = _numeric_data(df, columns)
    if numeric_data.empty:
        return pd.DataFrame()
    return numeric_data.corr(method=method)


def build_correlation_pairs_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Return one row per unique variable pair from a correlation matrix."""
    corr_matrix = build_correlation_matrix(df, columns=columns, method=method)
    rows = []
    for left_index, variable_1 in enumerate(corr_matrix.columns):
        for variable_2 in corr_matrix.columns[left_index + 1 :]:
            correlation = corr_matrix.loc[variable_1, variable_2]
            if pd.isna(correlation):
                continue
            rows.append(
                {
                    "variable_1": variable_1,
                    "variable_2": variable_2,
                    "correlation": float(correlation),
                    "abs_correlation": float(abs(correlation)),
                    "direction": "positive" if correlation > 0 else "negative" if correlation < 0 else "none",
                    "method": method,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["variable_1", "variable_2", "correlation", "abs_correlation", "direction", "method"]
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def identify_high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Return variable pairs with absolute correlation at or above threshold."""
    if corr_matrix is None or corr_matrix.empty:
        return pd.DataFrame(columns=["variable_1", "variable_2", "correlation", "abs_correlation", "direction"])
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1.")

    rows = []
    for left_index, variable_1 in enumerate(corr_matrix.columns):
        for variable_2 in corr_matrix.columns[left_index + 1 :]:
            correlation = corr_matrix.loc[variable_1, variable_2]
            if pd.isna(correlation) or abs(correlation) < threshold:
                continue
            rows.append(
                {
                    "variable_1": variable_1,
                    "variable_2": variable_2,
                    "correlation": float(correlation),
                    "abs_correlation": float(abs(correlation)),
                    "direction": "strong positive" if correlation > 0 else "strong negative",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["variable_1", "variable_2", "correlation", "abs_correlation", "direction"])
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def _validate_method(method: str) -> str:
    """Validate a pandas correlation method name."""
    normalized = str(method).lower()
    if normalized not in SUPPORTED_CORRELATION_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_CORRELATION_METHODS))
        raise ValueError(f"Unsupported correlation method `{method}`. Choose one of: {allowed}.")
    return normalized


def _numeric_data(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    """Return a numeric-only copy of the selected columns."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    selected_columns = list(columns) if columns is not None else list(df.columns)
    missing = [column for column in selected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Columns were not found in the dataset: {', '.join(missing)}")

    data = df[selected_columns].copy(deep=True)
    numeric_columns = []
    for column in data.columns:
        if pd.api.types.is_bool_dtype(data[column]):
            continue
        converted = pd.to_numeric(data[column], errors="coerce")
        if converted.notna().sum() > 0:
            data[column] = converted
            numeric_columns.append(column)

    return data[numeric_columns]
