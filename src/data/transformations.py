"""Transformation and feature engineering helpers.

Every function returns a new DataFrame and a log entry. The input DataFrame is
never modified in place, and transformations always create a new column.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox


LOG_FIELDS = [
    "operation_type",
    "method",
    "source_columns",
    "new_column",
    "parameters",
    "rows_before",
    "rows_after",
    "missing_before",
    "missing_after",
    "notes",
]


def apply_log_transform(
    df: pd.DataFrame,
    column: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a natural log transformed column. Values must be greater than 0."""
    values = _numeric_series(df, column)
    _require((values.dropna() > 0).all(), "Log transform requires all non-missing values to be > 0.")
    target = _new_column_name(column, "log", new_column)
    return _apply_single_column_transform(df, column, target, "log", np.log, {})


def apply_log1p_transform(
    df: pd.DataFrame,
    column: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a log(1 + x) transformed column. Values must be greater than -1."""
    values = _numeric_series(df, column)
    _require((values.dropna() > -1).all(), "Log1p transform requires all non-missing values to be > -1.")
    target = _new_column_name(column, "log1p", new_column)
    return _apply_single_column_transform(df, column, target, "log1p", np.log1p, {})


def apply_sqrt_transform(
    df: pd.DataFrame,
    column: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a square-root transformed column. Values must be at least 0."""
    values = _numeric_series(df, column)
    _require((values.dropna() >= 0).all(), "Square root transform requires all non-missing values to be >= 0.")
    target = _new_column_name(column, "sqrt", new_column)
    return _apply_single_column_transform(df, column, target, "sqrt", np.sqrt, {})


def apply_cube_root_transform(
    df: pd.DataFrame,
    column: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a cube-root transformed column."""
    target = _new_column_name(column, "cuberoot", new_column)
    return _apply_single_column_transform(df, column, target, "cube_root", np.cbrt, {})


def apply_reciprocal_transform(
    df: pd.DataFrame,
    column: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a reciprocal transformed column. Values must be nonzero."""
    values = _numeric_series(df, column)
    _require((values.dropna() != 0).all(), "Reciprocal transform requires all non-missing values to be nonzero.")
    target = _new_column_name(column, "reciprocal", new_column)
    return _apply_single_column_transform(df, column, target, "reciprocal", lambda series: 1 / series, {})


def apply_square_transform(
    df: pd.DataFrame,
    column: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a squared transformed column."""
    target = _new_column_name(column, "squared", new_column)
    return _apply_single_column_transform(df, column, target, "square", lambda series: series**2, {})


def apply_boxcox_transform(
    df: pd.DataFrame,
    column: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a Box-Cox transformed column. Values must be strictly positive."""
    values = _numeric_series(df, column)
    non_missing = values.dropna()
    _require((non_missing > 0).all(), "Box-Cox transform requires all non-missing values to be > 0.")
    _require(non_missing.nunique() > 1, "Box-Cox transform requires at least two distinct non-missing values.")

    transformed_values, lambda_value = stats.boxcox(non_missing)
    target = _new_column_name(column, "boxcox", new_column)
    new_df = _copy_with_empty_new_column(df, target)
    new_df.loc[non_missing.index, target] = transformed_values

    return new_df, _build_log_entry(
        method="boxcox",
        source_columns=[column],
        new_column=target,
        parameters={"lambda": float(lambda_value)},
        rows_before=len(df),
        rows_after=len(new_df),
        missing_before=int(values.isna().sum()),
        missing_after=int(new_df[target].isna().sum()),
        notes=f"Created {target} using Box-Cox transformation.",
    )


def apply_yeojohnson_transform(
    df: pd.DataFrame,
    column: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a Yeo-Johnson transformed column. Zero and negative values are allowed."""
    values = _numeric_series(df, column)
    non_missing = values.dropna()
    _require(non_missing.nunique() > 1, "Yeo-Johnson transform requires at least two distinct non-missing values.")

    transformed_values, lambda_value = stats.yeojohnson(non_missing)
    target = _new_column_name(column, "yeojohnson", new_column)
    new_df = _copy_with_empty_new_column(df, target)
    new_df.loc[non_missing.index, target] = transformed_values

    return new_df, _build_log_entry(
        method="yeojohnson",
        source_columns=[column],
        new_column=target,
        parameters={"lambda": float(lambda_value)},
        rows_before=len(df),
        rows_after=len(new_df),
        missing_before=int(values.isna().sum()),
        missing_after=int(new_df[target].isna().sum()),
        notes=f"Created {target} using Yeo-Johnson transformation.",
    )


def create_interaction_term(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a product interaction term from two numeric columns."""
    first = _numeric_series(df, column_a)
    second = _numeric_series(df, column_b)
    target = _new_column_name(f"{column_a}_x_{column_b}", "interaction", new_column)
    return _apply_two_column_transform(
        df,
        column_a,
        column_b,
        target,
        "interaction",
        first * second,
        {},
    )


def create_ratio_feature(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a ratio feature. The denominator must not contain zero values."""
    top = _numeric_series(df, numerator)
    bottom = _numeric_series(df, denominator)
    _require((bottom.dropna() != 0).all(), "Ratio feature requires denominator values to be nonzero.")
    target = _new_column_name(f"{numerator}_over_{denominator}", "ratio", new_column)
    return _apply_two_column_transform(df, numerator, denominator, target, "ratio", top / bottom, {})


def create_polynomial_feature(
    df: pd.DataFrame,
    column: str,
    power: int,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a polynomial feature for one numeric column."""
    _require(power >= 2, "Polynomial power must be 2 or greater.")
    target = _new_column_name(column, f"power_{power}", new_column)
    return _apply_single_column_transform(
        df,
        column,
        target,
        "polynomial",
        lambda series: series**power,
        {"power": power},
    )


def inverse_transform_values(method: str, values: Any, parameters: dict[str, Any] | None = None) -> pd.Series:
    """Invert supported transformations back to the original scale."""
    parameters = parameters or {}
    series = pd.Series(values, dtype="float64")

    if method == "log":
        return np.exp(series)
    if method == "log1p":
        return np.expm1(series)
    if method == "sqrt":
        return series**2
    if method == "cube_root":
        return series**3
    if method == "boxcox":
        return pd.Series(inv_boxcox(series, parameters["lambda"]), index=series.index)
    if method == "yeojohnson":
        return _inverse_yeojohnson(series, parameters["lambda"])

    raise ValueError(f"Inverse transform is not available for method: {method}.")


def inverse_transform_available(method: str) -> bool:
    """Return True when the method has an implemented inverse transform."""
    return method in {"log", "log1p", "sqrt", "cube_root", "boxcox", "yeojohnson"}


def _apply_single_column_transform(
    df: pd.DataFrame,
    column: str,
    new_column: str,
    method: str,
    transform: Callable[[pd.Series], Any],
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a numeric transformation to one column."""
    values = _numeric_series(df, column)
    new_df = _copy_with_empty_new_column(df, new_column)
    new_df[new_column] = transform(values)

    return new_df, _build_log_entry(
        method=method,
        source_columns=[column],
        new_column=new_column,
        parameters=parameters,
        rows_before=len(df),
        rows_after=len(new_df),
        missing_before=int(values.isna().sum()),
        missing_after=int(new_df[new_column].isna().sum()),
        notes=f"Created {new_column} using {method} transformation.",
    )


def _apply_two_column_transform(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
    new_column: str,
    method: str,
    transformed_values: pd.Series,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a feature from two numeric columns."""
    new_df = _copy_with_empty_new_column(df, new_column)
    new_df[new_column] = transformed_values
    missing_before = int(df[[column_a, column_b]].isna().any(axis=1).sum())

    return new_df, _build_log_entry(
        method=method,
        source_columns=[column_a, column_b],
        new_column=new_column,
        parameters=parameters,
        rows_before=len(df),
        rows_after=len(new_df),
        missing_before=missing_before,
        missing_after=int(new_df[new_column].isna().sum()),
        notes=f"Created {new_column} using {method} feature engineering.",
    )


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric version of a column or raise a clear error."""
    _require(column in df.columns, f"Column {column} was not found.")
    values = pd.to_numeric(df[column], errors="coerce")
    invalid_count = int(df[column].notna().sum() - values.notna().sum())
    _require(invalid_count == 0, f"Column {column} must contain numeric values for this transformation.")
    return values


def _copy_with_empty_new_column(df: pd.DataFrame, new_column: str) -> pd.DataFrame:
    """Return a deep copy with a validated empty target column."""
    _require(new_column not in df.columns, f"New column {new_column} already exists. Choose a different name.")
    new_df = df.copy(deep=True)
    new_df[new_column] = np.nan
    return new_df


def _new_column_name(source: str, suffix: str, new_column: str | None) -> str:
    """Create or clean a new column name."""
    if new_column is not None and new_column.strip():
        return new_column.strip()
    return f"{source}_{suffix}"


def _build_log_entry(
    method: str,
    source_columns: list[str],
    new_column: str,
    parameters: dict[str, Any],
    rows_before: int,
    rows_after: int,
    missing_before: int,
    missing_after: int,
    notes: str,
) -> dict[str, Any]:
    """Build a consistent transformation log entry."""
    return {
        "operation_type": "transformation",
        "method": method,
        "source_columns": source_columns,
        "new_column": new_column,
        "parameters": parameters,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "notes": notes,
    }


def _require(condition: bool, message: str) -> None:
    """Raise a clear ValueError when a validation rule fails."""
    if not condition:
        raise ValueError(message)


def _inverse_yeojohnson(values: pd.Series, lambda_value: float) -> pd.Series:
    """Invert a Yeo-Johnson transformation."""
    output = pd.Series(index=values.index, dtype="float64")
    positive_mask = values >= 0
    negative_mask = ~positive_mask

    if abs(lambda_value) < 1e-12:
        output.loc[positive_mask] = np.exp(values.loc[positive_mask]) - 1
    else:
        output.loc[positive_mask] = (values.loc[positive_mask] * lambda_value + 1) ** (1 / lambda_value) - 1

    if abs(lambda_value - 2) < 1e-12:
        output.loc[negative_mask] = 1 - np.exp(-values.loc[negative_mask])
    else:
        output.loc[negative_mask] = 1 - (1 - (2 - lambda_value) * values.loc[negative_mask]) ** (1 / (2 - lambda_value))

    return output
