"""Automatic transformation suggestions for numeric variables.

Suggestions are advisory only. They do not modify data and do not know about a
target variable yet.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


SUGGESTION_COLUMNS = [
    "variable",
    "detected_issue",
    "min",
    "max",
    "skewness",
    "kurtosis",
    "zero_count",
    "negative_count",
    "missing_count",
    "suggested_methods",
    "reason",
    "warning",
]


def build_transformation_suggestions(df: pd.DataFrame, schema: pd.DataFrame | None = None) -> pd.DataFrame:
    """Inspect numeric variables and return transformation suggestions."""
    rows = []

    for variable in _numeric_variables(df, schema):
        series = pd.to_numeric(df[variable], errors="coerce")
        non_missing = series.dropna()

        if non_missing.empty:
            rows.append(_empty_row(variable, series, "all_missing", "No non-missing numeric values are available."))
            continue

        unique_count = int(non_missing.nunique())
        minimum = float(non_missing.min())
        maximum = float(non_missing.max())
        skewness = float(non_missing.skew()) if len(non_missing) >= 3 else 0.0
        kurtosis = float(non_missing.kurtosis()) if len(non_missing) >= 4 else 0.0
        zero_count = int((non_missing == 0).sum())
        negative_count = int((non_missing < 0).sum())
        outlier_count = _iqr_outlier_count(non_missing)
        is_count_like = _is_count_like(non_missing)

        detected_issue, methods, reason, warning = _suggest_for_values(
            unique_count=unique_count,
            minimum=minimum,
            skewness=skewness,
            zero_count=zero_count,
            negative_count=negative_count,
            outlier_count=outlier_count,
            is_count_like=is_count_like,
            row_count=len(non_missing),
        )

        rows.append(
            {
                "variable": variable,
                "detected_issue": detected_issue,
                "min": minimum,
                "max": maximum,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "zero_count": zero_count,
                "negative_count": negative_count,
                "missing_count": int(series.isna().sum()),
                "suggested_methods": methods,
                "reason": reason,
                "warning": warning,
            }
        )

    return pd.DataFrame(rows, columns=SUGGESTION_COLUMNS)


def _suggest_for_values(
    unique_count: int,
    minimum: float,
    skewness: float,
    zero_count: int,
    negative_count: int,
    outlier_count: int,
    is_count_like: bool,
    row_count: int,
) -> tuple[str, list[str], str, str]:
    """Apply rule-based suggestion logic for one variable."""
    abs_skewness = abs(skewness)
    right_skewed = skewness >= 1.0
    skewed = abs_skewness >= 1.0
    outlier_rate = outlier_count / row_count if row_count else 0

    if unique_count <= 5:
        return (
            "too_few_unique_values",
            [],
            "The variable has very few unique values, so continuous transformations are not strongly suggested.",
            "Consider treating this variable as discrete or categorical.",
        )

    if is_count_like and right_skewed:
        return (
            "count_like_right_skew",
            ["Square root", "Log1p", "Yeo-Johnson"],
            "The variable looks count-like and right-skewed.",
            "If this is the target variable, count models such as Poisson or Negative Binomial may be more appropriate.",
        )

    if negative_count > 0 and skewed:
        return (
            "skewed_with_negative_values",
            ["Yeo-Johnson", "Cube root"],
            "The variable contains negative values and is skewed.",
            "",
        )

    if zero_count > 0 and right_skewed:
        return (
            "right_skewed_with_zero_values",
            ["Log1p", "Square root", "Yeo-Johnson"],
            "The variable contains zero values and is right-skewed.",
            "",
        )

    if minimum > 0 and skewed:
        return (
            "positive_high_skew",
            ["Log", "Square root", "Box-Cox", "Yeo-Johnson"],
            "The variable is strictly positive and has high absolute skewness.",
            "",
        )

    if outlier_rate >= 0.05:
        return (
            "many_iqr_outliers",
            [],
            "The variable has many outliers by the IQR rule.",
            "Robust scaling or winsorization may be useful future options.",
        )

    if abs_skewness <= 0.5:
        return (
            "roughly_symmetric",
            [],
            "No transformation is strongly suggested because the variable is roughly symmetric.",
            "",
        )

    return (
        "mild_skew",
        ["Yeo-Johnson"],
        "The variable has mild skewness. A transformation may help, but it is not strongly indicated.",
        "",
    )


def _numeric_variables(df: pd.DataFrame, schema: pd.DataFrame | None) -> list[str]:
    """Return numeric variables, optionally using a schema table."""
    if schema is not None and {"variable", "detected_type"}.issubset(schema.columns):
        numeric_types = {"continuous_numeric", "discrete_numeric"}
        return schema.loc[schema["detected_type"].isin(numeric_types), "variable"].tolist()

    return df.select_dtypes(include="number").columns.tolist()


def _is_count_like(values: pd.Series) -> bool:
    """Return True when values look like nonnegative integer counts."""
    if (values < 0).any():
        return False

    mostly_integer = ((values % 1).abs() < 1e-9).mean() >= 0.95
    modest_scale = values.max() <= 100
    return bool(mostly_integer and modest_scale)


def _iqr_outlier_count(values: pd.Series) -> int:
    """Count outliers using the 1.5 * IQR rule."""
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((values < lower) | (values > upper)).sum())


def _empty_row(variable: str, series: pd.Series, issue: str, reason: str) -> dict[str, Any]:
    """Build a suggestion row for columns that cannot be summarized."""
    return {
        "variable": variable,
        "detected_issue": issue,
        "min": None,
        "max": None,
        "skewness": None,
        "kurtosis": None,
        "zero_count": 0,
        "negative_count": 0,
        "missing_count": int(series.isna().sum()),
        "suggested_methods": [],
        "reason": reason,
        "warning": "",
    }
