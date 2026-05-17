"""Variable type detection helpers.

The goal of this module is to give beginner-friendly, practical labels for
columns before any cleaning or modeling happens. The rules are intentionally
transparent so a user can understand why a column received a type.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


ID_NAME_PATTERN = re.compile(r"(id|code|key|uuid|identifier)", re.IGNORECASE)

ORDINAL_PATTERNS = [
    {"low", "medium", "high"},
    {"poor", "fair", "good", "excellent"},
    {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
    {"very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"},
]


def detect_variable_types(data: pd.DataFrame) -> pd.DataFrame:
    """Return a variable type summary for every column in a DataFrame."""
    rows = []

    for column in data.columns:
        series = data[column]
        non_missing = series.dropna()
        example_values = non_missing.astype(str).head(5).tolist()

        rows.append(
            {
                "variable": column,
                "raw_dtype": str(series.dtype),
                "detected_type": detect_column_type(series, column_name=column),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "unique_count": int(non_missing.nunique()),
                "example_values": example_values,
            }
        )

    return pd.DataFrame(rows)


def detect_column_type(series: pd.Series, column_name: str = "") -> str:
    """Classify one column using transparent rule-based checks."""
    non_missing = series.dropna()
    non_missing_count = len(non_missing)
    unique_count = int(non_missing.nunique())

    if unique_count <= 1:
        return "constant"

    if unique_count == 2:
        return "binary"

    if _looks_like_datetime(non_missing):
        return "datetime"

    if _looks_like_id(column_name, unique_count, non_missing_count):
        return "id_like"

    if _looks_like_text(non_missing, unique_count, non_missing_count):
        return "text"

    if _looks_like_ordinal(non_missing):
        return "ordinal_categorical_candidate"

    numeric_string_type = _numeric_string_type(non_missing, unique_count, non_missing_count)
    if numeric_string_type is not None:
        return numeric_string_type

    if pd.api.types.is_numeric_dtype(non_missing):
        if _mostly_integer_values(non_missing) and _has_low_unique_count(unique_count, non_missing_count):
            return "discrete_numeric"
        return "continuous_numeric"

    return "nominal_categorical"


def _looks_like_datetime(non_missing: pd.Series) -> bool:
    """Check whether values can be parsed as datetimes with high success."""
    if pd.api.types.is_datetime64_any_dtype(non_missing):
        return True

    if pd.api.types.is_numeric_dtype(non_missing):
        return False

    parsed = pd.to_datetime(non_missing, errors="coerce", format="mixed")
    success_rate = parsed.notna().mean()
    return bool(success_rate >= 0.8)


def _looks_like_id(column_name: str, unique_count: int, non_missing_count: int) -> bool:
    """Detect high-cardinality identifier-style columns by name and uniqueness."""
    if non_missing_count == 0:
        return False

    unique_ratio = unique_count / non_missing_count
    return bool(ID_NAME_PATTERN.search(column_name) and unique_ratio >= 0.9)


def _looks_like_text(non_missing: pd.Series, unique_count: int, non_missing_count: int) -> bool:
    """Detect long, high-cardinality string columns."""
    if non_missing_count == 0 or pd.api.types.is_numeric_dtype(non_missing):
        return False

    as_text = non_missing.astype(str)
    average_length = as_text.str.len().mean()
    unique_ratio = unique_count / non_missing_count

    return bool(average_length >= 30 and unique_ratio >= 0.5 and unique_count >= 10)


def _looks_like_ordinal(non_missing: pd.Series) -> bool:
    """Detect common ordered categories and 1-to-5 rating scales."""
    normalized_values = {_normalize_value(value) for value in non_missing.unique()}

    if _is_numeric_rating_1_to_5(non_missing):
        return True

    for pattern in ORDINAL_PATTERNS:
        if normalized_values.issubset(pattern) and len(normalized_values) >= 3:
            return True

    return False


def _is_numeric_rating_1_to_5(non_missing: pd.Series) -> bool:
    """Return True when numeric values look like a 1-to-5 rating scale."""
    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    if numeric_values.isna().any():
        return False

    unique_values = set(numeric_values.dropna().astype(float).unique())
    valid_rating_values = {1.0, 2.0, 3.0, 4.0, 5.0}

    return unique_values.issubset(valid_rating_values) and len(unique_values) >= 3


def _numeric_string_type(non_missing: pd.Series, unique_count: int, non_missing_count: int) -> str | None:
    """Detect numeric columns that arrived as strings."""
    if pd.api.types.is_numeric_dtype(non_missing):
        return None

    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    if numeric_values.notna().mean() < 0.95:
        return None

    if _mostly_integer_values(numeric_values) and _has_low_unique_count(unique_count, non_missing_count):
        return "discrete_numeric"
    return "continuous_numeric"


def _mostly_integer_values(non_missing: pd.Series) -> bool:
    """Return True when numeric values are nearly all whole numbers."""
    numeric_values = pd.to_numeric(non_missing, errors="coerce").dropna()
    if numeric_values.empty:
        return False

    remainders = np.isclose(numeric_values % 1, 0)
    return bool(remainders.mean() >= 0.95)


def _has_low_unique_count(unique_count: int, non_missing_count: int) -> bool:
    """Return True when a numeric column has relatively few distinct values."""
    if non_missing_count == 0:
        return False

    unique_ratio = unique_count / non_missing_count
    return unique_count <= 20 or unique_ratio <= 0.1


def _normalize_value(value: Any) -> str:
    """Normalize category labels for simple pattern matching."""
    return str(value).strip().lower().replace("_", " ").replace("-", " ")
