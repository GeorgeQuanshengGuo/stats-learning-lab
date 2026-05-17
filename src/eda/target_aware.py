"""Target-aware EDA helpers.

These functions inspect the selected target and suggest appropriate analysis
paths. They do not fit models and do not modify the input DataFrame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import re

from src.data.type_detector import detect_column_type


NUMERIC_TARGET_TYPES = {"continuous_numeric", "discrete_numeric", "count_like_numeric"}
CATEGORICAL_TARGET_TYPES = {"binary", "nominal_categorical", "ordinal_categorical_candidate"}
COUNT_NAME_PATTERN = re.compile(
    r"(count|num|number|visits|events|frequency|qty|quantity|orders|claims|clicks|cases)",
    re.IGNORECASE,
)


def build_target_profile(
    df: pd.DataFrame,
    target_column: str,
    schema: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return a compact profile for the selected target variable."""
    _validate_column(df, target_column)
    target = df[target_column]
    detected_type = _detected_type(df, target_column, schema)
    target_type = classify_target_type(df, target_column, schema)
    non_missing = target.dropna()

    return {
        "target": target_column,
        "raw_dtype": str(target.dtype),
        "detected_type": detected_type,
        "target_type": target_type,
        "missing_count": int(target.isna().sum()),
        "missing_pct": float(target.isna().mean() * 100),
        "unique_count": int(non_missing.nunique()),
        "example_values": non_missing.astype(str).head(5).tolist(),
    }


def classify_target_type(
    df: pd.DataFrame,
    target_column: str,
    schema: pd.DataFrame | None = None,
) -> str:
    """Classify the target for target-aware EDA and recommendations."""
    _validate_column(df, target_column)
    detected_type = _detected_type(df, target_column, schema)
    target = df[target_column]

    if detected_type == "binary":
        return "binary"
    if detected_type == "ordinal_categorical_candidate":
        return "ordinal_categorical_candidate"
    if _looks_count_like(target, target_column):
        return "count_like_numeric"
    if detected_type in {"continuous_numeric", "discrete_numeric", "nominal_categorical"}:
        return detected_type

    return detected_type


def build_model_recommendations(target_type: str) -> pd.DataFrame:
    """Return beginner-friendly analysis/model recommendations for a target type."""
    recommendations = {
        "continuous_numeric": [
            ("Linear Regression", "available", "Use for interpretable continuous outcomes."),
            ("ML Regression", "available", "Use for predictive baseline regression models."),
        ],
        "discrete_numeric": [
            ("Linear Regression", "available", "Reasonable for numeric outcomes when assumptions are acceptable."),
            ("ML Regression", "available", "Use as a predictive option for numeric outcomes."),
        ],
        "count_like_numeric": [
            ("Poisson / Negative Binomial", "future feature", "Prefer count models for non-negative integer counts."),
            ("ML Regression", "available", "Use as a predictive option, but interpret count predictions carefully."),
        ],
        "binary": [
            ("Logistic Regression", "available", "Use for interpretable binary outcomes."),
            ("ML Binary Classification", "available", "Use for predictive baseline binary classification."),
        ],
        "nominal_categorical": [
            ("Multinomial Logistic Regression", "future feature", "Appropriate for multiclass nominal outcomes."),
            ("ML Multiclass Classification", "future feature", "Predictive option for more than two classes."),
        ],
        "ordinal_categorical_candidate": [
            ("Ordinal Regression", "future feature", "Appropriate when ordered categories are meaningful."),
        ],
    }
    rows = [
        {
            "target_type": target_type,
            "recommended_module": module,
            "status": status,
            "reason": reason,
        }
        for module, status, reason in recommendations.get(
            target_type,
            [("Review target type", "manual review", "No automated recommendation is available for this target type.")],
        )
    ]
    return pd.DataFrame(rows)


def variable_role(detected_type: str) -> str:
    """Map a detected variable type to a broad role for relationship EDA."""
    if detected_type in {"continuous_numeric", "discrete_numeric"}:
        return "numeric"
    if detected_type in CATEGORICAL_TARGET_TYPES:
        return "categorical"
    return "other"


def is_numeric_target(target_type: str) -> bool:
    """Return True when target-aware EDA should treat y as numeric."""
    return target_type in NUMERIC_TARGET_TYPES


def is_categorical_target(target_type: str) -> bool:
    """Return True when target-aware EDA should treat y as categorical."""
    return target_type in CATEGORICAL_TARGET_TYPES


def _detected_type(df: pd.DataFrame, column: str, schema: pd.DataFrame | None) -> str:
    """Read a detected type from schema or detect it directly."""
    if schema is not None and not schema.empty and {"variable", "detected_type"}.issubset(schema.columns):
        matches = schema.loc[schema["variable"] == column, "detected_type"]
        if not matches.empty:
            return str(matches.iloc[0])
    return detect_column_type(df[column], column_name=column)


def _looks_count_like(series: pd.Series, column_name: str) -> bool:
    """Return True for non-negative integer numeric targets with more than two values."""
    if not COUNT_NAME_PATTERN.search(column_name):
        return False

    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_values.empty or len(numeric_values.unique()) <= 2:
        return False
    if (numeric_values < 0).any():
        return False
    return bool(np.isclose(numeric_values % 1, 0).mean() >= 0.95)


def _validate_column(df: pd.DataFrame, column: str) -> None:
    """Raise a clear error when the requested column is not available."""
    if column not in df.columns:
        raise ValueError(f"Column was not found in the dataset: {column}")
