"""Summary tables for exploratory data analysis."""

from typing import Any

import numpy as np
import pandas as pd

from src.data.type_detector import detect_column_type


def dataset_overview(data: pd.DataFrame) -> dict[str, Any]:
    """Return a small dictionary with dataset-level facts."""
    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "duplicated_rows": int(data.duplicated().sum()),
        "missing_values": int(data.isna().sum().sum()),
    }


def build_summary_table(df: pd.DataFrame, schema: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one EDA summary row per variable.

    Numeric columns receive numeric statistics. Categorical-like columns receive
    top-value frequency statistics. Values that do not apply are left as NaN.
    """
    schema_lookup = _schema_type_lookup(schema)
    rows = []

    for variable in df.columns:
        series = df[variable]
        detected_type = schema_lookup.get(variable, detect_column_type(series, variable))
        numeric_values = pd.to_numeric(series, errors="coerce")
        non_missing = series.dropna()

        row: dict[str, Any] = {
            "variable": variable,
            "raw_dtype": str(series.dtype),
            "detected_type": detected_type,
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "unique_count": int(non_missing.nunique()),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "max": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "top_value": None,
            "top_count": np.nan,
            "top_pct": np.nan,
        }

        if _is_numeric_summary_type(detected_type):
            row.update(_numeric_stats(numeric_values))
        else:
            row.update(_top_value_stats(non_missing, len(series)))

        rows.append(row)

    return pd.DataFrame(rows)


def summary_table(data: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for the Version 0.1 EDA page."""
    return build_summary_table(data)


def _schema_type_lookup(schema: pd.DataFrame | None) -> dict[str, str]:
    """Read detected types from an optional schema table."""
    if schema is None or schema.empty:
        return {}

    if {"variable", "detected_type"}.issubset(schema.columns):
        return dict(zip(schema["variable"], schema["detected_type"]))

    return {}


def _is_numeric_summary_type(detected_type: str) -> bool:
    """Return True for variable types that should get numeric statistics."""
    return detected_type in {
        "continuous_numeric",
        "discrete_numeric",
        "ordinal_categorical_candidate",
    }


def _numeric_stats(values: pd.Series) -> dict[str, float]:
    """Calculate numeric statistics, safely handling all-missing columns."""
    clean_values = values.dropna()
    if clean_values.empty:
        return {}

    return {
        "mean": float(clean_values.mean()),
        "std": float(clean_values.std()),
        "min": float(clean_values.min()),
        "q1": float(clean_values.quantile(0.25)),
        "median": float(clean_values.median()),
        "q3": float(clean_values.quantile(0.75)),
        "max": float(clean_values.max()),
        "skewness": float(clean_values.skew()),
        "kurtosis": float(clean_values.kurtosis()),
    }


def _top_value_stats(non_missing: pd.Series, row_count: int) -> dict[str, Any]:
    """Calculate the most common value and its frequency."""
    if non_missing.empty:
        return {}

    counts = non_missing.value_counts(dropna=True)
    top_value = counts.index[0]
    top_count = int(counts.iloc[0])

    return {
        "top_value": top_value,
        "top_count": top_count,
        "top_pct": float(top_count / row_count * 100) if row_count else np.nan,
    }
