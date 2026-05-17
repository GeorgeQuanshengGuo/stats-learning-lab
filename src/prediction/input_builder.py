"""Helpers for building single-row prediction inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_input_specs(
    feature_names: list[str],
    reference_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Return simple UI-friendly input specifications for raw feature names."""
    specs = []
    for feature in feature_names:
        series = reference_df[feature] if reference_df is not None and feature in reference_df.columns else None
        specs.append(
            {
                "name": feature,
                "kind": _feature_kind(series),
                "options": _feature_options(series),
                "default": _feature_default(series),
            }
        )
    return specs


def build_single_row_input_frame(
    feature_names: list[str],
    input_values: dict[str, Any],
) -> pd.DataFrame:
    """Build a one-row DataFrame using required raw feature names."""
    missing = [feature for feature in feature_names if feature not in input_values]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing feature input values: {missing_text}")

    row = {feature: input_values[feature] for feature in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def _feature_kind(series: pd.Series | None) -> str:
    """Infer a small input kind for Streamlit widgets."""
    if series is None:
        return "text"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _feature_options(series: pd.Series | None) -> list[Any]:
    """Return compact options for categorical-like fields."""
    if series is None:
        return []
    values = series.dropna().unique().tolist()
    if len(values) <= 50 and not pd.api.types.is_numeric_dtype(series):
        return sorted(values, key=lambda value: str(value))
    return []


def _feature_default(series: pd.Series | None) -> Any:
    """Return a reasonable default value for one feature input."""
    if series is None or series.dropna().empty:
        return ""
    if pd.api.types.is_numeric_dtype(series):
        return float(series.dropna().median())
    if pd.api.types.is_bool_dtype(series):
        return bool(series.dropna().mode().iloc[0])
    mode = series.dropna().mode()
    if not mode.empty:
        return mode.iloc[0]
    return series.dropna().iloc[0]
