"""Multicollinearity diagnostics for selected predictor variables."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.modeling.diagnostics.diagnostic_rules import vif_warning


def compute_vif_table(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Compute VIF for numeric and one-hot encoded predictor columns.

    VIF is a multicollinearity diagnostic. It is not an overfitting diagnostic.
    """
    design = _encoded_design_matrix(df, feature_columns)
    if design.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "vif", "risk_level", "message"])
    if design.shape[1] == 1:
        return pd.DataFrame(
            [
                {
                    "feature": design.columns[0],
                    "vif": None,
                    "risk_level": "not_available",
                    "message": "VIF requires at least two numeric design matrix columns.",
                }
            ]
        )

    values = design.to_numpy(dtype=float)
    rows = []
    for index, feature in enumerate(design.columns):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vif_value = float(variance_inflation_factor(values, index))
        except Exception:
            vif_value = np.inf
        level, message = vif_warning(None if np.isnan(vif_value) else vif_value)
        rows.append(
            {
                "feature": feature,
                "vif": vif_value,
                "risk_level": level,
                "message": message,
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False, na_position="last").reset_index(drop=True)


def compute_predictor_correlation_table(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return pairwise correlations among encoded numeric predictor columns."""
    design = _encoded_design_matrix(df, feature_columns)
    if design.shape[1] < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation"])

    corr = design.corr()
    rows = []
    for left_index, feature_1 in enumerate(corr.columns):
        for feature_2 in corr.columns[left_index + 1 :]:
            value = corr.loc[feature_1, feature_2]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": float(value),
                    "abs_correlation": float(abs(value)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation"])
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def compute_condition_number(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Compute a condition number for the encoded numeric design matrix."""
    design = _encoded_design_matrix(df, feature_columns)
    if design.shape[1] < 2:
        return {
            "condition_number": None,
            "risk_level": "not_available",
            "message": "Condition number requires at least two numeric design matrix columns.",
        }

    values = design.to_numpy(dtype=float)
    values = values - values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1
    values = values / std
    try:
        condition_number = float(np.linalg.cond(values))
    except Exception:
        condition_number = None

    if condition_number is None or np.isnan(condition_number):
        return {
            "condition_number": None,
            "risk_level": "not_available",
            "message": "Condition number could not be computed.",
        }
    if condition_number > 30:
        level = "warning"
        message = "Condition number is above 30, which can indicate multicollinearity or numerical instability."
    else:
        level = "ok"
        message = "Condition number is not high by the common > 30 heuristic."
    return {
        "condition_number": condition_number,
        "risk_level": level,
        "message": message,
    }


def _encoded_design_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return a complete numeric design matrix for VIF/correlation diagnostics."""
    _validate_inputs(df, feature_columns)
    data = df[feature_columns].copy(deep=True)
    numeric_columns = [column for column in data.columns if pd.api.types.is_numeric_dtype(data[column])]
    categorical_columns = [column for column in data.columns if column not in numeric_columns]

    parts = []
    if numeric_columns:
        numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(numeric.median(numeric_only=True))
        parts.append(numeric)
    if categorical_columns:
        categorical = data[categorical_columns].copy(deep=True)
        for column in categorical.columns:
            mode = categorical[column].dropna().mode()
            fill_value = mode.iloc[0] if not mode.empty else "Missing"
            categorical[column] = categorical[column].fillna(fill_value).astype(str)
        encoded = pd.get_dummies(categorical, drop_first=True, dtype=float)
        if not encoded.empty:
            parts.append(encoded)

    if not parts:
        return pd.DataFrame(index=data.index)

    design = pd.concat(parts, axis=1)
    design = design.loc[:, design.nunique(dropna=False) > 1]
    return design.astype(float)


def _validate_inputs(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Validate requested columns."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if not feature_columns:
        raise ValueError("Choose at least one feature column.")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Feature columns were not found in the dataset: {', '.join(missing)}")
