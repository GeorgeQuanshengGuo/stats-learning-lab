"""Reusable preprocessing builders for machine learning models.

The helpers in this module only build sklearn preprocessing objects. They do
not call fit or transform, so model code can fit them later on training data
inside a Pipeline and avoid data leakage.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build an unfitted preprocessing object for selected feature columns.

    Numeric columns receive median imputation and optional standard scaling.
    Categorical columns receive most-frequent imputation and one-hot encoding.
    The returned ColumnTransformer must be fit later on training data only.
    """
    selected_columns = list(feature_columns)
    _validate_feature_columns(df, selected_columns)

    numeric_columns = _numeric_columns(df, selected_columns)
    categorical_columns = [column for column in selected_columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                _numeric_pipeline(scale_numeric=scale_numeric),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                _categorical_pipeline(),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _validate_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Raise a clear error if feature columns are missing or empty."""
    if not feature_columns:
        raise ValueError("Choose at least one feature column.")

    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Feature columns were not found in the dataset: {missing_text}")


def _numeric_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Return selected columns with numeric pandas dtypes."""
    return [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]


def _numeric_pipeline(scale_numeric: bool) -> Pipeline:
    """Build numeric preprocessing steps."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _categorical_pipeline() -> Pipeline:
    """Build categorical preprocessing steps."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
    )


def _one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with multiple sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
