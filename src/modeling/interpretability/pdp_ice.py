"""Partial dependence and ICE helpers for fitted sklearn Pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


PDP_NOTE = (
    "PDP shows the average model response as a feature changes. It is a model "
    "inspection tool, not evidence of a causal effect."
)
ICE_NOTE = (
    "ICE shows individual-level model response curves as a feature changes. "
    "Curves can be misleading when features are highly correlated."
)
CORRELATION_WARNING = (
    "Feature importance, PDP, and ICE can be misleading when features are highly correlated."
)


def build_partial_dependence_table(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    feature: str,
    task_type: str = "regression",
    grid_resolution: int = 20,
    max_samples: int = 500,
) -> pd.DataFrame:
    """Return a one-feature PDP table for a fitted sklearn Pipeline."""
    x_sample = _prepare_x_data(x_data, [feature], max_samples=max_samples)
    grid_values = _feature_grid(x_sample[feature], grid_resolution=grid_resolution)
    rows = []
    for value in grid_values:
        modified = x_sample.copy(deep=True)
        modified[feature] = value
        predictions = _model_response(pipeline, modified, task_type=task_type)
        rows.append(
            {
                "feature": feature,
                "feature_value": value,
                "average_prediction": float(np.mean(predictions)),
                "task_type": task_type,
            }
        )
    return pd.DataFrame(rows)


def build_ice_table(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    feature: str,
    task_type: str = "regression",
    grid_resolution: int = 20,
    max_samples: int = 50,
) -> pd.DataFrame:
    """Return ICE rows for one feature and a limited number of observations."""
    x_sample = _prepare_x_data(x_data, [feature], max_samples=max_samples)
    grid_values = _feature_grid(x_sample[feature], grid_resolution=grid_resolution)
    rows = []
    for row_position, (_, original_row) in enumerate(x_sample.iterrows()):
        base = pd.DataFrame([original_row.to_dict()])
        for value in grid_values:
            modified = base.copy(deep=True)
            modified[feature] = value
            prediction = _model_response(pipeline, modified, task_type=task_type)[0]
            rows.append(
                {
                    "observation_id": row_position,
                    "feature": feature,
                    "feature_value": value,
                    "prediction": float(prediction),
                    "task_type": task_type,
                }
            )
    return pd.DataFrame(rows)


def build_two_feature_partial_dependence_table(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    task_type: str = "regression",
    grid_resolution: int = 10,
    max_samples: int = 300,
) -> pd.DataFrame:
    """Return a two-feature PDP table suitable for a heatmap."""
    if feature_a == feature_b:
        raise ValueError("Choose two different features for a 2D PDP.")

    x_sample = _prepare_x_data(x_data, [feature_a, feature_b], max_samples=max_samples)
    grid_a = _feature_grid(x_sample[feature_a], grid_resolution=grid_resolution)
    grid_b = _feature_grid(x_sample[feature_b], grid_resolution=grid_resolution)
    rows = []
    for value_a in grid_a:
        for value_b in grid_b:
            modified = x_sample.copy(deep=True)
            modified[feature_a] = value_a
            modified[feature_b] = value_b
            predictions = _model_response(pipeline, modified, task_type=task_type)
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "feature_a_value": value_a,
                    "feature_b_value": value_b,
                    "average_prediction": float(np.mean(predictions)),
                    "task_type": task_type,
                }
            )
    return pd.DataFrame(rows)


def _prepare_x_data(
    x_data: pd.DataFrame,
    required_features: list[str],
    max_samples: int,
) -> pd.DataFrame:
    """Return a bounded copy of raw feature data."""
    if not isinstance(x_data, pd.DataFrame):
        raise ValueError("x_data must be a pandas DataFrame.")
    missing_features = [feature for feature in required_features if feature not in x_data.columns]
    if missing_features:
        missing_text = ", ".join(missing_features)
        raise ValueError(f"Feature columns were not found in x_data: {missing_text}")
    if x_data.empty:
        raise ValueError("x_data must contain at least one row.")
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1.")

    x_copy = x_data.copy(deep=True)
    if len(x_copy) > max_samples:
        return x_copy.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return x_copy.reset_index(drop=True)


def _feature_grid(series: pd.Series, grid_resolution: int) -> list[Any]:
    """Build numeric quantile or categorical unique-value grid."""
    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least 2.")

    non_missing = series.dropna()
    if non_missing.empty:
        raise ValueError(f"Feature `{series.name}` has no non-missing values.")

    if pd.api.types.is_numeric_dtype(non_missing):
        unique_values = np.sort(non_missing.unique())
        if len(unique_values) <= grid_resolution:
            return [float(value) for value in unique_values]
        quantiles = np.linspace(0, 1, grid_resolution)
        return [float(value) for value in np.unique(np.quantile(non_missing, quantiles))]

    values = sorted(non_missing.astype(str).unique().tolist())
    if len(values) > grid_resolution:
        return values[:grid_resolution]
    return values


def _model_response(pipeline: Pipeline, x_values: pd.DataFrame, task_type: str) -> np.ndarray:
    """Return predictions or positive-class probabilities for a fitted Pipeline."""
    _validate_pipeline(pipeline)
    if task_type == "binary_classification":
        if not hasattr(pipeline, "predict_proba"):
            raise ValueError("PDP/ICE for classification requires a model with predict_proba.")
        probabilities = pipeline.predict_proba(x_values)
        positive_index = _positive_class_index(pipeline)
        return np.asarray(probabilities[:, positive_index], dtype=float)
    return np.asarray(pipeline.predict(x_values), dtype=float)


def _positive_class_index(pipeline: Pipeline) -> int:
    """Return the encoded positive-class probability column."""
    model = pipeline.named_steps.get("model")
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return classes.index(1)
    if len(classes) == 2:
        return 1
    raise ValueError("Classification PDP/ICE requires a fitted binary classifier.")


def _validate_pipeline(pipeline: Pipeline) -> None:
    """Raise a clear error if the object is not a fitted sklearn Pipeline."""
    if not isinstance(pipeline, Pipeline):
        raise ValueError("A fitted sklearn Pipeline is required.")
    if "model" not in pipeline.named_steps:
        raise ValueError("The Pipeline must include a 'model' step.")
