"""Interpretability helpers for fitted sklearn decision tree Pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import export_text

from src.modeling.interpretability.coefficients import get_transformed_feature_names_from_pipeline
from src.visualization.tree_plots import create_tree_plot_figure as _create_tree_plot_figure


TREE_EXPLANATION_WARNING = (
    "Deep trees may overfit and become difficult to interpret. For explanation, prefer max_depth between 2 and 5."
)


def extract_tree_rules(pipeline, max_depth: int = 5) -> str:
    """Return text rules for a fitted DecisionTree Pipeline."""
    model = _tree_model(pipeline)
    if model is None:
        return "No decision tree rules are available for this model."

    feature_names = _feature_names_for_tree(pipeline)
    return export_text(model, feature_names=feature_names, max_depth=max_depth)


def extract_tree_feature_importance(pipeline) -> pd.DataFrame:
    """Return decision tree feature importance aligned with transformed features."""
    model = _tree_model(pipeline)
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = _feature_names_for_tree(pipeline)
    importances = model.feature_importances_
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{index}" for index in range(len(importances))]

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def create_tree_plot_figure(pipeline, max_depth: int = 4):
    """Create a matplotlib figure for a fitted DecisionTree Pipeline."""
    model = _tree_model(pipeline)
    if model is None:
        return None
    return _create_tree_plot_figure(
        pipeline,
        feature_names=_feature_names_for_tree(pipeline),
        max_depth=max_depth,
    )


def get_decision_path_for_observation(pipeline, input_row) -> pd.DataFrame:
    """Return the rule path followed by one raw input observation."""
    model = _tree_model(pipeline)
    if model is None:
        return pd.DataFrame(columns=["step", "node_id", "feature", "threshold", "value", "direction", "rule"])

    transformed = _transform_one_row(pipeline, input_row)
    transformed_array = _as_dense_array(transformed)
    node_indicator = model.decision_path(transformed)
    leaf_id = int(model.apply(transformed)[0])
    feature_names = _feature_names_for_tree(pipeline)

    node_ids = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
    rows = []
    step = 1
    for node_id in node_ids:
        node_id = int(node_id)
        if node_id == leaf_id:
            continue

        feature_index = int(model.tree_.feature[node_id])
        if feature_index < 0:
            continue

        threshold = float(model.tree_.threshold[node_id])
        value = float(transformed_array[0, feature_index])
        direction = "<=" if value <= threshold else ">"
        feature_name = feature_names[feature_index] if feature_index < len(feature_names) else f"feature_{feature_index}"
        rows.append(
            {
                "step": step,
                "node_id": node_id,
                "feature": feature_name,
                "threshold": threshold,
                "value": value,
                "direction": direction,
                "rule": f"{feature_name} {direction} {threshold:.4g}",
            }
        )
        step += 1

    return pd.DataFrame(rows)


def summarize_leaf_prediction(pipeline, input_row) -> dict[str, Any]:
    """Return the leaf prediction summary for one raw input observation."""
    model = _tree_model(pipeline)
    if model is None:
        return {"prediction_supported": False}

    row_df = _as_single_row_frame(input_row)
    transformed = _transform_one_row(pipeline, row_df)
    leaf_id = int(model.apply(transformed)[0])
    leaf_samples = int(model.tree_.n_node_samples[leaf_id])
    prediction = pipeline.predict(row_df)[0]

    if hasattr(model, "predict_proba"):
        probabilities = pipeline.predict_proba(row_df)[0]
        class_labels = [str(value) for value in model.classes_]
        return {
            "task_type": "binary_classification" if len(class_labels) == 2 else "classification",
            "leaf_id": leaf_id,
            "predicted_class": prediction,
            "predicted_probabilities": {
                label: float(probability)
                for label, probability in zip(class_labels, probabilities)
            },
            "leaf_samples": leaf_samples,
            "prediction_supported": True,
        }

    return {
        "task_type": "regression",
        "leaf_id": leaf_id,
        "predicted_value": float(prediction),
        "leaf_samples": leaf_samples,
        "prediction_supported": True,
    }


def _tree_model(pipeline):
    """Return the fitted decision tree model step if available."""
    if not hasattr(pipeline, "named_steps"):
        return None
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "tree_"):
        return None
    return model


def _feature_names_for_tree(pipeline) -> list[str]:
    """Return transformed feature names for a fitted tree Pipeline."""
    model = _tree_model(pipeline)
    feature_names = get_transformed_feature_names_from_pipeline(pipeline)
    expected_count = getattr(model, "n_features_in_", len(feature_names)) if model is not None else len(feature_names)
    if len(feature_names) != expected_count:
        return [f"feature_{index}" for index in range(expected_count)]
    return feature_names


def _transform_one_row(pipeline, input_row):
    """Transform one raw observation with the fitted preprocessing step."""
    preprocessing = pipeline.named_steps.get("preprocessing")
    if preprocessing is None:
        return _as_single_row_frame(input_row).to_numpy()
    return preprocessing.transform(_as_single_row_frame(input_row))


def _as_single_row_frame(input_row) -> pd.DataFrame:
    """Normalize dict, Series, or one-row DataFrame input into a DataFrame."""
    if isinstance(input_row, pd.DataFrame):
        if len(input_row) != 1:
            return input_row.iloc[[0]].copy()
        return input_row.copy()
    if isinstance(input_row, pd.Series):
        return input_row.to_frame().T
    if isinstance(input_row, dict):
        return pd.DataFrame([input_row])
    return pd.DataFrame(input_row).iloc[[0]].copy()


def _as_dense_array(values) -> np.ndarray:
    """Return a dense numpy array from sparse or dense transformed values."""
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)
