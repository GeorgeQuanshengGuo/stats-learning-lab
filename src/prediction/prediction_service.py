"""Prediction service for saved fitted model artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.prediction.input_builder import build_single_row_input_frame
from src.prediction.statistical_intervals import (
    predict_logit_probability_with_ci,
    predict_ols_with_intervals,
)
from src.modeling.statistical.ordinal_regression import predict_ordered_model
from src.modeling.statistical.count_regression import predict_count_with_mean_ci


class PredictionError(ValueError):
    """Raised when a saved run cannot be used for prediction."""


def is_artifact_usable_for_prediction(model_run: dict[str, Any], artifact: dict[str, Any] | None) -> bool:
    """Return True when a ModelRun has a fitted object usable for prediction."""
    if not artifact:
        return False
    if artifact.get("prediction_supported") is False:
        return False
    return artifact.get("fitted_pipeline") is not None or artifact.get("fitted_model") is not None


def predict_from_model_run(
    model_run: dict[str, Any],
    artifact: dict[str, Any] | None,
    input_values: dict[str, Any],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Predict one observation using a saved ModelRun and fitted artifact."""
    if not is_artifact_usable_for_prediction(model_run, artifact):
        run_id = model_run.get("run_id", "unknown")
        raise PredictionError(f"No usable fitted model artifact is available for run_id: {run_id}")

    features = list(artifact.get("features") or model_run.get("features") or [])
    input_frame = build_single_row_input_frame(features, input_values)

    if artifact.get("fitted_pipeline") is not None:
        result = _predict_with_pipeline(model_run, artifact, input_frame, threshold)
    else:
        result = _predict_with_statsmodels(model_run, artifact, input_frame, threshold)

    result["input_frame"] = input_frame
    return result


def build_prediction_log_entry(
    run_id: str,
    model_name: str,
    input_values: dict[str, Any],
    prediction_result: dict[str, Any],
) -> dict[str, Any]:
    """Return one session-safe prediction log entry."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_name": model_name,
        "input_values": dict(input_values),
        "prediction": _log_prediction_value(prediction_result),
        "interval": prediction_result.get("interval"),
        "threshold": prediction_result.get("threshold"),
    }


def _predict_with_pipeline(
    model_run: dict[str, Any],
    artifact: dict[str, Any],
    input_frame: pd.DataFrame,
    threshold: float | None,
) -> dict[str, Any]:
    """Predict with a fitted sklearn Pipeline."""
    pipeline = artifact["fitted_pipeline"]
    task_type = model_run.get("task_type") or artifact.get("task_type")

    if task_type == "regression":
        prediction = pipeline.predict(input_frame)[0]
        return {
            "task_type": "regression",
            "predicted_value": _python_scalar(prediction),
            "interval": None,
            "interval_available": False,
            "interval_message": "Interval is not available for this model.",
        }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_frame)[0]
        model = pipeline.named_steps.get("model")
        classes = list(getattr(model, "classes_", range(len(probabilities))))
        probability_by_class = _probability_dict(artifact, classes, probabilities)
    else:
        probabilities = None
        probability_by_class = {}

    if task_type == "binary_classification":
        used_threshold = _classification_threshold(model_run, artifact, threshold)
        positive_probability = _positive_class_probability(artifact, probabilities, probability_by_class)
        positive_class = artifact.get("positive_class") or _preprocessing_value(model_run, "positive_class")
        negative_class = _negative_class_label(artifact, positive_class)

        if positive_probability is not None:
            predicted_class = positive_class if positive_probability >= used_threshold else negative_class
        else:
            predicted_encoded = pipeline.predict(input_frame)[0]
            predicted_class = positive_class if predicted_encoded == 1 else negative_class

        return {
            "task_type": "binary_classification",
            "positive_class": positive_class,
            "positive_class_probability": positive_probability,
            "threshold": used_threshold,
            "predicted_class": _python_scalar(predicted_class),
            "class_probabilities": probability_by_class,
            "interval": None,
            "interval_available": False,
            "interval_message": "Interval is not available for this model.",
        }

    predicted_class = pipeline.predict(input_frame)[0]
    return {
        "task_type": "classification",
        "predicted_class": _python_scalar(predicted_class),
        "class_probabilities": probability_by_class,
        "interval": None,
        "interval_available": False,
        "interval_message": "Interval is not available for this model.",
    }


def _predict_with_statsmodels(
    model_run: dict[str, Any],
    artifact: dict[str, Any],
    input_frame: pd.DataFrame,
    threshold: float | None,
) -> dict[str, Any]:
    """Predict with a fitted statsmodels object when design columns are available."""
    model = artifact["fitted_model"]
    task_type = model_run.get("task_type") or artifact.get("task_type")

    if task_type == "ordinal_classification":
        ordinal_prediction = predict_ordered_model(
            model,
            input_frame,
            design_columns=list(artifact.get("design_columns") or []),
            category_order=list((artifact.get("target_encoder") or {}).get("category_order") or []),
        )
        predicted_category = ordinal_prediction["predicted_categories"].iloc[0]
        class_probabilities = ordinal_prediction["probabilities"].iloc[0].to_dict()
        return {
            "task_type": "ordinal_classification",
            "predicted_class": _python_scalar(predicted_category),
            "class_probabilities": {
                str(class_name): float(probability)
                for class_name, probability in class_probabilities.items()
            },
            "interval": None,
            "interval_available": False,
            "interval_message": "Interval is not available for this ordinal model.",
        }

    if task_type == "count_regression":
        prediction_table = predict_count_with_mean_ci(
            model,
            input_frame,
            design_columns=list(artifact.get("design_columns") or []),
            alpha=0.05,
        )
        row = prediction_table.iloc[0].to_dict()
        interval = prediction_table.where(pd.notna(prediction_table), None).to_dict(orient="records")
        return {
            "task_type": "regression",
            "predicted_value": float(row["predicted_expected_count"]),
            "predicted_expected_count": float(row["predicted_expected_count"]),
            "interval": interval,
            "interval_available": True,
            "interval_message": row.get("interval_explanation"),
        }

    design = _statsmodels_design_frame(input_frame, artifact)
    if task_type == "binary_classification":
        used_threshold = _classification_threshold(model_run, artifact, threshold)
        positive_class = artifact.get("positive_class") or _preprocessing_value(model_run, "positive_class")
        negative_class = _negative_class_label(artifact, positive_class)
        probability_table = predict_logit_probability_with_ci(
            model,
            design,
            alpha=0.05,
            threshold=used_threshold,
            positive_class=positive_class,
            negative_class=negative_class,
        )
        row = probability_table.iloc[0].to_dict()
        probability = float(row["predicted_probability"])
        interval = probability_table.where(pd.notna(probability_table), None).to_dict(orient="records")
        return {
            "task_type": "binary_classification",
            "positive_class": positive_class,
            "positive_class_probability": probability,
            "threshold": used_threshold,
            "predicted_class": row["predicted_class"],
            "class_probabilities": {
                str(negative_class): float(1 - probability),
                str(positive_class): probability,
            },
            "interval": interval,
            "interval_available": pd.notna(row.get("probability_ci_lower"))
            and pd.notna(row.get("probability_ci_upper")),
            "interval_message": row.get("interval_message"),
        }

    try:
        ols_prediction = predict_ols_with_intervals(model, design, alpha=0.05)
    except Exception as error:
        prediction = float(model.predict(design)[0])
        return {
            "task_type": "regression",
            "predicted_value": prediction,
            "interval": None,
            "interval_available": False,
            "interval_message": f"Interval is not available for this statistical model: {error}",
        }

    interval = ols_prediction.where(pd.notna(ols_prediction), None).to_dict(orient="records")
    predicted_value = float(ols_prediction["predicted_mean"].iloc[0])
    return {
        "task_type": "regression",
        "predicted_value": predicted_value,
        "interval": interval,
        "interval_available": True,
        "interval_message": None,
    }


def _statsmodels_design_frame(input_frame: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Build a statsmodels design row from raw inputs and saved design columns."""
    design_columns = list(artifact.get("design_columns") or [])
    if not design_columns:
        raise PredictionError("Statsmodels prediction requires saved design_columns metadata.")

    encoded = pd.get_dummies(input_frame, dtype=float)
    encoded = sm.add_constant(encoded, has_constant="add")
    return encoded.reindex(columns=design_columns, fill_value=0).astype(float)


def _classification_threshold(
    model_run: dict[str, Any],
    artifact: dict[str, Any],
    threshold: float | None,
) -> float:
    """Return selected classification threshold."""
    if threshold is not None:
        return float(threshold)
    preprocessing = artifact.get("preprocessing_summary") or model_run.get("preprocessing") or {}
    return float(preprocessing.get("threshold", 0.5))


def _positive_class_probability(
    artifact: dict[str, Any],
    probabilities: Any,
    probability_by_class: dict[str, float],
) -> float | None:
    """Return probability for encoded positive class when available."""
    if probabilities is None:
        return None
    positive_class = artifact.get("positive_class")
    if positive_class is not None and str(positive_class) in probability_by_class:
        return probability_by_class[str(positive_class)]
    if "1" in probability_by_class:
        return probability_by_class["1"]
    if len(probabilities) == 2:
        return float(probabilities[-1])
    return None


def _negative_class_label(artifact: dict[str, Any], positive_class: Any) -> Any:
    """Infer the negative class label for binary classification displays."""
    target_encoder = artifact.get("target_encoder") or {}
    target_classes = target_encoder.get("target_classes") or []
    for class_value in target_classes:
        if class_value != positive_class:
            return class_value
    return 0


def _probability_dict(artifact: dict[str, Any], classes: list[Any], probabilities: Any) -> dict[str, float]:
    """Return class probabilities, mapping encoded 0/1 labels back when possible."""
    positive_class = artifact.get("positive_class")
    negative_class = _negative_class_label(artifact, positive_class)
    labels = []
    for class_value in classes:
        if class_value == 1 and positive_class is not None:
            labels.append(positive_class)
        elif class_value == 0 and positive_class is not None:
            labels.append(negative_class)
        else:
            labels.append(class_value)
    return {
        str(class_value): float(probability)
        for class_value, probability in zip(labels, probabilities)
    }


def _preprocessing_value(model_run: dict[str, Any], key: str) -> Any:
    """Return one preprocessing value from a ModelRun."""
    return (model_run.get("preprocessing") or {}).get(key)


def _log_prediction_value(prediction_result: dict[str, Any]) -> Any:
    """Return the compact value stored in prediction_log."""
    if "predicted_value" in prediction_result:
        return prediction_result["predicted_value"]
    if "predicted_class" in prediction_result:
        return {
            "predicted_class": prediction_result["predicted_class"],
            "positive_class_probability": prediction_result.get("positive_class_probability"),
            "class_probabilities": prediction_result.get("class_probabilities"),
        }
    return None


def _python_scalar(value: Any) -> Any:
    """Convert numpy scalar values into plain Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value
