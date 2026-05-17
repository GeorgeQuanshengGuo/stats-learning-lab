import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from src.prediction.input_builder import build_single_row_input_frame
from src.prediction.prediction_service import (
    PredictionError,
    build_prediction_log_entry,
    predict_from_model_run,
)
from src.modeling.preprocessing import build_preprocessing_pipeline


def _fit_pipeline(data, features, target, model):
    preprocessing = build_preprocessing_pipeline(data, features, scale_numeric=False)
    pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
    pipeline.fit(data[features], data[target])
    return pipeline


def test_prediction_works_for_sklearn_pipeline_regression():
    data = pd.DataFrame(
        {
            "target": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "x": [1, 2, 3, 4, 5, 6],
            "group": ["A", "A", "B", "B", "C", "C"],
        }
    )
    pipeline = _fit_pipeline(data, ["x", "group"], "target", LinearRegression())
    model_run = _model_run("run-reg", "regression", "Linear Regression", ["x", "group"])
    artifact = _artifact(model_run, pipeline)

    result = predict_from_model_run(model_run, artifact, {"x": 4, "group": "B"})

    assert result["task_type"] == "regression"
    assert isinstance(result["predicted_value"], float)
    assert result["interval_available"] is False


def test_prediction_works_for_sklearn_pipeline_binary_classification():
    data = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 1, 0, 1, 0],
            "x": [8, 1, 7, 2, 9, 1, 8, 2],
            "group": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    pipeline = _fit_pipeline(data, ["x", "group"], "target", LogisticRegression(max_iter=1000))
    model_run = _model_run("run-clf", "binary_classification", "Logistic Regression", ["x", "group"])
    model_run["preprocessing"] = {"positive_class": "yes"}
    artifact = _artifact(
        model_run,
        pipeline,
        positive_class="yes",
        target_encoder={"positive_class": "yes", "target_classes": ["no", "yes"]},
    )

    result = predict_from_model_run(model_run, artifact, {"x": 8, "group": "A"}, threshold=0.5)

    assert result["task_type"] == "binary_classification"
    assert result["positive_class"] == "yes"
    assert result["threshold"] == 0.5
    assert result["predicted_class"] in {"yes", "no"}
    assert set(result["class_probabilities"]) == {"yes", "no"}
    assert result["positive_class_probability"] == result["class_probabilities"]["yes"]


def test_missing_artifact_raises_clear_error():
    model_run = _model_run("missing", "regression", "Linear Regression", ["x"])

    with pytest.raises(PredictionError, match="No usable fitted model artifact"):
        predict_from_model_run(model_run, None, {"x": 1})


def test_missing_feature_input_raises_clear_error():
    data = pd.DataFrame({"target": [1, 2, 3, 4], "x": [1, 2, 3, 4], "z": [4, 3, 2, 1]})
    pipeline = _fit_pipeline(data, ["x", "z"], "target", LinearRegression())
    model_run = _model_run("run-reg", "regression", "Linear Regression", ["x", "z"])
    artifact = _artifact(model_run, pipeline)

    with pytest.raises(ValueError, match="Missing feature input values: z"):
        predict_from_model_run(model_run, artifact, {"x": 1})


def test_input_values_are_not_modified():
    feature_names = ["x", "group"]
    input_values = {"x": 1, "group": "A"}
    original = dict(input_values)

    frame = build_single_row_input_frame(feature_names, input_values)

    assert input_values == original
    assert frame.to_dict(orient="records") == [original]


def test_prediction_log_format_is_correct():
    result = {
        "task_type": "binary_classification",
        "predicted_class": "yes",
        "positive_class_probability": 0.8,
        "class_probabilities": {"no": 0.2, "yes": 0.8},
        "threshold": 0.5,
        "interval": None,
    }

    entry = build_prediction_log_entry(
        run_id="run-1",
        model_name="Logistic Regression",
        input_values={"x": 1},
        prediction_result=result,
    )

    assert entry["timestamp"]
    assert entry["run_id"] == "run-1"
    assert entry["model_name"] == "Logistic Regression"
    assert entry["input_values"] == {"x": 1}
    assert entry["prediction"]["predicted_class"] == "yes"
    assert entry["prediction"]["positive_class_probability"] == 0.8
    assert entry["interval"] is None
    assert entry["threshold"] == 0.5


def _model_run(run_id, task_type, model_name, features):
    return {
        "run_id": run_id,
        "task_type": task_type,
        "model_name": model_name,
        "model_family": "machine_learning",
        "target": "target",
        "features": features,
        "preprocessing": {},
    }


def _artifact(model_run, pipeline, positive_class=None, target_encoder=None):
    return {
        "run_id": model_run["run_id"],
        "model_name": model_run["model_name"],
        "model_family": model_run["model_family"],
        "task_type": model_run["task_type"],
        "fitted_pipeline": pipeline,
        "fitted_model": pipeline.named_steps["model"],
        "target": model_run["target"],
        "features": model_run["features"],
        "positive_class": positive_class,
        "target_encoder": target_encoder,
        "prediction_supported": True,
        "interval_supported": False,
    }
