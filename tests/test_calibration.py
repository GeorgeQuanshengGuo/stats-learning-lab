import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from src.modeling.interpretability.calibration import build_binary_calibration_summary
from src.modeling.preprocessing import build_preprocessing_pipeline
from src.visualization.interpretability_plots import (
    plot_calibration_curve,
    plot_probability_histogram,
)


def test_calibration_runs_for_binary_classifier_with_predict_proba():
    data = _classification_data()
    pipeline = _fit_pipeline(data, ["x", "group"], "target", LogisticRegression(max_iter=1000))

    summary = build_binary_calibration_summary(
        pipeline,
        data[["x", "group"]],
        data["target"],
        positive_label=1,
        n_bins=4,
    )

    assert not summary["calibration_table"].empty
    assert not summary["probability_histogram"].empty
    assert 0 <= summary["brier_score"] <= 1
    assert "Calibration checks" in summary["note"]
    assert plot_calibration_curve(summary["calibration_table"]) is not None
    assert plot_probability_histogram(summary["probability_histogram"]) is not None


def test_calibration_handles_named_positive_label():
    data = _classification_data().copy()
    data["target"] = data["target"].map({0: "no", 1: "yes"})
    encoded = data.copy()
    encoded["target"] = (encoded["target"] == "yes").astype(int)
    pipeline = _fit_pipeline(encoded, ["x", "group"], "target", LogisticRegression(max_iter=1000))

    summary = build_binary_calibration_summary(
        pipeline,
        data[["x", "group"]],
        data["target"],
        positive_label="yes",
        n_bins=4,
    )

    assert not summary["calibration_table"].empty
    assert 0 <= summary["brier_score"] <= 1


def test_calibration_unsupported_model_returns_clear_message():
    data = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0], "x": [1, 2, 3, 4]})
    pipeline = _fit_pipeline(data, ["x"], "target", LinearRegression())

    with pytest.raises(ValueError, match="predict_proba"):
        build_binary_calibration_summary(
            pipeline,
            data[["x"]],
            [0, 1, 0, 1],
            positive_label=1,
            n_bins=3,
        )


def test_calibration_requires_binary_target():
    data = _classification_data()
    pipeline = _fit_pipeline(data, ["x", "group"], "target", LogisticRegression(max_iter=1000))

    with pytest.raises(ValueError, match="exactly two"):
        build_binary_calibration_summary(
            pipeline,
            data[["x", "group"]],
            [0, 1, 2, 0, 1, 2, 0, 1],
            positive_label=1,
            n_bins=4,
        )


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


def _classification_data():
    return pd.DataFrame(
        {
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
            "x": [1, 2, 2, 3, 7, 8, 9, 10],
            "group": ["A", "A", "B", "B", "C", "C", "A", "B"],
        }
    )
