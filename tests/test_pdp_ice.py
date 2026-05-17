import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from src.modeling.interpretability.pdp_ice import (
    build_ice_table,
    build_partial_dependence_table,
    build_two_feature_partial_dependence_table,
)
from src.modeling.preprocessing import build_preprocessing_pipeline
from src.visualization.interpretability_plots import (
    plot_ice_curves,
    plot_partial_dependence,
    plot_two_feature_pdp,
)


def test_pdp_runs_for_regression_pipeline():
    data = _regression_data()
    pipeline = _fit_pipeline(data, ["x", "group"], "target", LinearRegression())

    table = build_partial_dependence_table(
        pipeline,
        data[["x", "group"]],
        feature="x",
        task_type="regression",
        grid_resolution=5,
    )

    assert not table.empty
    assert set(table.columns) == {"feature", "feature_value", "average_prediction", "task_type"}
    assert table["feature"].unique().tolist() == ["x"]
    assert table["task_type"].unique().tolist() == ["regression"]
    assert plot_partial_dependence(table) is not None


def test_pdp_runs_for_binary_classification_pipeline():
    data = _classification_data()
    pipeline = _fit_pipeline(data, ["x", "group"], "target", LogisticRegression(max_iter=1000))

    table = build_partial_dependence_table(
        pipeline,
        data[["x", "group"]],
        feature="x",
        task_type="binary_classification",
        grid_resolution=5,
    )

    assert not table.empty
    assert table["average_prediction"].between(0, 1).all()
    assert table["task_type"].unique().tolist() == ["binary_classification"]


def test_ice_runs_for_one_feature():
    data = _regression_data()
    pipeline = _fit_pipeline(data, ["x", "group"], "target", LinearRegression())

    table = build_ice_table(
        pipeline,
        data[["x", "group"]],
        feature="x",
        task_type="regression",
        grid_resolution=4,
        max_samples=3,
    )

    assert not table.empty
    assert table["observation_id"].nunique() == 3
    assert plot_ice_curves(table) is not None


def test_two_feature_pdp_runs():
    data = _regression_data()
    pipeline = _fit_pipeline(data, ["x", "z", "group"], "target", LinearRegression())

    table = build_two_feature_partial_dependence_table(
        pipeline,
        data[["x", "z", "group"]],
        feature_a="x",
        feature_b="z",
        task_type="regression",
        grid_resolution=4,
    )

    assert not table.empty
    assert {"feature_a_value", "feature_b_value", "average_prediction"}.issubset(table.columns)
    assert plot_two_feature_pdp(table) is not None


def test_unsupported_model_returns_clear_message():
    data = _regression_data()
    pipeline = _fit_pipeline(data, ["x", "group"], "target", LinearRegression())

    with pytest.raises(ValueError, match="requires a model with predict_proba"):
        build_partial_dependence_table(
            pipeline,
            data[["x", "group"]],
            feature="x",
            task_type="binary_classification",
            grid_resolution=5,
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


def _regression_data():
    return pd.DataFrame(
        {
            "target": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "z": [8, 7, 6, 5, 4, 3, 2, 1],
            "group": ["A", "A", "B", "B", "C", "C", "A", "B"],
        }
    )


def _classification_data():
    return pd.DataFrame(
        {
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
            "x": [1, 2, 2, 3, 7, 8, 9, 10],
            "group": ["A", "A", "B", "B", "C", "C", "A", "B"],
        }
    )
