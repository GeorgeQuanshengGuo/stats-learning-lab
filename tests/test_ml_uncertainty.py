import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.modeling.preprocessing import build_preprocessing_pipeline
from src.prediction.ml_uncertainty import bootstrap_prediction_interval


def test_bootstrap_interval_returns_bounds():
    data = _regression_data()
    factory = _pipeline_factory(data, ["x", "group"])

    result = bootstrap_prediction_interval(
        base_pipeline_factory=factory,
        df=data,
        target_column="target",
        feature_columns=["x", "group"],
        input_row={"x": 4, "group": "B"},
        n_bootstrap=10,
        random_state=1,
    )

    assert result["interval_type"] == "bootstrap_empirical_interval"
    assert result["lower_bound"] <= result["upper_bound"]
    assert result["n_bootstrap"] == 10
    assert result["alpha"] == 0.05
    assert isinstance(result["point_prediction"], float)
    assert isinstance(result["bootstrap_mean"], float)


def test_bootstrap_interval_width_is_nonnegative():
    data = _regression_data()
    factory = _pipeline_factory(data, ["x", "group"])

    result = bootstrap_prediction_interval(
        base_pipeline_factory=factory,
        df=data,
        target_column="target",
        feature_columns=["x", "group"],
        input_row={"x": 5, "group": "C"},
        n_bootstrap=8,
        random_state=2,
    )

    assert result["upper_bound"] - result["lower_bound"] >= 0


def test_bootstrap_does_not_modify_input_dataframe():
    data = _regression_data()
    original = data.copy(deep=True)
    factory = _pipeline_factory(data, ["x", "group"])

    bootstrap_prediction_interval(
        base_pipeline_factory=factory,
        df=data,
        target_column="target",
        feature_columns=["x", "group"],
        input_row={"x": 3, "group": "A"},
        n_bootstrap=6,
        random_state=3,
    )

    assert_frame_equal(data, original)


def test_bootstrap_keeps_preprocessing_inside_pipeline():
    data = _regression_data()
    created_pipelines = []

    def factory(training_df):
        preprocessing = build_preprocessing_pipeline(training_df, ["x", "group"], scale_numeric=True)
        pipeline = Pipeline(
            [
                ("preprocessing", preprocessing),
                ("model", LinearRegression()),
            ]
        )
        created_pipelines.append(pipeline)
        return pipeline

    bootstrap_prediction_interval(
        base_pipeline_factory=factory,
        df=data,
        target_column="target",
        feature_columns=["x", "group"],
        input_row={"x": 3, "group": "A"},
        n_bootstrap=5,
        random_state=4,
    )

    assert created_pipelines
    assert all(isinstance(pipeline, Pipeline) for pipeline in created_pipelines)
    assert all("preprocessing" in pipeline.named_steps for pipeline in created_pipelines)
    assert all("model" in pipeline.named_steps for pipeline in created_pipelines)


def test_bootstrap_uses_attached_fitted_pipeline_for_point_prediction():
    data = _regression_data()
    factory = _pipeline_factory(data, ["x", "group"])
    fitted_pipeline = factory(data)
    fitted_pipeline.fit(data[["x", "group"]], data["target"])
    factory.fitted_pipeline = fitted_pipeline

    result = bootstrap_prediction_interval(
        base_pipeline_factory=factory,
        df=data,
        target_column="target",
        feature_columns=["x", "group"],
        input_row={"x": 4, "group": "B"},
        n_bootstrap=5,
        random_state=5,
    )

    expected = float(fitted_pipeline.predict(pd.DataFrame([{"x": 4, "group": "B"}]))[0])
    assert result["point_prediction"] == pytest.approx(expected)


def test_small_dataset_error_is_clear():
    data = pd.DataFrame({"target": [1.0, 2.0], "x": [1, 2]})
    factory = _pipeline_factory(data, ["x"])

    with pytest.raises(ValueError, match="At least 3 rows"):
        bootstrap_prediction_interval(
            base_pipeline_factory=factory,
            df=data,
            target_column="target",
            feature_columns=["x"],
            input_row={"x": 2},
            n_bootstrap=5,
        )


def _regression_data():
    return pd.DataFrame(
        {
            "target": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "group": ["A", "A", "B", "B", "C", "C", "A", "B"],
        }
    )


def _pipeline_factory(data, features):
    def factory(training_df=None):
        training_data = data if training_df is None else training_df
        preprocessing = build_preprocessing_pipeline(training_data, features, scale_numeric=False)
        return Pipeline(
            [
                ("preprocessing", preprocessing),
                ("model", LinearRegression()),
            ]
        )

    return factory
