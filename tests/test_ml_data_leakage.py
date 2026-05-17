import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.core.model_artifacts import clear_model_artifacts
from src.modeling.machine_learning.classification import run_ml_binary_classification_models
from src.modeling.machine_learning.regression import run_ml_regression_models
from src.modeling.machine_learning.tuning import (
    run_tuned_ml_binary_classification_models,
    run_tuned_ml_regression_models,
)
from src.prediction.prediction_service import predict_from_model_run


def _leakage_regression_data() -> pd.DataFrame:
    """Create data where fitting preprocessing on all rows would be visible."""
    return pd.DataFrame(
        {
            "target": [0.0, 1.2, 2.1, 3.4, 4.0, 5.1, 6.3, 7.2, 8.4, 9.5],
            "num": [0.0, 1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, -1000.0],
            "cat": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "RARE_TEST"],
        }
    )


def _binary_data() -> pd.DataFrame:
    """Create a balanced binary dataset with mixed feature types."""
    return pd.DataFrame(
        {
            "target": ["yes", "no"] * 12,
            "num": [9, 1, 8, 2, 9, 2, 7, 3, 8, 1, 9, 2, 7, 3, 8, 1, 9, 2, 7, 3, 8, 1, 9, 2],
            "cat": ["A", "B", "A", "B", "A", "C", "A", "C"] * 3,
        }
    )


def test_regression_preprocessing_is_fit_on_training_data_only():
    clear_model_artifacts()
    data = _leakage_regression_data()
    original = data.copy(deep=True)
    features = ["num", "cat"]

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=features,
        selected_models=["Linear Regression"],
        test_size=0.2,
        random_state=1,
        scale_numeric=True,
    )[0]

    train_df, test_df = train_test_split(data[["target", *features]], test_size=0.2, random_state=1)
    preprocessing = result["pipeline"].named_steps["preprocessing"]
    numeric_pipeline = preprocessing.named_transformers_["numeric"]
    categorical_pipeline = preprocessing.named_transformers_["categorical"]

    train_median = train_df["num"].median()
    full_median = data["num"].median()
    assert numeric_pipeline.named_steps["imputer"].statistics_[0] == train_median
    assert numeric_pipeline.named_steps["imputer"].statistics_[0] != full_median

    expected_scaled_mean = train_df["num"].fillna(train_median).mean()
    full_scaled_mean = data["num"].fillna(full_median).mean()
    assert np.isclose(numeric_pipeline.named_steps["scaler"].mean_[0], expected_scaled_mean)
    assert not np.isclose(numeric_pipeline.named_steps["scaler"].mean_[0], full_scaled_mean)

    encoded_categories = set(categorical_pipeline.named_steps["encoder"].categories_[0])
    assert "RARE_TEST" not in encoded_categories
    preprocessing.transform(test_df[features])
    pd.testing.assert_frame_equal(data, original)


def test_ml_binary_classification_uses_stratified_split_and_pipeline():
    clear_model_artifacts()
    data = _binary_data()
    original = data.copy(deep=True)

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["num", "cat"],
        selected_models=["Logistic Regression"],
        positive_class="yes",
        test_size=0.25,
        random_state=11,
    )[0]

    pipeline = result["pipeline"]
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["preprocessing"], ColumnTransformer)
    assert result["model_run"]["split_config"]["stratified"] is True
    assert set(result["train_actual"].unique()) == {0, 1}
    assert set(result["test_actual"].unique()) == {0, 1}
    pd.testing.assert_frame_equal(data, original)


def test_hyperparameter_tuning_wraps_search_around_full_pipeline():
    clear_model_artifacts()
    regression_result = run_tuned_ml_regression_models(
        _leakage_regression_data(),
        target_column="target",
        feature_columns=["num", "cat"],
        selected_models=["Ridge"],
        search_type="grid",
        cv_folds=3,
        random_state=3,
        scale_numeric=True,
    )[0]
    classification_result = run_tuned_ml_binary_classification_models(
        _binary_data(),
        target_column="target",
        feature_columns=["num", "cat"],
        selected_models=["Logistic Regression"],
        positive_class="yes",
        search_type="grid",
        cv_folds=3,
        random_state=3,
        scale_numeric=True,
    )[0]

    for result in [regression_result, classification_result]:
        search_estimator = result["search"].estimator
        best_pipeline = result["pipeline"]
        assert isinstance(search_estimator, Pipeline)
        assert isinstance(search_estimator.named_steps["preprocessing"], ColumnTransformer)
        assert isinstance(best_pipeline, Pipeline)
        assert isinstance(best_pipeline.named_steps["preprocessing"], ColumnTransformer)
        assert result["model_run"]["split_config"]["train_rows"] < result["model_run"]["split_config"]["rows_used"]


def test_prediction_service_uses_saved_pipeline_without_refitting():
    class NoRefitPipeline:
        def __init__(self):
            self.predict_calls = 0
            self.fit_calls = 0

        def fit(self, *_args, **_kwargs):
            self.fit_calls += 1
            raise AssertionError("Prediction should not refit the saved model artifact.")

        def predict(self, input_frame):
            self.predict_calls += 1
            assert list(input_frame.columns) == ["x"]
            return np.array([42.0])

    pipeline = NoRefitPipeline()
    model_run = {
        "run_id": "run-no-refit",
        "task_type": "regression",
        "model_family": "machine_learning",
        "model_name": "No Refit Regression",
        "target": "target",
        "features": ["x"],
        "preprocessing": {},
    }
    artifact = {
        "run_id": "run-no-refit",
        "model_family": "machine_learning",
        "task_type": "regression",
        "model_name": "No Refit Regression",
        "fitted_pipeline": pipeline,
        "fitted_model": object(),
        "features": ["x"],
        "prediction_supported": True,
    }

    prediction = predict_from_model_run(model_run, artifact, {"x": 5})

    assert prediction["predicted_value"] == 42.0
    assert pipeline.predict_calls == 1
    assert pipeline.fit_calls == 0
