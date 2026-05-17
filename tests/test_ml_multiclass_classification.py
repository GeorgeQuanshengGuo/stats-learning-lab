import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.pipeline import Pipeline

from src.core.model_artifacts import clear_model_artifacts, get_model_artifact
from src.modeling.machine_learning.multiclass_classification import (
    SUPPORTED_MULTICLASS_MODELS,
    run_ml_multiclass_classification_models,
)
from src.prediction.prediction_service import predict_from_model_run


def test_multiclass_ml_runs_without_crashing_and_saves_artifact():
    clear_model_artifacts()
    data = _sample_multiclass_data()

    results = run_ml_multiclass_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Random Forest Classifier"],
        random_state=7,
    )

    assert len(results) == 1
    model_run = results[0]["model_run"]
    artifact = get_model_artifact(model_run["run_id"])
    assert model_run["task_type"] == "multiclass_classification"
    assert artifact is not None
    assert artifact["fitted_pipeline"] is results[0]["pipeline"]
    assert artifact["prediction_supported"] is True


def test_each_multiclass_model_returns_a_result():
    data = _sample_multiclass_data()

    results = run_ml_multiclass_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=SUPPORTED_MULTICLASS_MODELS,
        random_state=7,
    )

    returned_model_names = [result["model_run"]["model_name"] for result in results]
    assert returned_model_names == SUPPORTED_MULTICLASS_MODELS


def test_multiclass_metrics_and_tables_exist():
    data = _sample_multiclass_data()

    result = run_ml_multiclass_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Multinomial Logistic Regression"],
        random_state=7,
    )[0]
    model_run = result["model_run"]

    assert {"train_accuracy", "train_macro_f1", "train_weighted_f1"}.issubset(model_run["train_metrics"])
    assert {"test_accuracy", "test_macro_f1", "test_weighted_f1", "test_log_loss"}.issubset(model_run["test_metrics"])
    assert result["confusion_matrix"].shape == (3, 3)
    assert set(result["class_wise_metrics"]["class"]) == {"A", "B", "C"}
    assert not result["probability_table"].empty


def test_non_multiclass_target_raises_clear_error():
    data = _sample_multiclass_data()
    data["target"] = data["target"].replace({"C": "B"})

    with pytest.raises(ValueError, match="more than two"):
        run_ml_multiclass_classification_models(
            data,
            target_column="target",
            feature_columns=["numeric_feature", "second_numeric", "category"],
            selected_models=["Random Forest Classifier"],
            random_state=7,
        )


def test_multiclass_pipeline_and_input_dataframe_are_safe():
    data = _sample_multiclass_data()
    original = data.copy(deep=True)

    result = run_ml_multiclass_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Decision Tree Classifier"],
        random_state=7,
    )[0]

    assert isinstance(result["pipeline"], Pipeline)
    assert "preprocessing" in result["pipeline"].named_steps
    assert "model" in result["pipeline"].named_steps
    assert_frame_equal(data, original)


def test_prediction_service_works_for_multiclass_ml_artifact():
    clear_model_artifacts()
    data = _sample_multiclass_data()
    result = run_ml_multiclass_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Random Forest Classifier"],
        random_state=7,
    )[0]
    model_run = result["model_run"]
    artifact = get_model_artifact(model_run["run_id"])

    prediction = predict_from_model_run(
        model_run,
        artifact,
        {"numeric_feature": 1.0, "second_numeric": 2.0, "category": "low"},
    )

    assert prediction["predicted_class"] in {"A", "B", "C"}
    assert set(prediction["class_probabilities"]) == {"A", "B", "C"}


def _sample_multiclass_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
            "numeric_feature": [
                0.5,
                0.8,
                1.0,
                1.2,
                1.4,
                1.6,
                1.8,
                2.0,
                3.0,
                3.2,
                3.4,
                3.6,
                3.8,
                4.0,
                4.2,
                4.4,
                6.0,
                6.2,
                6.4,
                6.6,
                6.8,
                7.0,
                7.2,
                7.4,
            ],
            "second_numeric": [
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                2.6,
                2.7,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                5.0,
                5.1,
                5.2,
                5.3,
                5.4,
                5.5,
                5.6,
                5.7,
            ],
            "category": ["low", "low", "mid", "mid"] * 2
            + ["mid", "mid", "high", "high"] * 2
            + ["high", "high", "mid", "mid"] * 2,
        }
    )
