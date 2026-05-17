import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.core.model_artifacts import clear_model_artifacts, get_model_artifact
from src.core.model_run import MODEL_RUN_FIELDS
import src.modeling.machine_learning.classification as ml_classification
from src.modeling.machine_learning.classification import (
    SUPPORTED_CLASSIFICATION_MODELS,
    run_ml_binary_classification_models,
)


def _sample_classification_data() -> pd.DataFrame:
    """Build a small mixed-type binary classification dataset."""
    return pd.DataFrame(
        {
            "target": [
                "yes",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "yes",
                "no",
            ],
            "numeric_feature": [1, 2, 3, 4, 5, 6, None, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "second_numeric": [5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10, -1, 11, -2, 12, -3],
            "category": ["A", "B", "A", "B", "C", "A", "B", "C", None, "A", "B", "C", "A", "B", "C", "A"],
        }
    )


def test_run_ml_binary_classification_models_runs_without_crashing():
    clear_model_artifacts()
    data = _sample_classification_data()

    results = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Logistic Regression"],
        positive_class="yes",
        random_state=7,
    )

    assert len(results) == 1
    artifact = get_model_artifact(results[0]["model_run"]["run_id"])
    assert artifact is not None
    assert artifact["fitted_pipeline"] is results[0]["pipeline"]
    assert artifact["positive_class"] == "yes"
    assert artifact["prediction_supported"] is True


def test_each_selected_model_returns_a_result():
    data = _sample_classification_data()

    results = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=SUPPORTED_CLASSIFICATION_MODELS,
        positive_class="yes",
        random_state=7,
    )

    returned_model_names = [result["model_run"]["model_name"] for result in results]

    assert returned_model_names == SUPPORTED_CLASSIFICATION_MODELS


def test_train_and_test_metrics_exist():
    data = _sample_classification_data()

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Random Forest Classifier"],
        positive_class="yes",
        random_state=7,
    )[0]
    model_run = result["model_run"]

    assert {"train_accuracy", "train_precision", "train_recall", "train_f1"}.issubset(model_run["train_metrics"])
    assert {"train_roc_auc", "train_pr_auc"}.issubset(model_run["train_metrics"])
    assert {"test_accuracy", "test_precision", "test_recall", "test_f1"}.issubset(model_run["test_metrics"])
    assert {"test_roc_auc", "test_pr_auc"}.issubset(model_run["test_metrics"])


def test_logistic_ml_classification_saves_coefficients_and_formula():
    clear_model_artifacts()
    data = _sample_classification_data()

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Logistic Regression"],
        positive_class="yes",
        random_state=7,
    )[0]
    model_run = result["model_run"]
    artifact = get_model_artifact(model_run["run_id"])

    coefficient_terms = [row["term"] for row in model_run["coefficient_table"]]

    assert "intercept" in coefficient_terms
    assert model_run["formula_latex"]["estimated"]
    assert "Positive class is `yes`" in model_run["formula_latex"]["note"]
    assert artifact["formula_latex"] == model_run["formula_latex"]


def test_binary_classification_cross_validation_metrics_are_saved():
    data = _sample_classification_data()

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Logistic Regression"],
        positive_class="yes",
        random_state=7,
        cv_folds=5,
    )[0]
    test_metrics = result["model_run"]["test_metrics"]

    assert {
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_precision_mean",
        "cv_precision_std",
        "cv_recall_mean",
        "cv_recall_std",
        "cv_f1_mean",
        "cv_f1_std",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
    }.issubset(test_metrics)
    assert test_metrics["cv_accuracy_mean"] is not None
    assert result["model_run"]["split_config"]["cv_folds"] == 5


def test_binary_classification_cross_validation_uses_full_pipeline(monkeypatch):
    data = _sample_classification_data()
    captured = {}

    def fake_cross_validate(estimator, x_values, y_values, cv, scoring, error_score):
        captured["estimator"] = estimator
        captured["x_columns"] = list(x_values.columns)
        captured["y_values"] = set(y_values.unique())
        captured["cv"] = cv
        return {
            "test_accuracy": np.array([0.75, 0.8]),
            "test_precision": np.array([0.7, 0.85]),
            "test_recall": np.array([0.6, 0.9]),
            "test_f1": np.array([0.65, 0.87]),
            "test_roc_auc": np.array([0.8, 0.9]),
        }

    monkeypatch.setattr(ml_classification, "cross_validate", fake_cross_validate)

    result = ml_classification.run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Logistic Regression"],
        positive_class="yes",
        random_state=7,
        cv_folds=5,
    )[0]

    assert isinstance(captured["estimator"], Pipeline)
    assert "preprocessing" in captured["estimator"].named_steps
    assert "model" in captured["estimator"].named_steps
    assert captured["x_columns"] == ["numeric_feature", "second_numeric", "category"]
    assert captured["y_values"] == {0, 1}
    assert result["model_run"]["test_metrics"]["cv_accuracy_mean"] == 0.775


def test_binary_classification_cross_validation_handles_too_few_rows_gracefully():
    data = pd.DataFrame(
        {
            "target": ["yes", "no", "yes", "no"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["feature"],
        selected_models=["Decision Tree Classifier"],
        positive_class="yes",
        test_size=0.5,
        random_state=7,
        cv_folds=5,
    )[0]

    assert result["model_run"]["test_metrics"]["cv_accuracy_mean"] is None
    assert result["model_run"]["test_metrics"]["cv_roc_auc_mean"] is None


def test_target_is_encoded_correctly_based_on_positive_class():
    data = _sample_classification_data()

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Decision Tree Classifier"],
        positive_class="no",
        random_state=7,
    )[0]

    assert result["positive_class"] == "no"
    assert set(result["train_actual"].unique()) == {0, 1}
    assert result["model_run"]["preprocessing"]["positive_class"] == "no"
    assert "no encoded as 1" in result["model_run"]["preprocessing"]["target_encoding"]


def test_non_binary_target_raises_clear_error():
    data = _sample_classification_data()
    data.loc[0, "target"] = "maybe"

    with pytest.raises(ValueError, match="exactly two non-missing target classes"):
        run_ml_binary_classification_models(
            data,
            target_column="target",
            feature_columns=["numeric_feature", "second_numeric", "category"],
            selected_models=["Logistic Regression"],
            positive_class="yes",
            random_state=7,
        )


def test_positive_class_must_be_one_of_two_classes():
    data = _sample_classification_data()

    with pytest.raises(ValueError, match="positive_class"):
        run_ml_binary_classification_models(
            data,
            target_column="target",
            feature_columns=["numeric_feature", "second_numeric", "category"],
            selected_models=["Logistic Regression"],
            positive_class="maybe",
            random_state=7,
        )


def test_model_run_has_unified_top_level_fields():
    data = _sample_classification_data()

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Gradient Boosting Classifier"],
        positive_class="yes",
        random_state=7,
    )[0]
    model_run = result["model_run"]

    assert list(model_run.keys()) == MODEL_RUN_FIELDS
    assert model_run["task_type"] == "binary_classification"
    assert model_run["model_family"] == "machine_learning"
    assert model_run["target"] == "target"
    assert model_run["features"] == ["numeric_feature", "second_numeric", "category"]
    assert model_run["statistical_summary"] == {}


def test_preprocessing_is_inside_sklearn_pipeline():
    data = _sample_classification_data()

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["KNN Classifier"],
        positive_class="yes",
        random_state=7,
    )[0]
    pipeline = result["pipeline"]

    assert isinstance(pipeline, Pipeline)
    assert "preprocessing" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_original_input_df_is_not_modified():
    data = _sample_classification_data()
    original = data.copy(deep=True)

    run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Logistic Regression", "Random Forest Classifier"],
        positive_class="yes",
        random_state=7,
    )

    pd.testing.assert_frame_equal(data, original)
