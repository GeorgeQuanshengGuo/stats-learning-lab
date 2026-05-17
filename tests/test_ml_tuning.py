import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.core.model_artifacts import clear_model_artifacts, get_model_artifact
from src.core.model_comparison import (
    build_binary_classification_comparison_table,
    build_regression_comparison_table,
)
from src.prediction.prediction_service import predict_from_model_run
from src.modeling.machine_learning.tuning import (
    run_tuned_ml_binary_classification_models,
    run_tuned_ml_regression_models,
)


def _regression_data() -> pd.DataFrame:
    """Build a small mixed-type regression dataset."""
    return pd.DataFrame(
        {
            "target": [10, 12, 13, 15, 18, 20, 22, 24, 25, 28, 30, 33, 35, 38, 40],
            "x1": [1, 2, 3, 4, 5, 6, 7, None, 9, 10, 11, 12, 13, 14, 15],
            "x2": [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16],
            "group": ["A", "A", "B", "B", "C", "A", "B", "C", None, "A", "B", "C", "A", "B", "C"],
        }
    )


def _classification_data() -> pd.DataFrame:
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
                "yes",
                "no",
            ],
            "x1": [9, 1, 8, 2, 9, 2, 7, 3, 8, 1, 9, 2, 7, 3, 8, 1, 9, 2],
            "x2": [1, 9, 2, 8, 1, 8, 3, 7, 2, 9, 1, 8, 3, 7, 2, 9, 1, 8],
            "group": ["A", "B", "A", "B", "A", "C", "A", "C", None, "B", "A", "B", "A", "C", "A", "B", "A", "C"],
        }
    )


def test_tuned_regression_returns_model_run_pipeline_and_artifact():
    clear_model_artifacts()
    data = _regression_data()

    result = run_tuned_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Ridge"],
        search_type="grid",
        cv_folds=3,
        scoring="RMSE",
        random_state=7,
    )[0]
    model_run = result["model_run"]
    artifact = get_model_artifact(model_run["run_id"])

    assert model_run["model_name"] == "Ridge (Tuned)"
    assert model_run["preprocessing"]["tuning"]["enabled"] is True
    assert model_run["preprocessing"]["tuning"]["best_params"]
    assert {"train_rmse", "train_mae", "train_r2"}.issubset(model_run["train_metrics"])
    assert {"test_rmse", "test_mae", "test_r2"}.issubset(model_run["test_metrics"])
    assert isinstance(result["pipeline"], Pipeline)
    assert "preprocessing" in result["pipeline"].named_steps
    assert "model" in result["pipeline"].named_steps
    assert isinstance(result["search"].estimator, Pipeline)
    assert "preprocessing" in result["search"].estimator.named_steps
    assert artifact["fitted_pipeline"] is result["pipeline"]
    assert artifact["best_estimator"] is result["pipeline"]
    assert not result["tuning_results_table"].empty


def test_tuned_binary_classification_returns_metrics_and_positive_class_artifact():
    clear_model_artifacts()
    data = _classification_data()

    result = run_tuned_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Logistic Regression"],
        positive_class="yes",
        search_type="randomized",
        cv_folds=3,
        scoring="F1",
        n_iter=2,
        random_state=7,
    )[0]
    model_run = result["model_run"]
    artifact = get_model_artifact(model_run["run_id"])

    assert model_run["model_name"] == "Logistic Regression (Tuned)"
    assert model_run["preprocessing"]["positive_class"] == "yes"
    assert model_run["preprocessing"]["tuning"]["search_type"] == "randomized"
    assert {"train_accuracy", "train_precision", "train_recall", "train_f1"}.issubset(model_run["train_metrics"])
    assert {"test_accuracy", "test_precision", "test_recall", "test_f1"}.issubset(model_run["test_metrics"])
    assert isinstance(result["pipeline"], Pipeline)
    assert "preprocessing" in result["pipeline"].named_steps
    assert artifact["fitted_pipeline"] is result["pipeline"]
    assert artifact["positive_class"] == "yes"
    assert artifact["target_encoder"]["positive_class_encoded_as"] == 1


def test_tuned_runs_appear_in_model_comparison_tables():
    regression_run = run_tuned_ml_regression_models(
        _regression_data(),
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Lasso"],
        search_type="grid",
        cv_folds=3,
        scoring="MAE",
        random_state=7,
    )[0]["model_run"]
    classification_run = run_tuned_ml_binary_classification_models(
        _classification_data(),
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Decision Tree Classifier"],
        positive_class="yes",
        search_type="grid",
        cv_folds=3,
        scoring="Accuracy",
        random_state=7,
    )[0]["model_run"]

    regression_table = build_regression_comparison_table([regression_run, classification_run])
    classification_table = build_binary_classification_comparison_table([regression_run, classification_run])

    assert len(regression_table) == 1
    assert regression_table.loc[0, "model_name"] == "Lasso (Tuned)"
    assert regression_table.loc[0, "tuning_method"] == "grid"
    assert regression_table.loc[0, "best_cv_score"] is not None
    assert len(classification_table) == 1
    assert classification_table.loc[0, "model_name"] == "Decision Tree Classifier (Tuned)"
    assert classification_table.loc[0, "tuning_scoring"] == "Accuracy"


def test_prediction_service_works_with_tuned_regression_artifact():
    clear_model_artifacts()
    result = run_tuned_ml_regression_models(
        _regression_data(),
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Ridge"],
        search_type="grid",
        cv_folds=3,
        random_state=7,
    )[0]
    model_run = result["model_run"]
    artifact = get_model_artifact(model_run["run_id"])

    prediction = predict_from_model_run(
        model_run,
        artifact,
        {"x1": 5, "x2": 6, "group": "A"},
    )

    assert prediction["task_type"] == "regression"
    assert isinstance(prediction["predicted_value"], float)


def test_prediction_service_works_with_tuned_binary_classification_artifact():
    clear_model_artifacts()
    result = run_tuned_ml_binary_classification_models(
        _classification_data(),
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Logistic Regression"],
        positive_class="yes",
        search_type="grid",
        cv_folds=3,
        random_state=7,
    )[0]
    model_run = result["model_run"]
    artifact = get_model_artifact(model_run["run_id"])

    prediction = predict_from_model_run(
        model_run,
        artifact,
        {"x1": 9, "x2": 1, "group": "A"},
        threshold=0.5,
    )

    assert prediction["task_type"] == "binary_classification"
    assert prediction["positive_class"] == "yes"
    assert prediction["predicted_class"] in {"yes", "no"}
    assert prediction["positive_class_probability"] is not None


def test_tuning_does_not_modify_original_input_df():
    data = _regression_data()
    original = data.copy(deep=True)

    run_tuned_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Decision Tree Regressor"],
        search_type="randomized",
        cv_folds=3,
        n_iter=2,
        random_state=7,
    )

    pd.testing.assert_frame_equal(data, original)


def test_unsupported_tuning_model_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported regression tuning model"):
        run_tuned_ml_regression_models(
            _regression_data(),
            target_column="target",
            feature_columns=["x1", "x2", "group"],
            selected_models=["Linear Regression"],
            cv_folds=3,
        )
