import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.core.model_artifacts import clear_model_artifacts, get_model_artifact
from src.core.model_run import MODEL_RUN_FIELDS
import src.modeling.machine_learning.regression as ml_regression
from src.modeling.machine_learning.regression import (
    SUPPORTED_REGRESSION_MODELS,
    run_ml_regression_models,
)


def _sample_regression_data() -> pd.DataFrame:
    """Build a small mixed-type regression dataset."""
    return pd.DataFrame(
        {
            "target": [
                10.0,
                12.5,
                13.0,
                15.5,
                18.0,
                19.5,
                22.0,
                24.5,
                25.0,
                27.5,
                30.0,
                31.5,
            ],
            "numeric_feature": [1, 2, 3, 4, 5, 6, 7, 8, None, 10, 11, 12],
            "second_numeric": [2, 1, 4, 3, 6, 5, 8, 7, 10, None, 12, 11],
            "category": ["A", "B", "A", "B", "C", "A", "B", "C", "A", None, "B", "C"],
        }
    )


def test_run_ml_regression_models_runs_without_crashing():
    clear_model_artifacts()
    data = _sample_regression_data()

    results = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Linear Regression"],
        random_state=7,
    )

    assert len(results) == 1
    artifact = get_model_artifact(results[0]["model_run"]["run_id"])
    assert artifact is not None
    assert artifact["fitted_pipeline"] is results[0]["pipeline"]
    assert artifact["prediction_supported"] is True


def test_each_selected_model_returns_a_result():
    data = _sample_regression_data()

    results = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=SUPPORTED_REGRESSION_MODELS,
        random_state=7,
    )

    returned_model_names = [result["model_run"]["model_name"] for result in results]

    assert returned_model_names == SUPPORTED_REGRESSION_MODELS


def test_train_and_test_metrics_exist():
    data = _sample_regression_data()

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Ridge"],
        random_state=7,
    )[0]
    model_run = result["model_run"]

    assert {"train_rmse", "train_mae", "train_r2"}.issubset(model_run["train_metrics"])
    assert {"test_rmse", "test_mae", "test_r2"}.issubset(model_run["test_metrics"])


def test_linear_ml_regression_saves_coefficients_and_formula():
    clear_model_artifacts()
    data = _sample_regression_data()

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Linear Regression"],
        random_state=7,
    )[0]
    model_run = result["model_run"]
    artifact = get_model_artifact(model_run["run_id"])

    coefficient_terms = [row["term"] for row in model_run["coefficient_table"]]

    assert "intercept" in coefficient_terms
    assert any(term.startswith("category_") for term in coefficient_terms)
    assert model_run["formula_latex"]["estimated"]
    assert "transformed feature scale" in model_run["formula_latex"]["note"]
    assert artifact["formula_latex"] == model_run["formula_latex"]


def test_regression_cross_validation_metrics_are_saved():
    data = _sample_regression_data()

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Ridge"],
        random_state=7,
        cv_folds=5,
    )[0]
    test_metrics = result["model_run"]["test_metrics"]

    assert {"cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_mae_std", "cv_r2_mean", "cv_r2_std"}.issubset(
        test_metrics
    )
    assert test_metrics["cv_rmse_mean"] is not None
    assert result["model_run"]["split_config"]["cv_folds"] == 5


def test_regression_cross_validation_uses_full_pipeline(monkeypatch):
    data = _sample_regression_data()
    captured = {}

    def fake_cross_validate(estimator, x_values, y_values, cv, scoring, error_score):
        captured["estimator"] = estimator
        captured["x_columns"] = list(x_values.columns)
        captured["cv"] = cv
        return {
            "test_rmse": np.array([-1.0, -2.0]),
            "test_mae": np.array([-0.5, -0.7]),
            "test_r2": np.array([0.8, 0.6]),
        }

    monkeypatch.setattr(ml_regression, "cross_validate", fake_cross_validate)

    result = ml_regression.run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Linear Regression"],
        random_state=7,
        cv_folds=5,
    )[0]

    assert isinstance(captured["estimator"], Pipeline)
    assert "preprocessing" in captured["estimator"].named_steps
    assert "model" in captured["estimator"].named_steps
    assert captured["x_columns"] == ["numeric_feature", "second_numeric", "category"]
    assert result["model_run"]["test_metrics"]["cv_rmse_mean"] == 1.5


def test_regression_cross_validation_handles_too_few_rows_gracefully():
    data = pd.DataFrame({"target": [1.0, 2.0, 3.0], "feature": [1.0, 2.0, 3.0]})

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["feature"],
        selected_models=["Linear Regression"],
        random_state=7,
        cv_folds=5,
    )[0]

    assert result["model_run"]["test_metrics"]["cv_rmse_mean"] is None
    assert result["model_run"]["test_metrics"]["cv_r2_mean"] is None


def test_model_run_has_unified_top_level_fields():
    data = _sample_regression_data()

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Decision Tree Regressor"],
        random_state=7,
    )[0]
    model_run = result["model_run"]

    assert list(model_run.keys()) == MODEL_RUN_FIELDS
    assert model_run["task_type"] == "regression"
    assert model_run["model_family"] == "machine_learning"
    assert model_run["target"] == "target"
    assert model_run["features"] == ["numeric_feature", "second_numeric", "category"]
    assert model_run["statistical_summary"] == {}


def test_preprocessing_is_inside_sklearn_pipeline():
    data = _sample_regression_data()

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Random Forest Regressor"],
        random_state=7,
        scale_numeric=True,
    )[0]
    pipeline = result["pipeline"]

    assert isinstance(pipeline, Pipeline)
    assert "preprocessing" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_original_input_df_is_not_modified():
    data = _sample_regression_data()
    original = data.copy(deep=True)

    run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "second_numeric", "category"],
        selected_models=["Gradient Boosting Regressor", "KNN Regressor"],
        random_state=7,
    )

    pd.testing.assert_frame_equal(data, original)
