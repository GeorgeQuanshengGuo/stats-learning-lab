import numpy as np
import pandas as pd

from src.core.model_artifacts import clear_model_artifacts
from src.modeling.machine_learning.classification import run_ml_binary_classification_models
from src.modeling.machine_learning.regression import run_ml_regression_models
from src.modeling.machine_learning.tuning import run_tuned_ml_regression_models


def _regression_data() -> pd.DataFrame:
    """Build deterministic regression data with numeric and categorical features."""
    rows = []
    for index in range(36):
        group = ["A", "B", "C"][index % 3]
        group_effect = {"A": 0.0, "B": 2.0, "C": -1.0}[group]
        x1 = float(index)
        x2 = float((index % 7) - 3)
        rows.append(
            {
                "target": 5.0 + 1.8 * x1 - 0.7 * x2 + group_effect,
                "x1": x1,
                "x2": x2,
                "group": group,
            }
        )
    return pd.DataFrame(rows)


def _classification_data() -> pd.DataFrame:
    """Build deterministic binary data with enough rows for stratified splits."""
    rows = []
    for index in range(40):
        group = "A" if index % 4 in {0, 1} else "B"
        score = (index % 10) + (2 if group == "A" else -1)
        rows.append(
            {
                "target": "yes" if score >= 5 else "no",
                "x1": float(score),
                "x2": float(index % 5),
                "group": group,
            }
        )
    return pd.DataFrame(rows)


def test_ml_regression_results_are_reproducible_with_same_random_state():
    clear_model_artifacts()
    data = _regression_data()

    first = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Random Forest Regressor"],
        random_state=23,
        test_size=0.25,
        scale_numeric=True,
    )[0]
    second = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Random Forest Regressor"],
        random_state=23,
        test_size=0.25,
        scale_numeric=True,
    )[0]

    np.testing.assert_allclose(first["test_predictions"], second["test_predictions"])
    assert first["model_run"]["test_metrics"]["test_rmse"] == second["model_run"]["test_metrics"]["test_rmse"]
    assert first["model_run"]["test_metrics"]["test_r2"] == second["model_run"]["test_metrics"]["test_r2"]


def test_ml_binary_classification_results_are_reproducible_with_same_random_state():
    clear_model_artifacts()
    data = _classification_data()

    first = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Random Forest Classifier"],
        positive_class="yes",
        random_state=17,
        test_size=0.25,
        scale_numeric=True,
    )[0]
    second = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Random Forest Classifier"],
        positive_class="yes",
        random_state=17,
        test_size=0.25,
        scale_numeric=True,
    )[0]

    np.testing.assert_array_equal(first["test_predictions"], second["test_predictions"])
    np.testing.assert_allclose(first["test_probabilities"], second["test_probabilities"])
    assert first["model_run"]["test_metrics"]["test_f1"] == second["model_run"]["test_metrics"]["test_f1"]


def test_randomized_tuning_is_reproducible_with_same_random_state():
    clear_model_artifacts()
    data = _regression_data()

    first = run_tuned_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Ridge"],
        search_type="randomized",
        cv_folds=3,
        scoring="RMSE",
        n_iter=2,
        random_state=31,
        scale_numeric=True,
    )[0]
    second = run_tuned_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "group"],
        selected_models=["Ridge"],
        search_type="randomized",
        cv_folds=3,
        scoring="RMSE",
        n_iter=2,
        random_state=31,
        scale_numeric=True,
    )[0]

    assert first["search"].best_params_ == second["search"].best_params_
    assert first["search"].best_score_ == second["search"].best_score_
    np.testing.assert_allclose(first["test_predictions"], second["test_predictions"])
