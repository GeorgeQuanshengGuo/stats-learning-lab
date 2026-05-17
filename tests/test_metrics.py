import numpy as np

from src.modeling.metrics import (
    compute_binary_classification_metrics,
    compute_regression_metrics,
)


def test_compute_regression_metrics_normal_case():
    metrics = compute_regression_metrics([1, 2, 3], [1, 2, 5])

    assert metrics["rmse"] == np.sqrt(4 / 3)
    assert metrics["mae"] == 2 / 3
    assert metrics["r2"] == -1


def test_compute_regression_metrics_perfect_prediction():
    metrics = compute_regression_metrics([1, 2, 3], [1, 2, 3])

    assert metrics == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}


def test_compute_binary_classification_metrics_normal_case():
    metrics = compute_binary_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 0],
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert "roc_auc" not in metrics
    assert "pr_auc" not in metrics


def test_compute_binary_classification_metrics_with_probabilities():
    metrics = compute_binary_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        y_proba=[0.1, 0.2, 0.8, 0.9],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_compute_binary_classification_metrics_uses_positive_label():
    metrics = compute_binary_classification_metrics(
        y_true=["no", "yes", "yes", "no"],
        y_pred=["no", "yes", "no", "no"],
        y_proba=[0.1, 0.9, 0.4, 0.2],
        positive_label="yes",
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert np.isclose(metrics["f1"], 2 / 3)
    assert metrics["roc_auc"] == 1.0


def test_compute_binary_classification_metrics_roc_auc_cannot_be_computed():
    metrics = compute_binary_classification_metrics(
        y_true=[1, 1, 1],
        y_pred=[1, 1, 1],
        y_proba=[0.8, 0.9, 0.7],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None


def test_compute_binary_classification_metrics_zero_division_cases():
    metrics = compute_binary_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 0, 0],
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
