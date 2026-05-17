import numpy as np
import pandas as pd
import pytest

from src.modeling.statistical.logistic_regression import (
    classification_metrics,
    confusion_matrix_table,
    find_best_f1_threshold,
    roc_auc_score,
    run_logistic_regression,
)


def test_classification_metrics_at_threshold():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.4], threshold=0.5)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["F1"] == 0.5


def test_confusion_matrix_table_counts_predictions():
    matrix = confusion_matrix_table([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.4], threshold=0.5)

    assert matrix.loc["actual_0", "predicted_0"] == 1
    assert matrix.loc["actual_0", "predicted_1"] == 1
    assert matrix.loc["actual_1", "predicted_0"] == 1
    assert matrix.loc["actual_1", "predicted_1"] == 1


def test_threshold_changes_metrics_and_confusion_matrix():
    y_true = [0, 0, 1, 1]
    probabilities = [0.2, 0.6, 0.7, 0.9]

    low_threshold_metrics = classification_metrics(y_true, probabilities, threshold=0.5)
    high_threshold_metrics = classification_metrics(y_true, probabilities, threshold=0.8)
    low_threshold_matrix = confusion_matrix_table(y_true, probabilities, threshold=0.5)
    high_threshold_matrix = confusion_matrix_table(y_true, probabilities, threshold=0.8)

    assert low_threshold_metrics["recall"] == 1.0
    assert high_threshold_metrics["recall"] == 0.5
    assert low_threshold_matrix.loc["actual_1", "predicted_1"] == 2
    assert high_threshold_matrix.loc["actual_1", "predicted_1"] == 1


def test_roc_auc_score_for_perfect_rankings():
    auc = roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert auc == 1.0


def test_find_best_f1_threshold_returns_valid_threshold():
    threshold = find_best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert np.isclose(threshold, 0.8)


def test_logistic_regression_rejects_non_binary_target():
    data = pd.DataFrame(
        {
            "target": ["a", "b", "c", "a", "b", "c"],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )

    with pytest.raises(ValueError, match="exactly two target classes"):
        run_logistic_regression(data, "target", "a", ["x"], 0.3, 42)


def test_logistic_regression_rejects_positive_class_not_in_target():
    data = pd.DataFrame(
        {
            "target": ["no", "yes", "no", "yes", "no", "yes"],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )

    with pytest.raises(ValueError, match="selected positive class"):
        run_logistic_regression(data, "target", "maybe", ["x"], 0.3, 42)
