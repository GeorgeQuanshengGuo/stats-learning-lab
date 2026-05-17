"""Reusable metric helpers for machine learning models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float | None]:
    """Return basic regression metrics for predictions."""
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)

    if len(true_values) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    errors = true_values - predicted_values
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(mean_absolute_error(true_values, predicted_values)),
        "r2": _safe_r2_score(true_values, predicted_values),
    }


def compute_binary_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None,
    positive_label: Any = 1,
) -> dict[str, float | None]:
    """Return binary classification metrics for predicted labels and probabilities."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    metrics = {
        "accuracy": _safe_accuracy_score(true_values, predicted_values),
        "precision": _safe_precision_score(true_values, predicted_values, positive_label),
        "recall": _safe_recall_score(true_values, predicted_values, positive_label),
        "f1": _safe_f1_score(true_values, predicted_values, positive_label),
    }

    if y_proba is not None:
        positive_scores = _positive_class_scores(y_proba)
        binary_true = (true_values == positive_label).astype(int)
        metrics["roc_auc"] = _safe_roc_auc_score(binary_true, positive_scores)
        metrics["pr_auc"] = _safe_pr_auc_score(binary_true, positive_scores)

    return metrics


def compute_multiclass_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None,
    labels: list[Any] | None = None,
) -> dict[str, float | None]:
    """Return multiclass classification metrics for labels and probabilities."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    if len(true_values) == 0:
        return {
            "accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "weighted_precision": np.nan,
            "weighted_recall": np.nan,
            "weighted_f1": np.nan,
        }

    metrics = {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "macro_precision": float(precision_score(true_values, predicted_values, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true_values, predicted_values, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true_values, predicted_values, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(true_values, predicted_values, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(true_values, predicted_values, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(true_values, predicted_values, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        metrics["log_loss"] = _safe_log_loss(true_values, y_proba, labels=labels)

    return metrics


def _safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return R-squared, or NaN when it is undefined."""
    if len(y_true) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def _safe_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return accuracy, or NaN for empty inputs."""
    if len(y_true) == 0:
        return np.nan
    return float(accuracy_score(y_true, y_pred))


def _safe_precision_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: Any) -> float:
    """Return precision without raising on zero division."""
    if len(y_true) == 0:
        return np.nan
    return float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0))


def _safe_recall_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: Any) -> float:
    """Return recall without raising on zero division."""
    if len(y_true) == 0:
        return np.nan
    return float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0))


def _safe_f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: Any) -> float:
    """Return F1 without raising on zero division."""
    if len(y_true) == 0:
        return np.nan
    return float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0))


def _positive_class_scores(y_proba: Any) -> np.ndarray:
    """Return one probability score per row for the positive class."""
    probability_values = np.asarray(y_proba, dtype=float)
    if probability_values.ndim == 2:
        return probability_values[:, -1]
    return probability_values


def _safe_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return ROC AUC, or None when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _safe_pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return average precision, or None when PR AUC is not feasible."""
    if len(np.unique(y_true)) < 2 or y_true.sum() == 0:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def _safe_log_loss(y_true: np.ndarray, y_proba: Any, labels: list[Any] | None) -> float | None:
    """Return multiclass log loss, or None when probabilities are not usable."""
    try:
        return float(log_loss(y_true, y_proba, labels=labels))
    except ValueError:
        return None
