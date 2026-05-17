"""Overfitting and CV-stability diagnostics for saved ModelRuns."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.modeling.diagnostics.diagnostic_rules import (
    cv_stability_warning,
    overfitting_warning_from_value,
)


def build_regression_overfitting_diagnostics(model_run: dict[str, Any]) -> pd.DataFrame:
    """Return train/test gap diagnostics for a regression ModelRun."""
    train_metrics = _metric_group(model_run.get("train_metrics") or {})
    test_metrics = _metric_group(model_run.get("test_metrics") or {})

    train_rmse = _metric_value(train_metrics, "train_rmse", "RMSE", "rmse")
    test_rmse = _metric_value(test_metrics, "test_rmse", "RMSE", "rmse")
    train_r2 = _metric_value(train_metrics, "train_r2", "R-squared", "r2")
    test_r2 = _metric_value(test_metrics, "test_r2", "R-squared", "r2")

    ratio = _safe_ratio(test_rmse, train_rmse)
    r2_gap = _safe_difference(train_r2, test_r2)
    rows = [
        {"metric": "train_rmse", "value": train_rmse, "risk_level": "info", "message": "Training RMSE."},
        {"metric": "test_rmse", "value": test_rmse, "risk_level": "info", "message": "Test RMSE."},
        overfitting_warning_from_value("test_rmse_train_rmse_ratio", ratio),
        {"metric": "train_r2", "value": train_r2, "risk_level": "info", "message": "Training R-squared."},
        {"metric": "test_r2", "value": test_r2, "risk_level": "info", "message": "Test R-squared."},
        overfitting_warning_from_value("train_r2_minus_test_r2", r2_gap),
    ]
    rows.extend(build_cv_stability_diagnostics(model_run).to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_classification_overfitting_diagnostics(model_run: dict[str, Any]) -> pd.DataFrame:
    """Return train/test gap diagnostics for classification ModelRuns."""
    train_metrics = model_run.get("train_metrics") or {}
    test_metrics = model_run.get("test_metrics") or {}

    accuracy_gap = _safe_difference(
        _metric_value(train_metrics, "train_accuracy", "accuracy"),
        _metric_value(test_metrics, "test_accuracy", "accuracy"),
    )
    f1_gap = _safe_difference(
        _metric_value(train_metrics, "train_f1", "f1", "F1"),
        _metric_value(test_metrics, "test_f1", "f1", "F1"),
    )
    auc_gap = _safe_difference(
        _metric_value(train_metrics, "train_roc_auc", "roc_auc", "ROC AUC"),
        _metric_value(test_metrics, "test_roc_auc", "roc_auc", "ROC AUC"),
    )

    rows = [
        overfitting_warning_from_value("train_accuracy_minus_test_accuracy", accuracy_gap),
        overfitting_warning_from_value("train_f1_minus_test_f1", f1_gap),
        overfitting_warning_from_value("train_auc_minus_test_auc", auc_gap),
    ]
    rows.extend(build_cv_stability_diagnostics(model_run).to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_cv_stability_diagnostics(model_run: dict[str, Any]) -> pd.DataFrame:
    """Return CV mean/std diagnostic rows when available."""
    test_metrics = _metric_group(model_run.get("test_metrics") or {})
    rows = []
    metric_prefixes = [
        "cv_rmse",
        "cv_mae",
        "cv_r2",
        "cv_accuracy",
        "cv_precision",
        "cv_recall",
        "cv_f1",
        "cv_roc_auc",
    ]
    for prefix in metric_prefixes:
        mean_key = f"{prefix}_mean"
        std_key = f"{prefix}_std"
        if mean_key not in test_metrics and std_key not in test_metrics:
            continue
        warning = cv_stability_warning(
            _as_float(test_metrics.get(mean_key)),
            _as_float(test_metrics.get(std_key)),
        )
        warning["metric"] = prefix
        rows.append(warning)
    return pd.DataFrame(rows, columns=["metric", "value", "risk_level", "message"])


def build_overfitting_diagnostics(model_run: dict[str, Any]) -> pd.DataFrame:
    """Dispatch overfitting diagnostics based on ModelRun task type."""
    task_type = model_run.get("task_type")
    if task_type in {"regression", "count_regression"}:
        return build_regression_overfitting_diagnostics(model_run)
    if task_type in {"binary_classification", "multiclass_classification", "ordinal_classification"}:
        return build_classification_overfitting_diagnostics(model_run)
    return pd.DataFrame(columns=["metric", "value", "risk_level", "message"])


def _metric_group(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return transformed-scale metrics when available."""
    if "transformed_scale" in metrics and isinstance(metrics["transformed_scale"], dict):
        return metrics["transformed_scale"]
    return metrics


def _metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
    """Return the first available numeric metric value."""
    for key in keys:
        if key in metrics:
            return _as_float(metrics[key])
    return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator / denominator when possible."""
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator / denominator)


def _safe_difference(left: float | None, right: float | None) -> float | None:
    """Return left - right when possible."""
    if left is None or right is None:
        return None
    return float(left - right)


def _as_float(value: Any) -> float | None:
    """Convert values to float, returning None when unavailable."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

