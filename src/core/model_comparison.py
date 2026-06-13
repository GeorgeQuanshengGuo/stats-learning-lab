"""Helpers for building model comparison tables from saved ModelRuns."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_model_run_overview(model_runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a compact overview table for all saved model runs."""
    rows = []
    for run in model_runs:
        rows.append(
            {
                "run_id": run["run_id"],
                "timestamp": run["timestamp"],
                "task_type": run["task_type"],
                "model_family": run["model_family"],
                "model_name": run["model_name"],
                "target": run["target"],
                "features": _feature_text(run["features"]),
                "notes": run["notes"],
            }
        )
    return pd.DataFrame(rows)


def build_regression_comparison_table(model_runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Return comparison metrics for regression model runs only."""
    rows = []
    for run in model_runs:
        if run["task_type"] != "regression":
            continue

        train_metrics = _metric_group(run["train_metrics"])
        test_metrics = _metric_group(run["test_metrics"])
        summary = _summary_lookup(run["statistical_summary"])
        tuning = _tuning_lookup(run)

        rows.append(
            {
                "model_name": run["model_name"],
                "model_family": run["model_family"],
                "target": run["target"],
                "features": _feature_text(run["features"]),
                "train_rmse": _metric_value(train_metrics, "train_rmse", "RMSE"),
                "test_rmse": _metric_value(test_metrics, "test_rmse", "RMSE"),
                "train_mae": _metric_value(train_metrics, "train_mae", "MAE"),
                "test_mae": _metric_value(test_metrics, "test_mae", "MAE"),
                "train_r2": _metric_value(train_metrics, "train_r2", "R-squared"),
                "test_r2": _metric_value(test_metrics, "test_r2", "R-squared"),
                "cv_rmse_mean": _metric_value(test_metrics, "cv_rmse_mean"),
                "cv_rmse_std": _metric_value(test_metrics, "cv_rmse_std"),
                "cv_mae_mean": _metric_value(test_metrics, "cv_mae_mean"),
                "cv_mae_std": _metric_value(test_metrics, "cv_mae_std"),
                "cv_r2_mean": _metric_value(test_metrics, "cv_r2_mean"),
                "cv_r2_std": _metric_value(test_metrics, "cv_r2_std"),
                "aic": summary.get("AIC"),
                "bic": summary.get("BIC"),
                "adjusted_r_squared": summary.get("Adjusted R-squared"),
                "tuning_method": tuning.get("search_type"),
                "tuning_scoring": tuning.get("scoring"),
                "best_cv_score": tuning.get("best_cv_score"),
                "tuning_runtime_seconds": tuning.get("runtime_seconds"),
                "validation_warnings": _validation_warning_text(run),
            }
        )
    return pd.DataFrame(rows)


def build_binary_classification_comparison_table(model_runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Return comparison metrics for binary classification model runs only."""
    rows = []
    for run in model_runs:
        if run["task_type"] != "binary_classification":
            continue

        train_metrics = run["train_metrics"]
        test_metrics = run["test_metrics"]
        summary = _summary_lookup(run["statistical_summary"])
        tuning = _tuning_lookup(run)

        rows.append(
            {
                "model_name": run["model_name"],
                "model_family": run["model_family"],
                "target": run["target"],
                "features": _feature_text(run["features"]),
                "positive_class": run.get("preprocessing", {}).get("positive_class"),
                "train_accuracy": _metric_value(train_metrics, "train_accuracy", "accuracy"),
                "test_accuracy": _metric_value(test_metrics, "test_accuracy", "accuracy"),
                "train_precision": _metric_value(train_metrics, "train_precision", "precision"),
                "test_precision": _metric_value(test_metrics, "test_precision", "precision"),
                "train_recall": _metric_value(train_metrics, "train_recall", "recall"),
                "test_recall": _metric_value(test_metrics, "test_recall", "recall"),
                "train_f1": _metric_value(train_metrics, "train_f1", "F1"),
                "test_f1": _metric_value(test_metrics, "test_f1", "F1"),
                "train_roc_auc": _metric_value(train_metrics, "train_roc_auc", "ROC AUC"),
                "test_roc_auc": _metric_value(test_metrics, "test_roc_auc", "ROC AUC"),
                "train_pr_auc": _metric_value(train_metrics, "train_pr_auc", "PR AUC"),
                "test_pr_auc": _metric_value(test_metrics, "test_pr_auc", "PR AUC"),
                "cv_accuracy_mean": _metric_value(test_metrics, "cv_accuracy_mean"),
                "cv_accuracy_std": _metric_value(test_metrics, "cv_accuracy_std"),
                "cv_precision_mean": _metric_value(test_metrics, "cv_precision_mean"),
                "cv_precision_std": _metric_value(test_metrics, "cv_precision_std"),
                "cv_recall_mean": _metric_value(test_metrics, "cv_recall_mean"),
                "cv_recall_std": _metric_value(test_metrics, "cv_recall_std"),
                "cv_f1_mean": _metric_value(test_metrics, "cv_f1_mean"),
                "cv_f1_std": _metric_value(test_metrics, "cv_f1_std"),
                "cv_roc_auc_mean": _metric_value(test_metrics, "cv_roc_auc_mean"),
                "cv_roc_auc_std": _metric_value(test_metrics, "cv_roc_auc_std"),
                "aic": summary.get("AIC"),
                "bic": summary.get("BIC"),
                "pseudo_r_squared": summary.get("Pseudo R-squared"),
                "tuning_method": tuning.get("search_type"),
                "tuning_scoring": tuning.get("scoring"),
                "best_cv_score": tuning.get("best_cv_score"),
                "tuning_runtime_seconds": tuning.get("runtime_seconds"),
                "validation_warnings": _validation_warning_text(run),
            }
        )
    return pd.DataFrame(rows)


def build_multiclass_classification_comparison_table(model_runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Return comparison metrics for multiclass classification model runs only."""
    rows = []
    for run in model_runs:
        if run["task_type"] != "multiclass_classification":
            continue

        train_metrics = run["train_metrics"]
        test_metrics = run["test_metrics"]
        summary = _summary_lookup(run["statistical_summary"])

        rows.append(
            {
                "model_name": run["model_name"],
                "model_family": run["model_family"],
                "target": run["target"],
                "features": _feature_text(run["features"]),
                "target_classes": ", ".join(str(value) for value in run.get("preprocessing", {}).get("target_classes", [])),
                "reference_class": run.get("preprocessing", {}).get("reference_class"),
                "train_accuracy": _metric_value(train_metrics, "train_accuracy", "accuracy"),
                "test_accuracy": _metric_value(test_metrics, "test_accuracy", "accuracy"),
                "train_macro_precision": _metric_value(train_metrics, "train_macro_precision", "macro_precision"),
                "test_macro_precision": _metric_value(test_metrics, "test_macro_precision", "macro_precision"),
                "train_macro_recall": _metric_value(train_metrics, "train_macro_recall", "macro_recall"),
                "test_macro_recall": _metric_value(test_metrics, "test_macro_recall", "macro_recall"),
                "train_macro_f1": _metric_value(train_metrics, "train_macro_f1", "macro_f1"),
                "test_macro_f1": _metric_value(test_metrics, "test_macro_f1", "macro_f1"),
                "train_weighted_f1": _metric_value(train_metrics, "train_weighted_f1", "weighted_f1"),
                "test_weighted_f1": _metric_value(test_metrics, "test_weighted_f1", "weighted_f1"),
                "train_log_loss": _metric_value(train_metrics, "train_log_loss", "log_loss"),
                "test_log_loss": _metric_value(test_metrics, "test_log_loss", "log_loss"),
                "aic": summary.get("AIC"),
                "bic": summary.get("BIC"),
            }
        )
    return pd.DataFrame(rows)


def build_ordinal_classification_comparison_table(model_runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Return comparison metrics for ordinal classification model runs only."""
    rows = []
    for run in model_runs:
        if run["task_type"] != "ordinal_classification":
            continue

        train_metrics = run["train_metrics"]
        test_metrics = run["test_metrics"]
        summary = _summary_lookup(run["statistical_summary"])

        rows.append(
            {
                "model_name": run["model_name"],
                "model_family": run["model_family"],
                "target": run["target"],
                "features": _feature_text(run["features"]),
                "category_order": ", ".join(str(value) for value in run.get("preprocessing", {}).get("category_order", [])),
                "train_accuracy": _metric_value(train_metrics, "accuracy", "train_accuracy"),
                "test_accuracy": _metric_value(test_metrics, "accuracy", "test_accuracy"),
                "train_macro_f1": _metric_value(train_metrics, "macro_f1", "train_macro_f1"),
                "test_macro_f1": _metric_value(test_metrics, "macro_f1", "test_macro_f1"),
                "train_weighted_f1": _metric_value(train_metrics, "weighted_f1", "train_weighted_f1"),
                "test_weighted_f1": _metric_value(test_metrics, "weighted_f1", "test_weighted_f1"),
                "train_ordered_mae": _metric_value(train_metrics, "ordered_mae", "train_ordered_mae"),
                "test_ordered_mae": _metric_value(test_metrics, "ordered_mae", "test_ordered_mae"),
                "aic": summary.get("AIC"),
                "bic": summary.get("BIC"),
            }
        )
    return pd.DataFrame(rows)


def build_count_regression_comparison_table(model_runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Return comparison metrics for count regression model runs only."""
    rows = []
    for run in model_runs:
        if run["task_type"] != "count_regression":
            continue

        train_metrics = run["train_metrics"]
        test_metrics = run["test_metrics"]
        summary = _summary_lookup(run["statistical_summary"])

        rows.append(
            {
                "model_name": run["model_name"],
                "model_family": run["model_family"],
                "target": run["target"],
                "features": _feature_text(run["features"]),
                "train_rmse": _metric_value(train_metrics, "rmse", "RMSE", "train_rmse"),
                "test_rmse": _metric_value(test_metrics, "rmse", "RMSE", "test_rmse"),
                "train_mae": _metric_value(train_metrics, "mae", "MAE", "train_mae"),
                "test_mae": _metric_value(test_metrics, "mae", "MAE", "test_mae"),
                "train_mean_deviance": _metric_value(train_metrics, "mean_deviance"),
                "test_mean_deviance": _metric_value(test_metrics, "mean_deviance"),
                "aic": summary.get("AIC"),
                "bic": summary.get("BIC"),
                "variance_mean_ratio": summary.get("variance / mean"),
                "pearson_chi_square_per_df": summary.get("Pearson chi-square / df"),
                "zero_proportion": summary.get("zero proportion"),
                "alpha": summary.get("alpha"),
            }
        )
    return pd.DataFrame(rows)


def filter_model_runs(
    model_runs: list[dict[str, Any]],
    task_type: str,
    target: str,
    model_family: str,
) -> list[dict[str, Any]]:
    """Filter saved model runs by task type, target, and model family."""
    filtered = model_runs
    if task_type != "All":
        filtered = [run for run in filtered if run["task_type"] == task_type]
    if target != "All":
        filtered = [run for run in filtered if run["target"] == target]
    if model_family != "All":
        filtered = [run for run in filtered if run["model_family"] == model_family]
    return filtered


def _metric_group(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return the transformed-scale metrics for regression runs."""
    if "transformed_scale" in metrics:
        return metrics["transformed_scale"]
    return metrics


def _summary_lookup(summary_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Convert statistical summary records into a lookup dictionary."""
    if isinstance(summary_rows, dict):
        return summary_rows

    lookup = {}
    for row in summary_rows or []:
        lookup[row.get("statistic")] = row.get("value")
    return lookup


def _metric_value(metrics: dict[str, Any], *keys: str) -> Any:
    """Return the first available metric value from several possible key names."""
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _tuning_lookup(run: dict[str, Any]) -> dict[str, Any]:
    """Return tuning metadata when a ModelRun was created by a search."""
    return (run.get("preprocessing") or {}).get("tuning") or {}


def _validation_warning_text(run: dict[str, Any]) -> str:
    """Return compact validation warning text from ModelRun metadata."""
    readiness = (run.get("preprocessing") or {}).get("validation_readiness") or {}
    warnings = readiness.get("warnings", []) if isinstance(readiness, dict) else []
    return " ".join(str(warning) for warning in warnings)


def _feature_text(features: list[str]) -> str:
    """Render feature lists compactly for tables and CSV export."""
    return ", ".join(features)
