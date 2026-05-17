"""Rule-based model interpretation helpers.

The explanations in this module are intentionally conservative. They summarize
saved model outputs and avoid unsupported causal language.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def explain_model_run(model_run: dict[str, Any]) -> str:
    """Return a concise rule-based explanation for one saved ModelRun."""
    model_family = model_run.get("model_family")
    task_type = model_run.get("task_type")

    if model_family == "statistical" and task_type == "regression":
        return explain_linear_regression_run(model_run)
    if model_family == "statistical" and task_type == "binary_classification":
        return explain_logistic_regression_run(model_run)
    if model_family == "machine_learning" and task_type == "regression":
        return explain_ml_regression_run(model_run)
    if model_family == "machine_learning" and task_type == "binary_classification":
        return explain_ml_binary_classification_run(model_run)

    return "No rule-based interpretation is available for this model run yet."


def explain_linear_regression_run(model_run: dict[str, Any]) -> str:
    """Explain a saved statistical linear regression run."""
    target = model_run.get("target")
    features = _feature_text(model_run.get("features") or [])
    summary = _summary_lookup(model_run.get("statistical_summary"))
    train_metrics = _metric_group(model_run.get("train_metrics") or {})
    test_metrics = _metric_group(model_run.get("test_metrics") or {})
    coefficient_text = _linear_coefficient_direction_text(model_run.get("coefficient_table"))

    parts = [
        f"This linear regression models `{target}` using selected features: {features}.",
        _available_metrics_text(
            "Model fit statistics",
            {
                "R-squared": _first_available(summary, "R-squared", "r_squared"),
                "Adjusted R-squared": _first_available(summary, "Adjusted R-squared", "adjusted_r_squared"),
                "AIC": _first_available(summary, "AIC", "aic"),
                "BIC": _first_available(summary, "BIC", "bic"),
            },
        ),
        _available_metrics_text(
            "Test-set predictive metrics",
            {
                "RMSE": _first_available(test_metrics, "RMSE", "test_rmse", "rmse"),
                "MAE": _first_available(test_metrics, "MAE", "test_mae", "mae"),
                "R-squared": _first_available(test_metrics, "R-squared", "test_r2", "r2"),
            },
        ),
    ]
    if coefficient_text:
        parts.append(coefficient_text)
    parts.append(_regression_overfit_text(train_metrics, test_metrics))
    parts.append("Coefficient directions describe associations in this fitted dataset, not causal effects unless the study design supports causal interpretation.")
    return _join_parts(parts)


def explain_logistic_regression_run(model_run: dict[str, Any]) -> str:
    """Explain a saved statistical binary logistic regression run."""
    target = model_run.get("target")
    features = _feature_text(model_run.get("features") or [])
    preprocessing = model_run.get("preprocessing") or {}
    positive_class = preprocessing.get("positive_class", "the selected positive class")
    threshold = preprocessing.get("threshold")
    test_metrics = model_run.get("test_metrics") or {}
    summary = _summary_lookup(model_run.get("statistical_summary"))
    odds_text = _odds_ratio_text(model_run.get("coefficient_table"))

    parts = [
        f"This logistic regression models `{target}` as a binary outcome, with `{positive_class}` treated as the positive class, using features: {features}.",
        _available_metrics_text(
            "Test-set classification metrics",
            {
                "accuracy": _first_available(test_metrics, "accuracy", "test_accuracy"),
                "precision": _first_available(test_metrics, "precision", "test_precision"),
                "recall": _first_available(test_metrics, "recall", "test_recall"),
                "F1": _first_available(test_metrics, "F1", "test_f1"),
                "ROC AUC": _first_available(test_metrics, "ROC AUC", "test_roc_auc"),
            },
        ),
        _available_metrics_text(
            "Model fit statistics",
            {
                "AIC": _first_available(summary, "AIC", "aic"),
                "BIC": _first_available(summary, "BIC", "bic"),
            },
        ),
    ]
    if odds_text:
        parts.append(odds_text)
    if threshold is not None:
        parts.append(f"The reported class metrics depend on the classification threshold `{_format_value(threshold)}`; changing the threshold can change precision, recall, F1, and the confusion matrix.")
    else:
        parts.append("Classification metrics depend on the chosen probability threshold; changing the threshold can change precision, recall, F1, and the confusion matrix.")
    parts.append("Odds ratios describe associations with the positive class in this fitted dataset, not causal effects unless the study design supports causal interpretation.")
    return _join_parts(parts)


def explain_ml_regression_run(model_run: dict[str, Any]) -> str:
    """Explain a saved machine learning regression run."""
    target = model_run.get("target")
    features = _feature_text(model_run.get("features") or [])
    train_metrics = model_run.get("train_metrics") or {}
    test_metrics = model_run.get("test_metrics") or {}

    parts = [
        f"This machine learning regression model predicts `{target}` from selected features: {features}.",
        _available_metrics_text(
            "Predictive performance",
            {
                "train RMSE": _first_available(train_metrics, "train_rmse", "RMSE"),
                "test RMSE": _first_available(test_metrics, "test_rmse", "RMSE"),
                "train MAE": _first_available(train_metrics, "train_mae", "MAE"),
                "test MAE": _first_available(test_metrics, "test_mae", "MAE"),
                "train R2": _first_available(train_metrics, "train_r2", "R-squared"),
                "test R2": _first_available(test_metrics, "test_r2", "R-squared"),
            },
        ),
        _regression_overfit_text(train_metrics, test_metrics),
    ]
    importance_text = _feature_importance_text(model_run.get("feature_importance_table"))
    if importance_text:
        parts.append(importance_text)
    parts.append("Feature importance can highlight predictive signals, but it does not automatically imply causality.")
    return _join_parts(parts)


def explain_ml_binary_classification_run(model_run: dict[str, Any]) -> str:
    """Explain a saved machine learning binary classification run."""
    target = model_run.get("target")
    features = _feature_text(model_run.get("features") or [])
    preprocessing = model_run.get("preprocessing") or {}
    positive_class = preprocessing.get("positive_class", "the selected positive class")
    train_metrics = model_run.get("train_metrics") or {}
    test_metrics = model_run.get("test_metrics") or {}

    parts = [
        f"This machine learning binary classifier predicts `{target}`, treating `{positive_class}` as the positive class, using features: {features}.",
        _available_metrics_text(
            "Predictive performance",
            {
                "train accuracy": _first_available(train_metrics, "train_accuracy", "accuracy"),
                "test accuracy": _first_available(test_metrics, "test_accuracy", "accuracy"),
                "train precision": _first_available(train_metrics, "train_precision", "precision"),
                "test precision": _first_available(test_metrics, "test_precision", "precision"),
                "train recall": _first_available(train_metrics, "train_recall", "recall"),
                "test recall": _first_available(test_metrics, "test_recall", "recall"),
                "train F1": _first_available(train_metrics, "train_f1", "F1"),
                "test F1": _first_available(test_metrics, "test_f1", "F1"),
                "test ROC AUC": _first_available(test_metrics, "test_roc_auc", "ROC AUC"),
            },
        ),
        _classification_overfit_text(train_metrics, test_metrics),
    ]
    importance_text = _feature_importance_text(model_run.get("feature_importance_table"))
    if importance_text:
        parts.append(importance_text)
    parts.append("Feature importance can highlight predictive signals, but it does not automatically imply causality.")
    return _join_parts(parts)


def _linear_coefficient_direction_text(coefficient_table: list[dict[str, Any]]) -> str:
    """Summarize coefficient directions for linear regression."""
    coefficient_table = _records_from_table(coefficient_table)
    coefficients = _non_intercept_coefficients(coefficient_table, "estimate")
    if not coefficients:
        return ""

    top_coefficients = _top_by_absolute_value(coefficients, "estimate")
    phrases = []
    for row in top_coefficients:
        estimate = row.get("estimate")
        if estimate is None:
            continue
        direction = "positive" if estimate > 0 else "negative" if estimate < 0 else "near-zero"
        phrases.append(f"`{row.get('term')}` has a {direction} coefficient ({_format_value(estimate)})")

    if not phrases:
        return ""
    return "Coefficient direction summary: " + "; ".join(phrases) + "."


def _odds_ratio_text(coefficient_table: list[dict[str, Any]]) -> str:
    """Summarize odds ratios for logistic regression."""
    coefficient_table = _records_from_table(coefficient_table)
    coefficients = _non_intercept_coefficients(coefficient_table, "odds_ratio")
    if not coefficients:
        return ""

    top_coefficients = _top_by_distance_from_one(coefficients)
    phrases = []
    for row in top_coefficients:
        odds_ratio = row.get("odds_ratio")
        if odds_ratio is None:
            continue
        direction = "higher odds" if odds_ratio > 1 else "lower odds" if odds_ratio < 1 else "similar odds"
        phrases.append(f"`{row.get('term')}` is associated with {direction} of the positive class (odds ratio {_format_value(odds_ratio)})")

    if not phrases:
        return ""
    return "Odds ratio summary: " + "; ".join(phrases) + "."


def _feature_importance_text(feature_importance_table: list[dict[str, Any]]) -> str:
    """Summarize the highest feature-importance rows."""
    feature_importance_table = _records_from_table(feature_importance_table)
    if not feature_importance_table:
        return ""

    rows = [
        row
        for row in feature_importance_table
        if isinstance(row, dict) and row.get("feature") is not None and row.get("importance") is not None
    ]
    if not rows:
        return ""

    rows = sorted(rows, key=lambda row: abs(float(row.get("importance", 0))), reverse=True)[:5]
    phrases = [
        f"`{row.get('feature')}` ({_format_value(row.get('importance'))}, {row.get('importance_type', 'importance')})"
        for row in rows
    ]
    return "Top feature-importance signals: " + "; ".join(phrases) + "."


def _regression_overfit_text(train_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> str:
    """Return a simple train/test gap warning for regression."""
    train_rmse = _first_available(train_metrics, "train_rmse", "RMSE", "rmse")
    test_rmse = _first_available(test_metrics, "test_rmse", "RMSE", "rmse")
    train_r2 = _first_available(train_metrics, "train_r2", "R-squared", "r2")
    test_r2 = _first_available(test_metrics, "test_r2", "R-squared", "r2")

    if _is_number(train_rmse) and _is_number(test_rmse) and train_rmse > 0 and test_rmse > train_rmse * 1.25:
        return "The test RMSE is noticeably higher than the train RMSE, which may indicate overfitting or a difficult test split."
    if _is_number(train_r2) and _is_number(test_r2) and train_r2 - test_r2 > 0.15:
        return "The train R-squared is noticeably higher than the test R-squared, which may indicate overfitting."
    return "No large train/test performance gap is apparent from the available regression metrics."


def _classification_overfit_text(train_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> str:
    """Return a simple train/test gap warning for binary classification."""
    train_f1 = _first_available(train_metrics, "train_f1", "F1", "f1")
    test_f1 = _first_available(test_metrics, "test_f1", "F1", "f1")
    train_accuracy = _first_available(train_metrics, "train_accuracy", "accuracy")
    test_accuracy = _first_available(test_metrics, "test_accuracy", "accuracy")

    if _is_number(train_f1) and _is_number(test_f1) and train_f1 - test_f1 > 0.10:
        return "The train F1 is noticeably higher than the test F1, which may indicate overfitting."
    if _is_number(train_accuracy) and _is_number(test_accuracy) and train_accuracy - test_accuracy > 0.10:
        return "The train accuracy is noticeably higher than the test accuracy, which may indicate overfitting."
    return "No large train/test performance gap is apparent from the available classification metrics."


def _available_metrics_text(label: str, metrics: dict[str, Any]) -> str:
    """Format available metrics without inventing missing values."""
    available = [
        f"{name}: {_format_value(value)}"
        for name, value in metrics.items()
        if value is not None
    ]
    if not available:
        return f"{label}: no values are available."
    return f"{label}: " + ", ".join(available) + "."


def _metric_group(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return transformed-scale metrics when statistical regression nests metrics."""
    if "transformed_scale" in metrics and isinstance(metrics["transformed_scale"], dict):
        return metrics["transformed_scale"]
    return metrics


def _summary_lookup(summary_rows: Any) -> dict[str, Any]:
    """Convert statistical summary records into a lookup dictionary."""
    if isinstance(summary_rows, dict):
        return summary_rows
    if isinstance(summary_rows, pd.DataFrame):
        summary_rows = summary_rows.where(pd.notna(summary_rows), None).to_dict(orient="records")
    lookup = {}
    for row in summary_rows or []:
        if isinstance(row, dict):
            lookup[row.get("statistic")] = row.get("value")
    return lookup


def _records_from_table(value: Any) -> list[dict[str, Any]]:
    """Normalize list/DataFrame table values without boolean-testing DataFrames."""
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        return value.where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _first_available(values: dict[str, Any], *keys: str) -> Any:
    """Return the first non-missing value from several possible keys."""
    for key in keys:
        if key in values and values[key] is not None:
            return values[key]
    return None


def _non_intercept_coefficients(coefficient_table: list[dict[str, Any]], value_key: str) -> list[dict[str, Any]]:
    """Return coefficient rows excluding intercept/constant terms."""
    rows = []
    for row in coefficient_table:
        if not isinstance(row, dict):
            continue
        term = str(row.get("term", "")).lower()
        if term in {"const", "intercept"}:
            continue
        if row.get(value_key) is not None:
            rows.append(row)
    return rows


def _top_by_absolute_value(rows: list[dict[str, Any]], value_key: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return rows with largest absolute numeric values."""
    return sorted(rows, key=lambda row: abs(float(row.get(value_key, 0))), reverse=True)[:limit]


def _top_by_distance_from_one(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Return odds-ratio rows farthest from 1."""
    return sorted(rows, key=lambda row: abs(float(row.get("odds_ratio", 1)) - 1), reverse=True)[:limit]


def _feature_text(features: list[str]) -> str:
    """Format a feature list for explanation text."""
    if not features:
        return "no features recorded"
    return ", ".join(f"`{feature}`" for feature in features)


def _format_value(value: Any) -> str:
    """Format metric values compactly."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _is_number(value: Any) -> bool:
    """Return True for int/float values that can be compared."""
    return isinstance(value, (int, float)) and value == value


def _join_parts(parts: list[str]) -> str:
    """Join non-empty explanation parts."""
    return " ".join(part for part in parts if part)
