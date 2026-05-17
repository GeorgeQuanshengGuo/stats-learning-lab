"""Rule helpers for model diagnostics.

The thresholds here are simple heuristics. They are meant to flag runs that
deserve review, not to prove that a model is good or bad.
"""

from __future__ import annotations

from typing import Any


def overfitting_warning_from_value(
    metric_name: str,
    value: float | None,
) -> dict[str, Any]:
    """Return a warning level and message for one overfitting diagnostic."""
    if value is None:
        return _rule_result(metric_name, value, "not_available", "Metric is not available.")

    if metric_name == "test_rmse_train_rmse_ratio":
        if value > 1.50:
            return _rule_result(metric_name, value, "strong_warning", "Test RMSE is more than 1.50 times train RMSE.")
        if value > 1.25:
            return _rule_result(metric_name, value, "warning", "Test RMSE is more than 1.25 times train RMSE.")
        return _rule_result(metric_name, value, "ok", "Train/test RMSE gap is not large by this heuristic.")

    if metric_name == "train_r2_minus_test_r2":
        if value > 0.20:
            return _rule_result(metric_name, value, "strong_warning", "Train R-squared exceeds test R-squared by more than 0.20.")
        if value > 0.10:
            return _rule_result(metric_name, value, "warning", "Train R-squared exceeds test R-squared by more than 0.10.")
        return _rule_result(metric_name, value, "ok", "Train/test R-squared gap is not large by this heuristic.")

    if metric_name in {"train_auc_minus_test_auc", "train_f1_minus_test_f1"}:
        display_name = "AUC" if "auc" in metric_name else "F1"
        if value > 0.10:
            return _rule_result(metric_name, value, "strong_warning", f"Train {display_name} exceeds test {display_name} by more than 0.10.")
        if value > 0.05:
            return _rule_result(metric_name, value, "warning", f"Train {display_name} exceeds test {display_name} by more than 0.05.")
        return _rule_result(metric_name, value, "ok", f"Train/test {display_name} gap is not large by this heuristic.")

    if metric_name == "train_accuracy_minus_test_accuracy":
        if value > 0.10:
            return _rule_result(metric_name, value, "warning", "Train accuracy exceeds test accuracy by more than 0.10.")
        return _rule_result(metric_name, value, "ok", "Train/test accuracy gap is not large by this heuristic.")

    return _rule_result(metric_name, value, "not_available", "No rule is configured for this metric.")


def vif_warning(vif_value: float | None) -> tuple[str, str]:
    """Return a heuristic VIF warning level and explanation."""
    if vif_value is None:
        return "not_available", "VIF could not be computed."
    if vif_value > 10:
        return "strong_warning", "VIF is above 10, a strong heuristic warning for multicollinearity."
    if vif_value > 5:
        return "warning", "VIF is above 5, a heuristic warning for multicollinearity."
    return "ok", "VIF is usually acceptable by the common <= 5 heuristic."


def cv_stability_warning(mean_value: float | None, std_value: float | None) -> dict[str, Any]:
    """Return a simple cross-validation stability warning."""
    if mean_value is None or std_value is None:
        return _rule_result("cv_stability", None, "not_available", "CV mean/std is not available.")
    if mean_value == 0:
        return _rule_result("cv_stability", None, "not_available", "CV mean is zero, so relative stability cannot be computed.")

    relative_std = abs(std_value / mean_value)
    if relative_std > 0.25:
        level = "warning"
        message = "CV standard deviation is more than 25% of the CV mean; results may be unstable."
    else:
        level = "ok"
        message = "CV standard deviation is not large relative to the CV mean by this heuristic."
    return _rule_result("cv_stability", relative_std, level, message)


def _rule_result(metric_name: str, value: Any, level: str, message: str) -> dict[str, Any]:
    """Build one rule result row."""
    return {
        "metric": metric_name,
        "value": value,
        "risk_level": level,
        "message": message,
    }

