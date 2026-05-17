"""Tests for chart explanation defaults."""

from src.ui.chart_explanations import GENERAL_WARNINGS, get_chart_explanation, merge_chart_explanation


def test_default_explanations_include_required_chart_types() -> None:
    required_types = [
        "histogram",
        "boxplot",
        "correlation_heatmap",
        "scatter_plot",
        "grouped_boxplot",
        "outlier_chart",
        "residuals_vs_fitted",
        "scale_location",
        "leverage",
        "cooks_distance",
        "qq_plot",
        "confusion_matrix",
        "roc_curve",
        "pr_curve",
        "calibration_plot",
        "feature_importance",
        "pdp",
        "ice",
        "prediction_interval_plot",
        "overfitting_diagnostics",
        "learning_curve",
    ]

    for chart_type in required_types:
        explanation = get_chart_explanation(chart_type)
        assert explanation["how_to_read"]
        assert "warnings" in explanation


def test_required_warnings_are_available() -> None:
    assert "Correlation does not imply causation." in GENERAL_WARNINGS.values()
    assert "Feature importance is not causal importance." in GENERAL_WARNINGS.values()
    assert "Prediction intervals depend on assumptions or the interval method." in GENERAL_WARNINGS.values()
    assert "ROC AUC may be misleading under class imbalance." in GENERAL_WARNINGS.values()
    assert "p-values do not measure effect size." in GENERAL_WARNINGS.values()
    assert "VIF measures multicollinearity, not overfitting." in GENERAL_WARNINGS.values()


def test_merge_chart_explanation_preserves_defaults_and_adds_warning() -> None:
    merged = merge_chart_explanation("histogram", warnings=["Custom warning."])

    assert "Bars show" in merged["how_to_read"]
    assert "Custom warning." in merged["warnings"]
