"""Default chart explanations and cautions."""

from __future__ import annotations


DEFAULT_CHART_EXPLANATIONS = {
    "histogram": {
        "how_to_read": "Bars show how observations are distributed across value ranges. Look for skewness, gaps, and unusual tails.",
        "warnings": ["Histograms are sensitive to binning choices."],
        "next_step": "Check a boxplot or summary table if the distribution looks skewed or has extreme values.",
        "glossary_terms": ["outlier"],
    },
    "boxplot": {
        "how_to_read": "The box shows the middle half of the data, the line marks the median, and points may flag possible outliers.",
        "warnings": ["A flagged point is not automatically an error."],
        "next_step": "Investigate flagged values before deciding whether to clean or transform them.",
        "glossary_terms": ["outlier"],
    },
    "categorical_bar": {
        "how_to_read": "Bars compare how often each category or value appears.",
        "warnings": ["Rare categories may be hard to see and may need table review."],
        "next_step": "Check whether categories need grouping, cleaning, or clearer labels.",
        "glossary_terms": ["summary_table"],
    },
    "missing_bar": {
        "how_to_read": "Bars show how many missing values each variable has.",
        "warnings": ["Missingness can bias results if it is related to the outcome or predictors."],
        "next_step": "Move to Data Cleaning if missingness needs a confirmed handling choice.",
        "glossary_terms": ["missing_value", "imputation"],
    },
    "datetime_counts": {
        "how_to_read": "The line shows how many observations occur on each date or time bucket.",
        "warnings": ["Date parsing errors or uneven collection periods can distort the pattern."],
        "next_step": "Check date formats and consider time-aware summaries if the pattern matters.",
        "glossary_terms": ["eda"],
    },
    "correlation_heatmap": {
        "how_to_read": "Colors show the strength and direction of pairwise numeric associations.",
        "warnings": ["Correlation does not imply causation.", "Pearson correlation may miss nonlinear relationships."],
        "next_step": "Inspect strong relationships with scatter plots or domain knowledge.",
        "glossary_terms": ["correlation", "pearson_correlation", "spearman_correlation"],
    },
    "scatter_plot": {
        "how_to_read": "Each point is one observation. Look for direction, shape, clusters, and unusual points.",
        "warnings": ["A visible pattern is exploratory and does not prove causation."],
        "next_step": "Consider correlation summaries or model diagnostics if using the variables for modeling.",
        "glossary_terms": ["correlation", "outlier"],
    },
    "grouped_boxplot": {
        "how_to_read": "Each box compares a numeric variable's distribution within a category.",
        "warnings": ["Groups with small counts can make patterns unstable."],
        "next_step": "Review the grouped summary table before modeling category effects.",
        "glossary_terms": ["summary_table", "outlier"],
    },
    "outlier_chart": {
        "how_to_read": "Highlighted points are observations flagged by the selected outlier method.",
        "warnings": ["A flagged row is not automatically an error."],
        "next_step": "Investigate flagged rows with domain context before cleaning or excluding them.",
        "glossary_terms": ["outlier"],
    },
    "residuals_vs_fitted": {
        "how_to_read": "Residuals should ideally be centered around zero without a strong pattern.",
        "warnings": ["Patterns can indicate nonlinearity, heteroscedasticity, or missing structure."],
        "next_step": "Check Q-Q plots, leverage, and model specification if patterns appear.",
        "glossary_terms": ["residual", "heteroscedasticity"],
    },
    "scale_location": {
        "how_to_read": "The vertical spread should be roughly even across fitted values.",
        "warnings": ["Increasing or decreasing spread can suggest heteroscedasticity."],
        "next_step": "Consider transformations, alternative models, or robust methods if spread changes strongly.",
        "glossary_terms": ["heteroscedasticity", "residual"],
    },
    "leverage": {
        "how_to_read": "Points far from the main cloud may have high leverage or influence.",
        "warnings": ["Influential observations are not automatically wrong."],
        "next_step": "Inspect flagged rows before deciding whether the model is overly sensitive.",
        "glossary_terms": ["leverage", "cooks_distance"],
    },
    "cooks_distance": {
        "how_to_read": "Tall bars indicate observations that may strongly affect fitted coefficients.",
        "warnings": ["Cook's distance is a diagnostic flag, not a deletion rule."],
        "next_step": "Investigate large values with residual and leverage diagnostics.",
        "glossary_terms": ["cooks_distance"],
    },
    "observed_vs_predicted": {
        "how_to_read": "Compare observed outcomes with model predictions. Better predictions sit closer to the diagonal pattern.",
        "warnings": ["Good visual fit on one split does not guarantee future performance."],
        "next_step": "Review test metrics and diagnostics before reporting the model.",
        "glossary_terms": ["residual", "rmse", "mae"],
    },
    "qq_plot": {
        "how_to_read": "Points close to the diagonal suggest residuals are roughly normal.",
        "warnings": ["Departures in the tails can affect inference in small samples."],
        "next_step": "Review residuals and consider transformations or robust methods if departures are severe.",
        "glossary_terms": ["normal_qq_plot", "residual"],
    },
    "confusion_matrix": {
        "how_to_read": "Rows and columns compare actual and predicted classes to show correct and incorrect classifications.",
        "warnings": ["The matrix changes when the classification threshold changes."],
        "next_step": "Review precision, recall, and the threshold choice for the use case.",
        "glossary_terms": ["confusion_matrix", "precision", "recall", "threshold"],
    },
    "roc_curve": {
        "how_to_read": "The curve shows true positive rate versus false positive rate across thresholds.",
        "warnings": ["ROC AUC may be misleading under class imbalance."],
        "next_step": "Check PR AUC and threshold-specific metrics when positives are rare.",
        "glossary_terms": ["roc_auc", "threshold"],
    },
    "pr_curve": {
        "how_to_read": "The curve shows precision and recall tradeoffs across thresholds.",
        "warnings": ["PR curves are especially useful when the positive class is rare."],
        "next_step": "Choose a threshold based on the cost of false positives and false negatives.",
        "glossary_terms": ["pr_auc", "precision", "recall", "threshold"],
    },
    "calibration_plot": {
        "how_to_read": "Compare predicted probabilities with observed event rates.",
        "warnings": ["Good ranking does not guarantee calibrated probabilities."],
        "next_step": "Consider calibration methods if probabilities will drive decisions.",
        "glossary_terms": ["calibration", "prediction_probability"],
    },
    "probability_distribution": {
        "how_to_read": "The distribution shows how predicted probabilities differ across actual classes.",
        "warnings": ["The chosen threshold determines final class labels."],
        "next_step": "Compare the probability distribution with the confusion matrix and threshold.",
        "glossary_terms": ["prediction_probability", "threshold"],
    },
    "feature_importance": {
        "how_to_read": "Larger values indicate stronger model reliance on a feature by the selected importance method.",
        "warnings": ["Feature importance is not causal importance.", "Correlated features can distort importance scores."],
        "next_step": "Use PDP, ICE, or domain review for deeper interpretation.",
        "glossary_terms": ["feature_importance", "permutation_importance"],
    },
    "pdp": {
        "how_to_read": "The line shows average model response as a feature changes.",
        "warnings": ["PDP shows model behavior, not causal effects.", "Correlated features can make PDP misleading."],
        "next_step": "Use ICE curves to inspect individual-level variation.",
        "glossary_terms": ["pdp", "ice"],
    },
    "ice": {
        "how_to_read": "Each line shows how one observation's model prediction changes as a feature changes.",
        "warnings": ["ICE curves show model behavior, not causal effects."],
        "next_step": "Compare ICE variation with the average PDP pattern.",
        "glossary_terms": ["ice", "pdp"],
    },
    "decision_tree": {
        "how_to_read": "Nodes show the split rules used by the tree. Leaves show final prediction regions.",
        "warnings": ["Deep trees may overfit and become difficult to interpret."],
        "next_step": "Review tree depth, leaf sizes, and test performance before trusting the rules.",
        "glossary_terms": ["overfitting", "feature_importance"],
    },
    "learning_curve": {
        "how_to_read": "The curves compare training and validation score as training size increases.",
        "warnings": ["Learning curves refit models and can vary by split and scoring metric."],
        "next_step": "Use the curve to decide whether more data or simpler models may help.",
        "glossary_terms": ["cross_validation", "overfitting"],
    },
    "pca_scree": {
        "how_to_read": "Bars show how much variance each principal component explains.",
        "warnings": ["PCA components are mathematical summaries, not causal factors."],
        "next_step": "Use cumulative variance to choose how many components to keep.",
        "glossary_terms": ["pca"],
    },
    "pca_scatter": {
        "how_to_read": "Each point is an observation projected onto the first two principal components.",
        "warnings": ["Separation in PCA space is exploratory and depends on selected features and scaling."],
        "next_step": "Use this as a visual guide, not as proof of real-world groups.",
        "glossary_terms": ["pca"],
    },
    "cluster_chart": {
        "how_to_read": "Colors or bars summarize clusters created by the selected algorithm.",
        "warnings": ["Clusters are exploratory groupings, not automatically real-world categories."],
        "next_step": "Compare cluster summaries and check whether the grouping is useful for the analysis question.",
        "glossary_terms": ["clustering"],
    },
    "anomaly_chart": {
        "how_to_read": "Highlighted observations are unusual under the selected anomaly detection method.",
        "warnings": ["Anomaly detection does not prove that rows are errors."],
        "next_step": "Inspect top anomalous rows before deciding on any cleaning action.",
        "glossary_terms": ["outlier", "anomaly"],
    },
    "prediction_interval_plot": {
        "how_to_read": "The interval shows uncertainty around a prediction using the selected method.",
        "warnings": ["Prediction intervals depend on assumptions or the interval method."],
        "next_step": "Report the interval method clearly and avoid overclaiming precision.",
        "glossary_terms": ["prediction_interval"],
    },
    "overfitting_diagnostics": {
        "how_to_read": "Compare train and test metrics. Large gaps often mean the model fits the training data better than new data.",
        "warnings": ["Train/test gaps are warning signs, not proof by themselves."],
        "next_step": "Review cross-validation stability, model complexity, and feature choices if the gaps are large.",
        "glossary_terms": ["overfitting", "cross_validation", "train_test_split"],
    },
    "vif_bar": {
        "how_to_read": "Higher VIF values indicate stronger predictor overlap in the design matrix.",
        "warnings": ["VIF measures multicollinearity, not overfitting."],
        "next_step": "Review high-VIF predictors before interpreting coefficients.",
        "glossary_terms": ["vif", "multicollinearity"],
    },
}

GENERAL_WARNINGS = {
    "correlation_causation": "Correlation does not imply causation.",
    "feature_importance_causality": "Feature importance is not causal importance.",
    "prediction_interval_assumptions": "Prediction intervals depend on assumptions or the interval method.",
    "roc_auc_imbalance": "ROC AUC may be misleading under class imbalance.",
    "p_value_effect_size": "p-values do not measure effect size.",
    "vif_not_overfitting": "VIF measures multicollinearity, not overfitting.",
}


def get_chart_explanation(chart_type: str) -> dict:
    """Return default explanation metadata for a chart type."""
    return DEFAULT_CHART_EXPLANATIONS.get(
        chart_type,
        {
            "how_to_read": "Read this chart in the context of the selected variables and method.",
            "warnings": [],
            "next_step": None,
            "glossary_terms": [],
        },
    )


def merge_chart_explanation(
    chart_type: str,
    how_to_read: str | None = None,
    warnings: list[str] | None = None,
    next_step: str | None = None,
) -> dict:
    """Merge caller-provided chart guidance with chart-type defaults."""
    defaults = get_chart_explanation(chart_type)
    merged_warnings = list(defaults.get("warnings", []))
    for warning in warnings or []:
        if warning not in merged_warnings:
            merged_warnings.append(warning)
    return {
        "how_to_read": how_to_read or defaults.get("how_to_read"),
        "warnings": merged_warnings,
        "next_step": next_step or defaults.get("next_step"),
        "glossary_terms": defaults.get("glossary_terms", []),
    }
