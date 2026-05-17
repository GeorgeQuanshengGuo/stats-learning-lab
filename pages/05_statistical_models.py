"""Streamlit page for statistical models."""

import pandas as pd
import streamlit as st

from src.core.model_artifacts import save_model_artifact
from src.core.model_run import add_model_run_to_session, create_model_run, get_model_runs
from src.data.transformations import inverse_transform_values
from src.eda.summary import build_summary_table
from src.modeling.statistical.linear_regression import (
    regression_metrics,
    plot_normal_qq,
    plot_residuals_vs_fitted,
    plot_residuals_vs_leverage,
    plot_scale_location,
    run_linear_regression,
)
from src.modeling.diagnostics.influence import INFLUENCE_WARNING
from src.modeling.statistical.logistic_regression import (
    classification_metrics,
    confusion_matrix_table,
    find_best_f1_threshold,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_probability_distribution,
    plot_roc_curve,
    run_logistic_regression,
)
from src.modeling.statistical.multinomial_logistic import run_multinomial_logistic_regression
from src.modeling.statistical.ordinal_regression import (
    PROPORTIONAL_ODDS_WARNING,
    run_ordinal_logistic_regression,
)
from src.modeling.statistical.count_regression import (
    COUNT_MODEL_OPTIONS,
    NEGATIVE_BINOMIAL_MODEL_NAME,
    POISSON_MODEL_NAME,
    plot_observed_vs_predicted_counts,
    run_count_regression,
)
from src.reporting.formula_builder import (
    build_estimated_linear_formula,
    build_estimated_logistic_formula,
    build_symbolic_linear_formula,
    build_symbolic_logistic_formula,
)
from src.ui.chart_cards import render_plotly_chart_card
from src.ui.page_templates import render_dataset_required_empty_state, render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles
from src.visualization.influence_plots import (
    plot_cooks_distance,
    plot_influence_residuals_vs_fitted,
    plot_leverage_vs_standardized_residual,
    plot_studentized_residuals,
)


def _combine_formula_parts(*formula_parts: dict | None) -> dict:
    """Merge formula dictionaries while keeping later notes readable."""
    combined = {}
    notes = []
    for formula in formula_parts:
        if not formula:
            continue
        for key, value in formula.items():
            if key == "note":
                notes.append(value)
            else:
                combined[key] = value
    if notes:
        combined["note"] = " ".join(notes)
    return combined


def _attach_formula_to_model_run(model_run: dict, formula_latex: dict | None) -> dict:
    """Attach formula metadata after creation for compatibility with older helpers."""
    model_run["formula_latex"] = formula_latex
    return model_run


def _build_linear_model_run(
    result: dict,
    target_metadata: dict | None = None,
    original_scale_metrics: dict | None = None,
) -> dict:
    """Create a unified ModelRun for linear regression."""
    preprocessing = {
        "categorical_encoding": "one-hot drop_first fitted on train columns",
        "missing_values": "complete-case rows for selected variables",
    }
    notes = "OLS linear regression fitted with statsmodels."
    if target_metadata is not None:
        preprocessing["target_transformation"] = {
            "method": target_metadata["method"],
            "original_target": target_metadata["original_target"],
            "transformed_target": target_metadata["transformed_target"],
            "parameters": target_metadata.get("parameters", {}),
            "inverse_transform_available": target_metadata["inverse_transform_available"],
        }
        notes = f"{notes} Target was transformed using {target_metadata['method']}."
    if original_scale_metrics is not None:
        notes = f"{notes} Original-scale test metrics were computed by inverse transforming predictions."
    formula_latex = _combine_formula_parts(
        build_symbolic_linear_formula(result["y_column"], result["x_columns"]),
        build_estimated_linear_formula(result["y_column"], result["coefficient_table"]),
    )

    test_metrics = {"transformed_scale": result["test_metrics"]}
    if original_scale_metrics is not None:
        test_metrics["original_scale"] = original_scale_metrics

    model_run = create_model_run(
        task_type="regression",
        model_family="statistical",
        model_name="linear_regression_ols",
        target=result["y_column"],
        features=result["x_columns"],
        split_config={
            "test_size": result["test_size"],
            "random_state": result["random_state"],
            "rows_used": result["rows_used"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
        },
        preprocessing=preprocessing,
        statistical_summary=result["model_statistics"],
        train_metrics={"transformed_scale": result["train_metrics"]},
        test_metrics=test_metrics,
        coefficient_table=result["coefficient_table"],
        diagnostic_plot_keys=[
            "residuals_vs_fitted",
            "normal_qq",
            "scale_location",
            "residuals_vs_leverage",
            "influence_residuals_vs_fitted",
            "cooks_distance",
            "studentized_residuals",
        ],
        notes=notes,
    )
    return _attach_formula_to_model_run(model_run, formula_latex)


def _build_logistic_model_run(result: dict) -> dict:
    """Create a unified ModelRun for binary logistic regression."""
    formula_latex = _combine_formula_parts(
        build_symbolic_logistic_formula(result["y_column"], result["x_columns"], result["positive_class"]),
        build_estimated_logistic_formula(result["y_column"], result["coefficient_table"], result["positive_class"]),
    )
    model_run = create_model_run(
        task_type="binary_classification",
        model_family="statistical",
        model_name="binary_logistic_regression_logit",
        target=result["y_column"],
        features=result["x_columns"],
        split_config={
            "test_size": result["test_size"],
            "random_state": result["random_state"],
            "rows_used": result["rows_used"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
        },
        preprocessing={
            "positive_class": result["positive_class"],
            "categorical_encoding": "one-hot drop_first fitted on train columns",
            "missing_values": "complete-case rows for selected variables",
            "threshold": result["threshold"],
            "threshold_strategy": result["threshold_strategy"],
        },
        statistical_summary=result["model_statistics"],
        train_metrics=result["train_metrics"],
        test_metrics=result["test_metrics"],
        coefficient_table=result["coefficient_table"],
        diagnostic_plot_keys=[
            "roc_curve",
            "precision_recall_curve",
            "confusion_matrix",
            "predicted_probability_distribution",
        ],
        notes="Binary Logit model fitted with statsmodels.",
    )
    return _attach_formula_to_model_run(model_run, formula_latex)


def _build_multinomial_model_run(result: dict) -> dict:
    """Create a unified ModelRun for multinomial logistic regression."""
    return create_model_run(
        task_type="multiclass_classification",
        model_family="statistical",
        model_name="multinomial_logistic_regression_mnlogit",
        target=result["y_column"],
        features=result["x_columns"],
        split_config={
            "test_size": result["test_size"],
            "random_state": result["random_state"],
            "rows_used": result["rows_used"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "stratified": True,
        },
        preprocessing={
            "reference_class": result["reference_class"],
            "target_classes": result["target_classes"],
            "categorical_encoding": "one-hot drop_first fitted on train columns",
            "missing_values": "complete-case rows for selected variables",
        },
        statistical_summary=result["model_statistics"],
        train_metrics=result["train_metrics"],
        test_metrics=result["test_metrics"],
        coefficient_table=result["coefficient_table"],
        diagnostic_plot_keys=["confusion_matrix", "class_wise_metrics"],
        formula_latex=result.get("formula_latex"),
        notes="Multinomial Logit model fitted with statsmodels.",
    )


def _build_ordinal_model_run(result: dict) -> dict:
    """Create a unified ModelRun for ordinal logistic regression."""
    return create_model_run(
        task_type="ordinal_classification",
        model_family="statistical",
        model_name="ordinal_logistic_regression_ordered_logit",
        target=result["y_column"],
        features=result["x_columns"],
        split_config={
            "test_size": result["test_size"],
            "random_state": result["random_state"],
            "rows_used": result["rows_used"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "stratified": True,
        },
        preprocessing={
            "category_order": result["category_order"],
            "categorical_encoding": "one-hot drop_first fitted on train columns",
            "missing_values": "complete-case rows for selected variables",
            "link": "logit",
        },
        statistical_summary=result["model_statistics"],
        train_metrics=result["train_metrics"],
        test_metrics=result["test_metrics"],
        coefficient_table=result["coefficient_table"],
        diagnostic_plot_keys=["confusion_matrix", "class_probability_table"],
        formula_latex=result.get("formula_latex"),
        notes=f"Ordinal Logit model fitted with statsmodels. {PROPORTIONAL_ODDS_WARNING}",
    )


def _build_count_model_run(result: dict) -> dict:
    """Create a unified ModelRun for count regression."""
    model_name = (
        "poisson_regression_glm"
        if result["model_type"] == POISSON_MODEL_NAME
        else "negative_binomial_regression"
    )
    notes = f"{result['model_type']} fitted with statsmodels. "
    notes = f"{notes}{result['overdispersion']['recommendation']}"
    if result["zero_warning"].get("warning"):
        notes = f"{notes} {result['zero_warning']['warning']}"

    return create_model_run(
        task_type="count_regression",
        model_family="statistical",
        model_name=model_name,
        target=result["y_column"],
        features=result["x_columns"],
        split_config={
            "test_size": result["test_size"],
            "random_state": result["random_state"],
            "rows_used": result["rows_used"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
        },
        preprocessing={
            "categorical_encoding": "one-hot drop_first fitted on train columns",
            "missing_values": "complete-case rows for selected variables",
            "link": "log",
            "model_type": result["model_type"],
        },
        statistical_summary=result["model_statistics"],
        train_metrics=result["train_metrics"],
        test_metrics=result["test_metrics"],
        coefficient_table=result["coefficient_table"],
        diagnostic_plot_keys=["observed_vs_predicted_counts"],
        formula_latex=result.get("formula_latex"),
        notes=notes,
    )


def _save_statistical_model_artifact(model_run: dict, result: dict) -> None:
    """Save the fitted statsmodels object for later prediction-related features."""
    is_linear = model_run["task_type"] == "regression"
    preprocessing = model_run.get("preprocessing", {})
    save_model_artifact(
        model_run["run_id"],
        {
            "model_name": model_run["model_name"],
            "model_family": model_run["model_family"],
            "task_type": model_run["task_type"],
            "fitted_model": result["model"],
            "fitted_pipeline": None,
            "target": model_run["target"],
            "features": model_run["features"],
            "transformed_feature_names": list(result["model"].params.index),
            "preprocessing_summary": preprocessing,
            "target_encoder": {
                "positive_class": result.get("positive_class"),
                "negative_class": "all other target values",
            }
            if not is_linear
            else None,
            "positive_class": result.get("positive_class"),
            "formula_latex": model_run.get("formula_latex"),
            "coefficient_table": model_run.get("coefficient_table"),
            "influence_table": result.get("influence_table"),
            "top_influential_rows": result.get("top_influential_rows"),
            "influence_warning": INFLUENCE_WARNING if is_linear else None,
            "prediction_supported": True,
            "interval_supported": is_linear,
            "interval_method": "statsmodels OLS prediction interval" if is_linear else None,
            "design_columns": list(result["model"].params.index),
        },
    )


def _save_multinomial_model_artifact(model_run: dict, result: dict) -> None:
    """Save the fitted MNLogit artifact for session inspection."""
    save_model_artifact(
        model_run["run_id"],
        {
            "model_name": model_run["model_name"],
            "model_family": model_run["model_family"],
            "task_type": model_run["task_type"],
            "fitted_model": result["model"],
            "fitted_pipeline": None,
            "target": model_run["target"],
            "features": model_run["features"],
            "transformed_feature_names": result["design_columns"],
            "preprocessing_summary": model_run.get("preprocessing", {}),
            "target_encoder": {
                "reference_class": result["reference_class"],
                "target_classes": result["target_classes"],
            },
            "positive_class": None,
            "formula_latex": model_run.get("formula_latex"),
            "coefficient_table": model_run.get("coefficient_table"),
            "prediction_supported": False,
            "interval_supported": False,
            "interval_method": None,
            "design_columns": result["design_columns"],
        },
    )


def _save_ordinal_model_artifact(model_run: dict, result: dict) -> None:
    """Save a fitted OrderedModel artifact for later prediction."""
    save_model_artifact(
        model_run["run_id"],
        {
            "model_name": model_run["model_name"],
            "model_family": model_run["model_family"],
            "task_type": model_run["task_type"],
            "fitted_model": result["model"],
            "fitted_pipeline": None,
            "target": model_run["target"],
            "features": model_run["features"],
            "transformed_feature_names": result["design_columns"],
            "preprocessing_summary": model_run.get("preprocessing", {}),
            "target_encoder": {
                "category_order": result["category_order"],
            },
            "positive_class": None,
            "formula_latex": model_run.get("formula_latex"),
            "coefficient_table": model_run.get("coefficient_table"),
            "threshold_table": result.get("threshold_table"),
            "prediction_supported": True,
            "interval_supported": False,
            "interval_method": None,
            "design_columns": result["design_columns"],
        },
    )


def _save_count_model_artifact(model_run: dict, result: dict) -> None:
    """Save a fitted count model artifact for later prediction."""
    save_model_artifact(
        model_run["run_id"],
        {
            "model_name": model_run["model_name"],
            "model_family": model_run["model_family"],
            "task_type": model_run["task_type"],
            "fitted_model": result["model"],
            "fitted_pipeline": None,
            "target": model_run["target"],
            "features": model_run["features"],
            "transformed_feature_names": result["design_columns"],
            "preprocessing_summary": model_run.get("preprocessing", {}),
            "target_encoder": None,
            "positive_class": None,
            "formula_latex": model_run.get("formula_latex"),
            "coefficient_table": model_run.get("coefficient_table"),
            "prediction_supported": True,
            "interval_supported": True,
            "interval_method": "delta-method confidence interval for expected count mean",
            "design_columns": result["design_columns"],
        },
    )


def _logistic_display_threshold(result: dict, threshold_mode: str, threshold: float) -> float:
    """Return the threshold currently requested for displaying logistic results."""
    if threshold_mode == "Optimize threshold for F1":
        return find_best_f1_threshold(result["train_actual"], result["train_probabilities"])
    return float(threshold)


def _target_transformation_metadata(y_column: str, transformation_log: list[dict]) -> dict | None:
    """Find target transformation metadata for a selected y column."""
    for entry in reversed(transformation_log):
        if entry.get("transformed_target") == y_column:
            return entry
    return None


def _original_scale_metrics(result: dict, target_metadata: dict) -> dict | None:
    """Calculate test metrics after inverting transformed predictions."""
    if not target_metadata.get("inverse_transform_available"):
        return None

    original_actual = inverse_transform_values(
        target_metadata["method"],
        result["y_test"],
        target_metadata.get("parameters", {}),
    )
    original_predictions = inverse_transform_values(
        target_metadata["method"],
        result["test_predictions"],
        target_metadata.get("parameters", {}),
    )
    return regression_metrics(original_actual, original_predictions)


def _show_formula(formula_latex: dict | None) -> None:
    """Display formula strings in Streamlit when they are available."""
    if not formula_latex:
        return
    st.subheader("Model Formula")
    for key in ["symbolic", "linear_predictor", "estimated", "thresholds", "probability", "mean"]:
        formula = formula_latex.get(key)
        if formula:
            st.latex(formula)
    if formula_latex.get("note"):
            st.caption(formula_latex["note"])


def _render_stat_chart(
    figure,
    title: str,
    chart_type: str,
    variables_used: list[str],
    method: str,
    sample_size: int | None,
    next_step: str,
    missing_handling: str = "Uses the complete-case data used by the fitted model.",
) -> None:
    """Render a statistical model chart with consistent context."""
    render_plotly_chart_card(
        figure,
        title=title,
        chart_type=chart_type,
        source_page="Statistical Models",
        variables_used=variables_used,
        method=method,
        sample_size=sample_size,
        missing_handling=missing_handling,
        next_step=next_step,
    )


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Statistical Models",
    purpose="Fit interpretable statistical models with formulas, coefficient tables, inference output, and diagnostics.",
    when_to_use="Use this page when you need model summaries with p-values, confidence intervals, AIC, BIC, or interpretable coefficients.",
    common_mistake="Do not read coefficients as causal effects unless the study design supports causal interpretation.",
    key_terms=["coefficient", "p_value", "confidence_interval", "aic", "bic"],
    next_step="After fitting models, compare saved runs and inspect diagnostics.",
    next_page="pages/06_model_comparison.py",
    tags=["Inference", "Statsmodels"],
)

working_df = st.session_state.get("working_df")

if working_df is None:
    render_dataset_required_empty_state()
else:
    summary = build_summary_table(working_df)
    linear_tab, logistic_tab, multinomial_tab, ordinal_tab, count_tab, runs_tab = st.tabs(
        [
            "Linear Regression",
            "Binary Logistic Regression",
            "Multinomial Logistic Regression",
            "Ordinal Logistic Regression",
            "Count Regression",
            "Saved Model Runs",
        ]
    )

    with linear_tab:
        continuous_y_options = summary.loc[
            summary["detected_type"] == "continuous_numeric",
            "variable",
        ].tolist()

        if not continuous_y_options:
            st.warning("No continuous numeric outcome variable was detected.")
        else:
            y_column = st.selectbox(
                "Outcome variable (y)",
                continuous_y_options,
                key="linear_y",
                help="Choose the continuous numeric target that linear regression will predict.",
            )
            x_options = [column for column in working_df.columns if column != y_column]
            x_columns = st.multiselect(
                "Predictor variables (x)",
                x_options,
                key="linear_x",
                help="Choose one or more predictors. Categorical predictors are one-hot encoded for the model.",
            )
            test_size = st.slider(
                "Test split ratio",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="linear_test_size",
                help="Fraction of usable rows held out for test metrics. The model is fit on the remaining training rows.",
            )
            random_state = st.number_input(
                "Random state",
                min_value=0,
                value=42,
                step=1,
                key="linear_seed",
                help="Keeps the train/test split reproducible.",
            )
            target_metadata = _target_transformation_metadata(
                y_column,
                st.session_state.get("transformation_log", []),
            )

            compute_original_scale = False
            if target_metadata is not None:
                st.info(
                    f"{y_column} was created from target transformation "
                    f"{target_metadata['method']} of {target_metadata['original_target']}."
                )
                st.caption("Linear regression will fit predictions on the transformed target scale.")
                if target_metadata.get("inverse_transform_available"):
                    compute_original_scale = st.checkbox(
                        "Also compute test metrics on the original target scale",
                        value=True,
                        help="Inverse-transform predictions so RMSE, MAE, and R-squared are also shown on the original target scale.",
                    )
                else:
                    st.warning("Inverse transformation is not available, so original-scale metrics cannot be computed.")

            if st.button(
                "Fit linear regression",
                help="Fit a statsmodels OLS model using the selected target, predictors, and train/test split.",
            ):
                if not x_columns:
                    st.error("Choose at least one predictor variable.")
                else:
                    try:
                        result = run_linear_regression(
                            working_df,
                            y_column=y_column,
                            x_columns=x_columns,
                            test_size=float(test_size),
                            random_state=int(random_state),
                        )
                    except Exception as error:
                        st.error(f"Could not fit the model: {error}")
                    else:
                        original_scale_metrics = (
                            _original_scale_metrics(result, target_metadata)
                            if target_metadata is not None and compute_original_scale
                            else None
                        )
                        result["target_transformation_metadata"] = target_metadata
                        result["original_scale_test_metrics"] = original_scale_metrics
                        model_run = _build_linear_model_run(result, target_metadata, original_scale_metrics)
                        add_model_run_to_session(model_run)
                        _save_statistical_model_artifact(model_run, result)
                        result["formula_latex"] = model_run.get("formula_latex")
                        st.session_state["latest_linear_regression_result"] = result
                        st.success("Linear regression model fitted.")

            result = st.session_state.get("latest_linear_regression_result")
            if result is not None:
                _show_formula(result.get("formula_latex"))
                st.subheader("Coefficient Table")
                st.dataframe(result["coefficient_table"], use_container_width=True)
                st.subheader("Model Statistics")
                st.dataframe(result["model_statistics"], use_container_width=True)
                st.subheader("Train Metrics")
                st.caption("Transformed target scale" if result.get("target_transformation_metadata") else "Target scale")
                st.dataframe(pd.DataFrame([result["train_metrics"]]), use_container_width=True)
                st.subheader("Test Metrics")
                st.caption("Transformed target scale" if result.get("target_transformation_metadata") else "Target scale")
                st.dataframe(pd.DataFrame([result["test_metrics"]]), use_container_width=True)
                if result.get("original_scale_test_metrics") is not None:
                    original_target = result["target_transformation_metadata"]["original_target"]
                    st.subheader("Test Metrics on Original Target Scale")
                    st.caption(f"Original target scale: {original_target}")
                    st.dataframe(pd.DataFrame([result["original_scale_test_metrics"]]), use_container_width=True)
                st.subheader("Diagnostic Plots")
                _render_stat_chart(
                    plot_residuals_vs_fitted(result["fitted_values"], result["residuals"]),
                    title="Residuals vs fitted values",
                    chart_type="residuals_vs_fitted",
                    variables_used=[result["y_column"], *result["x_columns"]],
                    method="OLS residual diagnostic",
                    sample_size=len(result["residuals"]),
                    next_step="If a pattern appears, review transformations, missing predictors, or model form.",
                )
                _render_stat_chart(
                    plot_normal_qq(result["residuals"]),
                    title="Normal Q-Q plot",
                    chart_type="qq_plot",
                    variables_used=[result["y_column"]],
                    method="Residual normality diagnostic",
                    sample_size=len(result["residuals"]),
                    next_step="Check whether tail departures could affect inference.",
                )
                _render_stat_chart(
                    plot_scale_location(result["fitted_values"], result["residuals"]),
                    title="Scale-location plot",
                    chart_type="scale_location",
                    variables_used=[result["y_column"], *result["x_columns"]],
                    method="Residual spread diagnostic",
                    sample_size=len(result["residuals"]),
                    next_step="Look for changing spread before relying on standard errors.",
                )
                _render_stat_chart(
                    plot_residuals_vs_leverage(result["influence"], result["residuals"]),
                    title="Residuals vs leverage",
                    chart_type="leverage",
                    variables_used=[result["y_column"], *result["x_columns"]],
                    method="Leverage and influence diagnostic",
                    sample_size=len(result["residuals"]),
                    next_step="Investigate high-leverage observations before deciding on any action.",
                )
                st.subheader("Influence Diagnostics")
                st.warning(INFLUENCE_WARNING)
                influence_table = result.get("influence_table")
                if influence_table is not None and not influence_table.empty:
                    st.dataframe(influence_table, use_container_width=True)
                    st.write("Top influential rows")
                    st.dataframe(result["top_influential_rows"], use_container_width=True)
                    flagged_indices = influence_table.loc[
                        influence_table["influential_observation"],
                        "row_index",
                    ].tolist()
                    if flagged_indices:
                        st.write("Flagged rows from working data")
                        available_indices = [index for index in flagged_indices if index in working_df.index]
                        st.dataframe(working_df.loc[available_indices], use_container_width=True)
                    _render_stat_chart(
                        plot_influence_residuals_vs_fitted(influence_table),
                        title="Influence: residuals vs fitted",
                        chart_type="residuals_vs_fitted",
                        variables_used=[result["y_column"], *result["x_columns"]],
                        method="OLSInfluence residual diagnostic",
                        sample_size=len(influence_table),
                        next_step="Inspect flagged observations; do not remove them automatically.",
                    )
                    _render_stat_chart(
                        plot_leverage_vs_standardized_residual(influence_table),
                        title="Leverage vs standardized residual",
                        chart_type="leverage",
                        variables_used=[result["y_column"], *result["x_columns"]],
                        method="OLSInfluence leverage diagnostic",
                        sample_size=len(influence_table),
                        next_step="Review rows with both high leverage and large residuals.",
                    )
                    _render_stat_chart(
                        plot_cooks_distance(influence_table, threshold=4 / len(influence_table)),
                        title="Cook's distance",
                        chart_type="cooks_distance",
                        variables_used=[result["y_column"], *result["x_columns"]],
                        method="Cook's distance with 4/n guide line",
                        sample_size=len(influence_table),
                        next_step="Investigate large Cook's distance values with domain context.",
                    )
                    _render_stat_chart(
                        plot_studentized_residuals(influence_table),
                        title="Studentized residuals by observation",
                        chart_type="leverage",
                        variables_used=[result["y_column"]],
                        method="Studentized residual diagnostic",
                        sample_size=len(influence_table),
                        next_step="Review observations beyond the reference thresholds.",
                    )
                else:
                    st.info("No influence diagnostics are available for this fit.")

    with logistic_tab:
        binary_y_options = summary.loc[
            summary["detected_type"] == "binary",
            "variable",
        ].tolist()

        if not binary_y_options:
            st.warning("No binary target variable was detected.")
        else:
            y_column = st.selectbox(
                "Binary outcome variable (y)",
                binary_y_options,
                key="logistic_y",
                help="Choose a target with exactly two non-missing classes.",
            )
            target_classes = working_df[y_column].dropna().unique().tolist()
            positive_class = st.selectbox(
                "Positive class",
                target_classes,
                key="logistic_positive",
                help="Choose the class treated as the event of interest. Metrics like recall and precision focus on this class.",
            )
            x_options = [column for column in working_df.columns if column != y_column]
            x_columns = st.multiselect(
                "Predictor variables (x)",
                x_options,
                key="logistic_x",
                help="Choose predictors for the logistic regression model. Categorical predictors are one-hot encoded.",
            )
            test_size = st.slider(
                "Test split ratio",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="logistic_test_size",
                help="Fraction of usable rows held out for test metrics.",
            )
            random_state = st.number_input(
                "Random state",
                min_value=0,
                value=42,
                step=1,
                key="logistic_seed",
                help="Keeps the train/test split reproducible.",
            )
            threshold_mode = st.radio(
                "Threshold mode",
                ["Custom threshold", "Optimize threshold for F1"],
                horizontal=True,
                help="Choose whether to use your own probability cutoff or find a cutoff that maximizes F1 on the training data.",
            )
            threshold = st.slider(
                "Classification threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.5,
                step=0.01,
                disabled=threshold_mode == "Optimize threshold for F1",
                help="Predicted probabilities at or above this value are classified as the positive class.",
            )

            if st.button(
                "Fit binary logistic regression",
                help="Fit a statsmodels Logit model and save the run for comparison and prediction.",
            ):
                if not x_columns:
                    st.error("Choose at least one predictor variable.")
                else:
                    try:
                        result = run_logistic_regression(
                            working_df,
                            y_column=y_column,
                            positive_class=positive_class,
                            x_columns=x_columns,
                            test_size=float(test_size),
                            random_state=int(random_state),
                            threshold=float(threshold),
                            optimize_threshold=threshold_mode == "Optimize threshold for F1",
                        )
                    except Exception as error:
                        st.error(f"Could not fit the model: {error}")
                    else:
                        model_run = _build_logistic_model_run(result)
                        add_model_run_to_session(model_run)
                        _save_statistical_model_artifact(model_run, result)
                        result["formula_latex"] = model_run.get("formula_latex")
                        st.session_state["latest_logistic_regression_result"] = result
                        st.success("Binary logistic regression model fitted.")

            result = st.session_state.get("latest_logistic_regression_result")
            if result is not None:
                for warning_message in result.get("warnings", []):
                    st.warning(warning_message)
                _show_formula(result.get("formula_latex"))
                display_threshold = _logistic_display_threshold(result, threshold_mode, float(threshold))
                train_metrics = classification_metrics(
                    result["train_actual"],
                    result["train_probabilities"],
                    display_threshold,
                )
                test_metrics = classification_metrics(
                    result["test_actual"],
                    result["test_probabilities"],
                    display_threshold,
                )
                test_confusion_matrix = confusion_matrix_table(
                    result["test_actual"],
                    result["test_probabilities"],
                    display_threshold,
                )

                st.caption(
                    f"Positive class: {result['positive_class']} | "
                    f"Displayed threshold: {display_threshold:.3f}"
                )
                st.subheader("Coefficient Table")
                st.dataframe(result["coefficient_table"], use_container_width=True)
                st.subheader("Model Statistics")
                st.dataframe(result["model_statistics"], use_container_width=True)
                st.subheader("Train Metrics")
                st.dataframe(pd.DataFrame([train_metrics]), use_container_width=True)
                st.subheader("Test Metrics")
                st.dataframe(pd.DataFrame([test_metrics]), use_container_width=True)
                st.subheader("Confusion Matrix")
                st.dataframe(test_confusion_matrix, use_container_width=True)
                _render_stat_chart(
                    plot_confusion_matrix(test_confusion_matrix),
                    title="Confusion matrix",
                    chart_type="confusion_matrix",
                    variables_used=[result["y_column"]],
                    method=f"Binary classification at threshold {display_threshold:.3f}",
                    sample_size=int(test_confusion_matrix.to_numpy().sum()),
                    next_step="Review precision, recall, and threshold choice together.",
                )
                st.subheader("Diagnostic Plots")
                _render_stat_chart(
                    plot_roc_curve(result["test_roc_curve"]),
                    title="ROC curve",
                    chart_type="roc_curve",
                    variables_used=[result["y_column"]],
                    method="ROC curve on test set",
                    sample_size=len(result["test_actual"]),
                    next_step="Use PR curve and threshold metrics if classes are imbalanced.",
                )
                _render_stat_chart(
                    plot_precision_recall_curve(result["test_pr_curve"]),
                    title="Precision-recall curve",
                    chart_type="pr_curve",
                    variables_used=[result["y_column"]],
                    method="Precision-recall curve on test set",
                    sample_size=len(result["test_actual"]),
                    next_step="Choose a threshold based on false-positive and false-negative costs.",
                )
                _render_stat_chart(
                    plot_probability_distribution(result["test_probabilities"], result["test_actual"]),
                    title="Predicted probability distribution",
                    chart_type="probability_distribution",
                    variables_used=[result["y_column"]],
                    method="Predicted probabilities by actual class",
                    sample_size=len(result["test_actual"]),
                    next_step="Compare probability separation with the confusion matrix.",
                )

    with multinomial_tab:
        multiclass_y_options = summary.loc[
            summary["detected_type"].isin(["nominal_categorical", "ordinal_categorical_candidate"]),
            "variable",
        ].tolist()
        multiclass_y_options = [
            column
            for column in multiclass_y_options
            if working_df[column].dropna().nunique() > 2
        ]

        if not multiclass_y_options:
            st.warning("No target variable with three or more classes was detected.")
        else:
            y_column = st.selectbox(
                "Multiclass outcome variable (y)",
                multiclass_y_options,
                key="multinomial_y",
                help="Choose an unordered target with three or more classes.",
            )
            target_classes = sorted(working_df[y_column].dropna().unique().tolist(), key=lambda value: str(value))
            reference_class = st.selectbox(
                "Reference class",
                target_classes,
                key="multinomial_reference",
                help="Class used as the comparison baseline for class-specific logits.",
            )
            st.info("Use this only for unordered target classes. Ordered categories should wait for ordinal regression.")
            confirm_unordered = st.checkbox(
                "I confirm this target should be treated as unordered multiclass",
                value=False,
                key="multinomial_confirm_unordered",
                help="Use this only when the target classes do not have a meaningful order.",
            )
            x_options = [column for column in working_df.columns if column != y_column]
            x_columns = st.multiselect(
                "Predictor variables (x)",
                x_options,
                key="multinomial_x",
                help="Choose predictors for the multinomial logistic model.",
            )
            test_size = st.slider(
                "Test split ratio",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="multinomial_test_size",
                help="Fraction of usable rows held out for test metrics.",
            )
            random_state = st.number_input(
                "Random state",
                min_value=0,
                value=42,
                step=1,
                key="multinomial_seed",
                help="Keeps the train/test split reproducible.",
            )

            if st.button(
                "Fit multinomial logistic regression",
                help="Fit a statsmodels multinomial logistic model and save the run.",
            ):
                if not confirm_unordered:
                    st.error("Confirm that the target should be treated as unordered multiclass.")
                elif not x_columns:
                    st.error("Choose at least one predictor variable.")
                else:
                    try:
                        result = run_multinomial_logistic_regression(
                            working_df,
                            y_column=y_column,
                            reference_class=reference_class,
                            x_columns=x_columns,
                            test_size=float(test_size),
                            random_state=int(random_state),
                        )
                    except Exception as error:
                        st.error(f"Could not fit the model: {error}")
                    else:
                        model_run = _build_multinomial_model_run(result)
                        add_model_run_to_session(model_run)
                        _save_multinomial_model_artifact(model_run, result)
                        st.session_state["latest_multinomial_logistic_result"] = result
                        st.success("Multinomial logistic regression model fitted.")

            result = st.session_state.get("latest_multinomial_logistic_result")
            if result is not None:
                _show_formula(result.get("formula_latex"))
                st.caption(f"Reference class: {result['reference_class']}")
                st.subheader("Coefficient Table")
                st.dataframe(result["coefficient_table"], use_container_width=True)
                st.subheader("Model Statistics")
                st.dataframe(result["model_statistics"], use_container_width=True)
                st.subheader("Train Metrics")
                st.dataframe(pd.DataFrame([result["train_metrics"]]), use_container_width=True)
                st.subheader("Test Metrics")
                st.dataframe(pd.DataFrame([result["test_metrics"]]), use_container_width=True)
                st.subheader("Confusion Matrix")
                st.dataframe(result["test_confusion_matrix"], use_container_width=True)
                _render_stat_chart(
                    plot_confusion_matrix(result["test_confusion_matrix"]),
                    title="Multiclass confusion matrix",
                    chart_type="confusion_matrix",
                    variables_used=[result["y_column"]],
                    method="Multinomial logistic regression test confusion matrix",
                    sample_size=int(result["test_confusion_matrix"].to_numpy().sum()),
                    next_step="Review class-wise metrics to see which classes are confused.",
                )
                st.subheader("Class-wise Metrics")
                st.dataframe(result["test_class_metrics"], use_container_width=True)

    with ordinal_tab:
        ordinal_y_options = summary.loc[
            summary["detected_type"].isin(["ordinal_categorical_candidate", "nominal_categorical"]),
            "variable",
        ].tolist()
        ordinal_y_options = [
            column
            for column in ordinal_y_options
            if working_df[column].dropna().nunique() >= 3
        ]

        if not ordinal_y_options:
            st.warning("No target variable with three or more ordered categories was detected.")
        else:
            y_column = st.selectbox(
                "Ordered outcome variable (y)",
                ordinal_y_options,
                key="ordinal_y",
                help="Choose a target whose categories have a meaningful order, such as low < medium < high.",
            )
            observed_categories = sorted(working_df[y_column].dropna().unique().tolist(), key=lambda value: str(value))
            st.info("Confirm the order from lowest to highest. Reorder the categories if needed before fitting.")
            category_order = st.multiselect(
                "Category order from lowest to highest",
                observed_categories,
                default=observed_categories,
                key="ordinal_category_order",
                help="Set the target category order from lowest to highest. Include every observed category exactly once.",
            )
            order_is_valid = len(category_order) == len(observed_categories) and set(category_order) == set(observed_categories)
            if not order_is_valid:
                st.warning("The order must include every observed category exactly once.")
            st.warning(PROPORTIONAL_ODDS_WARNING)

            x_options = [column for column in working_df.columns if column != y_column]
            x_columns = st.multiselect(
                "Predictor variables (x)",
                x_options,
                key="ordinal_x",
                help="Choose predictors for the ordinal logistic model.",
            )
            test_size = st.slider(
                "Test split ratio",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="ordinal_test_size",
                help="Fraction of usable rows held out for test metrics.",
            )
            random_state = st.number_input(
                "Random state",
                min_value=0,
                value=42,
                step=1,
                key="ordinal_seed",
                help="Keeps the train/test split reproducible.",
            )

            if st.button(
                "Fit ordinal logistic regression",
                help="Fit a proportional-odds ordinal model using the confirmed category order.",
            ):
                if not order_is_valid:
                    st.error("Define a complete category order before fitting.")
                elif not x_columns:
                    st.error("Choose at least one predictor variable.")
                else:
                    try:
                        result = run_ordinal_logistic_regression(
                            working_df,
                            y_column=y_column,
                            category_order=category_order,
                            x_columns=x_columns,
                            test_size=float(test_size),
                            random_state=int(random_state),
                        )
                    except Exception as error:
                        st.error(f"Could not fit the model: {error}")
                    else:
                        model_run = _build_ordinal_model_run(result)
                        add_model_run_to_session(model_run)
                        _save_ordinal_model_artifact(model_run, result)
                        st.session_state["latest_ordinal_logistic_result"] = result
                        st.success("Ordinal logistic regression model fitted.")

            result = st.session_state.get("latest_ordinal_logistic_result")
            if result is not None:
                _show_formula(result.get("formula_latex"))
                st.caption("Category order: " + " < ".join(str(value) for value in result["category_order"]))
                st.warning(result["diagnostic_warning"])
                st.subheader("Coefficient Table")
                st.dataframe(result["coefficient_table"], use_container_width=True)
                st.subheader("Threshold / Cutpoint Parameters")
                st.dataframe(result["threshold_table"], use_container_width=True)
                st.subheader("Model Statistics")
                st.dataframe(result["model_statistics"], use_container_width=True)
                st.subheader("Train Metrics")
                st.dataframe(pd.DataFrame([result["train_metrics"]]), use_container_width=True)
                st.subheader("Test Metrics")
                st.dataframe(pd.DataFrame([result["test_metrics"]]), use_container_width=True)
                st.subheader("Confusion Matrix")
                st.dataframe(result["test_confusion_matrix"], use_container_width=True)
                _render_stat_chart(
                    plot_confusion_matrix(result["test_confusion_matrix"]),
                    title="Ordinal confusion matrix",
                    chart_type="confusion_matrix",
                    variables_used=[result["y_column"]],
                    method="Ordinal logistic regression test confusion matrix",
                    sample_size=int(result["test_confusion_matrix"].to_numpy().sum()),
                    next_step="Check whether mistakes are mostly near neighboring ordered classes.",
                )
                st.subheader("Predicted Probability Table")
                st.dataframe(result["test_probability_table"], use_container_width=True)

    with count_tab:
        numeric_y_options = summary.loc[
            summary["detected_type"].isin(["discrete_numeric", "continuous_numeric"]),
            "variable",
        ].tolist()

        if not numeric_y_options:
            st.warning("No numeric target variables were detected for count regression.")
        else:
            y_column = st.selectbox(
                "Count outcome variable (y)",
                numeric_y_options,
                key="count_y",
                help="Choose a nonnegative integer-like target, such as event counts.",
            )
            model_type = st.radio(
                "Count model",
                COUNT_MODEL_OPTIONS,
                horizontal=True,
                help="Poisson is the basic count model. Negative Binomial is often better when counts are overdispersed.",
            )
            x_options = [column for column in working_df.columns if column != y_column]
            x_columns = st.multiselect(
                "Predictor variables (x)",
                x_options,
                key="count_x",
                help="Choose predictors for the count regression model.",
            )
            test_size = st.slider(
                "Test split ratio",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="count_test_size",
                help="Fraction of usable rows held out for test metrics.",
            )
            random_state = st.number_input(
                "Random state",
                min_value=0,
                value=42,
                step=1,
                key="count_seed",
                help="Keeps the train/test split reproducible.",
            )
            st.caption(
                "Count regression requires a nonnegative integer-like target. "
                "The prediction output is the expected count mean."
            )

            if st.button(
                "Fit count regression",
                help="Fit the selected statistical count model and save the run.",
            ):
                if not x_columns:
                    st.error("Choose at least one predictor variable.")
                else:
                    try:
                        result = run_count_regression(
                            working_df,
                            y_column=y_column,
                            x_columns=x_columns,
                            model_type=model_type,
                            test_size=float(test_size),
                            random_state=int(random_state),
                        )
                    except Exception as error:
                        st.error(f"Could not fit the model: {error}")
                    else:
                        model_run = _build_count_model_run(result)
                        add_model_run_to_session(model_run)
                        _save_count_model_artifact(model_run, result)
                        st.session_state["latest_count_regression_result"] = result
                        st.success(f"{model_type} fitted.")

            result = st.session_state.get("latest_count_regression_result")
            if result is not None:
                _show_formula(result.get("formula_latex"))
                st.subheader("Coefficient Table")
                st.dataframe(result["coefficient_table"], use_container_width=True)
                st.subheader("Model Statistics")
                st.dataframe(result["model_statistics"], use_container_width=True)
                st.subheader("Overdispersion Check")
                st.json(result["overdispersion"])
                if result["overdispersion"].get("overdispersion_detected") and result["model_type"] != NEGATIVE_BINOMIAL_MODEL_NAME:
                    st.warning(result["overdispersion"]["recommendation"])
                if result["zero_warning"].get("warning"):
                    st.warning(result["zero_warning"]["warning"])
                st.subheader("Train Metrics")
                st.dataframe(pd.DataFrame([result["train_metrics"]]), use_container_width=True)
                st.subheader("Test Metrics")
                st.dataframe(pd.DataFrame([result["test_metrics"]]), use_container_width=True)
                st.subheader("Observed vs Predicted Counts")
                _render_stat_chart(
                    plot_observed_vs_predicted_counts(result["test_actual"], result["test_predictions"]),
                    title="Observed vs predicted counts",
                    chart_type="observed_vs_predicted",
                    variables_used=[result["y_column"], *result["x_columns"]],
                    method=f"{result['model_type']} expected count predictions",
                    sample_size=len(result["test_actual"]),
                    next_step="Use overdispersion and zero warnings before reporting count-model results.",
                )

    with runs_tab:
        model_runs = get_model_runs()
        if model_runs:
            st.dataframe(pd.DataFrame(model_runs), use_container_width=True)
        else:
            st.info("No model runs saved yet.")
