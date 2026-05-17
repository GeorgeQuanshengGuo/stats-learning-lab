"""Streamlit dashboard for saved model diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import learning_curve

from src.modeling.diagnostics.multicollinearity import (
    compute_condition_number,
    compute_predictor_correlation_table,
    compute_vif_table,
)
from src.modeling.diagnostics.overfitting import build_overfitting_diagnostics
from src.modeling.diagnostics.influence import INFLUENCE_WARNING, top_influential_rows
from src.visualization.diagnostic_plots import (
    plot_learning_curve_table,
    plot_overfitting_metric_gaps,
    plot_vif_bar,
)
from src.visualization.influence_plots import (
    plot_cooks_distance,
    plot_influence_residuals_vs_fitted,
    plot_leverage_vs_standardized_residual,
    plot_studentized_residuals,
)
from src.ui.chart_cards import render_plotly_chart_card
from src.ui.components import render_empty_state
from src.ui.page_templates import render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles


def _run_label(model_run: dict[str, Any]) -> str:
    """Return a compact label for selecting saved model runs."""
    return (
        f"{model_run.get('model_name')} | {model_run.get('target')} | "
        f"{model_run.get('task_type')} | {model_run.get('run_id', '')[:8]}"
    )


def _metadata(model_run: dict[str, Any]) -> dict[str, Any]:
    """Return display-safe metadata for one model run."""
    return {
        "model_name": model_run.get("model_name"),
        "model_family": model_run.get("model_family"),
        "task_type": model_run.get("task_type"),
        "target": model_run.get("target"),
        "features": model_run.get("features"),
        "train/test split": model_run.get("split_config"),
        "preprocessing": model_run.get("preprocessing"),
    }


def _learning_curve_scoring_options(task_type: str) -> list[str]:
    """Return scoring metrics suitable for sklearn learning_curve."""
    if task_type == "regression":
        return ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
    if task_type == "binary_classification":
        return ["accuracy", "f1", "roc_auc"]
    return ["accuracy"]


def _prepare_learning_curve_data(
    working_df: pd.DataFrame,
    model_run: dict[str, Any],
    artifact: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw X and target y for learning_curve."""
    target = model_run.get("target")
    features = list(artifact.get("features") or model_run.get("features") or [])
    if target not in working_df.columns:
        raise ValueError("The model target is not available in the current working dataset.")
    missing_features = [feature for feature in features if feature not in working_df.columns]
    if missing_features:
        raise ValueError(f"Model features are not available in the current working dataset: {', '.join(missing_features)}")

    data = working_df[[target, *features]].copy(deep=True).dropna(subset=[target])
    x_data = data[features]
    y_data = data[target]
    if model_run.get("task_type") == "binary_classification":
        positive_class = artifact.get("positive_class") or (model_run.get("preprocessing") or {}).get("positive_class")
        y_data = (y_data == positive_class).astype(int)
    else:
        y_data = pd.to_numeric(y_data, errors="coerce")
        keep_rows = y_data.notna()
        x_data = x_data.loc[keep_rows]
        y_data = y_data.loc[keep_rows]
    if len(x_data) < 5:
        raise ValueError("Learning curve needs at least 5 usable rows.")
    return x_data, y_data


def _build_learning_curve_table(
    pipeline,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    scoring: str,
    cv_folds: int,
) -> pd.DataFrame:
    """Compute a learning curve table by refitting the full sklearn Pipeline."""
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=pipeline,
        X=x_data,
        y=y_data,
        cv=cv_folds,
        scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 5),
        error_score=np.nan,
    )
    if scoring.startswith("neg_"):
        train_scores = -train_scores
        validation_scores = -validation_scores
    return pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_score_mean": np.nanmean(train_scores, axis=1),
            "train_score_std": np.nanstd(train_scores, axis=1),
            "validation_score_mean": np.nanmean(validation_scores, axis=1),
            "validation_score_std": np.nanstd(validation_scores, axis=1),
            "scoring": scoring,
        }
    )


def _artifact_table(artifact: dict[str, Any], key: str) -> pd.DataFrame:
    """Return a DataFrame table from artifact metadata."""
    value = artifact.get(key)
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _render_diagnostic_chart(
    figure,
    title: str,
    chart_type: str,
    variables_used: list[str],
    method: str,
    sample_size: int | None,
    next_step: str,
) -> None:
    """Render a diagnostics chart with consistent context."""
    render_plotly_chart_card(
        figure,
        title=title,
        chart_type=chart_type,
        source_page="Model Diagnostics",
        variables_used=variables_used,
        method=method,
        sample_size=sample_size,
        missing_handling="Uses saved model results or the current working dataset as noted by the table.",
        next_step=next_step,
    )


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Model Diagnostics",
    purpose="Inspect saved model runs for overfitting, multicollinearity, residual patterns, and influence concerns.",
    when_to_use="Use this page after fitting a model and before trusting or reporting its results.",
    common_mistake="Do not treat VIF as an overfitting diagnostic; it is about predictor overlap and coefficient stability.",
    key_terms=["residual", "vif", "cooks_distance"],
    next_step="If diagnostics look acceptable, compare model runs or export a report.",
    next_page="pages/06_model_comparison.py",
    tags=["Diagnostics", "Interpretation"],
)

model_runs = st.session_state.get("model_runs", [])
model_artifacts = st.session_state.get("model_artifacts", {})
working_df = st.session_state.get("working_df")

if not model_runs:
    render_empty_state(
        "No saved model runs yet",
        "Fit a model first so diagnostics can inspect saved results.",
        action_label="Go to Statistical Models",
        action_page="pages/05_statistical_models.py",
        icon="Diagnostics:",
    )
else:
    selected_label = st.selectbox(
        "Saved model run",
        [_run_label(run) for run in model_runs],
        help="Choose a saved model run to inspect for overfitting, multicollinearity, and influence diagnostics.",
    )
    selected_run = model_runs[[_run_label(run) for run in model_runs].index(selected_label)]
    selected_artifact = model_artifacts.get(selected_run.get("run_id"), {})

    st.subheader("Model Metadata")
    st.json(_metadata(selected_run))

    st.subheader("Diagnostic Cards")
    overfitting_table = build_overfitting_diagnostics(selected_run)
    warning_count = int((overfitting_table["risk_level"].isin(["warning", "strong_warning"])).sum()) if not overfitting_table.empty else 0
    cv_rows = overfitting_table.loc[overfitting_table["metric"].astype(str).str.startswith("cv_")] if not overfitting_table.empty else pd.DataFrame()

    card_1, card_2, card_3, card_4 = st.columns(4)
    card_1.metric("Overfitting warnings", warning_count)
    card_2.metric("CV metrics found", len(cv_rows))
    card_3.metric("Artifact available", "Yes" if selected_artifact else "No")
    influence_table = _artifact_table(selected_artifact, "influence_table")
    card_4.metric(
        "Residual diagnostics",
        "Available" if selected_run.get("diagnostic_plot_keys") or not influence_table.empty else "Not saved",
    )

    st.subheader("Overfitting Risk")
    if overfitting_table.empty:
        st.info("No overfitting diagnostic metrics are available for this model run.")
    else:
        st.dataframe(overfitting_table, use_container_width=True)
        _render_diagnostic_chart(
            plot_overfitting_metric_gaps(overfitting_table),
            title="Overfitting Diagnostic Gaps",
            chart_type="overfitting_diagnostics",
            variables_used=list(selected_run.get("features") or []),
            method="Saved train/test metrics and heuristic diagnostic rules",
            sample_size=None,
            next_step="If gaps are large, inspect cross-validation, model complexity, and feature choices.",
        )

    st.subheader("Cross-validation Stability")
    if cv_rows.empty:
        st.info("No cross-validation mean/std metrics were saved for this model run.")
    else:
        st.dataframe(cv_rows, use_container_width=True)

    st.subheader("Multicollinearity Risk")
    st.caption(
        "VIF is a multicollinearity diagnostic, not an overfitting diagnostic. "
        "The common VIF > 5 and VIF > 10 cutoffs are heuristic thresholds."
    )
    features = list(selected_run.get("features") or [])
    if working_df is None:
        st.info("Load a working dataset to compute multicollinearity diagnostics.")
    elif not features:
        st.info("This model run does not list feature columns.")
    else:
        try:
            vif_table = compute_vif_table(working_df, features)
            predictor_correlation_table = compute_predictor_correlation_table(working_df, features)
            condition_number = compute_condition_number(working_df, features)
        except Exception as error:
            st.error(f"Could not compute multicollinearity diagnostics: {error}")
        else:
            st.write("VIF table")
            st.dataframe(vif_table, use_container_width=True)
            _render_diagnostic_chart(
                plot_vif_bar(vif_table),
                title="Variance inflation factors",
                chart_type="vif_bar",
                variables_used=features,
                method="VIF on numeric or encoded design matrix columns",
                sample_size=len(vif_table),
                next_step="Review high-VIF predictors before interpreting coefficients.",
            )
            st.write("Predictor correlation table")
            st.dataframe(predictor_correlation_table, use_container_width=True)
            st.write("Condition number")
            st.json(condition_number)

    st.subheader("Residual / Influence Diagnostics")
    diagnostic_keys = selected_run.get("diagnostic_plot_keys") or []
    if diagnostic_keys:
        st.write("Saved diagnostic plot metadata")
        st.json(diagnostic_keys)
    if not influence_table.empty:
        st.warning(INFLUENCE_WARNING)
        st.write("Influence table")
        st.dataframe(influence_table, use_container_width=True)
        st.write("Top influential rows")
        st.dataframe(top_influential_rows(influence_table), use_container_width=True)
        if working_df is not None and {"row_index", "influential_observation"}.issubset(influence_table.columns):
            flagged_indices = influence_table.loc[
                influence_table["influential_observation"].astype(bool),
                "row_index",
            ].tolist()
            available_indices = [index for index in flagged_indices if index in working_df.index]
            if available_indices:
                st.write("Flagged rows from working data")
                st.dataframe(working_df.loc[available_indices], use_container_width=True)
        _render_diagnostic_chart(
            plot_influence_residuals_vs_fitted(influence_table),
            title="Influence: residuals vs fitted",
            chart_type="residuals_vs_fitted",
            variables_used=list(selected_run.get("features") or []),
            method="OLSInfluence residual diagnostic",
            sample_size=len(influence_table),
            next_step="Inspect flagged observations before deciding whether action is justified.",
        )
        _render_diagnostic_chart(
            plot_leverage_vs_standardized_residual(influence_table),
            title="Leverage vs standardized residual",
            chart_type="leverage",
            variables_used=list(selected_run.get("features") or []),
            method="Leverage and standardized residual diagnostic",
            sample_size=len(influence_table),
            next_step="Review rows with both high leverage and large residuals.",
        )
        _render_diagnostic_chart(
            plot_cooks_distance(influence_table, threshold=4 / len(influence_table)),
            title="Cook's distance",
            chart_type="cooks_distance",
            variables_used=list(selected_run.get("features") or []),
            method="Cook's distance with 4/n guide line",
            sample_size=len(influence_table),
            next_step="Investigate high Cook's distance rows with domain knowledge.",
        )
        _render_diagnostic_chart(
            plot_studentized_residuals(influence_table),
            title="Studentized residuals by observation",
            chart_type="leverage",
            variables_used=[selected_run.get("target")],
            method="Studentized residual index plot",
            sample_size=len(influence_table),
            next_step="Review observations beyond the threshold lines.",
        )
    elif not diagnostic_keys:
        st.info("No residual or influence diagnostic plot metadata was saved for this run.")

    st.subheader("Learning Curve")
    st.info(
        "Learning curves refit the full sklearn Pipeline several times. This can take time, "
        "and preprocessing stays inside the Pipeline during each refit."
    )
    if selected_run.get("model_family") != "machine_learning" or not selected_artifact.get("fitted_pipeline"):
        st.info("Learning curve is available only for saved sklearn machine learning artifacts.")
    elif working_df is None:
        st.info("Load a working dataset before computing a learning curve.")
    else:
        scoring = st.selectbox(
            "Scoring metric",
            _learning_curve_scoring_options(selected_run.get("task_type")),
            help="Metric used for the learning curve. The model is refit several times using the full Pipeline.",
        )
        cv_folds = st.slider(
            "CV folds",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of cross-validation folds used when refitting the learning curve.",
        )
        if st.button(
            "Compute learning curve",
            help="Refit the saved sklearn Pipeline across training sizes. This may take time.",
        ):
            try:
                x_data, y_data = _prepare_learning_curve_data(working_df, selected_run, selected_artifact)
                curve_table = _build_learning_curve_table(
                    selected_artifact["fitted_pipeline"],
                    x_data,
                    y_data,
                    scoring=scoring,
                    cv_folds=int(cv_folds),
                )
            except Exception as error:
                st.error(f"Could not compute learning curve: {error}")
            else:
                st.dataframe(curve_table, use_container_width=True)
                _render_diagnostic_chart(
                    plot_learning_curve_table(curve_table),
                    title="Learning curve",
                    chart_type="learning_curve",
                    variables_used=features,
                    method=f"sklearn learning_curve with {scoring}",
                    sample_size=int(curve_table["train_size"].max()) if "train_size" in curve_table else None,
                    next_step="Use the train-validation gap to decide whether more data or simpler models may help.",
                )
