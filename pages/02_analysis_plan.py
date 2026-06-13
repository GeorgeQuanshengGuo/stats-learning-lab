"""Streamlit page for the learning-focused analysis plan."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.core.analysis_plan import (
    ANALYSIS_GOALS,
    TARGET_TYPES,
    analysis_plan_is_recorded,
    create_empty_analysis_plan,
    normalize_analysis_plan,
    update_analysis_plan,
)
from src.core.rigor import build_data_readiness, build_rigor_checklist
from src.ui.components import render_empty_state, render_section_header
from src.ui.page_templates import render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles


def _option_index(options: list[str], value: str) -> int:
    """Return a safe selectbox index."""
    return options.index(value) if value in options else 0


def _column_options(df: pd.DataFrame | None) -> list[str]:
    """Return column names for widget options."""
    if df is None:
        return []
    return list(df.columns)


def _render_checklist(checklist: list[dict]) -> None:
    """Render checklist records as a readable table."""
    status_order = {
        "needs_attention": 0,
        "not_started": 1,
        "completed": 2,
        "not_applicable": 3,
    }
    table = pd.DataFrame(checklist)
    if table.empty:
        st.info("No checklist items are available yet.")
        return
    table["_order"] = table["status"].map(status_order).fillna(9)
    table = table.sort_values(["_order", "group", "item"]).drop(columns=["_order"])
    st.dataframe(table, use_container_width=True)


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Analysis Plan",
    purpose="Record the analysis question, target, intended model path, and interpretation boundaries before reading results.",
    when_to_use="Use this page after upload and before treating EDA patterns or model outputs as conclusions.",
    common_mistake="Do not repeatedly try many models and then interpret the final p-values as if the model had been planned from the start.",
    key_terms=["p_value", "train_test_split", "confidence_interval", "overfitting"],
    next_step="After saving the plan, move to EDA and compare what the data shows against the planned question.",
    next_page="pages/02_eda.py",
    tags=["Planning", "Rigor"],
)

working_df = st.session_state.get("working_df")
current_plan = normalize_analysis_plan(st.session_state.get("analysis_plan"))
column_options = _column_options(working_df)

if working_df is None:
    render_empty_state(
        "No dataset loaded yet",
        "You can still draft the research question, but column selectors will appear after upload.",
        action_label="Go to Upload Data",
        action_page="pages/01_upload_data.py",
        icon="Plan:",
    )

with st.form("analysis_plan_form"):
    render_section_header(
        "Question and Goal",
        "Write down the intended analysis before interpreting model output.",
    )
    analysis_title = st.text_input(
        "Analysis title",
        value=current_plan["analysis_title"],
        help="A short name for this analysis session.",
    )
    research_question = st.text_area(
        "Research question",
        value=current_plan["research_question"],
        help="State the question in plain language before looking for statistical significance.",
    )
    analysis_goal = st.selectbox(
        "Analysis goal",
        ANALYSIS_GOALS,
        index=_option_index(ANALYSIS_GOALS, current_plan["analysis_goal"]),
        help="Choose whether this is exploration, prediction, inference, or a learning demo.",
    )
    unit_of_analysis = st.text_input(
        "Unit of analysis",
        value=current_plan["unit_of_analysis"],
        help="Describe what one row represents, such as one person, account, country-year, or transaction.",
    )

    render_section_header(
        "Variables",
        "Choose planned variables and columns that should be treated carefully.",
    )
    if column_options:
        target_options = [""] + column_options
        target_variable = st.selectbox(
            "Target variable",
            target_options,
            index=_option_index(target_options, current_plan["target_variable"]),
            help="The outcome you plan to explain or predict.",
        )
        candidate_features = st.multiselect(
            "Candidate features",
            column_options,
            default=[column for column in current_plan["candidate_features"] if column in column_options],
            help="Predictors you may use. This is a plan, not an automatic model fit.",
        )
        excluded_columns = st.multiselect(
            "Excluded columns",
            column_options,
            default=[column for column in current_plan["excluded_columns"] if column in column_options],
            help="Columns that should not be used for modeling, such as IDs, notes, or leakage-prone fields.",
        )
        known_id_columns = st.multiselect(
            "Known ID columns",
            column_options,
            default=[column for column in current_plan["known_id_columns"] if column in column_options],
            help="ID-like fields are usually identifiers rather than meaningful predictors.",
        )
        known_time_columns = st.multiselect(
            "Known time columns",
            column_options,
            default=[column for column in current_plan["known_time_columns"] if column in column_options],
            help="Time columns may require time-ordered validation rather than random train/test splitting.",
        )
        grouping_or_cluster_columns = st.multiselect(
            "Grouping or cluster columns",
            column_options,
            default=[column for column in current_plan["grouping_or_cluster_columns"] if column in column_options],
            help="Examples include school, hospital, region, user, batch, or repeated-measure groups.",
        )
    else:
        target_variable = st.text_input(
            "Target variable",
            value=current_plan["target_variable"],
            help="You can type a planned target now and choose from columns after upload.",
        )
        candidate_features = current_plan["candidate_features"]
        excluded_columns = current_plan["excluded_columns"]
        known_id_columns = current_plan["known_id_columns"]
        known_time_columns = current_plan["known_time_columns"]
        grouping_or_cluster_columns = current_plan["grouping_or_cluster_columns"]

    expected_target_type = st.selectbox(
        "Expected target type",
        TARGET_TYPES,
        index=_option_index(TARGET_TYPES, current_plan["expected_target_type"]),
        help="This helps the app explain which model families are appropriate.",
    )
    planned_model_family = st.text_input(
        "Planned model family",
        value=current_plan["planned_model_family"],
        help="Examples: OLS, logistic regression, ML regression baseline, count model.",
    )
    primary_metric_or_statistic = st.text_input(
        "Primary metric or statistic",
        value=current_plan["primary_metric_or_statistic"],
        help="Examples: test RMSE, ROC AUC, coefficient sign, confidence interval, AIC/BIC.",
    )

    render_section_header(
        "Validation and Interpretation",
        "Record how you plan to handle uncertainty, missingness, and interpretation boundaries.",
    )
    train_test_strategy = st.text_input(
        "Train/test or validation strategy",
        value=current_plan["train_test_strategy"],
        help="Examples: random 80/20 split, 5-fold CV, time-ordered split, no split for descriptive EDA.",
    )
    missing_data_strategy = st.text_area(
        "Missing data strategy",
        value=current_plan["missing_data_strategy"],
        help="Describe whether you will inspect, drop, impute, or leave missing values untouched.",
    )
    transformation_plan = st.text_area(
        "Transformation plan",
        value=current_plan["transformation_plan"],
        help="Describe planned transformations before creating many alternative variables.",
    )
    assumptions_to_check = st.text_area(
        "Assumptions to check",
        value=current_plan["assumptions_to_check"],
        help="Examples: linearity, residual pattern, class balance, overdispersion, proportional odds.",
    )
    interpretation_boundaries = st.text_area(
        "Interpretation boundaries",
        value=current_plan["interpretation_boundaries"],
        help="State what the analysis cannot prove, especially causal conclusions.",
    )
    update_reason = st.text_input(
        "Reason for this plan update",
        value="Initial analysis plan." if not analysis_plan_is_recorded(current_plan) else "Refined analysis plan.",
        help="This reason is stored in the decision log for fields that changed.",
    )

    submitted = st.form_submit_button("Save analysis plan")

if submitted:
    updates = {
        "analysis_title": analysis_title,
        "research_question": research_question,
        "analysis_goal": analysis_goal,
        "unit_of_analysis": unit_of_analysis,
        "target_variable": target_variable,
        "candidate_features": candidate_features,
        "excluded_columns": excluded_columns,
        "known_id_columns": known_id_columns,
        "known_time_columns": known_time_columns,
        "grouping_or_cluster_columns": grouping_or_cluster_columns,
        "expected_target_type": expected_target_type,
        "planned_model_family": planned_model_family,
        "primary_metric_or_statistic": primary_metric_or_statistic,
        "train_test_strategy": train_test_strategy,
        "missing_data_strategy": missing_data_strategy,
        "transformation_plan": transformation_plan,
        "assumptions_to_check": assumptions_to_check,
        "interpretation_boundaries": interpretation_boundaries,
    }
    new_plan, log_entries = update_analysis_plan(
        st.session_state.get("analysis_plan") or create_empty_analysis_plan(),
        updates,
        reason=update_reason,
    )
    st.session_state["analysis_plan"] = new_plan
    st.session_state.setdefault("analysis_decision_log", [])
    st.session_state["analysis_decision_log"].extend(log_entries)
    st.success("Analysis plan saved.")
    if not log_entries:
        st.info("No fields changed, so no new decision-log entry was added.")
    st.rerun()

if st.button(
    "Clear analysis plan",
    help="Clear the plan and decision log. This does not change original_df or working_df.",
):
    st.session_state["analysis_plan"] = None
    st.session_state["analysis_decision_log"] = []
    st.session_state["rigor_warnings"] = []
    st.session_state["analysis_stage_status"] = {}
    st.success("Analysis plan cleared.")
    st.rerun()

st.subheader("Current Analysis Plan")
if analysis_plan_is_recorded(st.session_state.get("analysis_plan")):
    st.json(st.session_state["analysis_plan"])
else:
    st.info("No analysis plan has been recorded yet.")

st.subheader("Statistical Rigor Checklist")
checklist = build_rigor_checklist(
    analysis_plan=st.session_state.get("analysis_plan"),
    working_df=working_df,
    cleaning_log=st.session_state.get("cleaning_log", []),
    transformation_log=st.session_state.get("transformation_log", []),
    model_runs=st.session_state.get("model_runs", []),
    prediction_log=st.session_state.get("prediction_log", []),
)
_render_checklist(checklist)
st.session_state["analysis_stage_status"] = {
    "needs_attention": sum(1 for item in checklist if item["status"] == "needs_attention"),
    "completed": sum(1 for item in checklist if item["status"] == "completed"),
}

if working_df is not None:
    st.subheader("Data Readiness Summary")
    readiness = build_data_readiness(
        working_df,
        target_column=(st.session_state.get("analysis_plan") or {}).get("target_variable"),
    )
    readiness_summary = {
        key: value
        for key, value in readiness.items()
        if key != "missing_summary"
    }
    st.json(readiness_summary)

st.subheader("Decision Log")
decision_log = st.session_state.get("analysis_decision_log", [])
if decision_log:
    st.dataframe(pd.DataFrame(decision_log), use_container_width=True)
else:
    st.info("No analysis-plan decisions have been logged yet.")
