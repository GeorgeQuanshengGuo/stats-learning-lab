"""Model comparison page."""

import pandas as pd
import streamlit as st

from src.core.model_comparison import (
    build_binary_classification_comparison_table,
    build_count_regression_comparison_table,
    build_model_run_overview,
    build_multiclass_classification_comparison_table,
    build_ordinal_classification_comparison_table,
    build_regression_comparison_table,
    filter_model_runs,
)
from src.core.model_run import clear_model_runs, get_model_runs
from src.reporting.interpretation import explain_model_run
from src.ui.components import render_empty_state
from src.ui.page_templates import render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles


def _download_table(table: pd.DataFrame, label: str, file_name: str) -> None:
    """Render a CSV download button for one displayed comparison table."""
    st.download_button(
        label,
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Model Comparison",
    purpose="Compare saved model runs using task-appropriate metrics and interpretation notes.",
    when_to_use="Use this page after fitting one or more statistical or machine learning models.",
    common_mistake="Do not compare regression and classification models with one universal score.",
    key_terms=[("model_run", "Model run"), "rmse", "f1_score", "roc_auc"],
    next_step="After choosing promising runs, inspect diagnostics or export a report.",
    next_page="pages/10_model_diagnostics.py",
    tags=["Model review", "Saved runs"],
)

model_runs = get_model_runs()

if not model_runs:
    render_empty_state(
        "No saved model runs yet",
        "Fit a model first so it can appear in comparison tables.",
        action_label="Go to Statistical Models",
        action_page="pages/05_statistical_models.py",
        icon="Models:",
    )
else:
    task_options = ["All"] + sorted({run["task_type"] for run in model_runs})
    target_options = ["All"] + sorted({run["target"] for run in model_runs})
    family_options = ["All"] + sorted({run["model_family"] for run in model_runs})

    selected_task = st.selectbox(
        "Task type",
        task_options,
        help="Filter saved runs by problem type, such as regression or binary classification.",
    )
    selected_target = st.selectbox(
        "Target variable",
        target_options,
        help="Filter saved runs to one target variable.",
    )
    selected_family = st.selectbox(
        "Model family",
        family_options,
        help="Filter saved runs by statistical or machine learning model family.",
    )

    filtered_runs = filter_model_runs(model_runs, selected_task, selected_target, selected_family)

    if st.button(
        "Clear saved model runs",
        help="Remove saved ModelRun metadata from this session. This does not modify uploaded data.",
    ):
        clear_model_runs()
        st.success("Saved model runs cleared.")
        st.rerun()

    st.subheader("Interpretation Assistant")
    for run in filtered_runs:
        label = f"{run['model_name']} | {run['target']} | {run['model_family']}"
        with st.expander(label):
            st.write(explain_model_run(run))

    st.subheader("All Saved Runs")
    overview = build_model_run_overview(filtered_runs)
    st.dataframe(overview, use_container_width=True)

    regression_table = build_regression_comparison_table(filtered_runs)
    classification_table = build_binary_classification_comparison_table(filtered_runs)
    multiclass_table = build_multiclass_classification_comparison_table(filtered_runs)
    ordinal_table = build_ordinal_classification_comparison_table(filtered_runs)
    count_table = build_count_regression_comparison_table(filtered_runs)

    st.subheader("Regression Runs")
    if regression_table.empty:
        st.info("No regression runs match the current filters.")
    else:
        st.dataframe(regression_table, use_container_width=True)
        _download_table(
            regression_table,
            "Download regression comparison table as CSV",
            "regression_model_comparison.csv",
        )

    st.subheader("Binary Classification Runs")
    if classification_table.empty:
        st.info("No binary classification runs match the current filters.")
    else:
        st.dataframe(classification_table, use_container_width=True)
        _download_table(
            classification_table,
            "Download binary classification comparison table as CSV",
            "binary_classification_model_comparison.csv",
        )

    st.subheader("Multiclass Classification Runs")
    if multiclass_table.empty:
        st.info("No multiclass classification runs match the current filters.")
    else:
        st.dataframe(multiclass_table, use_container_width=True)
        _download_table(
            multiclass_table,
            "Download multiclass classification comparison table as CSV",
            "multiclass_classification_model_comparison.csv",
        )

    st.subheader("Ordinal Classification Runs")
    if ordinal_table.empty:
        st.info("No ordinal classification runs match the current filters.")
    else:
        st.dataframe(ordinal_table, use_container_width=True)
        _download_table(
            ordinal_table,
            "Download ordinal classification comparison table as CSV",
            "ordinal_classification_model_comparison.csv",
        )

    st.subheader("Count Regression Runs")
    if count_table.empty:
        st.info("No count regression runs match the current filters.")
    else:
        st.dataframe(count_table, use_container_width=True)
        _download_table(
            count_table,
            "Download count regression comparison table as CSV",
            "count_regression_model_comparison.csv",
        )
