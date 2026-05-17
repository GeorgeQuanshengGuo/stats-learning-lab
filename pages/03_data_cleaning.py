"""Streamlit page for Version 0.2 missing value handling."""

import pandas as pd
import streamlit as st

from src.core.state import apply_cleaning_result, reset_working_data
from src.data.cleaning import (
    drop_all_rows_with_any_missing,
    drop_missing_rows_for_column,
    fill_categorical_missing_label,
    fill_categorical_mode,
    fill_numeric_mean,
    fill_numeric_median,
)
from src.eda.missing import build_missing_table
from src.ui.page_templates import render_dataset_required_empty_state, render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles


def _run_cleaning_operation(
    df: pd.DataFrame,
    operation_label: str,
    selected_column: str | None,
    label: str,
):
    """Run the selected cleaning operation for preview."""
    if operation_label == "Drop rows missing in selected column":
        return drop_missing_rows_for_column(df, selected_column)
    if operation_label == "Drop rows with any missing value":
        return drop_all_rows_with_any_missing(df)
    if operation_label == "Fill numeric mean":
        return fill_numeric_mean(df, selected_column)
    if operation_label == "Fill numeric median":
        return fill_numeric_median(df, selected_column)
    if operation_label == "Fill categorical mode":
        return fill_categorical_mode(df, selected_column)
    return fill_categorical_missing_label(df, selected_column, label=label)


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Data Cleaning",
    purpose="Preview and confirm missing-value cleaning actions on the working dataset only.",
    when_to_use="Use this page after EDA identifies missing values that need a deliberate handling choice.",
    common_mistake="Do not drop rows or impute values automatically without checking how many rows and values are affected.",
    key_terms=["imputation", ("complete_case_analysis", "Complete-case analysis")],
    next_step="After cleaning, revisit EDA or create transformations if the variables need reshaping.",
    next_page="pages/04_transformations.py",
    tags=["Data preparation", "Missing values"],
)

working_df = st.session_state.get("working_df")
original_df = st.session_state.get("original_df")

if working_df is None or original_df is None:
    render_dataset_required_empty_state()
else:
    st.subheader("Missing Values")
    st.dataframe(build_missing_table(working_df), use_container_width=True)

    if st.button(
        "Reset working data from original dataset",
        help="Restore the working dataset from the protected original upload. This clears applied cleaning changes from the working copy.",
    ):
        reset_working_data()
        st.success("Working data has been reset from the original dataset.")
        st.rerun()

    st.subheader("Choose a Cleaning Operation")
    operation_label = st.selectbox(
        "Operation",
        [
            "Drop rows missing in selected column",
            "Drop rows with any missing value",
            "Fill numeric mean",
            "Fill numeric median",
            "Fill categorical mode",
            "Fill categorical missing label",
        ],
        help="Choose how missing values should be handled. Preview first so you can see the affected rows and values.",
    )

    selected_column = None
    if operation_label != "Drop rows with any missing value":
        selected_column = st.selectbox(
            "Column",
            working_df.columns,
            help="Choose the variable whose missing values will be dropped or filled.",
        )

    label = "Missing"
    if operation_label == "Fill categorical missing label":
        label = st.text_input(
            "Missing label",
            value="Missing",
            help="Text to insert for missing categorical values. Use a clear label that will not be confused with real categories.",
        )

    if st.button(
        "Preview cleaning operation",
        help="Show what would change before applying anything to the working dataset.",
    ):
        preview_df, log_entry = _run_cleaning_operation(
            working_df,
            operation_label,
            selected_column,
            label,
        )
        st.session_state["cleaning_preview_df"] = preview_df
        st.session_state["cleaning_preview_log"] = log_entry

    preview_df = st.session_state.get("cleaning_preview_df")
    log_entry = st.session_state.get("cleaning_preview_log")

    if preview_df is not None and log_entry is not None:
        st.subheader("Preview")
        st.json(log_entry)
        st.dataframe(preview_df.head(50), use_container_width=True)

        if st.button(
            "Confirm and apply to working data",
            help="Apply the previewed cleaning result to working_df only. The original uploaded data stays unchanged.",
        ):
            apply_cleaning_result(preview_df, log_entry)
            st.session_state["cleaning_preview_df"] = None
            st.session_state["cleaning_preview_log"] = None
            st.success("Cleaning operation applied to working data.")
            st.rerun()

    st.subheader("Cleaning Log")
    cleaning_log = st.session_state.get("cleaning_log", [])
    if cleaning_log:
        st.dataframe(pd.DataFrame(cleaning_log), use_container_width=True)
    else:
        st.info("No cleaning operations have been applied yet.")
