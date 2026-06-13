"""Small helpers for managing Streamlit session state."""

import pandas as pd

import streamlit as st

from src.core.model_artifacts import clear_model_artifacts, initialize_model_artifacts
from src.core.model_run import clear_model_runs


def init_session_state() -> None:
    """Create the session keys used by the workbench.

    original_df stores the uploaded data exactly as it arrived.
    working_df is the copy that later cleaning steps will use.
    """
    st.session_state.setdefault("original_df", None)
    st.session_state.setdefault("working_df", None)
    st.session_state.setdefault("uploaded_file_name", None)
    st.session_state.setdefault("cleaning_log", [])
    st.session_state.setdefault("transformation_log", [])
    st.session_state.setdefault("model_runs", [])
    st.session_state.setdefault("prediction_log", [])
    st.session_state.setdefault("analysis_plan", None)
    st.session_state.setdefault("analysis_decision_log", [])
    st.session_state.setdefault("rigor_warnings", [])
    st.session_state.setdefault("analysis_stage_status", {})
    st.session_state.setdefault("pca_artifacts", [])
    st.session_state.setdefault("clustering_artifacts", [])
    st.session_state.setdefault("anomaly_artifacts", [])
    st.session_state.setdefault("workspace_restore_dismissed", False)
    initialize_model_artifacts()


def set_uploaded_data(data: pd.DataFrame) -> None:
    """Store safe copies of an uploaded dataset in session state."""
    st.session_state["original_df"] = data.copy(deep=True)
    st.session_state["working_df"] = data.copy(deep=True)
    st.session_state["cleaning_log"] = []
    st.session_state["transformation_log"] = []
    st.session_state["prediction_log"] = []
    st.session_state["analysis_plan"] = None
    st.session_state["analysis_decision_log"] = []
    st.session_state["rigor_warnings"] = []
    st.session_state["analysis_stage_status"] = {}
    st.session_state["pca_artifacts"] = []
    st.session_state["clustering_artifacts"] = []
    st.session_state["anomaly_artifacts"] = []
    clear_model_runs()
    clear_latest_model_results()


def apply_cleaning_result(new_df: pd.DataFrame, log_entry: dict) -> None:
    """Replace working_df after the user confirms a cleaning operation."""
    st.session_state["working_df"] = new_df.copy(deep=True)
    st.session_state["cleaning_log"].append(log_entry)
    clear_latest_model_results()


def apply_transformation_result(new_df: pd.DataFrame, log_entry: dict) -> None:
    """Replace working_df after the user confirms a transformation."""
    st.session_state["working_df"] = new_df.copy(deep=True)
    st.session_state["transformation_log"].append(log_entry)
    clear_latest_model_results()


def reset_working_data() -> None:
    """Restore working_df from original_df and clear the cleaning log."""
    original_df = st.session_state.get("original_df")
    if original_df is not None:
        st.session_state["working_df"] = original_df.copy(deep=True)
    st.session_state["cleaning_log"] = []
    st.session_state["transformation_log"] = []
    clear_latest_model_results()


def clear_latest_model_results() -> None:
    """Remove displayed model results that no longer match the working data."""
    st.session_state.pop("latest_linear_regression_result", None)
    st.session_state.pop("latest_logistic_regression_result", None)
    st.session_state.pop("latest_ml_regression_results", None)
    st.session_state.pop("latest_ml_classification_results", None)
    st.session_state.pop("latest_ml_multiclass_results", None)
    st.session_state.pop("latest_pca_result", None)
    st.session_state.pop("latest_clustering_result", None)
    st.session_state.pop("latest_anomaly_result", None)
    st.session_state["pca_artifacts"] = []
    st.session_state["clustering_artifacts"] = []
    st.session_state["anomaly_artifacts"] = []
    clear_model_artifacts()
