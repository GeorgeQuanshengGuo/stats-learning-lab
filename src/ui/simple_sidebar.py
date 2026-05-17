"""Simple, readable sidebar navigation for the learning app."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.core.workspace_snapshot import (
    AUTOSAVE_SNAPSHOT_PATH,
    MANUAL_SNAPSHOT_PATH,
    clear_workspace_session,
    has_meaningful_workspace,
    load_autosave_preference,
    load_workspace_snapshot,
    maybe_autosave_workspace,
    restore_workspace_snapshot,
    save_workspace_snapshot,
    set_autosave_preference,
    snapshot_exists,
    summarize_workspace_snapshot,
)
from src.ui.components import render_badge
from src.ui.home_sections import get_logo_path
from src.ui.layout import escape_html, render_html_block


NAV_GROUPS = [
    (
        "Start",
        [
            ("Home", "app.py"),
            ("Upload Data", "pages/01_upload_data.py"),
        ],
    ),
    (
        "Data",
        [
            ("Exploratory Data Analysis", "pages/02_eda.py"),
            ("Data Cleaning", "pages/03_data_cleaning.py"),
            ("Transformations", "pages/04_transformations.py"),
        ],
    ),
    (
        "Models",
        [
            ("Statistical Models", "pages/05_statistical_models.py"),
            ("Machine Learning", "pages/07_machine_learning.py"),
            ("Model Comparison", "pages/06_model_comparison.py"),
            ("Model Diagnostics", "pages/10_model_diagnostics.py"),
        ],
    ),
    (
        "Output",
        [
            ("Prediction", "pages/09_prediction.py"),
            ("Report", "pages/08_report.py"),
        ],
    ),
    (
        "Help",
        [
            ("Glossary", "pages/11_glossary.py"),
        ],
    ),
]


def build_simple_sidebar_status(session_state: dict[str, Any]) -> dict[str, Any]:
    """Build compact sidebar status values without mutating session state."""
    working_df = session_state.get("working_df")
    has_dataset = isinstance(working_df, pd.DataFrame)
    rows = 0
    columns = 0
    if has_dataset:
        rows, columns = working_df.shape

    return {
        "has_dataset": has_dataset,
        "rows": rows,
        "columns": columns,
    }


def _status_row(label: str, value: str) -> None:
    """Render one compact status row."""
    render_html_block(
        f"""
        <div class="simple-sidebar-status-row">
          <span class="simple-sidebar-status-label">{escape_html(label)}</span>
          <span class="simple-sidebar-status-value">{escape_html(value)}</span>
        </div>
        """
    )


def _render_status(status: dict[str, Any]) -> None:
    """Render compact dataset and model status."""
    render_html_block('<div class="simple-sidebar-section">Status</div>')
    render_html_block('<div class="simple-sidebar-status">')
    if status["has_dataset"]:
        render_badge("Dataset Loaded", color="success")
        _status_row("Rows", f"{status['rows']:,}")
        _status_row("Columns", f"{status['columns']:,}")
    else:
        render_badge("No Dataset", color="warning")
        st.caption("Upload a dataset to begin.")

    render_html_block("</div>")


def _render_nav_links() -> None:
    """Render grouped navigation with Title Case labels."""
    render_html_block('<div class="simple-sidebar-section">Navigation</div>')
    for group_name, links in NAV_GROUPS:
        st.caption(group_name)
        for label, page_path in links:
            st.page_link(page_path, label=label)


def _render_workspace_controls() -> None:
    """Render small save/restore controls without changing data logic."""
    render_html_block('<div class="simple-sidebar-section">Workspace</div>')
    has_workspace = has_meaningful_workspace(st.session_state)
    st.session_state.setdefault("workspace_autosave_enabled", load_autosave_preference())

    if (
        snapshot_exists(AUTOSAVE_SNAPSHOT_PATH)
        and not has_workspace
        and not st.session_state.get("workspace_restore_dismissed", False)
    ):
        snapshot = load_workspace_snapshot(AUTOSAVE_SNAPSHOT_PATH)
        summary = summarize_workspace_snapshot(snapshot)
        st.info("Found a saved workspace. Restore it?")
        st.caption(
            f"{summary['dataset_name']} · {summary['rows']:,} rows · "
            f"{summary['columns']:,} columns · {summary['model_runs']} model runs"
        )
        restore_col, fresh_col = st.columns(2)
        with restore_col:
            if st.button("Restore", key="workspace_restore_autosave"):
                restore_workspace_snapshot(st.session_state, snapshot)
                st.rerun()
        with fresh_col:
            if st.button("Start fresh", key="workspace_start_fresh_autosave"):
                autosave_enabled = st.session_state.get("workspace_autosave_enabled", False)
                clear_workspace_session(st.session_state, preserve_keys={"workspace_autosave_enabled"})
                st.session_state["workspace_autosave_enabled"] = autosave_enabled
                st.session_state["workspace_restore_dismissed"] = True
                st.rerun()

    with st.expander("Save / restore", expanded=False):
        if st.button(
            "Save workspace",
            key="workspace_manual_save",
            disabled=not has_workspace,
            help="Save the current dataset, logs, model runs, page selections, and fitted artifacts when possible.",
        ):
            snapshot = save_workspace_snapshot(st.session_state, path=MANUAL_SNAPSHOT_PATH, source="manual")
            skipped = snapshot.get("skipped_keys") or []
            if skipped:
                st.warning(f"Workspace saved, but skipped items that could not be saved: {', '.join(skipped)}")
            else:
                st.success("Workspace saved.")

        if snapshot_exists(MANUAL_SNAPSHOT_PATH):
            if st.button("Restore manual save", key="workspace_restore_manual"):
                snapshot = load_workspace_snapshot(MANUAL_SNAPSHOT_PATH)
                restore_workspace_snapshot(st.session_state, snapshot)
                st.rerun()

        autosave_enabled = st.checkbox(
            "Auto-save workspace",
            key="workspace_autosave_enabled",
            help="When on, the app saves a local autosave snapshot. Refreshing still asks before restoring.",
        )
        set_autosave_preference(bool(autosave_enabled))
        if autosave_enabled and has_workspace:
            try:
                maybe_autosave_workspace(st.session_state)
                st.caption("Auto-save is on.")
            except Exception as error:
                st.warning(f"Auto-save could not save this workspace: {error}")

        if has_workspace and st.button("Start fresh", key="workspace_start_fresh_current"):
            autosave_enabled = st.session_state.get("workspace_autosave_enabled", False)
            clear_workspace_session(st.session_state, preserve_keys={"workspace_autosave_enabled"})
            st.session_state["workspace_autosave_enabled"] = autosave_enabled
            st.session_state["workspace_restore_dismissed"] = True
            st.rerun()


def render_simple_sidebar() -> None:
    """Render the shared simple sidebar.

    This helper only reads session state. It does not modify datasets, logs, or
    model results.
    """
    with st.sidebar:
        logo_path = get_logo_path()
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            render_html_block('<div class="simple-sidebar-title">Stats Learning Lab</div>')
        render_html_block(
            """
            <div class="simple-sidebar-caption">
            A local practice space for learning statistics and machine learning.
            </div>
            """
        )
        _render_status(build_simple_sidebar_status(st.session_state))
        _render_workspace_controls()
        _render_nav_links()
