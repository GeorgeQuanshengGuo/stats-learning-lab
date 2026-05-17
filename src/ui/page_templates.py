"""Composable page templates built from the shared UI components."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from src.ui.components import (
    render_empty_state,
    render_next_step_panel,
    render_page_header,
    render_two_column_panel,
)
from src.ui.term_help import render_term_help


def render_standard_page_intro(
    title: str,
    purpose: str,
    when_to_use: str,
    does_not_do: str,
    next_step: str | None = None,
    icon: str | None = None,
    tags: Sequence[str] | None = None,
) -> None:
    """Render the recommended intro structure for major pages."""
    render_page_header(title=title, subtitle=purpose, icon=icon, tags=tags)
    render_two_column_panel(
        left_title="When to use this page",
        left_body=when_to_use,
        right_title="What this page does not do",
        right_body=does_not_do,
    )
    if next_step:
        render_next_step_panel(next_step)


def render_dataset_required_empty_state(action_page: str = "pages/01_upload_data.py") -> None:
    """Render the standard message for pages that need a loaded dataset."""
    render_empty_state(
        title="No dataset loaded yet",
        body="Upload a CSV or Excel file before using this analysis page.",
        action_label="Go to Upload Data",
        action_page=action_page,
        icon="Dataset:",
    )


def render_educational_page_header(
    title: str,
    purpose: str,
    when_to_use: str,
    common_mistake: str | None = None,
    key_terms: Sequence[tuple[str, str] | str] | None = None,
    next_step: str | None = None,
    next_page: str | None = None,
    tags: Sequence[str] | None = None,
) -> None:
    """Render a lightweight, consistent page header.

    This helper is intentionally UI-only. It does not read or modify session
    state, datasets, logs, or model objects.
    """
    render_page_header(title=title, subtitle=purpose)
    if when_to_use:
        st.caption(f"Use when: {when_to_use}")
    if key_terms:
        with st.expander("Term help", expanded=False):
            for term in key_terms:
                if isinstance(term, tuple):
                    key, label = term
                else:
                    key = term
                    label = None
                render_term_help(key, display_text=label)
