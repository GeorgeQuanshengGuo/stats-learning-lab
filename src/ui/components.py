"""Reusable Streamlit UI components for consistent pages."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import streamlit as st

from src.ui.layout import escape_html, join_css_classes, render_html_block


BADGE_COLOR_CLASSES = {
    "neutral": "workbench-badge-neutral",
    "primary": "",
    "success": "workbench-badge-success",
    "warning": "workbench-badge-warning",
    "danger": "workbench-badge-danger",
}

CARD_COLOR_CLASSES = {
    "neutral": "",
    "primary": "workbench-card-primary",
    "success": "workbench-card-success",
    "warning": "workbench-card-warning",
    "danger": "workbench-card-danger",
}

SEVERITY_CARD_CLASSES = {
    "info": "workbench-card-primary",
    "warning": "workbench-card-warning",
    "danger": "workbench-card-danger",
    "success": "workbench-card-success",
}


def _format_icon(icon: str | None) -> str:
    """Return a compact escaped icon prefix."""
    return f"{escape_html(icon)} " if icon else ""


def _format_body(body: object) -> str:
    """Escape body text and preserve simple line breaks."""
    return escape_html(body).replace("\n", "<br>")


def _render_badge_html(text: str, color: str = "neutral") -> str:
    """Build badge HTML used by badge-like components."""
    color_class = BADGE_COLOR_CLASSES.get(color, BADGE_COLOR_CLASSES["neutral"])
    class_name = join_css_classes(["workbench-badge", color_class])
    return f'<span class="{class_name}">{escape_html(text)}</span>'


def render_badge(text: str, color: str = "neutral") -> None:
    """Render a small badge."""
    render_html_block(_render_badge_html(text, color=color))


def render_page_header(
    title: str,
    subtitle: str | None = None,
    icon: str | None = None,
    tags: Sequence[str] | None = None,
    help_text: str | None = None,
) -> None:
    """Render a consistent page title area."""
    tag_html = ""
    if tags:
        badges = [_render_badge_html(tag, color="neutral") for tag in tags]
        tag_html = f'<div class="workbench-stack-sm">{" ".join(badges)}</div>'

    subtitle_html = (
        f'<p class="workbench-page-subtitle">{escape_html(subtitle)}</p>' if subtitle else ""
    )
    help_html = (
        f'<p class="workbench-help-text">{escape_html(help_text)}</p>' if help_text else ""
    )

    render_html_block(
        f"""
        <div class="workbench-page-header">
          <h1 class="workbench-page-title">{_format_icon(icon)}{escape_html(title)}</h1>
          {subtitle_html}
          {tag_html}
          {help_html}
        </div>
        """
    )


def render_section_header(
    title: str,
    description: str | None = None,
    icon: str | None = None,
    help_key: str | None = None,
) -> None:
    """Render a compact section heading with optional native help text."""
    heading = f"{_format_icon(icon)}{title}"
    st.subheader(heading, help=help_key)
    if description:
        st.caption(description)


def render_info_card(
    title: str,
    body: str,
    icon: str | None = None,
    badge: str | None = None,
    color: str = "neutral",
) -> None:
    """Render a reusable informational card."""
    color_class = CARD_COLOR_CLASSES.get(color, CARD_COLOR_CLASSES["neutral"])
    class_name = join_css_classes(["workbench-card", color_class])
    badge_html = _render_badge_html(badge, color=color if color in BADGE_COLOR_CLASSES else "neutral") if badge else ""

    render_html_block(
        f"""
        <div class="{class_name}">
          <div class="workbench-card-title">{_format_icon(icon)}{escape_html(title)}</div>
          <div class="workbench-card-body">{_format_body(body)}</div>
          {badge_html}
        </div>
        """
    )


def render_metric_card(
    label: str,
    value: object,
    helper_text: str | None = None,
    delta: object | None = None,
    status: str | None = None,
) -> None:
    """Render a metric inside a light card."""
    color_class = CARD_COLOR_CLASSES.get(status or "neutral", "")
    delta_html = (
        f'<div class="workbench-metric-delta">{escape_html(delta)}</div>' if delta is not None else ""
    )
    helper_html = (
        f'<div class="workbench-card-body">{_format_body(helper_text)}</div>' if helper_text else ""
    )
    render_html_block(
        f"""
        <div class="{join_css_classes(["workbench-card", color_class])}">
          <div class="workbench-metric-label">{escape_html(label)}</div>
          <div class="workbench-metric-value">{escape_html(value)}</div>
          {delta_html}
          {helper_html}
        </div>
        """
    )


def render_warning_card(
    title: str,
    body: str,
    severity: str = "warning",
) -> None:
    """Render an info, warning, danger, or success callout card."""
    severity_class = SEVERITY_CARD_CLASSES.get(severity, SEVERITY_CARD_CLASSES["warning"])
    class_name = join_css_classes(["workbench-card", severity_class])
    render_html_block(
        f"""
        <div class="{class_name}">
          <div class="workbench-card-title">{escape_html(title)}</div>
          <div class="workbench-card-body">{_format_body(body)}</div>
        </div>
        """
    )


def render_empty_state(
    title: str,
    body: str,
    action_label: str | None = None,
    action_page: str | None = None,
    icon: str | None = None,
) -> None:
    """Render a friendly empty state and optional page link."""
    render_html_block(
        f"""
        <div class="workbench-empty-state">
          <div class="workbench-empty-title">{_format_icon(icon)}{escape_html(title)}</div>
          <div class="workbench-card-body">{_format_body(body)}</div>
        </div>
        """
    )
    if action_label and action_page:
        st.page_link(action_page, label=action_label)


def render_next_step_panel(
    recommended_next_step: str,
    reason: str | None = None,
    page_link: str | None = None,
) -> None:
    """Render the next recommended workflow step."""
    body = recommended_next_step if reason is None else f"{recommended_next_step}\n\nWhy: {reason}"
    render_info_card("Next recommended step", body, icon="Next:", color="primary")
    if page_link:
        st.page_link(page_link, label="Go to next step")


def render_workflow_stepper(
    steps: Sequence[str],
    current_step: int | str,
    completed_steps: Iterable[int | str] | None = None,
) -> None:
    """Render a simple workflow stepper.

    ``current_step`` and ``completed_steps`` accept either zero-based indexes or
    exact step names. This keeps the helper easy to use from simple pages.
    """
    completed_set = set(completed_steps or [])
    step_html = []

    for index, step in enumerate(steps):
        is_current = current_step == index or current_step == step
        is_complete = index in completed_set or step in completed_set
        class_name = join_css_classes(
            [
                "workbench-workflow-step",
                "workbench-workflow-step-active" if is_current else None,
                "workbench-workflow-step-complete" if is_complete else None,
            ]
        )
        label = "Current" if is_current else "Done" if is_complete else "Next"
        step_html.append(
            f"""
            <div class="{class_name}">
              <div class="workbench-workflow-step-number">{index + 1}</div>
              <div class="workbench-card-title">{escape_html(step)}</div>
              <div class="workbench-muted">{label}</div>
            </div>
            """
        )

    render_html_block(f'<div class="workbench-workflow">{"".join(step_html)}</div>')


def render_two_column_panel(
    left_title: str,
    left_body: str,
    right_title: str,
    right_body: str,
) -> None:
    """Render two balanced explanatory panels."""
    left, right = st.columns(2)
    with left:
        render_info_card(left_title, left_body)
    with right:
        render_info_card(right_title, right_body)
