"""Reusable chart card rendering helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.ui.chart_explanations import get_chart_explanation
from src.ui.components import render_badge
from src.ui.layout import escape_html, render_html_block
from src.visualization.chart_metadata import ChartMetadata, create_chart_metadata


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "filename": "workbench_chart"},
}


def _metadata_value(value: Any) -> str:
    """Return a display-safe metadata value."""
    if value is None or value == "":
        return "Not specified"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "None"
    return str(value)


def render_chart_context(metadata: ChartMetadata) -> None:
    """Render chart context such as variables, method, and sample size."""
    render_html_block(
        f"""
        <div class="chart-card-context">
          <div class="chart-card-context-item"><strong>Variables</strong><br>{escape_html(_metadata_value(metadata.variables_used))}</div>
          <div class="chart-card-context-item"><strong>Method</strong><br>{escape_html(_metadata_value(metadata.method))}</div>
          <div class="chart-card-context-item"><strong>Sample size</strong><br>{escape_html(_metadata_value(metadata.sample_size))}</div>
          <div class="chart-card-context-item"><strong>Missing handling</strong><br>{escape_html(_metadata_value(metadata.missing_handling))}</div>
        </div>
        """
    )


def _apply_professional_plotly_style(fig: Any) -> Any:
    """Apply a restrained shared Plotly style without changing chart data."""
    if fig is None or not hasattr(fig, "update_layout"):
        return fig
    layout_updates = {
        "template": "plotly_white",
        "margin": {"l": 32, "r": 24, "t": 64, "b": 44},
        "font": {"size": 13},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    }
    if getattr(fig.layout, "height", None) is None:
        layout_updates["height"] = 420
    fig.update_layout(**layout_updates)
    return fig


def render_chart_interpretation(metadata: ChartMetadata) -> None:
    """Render how-to-read and next-step text."""
    how_to_read = metadata.how_to_read or get_chart_explanation(metadata.chart_type)["how_to_read"]
    next_step = metadata.next_step or get_chart_explanation(metadata.chart_type).get("next_step")
    note = f"<strong>How to read this:</strong> {escape_html(how_to_read)}"
    if next_step:
        note += f"<br><strong>Next step:</strong> {escape_html(next_step)}"
    render_html_block(f'<div class="chart-card-note">{note}</div>')


def render_chart_warning(metadata: ChartMetadata) -> None:
    """Render chart warnings when available."""
    warnings = metadata.warnings or get_chart_explanation(metadata.chart_type).get("warnings", [])
    if not warnings:
        return
    warning_text = " ".join(str(warning) for warning in warnings)
    render_html_block(
        f'<div class="chart-card-warning"><strong>Caution:</strong> {escape_html(warning_text)}</div>'
    )


def render_chart_actions(
    metadata: ChartMetadata,
    fig: Any | None = None,
    allow_download: bool = True,
    allow_add_to_report: bool = True,
) -> None:
    """Render optional chart actions."""
    columns = st.columns(3)
    if allow_download and fig is not None and hasattr(fig, "to_html"):
        with columns[0]:
            st.download_button(
                "Download Chart",
                data=fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                file_name=f"{metadata.chart_id}.html",
                mime="text/html",
                key=f"download_chart_{metadata.chart_id}",
                help="Download this interactive chart as an HTML file.",
            )
    if allow_add_to_report:
        with columns[1]:
            if st.button(
                "Add To Report",
                key=f"add_chart_{metadata.chart_id}",
                help="Save this chart's metadata so it can be included in the analysis report.",
            ):
                st.session_state.setdefault("report_chart_items", []).append(metadata.to_dict())
                st.success("Chart metadata added to report items.")
    with columns[2]:
        explanation = get_chart_explanation(metadata.chart_type)
        glossary_terms = explanation.get("glossary_terms", [])
        if glossary_terms:
            with st.expander("View Glossary Terms"):
                st.write(", ".join(term.replace("_", " ") for term in glossary_terms))


def render_chart_card(
    fig: Any,
    metadata: ChartMetadata,
    allow_download: bool = True,
    allow_add_to_report: bool = True,
) -> None:
    """Render a Plotly figure inside a professional chart card."""
    fig = _apply_professional_plotly_style(fig)
    with st.container(border=True):
        render_html_block(
            f"""
            <div class="chart-card-header">
              <div>
                <div class="chart-card-title">{escape_html(metadata.title)}</div>
                <div class="chart-card-subtitle">{escape_html(metadata.subtitle or metadata.caption or "")}</div>
              </div>
            </div>
            """
        )
        badge_cols = st.columns([0.2, 0.2, 0.6])
        with badge_cols[0]:
            render_badge(metadata.chart_type.replace("_", " ").title(), color="primary")
        with badge_cols[1]:
            render_badge(metadata.source_page, color="neutral")
        render_chart_context(metadata)
        st.plotly_chart(
            fig,
            width="stretch",
            config=PLOTLY_CONFIG,
            key=f"plotly_chart_{metadata.chart_id}",
        )
        render_chart_interpretation(metadata)
        render_chart_warning(metadata)
        render_chart_actions(
            metadata,
            fig=fig,
            allow_download=allow_download,
            allow_add_to_report=allow_add_to_report,
        )


def render_plotly_chart_card(
    fig: Any,
    title: str,
    chart_type: str,
    source_page: str,
    variables_used: list[str] | tuple[str, ...] | None = None,
    subtitle: str | None = None,
    method: str | None = None,
    sample_size: int | None = None,
    missing_handling: str | None = None,
    caption: str | None = None,
    how_to_read: str | None = None,
    warnings: list[str] | tuple[str, ...] | str | None = None,
    next_step: str | None = None,
    allow_download: bool = True,
    allow_add_to_report: bool = True,
) -> None:
    """Create metadata and render a Plotly chart card in one beginner-friendly call."""
    metadata = create_chart_metadata(
        title=title,
        subtitle=subtitle,
        chart_type=chart_type,
        source_page=source_page,
        variables_used=variables_used,
        method=method,
        sample_size=sample_size,
        missing_handling=missing_handling,
        caption=caption,
        how_to_read=how_to_read,
        warnings=warnings,
        next_step=next_step,
    )
    render_chart_card(
        fig,
        metadata,
        allow_download=allow_download,
        allow_add_to_report=allow_add_to_report,
    )
