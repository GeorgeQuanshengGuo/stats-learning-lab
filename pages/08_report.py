"""Streamlit report export page."""

import streamlit as st

from src.reporting.export_html import export_report_html
from src.reporting.export_markdown import export_report_markdown
from src.reporting.report_builder import build_report_context
from src.ui.components import render_empty_state
from src.ui.page_templates import render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Report",
    purpose="Preview and export a transparent Markdown or HTML summary of the current analysis state.",
    when_to_use="Use this page after exploring data, applying confirmed operations, or saving model runs.",
    common_mistake="Do not present metrics or feature importance without the limitations and assumptions section.",
    key_terms=[("model_run", "Model run"), ("limitations", "Limitations")],
    next_step="Download the Markdown or HTML report after reviewing the preview.",
    tags=["Export", "Documentation"],
)

working_df = st.session_state.get("working_df")
schema = st.session_state.get("schema")
cleaning_log = st.session_state.get("cleaning_log", [])
transformation_log = st.session_state.get("transformation_log", [])
model_runs = st.session_state.get("model_runs", [])
model_artifacts = st.session_state.get("model_artifacts", {})
prediction_log = st.session_state.get("prediction_log", [])

if working_df is None:
    render_empty_state(
        "No dataset loaded yet",
        "The report can still render, but dataset-specific sections will be unavailable.",
        action_label="Go to Upload Data",
        action_page="pages/01_upload_data.py",
        icon="Report:",
    )

report_title = st.text_input(
    "Report title",
    value="Analysis Report",
    help="Title shown at the top of the exported Markdown and HTML reports.",
)

report_context = build_report_context(
    working_df=working_df,
    schema=schema,
    cleaning_log=cleaning_log,
    transformation_log=transformation_log,
    model_runs=model_runs,
    model_artifacts=model_artifacts,
    prediction_log=prediction_log,
    title=report_title,
)
markdown_report = export_report_markdown(report_context)
html_report = export_report_html(report_context)

st.subheader("Preview")
st.markdown(markdown_report)

st.download_button(
    "Download Markdown report",
    data=markdown_report.encode("utf-8"),
    file_name="analysis_report.md",
    mime="text/markdown",
)

st.download_button(
    "Download HTML report",
    data=html_report.encode("utf-8"),
    file_name="analysis_report.html",
    mime="text/html",
)
