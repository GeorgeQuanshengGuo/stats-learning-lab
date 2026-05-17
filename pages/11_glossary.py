"""Searchable glossary page for analysis terms."""

import streamlit as st

from src.ui.components import render_info_card, render_page_header, render_section_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles
from src.ui.term_help import render_glossary_search, render_term_help


load_global_styles()
render_simple_sidebar()

render_page_header(
    title="Glossary",
    subtitle=(
        "Search short explanations for common EDA, statistical modeling, "
        "machine learning, diagnostics, prediction, and visualization terms."
    ),
    tags=["educational", "non-black-box", "plain language"],
)

render_info_card(
    title="How to use these explanations",
    body=(
        "Use the glossary when a chart, metric, model output, or control label is unfamiliar. "
        "Definitions are intentionally practical and include common misunderstandings."
    ),
    color="primary",
)

render_section_header("Quick term help", "Five examples of the same help pattern used across the app.")
quick_cols = st.columns(5)
quick_terms = ["p_value", "confidence_interval", "roc_auc", "feature_importance", "threshold"]
for column, term_key in zip(quick_cols, quick_terms, strict=True):
    with column:
        render_term_help(term_key, mode="icon")

render_section_header("Search glossary")
render_glossary_search()
