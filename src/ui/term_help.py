"""Streamlit render helpers for glossary term explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.ui.glossary import (
    get_glossary_entry,
    glossary_terms_as_table,
    list_glossary_categories,
    search_glossary_terms,
)
from src.ui.layout import escape_html, render_html_block


def _related_terms_text(entry: dict[str, Any]) -> str:
    """Return related term keys as readable text."""
    related_terms = entry.get("related_terms") or []
    return ", ".join(str(term).replace("_", " ") for term in related_terms) or "None listed"


def _render_entry_body(entry: dict[str, Any]) -> None:
    """Render the body of a glossary popover or expander."""
    st.markdown(f"**Definition:** {entry['short_definition']}")
    st.write(entry["long_definition"])
    st.markdown(f"**Example:** {entry['example']}")
    st.markdown(f"**Common misunderstanding:** {entry['common_misunderstanding']}")
    st.caption(f"Related terms: {_related_terms_text(entry)}")


def render_inline_term(key: str, display_text: str | None = None) -> None:
    """Render a term as inline text with a native browser title tooltip."""
    entry = get_glossary_entry(key)
    text = display_text or entry["term"]
    title = entry["short_definition"]
    render_html_block(
        (
            f'<span class="workbench-term" title="{escape_html(title)}">'
            f"{escape_html(text)}</span>"
        )
    )


def render_glossary_popover(key: str, display_text: str | None = None) -> None:
    """Render a compact popover for one glossary entry."""
    entry = get_glossary_entry(key)
    label = display_text or "?"

    if hasattr(st, "popover"):
        with st.popover(label, help=entry["short_definition"]):
            st.markdown(f"### {entry['term']}")
            _render_entry_body(entry)
    else:
        st.caption(f"{entry['term']}: {entry['short_definition']}")


def render_help_icon(key: str) -> None:
    """Render only a small help icon for a glossary term."""
    render_glossary_popover(key, display_text="?")


def render_term_help(
    key: str,
    display_text: str | None = None,
    mode: str = "popover",
) -> None:
    """Render glossary help in inline, icon, or popover mode."""
    if mode == "inline":
        render_inline_term(key, display_text=display_text)
        return
    if mode == "icon":
        render_help_icon(key)
        return

    entry = get_glossary_entry(key)
    label = display_text or entry["term"]
    left, right = st.columns([0.88, 0.12])
    with left:
        st.write(label)
    with right:
        render_help_icon(key)


def render_glossary_search() -> None:
    """Render the searchable glossary browser."""
    categories = ["All"] + list_glossary_categories()
    controls = st.columns([0.62, 0.38])
    with controls[0]:
        query = st.text_input(
            "Search glossary terms",
            placeholder="Try: p-value, ROC AUC, residual, calibration",
        )
    with controls[1]:
        category = st.selectbox("Category", categories)

    entries = search_glossary_terms(query=query, category=category)

    if not entries:
        st.info("No matching glossary terms were found. Try a broader search.")
        return

    st.caption(f"Showing {len(entries)} glossary terms.")
    table = pd.DataFrame(glossary_terms_as_table())
    if query.strip() or category != "All":
        keys = {entry["key"] for entry in entries}
        table = table[table["key"].isin(keys)]
    st.dataframe(table, use_container_width=True, hide_index=True)

    for entry in entries:
        with st.expander(f"{entry['term']} ({entry.get('category', 'general')})"):
            _render_entry_body(entry)
