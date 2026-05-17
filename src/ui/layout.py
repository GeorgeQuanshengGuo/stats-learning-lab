"""Small layout helpers shared by Streamlit pages."""

from __future__ import annotations

import html
import textwrap
from collections.abc import Iterable

import streamlit as st


def escape_html(value: object) -> str:
    """Escape text before it is placed in small trusted HTML snippets."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def join_css_classes(classes: Iterable[str | None]) -> str:
    """Join CSS class names while ignoring empty values."""
    return " ".join(class_name for class_name in classes if class_name)


def render_html_block(markup: str) -> None:
    """Render a trusted local HTML snippet through Streamlit markdown."""
    dedented_markup = textwrap.dedent(markup).strip()
    normalized_markup = "\n".join(line.strip() for line in dedented_markup.splitlines())
    if hasattr(st, "html"):
        st.html(normalized_markup)
    else:
        st.markdown(normalized_markup, unsafe_allow_html=True)


def render_spacer(size: str = "md") -> None:
    """Render a small vertical spacer."""
    size_map = {"sm": "0.5rem", "md": "1rem", "lg": "1.5rem"}
    height = size_map.get(size, size_map["md"])
    render_html_block(f'<div style="height: {height};"></div>')


def two_equal_columns():
    """Return two equal-width Streamlit columns."""
    return st.columns(2)
