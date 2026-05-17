"""Helpers for loading trusted local CSS into Streamlit pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ui.theme import PROJECT_ROOT, get_global_css_path


def safe_read_css_file(path: str | Path) -> str:
    """Read a trusted local CSS file.

    The loader accepts only local ``.css`` files inside the project directory.
    This keeps the styling layer simple and avoids loading remote or generated
    user content into custom HTML.
    """
    css_path = Path(path).expanduser().resolve()
    project_root = PROJECT_ROOT.resolve()

    if css_path.suffix.lower() != ".css":
        raise ValueError("Only CSS files can be loaded as global styles.")

    try:
        css_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("CSS files must live inside the project directory.") from exc

    if not css_path.is_file():
        raise FileNotFoundError(f"CSS file was not found: {css_path}")

    return css_path.read_text(encoding="utf-8")


def load_global_styles(path: str | Path | None = None) -> None:
    """Load the app's trusted CSS foundation into the current Streamlit page."""
    css_path = get_global_css_path() if path is None else Path(path)
    css = safe_read_css_file(css_path)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
