"""Main Streamlit entry point for the learning app."""

import streamlit as st

from src.core.state import init_session_state
from src.ui.home_sections import get_logo_path, render_home_page
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles

APP_NAME = "Stats Learning Lab"


def main() -> None:
    """Start the Streamlit app."""
    st.set_page_config(page_title=APP_NAME, layout="wide")
    logo_path = get_logo_path()
    if logo_path.exists() and hasattr(st, "logo"):
        st.logo(str(logo_path), size="large")
    load_global_styles()
    init_session_state()
    render_simple_sidebar()

    render_home_page(st.session_state)


if __name__ == "__main__":
    main()
