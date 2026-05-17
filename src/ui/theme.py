"""Shared theme constants for the Streamlit workbench UI."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_CSS_PATH = PROJECT_ROOT / "assets" / "styles" / "app.css"

APP_THEME = {
    "primary": "#3347B8",
    "primary_soft": "#E0E7FF",
    "background": "#F8FAFC",
    "secondary_background": "#EEF2F7",
    "surface": "#FFFFFF",
    "text": "#172033",
    "muted": "#64748B",
    "border": "#CBD5E1",
    "warning": "#B45309",
    "warning_soft": "#FEF3C7",
    "danger": "#B91C1C",
    "danger_soft": "#FEE2E2",
    "success": "#047857",
    "success_soft": "#DCFCE7",
}

CHART_CATEGORICAL_COLORS = [
    "#3347B8",
    "#0F766E",
    "#B45309",
    "#9F1239",
    "#6D28D9",
    "#047857",
    "#0369A1",
    "#7C2D12",
    "#475569",
    "#0F172A",
]

CHART_SEQUENTIAL_COLORS = [
    "#F8FAFC",
    "#E0E7FF",
    "#C7D2FE",
    "#A5B4FC",
    "#818CF8",
    "#6366F1",
    "#4F46E5",
    "#4338CA",
    "#3730A3",
    "#312E81",
]

CHART_DIVERGING_COLORS = [
    "#9F1239",
    "#BE123C",
    "#F59E0B",
    "#FDE68A",
    "#F8FAFC",
    "#C7D2FE",
    "#818CF8",
    "#4F46E5",
    "#3730A3",
    "#1E1B4B",
]


def get_theme_color(name: str) -> str:
    """Return a named theme color.

    Raises:
        KeyError: If the color name is not part of the workbench theme.
    """
    return APP_THEME[name]


def get_global_css_path() -> Path:
    """Return the trusted local CSS file used by the app."""
    return GLOBAL_CSS_PATH
