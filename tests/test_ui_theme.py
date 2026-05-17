"""Tests for the global UI theme and style loader."""

from __future__ import annotations

import tomllib

import pytest

from src.ui import style_loader
from src.ui.theme import APP_THEME, get_global_css_path


def test_streamlit_theme_config_contains_expected_keys() -> None:
    config_path = get_global_css_path().parents[2] / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    theme = config["theme"]

    expected_keys = {
        "base",
        "primaryColor",
        "backgroundColor",
        "secondaryBackgroundColor",
        "textColor",
        "linkColor",
        "baseRadius",
        "buttonRadius",
        "borderColor",
        "showSidebarBorder",
        "chartCategoricalColors",
        "chartSequentialColors",
        "chartDivergingColors",
    }

    assert expected_keys.issubset(theme.keys())
    assert theme["base"] == "light"
    assert theme["primaryColor"] == APP_THEME["primary"]


def test_global_css_file_exists_and_has_expected_classes() -> None:
    css = style_loader.safe_read_css_file(get_global_css_path())

    assert ".workbench-card" in css
    assert ".workbench-hero" in css
    assert ".workbench-badge" in css
    assert ".workbench-term" in css
    assert ".workbench-workflow-step" in css
    assert "<script" not in css.lower()
    assert "javascript:" not in css.lower()


def test_safe_read_css_file_rejects_non_css_file(tmp_path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not css", encoding="utf-8")

    with pytest.raises(ValueError, match="Only CSS"):
        style_loader.safe_read_css_file(text_file)


def test_safe_read_css_file_rejects_file_outside_project(tmp_path) -> None:
    css_file = tmp_path / "outside.css"
    css_file.write_text(".example { color: red; }", encoding="utf-8")

    with pytest.raises(ValueError, match="project directory"):
        style_loader.safe_read_css_file(css_file)


def test_load_global_styles_renders_style_tag(monkeypatch) -> None:
    rendered = {}

    def fake_markdown(body: str, unsafe_allow_html: bool = False) -> None:
        rendered["body"] = body
        rendered["unsafe_allow_html"] = unsafe_allow_html

    monkeypatch.setattr(style_loader.st, "markdown", fake_markdown)

    style_loader.load_global_styles()

    assert rendered["body"].startswith("<style>")
    assert ".workbench-card" in rendered["body"]
    assert rendered["unsafe_allow_html"] is True
