"""Tests for the polished home page section helpers."""

from __future__ import annotations

import pandas as pd

from src.ui.home_sections import (
    EDA_PREVIEW_PATH,
    LOGO_PATH,
    MODELING_PREVIEW_PATH,
    MODULE_CARDS,
    REPORT_PREVIEW_PATH,
    RECOMMENDED_WORKFLOW,
    WORKFLOW_ILLUSTRATION_PATH,
    WORKFLOW_STEPS,
    _render_cards,
    _render_simple_cards,
    build_home_status_summary,
)


def test_home_illustration_assets_exist() -> None:
    for path in [
        LOGO_PATH,
        WORKFLOW_ILLUSTRATION_PATH,
        EDA_PREVIEW_PATH,
        MODELING_PREVIEW_PATH,
        REPORT_PREVIEW_PATH,
    ]:
        assert path.exists()
        assert path.suffix == ".svg"


def test_home_status_without_dataset() -> None:
    status = build_home_status_summary({"model_runs": [{}, {}]})

    assert status["has_dataset"] is False
    assert status["rows"] == 0
    assert status["columns"] == 0
    assert status["model_runs"] == 2


def test_home_status_with_loaded_dataset() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    status = build_home_status_summary(
        {
            "working_df": df,
            "original_df": df.copy(),
            "uploaded_file_name": "sample.csv",
            "model_runs": [{}],
        }
    )

    assert status["has_dataset"] is True
    assert status["dataset_name"] == "sample.csv"
    assert status["rows"] == 3
    assert status["columns"] == 2
    assert status["model_runs"] == 1
    assert "matches" in status["working_status"]


def test_home_status_when_working_dataset_changed_shape() -> None:
    original_df = pd.DataFrame({"x": [1, 2, 3]})
    working_df = pd.DataFrame({"x": [1, 2]})

    status = build_home_status_summary(
        {
            "working_df": working_df,
            "original_df": original_df,
        }
    )

    assert status["has_dataset"] is True
    assert "differs" in status["working_status"]


def test_home_sections_cover_required_workflow_and_modules() -> None:
    workflow_titles = {step["title"] for step in WORKFLOW_STEPS}
    module_titles = {title for title, _ in MODULE_CARDS}
    recommended_titles = {title for _, title, _ in RECOMMENDED_WORKFLOW}

    assert {
        "Upload Data",
        "EDA",
        "Cleaning",
        "Transformations",
        "Modeling",
        "Diagnostics",
        "Prediction",
        "Report",
    }.issubset(workflow_titles)
    assert {
        "Explore data",
        "Handle missing values",
        "Transform variables",
        "Fit statistical models",
        "Run ML baselines",
        "Compare models",
        "Diagnose assumptions",
        "Make predictions",
        "Export reports",
    }.issubset(module_titles)
    assert "Upload data" in recommended_titles


def test_home_card_grid_markup_has_no_indented_code_blocks(monkeypatch) -> None:
    rendered = []

    monkeypatch.setattr("src.ui.home_sections.render_html_block", rendered.append)

    _render_cards(WORKFLOW_STEPS[:2])
    _render_simple_cards(MODULE_CARDS[:2])

    assert len(rendered) == 2
    for output in rendered:
        assert output.startswith('<div class="home-card-grid">')
        assert "\n            <div" not in output
        assert output.count('class="home-step-card"') == 2
