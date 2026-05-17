"""Home page sections for the Streamlit learning app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ui.components import (
    render_empty_state,
    render_info_card,
    render_metric_card,
    render_section_header,
)
from src.ui.layout import escape_html, render_html_block
from src.ui.theme import PROJECT_ROOT


ILLUSTRATION_DIR = PROJECT_ROOT / "assets" / "illustrations"
LOGO_PATH = ILLUSTRATION_DIR / "workbench_logo.svg"
WORKFLOW_ILLUSTRATION_PATH = ILLUSTRATION_DIR / "workflow.svg"
EDA_PREVIEW_PATH = ILLUSTRATION_DIR / "eda_preview.svg"
MODELING_PREVIEW_PATH = ILLUSTRATION_DIR / "modeling_preview.svg"
REPORT_PREVIEW_PATH = ILLUSTRATION_DIR / "report_preview.svg"

WORKFLOW_STEPS = [
    {
        "icon": "UP",
        "title": "Upload Data",
        "explanation": "Load a CSV or Excel dataset into a protected original copy and a working copy.",
        "user_controls": "Choose the file.",
        "app_automates": "Creates safe session copies.",
    },
    {
        "icon": "EDA",
        "title": "EDA",
        "explanation": "Inspect variables, missingness, distributions, correlations, and relationships.",
        "user_controls": "Choose variables and views.",
        "app_automates": "Builds summaries and charts.",
    },
    {
        "icon": "CL",
        "title": "Cleaning",
        "explanation": "Handle missing values only after previewing the effect.",
        "user_controls": "Confirm every cleaning operation.",
        "app_automates": "Logs each operation.",
    },
    {
        "icon": "TR",
        "title": "Transformations",
        "explanation": "Create new transformed variables without overwriting source columns.",
        "user_controls": "Pick methods and names.",
        "app_automates": "Checks mathematical eligibility.",
    },
    {
        "icon": "MD",
        "title": "Modeling",
        "explanation": "Fit statistical models and machine learning baselines.",
        "user_controls": "Choose target, features, models, and options.",
        "app_automates": "Calculates metrics and stores runs.",
    },
    {
        "icon": "DG",
        "title": "Diagnostics",
        "explanation": "Review overfitting, assumptions, multicollinearity, and influence signals.",
        "user_controls": "Decide what risks matter.",
        "app_automates": "Computes diagnostic summaries.",
    },
    {
        "icon": "PR",
        "title": "Prediction",
        "explanation": "Use saved fitted models to predict new user-entered observations.",
        "user_controls": "Enter feature values.",
        "app_automates": "Routes inputs through saved pipelines.",
    },
    {
        "icon": "RP",
        "title": "Report",
        "explanation": "Export a transparent summary of the current analysis state.",
        "user_controls": "Review what to share.",
        "app_automates": "Assembles Markdown and HTML.",
    },
]

MODULE_CARDS = [
    ("Explore data", "Summary tables, missingness, relationships, outlier checks, and target-aware EDA."),
    ("Handle missing values", "Preview row drops or imputations before changing the working dataset."),
    ("Transform variables", "Create log, square-root, Box-Cox, Yeo-Johnson, interaction, ratio, and PCA features."),
    ("Fit statistical models", "Run interpretable models with coefficient tables, formulas, diagnostics, and inference."),
    ("Run ML baselines", "Train sklearn Pipeline baselines with leakage-safe preprocessing, metrics, CV, and tuning."),
    ("Compare models", "Review saved runs without mixing incompatible regression and classification metrics."),
    ("Diagnose assumptions", "Inspect overfitting, multicollinearity, residuals, influence, and calibration."),
    ("Make predictions", "Use saved artifacts to predict from raw user-entered feature values."),
    ("Export reports", "Generate Markdown and HTML reports with formulas, metrics, logs, and interpretations."),
]

RECOMMENDED_WORKFLOW = [
    ("Step 1", "Upload data", "Start with a CSV or Excel file."),
    ("Step 2", "Inspect EDA", "Understand types, missingness, distributions, and relationships."),
    ("Step 3", "Clean data", "Preview and confirm cleaning choices on the working dataset."),
    ("Step 4", "Transform variables", "Create new variables when diagnostics or modeling goals call for them."),
    ("Step 5", "Choose target and model", "Select target, features, and modeling family deliberately."),
    ("Step 6", "Check diagnostics", "Look for overfitting, assumption issues, and influential observations."),
    ("Step 7", "Compare and report", "Save model runs, compare them, then export a transparent report."),
]

def get_logo_path() -> Path:
    """Return the local logo path used by ``st.logo``."""
    return LOGO_PATH


def _render_section_intro(title: str, body: str) -> None:
    """Render a consistent home section heading."""
    render_html_block(
        f"""
        <div class="home-section">
          <h2 class="home-section-title">{escape_html(title)}</h2>
          <p class="home-section-copy">{escape_html(body)}</p>
        </div>
        """
    )


def _render_cards(cards: list[dict[str, str]]) -> None:
    """Render a grid of home cards."""
    card_html = []
    for card in cards:
        card_html.append(
            (
                '<div class="home-step-card">'
                f'<div class="home-card-icon">{escape_html(card["icon"])}</div>'
                f'<div class="home-card-title">{escape_html(card["title"])}</div>'
                f'<div class="home-card-copy">{escape_html(card["explanation"])}</div>'
                f'<div class="home-card-meta"><strong>User controls:</strong> {escape_html(card["user_controls"])}</div>'
                f'<div class="home-card-meta"><strong>App automates:</strong> {escape_html(card["app_automates"])}</div>'
                "</div>"
            )
        )
    render_html_block(f'<div class="home-card-grid">{"".join(card_html)}</div>')


def _render_simple_cards(items: list[tuple[str, str]]) -> None:
    """Render compact module cards."""
    card_html = []
    for title, body in items:
        card_html.append(
            (
                '<div class="home-step-card">'
                f'<div class="home-card-title">{escape_html(title)}</div>'
                f'<div class="home-card-copy">{escape_html(body)}</div>'
                "</div>"
            )
        )
    render_html_block(f'<div class="home-card-grid">{"".join(card_html)}</div>')


def build_home_status_summary(session_state: dict[str, Any]) -> dict[str, Any]:
    """Build a display-ready status summary without mutating session state."""
    working_df = session_state.get("working_df")
    original_df = session_state.get("original_df")
    model_runs = session_state.get("model_runs", []) or []
    file_name = session_state.get("uploaded_file_name") or "Uploaded dataset"

    if working_df is None:
        return {
            "has_dataset": False,
            "dataset_name": None,
            "rows": 0,
            "columns": 0,
            "working_status": "No working dataset yet",
            "model_runs": len(model_runs),
        }

    if not isinstance(working_df, pd.DataFrame):
        return {
            "has_dataset": False,
            "dataset_name": None,
            "rows": 0,
            "columns": 0,
            "working_status": "Working dataset is unavailable",
            "model_runs": len(model_runs),
        }

    row_count, column_count = working_df.shape
    original_shape = original_df.shape if isinstance(original_df, pd.DataFrame) else None
    if original_shape and original_shape != working_df.shape:
        status = "Working dataset differs from the original after confirmed operations"
    else:
        status = "Working dataset matches the original shape"

    return {
        "has_dataset": True,
        "dataset_name": file_name,
        "rows": row_count,
        "columns": column_count,
        "working_status": status,
        "model_runs": len(model_runs),
    }


def render_home_hero() -> None:
    """Render the landing-page hero."""
    left, right = st.columns([1.08, 0.92], vertical_alignment="center")
    with left:
        render_html_block(
            """
            <div class="workbench-hero">
              <div class="home-hero-kicker">A learning-first statistics and ML practice app</div>
              <h1 class="home-hero-title">Stats Learning Lab</h1>
              <p class="home-hero-subtitle">A small guided app for practicing data analysis with familiar Python libraries.</p>
              <p class="home-hero-copy">
                Upload a dataset, explore it, try cleaning and transformations, fit selected statistical or machine learning models,
                inspect diagnostics, and export a simple report while learning what each step means.
              </p>
            </div>
            """
        )
        cta_left, cta_right = st.columns([0.45, 0.55])
        with cta_left:
            st.page_link("pages/01_upload_data.py", label="Start with data upload")
        with cta_right:
            st.markdown("[View workflow](#workflow)")
    with right:
        st.image(str(WORKFLOW_ILLUSTRATION_PATH), use_container_width=True)


def render_quick_status_panel(session_state: dict[str, Any]) -> None:
    """Render loaded-dataset status or a friendly empty state."""
    _render_section_intro(
        "Quick status",
        "See whether the app already has an active working dataset and saved model runs.",
    )
    status = build_home_status_summary(session_state)
    if not status["has_dataset"]:
        render_empty_state(
            title="No dataset loaded yet",
            body="Start by uploading a CSV or Excel file. The app will keep the original dataset immutable and create a separate working copy.",
            action_label="Go to Upload Data",
            action_page="pages/01_upload_data.py",
            icon="Data:",
        )
        return

    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        render_metric_card("Dataset", status["dataset_name"])
    with col_2:
        render_metric_card("Rows", f"{status['rows']:,}")
    with col_3:
        render_metric_card("Columns", f"{status['columns']:,}")
    with col_4:
        render_metric_card("Saved model runs", status["model_runs"])
    render_info_card("Working dataset status", status["working_status"], color="primary")


def render_visual_workflow_section() -> None:
    """Render the workflow overview and step cards."""
    render_html_block('<div id="workflow"></div>')
    _render_section_intro(
        "Visual workflow",
        "The app follows a deliberate path from data upload to report export. Each stage keeps the user in control.",
    )
    st.image(str(WORKFLOW_ILLUSTRATION_PATH), use_container_width=True)
    _render_cards(WORKFLOW_STEPS)


def render_platform_identity_section() -> None:
    """Render what the learning app is and is not."""
    _render_section_intro(
        "What this app is",
        "A modest learning tool for practicing applied statistics and machine learning workflows.",
    )
    left, middle, right = st.columns(3)
    with left:
        render_info_card(
            "A study helper",
            "The app wraps common Python functions into a guided workflow so you can focus on learning the analysis steps.",
            color="primary",
        )
    with middle:
        render_info_card(
            "Not black-box AutoML",
            "You choose targets, features, model families, preprocessing options, and interpretation context.",
            color="warning",
        )
    with right:
        render_info_card(
            "Designed for learning",
            "Glossary entries, warnings, formulas, and notes explain what outputs do and do not mean.",
            color="success",
        )


def render_module_cards_section() -> None:
    """Render the module capability cards."""
    _render_section_intro(
        "What you can do",
        "Each module supports one part of the analysis workflow, from exploration to final report.",
    )
    _render_simple_cards(MODULE_CARDS)


def render_recommended_workflow_section() -> None:
    """Render a practical recommended sequence."""
    _render_section_intro(
        "Recommended workflow",
        "A simple path for most analyses. You can revisit earlier steps whenever a diagnostic suggests a change.",
    )
    cards = [
        {
            "icon": step,
            "title": title,
            "explanation": body,
            "user_controls": "Review and decide.",
            "app_automates": "Provides the relevant summaries or outputs.",
        }
        for step, title, body in RECOMMENDED_WORKFLOW
    ]
    _render_cards(cards)


def render_home_page(session_state: dict[str, Any]) -> None:
    """Render the complete polished home page."""
    render_home_hero()
    render_quick_status_panel(session_state)
    render_visual_workflow_section()
    render_platform_identity_section()
    render_module_cards_section()
    render_recommended_workflow_section()
    render_section_header("First step", "Upload data, then move through the sidebar workflow.")
    st.page_link("pages/01_upload_data.py", label="Start with data upload")
