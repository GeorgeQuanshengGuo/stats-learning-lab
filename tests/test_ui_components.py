"""Tests for reusable Streamlit UI components."""

from __future__ import annotations

from src.ui import components, layout, page_templates


def test_escape_html_protects_component_markup() -> None:
    text = '<script>alert("x")</script>'

    assert layout.escape_html(text) == "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"


def test_render_html_block_dedents_markup(monkeypatch) -> None:
    rendered = []

    monkeypatch.setattr(layout.st, "html", rendered.append)

    layout.render_html_block(
        """
            <div class="example">
              Text
            </div>
        """
    )

    assert rendered[0].startswith('<div class="example">')
    assert "Text" in rendered[0]
    assert "\n            <" not in rendered[0]


def test_render_badge_uses_expected_css_class(monkeypatch) -> None:
    rendered = []

    monkeypatch.setattr(components, "render_html_block", rendered.append)

    components.render_badge("Ready", color="success")

    assert "workbench-badge" in rendered[0]
    assert "workbench-badge-success" in rendered[0]
    assert "Ready" in rendered[0]


def test_render_page_header_escapes_title_and_tags(monkeypatch) -> None:
    rendered = []

    monkeypatch.setattr(components, "render_html_block", rendered.append)

    components.render_page_header(
        title="<EDA>",
        subtitle="Explore data",
        tags=["safe", "<tag>"],
        help_text="Use before modeling.",
    )

    output = rendered[0]
    assert "workbench-page-header" in output
    assert "&lt;EDA&gt;" in output
    assert "&lt;tag&gt;" in output
    assert "<EDA>" not in output


def test_render_info_card_supports_color_and_badge(monkeypatch) -> None:
    rendered = []

    monkeypatch.setattr(components, "render_html_block", rendered.append)

    components.render_info_card(
        title="Model result",
        body="Saved for comparison.",
        badge="Saved",
        color="success",
    )

    output = rendered[0]
    assert "workbench-card-success" in output
    assert "workbench-badge-success" in output
    assert "Saved for comparison." in output


def test_render_warning_card_uses_severity_class(monkeypatch) -> None:
    rendered = []

    monkeypatch.setattr(components, "render_html_block", rendered.append)

    components.render_warning_card("Careful", "Check assumptions.", severity="danger")

    assert "workbench-card-danger" in rendered[0]
    assert "Check assumptions." in rendered[0]


def test_render_metric_card_escapes_value_and_helper(monkeypatch) -> None:
    rendered = []

    monkeypatch.setattr(components, "render_html_block", rendered.append)

    components.render_metric_card(
        label="Rows",
        value="<100>",
        helper_text="After filtering",
        delta="+5",
        status="success",
    )

    output = rendered[0]
    assert "workbench-metric-value" in output
    assert "workbench-card-success" in output
    assert "&lt;100&gt;" in output
    assert "+5" in output


def test_render_workflow_stepper_marks_current_and_completed(monkeypatch) -> None:
    rendered = []

    monkeypatch.setattr(components, "render_html_block", rendered.append)

    components.render_workflow_stepper(
        steps=["Upload", "EDA", "Model"],
        current_step="EDA",
        completed_steps=[0],
    )

    output = rendered[0]
    assert "workbench-workflow" in output
    assert "workbench-workflow-step-active" in output
    assert "workbench-workflow-step-complete" in output
    assert "EDA" in output


def test_empty_state_renders_optional_page_link(monkeypatch) -> None:
    rendered = []
    links = []

    monkeypatch.setattr(components, "render_html_block", rendered.append)
    monkeypatch.setattr(components.st, "page_link", lambda page, label: links.append((page, label)))

    components.render_empty_state(
        title="No data",
        body="Upload first.",
        action_label="Upload Data",
        action_page="pages/01_upload_data.py",
    )

    assert "workbench-empty-state" in rendered[0]
    assert links == [("pages/01_upload_data.py", "Upload Data")]


def test_standard_page_intro_composes_expected_sections(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(page_templates, "render_page_header", lambda **kwargs: calls.append(("header", kwargs)))
    monkeypatch.setattr(
        page_templates,
        "render_two_column_panel",
        lambda **kwargs: calls.append(("two_column", kwargs)),
    )
    monkeypatch.setattr(
        page_templates,
        "render_next_step_panel",
        lambda next_step: calls.append(("next", next_step)),
    )

    page_templates.render_standard_page_intro(
        title="EDA",
        purpose="Explore data.",
        when_to_use="Before modeling.",
        does_not_do="Fit models.",
        next_step="Clean missing values.",
    )

    assert [name for name, _ in calls] == ["header", "two_column", "next"]
