"""Tests for chart metadata and chart card rendering."""

from __future__ import annotations

from contextlib import contextmanager

import plotly.graph_objects as go

from src.ui import chart_cards
from src.visualization.chart_metadata import ChartMetadata, create_chart_metadata


def test_create_chart_metadata_returns_required_fields() -> None:
    metadata = create_chart_metadata(
        title="Histogram of age",
        chart_type="histogram",
        source_page="EDA",
        variables_used=["age"],
        method="Histogram",
        sample_size=10,
        missing_handling="Dropped missing values",
    )

    assert isinstance(metadata, ChartMetadata)
    assert metadata.chart_id.startswith("chart_")
    assert metadata.title == "Histogram of age"
    assert metadata.variables_used == ["age"]
    assert metadata.created_at


def test_render_chart_context_outputs_metadata(monkeypatch) -> None:
    rendered = []
    metadata = create_chart_metadata(
        title="Chart",
        chart_type="histogram",
        source_page="EDA",
        variables_used=["x"],
        method="Histogram",
        sample_size=5,
        missing_handling="dropna",
    )

    monkeypatch.setattr(chart_cards, "render_html_block", rendered.append)

    chart_cards.render_chart_context(metadata)

    assert "Variables" in rendered[0]
    assert "Histogram" in rendered[0]
    assert "5" in rendered[0]


def test_render_chart_warning_uses_default_warning(monkeypatch) -> None:
    rendered = []
    metadata = create_chart_metadata(
        title="Importance",
        chart_type="feature_importance",
        source_page="Machine Learning",
    )

    monkeypatch.setattr(chart_cards, "render_html_block", rendered.append)

    chart_cards.render_chart_warning(metadata)

    assert "Feature importance is not causal importance" in rendered[0]


def test_render_chart_card_calls_plotly_chart(monkeypatch) -> None:
    calls = {"plotly": 0, "download": 0}
    metadata = create_chart_metadata(
        title="Histogram",
        chart_type="histogram",
        source_page="EDA",
        variables_used=["x"],
    )
    fig = go.Figure()

    @contextmanager
    def fake_column():
        yield

    monkeypatch.setattr(chart_cards, "render_html_block", lambda markup: None)
    monkeypatch.setattr(chart_cards, "render_badge", lambda *args, **kwargs: None)
    monkeypatch.setattr(chart_cards.st, "container", lambda *args, **kwargs: fake_column())
    monkeypatch.setattr(chart_cards.st, "columns", lambda *args, **kwargs: [fake_column(), fake_column(), fake_column()])
    monkeypatch.setattr(chart_cards.st, "plotly_chart", lambda *args, **kwargs: calls.__setitem__("plotly", calls["plotly"] + 1))
    monkeypatch.setattr(chart_cards.st, "download_button", lambda *args, **kwargs: calls.__setitem__("download", calls["download"] + 1))
    monkeypatch.setattr(chart_cards.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(chart_cards.st, "expander", lambda *args, **kwargs: fake_column())
    monkeypatch.setattr(chart_cards.st, "write", lambda *args, **kwargs: None)

    chart_cards.render_chart_card(fig, metadata)

    assert calls["plotly"] == 1
    assert calls["download"] == 1


def test_render_plotly_chart_card_builds_metadata(monkeypatch) -> None:
    captured = {}
    fig = go.Figure()

    def fake_render_chart_card(chart, metadata, **kwargs):
        captured["chart"] = chart
        captured["metadata"] = metadata
        captured["kwargs"] = kwargs

    monkeypatch.setattr(chart_cards, "render_chart_card", fake_render_chart_card)

    chart_cards.render_plotly_chart_card(
        fig,
        title="Correlation",
        chart_type="correlation_heatmap",
        source_page="EDA",
        variables_used=["x", "y"],
        method="Pearson",
        sample_size=20,
    )

    assert captured["chart"] is fig
    assert captured["metadata"].title == "Correlation"
    assert captured["metadata"].variables_used == ["x", "y"]
    assert captured["metadata"].sample_size == 20
