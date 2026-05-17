"""Tests for glossary loading and rendering helpers."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from src.ui import term_help
from src.ui.glossary import (
    REQUIRED_ENTRY_FIELDS,
    get_glossary_entry,
    load_glossary_terms,
    search_glossary_terms,
)


REQUIRED_KEYS = {
    "eda",
    "summary_table",
    "missing_value",
    "imputation",
    "outlier",
    "correlation",
    "pearson_correlation",
    "spearman_correlation",
    "p_value",
    "confidence_interval",
    "prediction_interval",
    "coefficient",
    "odds_ratio",
    "r_squared",
    "adjusted_r_squared",
    "rmse",
    "mae",
    "aic",
    "bic",
    "residual",
    "normal_qq_plot",
    "heteroscedasticity",
    "vif",
    "multicollinearity",
    "overfitting",
    "cross_validation",
    "train_test_split",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "confusion_matrix",
    "calibration",
    "feature_importance",
    "permutation_importance",
    "pdp",
    "ice",
    "shap",
    "prediction_probability",
    "threshold",
}


def test_glossary_loads_required_terms() -> None:
    terms = load_glossary_terms()

    assert REQUIRED_KEYS.issubset(terms.keys())


def test_glossary_entries_have_required_fields() -> None:
    terms = load_glossary_terms()

    for entry in terms.values():
        assert REQUIRED_ENTRY_FIELDS.issubset(entry.keys())
        assert isinstance(entry["related_terms"], list)
        assert entry["short_definition"]


def test_missing_glossary_entry_returns_safe_fallback() -> None:
    entry = get_glossary_entry("not_a_real_term")

    assert entry["key"] == "not_a_real_term"
    assert entry["term"] == "Not A Real Term"
    assert "No glossary entry" in entry["short_definition"]


def test_glossary_search_filters_by_query_and_category() -> None:
    results = search_glossary_terms(query="probability", category="prediction")
    keys = {entry["key"] for entry in results}

    assert "prediction_probability" in keys
    assert "threshold" in keys


def test_render_inline_term_uses_safe_html(monkeypatch) -> None:
    rendered = []

    monkeypatch.setattr(term_help, "render_html_block", rendered.append)

    term_help.render_inline_term("p_value", display_text="<p-value>")

    assert "workbench-term" in rendered[0]
    assert "&lt;p-value&gt;" in rendered[0]
    assert "<p-value>" not in rendered[0]


def test_render_glossary_popover_does_not_crash_when_monkeypatched(monkeypatch) -> None:
    calls = []

    @contextmanager
    def fake_popover(label, help=None):
        calls.append((label, help))
        yield

    monkeypatch.setattr(term_help.st, "popover", fake_popover)
    monkeypatch.setattr(term_help.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(term_help.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(term_help.st, "caption", lambda *args, **kwargs: None)

    term_help.render_glossary_popover("roc_auc")

    assert calls
    assert calls[0][0] == "?"


def test_load_glossary_rejects_incomplete_entry(tmp_path) -> None:
    bad_file = tmp_path / "bad_glossary.json"
    bad_file.write_text('[{"key": "bad"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_glossary_terms(bad_file)
