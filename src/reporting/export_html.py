"""HTML export for analysis reports."""

from __future__ import annotations

import html
import json
from typing import Any


def export_report_html(report_context: dict[str, Any]) -> str:
    """Render a report context dictionary as standalone HTML."""
    body = [
        f"<h1>{_escape(report_context.get('title', 'Analysis Report'))}</h1>",
        f"<p><strong>Generated timestamp:</strong> {_escape(report_context.get('generated_timestamp', 'Unknown'))}</p>",
        _dataset_overview_html(report_context.get("dataset_overview")),
        _record_section_html(
            "Working Dataset Status",
            report_context.get("working_dataset_status"),
            "No working dataset status is available.",
        ),
        _table_section_html("Schema Summary", report_context.get("schema_summary"), "No schema summary is available."),
        _table_section_html(
            "Missing Value Summary",
            report_context.get("missing_value_summary"),
            "No missing value summary is available.",
        ),
        _table_section_html("Cleaning Log", report_context.get("cleaning_log"), "No cleaning operations have been logged."),
        _table_section_html(
            "Transformation Log",
            report_context.get("transformation_log"),
            "No transformation operations have been logged.",
        ),
        _model_runs_html(report_context.get("model_runs_summary") or []),
        _model_comparison_html(report_context.get("model_comparison_summary") or {}),
        _table_section_html(
            "Interval And Coefficient Notes",
            report_context.get("interval_and_coefficient_notes"),
            "No interval or coefficient notes are available.",
        ),
        _limitations_html(report_context.get("limitations") or []),
    ]
    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            "<meta charset=\"utf-8\">",
            f"<title>{_escape(report_context.get('title', 'Analysis Report'))}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 2rem; line-height: 1.45; color: #222; }",
            "table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }",
            "th, td { border: 1px solid #ddd; padding: 0.45rem; text-align: left; vertical-align: top; }",
            "th { background: #f5f5f5; }",
            "code { background: #f7f7f7; padding: 0.1rem 0.25rem; }",
            "</style>",
            "</head>",
            "<body>",
            *body,
            "</body>",
            "</html>",
        ]
    )


def _dataset_overview_html(overview: dict[str, Any] | None) -> str:
    """Render dataset overview as HTML."""
    if not overview:
        return "<h2>Dataset Overview</h2><p>No dataset is currently loaded.</p>"

    columns = _format_value(overview.get("column_names") or [])
    return "\n".join(
        [
            "<h2>Dataset Overview</h2>",
            "<ul>",
            f"<li><strong>Number of rows:</strong> {_escape(overview.get('rows'))}</li>",
            f"<li><strong>Number of columns:</strong> {_escape(overview.get('columns'))}</li>",
            f"<li><strong>Column names:</strong> {_escape(columns)}</li>",
            "</ul>",
        ]
    )


def _model_runs_html(model_runs: list[dict[str, Any]]) -> str:
    """Render saved model run details."""
    if not model_runs:
        return "<h2>Saved Model Runs</h2><p>No saved model runs are available.</p>"

    parts = ["<h2>Saved Model Runs</h2>"]
    for index, run in enumerate(model_runs, start=1):
        parts.extend(
            [
                f"<h3>Model Run {index}: {_escape(run.get('model_name'))}</h3>",
                "<ul>",
                f"<li><strong>Model family:</strong> {_escape(run.get('model_family'))}</li>",
                f"<li><strong>Task type:</strong> {_escape(run.get('task_type'))}</li>",
                f"<li><strong>Target:</strong> {_escape(run.get('target'))}</li>",
                f"<li><strong>Features:</strong> {_escape(_format_value(run.get('features') or []))}</li>",
                f"<li><strong>AIC:</strong> {_escape(_format_value(run.get('aic')))}</li>",
                f"<li><strong>BIC:</strong> {_escape(_format_value(run.get('bic')))}</li>",
                f"<li><strong>Interval method:</strong> {_escape(_format_value(run.get('interval_method')))}</li>",
                "</ul>",
                "<h4>Split Config</h4>",
                _records_to_html_table([run.get("split_config") or {}]),
                "<h4>Preprocessing Summary</h4>",
                _records_to_html_table([run.get("preprocessing_summary") or {}]),
                "<h4>Train Metrics</h4>",
                _records_to_html_table([run.get("train_metrics") or {}]),
                "<h4>Test Metrics</h4>",
                _records_to_html_table([run.get("test_metrics") or {}]),
            ]
        )
        cv_metrics = run.get("cv_metrics") or {}
        if cv_metrics:
            parts.extend(["<h4>CV Metrics</h4>", _records_to_html_table([cv_metrics])])
        parts.extend(
            [
                _diagnostics_html(run),
                _formula_html(run),
                _coefficient_html(run),
                _interpretation_notes_html(run),
                _prediction_examples_html(run),
                _prediction_intervals_html(run),
                _decision_tree_html(run),
                _feature_importance_html(run),
                _pdp_ice_calibration_html(run),
            ]
        )
    return "\n".join(parts)


def _diagnostics_html(run: dict[str, Any]) -> str:
    """Render compact diagnostics metadata."""
    diagnostics = run.get("diagnostics_summary") or {}
    if not diagnostics:
        return "<h4>Diagnostics</h4><p>No diagnostics summary is available for this run.</p>"

    parts = ["<h4>Diagnostics</h4>", "<ul>"]
    if diagnostics.get("diagnostic_plot_keys"):
        parts.append(
            f"<li><strong>Diagnostic plot keys:</strong> {_escape(_format_value(diagnostics.get('diagnostic_plot_keys')))}</li>"
        )
    if diagnostics.get("influence_rows") is not None:
        parts.append(f"<li><strong>Influence diagnostic rows:</strong> {_escape(diagnostics.get('influence_rows'))}</li>")
    if diagnostics.get("tree_explanation_warning"):
        parts.append(f"<li><strong>Tree warning:</strong> {_escape(diagnostics.get('tree_explanation_warning'))}</li>")
    parts.append("</ul>")

    overfitting = diagnostics.get("overfitting_diagnostics") or []
    if overfitting:
        parts.extend(["<h5>Overfitting / CV Stability Diagnostics</h5>", _records_to_html_table(overfitting)])
    top_influential = diagnostics.get("top_influential_rows") or []
    if top_influential:
        parts.extend(["<h5>Top Influential Rows</h5>", _records_to_html_table(top_influential)])
    return "\n".join(parts)


def _model_comparison_html(comparison: dict[str, list[dict[str, Any]]]) -> str:
    """Render model comparison tables."""
    parts = ["<h2>Model Comparison Summary</h2>"]
    sections = [
        ("Regression", comparison.get("regression") or []),
        ("Binary Classification", comparison.get("binary_classification") or []),
        ("Multiclass Classification", comparison.get("multiclass_classification") or []),
        ("Ordinal Classification", comparison.get("ordinal_classification") or []),
        ("Count Regression", comparison.get("count_regression") or []),
    ]
    for title, records in sections:
        parts.append(f"<h3>{_escape(title)}</h3>")
        if records:
            parts.append(_records_to_html_table(records))
        else:
            parts.append(f"<p>No {_escape(title.lower())} comparison table is available.</p>")
    return "\n".join(parts)


def _formula_html(run: dict[str, Any]) -> str:
    """Render formula sections."""
    parts = ["<h4>Model Formula</h4>"]
    model_formula = run.get("model_formula") or {}
    if model_formula:
        items = "".join(
            f"<li><strong>{_escape(label)}:</strong> <code>{_escape(formula)}</code></li>"
            for label, formula in model_formula.items()
        )
        parts.append(f"<ul>{items}</ul>")
    else:
        parts.append("<p>No model formula is available for this run.</p>")

    parts.append("<h4>Estimated Formula</h4>")
    estimated = run.get("estimated_formula")
    if estimated:
        parts.append(f"<p><code>{_escape(estimated)}</code></p>")
        formula_latex = run.get("formula_latex") or {}
        if isinstance(formula_latex, dict) and formula_latex.get("note"):
            parts.append(f"<p><strong>Note:</strong> {_escape(formula_latex['note'])}</p>")
    else:
        parts.append("<p>No estimated formula is available for this run.</p>")
    return "\n".join(parts)


def _coefficient_html(run: dict[str, Any]) -> str:
    """Render coefficient table section."""
    table = run.get("coefficient_table") or []
    parts = ["<h4>Coefficient Table</h4>"]
    if table:
        note = run.get("coefficient_table_note")
        if note:
            parts.append(f"<p><strong>Note:</strong> {_escape(note)}</p>")
        parts.append(_records_to_html_table(table))
    else:
        parts.append("<p>No coefficient table is available for this run.</p>")
    return "\n".join(parts)


def _interpretation_notes_html(run: dict[str, Any]) -> str:
    """Render interpretation notes."""
    notes = run.get("interpretation_notes") or []
    if not notes and run.get("interpretation"):
        notes = [run["interpretation"]]
    if not notes:
        return "<h4>Interpretation Notes</h4><p>No interpretation notes are available for this run.</p>"
    items = "".join(f"<li>{_escape(note)}</li>" for note in notes)
    return f"<h4>Interpretation Notes</h4><ul>{items}</ul>"


def _prediction_examples_html(run: dict[str, Any]) -> str:
    """Render prediction examples."""
    examples = run.get("prediction_examples") or []
    if examples:
        return "<h4>Prediction Examples</h4>" + _records_to_html_table(examples)
    return "<h4>Prediction Examples</h4><p>No prediction examples are logged for this run.</p>"


def _prediction_intervals_html(run: dict[str, Any]) -> str:
    """Render prediction intervals or uncertainty intervals."""
    rows = run.get("prediction_intervals") or []
    if rows:
        return "<h4>Prediction Intervals or Uncertainty Intervals</h4>" + _records_to_html_table(rows)
    interval_method = run.get("interval_method")
    if interval_method:
        return (
            "<h4>Prediction Intervals or Uncertainty Intervals</h4>"
            f"<p>Interval method recorded for this artifact: <code>{_escape(interval_method)}</code>.</p>"
        )
    return (
        "<h4>Prediction Intervals or Uncertainty Intervals</h4>"
        "<p>No prediction interval or uncertainty interval output is available for this run.</p>"
    )


def _decision_tree_html(run: dict[str, Any]) -> str:
    """Render decision tree rules."""
    rules = run.get("decision_tree_rules")
    if not rules:
        return "<h4>Decision Tree Rules</h4><p>No decision tree rules are available for this run.</p>"
    warning = run.get("decision_tree_warning")
    warning_html = f"<p><strong>Note:</strong> {_escape(warning)}</p>" if warning else ""
    return f"<h4>Decision Tree Rules</h4>{warning_html}<pre>{_escape(rules)}</pre>"


def _feature_importance_html(run: dict[str, Any]) -> str:
    """Render feature importance."""
    table = run.get("feature_importance_table") or []
    parts = ["<h4>Feature Importance</h4>"]
    if table:
        importance_type = run.get("importance_type")
        if importance_type:
            parts.append(f"<p><strong>Importance type:</strong> {_escape(importance_type)}</p>")
        parts.append(_records_to_html_table(table))
    else:
        parts.append("<p>No feature importance table is available for this run.</p>")
    return "\n".join(parts)


def _pdp_ice_calibration_html(run: dict[str, Any]) -> str:
    """Render saved PDP, ICE, and calibration outputs."""
    outputs = run.get("interpretability_outputs") or {}
    parts = ["<h4>PDP / ICE / Calibration</h4>"]
    if not outputs:
        parts.append("<p>No saved PDP, ICE, or calibration output is available for this run.</p>")
        return "\n".join(parts)
    for label, output in outputs.items():
        parts.append(f"<h5>{_escape(label.replace('_', ' ').title())}</h5>")
        if isinstance(output, list) and all(isinstance(row, dict) for row in output):
            parts.append(_records_to_html_table(output))
        else:
            parts.append(f"<p>{_escape(output)}</p>")
    return "\n".join(parts)


def _limitations_html(limitations: list[str]) -> str:
    """Render limitations as HTML."""
    if not limitations:
        return "<h2>Limitations</h2><p>No limitations were provided.</p>"
    items = "\n".join(f"<li>{_escape(limitation)}</li>" for limitation in limitations)
    return f"<h2>Limitations</h2><ul>{items}</ul>"


def _table_section_html(title: str, records: list[dict[str, Any]] | None, empty_text: str) -> str:
    """Render a table section as HTML."""
    if records:
        content = _records_to_html_table(records)
    else:
        content = f"<p>{_escape(empty_text)}</p>"
    return f"<h2>{_escape(title)}</h2>\n{content}"


def _record_section_html(title: str, record: dict[str, Any] | None, empty_text: str) -> str:
    """Render one dictionary as an HTML section."""
    records = [record] if record else None
    return _table_section_html(title, records, empty_text)


def _records_to_html_table(records: list[dict[str, Any]]) -> str:
    """Render records as an HTML table."""
    if not records:
        return "<p>No data available.</p>"

    columns = _ordered_columns(records)
    header = "".join(f"<th>{_escape(column)}</th>" for column in columns)
    rows = []
    for record in records:
        cells = "".join(f"<td>{_escape(_format_value(record.get(column)))}</td>" for column in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _ordered_columns(records: list[dict[str, Any]]) -> list[str]:
    """Return columns in first-seen order across records."""
    columns = []
    for record in records:
        for key in record.keys():
            if key not in columns:
                columns.append(key)
    return columns


def _format_value(value: Any) -> str:
    """Format a value for HTML export."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _escape(value: Any) -> str:
    """Escape one value for HTML output."""
    return html.escape(_format_value(value))
