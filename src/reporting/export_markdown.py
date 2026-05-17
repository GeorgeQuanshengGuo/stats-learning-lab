"""Markdown export for analysis reports."""

from __future__ import annotations

import json
from typing import Any


def export_report_markdown(report_context: dict[str, Any]) -> str:
    """Render a report context dictionary as Markdown."""
    lines = [
        f"# {report_context.get('title', 'Analysis Report')}",
        "",
        f"Generated timestamp: {report_context.get('generated_timestamp', 'Unknown')}",
        "",
    ]

    lines.extend(_dataset_overview_section(report_context.get("dataset_overview")))
    lines.extend(
        _record_section(
            "Working Dataset Status",
            report_context.get("working_dataset_status"),
            "No working dataset status is available.",
        )
    )
    lines.extend(_table_section("Schema Summary", report_context.get("schema_summary"), "No schema summary is available."))
    lines.extend(
        _table_section(
            "Missing Value Summary",
            report_context.get("missing_value_summary"),
            "No missing value summary is available.",
        )
    )
    lines.extend(_table_section("Cleaning Log", report_context.get("cleaning_log"), "No cleaning operations have been logged."))
    lines.extend(
        _table_section(
            "Transformation Log",
            report_context.get("transformation_log"),
            "No transformation operations have been logged.",
        )
    )
    lines.extend(_model_runs_section(report_context.get("model_runs_summary") or []))
    lines.extend(_model_comparison_section(report_context.get("model_comparison_summary") or {}))
    lines.extend(
        _table_section(
            "Interval And Coefficient Notes",
            report_context.get("interval_and_coefficient_notes"),
            "No interval or coefficient notes are available.",
        )
    )
    lines.extend(_limitations_section(report_context.get("limitations") or []))

    return "\n".join(lines).strip() + "\n"


def _dataset_overview_section(overview: dict[str, Any] | None) -> list[str]:
    """Render dataset overview as Markdown."""
    lines = ["## Dataset Overview", ""]
    if not overview:
        return lines + ["No dataset is currently loaded.", ""]

    column_names = overview.get("column_names") or []
    lines.extend(
        [
            f"- Number of rows: {overview.get('rows')}",
            f"- Number of columns: {overview.get('columns')}",
            f"- Column names: {_format_value(column_names)}",
            "",
        ]
    )
    return lines


def _model_runs_section(model_runs: list[dict[str, Any]]) -> list[str]:
    """Render saved model run details."""
    lines = ["## Saved Model Runs", ""]
    if not model_runs:
        return lines + ["No saved model runs are available.", ""]

    for index, run in enumerate(model_runs, start=1):
        lines.extend(
            [
                f"### Model Run {index}: {run.get('model_name')}",
                "",
                f"- Model family: {run.get('model_family')}",
                f"- Task type: {run.get('task_type')}",
                f"- Target: {run.get('target')}",
                f"- Features: {_format_value(run.get('features') or [])}",
                f"- AIC: {_format_value(run.get('aic'))}",
                f"- BIC: {_format_value(run.get('bic'))}",
                f"- Interval method: {_format_value(run.get('interval_method'))}",
                "",
                "Split config",
                "",
                _records_to_markdown_table([run.get("split_config") or {}]),
                "",
                "Preprocessing summary",
                "",
                _records_to_markdown_table([run.get("preprocessing_summary") or {}]),
                "",
                "Train metrics",
                "",
                _records_to_markdown_table([run.get("train_metrics") or {}]),
                "",
                "Test metrics",
                "",
                _records_to_markdown_table([run.get("test_metrics") or {}]),
                "",
            ]
        )
        cv_metrics = run.get("cv_metrics") or {}
        if cv_metrics:
            lines.extend(["CV metrics", "", _records_to_markdown_table([cv_metrics]), ""])

        lines.extend(_diagnostics_section(run))
        lines.extend(_formula_sections(run))
        lines.extend(_coefficient_section(run))
        lines.extend(_interpretation_notes_section(run))
        lines.extend(_prediction_examples_section(run))
        lines.extend(_prediction_interval_section(run))
        lines.extend(_decision_tree_section(run))
        lines.extend(_feature_importance_section(run))
        lines.extend(_pdp_ice_calibration_section(run))

    return lines


def _diagnostics_section(run: dict[str, Any]) -> list[str]:
    """Render compact diagnostics metadata for a saved model run."""
    lines = ["#### Diagnostics", ""]
    diagnostics = run.get("diagnostics_summary") or {}
    if not diagnostics:
        return lines + ["No diagnostics summary is available for this run.", ""]

    diagnostic_plot_keys = diagnostics.get("diagnostic_plot_keys") or []
    if diagnostic_plot_keys:
        lines.extend([f"- Diagnostic plot keys: {_format_value(diagnostic_plot_keys)}"])
    if diagnostics.get("influence_rows") is not None:
        lines.extend([f"- Influence diagnostic rows: {_format_value(diagnostics.get('influence_rows'))}"])
    if diagnostics.get("tree_explanation_warning"):
        lines.extend([f"- Tree warning: {_format_value(diagnostics.get('tree_explanation_warning'))}"])
    lines.append("")

    overfitting = diagnostics.get("overfitting_diagnostics") or []
    if overfitting:
        lines.extend(["Overfitting / CV stability diagnostics", "", _records_to_markdown_table(overfitting), ""])
    top_influential = diagnostics.get("top_influential_rows") or []
    if top_influential:
        lines.extend(["Top influential rows", "", _records_to_markdown_table(top_influential), ""])
    return lines


def _model_comparison_section(comparison: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Render model comparison tables."""
    lines = ["## Model Comparison Summary", ""]
    sections = [
        ("Regression", comparison.get("regression") or []),
        ("Binary Classification", comparison.get("binary_classification") or []),
        ("Multiclass Classification", comparison.get("multiclass_classification") or []),
        ("Ordinal Classification", comparison.get("ordinal_classification") or []),
        ("Count Regression", comparison.get("count_regression") or []),
    ]

    for title, records in sections:
        lines.extend([f"### {title}", ""])
        if records:
            lines.extend([_records_to_markdown_table(records), ""])
        else:
            lines.extend([f"No {title.lower()} comparison table is available.", ""])

    return lines


def _formula_sections(run: dict[str, Any]) -> list[str]:
    """Render symbolic/model and estimated formulas."""
    lines = ["#### Model Formula", ""]
    model_formula = run.get("model_formula") or {}
    if model_formula:
        for label, formula in model_formula.items():
            lines.extend([f"- {label}: `{formula}`"])
        lines.append("")
    else:
        lines.extend(["No model formula is available for this run.", ""])

    lines.extend(["#### Estimated Formula", ""])
    estimated = run.get("estimated_formula")
    formula_note = (run.get("formula_latex") or {}).get("note") if isinstance(run.get("formula_latex"), dict) else None
    if estimated:
        lines.extend([f"`{estimated}`", ""])
        if formula_note:
            lines.extend([f"Note: {formula_note}", ""])
    else:
        lines.extend(["No estimated formula is available for this run.", ""])
    return lines


def _coefficient_section(run: dict[str, Any]) -> list[str]:
    """Render coefficient table and caveat."""
    lines = ["#### Coefficient Table", ""]
    coefficient_table = run.get("coefficient_table") or []
    if coefficient_table:
        note = run.get("coefficient_table_note")
        if note:
            lines.extend([f"Note: {note}", ""])
        lines.extend([_records_to_markdown_table(coefficient_table), ""])
    else:
        lines.extend(["No coefficient table is available for this run.", ""])
    return lines


def _interpretation_notes_section(run: dict[str, Any]) -> list[str]:
    """Render rule-based interpretation notes."""
    notes = run.get("interpretation_notes") or []
    if not notes and run.get("interpretation"):
        notes = [run["interpretation"]]
    lines = ["#### Interpretation Notes", ""]
    if notes:
        if run.get("interpretation"):
            lines.extend([f"Interpretation: {_format_value(run.get('interpretation'))}", ""])
        lines.extend([f"- {_format_value(note)}" for note in notes])
        lines.append("")
    else:
        lines.extend(["No interpretation notes are available for this run.", ""])
    return lines


def _prediction_examples_section(run: dict[str, Any]) -> list[str]:
    """Render prediction examples for one model run."""
    lines = ["#### Prediction Examples", ""]
    examples = run.get("prediction_examples") or []
    if examples:
        lines.extend([_records_to_markdown_table(examples), ""])
    else:
        lines.extend(["No prediction examples are logged for this run.", ""])
    return lines


def _prediction_interval_section(run: dict[str, Any]) -> list[str]:
    """Render statistical prediction intervals or ML uncertainty intervals."""
    lines = ["#### Prediction Intervals or Uncertainty Intervals", ""]
    interval_rows = run.get("prediction_intervals") or []
    if interval_rows:
        lines.extend([_records_to_markdown_table(interval_rows), ""])
        return lines

    interval_method = run.get("interval_method")
    if interval_method:
        lines.extend([f"Interval method recorded for this artifact: `{interval_method}`.", ""])
    else:
        lines.extend(["No prediction interval or uncertainty interval output is available for this run.", ""])
    return lines


def _decision_tree_section(run: dict[str, Any]) -> list[str]:
    """Render decision tree rules when available."""
    lines = ["#### Decision Tree Rules", ""]
    rules = run.get("decision_tree_rules")
    if rules:
        warning = run.get("decision_tree_warning")
        if warning:
            lines.extend([f"Note: {warning}", ""])
        lines.extend(["```text", str(rules), "```", ""])
    else:
        lines.extend(["No decision tree rules are available for this run.", ""])
    return lines


def _feature_importance_section(run: dict[str, Any]) -> list[str]:
    """Render feature importance when available."""
    lines = ["#### Feature Importance", ""]
    table = run.get("feature_importance_table") or []
    if table:
        importance_type = run.get("importance_type")
        if importance_type:
            lines.extend([f"Importance type: `{importance_type}`", ""])
        lines.extend([_records_to_markdown_table(table), ""])
    else:
        lines.extend(["No feature importance table is available for this run.", ""])
    return lines


def _pdp_ice_calibration_section(run: dict[str, Any]) -> list[str]:
    """Render saved PDP, ICE, or calibration output if present."""
    lines = ["#### PDP / ICE / Calibration", ""]
    outputs = run.get("interpretability_outputs") or {}
    if not outputs:
        return lines + ["No saved PDP, ICE, or calibration output is available for this run.", ""]

    for label, output in outputs.items():
        title = label.replace("_", " ").title()
        lines.extend([f"##### {title}", ""])
        if isinstance(output, list) and all(isinstance(row, dict) for row in output):
            lines.extend([_records_to_markdown_table(output), ""])
        else:
            lines.extend([_format_value(output), ""])
    return lines


def _limitations_section(limitations: list[str]) -> list[str]:
    """Render report limitations."""
    lines = ["## Limitations", ""]
    if not limitations:
        return lines + ["No limitations were provided.", ""]
    return lines + [f"- {limitation}" for limitation in limitations] + [""]


def _table_section(title: str, records: list[dict[str, Any]] | None, empty_text: str) -> list[str]:
    """Render a list-of-records table section."""
    lines = [f"## {title}", ""]
    if records:
        lines.extend([_records_to_markdown_table(records), ""])
    else:
        lines.extend([empty_text, ""])
    return lines


def _record_section(title: str, record: dict[str, Any] | None, empty_text: str) -> list[str]:
    """Render one dictionary as a one-row table section."""
    records = [record] if record else None
    return _table_section(title, records, empty_text)


def _records_to_markdown_table(records: list[dict[str, Any]]) -> str:
    """Render records as a simple Markdown table."""
    if not records:
        return "No data available."

    columns = _ordered_columns(records)
    header = "| " + " | ".join(_escape_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in records:
        rows.append("| " + " | ".join(_escape_cell(_format_value(record.get(column))) for column in columns) + " |")

    return "\n".join([header, separator, *rows])


def _ordered_columns(records: list[dict[str, Any]]) -> list[str]:
    """Return columns in first-seen order across records."""
    columns = []
    for record in records:
        for key in record.keys():
            if key not in columns:
                columns.append(key)
    return columns


def _format_value(value: Any) -> str:
    """Format a value for text export."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _escape_cell(value: Any) -> str:
    """Escape Markdown table pipe characters."""
    return _format_value(value).replace("|", "\\|").replace("\n", " ")
