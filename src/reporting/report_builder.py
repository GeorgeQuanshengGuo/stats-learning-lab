"""Build report data from the current analysis state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.core.model_comparison import (
    build_binary_classification_comparison_table,
    build_count_regression_comparison_table,
    build_multiclass_classification_comparison_table,
    build_ordinal_classification_comparison_table,
    build_regression_comparison_table,
)
from src.eda.missing import build_missing_table
from src.eda.summary import build_summary_table
from src.modeling.diagnostics.overfitting import build_overfitting_diagnostics
from src.reporting.interpretation import explain_model_run


DEFAULT_LIMITATIONS = [
    "This report summarizes the current app session only; saved model runs are not persisted to disk.",
    "Exploratory summaries are descriptive and do not establish causality.",
    "Model metrics reflect the selected data, split settings, preprocessing, and modeling choices.",
    "Feature importance is a predictive diagnostic and does not establish causality.",
    "Machine learning coefficients may be on a transformed or standardized feature scale.",
    "Prediction intervals and uncertainty intervals depend on the model assumptions or interval method used.",
    "Confidence intervals describe uncertainty in an expected model quantity; prediction intervals describe uncertainty for a new observation.",
    "Machine learning empirical uncertainty intervals, such as bootstrap intervals, are not classical confidence intervals.",
    "No unsupported statistical interpretation is generated automatically.",
    "PDF and Word export are not implemented in this report version.",
]

PREDICTIVE_ML_COEFFICIENT_NOTE = (
    "This is a predictive ML coefficient table. It does not include inferential "
    "p-values or confidence intervals. Use the statistical model module for inference."
)

INTERVAL_AND_COEFFICIENT_NOTES = [
    {
        "topic": "Confidence interval",
        "meaning": "A confidence interval summarizes uncertainty for an expected model quantity, such as the expected mean response.",
    },
    {
        "topic": "Prediction interval",
        "meaning": "A prediction interval summarizes uncertainty for a new individual observation and is usually wider than a mean confidence interval.",
    },
    {
        "topic": "ML empirical uncertainty interval",
        "meaning": "An ML empirical interval, such as a bootstrap interval, is produced by a resampling or algorithmic method and is not a classical confidence interval.",
    },
    {
        "topic": "Statistical inference coefficients",
        "meaning": "Statsmodels coefficient tables may include standard errors, p-values, confidence intervals, and inferential summaries.",
    },
    {
        "topic": "Predictive ML coefficients",
        "meaning": PREDICTIVE_ML_COEFFICIENT_NOTE,
    },
]


def build_report_context(
    working_df: pd.DataFrame | None = None,
    schema: pd.DataFrame | None = None,
    missing_table: pd.DataFrame | None = None,
    cleaning_log: list[dict[str, Any]] | None = None,
    transformation_log: list[dict[str, Any]] | None = None,
    model_runs: list[dict[str, Any]] | None = None,
    model_artifacts: dict[str, dict[str, Any]] | list[dict[str, Any]] | None = None,
    prediction_log: list[dict[str, Any]] | None = None,
    title: str = "Analysis Report",
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a session-safe report context dictionary."""
    model_runs = model_runs or []
    generated_at = generated_at or datetime.now(timezone.utc)
    schema_summary = _schema_summary(working_df, schema)
    missing_summary = _missing_summary(working_df, missing_table)

    return {
        "title": title,
        "generated_timestamp": generated_at.isoformat(),
        "dataset_overview": _dataset_overview(working_df),
        "working_dataset_status": _working_dataset_status(
            working_df=working_df,
            cleaning_log=cleaning_log,
            transformation_log=transformation_log,
        ),
        "schema_summary": schema_summary,
        "missing_value_summary": missing_summary,
        "cleaning_log": list(cleaning_log or []),
        "transformation_log": list(transformation_log or []),
        "model_runs_summary": summarize_model_runs(
            model_runs,
            model_artifacts=model_artifacts,
            prediction_log=prediction_log,
        ),
        "model_comparison_summary": build_model_comparison_summary(model_runs),
        "prediction_examples": list(prediction_log or []),
        "interval_and_coefficient_notes": list(INTERVAL_AND_COEFFICIENT_NOTES),
        "limitations": list(DEFAULT_LIMITATIONS),
    }


def summarize_model_runs(
    model_runs: list[dict[str, Any]],
    model_artifacts: dict[str, dict[str, Any]] | list[dict[str, Any]] | None = None,
    prediction_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return compact report rows for saved ModelRun dictionaries."""
    summaries = []
    artifact_lookup = _artifact_lookup(model_artifacts)
    prediction_log = list(prediction_log or [])
    for run in model_runs:
        run_id = run.get("run_id")
        artifact = artifact_lookup.get(run_id, {})
        summary_lookup = _summary_lookup(run.get("statistical_summary"))
        test_metrics = run.get("test_metrics") or {}
        coefficient_table = _records_from_any(
            _first_present(run.get("coefficient_table"), artifact.get("coefficient_table"))
        )
        feature_importance_table = _records_from_any(
            _first_present(
                run.get("feature_importance_table"),
                artifact.get("feature_importance_table"),
                artifact.get("tree_feature_importance_table"),
            )
        )
        formula_latex = _first_present(run.get("formula_latex"), artifact.get("formula_latex")) or {}
        prediction_examples = _prediction_examples_for_run(run_id, prediction_log)
        interpretation_notes = _interpretation_notes(run, artifact, coefficient_table)
        summaries.append(
            {
                "run_id": run_id,
                "model_name": run.get("model_name"),
                "model_family": run.get("model_family"),
                "task_type": run.get("task_type"),
                "target": run.get("target"),
                "features": run.get("features") or [],
                "split_config": run.get("split_config") or {},
                "preprocessing_summary": run.get("preprocessing") or {},
                "train_metrics": run.get("train_metrics") or {},
                "test_metrics": test_metrics,
                "cv_metrics": _cv_metrics(test_metrics),
                "diagnostics_summary": _diagnostics_summary(run, artifact),
                "aic": _first_available(summary_lookup, "AIC", "aic"),
                "bic": _first_available(summary_lookup, "BIC", "bic"),
                "formula_latex": formula_latex,
                "model_formula": _model_formula_parts(formula_latex),
                "estimated_formula": formula_latex.get("estimated") if isinstance(formula_latex, dict) else None,
                "coefficient_table": coefficient_table,
                "coefficient_table_note": _coefficient_table_note(run, coefficient_table),
                "prediction_examples": prediction_examples,
                "interval_method": artifact.get("interval_method"),
                "interval_supported": artifact.get("interval_supported"),
                "prediction_intervals": _prediction_intervals(prediction_examples),
                "decision_tree_rules": artifact.get("tree_rules"),
                "decision_tree_warning": artifact.get("tree_explanation_warning"),
                "feature_importance_table": feature_importance_table,
                "importance_type": run.get("importance_type") or artifact.get("importance_type"),
                "interpretability_outputs": _interpretability_outputs(artifact),
                "interpretation": explain_model_run(run),
                "interpretation_notes": interpretation_notes,
            }
        )
    return summaries


def build_model_comparison_summary(model_runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return report-friendly model comparison tables when model runs exist."""
    if not model_runs:
        return {
            "regression": [],
            "binary_classification": [],
        }

    regression_table = build_regression_comparison_table(model_runs)
    classification_table = build_binary_classification_comparison_table(model_runs)
    return {
        "regression": _records(regression_table),
        "binary_classification": _records(classification_table),
        "multiclass_classification": _records(build_multiclass_classification_comparison_table(model_runs)),
        "ordinal_classification": _records(build_ordinal_classification_comparison_table(model_runs)),
        "count_regression": _records(build_count_regression_comparison_table(model_runs)),
    }


def _dataset_overview(working_df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Return dataset-level report facts."""
    if working_df is None:
        return None
    return {
        "rows": int(working_df.shape[0]),
        "columns": int(working_df.shape[1]),
        "column_names": list(working_df.columns),
    }


def _working_dataset_status(
    working_df: pd.DataFrame | None,
    cleaning_log: list[dict[str, Any]] | None,
    transformation_log: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return reproducibility facts about the current working dataset."""
    if working_df is None:
        return None
    return {
        "dataset_role": "working_df",
        "rows": int(working_df.shape[0]),
        "columns": int(working_df.shape[1]),
        "cleaning_steps": len(cleaning_log or []),
        "transformation_steps": len(transformation_log or []),
        "note": (
            "This report describes the current working dataset. The original uploaded dataset "
            "is kept separately by the app and is not modified by cleaning or transformation actions."
        ),
    }


def _schema_summary(working_df: pd.DataFrame | None, schema: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Use provided schema or compute a current schema summary."""
    if schema is not None and not schema.empty:
        return _records(schema)
    if working_df is not None:
        return _records(build_summary_table(working_df))
    return []


def _missing_summary(
    working_df: pd.DataFrame | None,
    missing_table: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Use provided missing table or compute the current missing summary."""
    if missing_table is not None and not missing_table.empty:
        return _records(missing_table)
    if working_df is not None:
        return _records(build_missing_table(working_df))
    return []


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-like records."""
    if table is None or table.empty:
        return []
    return table.where(pd.notna(table), None).to_dict(orient="records")


def _records_from_any(value: Any) -> list[dict[str, Any]]:
    """Normalize saved table-like values into list-of-records."""
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return _records(value)
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [dict(value)]
    return []


def _first_present(*values: Any) -> Any:
    """Return the first non-empty value without boolean-testing DataFrames."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, pd.DataFrame):
            if not value.empty:
                return value
            continue
        if isinstance(value, (list, tuple, dict, str)):
            if len(value) > 0:
                return value
            continue
        return value
    return None


def _artifact_lookup(
    model_artifacts: dict[str, dict[str, Any]] | list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Return artifacts keyed by run_id."""
    if isinstance(model_artifacts, dict):
        return {
            run_id: artifact
            for run_id, artifact in model_artifacts.items()
            if isinstance(artifact, dict)
        }
    lookup = {}
    for artifact in model_artifacts or []:
        if isinstance(artifact, dict) and artifact.get("run_id"):
            lookup[artifact["run_id"]] = artifact
    return lookup


def _prediction_examples_for_run(
    run_id: str | None,
    prediction_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return prediction log entries for one saved model run."""
    if not run_id:
        return []
    return [dict(entry) for entry in prediction_log if entry.get("run_id") == run_id]


def _prediction_intervals(prediction_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten interval records from prediction examples."""
    rows = []
    for index, example in enumerate(prediction_examples, start=1):
        interval = example.get("interval")
        if not interval:
            continue
        if isinstance(interval, list):
            interval_rows = interval
        elif isinstance(interval, dict):
            interval_rows = [interval]
        else:
            interval_rows = [{"interval": interval}]
        for row in interval_rows:
            if isinstance(row, dict):
                interval_row = dict(row)
                interval_row.setdefault("prediction_example", index)
                rows.append(interval_row)
    return rows


def _model_formula_parts(formula_latex: Any) -> dict[str, Any]:
    """Return symbolic/model-level formula parts without the estimated equation."""
    if not isinstance(formula_latex, dict):
        return {}
    return {
        key: value
        for key, value in formula_latex.items()
        if key not in {"estimated", "note"} and value
    }


def _coefficient_table_note(run: dict[str, Any], coefficient_table: list[dict[str, Any]]) -> str | None:
    """Return the proper coefficient caveat for the model family."""
    if not coefficient_table:
        return None
    if run.get("model_family") == "machine_learning":
        return PREDICTIVE_ML_COEFFICIENT_NOTE
    return "This statistical coefficient table may include inferential quantities produced by statsmodels."


def _diagnostics_summary(run: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Collect compact diagnostics metadata for one model run."""
    diagnostics = {
        "diagnostic_plot_keys": run.get("diagnostic_plot_keys") or [],
        "overfitting_diagnostics": _records(build_overfitting_diagnostics(run)),
    }
    influence_table = _records_from_any(artifact.get("influence_table"))
    if influence_table:
        diagnostics["influence_rows"] = len(influence_table)
        diagnostics["top_influential_rows"] = _records_from_any(artifact.get("top_influential_rows"))
    if artifact.get("tree_explanation_warning"):
        diagnostics["tree_explanation_warning"] = artifact.get("tree_explanation_warning")
    return {key: value for key, value in diagnostics.items() if value not in (None, [], {})}


def _interpretation_notes(
    run: dict[str, Any],
    artifact: dict[str, Any],
    coefficient_table: list[dict[str, Any]],
) -> list[str]:
    """Gather concise interpretation notes for report display."""
    notes = [explain_model_run(run)]
    artifact_notes = artifact.get("coefficient_interpretation_notes") or []
    if isinstance(artifact_notes, str):
        notes.append(artifact_notes)
    else:
        notes.extend(str(note) for note in artifact_notes)
    coefficient_note = _coefficient_table_note(run, coefficient_table)
    if coefficient_note and coefficient_note not in notes:
        notes.append(coefficient_note)
    if artifact.get("tree_explanation_warning"):
        notes.append(str(artifact["tree_explanation_warning"]))
    return [note for note in notes if note]


def _interpretability_outputs(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return saved PDP, ICE, or calibration outputs if an artifact contains them."""
    if not artifact:
        return {}
    candidate_keys = {
        "pdp": ["pdp_table", "partial_dependence_table", "pdp_summary"],
        "ice": ["ice_table", "ice_summary"],
        "calibration": ["calibration_table", "calibration_summary"],
    }
    outputs = {}
    for label, keys in candidate_keys.items():
        for key in keys:
            if artifact.get(key) is not None:
                outputs[label] = _records_from_any(artifact.get(key)) or artifact.get(key)
                break
    if artifact.get("calibration_note"):
        outputs["calibration_note"] = artifact["calibration_note"]
    return outputs


def _cv_metrics(test_metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract cross-validation metrics from a model run's test metrics."""
    return {
        key: value
        for key, value in test_metrics.items()
        if key.startswith("cv_")
    }


def _summary_lookup(summary_rows: Any) -> dict[str, Any]:
    """Convert statistical summary records into a lookup dictionary."""
    if isinstance(summary_rows, dict):
        return summary_rows

    lookup = {}
    for row in summary_rows or []:
        if isinstance(row, dict):
            lookup[row.get("statistic")] = row.get("value")
    return lookup


def _first_available(values: dict[str, Any], *keys: str) -> Any:
    """Return the first available value from a dictionary."""
    for key in keys:
        if key in values:
            return values[key]
    return None
