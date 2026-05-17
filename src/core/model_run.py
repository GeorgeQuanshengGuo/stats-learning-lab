"""Helpers for storing model run results in one consistent structure."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.core.model_artifacts import clear_model_artifacts


MODEL_RUN_FIELDS = [
    "run_id",
    "timestamp",
    "task_type",
    "model_family",
    "model_name",
    "target",
    "features",
    "split_config",
    "preprocessing",
    "statistical_summary",
    "train_metrics",
    "test_metrics",
    "coefficient_table",
    "diagnostic_plot_keys",
    "feature_importance_table",
    "importance_type",
    "formula_latex",
    "notes",
]


def create_model_run(
    task_type: str,
    model_family: str,
    model_name: str,
    target: str,
    features: list[str],
    split_config: dict[str, Any],
    preprocessing: dict[str, Any] | None = None,
    statistical_summary: Any | None = None,
    train_metrics: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
    coefficient_table: Any | None = None,
    diagnostic_plot_keys: list[str] | None = None,
    feature_importance_table: Any | None = None,
    importance_type: str | None = None,
    formula_latex: dict[str, str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Create a consistent, session-safe model run dictionary."""
    return {
        "run_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type,
        "model_family": model_family,
        "model_name": model_name,
        "target": target,
        "features": list(features),
        "split_config": split_config,
        "preprocessing": preprocessing or {},
        "statistical_summary": _to_records(statistical_summary),
        "train_metrics": train_metrics or {},
        "test_metrics": test_metrics or {},
        "coefficient_table": _to_records(coefficient_table),
        "diagnostic_plot_keys": diagnostic_plot_keys or [],
        "feature_importance_table": _to_records(feature_importance_table),
        "importance_type": importance_type,
        "formula_latex": formula_latex,
        "notes": notes,
    }


def add_model_run_to_session(model_run: dict[str, Any]) -> None:
    """Append one model run to Streamlit session state."""
    st.session_state.setdefault("model_runs", [])
    st.session_state["model_runs"].append(model_run)


def get_model_runs() -> list[dict[str, Any]]:
    """Return saved model runs from Streamlit session state."""
    return st.session_state.setdefault("model_runs", [])


def clear_model_runs() -> None:
    """Clear saved model runs from Streamlit session state."""
    st.session_state["model_runs"] = []
    clear_model_artifacts()


def _to_records(value: Any) -> Any:
    """Convert DataFrames to records so model runs are easier to display/export."""
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value
