"""Session helpers for fitted model objects.

ModelRun dictionaries stay lightweight so they can be displayed, exported, and
tested easily. Fitted model objects can be large and are only useful inside the
current Streamlit session, so they live in this separate registry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st


MODEL_ARTIFACTS_KEY = "model_artifacts"


def initialize_model_artifacts() -> dict[str, dict[str, Any]]:
    """Create the model artifact registry if it does not already exist."""
    return st.session_state.setdefault(MODEL_ARTIFACTS_KEY, {})


def save_model_artifact(run_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
    """Save one fitted model artifact under the matching ModelRun id."""
    if not run_id:
        raise ValueError("A run_id is required to save a model artifact.")

    artifact_run_id = artifact.get("run_id")
    if artifact_run_id is not None and artifact_run_id != run_id:
        raise ValueError("Artifact run_id does not match the provided run_id.")

    registry = initialize_model_artifacts()
    saved_artifact = dict(artifact)
    saved_artifact["run_id"] = run_id
    saved_artifact.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    registry[run_id] = saved_artifact
    return saved_artifact


def get_model_artifact(run_id: str) -> dict[str, Any] | None:
    """Return a saved model artifact, or None when no artifact exists."""
    registry = initialize_model_artifacts()
    return registry.get(run_id)


def list_model_artifacts() -> list[dict[str, Any]]:
    """Return all saved model artifacts for the current session."""
    registry = initialize_model_artifacts()
    return list(registry.values())


def remove_model_artifact(run_id: str) -> dict[str, Any] | None:
    """Remove one model artifact and return it if it existed."""
    registry = initialize_model_artifacts()
    return registry.pop(run_id, None)


def clear_model_artifacts() -> None:
    """Remove all fitted model artifacts from the current session."""
    st.session_state[MODEL_ARTIFACTS_KEY] = {}
