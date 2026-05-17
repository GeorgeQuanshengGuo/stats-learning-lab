"""Tests for local workspace snapshot helpers."""

from __future__ import annotations

import pandas as pd

from src.core.workspace_snapshot import (
    create_workspace_snapshot,
    has_meaningful_workspace,
    load_workspace_snapshot,
    restore_workspace_snapshot,
    save_workspace_snapshot,
    summarize_workspace_snapshot,
)


def test_has_meaningful_workspace_detects_dataset_or_results():
    assert has_meaningful_workspace({}) is False
    assert has_meaningful_workspace({"working_df": pd.DataFrame({"x": [1, 2]})}) is True
    assert has_meaningful_workspace({"model_runs": [{"model_name": "Linear Regression"}]}) is True


def test_save_and_load_workspace_snapshot_round_trips_dataframe(tmp_path):
    path = tmp_path / "workspace.pkl"
    session_state = {
        "original_df": pd.DataFrame({"x": [1, 2]}),
        "working_df": pd.DataFrame({"x": [1, 3]}),
        "uploaded_file_name": "sample.csv",
        "model_runs": [{"model_name": "Ridge"}],
        "ml_task_type": "Regression",
    }

    saved = save_workspace_snapshot(session_state, path=path, source="manual")
    loaded = load_workspace_snapshot(path)

    assert saved["source"] == "manual"
    assert loaded["state"]["uploaded_file_name"] == "sample.csv"
    assert loaded["state"]["ml_task_type"] == "Regression"
    pd.testing.assert_frame_equal(loaded["state"]["working_df"], session_state["working_df"])


def test_restore_workspace_snapshot_replaces_session_state():
    snapshot = create_workspace_snapshot(
        {
            "working_df": pd.DataFrame({"x": [10]}),
            "model_runs": [{"model_name": "Random Forest"}],
            "pca_n_components": 2,
        }
    )
    session_state = {
        "working_df": pd.DataFrame({"old": [1]}),
        "temporary_key": "remove me",
        "workspace_autosave_enabled": True,
    }

    restore_workspace_snapshot(session_state, snapshot)

    assert "temporary_key" not in session_state
    assert session_state["workspace_autosave_enabled"] is True
    assert session_state["workspace_restore_dismissed"] is True
    assert session_state["pca_n_components"] == 2
    assert session_state["model_runs"][0]["model_name"] == "Random Forest"


def test_unpickleable_values_are_skipped():
    session_state = {
        "working_df": pd.DataFrame({"x": [1]}),
        "ml_task_type": "Regression",
        "ml_bad_value": lambda value: value,
    }

    snapshot = create_workspace_snapshot(session_state)

    assert "ml_task_type" in snapshot["state"]
    assert "ml_bad_value" not in snapshot["state"]
    assert "ml_bad_value" in snapshot["skipped_keys"]


def test_summarize_workspace_snapshot_is_display_ready():
    snapshot = create_workspace_snapshot(
        {
            "working_df": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
            "uploaded_file_name": "data.csv",
            "model_runs": [{}, {}],
        }
    )

    summary = summarize_workspace_snapshot(snapshot)

    assert summary["dataset_name"] == "data.csv"
    assert summary["rows"] == 2
    assert summary["columns"] == 2
    assert summary["model_runs"] == 2
