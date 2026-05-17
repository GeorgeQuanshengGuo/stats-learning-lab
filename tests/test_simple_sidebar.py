"""Tests for the simple sidebar navigation helpers."""

from __future__ import annotations

import pandas as pd

from src.ui.simple_sidebar import NAV_GROUPS, build_simple_sidebar_status, workspace_persistence_enabled


def test_navigation_labels_are_title_case() -> None:
    labels = [label for _, links in NAV_GROUPS for label, _ in links]

    assert "Upload Data" in labels
    assert "Machine Learning" in labels
    assert "Model Comparison" in labels
    assert all(label[0].isupper() for label in labels)


def test_navigation_groups_are_simple_and_complete() -> None:
    group_names = [group_name for group_name, _ in NAV_GROUPS]

    assert group_names == ["Start", "Data", "Models", "Output", "Help"]


def test_simple_sidebar_status_without_dataset() -> None:
    status = build_simple_sidebar_status({})

    assert status["has_dataset"] is False
    assert status["rows"] == 0
    assert status["columns"] == 0


def test_simple_sidebar_status_with_dataset_and_logs() -> None:
    status = build_simple_sidebar_status(
        {
            "working_df": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "cleaning_log": [{}],
            "transformation_log": [{}, {}],
            "model_runs": [{}, {}, {}],
        }
    )

    assert status["has_dataset"] is True
    assert status["rows"] == 2
    assert status["columns"] == 2
    assert "cleaning_steps" not in status
    assert "transformation_steps" not in status
    assert "model_runs" not in status


def test_workspace_persistence_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("STATS_LAB_ENABLE_WORKSPACE_SNAPSHOTS", raising=False)

    assert workspace_persistence_enabled() is False


def test_workspace_persistence_can_be_enabled_explicitly(monkeypatch) -> None:
    monkeypatch.setenv("STATS_LAB_ENABLE_WORKSPACE_SNAPSHOTS", "true")

    assert workspace_persistence_enabled() is True
