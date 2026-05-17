"""Save and restore local workspace snapshots.

Snapshots are local files created by this app. They are meant to protect users
from accidental refreshes, not to exchange untrusted model files.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.ui.theme import PROJECT_ROOT


SNAPSHOT_VERSION = 1
WORKSPACE_DIR = PROJECT_ROOT / "outputs" / "workspaces"
AUTOSAVE_SNAPSHOT_PATH = WORKSPACE_DIR / "autosave_workspace.pkl"
MANUAL_SNAPSHOT_PATH = WORKSPACE_DIR / "manual_workspace.pkl"
AUTOSAVE_PREFERENCE_PATH = WORKSPACE_DIR / "autosave_enabled.pkl"

CORE_SESSION_KEYS = [
    "original_df",
    "working_df",
    "uploaded_file_name",
    "schema",
    "cleaning_log",
    "transformation_log",
    "model_runs",
    "model_artifacts",
    "prediction_log",
    "pca_artifacts",
    "clustering_artifacts",
    "anomaly_artifacts",
    "report_chart_items",
]

SAVE_KEY_PREFIXES = (
    "latest_",
    "ml_",
    "pca_",
    "clustering_",
    "anomaly_",
    "prediction_",
    "statistical_",
    "eda_",
    "cleaning_",
    "transformation_",
)

WORKSPACE_INTERNAL_KEYS = {
    "workspace_restore_dismissed",
    "workspace_last_save_message",
}


def has_meaningful_workspace(session_state: dict[str, Any]) -> bool:
    """Return True when there is analysis state worth saving."""
    return any(
        [
            isinstance(session_state.get("working_df"), pd.DataFrame),
            bool(session_state.get("cleaning_log")),
            bool(session_state.get("transformation_log")),
            bool(session_state.get("model_runs")),
            bool(session_state.get("prediction_log")),
            bool(session_state.get("latest_pca_result")),
            bool(session_state.get("latest_clustering_result")),
            bool(session_state.get("latest_anomaly_result")),
        ]
    )


def create_workspace_snapshot(session_state: dict[str, Any], source: str = "manual") -> dict[str, Any]:
    """Build a pickle-safe snapshot from selected session state keys."""
    saved_state = {}
    skipped_keys = []
    for key in _keys_to_save(session_state):
        value = session_state.get(key)
        if _is_pickleable(value):
            saved_state[key] = value
        else:
            skipped_keys.append(key)

    return {
        "version": SNAPSHOT_VERSION,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": saved_state,
        "skipped_keys": skipped_keys,
    }


def save_workspace_snapshot(
    session_state: dict[str, Any],
    path: Path = MANUAL_SNAPSHOT_PATH,
    source: str = "manual",
) -> dict[str, Any]:
    """Save the current workspace snapshot to a local file."""
    snapshot = create_workspace_snapshot(session_state, source=source)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(snapshot, file)
    return snapshot


def load_workspace_snapshot(path: Path = AUTOSAVE_SNAPSHOT_PATH) -> dict[str, Any] | None:
    """Load a local workspace snapshot, returning None when it does not exist."""
    if not path.exists():
        return None
    with path.open("rb") as file:
        snapshot = pickle.load(file)
    if not isinstance(snapshot, dict) or "state" not in snapshot:
        raise ValueError("Saved workspace file is not a valid workspace snapshot.")
    return snapshot


def restore_workspace_snapshot(session_state: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Replace the current session values with values from a snapshot."""
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("state"), dict):
        raise ValueError("Saved workspace file is not a valid workspace snapshot.")

    preserve_autosave = session_state.get("workspace_autosave_enabled", load_autosave_preference())
    clear_workspace_session(session_state, preserve_keys={"workspace_autosave_enabled"})
    session_state.update(snapshot["state"])
    session_state["workspace_autosave_enabled"] = preserve_autosave
    session_state["workspace_restore_dismissed"] = True


def clear_workspace_session(session_state: dict[str, Any], preserve_keys: set[str] | None = None) -> None:
    """Clear the current Streamlit session state, preserving optional keys."""
    preserve_keys = preserve_keys or set()
    preserved = {key: session_state[key] for key in preserve_keys if key in session_state}
    for key in list(session_state.keys()):
        del session_state[key]
    session_state.update(preserved)


def snapshot_exists(path: Path = AUTOSAVE_SNAPSHOT_PATH) -> bool:
    """Return True when a snapshot file exists."""
    return path.exists()


def summarize_workspace_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Return a small summary suitable for UI display."""
    if not snapshot:
        return {
            "created_at": None,
            "dataset_name": None,
            "rows": 0,
            "columns": 0,
            "model_runs": 0,
        }

    state = snapshot.get("state") or {}
    working_df = state.get("working_df")
    rows = 0
    columns = 0
    if isinstance(working_df, pd.DataFrame):
        rows, columns = working_df.shape

    return {
        "created_at": snapshot.get("created_at"),
        "dataset_name": state.get("uploaded_file_name") or "Saved workspace",
        "rows": rows,
        "columns": columns,
        "model_runs": len(state.get("model_runs") or []),
    }


def set_autosave_preference(enabled: bool) -> None:
    """Persist the user's autosave preference locally."""
    AUTOSAVE_PREFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUTOSAVE_PREFERENCE_PATH.open("wb") as file:
        pickle.dump(bool(enabled), file)


def load_autosave_preference() -> bool:
    """Load the local autosave preference."""
    if not AUTOSAVE_PREFERENCE_PATH.exists():
        return False
    try:
        with AUTOSAVE_PREFERENCE_PATH.open("rb") as file:
            return bool(pickle.load(file))
    except Exception:
        return False


def maybe_autosave_workspace(session_state: dict[str, Any]) -> dict[str, Any] | None:
    """Autosave only when the user enabled it and useful state exists."""
    if not session_state.get("workspace_autosave_enabled", load_autosave_preference()):
        return None
    if not has_meaningful_workspace(session_state):
        return None
    return save_workspace_snapshot(
        session_state,
        path=AUTOSAVE_SNAPSHOT_PATH,
        source="autosave",
    )


def _keys_to_save(session_state: dict[str, Any]) -> list[str]:
    """Return session keys that belong in a workspace snapshot."""
    keys = set(CORE_SESSION_KEYS)
    for key in session_state:
        if key in WORKSPACE_INTERNAL_KEYS:
            continue
        if key.startswith(SAVE_KEY_PREFIXES):
            keys.add(key)
    return [key for key in keys if key in session_state]


def _is_pickleable(value: Any) -> bool:
    """Return True when a value can be written to a snapshot file."""
    try:
        pickle.dumps(value)
    except Exception:
        return False
    return True
