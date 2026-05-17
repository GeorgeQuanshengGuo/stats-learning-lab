from src.core.model_artifacts import (
    clear_model_artifacts,
    get_model_artifact,
    initialize_model_artifacts,
    list_model_artifacts,
    remove_model_artifact,
    save_model_artifact,
)
from src.core.model_run import clear_model_runs, create_model_run


class DummyFittedModel:
    """Tiny stand-in for a fitted model object."""


def test_initialize_model_artifacts_creates_registry():
    clear_model_artifacts()

    registry = initialize_model_artifacts()

    assert registry == {}


def test_save_and_retrieve_model_artifact():
    clear_model_artifacts()
    fitted_model = DummyFittedModel()

    saved = save_model_artifact(
        "run-1",
        {
            "model_name": "Linear Regression",
            "model_family": "machine_learning",
            "task_type": "regression",
            "fitted_model": fitted_model,
            "target": "y",
            "features": ["x"],
            "prediction_supported": True,
        },
    )

    retrieved = get_model_artifact("run-1")

    assert saved["run_id"] == "run-1"
    assert saved["created_at"]
    assert retrieved["fitted_model"] is fitted_model
    assert retrieved["target"] == "y"


def test_list_and_remove_model_artifacts():
    clear_model_artifacts()
    save_model_artifact("run-1", {"model_name": "Model A"})
    save_model_artifact("run-2", {"model_name": "Model B"})

    artifacts = list_model_artifacts()
    removed = remove_model_artifact("run-1")

    assert len(artifacts) == 2
    assert removed["model_name"] == "Model A"
    assert get_model_artifact("run-1") is None
    assert get_model_artifact("run-2")["model_name"] == "Model B"


def test_clear_model_artifacts_removes_all_artifacts():
    clear_model_artifacts()
    save_model_artifact("run-1", {"model_name": "Model A"})

    clear_model_artifacts()

    assert list_model_artifacts() == []


def test_missing_run_id_returns_none():
    clear_model_artifacts()

    assert get_model_artifact("does-not-exist") is None


def test_model_run_and_artifact_can_share_same_run_id():
    clear_model_artifacts()
    model_run = create_model_run(
        task_type="regression",
        model_family="machine_learning",
        model_name="Linear Regression",
        target="y",
        features=["x"],
        split_config={"test_size": 0.2},
    )

    save_model_artifact(
        model_run["run_id"],
        {
            "model_name": model_run["model_name"],
            "model_family": model_run["model_family"],
            "task_type": model_run["task_type"],
            "target": model_run["target"],
            "features": model_run["features"],
            "fitted_model": DummyFittedModel(),
        },
    )

    artifact = get_model_artifact(model_run["run_id"])

    assert artifact["run_id"] == model_run["run_id"]
    assert artifact["target"] == model_run["target"]
    assert "fitted_model" not in model_run


def test_clear_model_runs_also_clears_model_artifacts():
    clear_model_artifacts()
    save_model_artifact("run-1", {"model_name": "Model A"})

    clear_model_runs()

    assert list_model_artifacts() == []
