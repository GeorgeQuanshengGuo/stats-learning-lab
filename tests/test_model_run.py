import pandas as pd

from src.core.model_run import (
    MODEL_RUN_FIELDS,
    add_model_run_to_session,
    clear_model_runs,
    create_model_run,
    get_model_runs,
)


def test_create_model_run_contains_required_top_level_fields():
    model_run = create_model_run(
        task_type="regression",
        model_family="statistical",
        model_name="linear_regression_ols",
        target="y",
        features=["x1", "x2"],
        split_config={"test_size": 0.2, "random_state": 42},
    )

    assert list(model_run.keys()) == MODEL_RUN_FIELDS
    assert model_run["run_id"]
    assert model_run["timestamp"]
    assert model_run["target"] == "y"
    assert model_run["features"] == ["x1", "x2"]


def test_create_model_run_converts_dataframes_to_records():
    coefficients = pd.DataFrame({"term": ["const", "x"], "estimate": [1.0, 2.0]})
    summary = pd.DataFrame({"statistic": ["AIC"], "value": [10.0]})

    model_run = create_model_run(
        task_type="regression",
        model_family="statistical",
        model_name="linear_regression_ols",
        target="y",
        features=["x"],
        split_config={"test_size": 0.2},
        statistical_summary=summary,
        coefficient_table=coefficients,
    )

    assert model_run["coefficient_table"] == [
        {"term": "const", "estimate": 1.0},
        {"term": "x", "estimate": 2.0},
    ]
    assert model_run["statistical_summary"] == [{"statistic": "AIC", "value": 10.0}]


def test_model_run_session_helpers_add_get_and_clear():
    clear_model_runs()
    model_run = create_model_run(
        task_type="binary_classification",
        model_family="statistical",
        model_name="binary_logistic_regression_logit",
        target="target",
        features=["x"],
        split_config={"test_size": 0.3},
    )

    add_model_run_to_session(model_run)

    assert get_model_runs() == [model_run]

    clear_model_runs()

    assert get_model_runs() == []
