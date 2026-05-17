import pandas as pd
import pytest
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.core.model_artifacts import get_model_artifact, save_model_artifact
from src.core.model_run import add_model_run_to_session, create_model_run, get_model_runs
from src.core.state import (
    apply_cleaning_result,
    apply_transformation_result,
    init_session_state,
    reset_working_data,
    set_uploaded_data,
)
from src.data.cleaning import fill_numeric_median
from src.data.transformations import apply_log1p_transform
from src.modeling.machine_learning.regression import run_ml_regression_models
from src.modeling.preprocessing import build_preprocessing_pipeline
from src.prediction.prediction_service import PredictionError, predict_from_model_run
from src.reporting.report_builder import build_report_context


@pytest.fixture(autouse=True)
def _clean_session_state():
    st.session_state.clear()
    init_session_state()
    yield
    st.session_state.clear()


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "x": [1.0, None, 3.0, 4.0, 5.0, 6.0],
            "group": ["A", "A", "B", "B", "C", "C"],
        }
    )


def test_upload_stores_independent_original_and_working_copies():
    uploaded = _sample_df()

    set_uploaded_data(uploaded)
    uploaded.loc[0, "target"] = -999
    st.session_state["working_df"].loc[1, "target"] = -111

    assert st.session_state["original_df"].loc[0, "target"] == 10.0
    assert st.session_state["original_df"].loc[1, "target"] == 12.0
    assert st.session_state["working_df"].loc[0, "target"] == 10.0


def test_cleaning_operation_and_state_apply_do_not_mutate_original_data():
    set_uploaded_data(_sample_df())
    original_before = st.session_state["original_df"].copy(deep=True)
    working_before = st.session_state["working_df"].copy(deep=True)

    cleaned_df, log_entry = fill_numeric_median(st.session_state["working_df"], "x")
    pd.testing.assert_frame_equal(st.session_state["working_df"], working_before)

    apply_cleaning_result(cleaned_df, log_entry)

    pd.testing.assert_frame_equal(st.session_state["original_df"], original_before)
    assert st.session_state["working_df"]["x"].isna().sum() == 0
    assert len(st.session_state["cleaning_log"]) == 1
    assert st.session_state["cleaning_log"][0]["operation"] == "fill_numeric_median"


def test_transformation_operation_and_state_apply_do_not_mutate_original_data():
    set_uploaded_data(_sample_df())
    original_before = st.session_state["original_df"].copy(deep=True)
    working_before = st.session_state["working_df"].copy(deep=True)

    transformed_df, log_entry = apply_log1p_transform(st.session_state["working_df"], "target")
    pd.testing.assert_frame_equal(st.session_state["working_df"], working_before)

    apply_transformation_result(transformed_df, log_entry)

    pd.testing.assert_frame_equal(st.session_state["original_df"], original_before)
    assert "target_log1p" in st.session_state["working_df"].columns
    assert "target_log1p" not in st.session_state["original_df"].columns
    assert len(st.session_state["transformation_log"]) == 1
    assert st.session_state["transformation_log"][0]["method"] == "log1p"


def test_model_fitting_records_run_and_artifact_without_mutating_session_data():
    set_uploaded_data(_sample_df())
    original_before = st.session_state["original_df"].copy(deep=True)
    working_before = st.session_state["working_df"].copy(deep=True)

    result = run_ml_regression_models(
        st.session_state["working_df"],
        target_column="target",
        feature_columns=["x", "group"],
        selected_models=["Linear Regression"],
        random_state=7,
    )[0]
    add_model_run_to_session(result["model_run"])
    artifact = get_model_artifact(result["model_run"]["run_id"])

    pd.testing.assert_frame_equal(st.session_state["original_df"], original_before)
    pd.testing.assert_frame_equal(st.session_state["working_df"], working_before)
    assert get_model_runs()[0]["run_id"] == result["model_run"]["run_id"]
    assert artifact is not None
    assert artifact["run_id"] == result["model_run"]["run_id"]


def test_prediction_does_not_mutate_original_or_working_data():
    set_uploaded_data(_sample_df())
    data = st.session_state["working_df"].copy(deep=True)
    preprocessing = build_preprocessing_pipeline(data, ["x", "group"])
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", LinearRegression())])
    pipeline.fit(data[["x", "group"]], data["target"])
    model_run = create_model_run(
        task_type="regression",
        model_family="machine_learning",
        model_name="Linear Regression",
        target="target",
        features=["x", "group"],
        split_config={"test_size": 0.2},
    )
    artifact = {
        "run_id": model_run["run_id"],
        "task_type": "regression",
        "model_family": "machine_learning",
        "model_name": "Linear Regression",
        "fitted_pipeline": pipeline,
        "fitted_model": pipeline.named_steps["model"],
        "features": ["x", "group"],
        "prediction_supported": True,
    }
    original_before = st.session_state["original_df"].copy(deep=True)
    working_before = st.session_state["working_df"].copy(deep=True)

    prediction = predict_from_model_run(model_run, artifact, {"x": 3.0, "group": "B"})

    assert prediction["task_type"] == "regression"
    pd.testing.assert_frame_equal(st.session_state["original_df"], original_before)
    pd.testing.assert_frame_equal(st.session_state["working_df"], working_before)


def test_report_generation_does_not_mutate_original_or_working_data():
    set_uploaded_data(_sample_df())
    cleaning_log = [{"operation": "fill_numeric_median", "column": "x"}]
    transformation_log = [{"method": "log1p", "source_columns": ["target"], "new_column": "target_log1p"}]
    model_run = create_model_run(
        task_type="regression",
        model_family="machine_learning",
        model_name="Linear Regression",
        target="target",
        features=["x", "group"],
        split_config={"test_size": 0.2},
    )
    original_before = st.session_state["original_df"].copy(deep=True)
    working_before = st.session_state["working_df"].copy(deep=True)

    context = build_report_context(
        working_df=st.session_state["working_df"],
        cleaning_log=cleaning_log,
        transformation_log=transformation_log,
        model_runs=[model_run],
    )

    assert context["dataset_overview"]["rows"] == len(working_before)
    pd.testing.assert_frame_equal(st.session_state["original_df"], original_before)
    pd.testing.assert_frame_equal(st.session_state["working_df"], working_before)


def test_reset_working_dataset_restores_original_and_clears_edit_logs_and_artifacts():
    set_uploaded_data(_sample_df())
    transformed_df, transform_log = apply_log1p_transform(st.session_state["working_df"], "target")
    apply_transformation_result(transformed_df, transform_log)
    cleaned_df, cleaning_log = fill_numeric_median(st.session_state["working_df"], "x")
    apply_cleaning_result(cleaned_df, cleaning_log)
    save_model_artifact("old-run", {"model_name": "Old model", "prediction_supported": True})

    reset_working_data()

    pd.testing.assert_frame_equal(st.session_state["working_df"], st.session_state["original_df"])
    assert st.session_state["cleaning_log"] == []
    assert st.session_state["transformation_log"] == []
    assert get_model_artifact("old-run") is None


def test_old_model_run_without_artifact_is_handled_gracefully():
    model_run = create_model_run(
        task_type="regression",
        model_family="machine_learning",
        model_name="Old Linear Regression",
        target="target",
        features=["x"],
        split_config={"test_size": 0.2},
    )
    add_model_run_to_session(model_run)

    assert get_model_runs()[0]["run_id"] == model_run["run_id"]
    assert get_model_artifact(model_run["run_id"]) is None
    with pytest.raises(PredictionError, match="No usable fitted model artifact"):
        predict_from_model_run(model_run, None, {"x": 1.0})
