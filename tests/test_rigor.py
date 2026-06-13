import pandas as pd

from src.core.analysis_plan import create_analysis_plan
from src.core.rigor import (
    build_data_readiness,
    build_model_readiness,
    build_reproducibility_manifest,
    build_rigor_checklist,
)


def _messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Subject ID": ["A001", "A002", "A003", "A004", "A005"],
            "target": [1, 0, 0, 0, 0],
            "x1": [1.0, 2.0, None, 4.0, 100.0],
            "constant": [1, 1, 1, 1, 1],
            "event_date": pd.date_range("2026-01-01", periods=5),
            "category": ["a", "b", "c", "d", "e"],
        }
    )


def test_build_data_readiness_flags_common_data_risks() -> None:
    readiness = build_data_readiness(_messy_df(), target_column="target")

    assert readiness["row_count"] == 5
    assert readiness["column_count"] == 6
    assert "constant" in readiness["constant_columns"]
    assert "Subject ID" in readiness["id_like_columns"]
    assert "event_date" in readiness["datetime_columns"]
    assert readiness["target_missing_count"] == 0
    assert readiness["target_unique_count"] == 2


def test_build_model_readiness_flags_class_imbalance_and_id_like_features() -> None:
    readiness = build_model_readiness(
        _messy_df(),
        target_column="target",
        feature_columns=["Subject ID", "x1", "event_date"],
        task_type="binary_classification",
        split_strategy="random",
    )

    assert readiness["target_detected_type"] == "binary"
    assert readiness["sample_size_after_drop_missing"] == 4
    assert readiness["leakage_column_warning"] is True
    assert readiness["time_split_warning"] is True
    assert readiness["rare_class_warning"]
    assert any("ID-like" in warning for warning in readiness["warnings"])


def test_build_model_readiness_flags_count_target_issues() -> None:
    df = pd.DataFrame(
        {
            "count_y": [0, 0, 1.5, -1, 4],
            "x": [1, 2, 3, 4, 5],
        }
    )

    readiness = build_model_readiness(
        df,
        target_column="count_y",
        feature_columns=["x"],
        task_type="count_regression",
    )

    assert any("nonnegative" in warning for warning in readiness["warnings"])
    assert any("integer-like" in warning for warning in readiness["warnings"])
    assert any("zero" in warning.lower() for warning in readiness["warnings"])


def test_build_rigor_checklist_marks_missing_plan_as_attention() -> None:
    checklist = build_rigor_checklist(
        analysis_plan=None,
        working_df=_messy_df(),
        cleaning_log=[],
        transformation_log=[],
        model_runs=[],
        prediction_log=[],
    )

    status_by_item = {row["item"]: row["status"] for row in checklist}

    assert status_by_item["Analysis plan exists"] == "needs_attention"
    assert status_by_item["Dataset has been uploaded"] == "completed"


def test_build_rigor_checklist_marks_recorded_plan_items_completed() -> None:
    plan = create_analysis_plan(
        research_question="What predicts target?",
        analysis_goal="prediction",
        target_variable="target",
        known_id_columns=["Subject ID"],
        interpretation_boundaries="Associations only.",
        train_test_strategy="random 80/20 split",
        now="2026-06-13T00:00:00+00:00",
    )

    checklist = build_rigor_checklist(
        analysis_plan=plan,
        working_df=_messy_df(),
        cleaning_log=[{"operation": "fill_numeric_median"}],
        transformation_log=[{"method": "log1p"}],
        model_runs=[{"test_metrics": {"test_rmse": 1.0}}],
        prediction_log=[],
    )
    status_by_item = {row["item"]: row["status"] for row in checklist}

    assert status_by_item["Research question is written"] == "completed"
    assert status_by_item["Analysis goal is clear"] == "completed"
    assert status_by_item["Cleaning decisions are logged"] == "completed"
    assert status_by_item["Transformations are documented"] == "completed"
    assert status_by_item["Interpretation boundaries are stated"] == "completed"


def test_build_reproducibility_manifest_contains_core_fields() -> None:
    plan = create_analysis_plan(
        analysis_goal="prediction",
        target_variable="target",
        candidate_features=["x1"],
        excluded_columns=["Subject ID"],
        now="2026-06-13T00:00:00+00:00",
    )
    model_runs = [
        {
            "model_name": "Ridge",
            "model_family": "machine_learning",
            "target": "target",
            "features": ["x1"],
            "split_config": {"test_size": 0.2, "random_state": 42},
            "preprocessing": {"cross_validation": {"folds": 5}},
        }
    ]

    manifest = build_reproducibility_manifest(
        working_df=_messy_df(),
        original_df=_messy_df(),
        uploaded_file_name="synthetic.csv",
        analysis_plan=plan,
        cleaning_log=[{}],
        transformation_log=[{}],
        model_runs=model_runs,
        generated_at="2026-06-13T00:00:00+00:00",
        app_version="test-version",
    )

    assert manifest["app_version"] == "test-version"
    assert manifest["uploaded_file_name"] == "synthetic.csv"
    assert manifest["analysis_goal"] == "prediction"
    assert manifest["target"] == "target"
    assert manifest["features"] == ["x1"]
    assert manifest["random_state"] == 42
    assert manifest["cv_folds"] == 5
    assert manifest["known_limitations"]
