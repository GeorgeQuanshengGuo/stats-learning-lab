from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.data.loader import load_uploaded_file
from src.data.type_detector import detect_variable_types
from src.eda.outliers import run_outlier_detection
from src.eda.summary import build_summary_table
from src.modeling.diagnostics.multicollinearity import compute_condition_number, compute_vif_table
from src.modeling.machine_learning.classification import run_ml_binary_classification_models
from src.modeling.machine_learning.regression import run_ml_regression_models
from src.modeling.preprocessing import build_preprocessing_pipeline
from src.modeling.statistical.logistic_regression import run_logistic_regression
from src.prediction.prediction_service import PredictionError, predict_from_model_run
from src.reporting.report_builder import build_report_context


class NamedBytesIO(BytesIO):
    """Small uploaded-file stand-in with a Streamlit-like name attribute."""

    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


def test_schema_handles_all_missing_constant_id_dates_and_unusual_names():
    data = pd.DataFrame(
        {
            "all_missing": [None, None, None, None, None],
            "constant": [7, 7, 7, 7, 7],
            "user_id": ["id-1", "id-2", "id-3", "id-4", "id-5"],
            "日期": ["2026-01-01", "2026-01-02", "bad date", "2026-01-04", "2026-01-05"],
            "score value (%)": ["1.2", "2.4", "3.6", "4.8", "6.0"],
        }
    )

    schema = detect_variable_types(data)
    detected = dict(zip(schema["variable"], schema["detected_type"]))
    summary = build_summary_table(data, schema=schema)

    assert detected["all_missing"] == "constant"
    assert detected["constant"] == "constant"
    assert detected["user_id"] == "id_like"
    assert detected["日期"] == "datetime"
    assert detected["score value (%)"] == "continuous_numeric"
    assert summary.loc[summary["variable"] == "all_missing", "missing_count"].iloc[0] == 5
    assert "score value (%)" in summary["variable"].tolist()


def test_duplicate_column_names_from_csv_are_disambiguated_by_loader():
    uploaded = NamedBytesIO(b"value,value,target\n1,2,10\n3,4,20\n", "duplicate_columns.csv")

    loaded = load_uploaded_file(uploaded)

    assert loaded.columns.tolist() == ["value", "value.1", "target"]
    assert loaded.shape == (2, 3)


def test_tiny_dataset_fails_with_clear_modeling_message():
    data = pd.DataFrame({"target": [1.0, 2.0], "x": [1.0, 2.0]})

    with pytest.raises(ValueError, match="At least 3 rows"):
        run_ml_regression_models(
            data,
            target_column="target",
            feature_columns=["x"],
            selected_models=["Linear Regression"],
        )


def test_binary_classification_target_with_one_class_fails_clearly():
    data = pd.DataFrame({"target": ["yes"] * 6, "x": range(6)})

    with pytest.raises(ValueError, match="exactly two non-missing target classes"):
        run_ml_binary_classification_models(
            data,
            target_column="target",
            feature_columns=["x"],
            selected_models=["Logistic Regression"],
            positive_class="yes",
        )


def test_severe_class_imbalance_returns_warning_note_in_model_run():
    data = pd.DataFrame(
        {
            "target": ["yes"] * 2 + ["no"] * 30,
            "x": list(range(32)),
        }
    )

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["x"],
        selected_models=["Logistic Regression"],
        positive_class="yes",
        test_size=0.25,
        random_state=4,
    )[0]

    assert "test split contains only one target class" in result["model_run"]["notes"]
    assert result["model_run"]["test_metrics"]["test_roc_auc"] is None


def test_statistical_logistic_regression_records_perfect_separation_warning():
    data = pd.DataFrame(
        {
            "target": ["no"] * 6 + ["yes"] * 6,
            "x": [0] * 6 + [1] * 6,
        }
    )

    result = run_logistic_regression(
        data,
        y_column="target",
        positive_class="yes",
        x_columns=["x"],
        test_size=0.25,
        random_state=1,
    )

    assert any("Perfect separation" in warning for warning in result["warnings"])


def test_high_cardinality_categorical_variable_can_be_encoded_with_unknowns():
    train = pd.DataFrame(
        {
            "category": [f"level_{index}" for index in range(30)],
            "x": range(30),
        }
    )
    test = pd.DataFrame({"category": ["new_level"], "x": [100]})
    preprocessing = build_preprocessing_pipeline(train, ["category", "x"])

    transformed_train = preprocessing.fit_transform(train[["category", "x"]])
    transformed_test = preprocessing.transform(test[["category", "x"]])

    assert transformed_train.shape[0] == 30
    assert transformed_test.shape[0] == 1


def test_extreme_outlier_is_flagged_without_modifying_input():
    data = pd.DataFrame({"x": [10, 11, 9, 10, 12, 10, 999]})
    original = data.copy(deep=True)

    result = run_outlier_detection(data, ["x"], method="IQR rule")

    assert result["summary"]["flagged_count"] >= 1
    assert result["labels"].iloc[-1]
    pd.testing.assert_frame_equal(data, original)


def test_missing_target_values_are_dropped_for_ml_regression_without_mutating_input():
    data = pd.DataFrame({"target": [1.0, None, 3.0, 4.0, 5.0], "x": [1, 2, 3, 4, 5]})
    original = data.copy(deep=True)

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["x"],
        selected_models=["Linear Regression"],
        random_state=2,
    )[0]

    assert result["model_run"]["split_config"]["rows_used"] == 4
    pd.testing.assert_frame_equal(data, original)


def test_more_predictors_than_rows_can_use_regularized_ml_model_without_crashing():
    data = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "x1": [1, 2, 3, 4],
            "x2": [2, 3, 4, 5],
            "x3": [3, 4, 5, 6],
            "x4": [4, 5, 6, 7],
            "x5": [5, 6, 7, 8],
        }
    )

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["x1", "x2", "x3", "x4", "x5"],
        selected_models=["Ridge"],
        random_state=1,
    )[0]

    assert result["model_run"]["model_name"] == "Ridge"
    assert "test_rmse" in result["model_run"]["test_metrics"]


def test_vif_with_singular_matrix_returns_warnings_not_crash():
    data = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [2, 4, 6, 8, 10]})

    vif_table = compute_vif_table(data, ["x1", "x2"])
    condition = compute_condition_number(data, ["x1", "x2"])

    assert np.isinf(vif_table["vif"]).all()
    assert set(vif_table["risk_level"]) == {"strong_warning"}
    assert condition["risk_level"] == "warning"


def test_prediction_with_missing_input_value_uses_pipeline_imputation():
    data = pd.DataFrame(
        {
            "target": [10.0, 12.0, 14.0, 16.0, 18.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group": ["A", "A", "B", "B", "C"],
        }
    )
    preprocessing = build_preprocessing_pipeline(data, ["x", "group"])
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", LinearRegression())])
    pipeline.fit(data[["x", "group"]], data["target"])
    model_run = {
        "run_id": "run-missing-input",
        "task_type": "regression",
        "model_family": "machine_learning",
        "model_name": "Linear Regression",
        "target": "target",
        "features": ["x", "group"],
        "preprocessing": {},
    }
    artifact = {
        "run_id": "run-missing-input",
        "task_type": "regression",
        "model_family": "machine_learning",
        "model_name": "Linear Regression",
        "fitted_pipeline": pipeline,
        "fitted_model": pipeline.named_steps["model"],
        "features": ["x", "group"],
        "prediction_supported": True,
    }

    prediction = predict_from_model_run(model_run, artifact, {"x": None, "group": "B"})

    assert prediction["task_type"] == "regression"
    assert isinstance(prediction["predicted_value"], float)


def test_model_run_without_saved_artifact_fails_prediction_gracefully():
    model_run = {
        "run_id": "old-run",
        "task_type": "regression",
        "model_family": "machine_learning",
        "model_name": "Old model",
        "target": "target",
        "features": ["x"],
    }

    with pytest.raises(PredictionError, match="No usable fitted model artifact"):
        predict_from_model_run(model_run, None, {"x": 1})


def test_report_generation_with_no_model_runs_has_honest_empty_sections():
    data = pd.DataFrame({"x": [1, 2, 3]})

    context = build_report_context(working_df=data, model_runs=[])

    assert context["dataset_overview"]["rows"] == 3
    assert context["model_runs_summary"] == []
    assert context["model_comparison_summary"]["regression"] == []
    assert context["model_comparison_summary"]["binary_classification"] == []
