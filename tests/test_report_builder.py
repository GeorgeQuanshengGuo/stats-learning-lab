from datetime import datetime, timezone

import pandas as pd

from src.core.model_run import create_model_run
from src.core.analysis_plan import create_analysis_plan
from src.reporting.export_html import export_report_html
from src.reporting.export_markdown import export_report_markdown
from src.reporting.report_builder import build_report_context, summarize_model_runs


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [10.0, None, 30.0],
            "group": ["A", "B", "A"],
        }
    )


def _sample_model_run() -> dict:
    model_run = create_model_run(
        task_type="regression",
        model_family="statistical",
        model_name="linear_regression_ols",
        target="y",
        features=["x", "group"],
        split_config={"test_size": 0.2, "random_state": 42},
        preprocessing={"encoding": "one-hot"},
        statistical_summary=[
            {"statistic": "AIC", "value": 10.5},
            {"statistic": "BIC", "value": 12.5},
        ],
        train_metrics={"RMSE": 0.5},
        test_metrics={
            "RMSE": 0.8,
            "cv_rmse_mean": 0.9,
            "cv_rmse_std": 0.1,
        },
        coefficient_table=[
            {"term": "intercept", "estimate": 1.0, "p_value": 0.01},
            {"term": "x", "estimate": 0.5, "p_value": 0.04},
        ],
        feature_importance_table=[
            {"feature": "x", "importance": 0.8, "importance_type": "tree_based"},
        ],
    )
    model_run["run_id"] = "run-stat-1"
    model_run["formula_latex"] = {
        "symbolic": r"\mathrm{y}_i = \beta_0 + \beta_1 \mathrm{x}_i + \epsilon_i",
        "estimated": r"\hat{\mathrm{y}}_i = 1 + 0.5 \mathrm{x}_i",
        "note": "Estimated linear regression formula.",
    }
    return model_run


def _sample_model_artifacts() -> dict:
    return {
        "run-stat-1": {
            "run_id": "run-stat-1",
            "model_name": "linear_regression_ols",
            "model_family": "statistical",
            "task_type": "regression",
            "formula_latex": _sample_model_run()["formula_latex"],
            "coefficient_table": _sample_model_run()["coefficient_table"],
            "interval_supported": True,
            "interval_method": "statsmodels_get_prediction",
            "tree_rules": "if x <= 20 then value = 1.5",
            "tree_explanation_warning": "Deep trees may overfit and become difficult to interpret.",
            "feature_importance_table": [
                {"feature": "x", "importance": 0.8, "importance_type": "tree_based"},
            ],
            "calibration_summary": [{"bin": 1, "observed_frequency": 0.5}],
        }
    }


def _sample_prediction_log() -> list[dict]:
    return [
        {
            "timestamp": "2026-05-16T00:00:00+00:00",
            "run_id": "run-stat-1",
            "model_name": "linear_regression_ols",
            "input_values": {"x": 25, "group": "A"},
            "prediction": 13.5,
            "interval": [
                {
                    "predicted_mean": 13.5,
                    "mean_ci_lower": 12.0,
                    "mean_ci_upper": 15.0,
                    "obs_ci_lower": 10.0,
                    "obs_ci_upper": 17.0,
                    "interval_type": "statsmodels_prediction_interval",
                }
            ],
            "threshold": None,
        }
    ]


def _sample_ml_model_run() -> dict:
    model_run = create_model_run(
        task_type="regression",
        model_family="machine_learning",
        model_name="Linear Regression",
        target="y",
        features=["x", "group"],
        split_config={"test_size": 0.2, "random_state": 42},
        preprocessing={"pipeline": "sklearn Pipeline", "scale_numeric": True},
        train_metrics={"train_rmse": 0.4},
        test_metrics={"test_rmse": 0.7},
        coefficient_table=[
            {
                "term": "x",
                "coefficient": 0.3,
                "absolute_coefficient": 0.3,
                "coefficient_scale_note": "Coefficients are on the transformed feature scale.",
            }
        ],
    )
    model_run["run_id"] = "run-ml-1"
    model_run["formula_latex"] = {
        "estimated": r"\hat{\mathrm{y}}_i = 0.2 + 0.3 \mathrm{x}_i",
        "note": "sklearn coefficients are on the transformed feature scale.",
    }
    return model_run


def test_build_report_context_includes_dataset_schema_missing_logs_and_models():
    data = _sample_df()
    cleaning_log = [{"operation": "fill_numeric_median", "column": "x"}]
    transformation_log = [{"method": "log1p", "source_columns": ["y"], "new_column": "log1p_y"}]
    model_run = _sample_model_run()

    context = build_report_context(
        working_df=data,
        original_df=data,
        uploaded_file_name="synthetic.csv",
        cleaning_log=cleaning_log,
        transformation_log=transformation_log,
        model_runs=[model_run],
        model_artifacts=_sample_model_artifacts(),
        prediction_log=_sample_prediction_log(),
        analysis_plan=create_analysis_plan(
            research_question="What predicts y?",
            analysis_goal="prediction",
            target_variable="y",
            candidate_features=["x", "group"],
            interpretation_boundaries="Associations only.",
            now="2026-05-16T00:00:00+00:00",
        ),
        analysis_decision_log=[
            {
                "timestamp": "2026-05-16T00:00:00+00:00",
                "section": "research_question",
                "old_value": "",
                "new_value": "What predicts y?",
                "reason": "Initial plan.",
            }
        ],
        title="Demo Report",
        generated_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )

    assert context["title"] == "Demo Report"
    assert context["analysis_plan"]["target_variable"] == "y"
    assert context["analysis_decision_log"]
    assert context["statistical_rigor_checklist"]
    assert context["data_readiness_summary"]["row_count"] == 3
    assert context["reproducibility_manifest"]["uploaded_file_name"] == "synthetic.csv"
    assert context["dataset_overview"]["rows"] == 3
    assert context["dataset_overview"]["columns"] == 3
    assert context["dataset_overview"]["column_names"] == ["y", "x", "group"]
    assert context["schema_summary"]
    assert context["missing_value_summary"]
    assert context["cleaning_log"] == cleaning_log
    assert context["transformation_log"] == transformation_log
    assert context["model_runs_summary"][0]["model_name"] == "linear_regression_ols"
    assert context["model_runs_summary"][0]["aic"] == 10.5
    assert context["model_runs_summary"][0]["bic"] == 12.5
    assert context["model_runs_summary"][0]["cv_metrics"] == {
        "cv_rmse_mean": 0.9,
        "cv_rmse_std": 0.1,
    }
    assert context["model_runs_summary"][0]["estimated_formula"]
    assert context["model_runs_summary"][0]["coefficient_table"]
    assert context["model_runs_summary"][0]["prediction_examples"]
    assert context["model_runs_summary"][0]["prediction_intervals"]
    assert context["model_runs_summary"][0]["decision_tree_rules"]
    assert context["model_runs_summary"][0]["feature_importance_table"]
    assert context["model_runs_summary"][0]["interpretability_outputs"]["calibration"]
    assert context["model_comparison_summary"]["regression"]


def test_summarize_model_runs_includes_required_model_fields():
    summary = summarize_model_runs([_sample_model_run()])[0]

    assert summary["model_name"] == "linear_regression_ols"
    assert summary["model_family"] == "statistical"
    assert summary["task_type"] == "regression"
    assert summary["target"] == "y"
    assert summary["features"] == ["x", "group"]
    assert summary["split_config"]["test_size"] == 0.2
    assert summary["preprocessing_summary"]["encoding"] == "one-hot"
    assert summary["train_metrics"]["RMSE"] == 0.5
    assert summary["test_metrics"]["RMSE"] == 0.8


def test_report_marks_ml_coefficients_as_predictive_not_inferential():
    summary = summarize_model_runs([_sample_ml_model_run()])[0]

    assert "predictive ML coefficient table" in summary["coefficient_table_note"]
    assert "p-values" in summary["coefficient_table_note"]
    assert summary["coefficient_table"][0]["coefficient"] == 0.3


def test_report_builder_accepts_dataframe_tables_without_ambiguous_truth_value():
    model_run = _sample_model_run()
    model_run["coefficient_table"] = pd.DataFrame(
        [
            {"term": "intercept", "estimate": 1.0},
            {"term": "x", "estimate": 0.5},
        ]
    )
    model_run["feature_importance_table"] = pd.DataFrame(
        [{"feature": "x", "importance": 0.9, "importance_type": "tree_based"}]
    )
    artifacts = {
        model_run["run_id"]: {
            "coefficient_table": pd.DataFrame([{"term": "artifact_x", "estimate": 0.4}]),
            "feature_importance_table": pd.DataFrame(
                [{"feature": "artifact_x", "importance": 0.7}]
            ),
        }
    }

    context = build_report_context(
        working_df=_sample_df(),
        model_runs=[model_run],
        model_artifacts=artifacts,
    )

    summary = context["model_runs_summary"][0]
    assert summary["coefficient_table"][0]["term"] == "intercept"
    assert summary["feature_importance_table"][0]["feature"] == "x"


def test_markdown_export_contains_expected_sections_and_honest_empty_text():
    context = build_report_context(
        working_df=None,
        cleaning_log=[],
        transformation_log=[],
        model_runs=[],
        title="Empty Report",
        generated_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )

    markdown = export_report_markdown(context)

    assert "# Empty Report" in markdown
    assert "## Dataset Overview" in markdown
    assert "No dataset is currently loaded." in markdown
    assert "No cleaning operations have been logged." in markdown
    assert "No saved model runs are available." in markdown
    assert "No analysis plan was recorded before modeling." in markdown
    assert "Reproducibility Manifest" in markdown
    assert "## Limitations" in markdown


def test_markdown_export_includes_model_metrics_and_comparison_summary():
    context = build_report_context(
        working_df=_sample_df(),
        model_runs=[_sample_model_run()],
        model_artifacts=_sample_model_artifacts(),
        prediction_log=_sample_prediction_log(),
        generated_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )

    markdown = export_report_markdown(context)

    assert "linear_regression_ols" in markdown
    assert "cv_rmse_mean" in markdown
    assert "AIC" in markdown
    assert "Model Comparison Summary" in markdown
    assert "Analysis Plan" in markdown
    assert "Statistical Rigor Checklist" in markdown
    assert "Data Readiness Summary" in markdown
    assert "Reproducibility Manifest" in markdown
    assert "#### Model Formula" in markdown
    assert "#### Estimated Formula" in markdown
    assert "#### Coefficient Table" in markdown
    assert "#### Prediction Examples" in markdown
    assert "#### Prediction Intervals or Uncertainty Intervals" in markdown
    assert "#### Decision Tree Rules" in markdown
    assert "#### Feature Importance" in markdown
    assert "#### PDP / ICE / Calibration" in markdown
    assert "statsmodels_prediction_interval" in markdown


def test_html_export_contains_expected_sections_and_escapes_content():
    context = build_report_context(
        working_df=_sample_df(),
        cleaning_log=[{"operation": "<fill>", "column": "x"}],
        model_runs=[_sample_model_run()],
        model_artifacts=_sample_model_artifacts(),
        prediction_log=_sample_prediction_log(),
        title="HTML Report",
        generated_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )

    html = export_report_html(context)

    assert "<!doctype html>" in html
    assert "<h1>HTML Report</h1>" in html
    assert "Dataset Overview" in html
    assert "Analysis Plan" in html
    assert "Statistical Rigor Checklist" in html
    assert "Reproducibility Manifest" in html
    assert "linear_regression_ols" in html
    assert "&lt;fill&gt;" in html
    assert "Model Formula" in html
    assert "Estimated Formula" in html
    assert "Coefficient Table" in html
    assert "Prediction Examples" in html
    assert "Prediction Intervals or Uncertainty Intervals" in html


def test_report_builder_does_not_modify_input_df():
    data = _sample_df()
    original = data.copy(deep=True)

    build_report_context(working_df=data, model_runs=[_sample_model_run()])

    pd.testing.assert_frame_equal(data, original)
