from datetime import datetime, timezone

import pandas as pd

from src.core.model_run import create_model_run
from src.reporting.export_html import export_report_html
from src.reporting.export_markdown import export_report_markdown
from src.reporting.report_builder import build_report_context


def _working_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [10.0, 12.0, 14.0, 16.0],
            "x": [1.0, 2.0, None, 4.0],
            "group": ["A", "A", "B", "B"],
        }
    )


def _statistical_run() -> dict:
    run = create_model_run(
        task_type="regression",
        model_family="statistical",
        model_name="linear_regression_ols",
        target="target",
        features=["x", "group"],
        split_config={"test_size": 0.25, "random_state": 42, "train_rows": 3, "test_rows": 1},
        preprocessing={"categorical_encoding": "one-hot", "missing_rows": "dropped before fitting"},
        statistical_summary=[
            {"statistic": "R-squared", "value": 0.91},
            {"statistic": "Adjusted R-squared", "value": 0.84},
            {"statistic": "AIC", "value": 12.3},
            {"statistic": "BIC", "value": 13.4},
        ],
        train_metrics={"RMSE": 0.4, "MAE": 0.3, "R-squared": 0.95},
        test_metrics={"RMSE": 0.8, "MAE": 0.7, "R-squared": 0.75},
        coefficient_table=[
            {"term": "const", "estimate": 2.0, "std_error": 0.2, "p_value": 0.01},
            {"term": "x", "estimate": 1.5, "std_error": 0.3, "p_value": 0.03},
        ],
        diagnostic_plot_keys=["residuals_vs_fitted", "normal_qq"],
        formula_latex={
            "symbolic": r"y_i = \beta_0 + \beta_1 x_i + \epsilon_i",
            "estimated": r"\hat{y}_i = 2.0 + 1.5x_i",
            "note": "OLS estimated formula.",
        },
        notes="OLS model run for report reproducibility testing.",
    )
    run["run_id"] = "stat-run"
    return run


def _ml_run() -> dict:
    run = create_model_run(
        task_type="regression",
        model_family="machine_learning",
        model_name="Linear Regression",
        target="target",
        features=["x", "group"],
        split_config={"test_size": 0.25, "random_state": 42, "train_rows": 3, "test_rows": 1},
        preprocessing={"pipeline": "sklearn Pipeline", "scale_numeric": True},
        train_metrics={"train_rmse": 0.5, "train_mae": 0.4, "train_r2": 0.93},
        test_metrics={
            "test_rmse": 0.9,
            "test_mae": 0.8,
            "test_r2": 0.7,
            "cv_rmse_mean": 1.0,
            "cv_rmse_std": 0.1,
        },
        coefficient_table=[
            {
                "term": "x",
                "coefficient": 0.5,
                "absolute_coefficient": 0.5,
                "coefficient_scale_note": "Coefficients are on the scaled feature scale.",
            }
        ],
        formula_latex={
            "estimated": r"\hat{y}_i = 0.1 + 0.5x_i",
            "note": "sklearn coefficients are on the transformed feature scale.",
        },
        notes="ML model run for report reproducibility testing.",
    )
    run["run_id"] = "ml-run"
    return run


def _artifacts() -> dict:
    return {
        "stat-run": {
            "run_id": "stat-run",
            "interval_supported": True,
            "interval_method": "statsmodels_get_prediction",
            "influence_table": [
                {"row_index": 0, "cooks_distance": 0.1, "leverage": 0.2},
                {"row_index": 1, "cooks_distance": 0.4, "leverage": 0.5},
            ],
            "top_influential_rows": [{"row_index": 1, "cooks_distance": 0.4}],
        },
        "ml-run": {
            "run_id": "ml-run",
            "interval_supported": False,
            "interval_method": "bootstrap_empirical_interval",
            "coefficient_interpretation_notes": [
                "This is a predictive ML coefficient table. It does not include inferential p-values or confidence intervals."
            ],
        },
    }


def _prediction_log() -> list[dict]:
    return [
        {
            "timestamp": "2026-05-17T12:00:00+00:00",
            "run_id": "stat-run",
            "model_name": "linear_regression_ols",
            "input_values": {"x": 3.0, "group": "B"},
            "prediction": 6.5,
            "interval": [
                {
                    "predicted_mean": 6.5,
                    "mean_ci_lower": 5.8,
                    "mean_ci_upper": 7.2,
                    "obs_ci_lower": 4.1,
                    "obs_ci_upper": 8.9,
                    "interval_type": "statsmodels_prediction_interval",
                }
            ],
        },
        {
            "timestamp": "2026-05-17T12:05:00+00:00",
            "run_id": "ml-run",
            "model_name": "Linear Regression",
            "input_values": {"x": 3.0, "group": "B"},
            "prediction": 6.1,
            "interval": {
                "lower_bound": 5.0,
                "upper_bound": 7.4,
                "interval_type": "bootstrap_empirical_interval",
            },
        },
    ]


def _context() -> dict:
    return build_report_context(
        working_df=_working_df(),
        cleaning_log=[{"operation": "fill_numeric_median", "column": "x"}],
        transformation_log=[{"method": "log1p", "source_columns": ["target"], "new_column": "target_log1p"}],
        model_runs=[_statistical_run(), _ml_run()],
        model_artifacts=_artifacts(),
        prediction_log=_prediction_log(),
        title="Reproducibility Report",
        generated_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )


def test_report_context_contains_reproducibility_fields():
    context = _context()

    assert context["generated_timestamp"] == "2026-05-17T00:00:00+00:00"
    assert context["dataset_overview"]["rows"] == 4
    assert context["working_dataset_status"]["dataset_role"] == "working_df"
    assert context["working_dataset_status"]["cleaning_steps"] == 1
    assert context["schema_summary"]
    assert context["missing_value_summary"]
    assert context["cleaning_log"][0]["operation"] == "fill_numeric_median"
    assert context["transformation_log"][0]["method"] == "log1p"
    assert len(context["model_runs_summary"]) == 2
    assert context["model_comparison_summary"]["regression"]
    assert context["interval_and_coefficient_notes"]
    assert context["limitations"]


def test_report_model_summaries_include_targets_features_metrics_diagnostics_and_intervals():
    context = _context()
    stat_summary = context["model_runs_summary"][0]
    ml_summary = context["model_runs_summary"][1]

    assert stat_summary["target"] == "target"
    assert stat_summary["features"] == ["x", "group"]
    assert stat_summary["split_config"]["test_size"] == 0.25
    assert stat_summary["preprocessing_summary"]["categorical_encoding"] == "one-hot"
    assert stat_summary["estimated_formula"]
    assert stat_summary["coefficient_table"][0]["term"] == "const"
    assert stat_summary["train_metrics"]["RMSE"] == 0.4
    assert stat_summary["test_metrics"]["RMSE"] == 0.8
    assert stat_summary["diagnostics_summary"]["diagnostic_plot_keys"] == ["residuals_vs_fitted", "normal_qq"]
    assert stat_summary["diagnostics_summary"]["influence_rows"] == 2
    assert stat_summary["prediction_intervals"][0]["mean_ci_lower"] == 5.8
    assert stat_summary["prediction_intervals"][0]["obs_ci_lower"] == 4.1

    assert "predictive ML coefficient table" in ml_summary["coefficient_table_note"]
    assert ml_summary["cv_metrics"]["cv_rmse_mean"] == 1.0
    assert ml_summary["prediction_intervals"][0]["interval_type"] == "bootstrap_empirical_interval"


def test_markdown_export_contains_required_reproducibility_sections_and_caveats():
    markdown = export_report_markdown(_context())

    required_text = [
        "# Reproducibility Report",
        "Generated timestamp:",
        "## Dataset Overview",
        "## Working Dataset Status",
        "## Schema Summary",
        "## Missing Value Summary",
        "## Cleaning Log",
        "## Transformation Log",
        "Selected",
    ]
    for text in required_text[:-1]:
        assert text in markdown

    assert "Target:" in markdown
    assert "Features:" in markdown
    assert "Split config" in markdown
    assert "Preprocessing summary" in markdown
    assert "#### Model Formula" in markdown
    assert "#### Estimated Formula" in markdown
    assert "#### Coefficient Table" in markdown
    assert "Train metrics" in markdown
    assert "Test metrics" in markdown
    assert "#### Diagnostics" in markdown
    assert "#### Prediction Examples" in markdown
    assert "statsmodels_prediction_interval" in markdown
    assert "bootstrap_empirical_interval" in markdown
    assert "## Model Comparison Summary" in markdown
    assert "## Interval And Coefficient Notes" in markdown
    assert "## Limitations" in markdown
    assert "Confidence interval" in markdown
    assert "Prediction interval" in markdown
    assert "ML empirical uncertainty interval" in markdown
    assert "predictive ML coefficient table" in markdown
    assert "do not establish causality" in markdown


def test_html_export_contains_same_major_report_sections():
    html = export_report_html(_context())

    assert "<!doctype html>" in html
    assert "Working Dataset Status" in html
    assert "Schema Summary" in html
    assert "Missing Value Summary" in html
    assert "Cleaning Log" in html
    assert "Transformation Log" in html
    assert "Diagnostics" in html
    assert "Prediction Examples" in html
    assert "Model Comparison Summary" in html
    assert "Interval And Coefficient Notes" in html
    assert "bootstrap_empirical_interval" in html


def test_report_with_no_model_runs_is_honest_about_empty_sections():
    context = build_report_context(
        working_df=_working_df(),
        model_runs=[],
        generated_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )
    markdown = export_report_markdown(context)

    assert context["model_runs_summary"] == []
    assert "No saved model runs are available." in markdown
    assert "No regression comparison table is available." in markdown
    assert "No binary classification comparison table is available." in markdown
