import pandas as pd

from src.core.model_run import create_model_run
from src.reporting.export_markdown import export_report_markdown
from src.reporting.interpretation import explain_model_run
from src.reporting.report_builder import build_report_context


def _linear_model_run():
    return create_model_run(
        task_type="regression",
        model_family="statistical",
        model_name="linear_regression_ols",
        target="price",
        features=["size", "neighborhood"],
        split_config={"test_size": 0.2},
        statistical_summary=[
            {"statistic": "R-squared", "value": 0.82},
            {"statistic": "Adjusted R-squared", "value": 0.79},
            {"statistic": "AIC", "value": 120.0},
            {"statistic": "BIC", "value": 130.0},
        ],
        train_metrics={"transformed_scale": {"RMSE": 8.0, "MAE": 6.0, "R-squared": 0.86}},
        test_metrics={"transformed_scale": {"RMSE": 10.0, "MAE": 7.5, "R-squared": 0.74}},
        coefficient_table=pd.DataFrame(
            {
                "term": ["const", "size", "neighborhood_B"],
                "estimate": [1.0, 2.5, -0.8],
                "std_error": [0.1, 0.2, 0.3],
            }
        ),
    )


def _logistic_model_run():
    return create_model_run(
        task_type="binary_classification",
        model_family="statistical",
        model_name="binary_logistic_regression_logit",
        target="churn",
        features=["usage", "plan"],
        split_config={"test_size": 0.2},
        preprocessing={"positive_class": "yes", "threshold": 0.4},
        statistical_summary=[
            {"statistic": "AIC", "value": 80.0},
            {"statistic": "BIC", "value": 90.0},
        ],
        test_metrics={
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.7,
            "F1": 0.72,
            "ROC AUC": 0.84,
        },
        coefficient_table=pd.DataFrame(
            {
                "term": ["const", "usage", "plan_premium"],
                "odds_ratio": [1.0, 1.5, 0.6],
            }
        ),
    )


def _ml_regression_run():
    return create_model_run(
        task_type="regression",
        model_family="machine_learning",
        model_name="Random Forest Regressor",
        target="price",
        features=["size", "segment"],
        split_config={"test_size": 0.2},
        train_metrics={"train_rmse": 4.0, "train_mae": 3.0, "train_r2": 0.95},
        test_metrics={"test_rmse": 8.0, "test_mae": 6.0, "test_r2": 0.72},
        feature_importance_table=pd.DataFrame(
            {
                "feature": ["size", "segment_A"],
                "importance": [0.8, 0.2],
                "importance_type": ["tree_based", "tree_based"],
            }
        ),
        importance_type="tree_based",
    )


def _ml_classification_run():
    return create_model_run(
        task_type="binary_classification",
        model_family="machine_learning",
        model_name="Random Forest Classifier",
        target="churn",
        features=["usage", "plan"],
        split_config={"test_size": 0.2},
        preprocessing={"positive_class": "yes"},
        train_metrics={"train_accuracy": 0.95, "train_f1": 0.94},
        test_metrics={"test_accuracy": 0.78, "test_precision": 0.7, "test_recall": 0.68, "test_f1": 0.69, "test_roc_auc": 0.81},
        feature_importance_table=pd.DataFrame(
            {
                "feature": ["usage", "plan_basic"],
                "importance": [0.7, 0.3],
                "importance_type": ["tree_based", "tree_based"],
            }
        ),
        importance_type="tree_based",
    )


def test_linear_regression_explanation_mentions_fit_metrics_coefficients_and_causal_warning():
    explanation = explain_model_run(_linear_model_run())

    assert "price" in explanation
    assert "size" in explanation
    assert "R-squared" in explanation
    assert "Adjusted R-squared" in explanation
    assert "RMSE" in explanation
    assert "AIC" in explanation
    assert "positive coefficient" in explanation
    assert "not causal effects" in explanation


def test_logistic_regression_explanation_mentions_positive_class_metrics_odds_and_threshold_warning():
    explanation = explain_model_run(_logistic_model_run())

    assert "yes" in explanation
    assert "accuracy" in explanation
    assert "precision" in explanation
    assert "ROC AUC" in explanation
    assert "odds ratio" in explanation
    assert "threshold" in explanation
    assert "not causal effects" in explanation


def test_ml_regression_explanation_mentions_performance_overfit_and_feature_importance():
    explanation = explain_model_run(_ml_regression_run())

    assert "machine learning regression" in explanation
    assert "test RMSE" in explanation
    assert "overfitting" in explanation
    assert "size" in explanation
    assert "does not automatically imply causality" in explanation


def test_ml_binary_classification_explanation_mentions_positive_class_gap_and_feature_importance():
    explanation = explain_model_run(_ml_classification_run())

    assert "binary classifier" in explanation
    assert "yes" in explanation
    assert "test F1" in explanation
    assert "overfitting" in explanation
    assert "usage" in explanation
    assert "does not automatically imply causality" in explanation


def test_unknown_model_run_gets_clear_fallback():
    model_run = create_model_run(
        task_type="clustering",
        model_family="machine_learning",
        model_name="KMeans",
        target="",
        features=["x"],
        split_config={},
    )

    explanation = explain_model_run(model_run)

    assert "No rule-based interpretation" in explanation


def test_report_context_and_markdown_include_interpretation():
    context = build_report_context(
        working_df=pd.DataFrame({"price": [1, 2, 3], "size": [10, 20, 30]}),
        model_runs=[_linear_model_run()],
    )
    markdown = export_report_markdown(context)

    assert context["model_runs_summary"][0]["interpretation"]
    assert "Interpretation:" in markdown
    assert "not causal effects" in markdown
