from src.core.model_comparison import (
    build_binary_classification_comparison_table,
    build_multiclass_classification_comparison_table,
    build_regression_comparison_table,
    filter_model_runs,
)
from src.core.model_run import create_model_run


def test_regression_comparison_uses_only_regression_metrics():
    regression_run = create_model_run(
        task_type="regression",
        model_family="statistical",
        model_name="linear_regression_ols",
        target="y",
        features=["x"],
        split_config={"test_size": 0.2},
        statistical_summary=[
            {"statistic": "AIC", "value": 10.0},
            {"statistic": "BIC", "value": 12.0},
            {"statistic": "Adjusted R-squared", "value": 0.8},
        ],
        train_metrics={"transformed_scale": {"RMSE": 1.0, "MAE": 0.8, "R-squared": 0.9}},
        test_metrics={"transformed_scale": {"RMSE": 1.2, "MAE": 1.0, "R-squared": 0.7}},
    )
    classification_run = create_model_run(
        task_type="binary_classification",
        model_family="statistical",
        model_name="binary_logistic_regression_logit",
        target="churn",
        features=["x"],
        split_config={"test_size": 0.2},
        train_metrics={"accuracy": 0.8},
        test_metrics={"accuracy": 0.75},
    )

    table = build_regression_comparison_table([regression_run, classification_run])

    assert len(table) == 1
    assert table.loc[0, "model_name"] == "linear_regression_ols"
    assert table.loc[0, "model_family"] == "statistical"
    assert table.loc[0, "test_rmse"] == 1.2
    assert "test_accuracy" not in table.columns


def test_regression_comparison_handles_machine_learning_metrics():
    ml_run = create_model_run(
        task_type="regression",
        model_family="machine_learning",
        model_name="Random Forest Regressor",
        target="income",
        features=["age", "segment"],
        split_config={"test_size": 0.2},
        statistical_summary={},
        train_metrics={
            "train_rmse": 0.5,
            "train_mae": 0.4,
            "train_r2": 0.95,
        },
        test_metrics={
            "test_rmse": 1.5,
            "test_mae": 1.2,
            "test_r2": 0.75,
            "cv_rmse_mean": 1.6,
            "cv_rmse_std": 0.2,
            "cv_mae_mean": 1.3,
            "cv_mae_std": 0.1,
            "cv_r2_mean": 0.72,
            "cv_r2_std": 0.05,
        },
    )

    table = build_regression_comparison_table([ml_run])

    assert len(table) == 1
    assert table.loc[0, "model_family"] == "machine_learning"
    assert table.loc[0, "train_rmse"] == 0.5
    assert table.loc[0, "test_rmse"] == 1.5
    assert table.loc[0, "cv_rmse_mean"] == 1.6
    assert table.loc[0, "cv_r2_std"] == 0.05
    assert table.loc[0, "aic"] is None


def test_binary_classification_comparison_uses_only_classification_metrics():
    classification_run = create_model_run(
        task_type="binary_classification",
        model_family="statistical",
        model_name="binary_logistic_regression_logit",
        target="churn",
        features=["x"],
        split_config={"test_size": 0.2},
        statistical_summary=[
            {"statistic": "AIC", "value": 20.0},
            {"statistic": "BIC", "value": 22.0},
            {"statistic": "Pseudo R-squared", "value": 0.25},
        ],
        train_metrics={
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "F1": 0.65,
            "ROC AUC": 0.85,
            "PR AUC": 0.86,
        },
        test_metrics={
            "accuracy": 0.75,
            "precision": 0.6,
            "recall": 0.5,
            "F1": 0.55,
            "ROC AUC": 0.8,
            "PR AUC": 0.81,
        },
    )
    regression_run = create_model_run(
        task_type="regression",
        model_family="statistical",
        model_name="linear_regression_ols",
        target="y",
        features=["x"],
        split_config={"test_size": 0.2},
        train_metrics={"transformed_scale": {"RMSE": 1.0}},
        test_metrics={"transformed_scale": {"RMSE": 1.2}},
    )

    table = build_binary_classification_comparison_table([classification_run, regression_run])

    assert len(table) == 1
    assert table.loc[0, "model_name"] == "binary_logistic_regression_logit"
    assert table.loc[0, "model_family"] == "statistical"
    assert table.loc[0, "test_accuracy"] == 0.75
    assert table.loc[0, "test_pr_auc"] == 0.81
    assert table.loc[0, "pseudo_r_squared"] == 0.25
    assert "test_rmse" not in table.columns


def test_binary_classification_comparison_handles_machine_learning_metrics_and_positive_class():
    ml_run = create_model_run(
        task_type="binary_classification",
        model_family="machine_learning",
        model_name="Random Forest Classifier",
        target="churn",
        features=["age", "segment"],
        split_config={"test_size": 0.2},
        preprocessing={"positive_class": "yes"},
        statistical_summary={},
        train_metrics={
            "train_accuracy": 0.9,
            "train_precision": 0.85,
            "train_recall": 0.8,
            "train_f1": 0.82,
            "train_roc_auc": 0.93,
            "train_pr_auc": 0.91,
        },
        test_metrics={
            "test_accuracy": 0.8,
            "test_precision": 0.75,
            "test_recall": 0.7,
            "test_f1": 0.72,
            "test_roc_auc": 0.84,
            "test_pr_auc": 0.82,
            "cv_accuracy_mean": 0.79,
            "cv_accuracy_std": 0.04,
            "cv_precision_mean": 0.74,
            "cv_precision_std": 0.05,
            "cv_recall_mean": 0.69,
            "cv_recall_std": 0.06,
            "cv_f1_mean": 0.71,
            "cv_f1_std": 0.03,
            "cv_roc_auc_mean": 0.83,
            "cv_roc_auc_std": 0.02,
        },
    )

    table = build_binary_classification_comparison_table([ml_run])

    assert len(table) == 1
    assert table.loc[0, "model_family"] == "machine_learning"
    assert table.loc[0, "positive_class"] == "yes"
    assert table.loc[0, "train_accuracy"] == 0.9
    assert table.loc[0, "test_pr_auc"] == 0.82
    assert table.loc[0, "cv_accuracy_mean"] == 0.79
    assert table.loc[0, "cv_roc_auc_std"] == 0.02
    assert table.loc[0, "aic"] is None


def test_multiclass_classification_comparison_uses_only_multiclass_metrics():
    multiclass_run = create_model_run(
        task_type="multiclass_classification",
        model_family="machine_learning",
        model_name="Random Forest Classifier",
        target="species",
        features=["x1", "x2"],
        split_config={"test_size": 0.2},
        preprocessing={"target_classes": ["A", "B", "C"]},
        train_metrics={
            "train_accuracy": 0.9,
            "train_macro_f1": 0.88,
            "train_weighted_f1": 0.89,
            "train_log_loss": 0.4,
        },
        test_metrics={
            "test_accuracy": 0.8,
            "test_macro_precision": 0.78,
            "test_macro_recall": 0.79,
            "test_macro_f1": 0.77,
            "test_weighted_f1": 0.78,
            "test_log_loss": 0.6,
        },
    )
    binary_run = create_model_run(
        task_type="binary_classification",
        model_family="machine_learning",
        model_name="Logistic Regression",
        target="churn",
        features=["x1"],
        split_config={},
        test_metrics={"test_f1": 0.7},
    )

    table = build_multiclass_classification_comparison_table([multiclass_run, binary_run])

    assert len(table) == 1
    assert table.loc[0, "model_name"] == "Random Forest Classifier"
    assert table.loc[0, "target_classes"] == "A, B, C"
    assert table.loc[0, "test_accuracy"] == 0.8
    assert table.loc[0, "test_macro_f1"] == 0.77
    assert table.loc[0, "test_log_loss"] == 0.6
    assert "test_roc_auc" not in table.columns


def test_filter_model_runs_filters_by_task_target_and_family():
    first = create_model_run(
        task_type="regression",
        model_family="statistical",
        model_name="linear_regression_ols",
        target="income",
        features=["age"],
        split_config={},
    )
    second = create_model_run(
        task_type="binary_classification",
        model_family="statistical",
        model_name="binary_logistic_regression_logit",
        target="churn",
        features=["age"],
        split_config={},
    )

    filtered = filter_model_runs([first, second], "regression", "income", "statistical")

    assert filtered == [first]
