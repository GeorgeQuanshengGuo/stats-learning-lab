import pandas as pd

from src.modeling.diagnostics.overfitting import (
    build_classification_overfitting_diagnostics,
    build_cv_stability_diagnostics,
    build_overfitting_diagnostics,
    build_regression_overfitting_diagnostics,
)
from src.visualization.diagnostic_plots import plot_overfitting_metric_gaps


def test_regression_overfitting_diagnostics_flags_large_rmse_and_r2_gaps():
    model_run = {
        "task_type": "regression",
        "train_metrics": {"train_rmse": 1.0, "train_r2": 0.95},
        "test_metrics": {"test_rmse": 1.7, "test_r2": 0.60},
    }

    table = build_regression_overfitting_diagnostics(model_run)

    ratio_row = table.loc[table["metric"] == "test_rmse_train_rmse_ratio"].iloc[0]
    r2_row = table.loc[table["metric"] == "train_r2_minus_test_r2"].iloc[0]
    assert ratio_row["risk_level"] == "strong_warning"
    assert r2_row["risk_level"] == "strong_warning"
    assert ratio_row["value"] == 1.7


def test_classification_overfitting_diagnostics_flags_auc_and_f1_gaps():
    model_run = {
        "task_type": "binary_classification",
        "train_metrics": {"train_accuracy": 0.95, "train_f1": 0.91, "train_roc_auc": 0.96},
        "test_metrics": {"test_accuracy": 0.82, "test_f1": 0.75, "test_roc_auc": 0.82},
    }

    table = build_classification_overfitting_diagnostics(model_run)

    assert table.loc[table["metric"] == "train_auc_minus_test_auc", "risk_level"].iloc[0] == "strong_warning"
    assert table.loc[table["metric"] == "train_f1_minus_test_f1", "risk_level"].iloc[0] == "strong_warning"
    assert table.loc[table["metric"] == "train_accuracy_minus_test_accuracy", "risk_level"].iloc[0] == "warning"


def test_cv_stability_diagnostics_uses_saved_cv_mean_and_std():
    model_run = {
        "task_type": "regression",
        "train_metrics": {},
        "test_metrics": {"cv_rmse_mean": 1.0, "cv_rmse_std": 0.4},
    }

    table = build_cv_stability_diagnostics(model_run)

    assert table.loc[0, "metric"] == "cv_rmse"
    assert table.loc[0, "risk_level"] == "warning"


def test_overfitting_dispatch_handles_supported_and_unknown_task_types():
    regression = build_overfitting_diagnostics(
        {
            "task_type": "regression",
            "train_metrics": {"train_rmse": 1.0},
            "test_metrics": {"test_rmse": 1.1},
        }
    )
    unknown = build_overfitting_diagnostics({"task_type": "unsupported"})

    assert not regression.empty
    assert unknown.empty


def test_overfitting_plot_returns_figure():
    table = pd.DataFrame(
        [
            {"metric": "train_f1_minus_test_f1", "value": 0.2, "risk_level": "strong_warning", "message": "gap"},
        ]
    )

    figure = plot_overfitting_metric_gaps(table)

    assert figure.data
