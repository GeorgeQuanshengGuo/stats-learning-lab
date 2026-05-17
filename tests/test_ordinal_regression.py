import numpy as np
import pandas as pd
import pytest

from src.core.model_comparison import build_ordinal_classification_comparison_table
from src.core.model_run import create_model_run
from src.modeling.statistical.ordinal_regression import (
    PROPORTIONAL_ODDS_WARNING,
    predict_ordered_model,
    run_ordinal_logistic_regression,
)
from src.prediction.prediction_service import predict_from_model_run


def test_ordinal_logistic_regression_outputs_tables_formula_and_metrics():
    data = _ordered_data()
    order = ["low", "medium", "high"]

    result = run_ordinal_logistic_regression(
        data,
        y_column="satisfaction",
        category_order=order,
        x_columns=["x1", "x2", "group"],
        random_state=7,
    )

    assert not result["coefficient_table"].empty
    assert not result["threshold_table"].empty
    assert {
        "term",
        "estimate",
        "std_error",
        "z_value",
        "p_value",
        "ci_lower",
        "ci_upper",
        "odds_ratio",
    }.issubset(result["coefficient_table"].columns)
    assert {"cutpoint", "estimate", "std_error", "z_value", "p_value"}.issubset(result["threshold_table"].columns)
    assert {"accuracy", "macro_f1", "weighted_f1", "ordered_mae"}.issubset(result["test_metrics"])
    assert result["test_confusion_matrix"].shape == (3, 3)
    assert not result["test_probability_table"].empty
    assert "cumulative logit" in result["formula_latex"]["note"].lower()
    assert PROPORTIONAL_ODDS_WARNING == result["diagnostic_warning"]


def test_ordinal_regression_rejects_incomplete_order():
    data = _ordered_data()

    with pytest.raises(ValueError, match="category_order"):
        run_ordinal_logistic_regression(
            data,
            y_column="satisfaction",
            category_order=["low", "medium"],
            x_columns=["x1", "x2"],
        )


def test_ordinal_regression_rejects_two_categories():
    data = _ordered_data()
    data["satisfaction"] = data["satisfaction"].replace({"high": "medium"})

    with pytest.raises(ValueError, match="at least three"):
        run_ordinal_logistic_regression(
            data,
            y_column="satisfaction",
            category_order=["low", "medium"],
            x_columns=["x1", "x2"],
        )


def test_ordered_model_prediction_returns_probabilities_and_category():
    data = _ordered_data()
    order = ["low", "medium", "high"]
    result = run_ordinal_logistic_regression(
        data,
        y_column="satisfaction",
        category_order=order,
        x_columns=["x1", "x2", "group"],
        random_state=7,
    )

    prediction = predict_ordered_model(
        result["model"],
        pd.DataFrame([{"x1": 0.2, "x2": 0.1, "group": "a"}]),
        design_columns=result["design_columns"],
        category_order=order,
    )

    assert set(prediction["probabilities"].columns) == set(order)
    assert prediction["predicted_categories"].iloc[0] in order
    assert prediction["probabilities"].iloc[0].sum() == pytest.approx(1.0)


def test_prediction_service_supports_ordinal_artifact():
    data = _ordered_data()
    order = ["low", "medium", "high"]
    result = run_ordinal_logistic_regression(
        data,
        y_column="satisfaction",
        category_order=order,
        x_columns=["x1", "x2", "group"],
        random_state=7,
    )
    model_run = _ordinal_model_run(result)
    artifact = _ordinal_artifact(model_run, result)

    prediction = predict_from_model_run(
        model_run,
        artifact,
        {"x1": 0.2, "x2": 0.1, "group": "a"},
    )

    assert prediction["task_type"] == "ordinal_classification"
    assert prediction["predicted_class"] in order
    assert set(prediction["class_probabilities"]) == set(order)


def test_ordinal_model_comparison_table():
    result = run_ordinal_logistic_regression(
        _ordered_data(),
        y_column="satisfaction",
        category_order=["low", "medium", "high"],
        x_columns=["x1", "x2"],
        random_state=7,
    )
    model_run = _ordinal_model_run(result)

    table = build_ordinal_classification_comparison_table([model_run])

    assert len(table) == 1
    assert table.loc[0, "model_name"] == "ordinal_logistic_regression_ordered_logit"
    assert table.loc[0, "category_order"] == "low, medium, high"
    assert table.loc[0, "test_ordered_mae"] == result["test_metrics"]["ordered_mae"]


def _ordinal_model_run(result):
    return create_model_run(
        task_type="ordinal_classification",
        model_family="statistical",
        model_name="ordinal_logistic_regression_ordered_logit",
        target=result["y_column"],
        features=result["x_columns"],
        split_config={"test_size": result["test_size"]},
        preprocessing={"category_order": result["category_order"]},
        statistical_summary=result["model_statistics"],
        train_metrics=result["train_metrics"],
        test_metrics=result["test_metrics"],
        coefficient_table=result["coefficient_table"],
        formula_latex=result["formula_latex"],
    )


def _ordinal_artifact(model_run, result):
    return {
        "run_id": model_run["run_id"],
        "model_name": model_run["model_name"],
        "model_family": model_run["model_family"],
        "task_type": model_run["task_type"],
        "fitted_model": result["model"],
        "fitted_pipeline": None,
        "target": model_run["target"],
        "features": model_run["features"],
        "target_encoder": {"category_order": result["category_order"]},
        "prediction_supported": True,
        "design_columns": result["design_columns"],
    }


def _ordered_data():
    rng = np.random.default_rng(12)
    rows = []
    for label, x1_center, x2_center in [
        ("low", -1.0, -0.6),
        ("medium", 0.0, 0.0),
        ("high", 1.0, 0.6),
    ]:
        for _ in range(30):
            rows.append(
                {
                    "satisfaction": label,
                    "x1": rng.normal(x1_center, 1.2),
                    "x2": rng.normal(x2_center, 1.2),
                    "group": rng.choice(["a", "b"]),
                }
            )
    return pd.DataFrame(rows)
