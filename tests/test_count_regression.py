import numpy as np
import pandas as pd
import pytest

from src.core.model_comparison import build_count_regression_comparison_table
from src.core.model_run import create_model_run
from src.modeling.statistical.count_regression import (
    NEGATIVE_BINOMIAL_MODEL_NAME,
    POISSON_MODEL_NAME,
    build_overdispersion_summary,
    predict_count_with_mean_ci,
    run_count_regression,
    zero_inflation_warning,
)
from src.prediction.prediction_service import predict_from_model_run


def test_poisson_regression_outputs_tables_formula_and_metrics():
    data = _poisson_data()

    result = run_count_regression(
        data,
        y_column="count",
        x_columns=["x", "group"],
        model_type=POISSON_MODEL_NAME,
        random_state=7,
    )

    assert not result["coefficient_table"].empty
    assert {
        "term",
        "estimate",
        "std_error",
        "z_value",
        "p_value",
        "ci_lower",
        "ci_upper",
        "incidence_rate_ratio",
    }.issubset(result["coefficient_table"].columns)
    assert {"rmse", "mae", "r2", "mean_deviance"}.issubset(result["test_metrics"])
    assert "estimated" in result["formula_latex"]
    assert result["overdispersion"]["variance_mean_ratio"] >= 0


def test_negative_binomial_outputs_alpha_and_metrics():
    data = _negative_binomial_data()

    result = run_count_regression(
        data,
        y_column="count",
        x_columns=["x", "group"],
        model_type=NEGATIVE_BINOMIAL_MODEL_NAME,
        random_state=7,
    )

    statistics = {row["statistic"]: row["value"] for row in result["model_statistics"].to_dict("records")}
    assert "alpha" in statistics
    assert statistics["alpha"] >= 0
    assert {"rmse", "mae", "mean_deviance"}.issubset(result["test_metrics"])


def test_count_regression_rejects_negative_or_non_integer_target():
    data = _poisson_data()
    data.loc[0, "count"] = -1

    with pytest.raises(ValueError, match="nonnegative"):
        run_count_regression(data, "count", ["x"], POISSON_MODEL_NAME)

    data = _poisson_data()
    data["count"] = data["count"].astype(float)
    data.loc[0, "count"] = 1.5
    with pytest.raises(ValueError, match="integer-like"):
        run_count_regression(data, "count", ["x"], POISSON_MODEL_NAME)


def test_overdispersion_and_zero_warnings_work():
    counts = pd.Series([0, 0, 0, 0, 1, 2, 8, 15, 20, 25])

    overdispersion = build_overdispersion_summary(counts)
    zero_warning = zero_inflation_warning(counts, threshold=0.3)

    assert overdispersion["overdispersion_detected"] is True
    assert "Negative Binomial" in overdispersion["recommendation"]
    assert zero_warning["zero_proportion"] >= 0.3
    assert "Zero-inflated" in zero_warning["warning"]


def test_count_prediction_returns_expected_count_and_mean_ci():
    data = _poisson_data()
    result = run_count_regression(
        data,
        y_column="count",
        x_columns=["x", "group"],
        model_type=POISSON_MODEL_NAME,
        random_state=7,
    )

    prediction = predict_count_with_mean_ci(
        result["model"],
        pd.DataFrame([{"x": 0.1, "group": "a"}]),
        design_columns=result["design_columns"],
    )

    assert prediction.loc[0, "predicted_expected_count"] > 0
    assert prediction.loc[0, "mean_ci_lower"] <= prediction.loc[0, "mean_ci_upper"]
    assert "not an individual count prediction interval" in prediction.loc[0, "interval_explanation"]


def test_prediction_service_supports_count_regression_artifact():
    data = _poisson_data()
    result = run_count_regression(
        data,
        y_column="count",
        x_columns=["x", "group"],
        model_type=POISSON_MODEL_NAME,
        random_state=7,
    )
    model_run = _count_model_run(result)
    artifact = _count_artifact(model_run, result)

    prediction = predict_from_model_run(
        model_run,
        artifact,
        {"x": 0.1, "group": "a"},
    )

    assert prediction["predicted_expected_count"] > 0
    assert prediction["interval_available"] is True
    assert prediction["interval"][0]["mean_ci_lower"] <= prediction["interval"][0]["mean_ci_upper"]


def test_count_regression_model_comparison_table():
    result = run_count_regression(
        _poisson_data(),
        y_column="count",
        x_columns=["x", "group"],
        model_type=POISSON_MODEL_NAME,
        random_state=7,
    )
    model_run = _count_model_run(result)

    table = build_count_regression_comparison_table([model_run])

    assert len(table) == 1
    assert table.loc[0, "model_name"] == "poisson_regression_glm"
    assert table.loc[0, "test_rmse"] == result["test_metrics"]["rmse"]
    assert table.loc[0, "variance_mean_ratio"] == result["overdispersion"]["variance_mean_ratio"]


def _count_model_run(result):
    return create_model_run(
        task_type="count_regression",
        model_family="statistical",
        model_name="poisson_regression_glm"
        if result["model_type"] == POISSON_MODEL_NAME
        else "negative_binomial_regression",
        target=result["y_column"],
        features=result["x_columns"],
        split_config={"test_size": result["test_size"]},
        preprocessing={"model_type": result["model_type"]},
        statistical_summary=result["model_statistics"],
        train_metrics=result["train_metrics"],
        test_metrics=result["test_metrics"],
        coefficient_table=result["coefficient_table"],
        formula_latex=result["formula_latex"],
    )


def _count_artifact(model_run, result):
    return {
        "run_id": model_run["run_id"],
        "model_name": model_run["model_name"],
        "model_family": model_run["model_family"],
        "task_type": model_run["task_type"],
        "fitted_model": result["model"],
        "fitted_pipeline": None,
        "target": model_run["target"],
        "features": model_run["features"],
        "prediction_supported": True,
        "design_columns": result["design_columns"],
    }


def _poisson_data():
    rng = np.random.default_rng(21)
    rows = []
    for _ in range(120):
        x = rng.normal()
        group = rng.choice(["a", "b"])
        mean = np.exp(0.8 + 0.35 * x + (0.25 if group == "b" else 0))
        rows.append({"count": rng.poisson(mean), "x": x, "group": group})
    return pd.DataFrame(rows)


def _negative_binomial_data():
    rng = np.random.default_rng(22)
    rows = []
    alpha = 1.2
    size = 1 / alpha
    for _ in range(120):
        x = rng.normal()
        group = rng.choice(["a", "b"])
        mean = np.exp(0.8 + 0.35 * x + (0.25 if group == "b" else 0))
        probability = size / (size + mean)
        rows.append({"count": rng.negative_binomial(size, probability), "x": x, "group": group})
    return pd.DataFrame(rows)
