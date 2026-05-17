import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.prediction.statistical_intervals import (
    OLS_INTERVAL_EXPLANATION,
    predict_logit_probability_with_ci,
    predict_ols_with_intervals,
    unsupported_interval_result,
)


def test_ols_interval_output_has_required_columns():
    x_values = pd.DataFrame({"const": 1.0, "x": [1, 2, 3, 4, 5, 6, 7, 8]})
    y_values = pd.Series([2.0, 3.1, 4.0, 5.2, 6.1, 7.0, 8.2, 9.1])
    results = sm.OLS(y_values, x_values).fit()
    new_data = pd.DataFrame({"const": [1.0], "x": [4.5]})

    intervals = predict_ols_with_intervals(results, new_data, alpha=0.05)

    assert {
        "predicted_mean",
        "mean_ci_lower",
        "mean_ci_upper",
        "obs_ci_lower",
        "obs_ci_upper",
        "alpha",
        "interval_explanation",
    }.issubset(intervals.columns)
    assert intervals.loc[0, "alpha"] == 0.05
    assert intervals.loc[0, "interval_explanation"] == OLS_INTERVAL_EXPLANATION


def test_ols_prediction_interval_is_present_and_wider_than_mean_ci():
    x_values = pd.DataFrame({"const": 1.0, "x": [1, 2, 3, 4, 5, 6, 7, 8]})
    y_values = pd.Series([2.0, 3.1, 4.0, 5.2, 6.1, 7.0, 8.2, 9.1])
    results = sm.OLS(y_values, x_values).fit()
    new_data = pd.DataFrame({"const": [1.0], "x": [4.5]})

    intervals = predict_ols_with_intervals(results, new_data, alpha=0.05)
    mean_width = intervals.loc[0, "mean_ci_upper"] - intervals.loc[0, "mean_ci_lower"]
    obs_width = intervals.loc[0, "obs_ci_upper"] - intervals.loc[0, "obs_ci_lower"]

    assert np.isfinite(intervals.loc[0, "obs_ci_lower"])
    assert np.isfinite(intervals.loc[0, "obs_ci_upper"])
    assert obs_width > mean_width


def test_logit_probability_prediction_works_when_get_prediction_is_available():
    x_values = pd.DataFrame(
        {
            "const": 1.0,
            "x": [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4],
        }
    )
    y_values = pd.Series([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
    results = sm.Logit(y_values, x_values).fit(disp=False)
    new_data = pd.DataFrame({"const": [1.0], "x": [1.0]})

    prediction = predict_logit_probability_with_ci(
        results,
        new_data,
        alpha=0.05,
        threshold=0.5,
        positive_class="yes",
        negative_class="no",
    )

    assert 0 <= prediction.loc[0, "predicted_probability"] <= 1
    assert prediction.loc[0, "predicted_class"] in {"yes", "no"}
    assert prediction.loc[0, "threshold"] == 0.5
    if hasattr(results, "get_prediction"):
        assert "probability_ci_lower" in prediction.columns
        assert "probability_ci_upper" in prediction.columns


def test_unsupported_model_returns_clear_message():
    result = unsupported_interval_result("Random Forest Regressor")

    assert result["interval_available"] is False
    assert "not available" in result["interval_message"]
    assert "Random Forest Regressor" in result["interval_message"]
