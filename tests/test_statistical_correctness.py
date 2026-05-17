"""Numerical correctness checks against trusted statistical libraries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

from src.eda.correlation import (
    build_correlation_matrix,
    build_correlation_pairs_table,
    identify_high_correlation_pairs,
)
from src.eda.missing import build_missing_table, build_row_missing_summary
from src.modeling.diagnostics.diagnostic_rules import vif_warning
from src.modeling.diagnostics.influence import (
    compute_ols_influence_table,
    identify_high_leverage_points,
    identify_large_cooks_distance,
    identify_large_studentized_residuals,
)
from src.modeling.diagnostics.multicollinearity import compute_vif_table
from src.modeling.statistical.linear_regression import (
    build_coefficient_table,
    build_model_statistics,
)
from src.modeling.statistical.logistic_regression import (
    build_logistic_coefficient_table,
    build_logistic_model_statistics,
    classification_metrics,
)
from src.prediction.statistical_intervals import predict_ols_with_intervals


def _ols_results():
    """Fit a small stable OLS model directly with statsmodels."""
    df = pd.DataFrame(
        {
            "y": [3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.2, 16.8, 19.0, 21.1, 22.8, 25.2],
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "x2": [2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
        }
    )
    x = sm.add_constant(df[["x1", "x2"]])
    return sm.OLS(df["y"], x).fit(), x


def _logit_results():
    """Fit a small stable Logit model directly with statsmodels."""
    df = pd.DataFrame(
        {
            "y": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            "x1": [0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.7, 1.9, 2.1, 2.4, 2.6, 2.9, 3.1, 3.4],
            "x2": [1.2, 0.8, 1.0, 1.5, 1.1, 1.7, 1.8, 1.4, 2.2, 1.9, 2.4, 2.5, 2.1, 2.8],
        }
    )
    x = sm.add_constant(df[["x1", "x2"]])
    return sm.Logit(df["y"], x).fit(disp=False), x, df["y"]


def test_ols_coefficient_table_matches_statsmodels_results():
    results, _ = _ols_results()

    table = build_coefficient_table(results).set_index("term")

    for term in results.params.index:
        assert table.loc[term, "estimate"] == pytest.approx(results.params[term], rel=1e-10)
        assert table.loc[term, "std_error"] == pytest.approx(results.bse[term], rel=1e-10)
        assert table.loc[term, "t_value"] == pytest.approx(results.tvalues[term], rel=1e-10)
        assert table.loc[term, "p_value"] == pytest.approx(results.pvalues[term], rel=1e-10)
        ci_lower, ci_upper = results.conf_int().loc[term]
        assert table.loc[term, "ci_lower"] == pytest.approx(ci_lower, rel=1e-10)
        assert table.loc[term, "ci_upper"] == pytest.approx(ci_upper, rel=1e-10)


def test_ols_model_statistics_match_statsmodels_results():
    results, _ = _ols_results()

    stats_table = build_model_statistics(results).set_index("statistic")["value"]

    assert stats_table["R-squared"] == pytest.approx(results.rsquared, rel=1e-10)
    assert stats_table["Adjusted R-squared"] == pytest.approx(results.rsquared_adj, rel=1e-10)
    assert stats_table["F-statistic"] == pytest.approx(results.fvalue, rel=1e-10)
    assert stats_table["AIC"] == pytest.approx(results.aic, rel=1e-10)
    assert stats_table["BIC"] == pytest.approx(results.bic, rel=1e-10)


def test_ols_prediction_intervals_match_statsmodels_summary_frame():
    results, x = _ols_results()
    new_data = x.iloc[[0, 5]].copy()

    app_intervals = predict_ols_with_intervals(results, new_data, alpha=0.1)
    trusted = results.get_prediction(new_data).summary_frame(alpha=0.1).reset_index(drop=True)

    for column, trusted_column in [
        ("predicted_mean", "mean"),
        ("mean_ci_lower", "mean_ci_lower"),
        ("mean_ci_upper", "mean_ci_upper"),
        ("obs_ci_lower", "obs_ci_lower"),
        ("obs_ci_upper", "obs_ci_upper"),
    ]:
        np.testing.assert_allclose(app_intervals[column], trusted[trusted_column], rtol=1e-10)
    assert (app_intervals["obs_ci_upper"] - app_intervals["obs_ci_lower"]).gt(
        app_intervals["mean_ci_upper"] - app_intervals["mean_ci_lower"]
    ).all()


def test_logistic_coefficient_table_and_odds_ratios_match_statsmodels():
    results, _, _ = _logit_results()

    table = build_logistic_coefficient_table(results).set_index("term")

    for term in results.params.index:
        assert table.loc[term, "estimate"] == pytest.approx(results.params[term], rel=1e-10)
        assert table.loc[term, "std_error"] == pytest.approx(results.bse[term], rel=1e-10)
        assert table.loc[term, "z_value"] == pytest.approx(results.tvalues[term], rel=1e-10)
        assert table.loc[term, "p_value"] == pytest.approx(results.pvalues[term], rel=1e-10)
        assert table.loc[term, "odds_ratio"] == pytest.approx(np.exp(results.params[term]), rel=1e-10)


def test_logistic_model_statistics_and_probabilities_match_statsmodels():
    results, x, _ = _logit_results()

    stats_table = build_logistic_model_statistics(results).set_index("statistic")["value"]
    probabilities = results.predict(x)

    assert stats_table["AIC"] == pytest.approx(results.aic, rel=1e-10)
    assert stats_table["BIC"] == pytest.approx(results.bic, rel=1e-10)
    assert stats_table["Pseudo R-squared"] == pytest.approx(results.prsquared, rel=1e-10)
    assert probabilities.between(0, 1).all()
    assert probabilities.iloc[0] == pytest.approx(float(results.predict(x.iloc[[0]])[0]), rel=1e-10)


def test_logistic_classification_metrics_match_sklearn():
    results, x, y_true = _logit_results()
    probabilities = results.predict(x)
    threshold = 0.5
    predicted = (probabilities >= threshold).astype(int)

    metrics = classification_metrics(y_true, probabilities, threshold)

    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, predicted), rel=1e-10)
    assert metrics["precision"] == pytest.approx(precision_score(y_true, predicted, zero_division=0), rel=1e-10)
    assert metrics["recall"] == pytest.approx(recall_score(y_true, predicted, zero_division=0), rel=1e-10)
    assert metrics["F1"] == pytest.approx(f1_score(y_true, predicted, zero_division=0), rel=1e-10)
    assert metrics["ROC AUC"] == pytest.approx(roc_auc_score(y_true, probabilities), rel=1e-10)


def test_correlation_helpers_match_pandas_and_scipy():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 4, 6, 8, 10, 12],
            "z": [6, 5, 4, 3, 2, 1],
            "category": ["a", "b", "a", "b", "a", "b"],
        }
    )

    pearson = build_correlation_matrix(df, ["x", "y", "z"], method="pearson")
    spearman = build_correlation_matrix(df, ["x", "z"], method="spearman")
    pairs = build_correlation_pairs_table(df, ["x", "y", "z"], method="pearson")
    high_pairs = identify_high_correlation_pairs(pearson, threshold=0.95)

    pd.testing.assert_frame_equal(pearson, df[["x", "y", "z"]].corr(method="pearson"))
    assert spearman.loc["x", "z"] == pytest.approx(stats.spearmanr(df["x"], df["z"]).correlation, rel=1e-10)
    assert pairs.iloc[0]["abs_correlation"] == pytest.approx(1.0)
    assert set(high_pairs["direction"]) == {"strong positive", "strong negative"}


def test_vif_values_and_warning_thresholds_match_statsmodels():
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": [1.1, 2.1, 2.9, 4.2, 5.1, 5.8, 7.2, 8.1],
            "x3": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    design = df[["x1", "x2", "x3"]].astype(float)
    trusted = {
        column: variance_inflation_factor(design.to_numpy(dtype=float), index)
        for index, column in enumerate(design.columns)
    }

    table = compute_vif_table(df, ["x1", "x2", "x3"]).set_index("feature")

    for feature, trusted_value in trusted.items():
        assert table.loc[feature, "vif"] == pytest.approx(trusted_value, rel=1e-10)

    assert vif_warning(4.9)[0] == "ok"
    assert vif_warning(5.1)[0] == "warning"
    assert vif_warning(10.1)[0] == "strong_warning"


def test_ols_influence_table_matches_statsmodels_influence():
    results, _ = _ols_results()
    trusted = OLSInfluence(results)

    table = compute_ols_influence_table(results)

    np.testing.assert_allclose(table["leverage"], trusted.hat_matrix_diag, rtol=1e-10)
    np.testing.assert_allclose(table["cooks_distance"], trusted.cooks_distance[0], rtol=1e-10)
    np.testing.assert_allclose(table["standardized_residual"], trusted.resid_studentized_internal, rtol=1e-10)
    assert {"large_residual", "high_leverage", "high_cooks_distance", "influential_observation"}.issubset(table.columns)

    p = len(results.params)
    n = int(results.nobs)
    assert identify_large_studentized_residuals(table, threshold=0).shape[0] == table.shape[0]
    assert identify_high_leverage_points(table, p=p, n=n, multiplier=2).equals(table.loc[table["leverage"] > (2 * p / n)])
    assert identify_large_cooks_distance(table, n=n).equals(table.loc[table["cooks_distance"] > (4 / n)])


def test_missing_value_summaries_match_pandas_counts():
    df = pd.DataFrame(
        {
            "a": [1, None, 3, None],
            "b": ["x", "y", None, None],
            "c": [None, None, None, None],
        }
    )

    missing_table = build_missing_table(df).set_index("variable")
    row_summary = build_row_missing_summary(df).set_index("row_number")

    expected_missing = df.isna().sum()
    for column in df.columns:
        assert missing_table.loc[column, "missing_count"] == expected_missing[column]
        assert missing_table.loc[column, "missing_pct"] == pytest.approx(expected_missing[column] / len(df) * 100)

    expected_row_missing = df.isna().sum(axis=1)
    for row_number, missing_count in enumerate(expected_row_missing, start=1):
        assert row_summary.loc[row_number, "missing_count"] == missing_count
        assert row_summary.loc[row_number, "missing_pct"] == pytest.approx(missing_count / len(df.columns) * 100)
