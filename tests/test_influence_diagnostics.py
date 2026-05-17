import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.modeling.diagnostics.influence import (
    compute_ols_influence_table,
    identify_high_leverage_points,
    identify_large_cooks_distance,
    identify_large_dffits,
    identify_large_studentized_residuals,
    top_influential_rows,
)
from src.visualization.influence_plots import (
    plot_cooks_distance,
    plot_influence_residuals_vs_fitted,
    plot_leverage_vs_standardized_residual,
    plot_studentized_residuals,
)


def _fit_ols_with_influential_point():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 30], dtype=float)
    y = np.array([3, 5, 7, 9, 11, 13, 15, 17, -30], dtype=float)
    original_index = pd.Index([f"row_{index}" for index in range(len(x))])
    design = sm.add_constant(pd.DataFrame({"x": x}, index=original_index))
    result = sm.OLS(pd.Series(y, index=original_index), design).fit()
    return result, original_index


def test_compute_ols_influence_table_contains_required_columns_and_flags():
    result, original_index = _fit_ols_with_influential_point()

    table = compute_ols_influence_table(result, original_index=original_index)

    expected_columns = {
        "row_index",
        "fitted_value",
        "residual",
        "standardized_residual",
        "studentized_residual",
        "leverage",
        "cooks_distance",
        "dffits",
        "large_residual",
        "high_leverage",
        "high_cooks_distance",
        "influential_observation",
    }
    assert expected_columns.issubset(table.columns)
    assert table.loc[table["row_index"] == "row_8", "high_leverage"].iloc[0]
    assert table.loc[table["row_index"] == "row_8", "high_cooks_distance"].iloc[0]
    assert table.loc[table["row_index"] == "row_8", "influential_observation"].iloc[0]
    assert any(column.startswith("dfbeta_") for column in table.columns)


def test_identify_large_studentized_residuals_high_leverage_cooks_and_dffits():
    result, original_index = _fit_ols_with_influential_point()
    table = compute_ols_influence_table(result, original_index=original_index)
    p = len(result.params)
    n = int(result.nobs)

    high_leverage = identify_high_leverage_points(table, p=p, n=n, multiplier=2)
    large_cooks = identify_large_cooks_distance(table, n=n, threshold_rule="4/n")
    large_dffits = identify_large_dffits(table, p=p, n=n)
    large_residuals = identify_large_studentized_residuals(table, threshold=2)

    assert "row_8" in high_leverage["row_index"].tolist()
    assert "row_8" in large_cooks["row_index"].tolist()
    assert "row_8" in large_dffits["row_index"].tolist()
    assert not large_residuals.empty


def test_top_influential_rows_sorts_by_cooks_distance():
    result, original_index = _fit_ols_with_influential_point()
    table = compute_ols_influence_table(result, original_index=original_index)

    top_rows = top_influential_rows(table, top_n=3)

    assert len(top_rows) == 3
    assert top_rows["cooks_distance"].is_monotonic_decreasing
    assert top_rows.iloc[0]["row_index"] == "row_8"


def test_influence_plots_return_figures():
    result, original_index = _fit_ols_with_influential_point()
    table = compute_ols_influence_table(result, original_index=original_index)

    assert plot_influence_residuals_vs_fitted(table).data
    assert plot_leverage_vs_standardized_residual(table).data
    assert plot_cooks_distance(table, threshold=4 / len(table)).data
    assert plot_studentized_residuals(table).data
