import pandas as pd
import pytest

from src.modeling.diagnostics.multicollinearity import (
    compute_condition_number,
    compute_predictor_correlation_table,
    compute_vif_table,
)
from src.visualization.diagnostic_plots import plot_vif_bar


def _correlated_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6],
            "x2": [2.1, 4.0, 6.1, 8.0, 10.1, 12.0],
            "x3": [6, 5, 3, 4, 2, 1],
            "group": ["A", "A", "B", "B", "C", "C"],
        }
    )


def test_compute_vif_table_flags_correlated_numeric_predictors():
    data = _correlated_df()

    table = compute_vif_table(data, ["x1", "x2", "x3"])

    assert {"feature", "vif", "risk_level", "message"}.issubset(table.columns)
    assert table["vif"].max() > 10
    assert "strong_warning" in set(table["risk_level"])
    assert table["message"].str.contains("multicollinearity", case=False).any()


def test_compute_vif_table_handles_categorical_one_hot_predictors():
    data = _correlated_df()

    table = compute_vif_table(data, ["x1", "group"])

    assert not table.empty
    assert any(feature.startswith("group_") for feature in table["feature"])


def test_predictor_correlation_table_returns_pairwise_encoded_correlations():
    data = _correlated_df()

    table = compute_predictor_correlation_table(data, ["x1", "x2", "group"])

    assert {"feature_1", "feature_2", "correlation", "abs_correlation"}.issubset(table.columns)
    assert table["abs_correlation"].max() > 0.9


def test_condition_number_returns_risk_metadata():
    data = _correlated_df()

    result = compute_condition_number(data, ["x1", "x2", "x3"])

    assert "condition_number" in result
    assert "risk_level" in result
    assert result["condition_number"] is not None


def test_multicollinearity_helpers_reject_missing_columns():
    data = _correlated_df()

    with pytest.raises(ValueError, match="Feature columns were not found"):
        compute_vif_table(data, ["x1", "missing"])


def test_vif_plot_returns_figure():
    table = compute_vif_table(_correlated_df(), ["x1", "x2", "x3"])

    figure = plot_vif_bar(table)

    assert figure.data
