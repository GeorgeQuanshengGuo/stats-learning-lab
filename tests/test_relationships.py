import pandas as pd
import pytest

from src.eda.relationships import (
    categorical_categorical_relationship_tables,
    datetime_numeric_relationship_summary,
    numeric_categorical_relationship_summary,
    numeric_numeric_relationship_summary,
)
from src.visualization.relationship_plots import (
    plot_categorical_relationship_bar,
    plot_datetime_numeric_line,
    plot_numeric_by_category_violin,
    plot_numeric_vs_numeric_scatter,
)


def _relationship_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, None],
            "y": [2, 3, 5, 7, 11, 13],
            "group": ["A", "A", "B", "B", "B", "A"],
            "segment": ["low", "high", "low", "high", "high", "low"],
            "date": pd.date_range("2026-01-01", periods=6, freq="D"),
        }
    )


def test_numeric_numeric_relationship_summary_includes_correlations_and_sample_size():
    data = _relationship_df()

    summary = numeric_numeric_relationship_summary(data, "x", "y")

    assert summary.loc[0, "complete_rows"] == 5
    assert summary.loc[0, "pearson_correlation"] > 0
    assert summary.loc[0, "spearman_correlation"] > 0


def test_numeric_categorical_relationship_summary_returns_grouped_statistics():
    data = _relationship_df()

    summary = numeric_categorical_relationship_summary(data, "y", "group")

    assert {"category", "count", "mean", "median", "std", "min", "max"}.issubset(summary.columns)
    assert set(summary["category"]) == {"A", "B"}
    assert summary["count"].sum() == 6


def test_categorical_categorical_relationship_tables_include_percentages_and_chi_square():
    data = _relationship_df()

    result = categorical_categorical_relationship_tables(data, "group", "segment")

    assert not result["contingency_table"].empty
    assert not result["row_percentage_table"].empty
    assert not result["column_percentage_table"].empty
    assert {"chi_square", "p_value", "degrees_of_freedom", "note"}.issubset(
        result["chi_square_summary"].columns
    )


def test_datetime_numeric_relationship_summary_returns_date_range():
    data = _relationship_df()

    summary = datetime_numeric_relationship_summary(data, "date", "y")

    assert summary.loc[0, "complete_rows"] == 6
    assert summary.loc[0, "start"] == pd.Timestamp("2026-01-01")
    assert summary.loc[0, "end"] == pd.Timestamp("2026-01-06")


def test_relationship_helpers_reject_missing_columns():
    data = _relationship_df()

    with pytest.raises(ValueError, match="Columns were not found"):
        numeric_numeric_relationship_summary(data, "x", "missing")


def test_relationship_plots_return_figures():
    data = _relationship_df()

    assert plot_numeric_vs_numeric_scatter(data, "x", "y", show_trendline=False).data
    assert plot_numeric_by_category_violin(data, "y", "group").data
    assert plot_categorical_relationship_bar(data, "group", "segment").data
    assert plot_datetime_numeric_line(data, "date", "y", rolling_window=2).data
