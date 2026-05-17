import pandas as pd
import pytest

from src.eda.correlation import (
    build_correlation_matrix,
    build_correlation_pairs_table,
    identify_high_correlation_pairs,
)
from src.visualization.correlation_plots import plot_correlation_heatmap


def _sample_numeric_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "z": [5, 4, 3, 2, 1],
            "group": ["A", "A", "B", "B", "B"],
        }
    )


def test_build_correlation_matrix_pearson_for_numeric_columns():
    data = _sample_numeric_df()

    matrix = build_correlation_matrix(data, columns=["x", "y", "z"], method="pearson")

    assert list(matrix.columns) == ["x", "y", "z"]
    assert matrix.loc["x", "y"] == pytest.approx(1.0)
    assert matrix.loc["x", "z"] == pytest.approx(-1.0)


def test_build_correlation_pairs_table_sorts_by_absolute_correlation():
    data = _sample_numeric_df()

    pairs = build_correlation_pairs_table(data, columns=["x", "y", "z"], method="spearman")

    assert {"variable_1", "variable_2", "correlation", "abs_correlation", "direction", "method"}.issubset(
        pairs.columns
    )
    assert pairs.iloc[0]["abs_correlation"] == pytest.approx(1.0)
    assert set(pairs["method"]) == {"spearman"}


def test_identify_high_correlation_pairs_finds_positive_and_negative_pairs():
    data = _sample_numeric_df()
    matrix = build_correlation_matrix(data, columns=["x", "y", "z"])

    high_pairs = identify_high_correlation_pairs(matrix, threshold=0.9)

    assert len(high_pairs) == 3
    assert "strong positive" in set(high_pairs["direction"])
    assert "strong negative" in set(high_pairs["direction"])


def test_correlation_rejects_unsupported_method_and_bad_threshold():
    data = _sample_numeric_df()

    with pytest.raises(ValueError, match="Unsupported correlation method"):
        build_correlation_matrix(data, method="distance")

    matrix = build_correlation_matrix(data, columns=["x", "y"])
    with pytest.raises(ValueError, match="threshold"):
        identify_high_correlation_pairs(matrix, threshold=1.2)


def test_correlation_heatmap_returns_plotly_figure():
    matrix = build_correlation_matrix(_sample_numeric_df(), columns=["x", "y", "z"])

    figure = plot_correlation_heatmap(matrix, threshold=0.8)

    assert figure.data
