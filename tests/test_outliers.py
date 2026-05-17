import pandas as pd
import pytest

from src.eda.outliers import (
    add_outlier_flag_column,
    detect_iqr_outliers,
    detect_modified_zscore_outliers,
    detect_zscore_outliers,
    run_outlier_detection,
)
from src.visualization.outlier_plots import (
    plot_outlier_boxplot,
    plot_outlier_histogram,
    plot_outlier_pca,
    plot_outlier_scatter,
)


def _univariate_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [10, 11, 10, 9, 10, 11, 10, 200]})


def _multivariate_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0, 0.1, -0.2, 0.2, 0.1, -0.1, 6.0, 0.0, 0.2, -0.1],
            "y": [0, -0.1, 0.1, 0.2, -0.2, 0.0, 6.2, 0.1, -0.1, 0.2],
        }
    )


def test_iqr_detects_known_extreme_value():
    data = _univariate_df()

    labels, scores, details = detect_iqr_outliers(data[["x"]], multiplier=1.5)

    assert bool(labels.iloc[-1]) is True
    assert labels.sum() == 1
    assert details["upper_fences"]["x"] < 200
    assert scores.iloc[-1] > 0


def test_z_score_detects_known_extreme_value():
    data = pd.DataFrame({"x": [1, 2, 1, 2, 1, 2, 1, 2, 100]})

    labels, scores, details = detect_zscore_outliers(data[["x"]], threshold=2.0)

    assert bool(labels.iloc[-1]) is True
    assert labels.sum() == 1
    assert details["upper_thresholds"]["x"] < 100
    assert scores.iloc[-1] > 2.0


def test_modified_z_score_uses_robust_median_and_mad():
    data = _univariate_df()

    labels, scores, details = detect_modified_zscore_outliers(data[["x"]], threshold=3.5)

    assert bool(labels.iloc[-1]) is True
    assert labels.sum() == 1
    assert details["upper_thresholds"]["x"] < 200
    assert scores.iloc[-1] > 3.5


def test_isolation_forest_returns_labels_and_scores():
    data = _multivariate_df()

    result = run_outlier_detection(
        data,
        numeric_columns=["x", "y"],
        method="Isolation Forest",
        contamination=0.1,
        random_state=42,
    )

    assert "is_outlier" in result["result_table"].columns
    assert "outlier_score" in result["result_table"].columns
    assert result["summary"]["rows_evaluated"] == len(data)
    assert result["labels"].dtype == bool


def test_lof_returns_labels_and_scores():
    data = _multivariate_df()

    result = run_outlier_detection(
        data,
        numeric_columns=["x", "y"],
        method="Local Outlier Factor",
        contamination=0.1,
        n_neighbors=3,
    )

    assert "is_outlier" in result["result_table"].columns
    assert "outlier_score" in result["result_table"].columns
    assert result["summary"]["rows_evaluated"] == len(data)


def test_mahalanobis_distance_flags_multivariate_extreme_row():
    data = _multivariate_df()

    result = run_outlier_detection(
        data,
        numeric_columns=["x", "y"],
        method="Mahalanobis distance",
        mahalanobis_quantile=0.90,
    )

    assert bool(result["labels"].iloc[6]) is True
    assert result["details"]["score"] == "squared_mahalanobis_distance"


def test_input_df_is_not_modified_in_place():
    data = _multivariate_df()
    original = data.copy(deep=True)

    run_outlier_detection(data, numeric_columns=["x", "y"], method="IQR rule")

    pd.testing.assert_frame_equal(data, original)


def test_outlier_flag_column_is_added_only_when_explicitly_requested():
    data = _univariate_df()
    result = run_outlier_detection(data, numeric_columns=["x"], method="IQR rule")

    assert "outlier_flag" not in data.columns
    new_df, log_entry = add_outlier_flag_column(data, result, new_column="outlier_flag")

    assert "outlier_flag" not in data.columns
    assert "outlier_flag" in new_df.columns
    assert new_df["outlier_flag"].sum() == 1
    assert log_entry["method"] == "outlier_flag"
    assert log_entry["new_column"] == "outlier_flag"


def test_outlier_flag_rejects_existing_column():
    data = _univariate_df()
    result = run_outlier_detection(data, numeric_columns=["x"], method="IQR rule")

    with pytest.raises(ValueError, match="Column already exists"):
        add_outlier_flag_column(data, result, new_column="x")


def test_outlier_plots_return_figures():
    data = _multivariate_df()
    result = run_outlier_detection(data, numeric_columns=["x", "y"], method="IQR rule")
    labels = result["labels"]

    assert plot_outlier_boxplot(data, "x", labels).data
    assert plot_outlier_histogram(data, "x", labels).data
    assert plot_outlier_scatter(data, "x", "y", labels).data
    assert plot_outlier_pca(result["pca_scores"], labels).data
