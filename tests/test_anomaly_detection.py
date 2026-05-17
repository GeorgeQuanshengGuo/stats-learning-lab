import pandas as pd
import pytest

from src.modeling.machine_learning.anomaly_detection import (
    ANOMALY_LOG_FIELDS,
    add_anomaly_flag_to_dataframe,
    run_anomaly_detection,
    save_anomaly_result_to_session,
)
from src.visualization.anomaly_plots import (
    plot_anomaly_pca_scatter,
    plot_anomaly_score_distribution,
    plot_feature_boxplots,
)


def _anomaly_data() -> pd.DataFrame:
    """Build a small numeric dataset with obvious unusual rows."""
    return pd.DataFrame(
        {
            "x1": [1.0, 1.1, 0.9, 1.2, 1.0, 0.8, 1.1, 10.0, 11.0],
            "x2": [2.0, 2.1, 1.9, 2.2, 2.0, 1.8, None, 20.0, 21.0],
            "group": ["A", "A", "A", "A", "A", "A", "A", "B", "B"],
        }
    )


def test_iqr_anomaly_detection_flags_obvious_rows():
    data = _anomaly_data()

    result = run_anomaly_detection(
        data,
        feature_columns=["x1", "x2"],
        method="IQR rule",
        iqr_multiplier=1.5,
    )

    assert result["method"] == "IQR rule"
    assert len(result["anomaly_labels"]) == len(data)
    assert result["summary"]["number_of_anomalies"] >= 1
    assert result["summary"]["anomaly_percentage"] > 0
    assert "anomaly_score" in result["result_table"].columns
    assert not result["top_anomalies"].empty
    assert list(result["pca_scores"].columns) == ["PC1", "PC2"]


def test_isolation_forest_anomaly_detection_runs():
    data = _anomaly_data()

    result = run_anomaly_detection(
        data,
        feature_columns=["x1", "x2"],
        method="Isolation Forest",
        contamination=0.2,
        random_state=7,
    )

    assert result["method"] == "Isolation Forest"
    assert len(result["anomaly_labels"]) == len(data)
    assert result["model"] is not None
    assert result["summary"]["number_of_anomalies"] >= 1
    assert result["result_table"]["anomaly_score"].notna().all()


def test_local_outlier_factor_runs():
    data = _anomaly_data()

    result = run_anomaly_detection(
        data,
        feature_columns=["x1", "x2"],
        method="Local Outlier Factor",
        contamination=0.2,
        n_neighbors=3,
    )

    assert result["method"] == "Local Outlier Factor"
    assert len(result["anomaly_labels"]) == len(data)
    assert result["summary"]["rows_evaluated"] == len(data)


def test_z_score_runs_and_returns_scores():
    data = _anomaly_data()

    result = run_anomaly_detection(
        data,
        feature_columns=["x1", "x2"],
        method="Z-score",
        z_threshold=2.0,
    )

    assert result["method"] == "Z-score"
    assert result["anomaly_scores"].notna().all()
    assert "anomaly_score" in result["result_table"].columns


def test_anomaly_detection_does_not_modify_input_dataframe():
    data = _anomaly_data()
    original = data.copy(deep=True)

    run_anomaly_detection(data, ["x1", "x2"], method="IQR rule")

    pd.testing.assert_frame_equal(data, original)


def test_add_anomaly_flag_only_after_confirmation_helper_is_called():
    data = _anomaly_data()
    result = run_anomaly_detection(data, ["x1", "x2"], method="IQR rule")

    assert "anomaly_flag" not in data.columns
    new_df, log_entry = add_anomaly_flag_to_dataframe(data, result, new_column="anomaly_flag")

    assert "anomaly_flag" not in data.columns
    assert "anomaly_flag" in new_df.columns
    assert new_df["anomaly_flag"].dtype == bool
    assert list(log_entry.keys()) == ANOMALY_LOG_FIELDS
    assert log_entry["operation_type"] == "transformation"
    assert log_entry["method"] == "anomaly_flag"
    assert log_entry["source_columns"] == ["x1", "x2"]


def test_add_anomaly_flag_rejects_existing_column():
    data = _anomaly_data()
    data["anomaly_flag"] = False
    result = run_anomaly_detection(data, ["x1", "x2"], method="IQR rule")

    with pytest.raises(ValueError, match="already exists"):
        add_anomaly_flag_to_dataframe(data, result, new_column="anomaly_flag")


def test_anomaly_detection_rejects_non_numeric_feature():
    data = _anomaly_data()

    with pytest.raises(ValueError, match="numeric feature"):
        run_anomaly_detection(data, ["x1", "group"], method="IQR rule")


def test_anomaly_plots_are_created():
    data = _anomaly_data()
    result = run_anomaly_detection(data, ["x1", "x2"], method="IQR rule")

    assert plot_anomaly_pca_scatter(result["pca_scores"], result["anomaly_labels"]) is not None
    assert plot_anomaly_score_distribution(result["result_table"]) is not None
    assert plot_feature_boxplots(result["feature_boxplot_data"]) is not None


def test_save_anomaly_result_to_session_stores_latest_and_summary():
    data = _anomaly_data()
    result = run_anomaly_detection(data, ["x1", "x2"], method="IQR rule")
    session_state = {}

    save_anomaly_result_to_session(session_state, result)

    assert session_state["latest_anomaly_result"] is result
    assert len(session_state["anomaly_artifacts"]) == 1
    assert session_state["anomaly_artifacts"][0]["method"] == "IQR rule"
