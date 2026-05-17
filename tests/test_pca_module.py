import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.modeling.machine_learning.dimensionality_reduction import (
    PCA_TRANSFORMATION_LOG_FIELDS,
    add_pca_scores_to_dataframe,
    build_pca_component_interpretation,
    run_pca_analysis,
    save_pca_result_to_session,
)
from src.visualization.pca_plots import (
    plot_cumulative_variance,
    plot_pc1_pc2_scatter,
    plot_scree,
)


def _pca_data() -> pd.DataFrame:
    """Build a small numeric dataset with one missing value."""
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": [2.0, 4.0, 6.0, 8.0, None, 12.0],
            "x3": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "group": ["A", "A", "B", "B", "C", "C"],
        }
    )


def test_run_pca_analysis_returns_expected_tables():
    data = _pca_data()

    result = run_pca_analysis(
        data,
        feature_columns=["x1", "x2", "x3"],
        n_components=2,
        scale_numeric=True,
    )

    assert isinstance(result["pipeline"], Pipeline)
    assert list(result["explained_variance_table"].columns) == [
        "component",
        "explained_variance_ratio",
        "cumulative_explained_variance",
    ]
    assert list(result["loadings_table"].columns) == ["feature", "PC1", "PC2"]
    assert result["scores"].shape == (6, 2)
    assert result["scores"].isna().sum().sum() == 0
    assert result["explained_variance_table"]["cumulative_explained_variance"].iloc[-1] <= 1.0


def test_pca_input_df_is_not_modified():
    data = _pca_data()
    original = data.copy(deep=True)

    run_pca_analysis(data, ["x1", "x2", "x3"], n_components=2)

    pd.testing.assert_frame_equal(data, original)


def test_pca_rejects_non_numeric_features():
    data = _pca_data()

    with pytest.raises(ValueError, match="numeric feature"):
        run_pca_analysis(data, ["x1", "group"], n_components=2)


def test_pca_rejects_too_many_components():
    data = _pca_data()

    with pytest.raises(ValueError, match="n_components"):
        run_pca_analysis(data, ["x1", "x2"], n_components=3)


def test_add_pca_scores_to_dataframe_creates_new_columns_and_log():
    data = _pca_data()
    result = run_pca_analysis(data, ["x1", "x2", "x3"], n_components=2)

    new_df, log_entry = add_pca_scores_to_dataframe(data, result, score_prefix="PC")

    assert "PC1" in new_df.columns
    assert "PC2" in new_df.columns
    assert "PC1" not in data.columns
    assert list(log_entry.keys()) == PCA_TRANSFORMATION_LOG_FIELDS
    assert log_entry["operation_type"] == "transformation"
    assert log_entry["method"] == "pca_scores"
    assert log_entry["source_columns"] == ["x1", "x2", "x3"]
    assert log_entry["parameters"]["score_columns"] == ["PC1", "PC2"]


def test_add_pca_scores_rejects_existing_columns():
    data = _pca_data()
    data["PC1"] = 0.0
    result = run_pca_analysis(data, ["x1", "x2", "x3"], n_components=2)

    with pytest.raises(ValueError, match="already exist"):
        add_pca_scores_to_dataframe(data, result, score_prefix="PC")


def test_pca_plots_are_created():
    data = _pca_data()
    result = run_pca_analysis(data, ["x1", "x2", "x3"], n_components=2)

    assert plot_scree(result["explained_variance_table"]) is not None
    assert plot_cumulative_variance(result["explained_variance_table"]) is not None
    assert plot_pc1_pc2_scatter(result["scores"], color_values=data["group"]) is not None


def test_build_pca_component_interpretation_summarizes_top_loadings():
    data = _pca_data()
    result = run_pca_analysis(data, ["x1", "x2", "x3"], n_components=2)

    interpretation = build_pca_component_interpretation(result["loadings_table"], top_n=2)

    assert list(interpretation["component"]) == ["PC1", "PC2"]
    assert "dominant_features" in interpretation.columns
    assert interpretation["plain_language_summary"].str.contains("PC").all()
    assert interpretation["dominant_features"].str.len().min() > 0


def test_save_pca_result_to_session_stores_latest_and_artifact_summary():
    data = _pca_data()
    result = run_pca_analysis(data, ["x1", "x2", "x3"], n_components=2)
    session_state = {}

    save_pca_result_to_session(session_state, result)

    assert session_state["latest_pca_result"] is result
    assert len(session_state["pca_artifacts"]) == 1
    assert session_state["pca_artifacts"][0]["feature_columns"] == ["x1", "x2", "x3"]
