import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.modeling.machine_learning.clustering import (
    CLUSTERING_LOG_FIELDS,
    add_cluster_labels_to_dataframe,
    compute_kmeans_elbow_table,
    run_clustering_analysis,
    save_clustering_result_to_session,
)
from src.visualization.clustering_plots import (
    plot_cluster_pca_scatter,
    plot_cluster_size_bar,
    plot_kmeans_elbow,
    plot_silhouette,
)


def _cluster_data() -> pd.DataFrame:
    """Build a simple two-cluster numeric dataset."""
    return pd.DataFrame(
        {
            "x1": [0.0, 0.2, -0.1, 0.1, 5.0, 5.2, 4.9, 5.1],
            "x2": [0.0, 0.1, -0.2, 0.2, 5.0, 5.1, 4.8, 5.2],
            "x3": [1.0, 1.1, 0.9, None, 10.0, 10.2, 9.8, 10.1],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )


def test_kmeans_clustering_returns_labels_tables_and_metrics():
    data = _cluster_data()

    result = run_clustering_analysis(
        data,
        feature_columns=["x1", "x2", "x3"],
        method="KMeans",
        n_clusters=2,
        random_state=7,
    )

    assert result["method"] == "KMeans"
    assert isinstance(result["preprocessing"], Pipeline)
    assert len(result["cluster_labels"]) == len(data)
    assert result["cluster_size_table"]["count"].sum() == len(data)
    assert not result["cluster_summary_table"].empty
    assert result["metrics"]["silhouette_score"] is not None
    assert result["metrics"]["davies_bouldin_score"] is not None
    assert result["metrics"]["calinski_harabasz_score"] is not None
    assert list(result["pca_scores"].columns) == ["PC1", "PC2"]


def test_dbscan_clustering_runs_on_numeric_dataset():
    data = _cluster_data()

    result = run_clustering_analysis(
        data,
        feature_columns=["x1", "x2", "x3"],
        method="DBSCAN",
        dbscan_eps=0.6,
        dbscan_min_samples=2,
    )

    assert result["method"] == "DBSCAN"
    assert len(result["cluster_labels"]) == len(data)
    assert result["cluster_size_table"]["count"].sum() == len(data)
    assert set(result["metrics"]).issuperset(
        {"silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"}
    )


def test_agglomerative_clustering_runs():
    data = _cluster_data()

    result = run_clustering_analysis(
        data,
        feature_columns=["x1", "x2", "x3"],
        method="Agglomerative Clustering",
        n_clusters=2,
    )

    assert result["method"] == "Agglomerative Clustering"
    assert len(result["cluster_labels"].unique()) == 2


def test_clustering_does_not_modify_input_dataframe():
    data = _cluster_data()
    original = data.copy(deep=True)

    run_clustering_analysis(data, ["x1", "x2", "x3"], method="KMeans", n_clusters=2)

    pd.testing.assert_frame_equal(data, original)


def test_add_cluster_labels_only_after_confirmation_helper_is_called():
    data = _cluster_data()
    result = run_clustering_analysis(data, ["x1", "x2", "x3"], method="KMeans", n_clusters=2)

    assert "cluster_label" not in data.columns
    new_df, log_entry = add_cluster_labels_to_dataframe(data, result, new_column="cluster_label")

    assert "cluster_label" not in data.columns
    assert "cluster_label" in new_df.columns
    assert len(new_df["cluster_label"]) == len(data)
    assert list(log_entry.keys()) == CLUSTERING_LOG_FIELDS
    assert log_entry["operation_type"] == "transformation"
    assert log_entry["method"] == "cluster_labels"
    assert log_entry["source_columns"] == ["x1", "x2", "x3"]


def test_add_cluster_labels_rejects_existing_column():
    data = _cluster_data()
    data["cluster_label"] = 0
    result = run_clustering_analysis(data, ["x1", "x2", "x3"], method="KMeans", n_clusters=2)

    with pytest.raises(ValueError, match="already exists"):
        add_cluster_labels_to_dataframe(data, result, new_column="cluster_label")


def test_clustering_rejects_non_numeric_feature():
    data = _cluster_data()

    with pytest.raises(ValueError, match="numeric feature"):
        run_clustering_analysis(data, ["x1", "group"], method="KMeans", n_clusters=2)


def test_kmeans_elbow_table_and_plots_are_created():
    data = _cluster_data()
    result = run_clustering_analysis(data, ["x1", "x2", "x3"], method="KMeans", n_clusters=2)
    elbow_table = compute_kmeans_elbow_table(data, ["x1", "x2", "x3"], max_k=4)

    assert list(elbow_table.columns) == ["k", "inertia"]
    assert len(elbow_table) == 4
    assert plot_cluster_pca_scatter(result["pca_scores"], result["cluster_labels"]) is not None
    assert plot_cluster_size_bar(result["cluster_size_table"]) is not None
    assert plot_kmeans_elbow(elbow_table) is not None
    assert plot_silhouette(result["silhouette_table"]) is not None


def test_save_clustering_result_to_session_stores_latest_and_summary():
    data = _cluster_data()
    result = run_clustering_analysis(data, ["x1", "x2", "x3"], method="KMeans", n_clusters=2)
    session_state = {}

    save_clustering_result_to_session(session_state, result)

    assert session_state["latest_clustering_result"] is result
    assert len(session_state["clustering_artifacts"]) == 1
    assert session_state["clustering_artifacts"][0]["method"] == "KMeans"
