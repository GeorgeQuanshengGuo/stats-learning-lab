"""Plot helpers for exploratory clustering results."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_cluster_pca_scatter(pca_scores: pd.DataFrame, labels: pd.Series):
    """Build a PCA 2D scatter plot colored by cluster label."""
    if pca_scores is None or pca_scores.empty or "PC1" not in pca_scores.columns or "PC2" not in pca_scores.columns:
        return None

    plot_data = pca_scores[["PC1", "PC2"]].copy()
    plot_data["cluster"] = labels.astype(str).to_numpy()
    return px.scatter(
        plot_data,
        x="PC1",
        y="PC2",
        color="cluster",
        title="Cluster Plot: PCA 2D Projection",
        labels={"PC1": "PC1", "PC2": "PC2", "cluster": "Cluster"},
    )


def plot_cluster_size_bar(cluster_size_table: pd.DataFrame):
    """Build a cluster size bar chart."""
    if cluster_size_table is None or cluster_size_table.empty:
        return None
    return px.bar(
        cluster_size_table,
        x="cluster",
        y="count",
        title="Cluster Sizes",
        labels={"cluster": "Cluster", "count": "Rows"},
    )


def plot_kmeans_elbow(elbow_table: pd.DataFrame):
    """Build a KMeans elbow plot from inertia values."""
    if elbow_table is None or elbow_table.empty:
        return None
    return px.line(
        elbow_table,
        x="k",
        y="inertia",
        markers=True,
        title="KMeans Elbow Plot",
        labels={"k": "Number of Clusters (k)", "inertia": "Inertia"},
    )


def plot_silhouette(silhouette_table: pd.DataFrame):
    """Build a simple silhouette value plot."""
    if silhouette_table is None or silhouette_table.empty:
        return None

    plot_data = silhouette_table.copy()
    plot_data = plot_data.sort_values(["cluster", "silhouette_value"]).reset_index(drop=True)
    plot_data["sample_order"] = range(1, len(plot_data) + 1)
    return px.bar(
        plot_data,
        x="sample_order",
        y="silhouette_value",
        color="cluster",
        title="Silhouette Values",
        labels={
            "sample_order": "Samples ordered by cluster",
            "silhouette_value": "Silhouette Value",
            "cluster": "Cluster",
        },
    )
