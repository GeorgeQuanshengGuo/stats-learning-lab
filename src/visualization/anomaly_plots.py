"""Plot helpers for exploratory anomaly detection results."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_anomaly_pca_scatter(pca_scores: pd.DataFrame, labels: pd.Series):
    """Build a PCA 2D scatter plot with anomalies highlighted."""
    if pca_scores is None or pca_scores.empty or "PC1" not in pca_scores.columns or "PC2" not in pca_scores.columns:
        return None

    plot_data = pca_scores[["PC1", "PC2"]].copy()
    plot_data["anomaly_status"] = labels.map({True: "Anomaly", False: "Typical"}).to_numpy()
    return px.scatter(
        plot_data,
        x="PC1",
        y="PC2",
        color="anomaly_status",
        title="Anomaly Plot: PCA 2D Projection",
        labels={"PC1": "PC1", "PC2": "PC2", "anomaly_status": "Status"},
    )


def plot_anomaly_score_distribution(result_table: pd.DataFrame):
    """Build an anomaly score histogram when scores are available."""
    if result_table is None or result_table.empty or "anomaly_score" not in result_table.columns:
        return None

    plot_data = result_table.copy()
    plot_data["anomaly_status"] = plot_data["is_anomaly"].map({True: "Anomaly", False: "Typical"})
    return px.histogram(
        plot_data,
        x="anomaly_score",
        color="anomaly_status",
        marginal="box",
        title="Anomaly Score Distribution",
        labels={"anomaly_score": "Anomaly Score", "anomaly_status": "Status"},
    )


def plot_feature_boxplots(feature_boxplot_data: pd.DataFrame):
    """Build per-feature boxplots grouped by anomaly status."""
    if feature_boxplot_data is None or feature_boxplot_data.empty:
        return None
    return px.box(
        feature_boxplot_data,
        x="feature",
        y="value",
        color="anomaly_status",
        points="outliers",
        title="Feature Distributions by Anomaly Status",
        labels={"feature": "Feature", "value": "Value", "anomaly_status": "Status"},
    )
