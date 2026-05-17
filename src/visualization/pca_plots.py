"""Plot helpers for PCA exploratory results."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_scree(explained_variance_table: pd.DataFrame):
    """Build a scree plot from PCA explained variance ratios."""
    if explained_variance_table is None or explained_variance_table.empty:
        return None
    return px.bar(
        explained_variance_table,
        x="component",
        y="explained_variance_ratio",
        title="PCA Scree Plot",
        labels={
            "component": "Component",
            "explained_variance_ratio": "Explained Variance Ratio",
        },
    )


def plot_cumulative_variance(explained_variance_table: pd.DataFrame):
    """Build a cumulative explained variance plot."""
    if explained_variance_table is None or explained_variance_table.empty:
        return None
    return px.line(
        explained_variance_table,
        x="component",
        y="cumulative_explained_variance",
        markers=True,
        title="PCA Cumulative Explained Variance",
        labels={
            "component": "Component",
            "cumulative_explained_variance": "Cumulative Explained Variance",
        },
    )


def plot_pc1_pc2_scatter(scores: pd.DataFrame, color_values: pd.Series | None = None):
    """Build a PC1 vs PC2 scatter plot, optionally colored by a variable."""
    if scores is None or scores.empty or "PC1" not in scores.columns or "PC2" not in scores.columns:
        return None

    plot_data = scores[["PC1", "PC2"]].copy()
    color_column = None
    if color_values is not None:
        color_column = str(color_values.name or "color")
        plot_data[color_column] = color_values.astype("object").to_numpy()

    return px.scatter(
        plot_data,
        x="PC1",
        y="PC2",
        color=color_column,
        title="PCA Scores: PC1 vs PC2",
        labels={"PC1": "PC1", "PC2": "PC2"},
    )
