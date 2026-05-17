"""Plotly charts for EDA-level outlier detection."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_outlier_boxplot(df: pd.DataFrame, column: str, labels: pd.Series) -> go.Figure:
    """Create a boxplot with flagged points highlighted."""
    plot_data = _plot_frame(df, [column], labels)
    plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
    plot_data = plot_data.dropna(subset=[column])
    if plot_data.empty:
        return _empty_figure(f"No complete rows for {column}")

    figure = px.box(
        plot_data,
        y=column,
        points=False,
        title=f"Outlier Boxplot: {column}",
        template="plotly_white",
    )
    flagged = plot_data.loc[plot_data["outlier_status"] == "Flagged"]
    if not flagged.empty:
        figure.add_trace(
            go.Scatter(
                x=["flagged"] * len(flagged),
                y=flagged[column],
                mode="markers",
                name="Flagged rows",
                marker={"color": "#d62728", "size": 9},
                text=flagged["row_index"],
                hovertemplate="row_index=%{text}<br>value=%{y}<extra></extra>",
            )
        )
    return figure


def plot_outlier_histogram(
    df: pd.DataFrame,
    column: str,
    labels: pd.Series,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
) -> go.Figure:
    """Create a histogram with optional threshold lines."""
    plot_data = _plot_frame(df, [column], labels)
    plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
    plot_data = plot_data.dropna(subset=[column])
    if plot_data.empty:
        return _empty_figure(f"No complete rows for {column}")

    figure = px.histogram(
        plot_data,
        x=column,
        color="outlier_status",
        marginal="box",
        title=f"Outlier Histogram: {column}",
        template="plotly_white",
    )
    if lower_threshold is not None:
        figure.add_vline(x=lower_threshold, line_dash="dash", line_color="red", annotation_text="lower")
    if upper_threshold is not None:
        figure.add_vline(x=upper_threshold, line_dash="dash", line_color="red", annotation_text="upper")
    return figure


def plot_outlier_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    labels: pd.Series,
) -> go.Figure:
    """Create a two-variable scatter plot with flagged rows highlighted."""
    plot_data = _plot_frame(df, [x_column, y_column], labels)
    plot_data[x_column] = pd.to_numeric(plot_data[x_column], errors="coerce")
    plot_data[y_column] = pd.to_numeric(plot_data[y_column], errors="coerce")
    plot_data = plot_data.dropna(subset=[x_column, y_column])
    if plot_data.empty:
        return _empty_figure(f"No complete rows for {y_column} vs {x_column}")

    return px.scatter(
        plot_data,
        x=x_column,
        y=y_column,
        color="outlier_status",
        hover_data=["row_index"],
        title=f"Outlier Scatter: {y_column} vs {x_column}",
        template="plotly_white",
    )


def plot_outlier_pca(pca_scores: pd.DataFrame, labels: pd.Series) -> go.Figure:
    """Create a PCA 2D plot with flagged rows highlighted."""
    if pca_scores is None or pca_scores.empty or "PC1" not in pca_scores.columns or "PC2" not in pca_scores.columns:
        return _empty_figure("No PCA projection is available for selected variables.")

    plot_data = pca_scores[["PC1", "PC2"]].copy(deep=True)
    plot_data["row_index"] = plot_data.index
    plot_data["outlier_status"] = labels.reindex(plot_data.index).fillna(False).map(
        {True: "Flagged", False: "Typical"}
    )
    return px.scatter(
        plot_data,
        x="PC1",
        y="PC2",
        color="outlier_status",
        hover_data=["row_index"],
        title="Outlier PCA Projection",
        template="plotly_white",
    )


def _plot_frame(df: pd.DataFrame, columns: list[str], labels: pd.Series) -> pd.DataFrame:
    """Return a plotting copy with row index and outlier labels."""
    plot_data = df[columns].copy(deep=True)
    plot_data["row_index"] = plot_data.index
    plot_data["outlier_status"] = labels.reindex(plot_data.index).fillna(False).map(
        {True: "Flagged", False: "Typical"}
    )
    return plot_data


def _empty_figure(title: str) -> go.Figure:
    """Return an empty figure with a helpful title."""
    figure = go.Figure()
    figure.update_layout(title=title, template="plotly_white")
    return figure
