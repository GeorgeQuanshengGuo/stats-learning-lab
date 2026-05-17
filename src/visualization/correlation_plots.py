"""Plotly charts for correlation EDA."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
) -> go.Figure:
    """Create a diverging heatmap for a correlation matrix."""
    if corr_matrix is None or corr_matrix.empty:
        figure = go.Figure()
        figure.update_layout(title="No correlation matrix is available.", template="plotly_white")
        return figure

    hover_text = []
    for row_name in corr_matrix.index:
        hover_row = []
        for column_name in corr_matrix.columns:
            value = corr_matrix.loc[row_name, column_name]
            if pd.isna(value):
                label = "missing"
            elif abs(value) >= threshold and row_name != column_name:
                label = "strong positive" if value > 0 else "strong negative"
            else:
                label = "correlation"
            hover_row.append(f"{row_name} vs {column_name}<br>{label}: {value:.3f}" if pd.notna(value) else label)
        hover_text.append(hover_row)

    figure = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "Correlation"},
            text=hover_text,
            hoverinfo="text",
        )
    )
    figure.update_layout(
        title="Correlation Heatmap",
        template="plotly_white",
        xaxis_title="Variable",
        yaxis_title="Variable",
    )
    return figure
