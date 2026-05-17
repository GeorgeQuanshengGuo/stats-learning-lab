"""Plot helpers for transformation previews."""

import pandas as pd
import plotly.graph_objects as go


def plot_before_after_histograms(
    original_values: pd.Series,
    transformed_values: pd.Series,
    original_label: str,
    transformed_label: str,
) -> go.Figure:
    """Show before/after histograms in one interactive Plotly figure."""
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=original_values.dropna(),
            name=original_label,
            opacity=0.65,
        )
    )
    figure.add_trace(
        go.Histogram(
            x=transformed_values.dropna(),
            name=transformed_label,
            opacity=0.65,
        )
    )
    figure.update_layout(
        title="Before and after distribution",
        barmode="overlay",
        template="plotly_white",
        xaxis_title="Value",
        yaxis_title="Count",
    )
    return figure
