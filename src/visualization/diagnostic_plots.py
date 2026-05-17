"""Plotly charts for model diagnostics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_overfitting_metric_gaps(diagnostics_table: pd.DataFrame) -> go.Figure:
    """Plot overfitting diagnostic values as a compact bar chart."""
    if diagnostics_table is None or diagnostics_table.empty:
        return _empty_figure("No overfitting diagnostics are available.")
    plot_data = diagnostics_table.dropna(subset=["value"]).copy(deep=True)
    if plot_data.empty:
        return _empty_figure("No numeric overfitting diagnostic values are available.")

    figure = px.bar(
        plot_data,
        x="metric",
        y="value",
        color="risk_level",
        hover_data=["message"],
        title="Overfitting Diagnostics",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title="Metric", yaxis_title="Value")
    return figure


def plot_vif_bar(vif_table: pd.DataFrame) -> go.Figure:
    """Plot VIF values for predictor design matrix columns."""
    if vif_table is None or vif_table.empty or "vif" not in vif_table.columns:
        return _empty_figure("No VIF values are available.")
    plot_data = vif_table.dropna(subset=["vif"]).copy(deep=True)
    if plot_data.empty:
        return _empty_figure("No numeric VIF values are available.")

    figure = px.bar(
        plot_data,
        x="feature",
        y="vif",
        color="risk_level",
        hover_data=["message"],
        title="Variance Inflation Factors",
        template="plotly_white",
    )
    figure.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="VIF = 5")
    figure.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="VIF = 10")
    figure.update_layout(xaxis_title="Encoded predictor", yaxis_title="VIF")
    return figure


def plot_learning_curve_table(learning_curve_table: pd.DataFrame) -> go.Figure:
    """Plot train and validation scores from sklearn learning_curve output."""
    if learning_curve_table is None or learning_curve_table.empty:
        return _empty_figure("No learning curve data is available.")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=learning_curve_table["train_size"],
            y=learning_curve_table["train_score_mean"],
            mode="lines+markers",
            name="Train score",
            error_y={"type": "data", "array": learning_curve_table.get("train_score_std")},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=learning_curve_table["train_size"],
            y=learning_curve_table["validation_score_mean"],
            mode="lines+markers",
            name="Validation score",
            error_y={"type": "data", "array": learning_curve_table.get("validation_score_std")},
        )
    )
    figure.update_layout(
        title="Learning Curve",
        template="plotly_white",
        xaxis_title="Training rows",
        yaxis_title="Score",
    )
    return figure


def _empty_figure(title: str) -> go.Figure:
    """Return an empty figure with a helpful title."""
    figure = go.Figure()
    figure.update_layout(title=title, template="plotly_white")
    return figure
