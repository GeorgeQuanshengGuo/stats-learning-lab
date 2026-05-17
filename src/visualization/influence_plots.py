"""Plotly charts for OLS influence diagnostics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_influence_residuals_vs_fitted(influence_table: pd.DataFrame) -> go.Figure:
    """Plot residuals vs fitted values with influential observations highlighted."""
    if _empty(influence_table):
        return _empty_figure("No influence table is available.")
    return px.scatter(
        influence_table,
        x="fitted_value",
        y="residual",
        color="influential_observation",
        hover_data=["row_index", "studentized_residual", "leverage", "cooks_distance"],
        title="Residuals vs Fitted with Influence Flags",
        template="plotly_white",
    ).add_hline(y=0, line_dash="dash", line_color="gray")


def plot_leverage_vs_standardized_residual(influence_table: pd.DataFrame) -> go.Figure:
    """Plot leverage against standardized residuals."""
    if _empty(influence_table):
        return _empty_figure("No influence table is available.")
    figure = px.scatter(
        influence_table,
        x="leverage",
        y="standardized_residual",
        color="influential_observation",
        hover_data=["row_index", "studentized_residual", "cooks_distance", "dffits"],
        title="Leverage vs Standardized Residual",
        template="plotly_white",
    )
    figure.add_hline(y=0, line_dash="dash", line_color="gray")
    return figure


def plot_cooks_distance(influence_table: pd.DataFrame, threshold: float | None = None) -> go.Figure:
    """Plot Cook's distance by row position."""
    if _empty(influence_table):
        return _empty_figure("No influence table is available.")
    plot_data = influence_table.reset_index(drop=True).copy()
    plot_data["observation_number"] = plot_data.index
    figure = px.bar(
        plot_data,
        x="observation_number",
        y="cooks_distance",
        color="high_cooks_distance",
        hover_data=["row_index", "studentized_residual", "leverage"],
        title="Cook's Distance",
        template="plotly_white",
    )
    if threshold is not None:
        figure.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="threshold")
    return figure


def plot_studentized_residuals(influence_table: pd.DataFrame, threshold: float = 3) -> go.Figure:
    """Plot studentized residuals by row position."""
    if _empty(influence_table):
        return _empty_figure("No influence table is available.")
    plot_data = influence_table.reset_index(drop=True).copy()
    plot_data["observation_number"] = plot_data.index
    figure = px.scatter(
        plot_data,
        x="observation_number",
        y="studentized_residual",
        color="large_residual",
        hover_data=["row_index", "fitted_value", "residual", "cooks_distance"],
        title="Studentized Residual Index Plot",
        template="plotly_white",
    )
    figure.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"+{threshold}")
    figure.add_hline(y=-threshold, line_dash="dash", line_color="red", annotation_text=f"-{threshold}")
    figure.add_hline(y=0, line_dash="dash", line_color="gray")
    return figure


def _empty(influence_table: pd.DataFrame) -> bool:
    """Return True when plotting data is unavailable."""
    return influence_table is None or influence_table.empty


def _empty_figure(title: str) -> go.Figure:
    """Return a blank figure with a helpful title."""
    figure = go.Figure()
    figure.update_layout(title=title, template="plotly_white")
    return figure
