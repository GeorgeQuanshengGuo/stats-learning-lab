"""Plotly charts for machine learning interpretability outputs."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_partial_dependence(pdp_table: pd.DataFrame, title: str = "Partial Dependence"):
    """Plot a one-feature PDP table."""
    if pdp_table is None or pdp_table.empty:
        return None
    return px.line(
        pdp_table,
        x="feature_value",
        y="average_prediction",
        markers=True,
        title=title,
        labels={
            "feature_value": "Feature value",
            "average_prediction": "Average model response",
        },
    )


def plot_ice_curves(ice_table: pd.DataFrame, title: str = "ICE Curves"):
    """Plot ICE curves for individual observations."""
    if ice_table is None or ice_table.empty:
        return None
    return px.line(
        ice_table,
        x="feature_value",
        y="prediction",
        color="observation_id",
        title=title,
        labels={
            "feature_value": "Feature value",
            "prediction": "Model response",
            "observation_id": "Observation",
        },
    )


def plot_two_feature_pdp(pdp_table: pd.DataFrame, title: str = "2D Partial Dependence"):
    """Plot a two-feature PDP heatmap."""
    if pdp_table is None or pdp_table.empty:
        return None
    return px.density_heatmap(
        pdp_table,
        x="feature_a_value",
        y="feature_b_value",
        z="average_prediction",
        histfunc="avg",
        title=title,
        labels={
            "feature_a_value": str(pdp_table["feature_a"].iloc[0]),
            "feature_b_value": str(pdp_table["feature_b"].iloc[0]),
            "average_prediction": "Average model response",
        },
    )


def plot_calibration_curve(calibration_table: pd.DataFrame, title: str = "Calibration Curve"):
    """Plot observed event frequency against predicted probability."""
    if calibration_table is None or calibration_table.empty:
        return None

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Perfect calibration",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=calibration_table["mean_predicted_probability"],
            y=calibration_table["observed_frequency"],
            mode="lines+markers",
            name="Model",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed event frequency",
        yaxis_range=[0, 1],
        xaxis_range=[0, 1],
    )
    return figure


def plot_probability_histogram(histogram_table: pd.DataFrame, title: str = "Predicted Probability Histogram"):
    """Plot predicted-probability bin counts."""
    if histogram_table is None or histogram_table.empty:
        return None
    return px.bar(
        histogram_table,
        x="bin_midpoint",
        y="count",
        title=title,
        labels={
            "bin_midpoint": "Predicted probability",
            "count": "Count",
        },
    )
