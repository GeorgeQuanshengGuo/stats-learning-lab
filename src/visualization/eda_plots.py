"""Plotly charts for basic exploratory data analysis."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_numeric_histogram(df: pd.DataFrame, column: str) -> go.Figure:
    """Create an interactive histogram for a numeric column."""
    figure = px.histogram(
        df,
        x=column,
        title=f"Histogram of {column}",
        template="plotly_white",
    )
    figure.update_layout(bargap=0.05, xaxis_title=column, yaxis_title="Count")
    return figure


def plot_numeric_boxplot(df: pd.DataFrame, column: str) -> go.Figure:
    """Create an interactive boxplot for a numeric column."""
    figure = px.box(
        df,
        y=column,
        points="outliers",
        title=f"Boxplot of {column}",
        template="plotly_white",
    )
    figure.update_layout(yaxis_title=column)
    return figure


def plot_categorical_bar(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a bar chart of value counts for a categorical or discrete column."""
    counts = (
        df[column]
        .dropna()
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )
    counts[column] = counts[column].astype(str)

    figure = px.bar(
        counts,
        x=column,
        y="count",
        title=f"Value counts for {column}",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title=column, yaxis_title="Count")
    return figure


def plot_missing_bar(missing_table: pd.DataFrame) -> go.Figure:
    """Create a bar chart of missing values by variable."""
    figure = px.bar(
        missing_table,
        x="variable",
        y="missing_count",
        hover_data=["missing_pct"],
        title="Missing values by variable",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title="Variable", yaxis_title="Missing count")
    return figure


def plot_datetime_counts(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a line chart of observation counts over time."""
    datetime_values = pd.to_datetime(df[column], errors="coerce", format="mixed").dropna()

    if datetime_values.empty:
        return _empty_figure(f"No valid datetime values found for {column}")

    counts = (
        datetime_values.dt.floor("D")
        .value_counts()
        .sort_index()
        .rename_axis("date")
        .reset_index(name="count")
    )

    figure = px.line(
        counts,
        x="date",
        y="count",
        markers=True,
        title=f"Observations over time for {column}",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title=column, yaxis_title="Count")
    return figure


def _empty_figure(title: str) -> go.Figure:
    """Return a simple empty figure with a helpful title."""
    figure = go.Figure()
    figure.update_layout(title=title, template="plotly_white")
    return figure
