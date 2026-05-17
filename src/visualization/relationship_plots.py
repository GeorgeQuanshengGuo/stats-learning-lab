"""Plotly charts for target-aware EDA relationships."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_numeric_vs_numeric_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    show_trendline: bool = True,
) -> go.Figure:
    """Create a scatter plot for numeric x and numeric y."""
    plot_data = df[[x_column, y_column]].copy(deep=True)
    plot_data[x_column] = pd.to_numeric(plot_data[x_column], errors="coerce")
    plot_data[y_column] = pd.to_numeric(plot_data[y_column], errors="coerce")
    plot_data = plot_data.dropna()

    if plot_data.empty:
        return _empty_figure(f"No complete numeric rows for {y_column} vs {x_column}")

    figure = px.scatter(
        plot_data,
        x=x_column,
        y=y_column,
        trendline="ols" if show_trendline and len(plot_data) >= 3 else None,
        title=f"{y_column} vs {x_column}",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return figure


def plot_numeric_by_category_box(
    df: pd.DataFrame,
    numeric_column: str,
    category_column: str,
    title: str | None = None,
) -> go.Figure:
    """Create a boxplot for a numeric variable grouped by a category."""
    plot_data = df[[numeric_column, category_column]].copy(deep=True)
    plot_data[numeric_column] = pd.to_numeric(plot_data[numeric_column], errors="coerce")
    plot_data = plot_data.dropna()

    if plot_data.empty:
        return _empty_figure(f"No complete rows for {numeric_column} by {category_column}")

    plot_data[category_column] = plot_data[category_column].astype(str)
    figure = px.box(
        plot_data,
        x=category_column,
        y=numeric_column,
        points="outliers",
        title=title or f"{numeric_column} by {category_column}",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title=category_column, yaxis_title=numeric_column)
    return figure


def plot_numeric_by_category_violin(
    df: pd.DataFrame,
    numeric_column: str,
    category_column: str,
    title: str | None = None,
) -> go.Figure:
    """Create a violin plot for a numeric variable grouped by a category."""
    plot_data = df[[numeric_column, category_column]].copy(deep=True)
    plot_data[numeric_column] = pd.to_numeric(plot_data[numeric_column], errors="coerce")
    plot_data = plot_data.dropna()

    if plot_data.empty:
        return _empty_figure(f"No complete rows for {numeric_column} by {category_column}")

    plot_data[category_column] = plot_data[category_column].astype(str)
    figure = px.violin(
        plot_data,
        x=category_column,
        y=numeric_column,
        box=True,
        points=False,
        title=title or f"{numeric_column} by {category_column}",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title=category_column, yaxis_title=numeric_column)
    return figure


def plot_categorical_relationship_bar(
    df: pd.DataFrame,
    x_column: str,
    target_column: str,
    normalize: bool = True,
) -> go.Figure:
    """Create a grouped/stacked bar chart for categorical x and categorical y."""
    plot_data = df[[x_column, target_column]].dropna().copy(deep=True)
    if plot_data.empty:
        return _empty_figure(f"No complete rows for {target_column} by {x_column}")

    plot_data[x_column] = plot_data[x_column].astype(str)
    plot_data[target_column] = plot_data[target_column].astype(str)
    counts = (
        plot_data.groupby([x_column, target_column])
        .size()
        .reset_index(name="count")
    )

    if normalize:
        row_totals = counts.groupby(x_column)["count"].transform("sum")
        counts["percent"] = counts["count"] / row_totals * 100
        y_column = "percent"
        y_title = "Row percent"
    else:
        y_column = "count"
        y_title = "Count"

    figure = px.bar(
        counts,
        x=x_column,
        y=y_column,
        color=target_column,
        barmode="stack",
        title=f"{target_column} distribution by {x_column}",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title=x_column, yaxis_title=y_title)
    return figure


def plot_datetime_numeric_line(
    df: pd.DataFrame,
    datetime_column: str,
    numeric_column: str,
    rolling_window: int | None = None,
) -> go.Figure:
    """Create a time plot for a numeric variable over a datetime variable."""
    plot_data = df[[datetime_column, numeric_column]].copy(deep=True)
    plot_data[datetime_column] = pd.to_datetime(plot_data[datetime_column], errors="coerce")
    plot_data[numeric_column] = pd.to_numeric(plot_data[numeric_column], errors="coerce")
    plot_data = plot_data.dropna().sort_values(datetime_column)

    if plot_data.empty:
        return _empty_figure(f"No complete rows for {numeric_column} over {datetime_column}")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=plot_data[datetime_column],
            y=plot_data[numeric_column],
            mode="lines+markers",
            name=numeric_column,
        )
    )
    if rolling_window and rolling_window >= 2 and len(plot_data) >= rolling_window:
        rolling = plot_data[numeric_column].rolling(window=rolling_window, min_periods=1).mean()
        figure.add_trace(
            go.Scatter(
                x=plot_data[datetime_column],
                y=rolling,
                mode="lines",
                name=f"{rolling_window}-row rolling mean",
            )
        )
    figure.update_layout(
        title=f"{numeric_column} over {datetime_column}",
        template="plotly_white",
        xaxis_title=datetime_column,
        yaxis_title=numeric_column,
    )
    return figure


def _empty_figure(title: str) -> go.Figure:
    """Return a simple empty figure with a helpful title."""
    figure = go.Figure()
    figure.update_layout(title=title, template="plotly_white")
    return figure
