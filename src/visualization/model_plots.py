"""Plot helpers for model result displays."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_feature_importance_bar(
    importance_table: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 20,
):
    """Build a horizontal feature-importance bar chart."""
    if importance_table is None or importance_table.empty:
        return None

    plot_data = importance_table.sort_values("importance", ascending=False).head(top_n)
    plot_data = plot_data.sort_values("importance", ascending=True)
    return px.bar(
        plot_data,
        x="importance",
        y="feature",
        orientation="h",
        color="importance_type" if "importance_type" in plot_data.columns else None,
        title=title,
        labels={
            "feature": "Feature",
            "importance": "Importance",
            "importance_type": "Importance Type",
        },
    )
