"""Plotly visualization helpers for fitted decision tree Pipelines."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def create_tree_plot_figure(pipeline, feature_names: list[str] | None = None, max_depth: int = 4):
    """Create a compact Plotly decision tree figure from a fitted Pipeline."""
    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
    if model is None or not hasattr(model, "tree_"):
        return None

    feature_names = feature_names or [f"feature_{index}" for index in range(model.n_features_in_)]
    nodes = []
    edges = []
    _collect_tree_layout(
        model=model,
        feature_names=feature_names,
        node_id=0,
        depth=0,
        max_depth=max_depth,
        x_min=0.0,
        x_max=1.0,
        nodes=nodes,
        edges=edges,
    )

    edge_x = []
    edge_y = []
    node_lookup = {node["node_id"]: node for node in nodes}
    for parent_id, child_id in edges:
        parent = node_lookup[parent_id]
        child = node_lookup[child_id]
        edge_x.extend([parent["x"], child["x"], None])
        edge_y.extend([parent["y"], child["y"], None])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#9ca3af", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[node["x"] for node in nodes],
            y=[node["y"] for node in nodes],
            mode="markers+text",
            marker={"size": 28, "color": [node["color"] for node in nodes], "line": {"width": 1, "color": "#374151"}},
            text=[node["short_label"] for node in nodes],
            hovertext=[node["label"] for node in nodes],
            hoverinfo="text",
            textposition="bottom center",
            showlegend=False,
        )
    )
    figure.update_layout(
        title=f"Decision Tree Plot (depth shown: {max_depth})",
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=520,
    )
    return figure


def _collect_tree_layout(
    model,
    feature_names: list[str],
    node_id: int,
    depth: int,
    max_depth: int,
    x_min: float,
    x_max: float,
    nodes: list[dict],
    edges: list[tuple[int, int]],
) -> None:
    """Collect node positions and labels with a simple recursive layout."""
    tree = model.tree_
    x = (x_min + x_max) / 2
    is_leaf = tree.children_left[node_id] == tree.children_right[node_id]
    hidden_children = depth >= max_depth and not is_leaf
    nodes.append(
        {
            "node_id": node_id,
            "x": x,
            "y": -depth,
            "label": _node_label(model, feature_names, node_id, hidden_children),
            "short_label": "leaf" if is_leaf else f"n{node_id}",
            "color": "#86efac" if is_leaf else "#bfdbfe",
        }
    )

    if is_leaf or depth >= max_depth:
        return

    left_id = int(tree.children_left[node_id])
    right_id = int(tree.children_right[node_id])
    edges.extend([(node_id, left_id), (node_id, right_id)])
    _collect_tree_layout(model, feature_names, left_id, depth + 1, max_depth, x_min, x, nodes, edges)
    _collect_tree_layout(model, feature_names, right_id, depth + 1, max_depth, x, x_max, nodes, edges)


def _node_label(model, feature_names: list[str], node_id: int, hidden_children: bool) -> str:
    """Build a readable hover label for one tree node."""
    tree = model.tree_
    samples = int(tree.n_node_samples[node_id])
    value = np.asarray(tree.value[node_id]).ravel()
    value_text = ", ".join(f"{item:.4g}" for item in value[:5])
    if len(value) > 5:
        value_text = f"{value_text}, ..."

    if hidden_children:
        return f"node {node_id}<br>subtree hidden at selected max_depth<br>samples: {samples}<br>value: {value_text}"

    feature_index = int(tree.feature[node_id])
    if feature_index < 0:
        return f"leaf {node_id}<br>samples: {samples}<br>value: {value_text}"

    feature_name = feature_names[feature_index] if feature_index < len(feature_names) else f"feature_{feature_index}"
    threshold = float(tree.threshold[node_id])
    return f"node {node_id}<br>{feature_name} <= {threshold:.4g}<br>samples: {samples}<br>value: {value_text}"
