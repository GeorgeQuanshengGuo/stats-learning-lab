"""Exploratory clustering helpers for numeric feature sets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLUSTERING_NOTE = (
    "Clusters are exploratory groupings from the selected algorithm and features. "
    "They are not automatically real-world categories."
)

CLUSTERING_LOG_FIELDS = [
    "operation_type",
    "method",
    "source_columns",
    "new_column",
    "parameters",
    "rows_before",
    "rows_after",
    "missing_before",
    "missing_after",
    "notes",
]


def run_clustering_analysis(
    df: pd.DataFrame,
    feature_columns: list[str],
    method: str = "KMeans",
    scale_numeric: bool = True,
    n_clusters: int = 3,
    random_state: int = 42,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    linkage: str = "ward",
) -> dict[str, Any]:
    """Fit an exploratory clustering model on selected numeric features."""
    feature_columns = list(feature_columns)
    _validate_clustering_inputs(df, feature_columns, method, n_clusters, dbscan_eps, dbscan_min_samples)
    model_df = df[feature_columns].copy(deep=True)
    model_df = model_df.apply(pd.to_numeric, errors="coerce")

    preprocessing = _build_numeric_preprocessing(scale_numeric=scale_numeric)
    processed_values = preprocessing.fit_transform(model_df)
    model = _build_clusterer(
        method=method,
        n_clusters=n_clusters,
        random_state=random_state,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        linkage=linkage,
    )
    labels = model.fit_predict(processed_values)
    label_series = pd.Series(labels, index=df.index, name="cluster")

    return {
        "method": method,
        "feature_columns": feature_columns,
        "scale_numeric": scale_numeric,
        "parameters": _parameters_for_method(
            method=method,
            n_clusters=n_clusters,
            random_state=random_state,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            linkage=linkage,
        ),
        "preprocessing": preprocessing,
        "model": model,
        "processed_values": processed_values,
        "cluster_labels": label_series,
        "cluster_size_table": build_cluster_size_table(label_series),
        "cluster_summary_table": build_cluster_summary_table(model_df, label_series),
        "metrics": compute_clustering_metrics(processed_values, labels),
        "pca_scores": build_cluster_pca_scores(processed_values, index=df.index),
        "silhouette_table": build_silhouette_table(processed_values, label_series),
        "notes": CLUSTERING_NOTE,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def compute_kmeans_elbow_table(
    df: pd.DataFrame,
    feature_columns: list[str],
    scale_numeric: bool = True,
    max_k: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return KMeans inertia values for an elbow plot."""
    feature_columns = list(feature_columns)
    _validate_clustering_inputs(df, feature_columns, "KMeans", n_clusters=2)
    model_df = df[feature_columns].copy(deep=True).apply(pd.to_numeric, errors="coerce")
    preprocessing = _build_numeric_preprocessing(scale_numeric=scale_numeric)
    processed_values = preprocessing.fit_transform(model_df)

    upper_k = max(1, min(int(max_k), len(df)))
    rows = []
    for k_value in range(1, upper_k + 1):
        model = KMeans(n_clusters=k_value, n_init=10, random_state=random_state)
        model.fit(processed_values)
        rows.append({"k": k_value, "inertia": float(model.inertia_)})
    return pd.DataFrame(rows)


def build_cluster_size_table(labels: pd.Series) -> pd.DataFrame:
    """Return cluster counts and percentages."""
    counts = labels.value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "cluster": [_cluster_label_text(label) for label in counts.index],
            "raw_cluster_label": counts.index.tolist(),
            "count": counts.to_numpy(),
            "pct": counts.to_numpy() / total if total else 0,
        }
    )


def build_cluster_summary_table(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Return feature summaries by cluster in a long table."""
    working = features.copy(deep=True)
    working["cluster"] = labels.to_numpy()
    rows = []
    for cluster_label, cluster_df in working.groupby("cluster", dropna=False):
        for feature in features.columns:
            values = cluster_df[feature]
            rows.append(
                {
                    "cluster": _cluster_label_text(cluster_label),
                    "feature": feature,
                    "count": int(values.notna().sum()),
                    "mean": float(values.mean()) if values.notna().any() else np.nan,
                    "median": float(values.median()) if values.notna().any() else np.nan,
                    "std": float(values.std()) if values.notna().sum() > 1 else np.nan,
                    "min": float(values.min()) if values.notna().any() else np.nan,
                    "max": float(values.max()) if values.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_clustering_metrics(processed_values: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Compute clustering quality metrics when label structure allows it."""
    if not _metrics_are_applicable(labels):
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
        }

    return {
        "silhouette_score": float(silhouette_score(processed_values, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(processed_values, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(processed_values, labels)),
    }


def build_cluster_pca_scores(processed_values: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Project processed features to two PCA dimensions for cluster plotting."""
    if processed_values.shape[0] < 2 or processed_values.shape[1] < 2:
        return pd.DataFrame(index=index, columns=["PC1", "PC2"])
    scores = PCA(n_components=2).fit_transform(processed_values)
    return pd.DataFrame(scores, columns=["PC1", "PC2"], index=index)


def build_silhouette_table(processed_values: np.ndarray, labels: pd.Series) -> pd.DataFrame:
    """Return per-sample silhouette values when available."""
    label_values = labels.to_numpy()
    if not _metrics_are_applicable(label_values):
        return pd.DataFrame(columns=["row_index", "cluster", "silhouette_value"])

    values = silhouette_samples(processed_values, label_values)
    return pd.DataFrame(
        {
            "row_index": labels.index,
            "cluster": [_cluster_label_text(label) for label in label_values],
            "silhouette_value": values,
        }
    )


def add_cluster_labels_to_dataframe(
    df: pd.DataFrame,
    clustering_result: dict[str, Any],
    new_column: str = "cluster_label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a new DataFrame with cluster labels and a transformation log."""
    if new_column in df.columns:
        raise ValueError(f"Column already exists: {new_column}")
    labels = clustering_result.get("cluster_labels")
    if not isinstance(labels, pd.Series) or labels.empty:
        raise ValueError("Clustering result does not contain labels to add.")

    new_df = df.copy(deep=True)
    new_df[new_column] = labels.reindex(df.index).to_numpy()
    feature_columns = list(clustering_result.get("feature_columns") or [])

    log_entry = {
        "operation_type": "transformation",
        "method": "cluster_labels",
        "source_columns": feature_columns,
        "new_column": new_column,
        "parameters": {
            "clustering_method": clustering_result.get("method"),
            "scale_numeric": bool(clustering_result.get("scale_numeric", True)),
            **dict(clustering_result.get("parameters") or {}),
        },
        "rows_before": len(df),
        "rows_after": len(new_df),
        "missing_before": int(df[feature_columns].isna().sum().sum()) if feature_columns else 0,
        "missing_after": int(new_df[new_column].isna().sum()),
        "notes": f"Added cluster labels using {clustering_result.get('method')}. {CLUSTERING_NOTE}",
    }
    return new_df, log_entry


def save_clustering_result_to_session(session_state: Any, clustering_result: dict[str, Any]) -> None:
    """Store the latest clustering artifact in Streamlit session state."""
    session_state["latest_clustering_result"] = clustering_result
    session_state.setdefault("clustering_artifacts", [])
    session_state["clustering_artifacts"].append(
        {
            "created_at": clustering_result.get("created_at"),
            "method": clustering_result.get("method"),
            "feature_columns": clustering_result.get("feature_columns"),
            "scale_numeric": clustering_result.get("scale_numeric"),
            "parameters": clustering_result.get("parameters"),
            "metrics": clustering_result.get("metrics"),
            "cluster_size_table": clustering_result.get("cluster_size_table"),
        }
    )


def _build_numeric_preprocessing(scale_numeric: bool) -> Pipeline:
    """Build numeric preprocessing for clustering."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _build_clusterer(
    method: str,
    n_clusters: int,
    random_state: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    linkage: str,
):
    """Create a clustering estimator for the selected method."""
    if method == "KMeans":
        return KMeans(n_clusters=int(n_clusters), n_init=10, random_state=random_state)
    if method == "Agglomerative Clustering":
        return AgglomerativeClustering(n_clusters=int(n_clusters), linkage=linkage)
    if method == "DBSCAN":
        return DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
    raise ValueError(f"Unsupported clustering method: {method}")


def _validate_clustering_inputs(
    df: pd.DataFrame,
    feature_columns: list[str],
    method: str,
    n_clusters: int = 2,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> None:
    """Validate feature and method choices."""
    if method not in {"KMeans", "Agglomerative Clustering", "DBSCAN"}:
        raise ValueError(f"Unsupported clustering method: {method}")
    if len(feature_columns) < 2:
        raise ValueError("Clustering needs at least two numeric features.")

    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns were not found in the dataset: {', '.join(missing_features)}")

    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"Clustering requires numeric feature columns: {', '.join(non_numeric)}")

    all_missing = [column for column in feature_columns if df[column].notna().sum() == 0]
    if all_missing:
        raise ValueError(f"Clustering cannot use all-missing numeric columns: {', '.join(all_missing)}")

    if method in {"KMeans", "Agglomerative Clustering"}:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if n_clusters > len(df):
            raise ValueError("n_clusters cannot be greater than the number of rows.")

    if method == "DBSCAN":
        if dbscan_eps <= 0:
            raise ValueError("DBSCAN eps must be greater than 0.")
        if dbscan_min_samples < 1:
            raise ValueError("DBSCAN min_samples must be at least 1.")


def _parameters_for_method(
    method: str,
    n_clusters: int,
    random_state: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    linkage: str,
) -> dict[str, Any]:
    """Return method-specific parameters for logs and artifacts."""
    if method == "DBSCAN":
        return {"eps": float(dbscan_eps), "min_samples": int(dbscan_min_samples)}
    if method == "Agglomerative Clustering":
        return {"n_clusters": int(n_clusters), "linkage": linkage}
    return {"n_clusters": int(n_clusters), "random_state": int(random_state)}


def _metrics_are_applicable(labels: np.ndarray) -> bool:
    """Return True when cluster metrics can be computed."""
    unique_labels = np.unique(labels)
    return 2 <= len(unique_labels) < len(labels)


def _cluster_label_text(label: Any) -> str:
    """Render cluster labels, using a clearer name for DBSCAN noise."""
    if label == -1:
        return "Noise (-1)"
    return str(label)
