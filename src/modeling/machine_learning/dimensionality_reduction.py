"""Dimension reduction helpers for exploratory machine learning workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PCA_TRANSFORMATION_LOG_FIELDS = [
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


def run_pca_analysis(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_components: int = 2,
    scale_numeric: bool = True,
) -> dict[str, Any]:
    """Fit PCA on selected numeric features and return display tables.

    Missing numeric values are imputed with the median. Scaling is on by
    default because PCA is sensitive to feature scale.
    """
    feature_columns = list(feature_columns)
    _validate_pca_inputs(df, feature_columns, n_components)
    model_df = df[feature_columns].copy(deep=True)
    model_df = model_df.apply(pd.to_numeric, errors="coerce")

    pipeline = _build_pca_pipeline(n_components=n_components, scale_numeric=scale_numeric)
    scores_array = pipeline.fit_transform(model_df)
    pca = pipeline.named_steps["pca"]

    component_names = [f"PC{index}" for index in range(1, n_components + 1)]
    explained_variance_table = _build_explained_variance_table(pca, component_names)
    loadings_table = _build_loadings_table(pca, feature_columns, component_names)
    scores = pd.DataFrame(scores_array, columns=component_names, index=df.index)

    return {
        "feature_columns": feature_columns,
        "n_components": n_components,
        "scale_numeric": scale_numeric,
        "pipeline": pipeline,
        "explained_variance_table": explained_variance_table,
        "loadings_table": loadings_table,
        "scores": scores,
        "notes": (
            "PCA is an exploratory dimension reduction method. Components are mathematical "
            "summaries of variation and should not be interpreted as causal explanations."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def add_pca_scores_to_dataframe(
    df: pd.DataFrame,
    pca_result: dict[str, Any],
    score_prefix: str = "PC",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a new DataFrame with PCA score columns and a transformation log."""
    scores = pca_result.get("scores")
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        raise ValueError("PCA result does not contain score columns to add.")

    new_columns = [f"{score_prefix}{index}" for index in range(1, scores.shape[1] + 1)]
    _validate_new_columns(df, new_columns)

    new_df = df.copy(deep=True)
    for source_column, new_column in zip(scores.columns, new_columns):
        new_df[new_column] = scores[source_column].to_numpy()

    log_entry = {
        "operation_type": "transformation",
        "method": "pca_scores",
        "source_columns": list(pca_result.get("feature_columns") or []),
        "new_column": ", ".join(new_columns),
        "parameters": {
            "n_components": int(pca_result.get("n_components", scores.shape[1])),
            "scale_numeric": bool(pca_result.get("scale_numeric", True)),
            "score_columns": new_columns,
        },
        "rows_before": len(df),
        "rows_after": len(new_df),
        "missing_before": int(df[list(pca_result.get("feature_columns") or [])].isna().sum().sum())
        if pca_result.get("feature_columns")
        else 0,
        "missing_after": int(new_df[new_columns].isna().sum().sum()),
        "notes": (
            "Added PCA component score columns. PCA components summarize variation in selected numeric "
            "features and are not causal explanations."
        ),
    }
    return new_df, log_entry


def save_pca_result_to_session(session_state: Any, pca_result: dict[str, Any]) -> None:
    """Store the latest PCA artifact in Streamlit session state when available."""
    session_state["latest_pca_result"] = pca_result
    session_state.setdefault("pca_artifacts", [])
    session_state["pca_artifacts"].append(
        {
            "created_at": pca_result.get("created_at"),
            "feature_columns": pca_result.get("feature_columns"),
            "n_components": pca_result.get("n_components"),
            "scale_numeric": pca_result.get("scale_numeric"),
            "explained_variance_table": pca_result.get("explained_variance_table"),
            "loadings_table": pca_result.get("loadings_table"),
        }
    )


def build_pca_component_interpretation(
    loadings_table: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Summarize which original variables most define each PCA component.

    PCA components are weighted combinations of the selected variables. The
    variables with the largest absolute loadings are the most useful clues for
    naming or describing a component. The sign can flip between equivalent PCA
    solutions, so the positive/negative direction should be read cautiously.
    """
    if not isinstance(loadings_table, pd.DataFrame) or loadings_table.empty:
        return pd.DataFrame(
            columns=[
                "component",
                "dominant_features",
                "top_positive_features",
                "top_negative_features",
                "plain_language_summary",
            ]
        )
    if "feature" not in loadings_table.columns:
        raise ValueError("PCA loadings table must include a feature column.")

    component_columns = [column for column in loadings_table.columns if str(column).startswith("PC")]
    rows = []
    for component in component_columns:
        component_loadings = loadings_table[["feature", component]].copy()
        component_loadings[component] = pd.to_numeric(component_loadings[component], errors="coerce")
        component_loadings = component_loadings.dropna(subset=[component])
        if component_loadings.empty:
            continue

        by_abs = component_loadings.assign(abs_loading=component_loadings[component].abs())
        by_abs = by_abs.sort_values("abs_loading", ascending=False).head(top_n)
        positive = component_loadings.sort_values(component, ascending=False).head(top_n)
        negative = component_loadings.sort_values(component, ascending=True).head(top_n)

        dominant_features = _format_loading_features(by_abs, component)
        rows.append(
            {
                "component": component,
                "dominant_features": dominant_features,
                "top_positive_features": _format_loading_features(positive, component),
                "top_negative_features": _format_loading_features(negative, component),
                "plain_language_summary": (
                    f"{component} is mainly shaped by {dominant_features}. "
                    "Use these high-loading variables as clues, not as a causal label."
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_pca_pipeline(n_components: int, scale_numeric: bool) -> Pipeline:
    """Build the PCA preprocessing pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    steps.append(("pca", PCA(n_components=n_components)))
    return Pipeline(steps)


def _validate_pca_inputs(df: pd.DataFrame, feature_columns: list[str], n_components: int) -> None:
    """Validate PCA feature and component choices."""
    if not feature_columns:
        raise ValueError("Choose at least one numeric feature for PCA.")

    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns were not found in the dataset: {', '.join(missing_features)}")

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"PCA requires numeric feature columns: {', '.join(non_numeric)}")

    max_components = min(len(feature_columns), len(df))
    if n_components < 1:
        raise ValueError("PCA requires at least one component.")
    if n_components > max_components:
        raise ValueError(f"n_components cannot be greater than {max_components}.")

    all_missing = [column for column in feature_columns if df[column].notna().sum() == 0]
    if all_missing:
        raise ValueError(f"PCA cannot use all-missing numeric columns: {', '.join(all_missing)}")


def _build_explained_variance_table(pca: PCA, component_names: list[str]) -> pd.DataFrame:
    """Return explained and cumulative variance for each component."""
    explained = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": np.cumsum(explained),
        }
    )


def _build_loadings_table(pca: PCA, feature_columns: list[str], component_names: list[str]) -> pd.DataFrame:
    """Return one loading row per feature/component pair."""
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_columns,
        columns=component_names,
    )
    loadings.insert(0, "feature", loadings.index)
    return loadings.reset_index(drop=True)


def _validate_new_columns(df: pd.DataFrame, new_columns: list[str]) -> None:
    """Reject PC score names that would overwrite existing data."""
    conflicts = [column for column in new_columns if column in df.columns]
    if conflicts:
        raise ValueError(f"These PCA score columns already exist: {', '.join(conflicts)}")


def _format_loading_features(loadings: pd.DataFrame, component: str) -> str:
    """Format feature loading names for display."""
    pieces = []
    for _, row in loadings.iterrows():
        pieces.append(f"{row['feature']} ({float(row[component]):+.3f})")
    return ", ".join(pieces)
