"""Feature-importance helpers for fitted machine learning Pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return output feature names from a fitted preprocessing Pipeline step."""
    preprocessing = _preprocessing_step(pipeline)
    if preprocessing is None:
        return []

    try:
        names = preprocessing.get_feature_names_out()
    except Exception:
        return []

    return [_clean_feature_name(str(name)) for name in names]


def compute_tree_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return tree-based feature importances when the fitted model exposes them."""
    model = _model_step(pipeline)
    if model is None or not hasattr(model, "feature_importances_"):
        return _empty_importance_table()

    importances = model.feature_importances_
    feature_names = get_transformed_feature_names(pipeline)
    if not feature_names or len(feature_names) != len(importances):
        feature_names = [f"feature_{index}" for index in range(len(importances))]

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "importance_std": None,
            "importance_type": "tree_based",
        }
    )
    return _sort_importance_table(table)


def compute_permutation_importance_table(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test,
    scoring: str | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on test data for a fitted Pipeline."""
    if x_test.empty:
        return _empty_importance_table()

    result = permutation_importance(
        pipeline,
        x_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    table = pd.DataFrame(
        {
            "feature": list(x_test.columns),
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
            "importance_type": "permutation",
        }
    )
    return _sort_importance_table(table)


def combine_feature_importance_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine non-empty importance tables into one display-friendly table."""
    non_empty_tables = [table for table in tables if table is not None and not table.empty]
    if not non_empty_tables:
        return _empty_importance_table()
    combined = pd.concat(non_empty_tables, ignore_index=True)
    combined["_importance_order"] = combined["importance_type"].map({"tree_based": 0, "permutation": 1}).fillna(2)
    combined = combined.sort_values(["_importance_order", "importance"], ascending=[True, False])
    return combined.drop(columns=["_importance_order"]).reset_index(drop=True)


def importance_type_label(importance_table: pd.DataFrame) -> str | None:
    """Return a compact label for the importance types in a table."""
    if importance_table is None or importance_table.empty:
        return None
    available_types = importance_table["importance_type"].dropna().unique().tolist()
    importance_types = [
        importance_type
        for importance_type in ["tree_based", "permutation"]
        if importance_type in available_types
    ]
    return "_and_".join(importance_types)


def _preprocessing_step(pipeline: Pipeline) -> Any | None:
    """Return the fitted preprocessing step from a Pipeline if present."""
    if not hasattr(pipeline, "named_steps"):
        return None
    return pipeline.named_steps.get("preprocessing")


def _model_step(pipeline: Pipeline) -> Any | None:
    """Return the final model step from a Pipeline if present."""
    if not hasattr(pipeline, "named_steps"):
        return None
    return pipeline.named_steps.get("model")


def _clean_feature_name(feature_name: str) -> str:
    """Remove ColumnTransformer prefixes while keeping one-hot category detail."""
    if "__" in feature_name:
        return feature_name.split("__", maxsplit=1)[1]
    return feature_name


def _sort_importance_table(table: pd.DataFrame) -> pd.DataFrame:
    """Sort feature importance from largest to smallest."""
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def _empty_importance_table() -> pd.DataFrame:
    """Return an empty table with stable columns."""
    return pd.DataFrame(columns=["feature", "importance", "importance_std", "importance_type"])
