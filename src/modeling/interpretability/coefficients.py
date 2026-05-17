"""Coefficient summaries for interpretable sklearn models.

These helpers are intentionally descriptive, not inferential. sklearn baseline
models provide fitted coefficients, but not p-values or confidence intervals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PREDICTIVE_COEFFICIENT_NOTE = (
    "This is a predictive ML coefficient table. It does not include inferential "
    "p-values or confidence intervals. Use the statistical model module for inference."
)


def get_transformed_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Return fitted feature names after ColumnTransformer preprocessing."""
    preprocessing = _preprocessing_step(pipeline)
    if preprocessing is None:
        return []

    try:
        names = preprocessing.get_feature_names_out()
    except Exception:
        return []

    return [_clean_feature_name(str(name)) for name in names]


def extract_linear_model_coefficients(pipeline: Pipeline, target_name: str) -> pd.DataFrame | None:
    """Extract coefficients from sklearn linear regression-style estimators."""
    model = _model_step(pipeline)
    if model is None or not hasattr(model, "coef_"):
        return None

    coefficients = np.asarray(model.coef_, dtype=float).ravel()
    terms = _aligned_feature_names(pipeline, len(coefficients))
    intercept = float(np.asarray(getattr(model, "intercept_", 0.0), dtype=float).ravel()[0])

    return build_ml_coefficient_table(
        terms=terms,
        coefficients=coefficients,
        model_name=_model_display_name(model),
        target_name=target_name,
        intercept=intercept,
        task_type="regression",
        scale_numeric=_pipeline_uses_standard_scaler(pipeline),
    )


def extract_logistic_model_coefficients(
    pipeline: Pipeline,
    target_name: str,
    positive_class: Any,
) -> pd.DataFrame | None:
    """Extract coefficients and odds ratios from sklearn LogisticRegression."""
    model = _model_step(pipeline)
    if model is None or not hasattr(model, "coef_"):
        return None

    coefficients = np.asarray(model.coef_, dtype=float).ravel()
    terms = _aligned_feature_names(pipeline, len(coefficients))
    intercept = float(np.asarray(getattr(model, "intercept_", [0.0]), dtype=float).ravel()[0])

    return build_ml_coefficient_table(
        terms=terms,
        coefficients=coefficients,
        model_name=_model_display_name(model),
        target_name=target_name,
        intercept=intercept,
        task_type="binary_classification",
        positive_class=positive_class,
        scale_numeric=_pipeline_uses_standard_scaler(pipeline),
    )


def build_ml_coefficient_table(
    terms: list[str],
    coefficients: Any,
    model_name: str,
    target_name: str,
    intercept: float | None = None,
    task_type: str = "regression",
    positive_class: Any | None = None,
    scale_numeric: bool = False,
) -> pd.DataFrame:
    """Build a user-facing ML coefficient table."""
    coefficient_values = np.asarray(coefficients, dtype=float).ravel()
    scale_note = _coefficient_scale_note(scale_numeric=scale_numeric)

    rows = []
    if intercept is not None:
        rows.append(
            _coefficient_row(
                term="intercept",
                coefficient=float(intercept),
                model_name=model_name,
                scale_note=scale_note,
                task_type=task_type,
                positive_class=positive_class,
            )
        )

    for term, coefficient in zip(terms, coefficient_values):
        rows.append(
            _coefficient_row(
                term=term,
                coefficient=float(coefficient),
                model_name=model_name,
                scale_note=scale_note,
                task_type=task_type,
                positive_class=positive_class,
            )
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table.attrs["interpretation_notes"] = add_coefficient_interpretation_notes(
            table,
            scale_numeric=scale_numeric,
        )
        table.attrs["target_name"] = target_name
    return table


def add_coefficient_interpretation_notes(
    coefficient_table: pd.DataFrame | list[dict[str, Any]] | None,
    scale_numeric: bool = False,
) -> list[str]:
    """Return concise notes for a predictive ML coefficient table."""
    if coefficient_table is None:
        return []

    notes = [PREDICTIVE_COEFFICIENT_NOTE]
    if scale_numeric:
        notes.append(
            "Numeric coefficients are on the scaled feature scale. Do not interpret them as one-unit changes in the original raw feature."
        )
    else:
        notes.append(
            "Numeric coefficients are on the preprocessed feature scale. One-hot encoded categorical levels are indicator terms."
        )
    notes.append(
        "Categorical terms created by one-hot encoding compare the shown level against the omitted reference level."
    )
    return notes


def _coefficient_row(
    term: str,
    coefficient: float,
    model_name: str,
    scale_note: str,
    task_type: str,
    positive_class: Any | None,
) -> dict[str, Any]:
    """Build one row for either regression or logistic coefficients."""
    if task_type == "binary_classification":
        return {
            "term": term,
            "coefficient_log_odds": coefficient,
            "odds_ratio": float(np.exp(coefficient)),
            "absolute_coefficient": abs(coefficient),
            "model_name": model_name,
            "positive_class": positive_class,
            "coefficient_scale_note": scale_note,
        }

    return {
        "term": term,
        "coefficient": coefficient,
        "absolute_coefficient": abs(coefficient),
        "model_name": model_name,
        "coefficient_scale_note": scale_note,
    }


def _aligned_feature_names(pipeline: Pipeline, expected_count: int) -> list[str]:
    """Return feature names, falling back to generic names when needed."""
    feature_names = get_transformed_feature_names_from_pipeline(pipeline)
    if len(feature_names) != expected_count:
        return [f"feature_{index}" for index in range(expected_count)]
    return feature_names


def _coefficient_scale_note(scale_numeric: bool) -> str:
    """Return the scale note stored on every row."""
    if scale_numeric:
        return "Coefficients are on the transformed feature scale; numeric features were standardized in the sklearn Pipeline."
    return "Coefficients are on the transformed feature scale; categorical variables may appear as one-hot indicator terms."


def _pipeline_uses_standard_scaler(pipeline: Pipeline) -> bool:
    """Detect whether a fitted preprocessing step contains StandardScaler."""
    preprocessing = _preprocessing_step(pipeline)
    if preprocessing is None:
        return False

    for _, transformer, _ in getattr(preprocessing, "transformers_", []):
        if isinstance(transformer, StandardScaler):
            return True
        if hasattr(transformer, "steps"):
            for _, step in transformer.steps:
                if isinstance(step, StandardScaler):
                    return True
    return False


def _preprocessing_step(pipeline: Pipeline) -> Any | None:
    """Return the preprocessing step from a Pipeline."""
    if not hasattr(pipeline, "named_steps"):
        return None
    return pipeline.named_steps.get("preprocessing")


def _model_step(pipeline: Pipeline) -> Any | None:
    """Return the fitted model step from a Pipeline."""
    if not hasattr(pipeline, "named_steps"):
        return None
    return pipeline.named_steps.get("model")


def _clean_feature_name(feature_name: str) -> str:
    """Remove ColumnTransformer prefixes from transformed feature names."""
    if "__" in feature_name:
        return feature_name.split("__", maxsplit=1)[1]
    return feature_name


def _model_display_name(model: Any) -> str:
    """Return the sklearn estimator class name."""
    return model.__class__.__name__
