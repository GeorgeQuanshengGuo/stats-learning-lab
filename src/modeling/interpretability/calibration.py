"""Calibration helpers for fitted binary classification Pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline


CALIBRATION_NOTE = (
    "Calibration checks whether predicted probabilities match observed event "
    "frequencies. A well-calibrated model's points should be close to the "
    "diagonal reference line."
)


def build_binary_calibration_summary(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    y_true,
    positive_label: Any = 1,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return calibration curve, probability histogram, and Brier score."""
    _validate_calibration_inputs(pipeline, x_data, n_bins)
    y_binary = _encode_binary_target(y_true, positive_label)
    probabilities = _positive_probabilities(pipeline, x_data)

    if len(y_binary) != len(probabilities):
        raise ValueError("x_data and y_true must contain the same number of rows.")

    calibration_table = _calibration_table(y_binary, probabilities, n_bins=n_bins)
    histogram_table = _probability_histogram(probabilities, n_bins=n_bins)
    return {
        "calibration_table": calibration_table,
        "probability_histogram": histogram_table,
        "brier_score": float(brier_score_loss(y_binary, probabilities)),
        "note": CALIBRATION_NOTE,
    }


def _validate_calibration_inputs(pipeline: Pipeline, x_data: pd.DataFrame, n_bins: int) -> None:
    """Raise clear errors for unsupported calibration requests."""
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Calibration requires a fitted sklearn Pipeline.")
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Calibration requires a binary classifier with predict_proba.")
    if not isinstance(x_data, pd.DataFrame) or x_data.empty:
        raise ValueError("x_data must be a non-empty pandas DataFrame.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")


def _encode_binary_target(y_true, positive_label: Any) -> np.ndarray:
    """Encode a binary target as 0/1 using the chosen positive label."""
    series = pd.Series(y_true).dropna()
    classes = series.unique().tolist()
    if len(classes) != 2:
        raise ValueError("Calibration requires exactly two non-missing target classes.")
    if positive_label not in classes and positive_label != 1:
        raise ValueError("positive_label must be one of the two target classes.")
    if set(classes).issubset({0, 1}) and positive_label == 1:
        return series.astype(int).to_numpy()
    return (series == positive_label).astype(int).to_numpy()


def _positive_probabilities(pipeline: Pipeline, x_data: pd.DataFrame) -> np.ndarray:
    """Return probabilities for encoded class 1."""
    probabilities = pipeline.predict_proba(x_data)
    model = pipeline.named_steps.get("model")
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        positive_index = classes.index(1)
    elif len(classes) == 2:
        positive_index = 1
    else:
        raise ValueError("Calibration requires a fitted binary classifier.")
    return np.asarray(probabilities[:, positive_index], dtype=float)


def _calibration_table(y_binary: np.ndarray, probabilities: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Build observed-frequency rows by probability bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probabilities, bins[1:-1], right=True)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": int(bin_id + 1),
                "probability_lower": float(bins[bin_id]),
                "probability_upper": float(bins[bin_id + 1]),
                "mean_predicted_probability": float(np.mean(probabilities[mask])),
                "observed_frequency": float(np.mean(y_binary[mask])),
                "count": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


def _probability_histogram(probabilities: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Return predicted probability counts by bin."""
    counts, edges = np.histogram(probabilities, bins=np.linspace(0, 1, n_bins + 1))
    return pd.DataFrame(
        {
            "probability_lower": edges[:-1],
            "probability_upper": edges[1:],
            "bin_midpoint": (edges[:-1] + edges[1:]) / 2,
            "count": counts,
        }
    )
