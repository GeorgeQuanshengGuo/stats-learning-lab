"""Prediction intervals and probability intervals for statsmodels results."""

from __future__ import annotations

from typing import Any

import pandas as pd


OLS_INTERVAL_EXPLANATION = (
    "mean_ci_lower and mean_ci_upper are confidence intervals for the expected mean response. "
    "obs_ci_lower and obs_ci_upper are prediction intervals for a new individual observation. "
    "Prediction intervals are usually wider than mean confidence intervals."
)


def predict_ols_with_intervals(results: Any, new_data: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Predict OLS mean and interval estimates for new design rows."""
    prediction = results.get_prediction(new_data)
    frame = prediction.summary_frame(alpha=alpha)

    return pd.DataFrame(
        {
            "predicted_mean": frame["mean"],
            "mean_ci_lower": frame["mean_ci_lower"],
            "mean_ci_upper": frame["mean_ci_upper"],
            "obs_ci_lower": frame["obs_ci_lower"],
            "obs_ci_upper": frame["obs_ci_upper"],
            "alpha": alpha,
            "interval_explanation": OLS_INTERVAL_EXPLANATION,
        }
    ).reset_index(drop=True)


def predict_logit_probability_with_ci(
    results: Any,
    new_data: pd.DataFrame,
    alpha: float = 0.05,
    threshold: float = 0.5,
    positive_class: Any = 1,
    negative_class: Any = 0,
) -> pd.DataFrame:
    """Predict statsmodels Logit probabilities with CI when available."""
    predicted_probability = float(results.predict(new_data)[0])
    probability_ci_lower = None
    probability_ci_upper = None
    interval_message = None

    if hasattr(results, "get_prediction"):
        try:
            summary = results.get_prediction(new_data).summary_frame(alpha=alpha)
            probability_ci_lower, probability_ci_upper = _probability_ci_from_summary(summary)
        except Exception as error:
            interval_message = f"Probability confidence interval is not available from this fitted Logit result: {error}"
    else:
        interval_message = "Probability confidence interval is not available because this fitted Logit result does not expose get_prediction."

    if interval_message is None and (probability_ci_lower is None or probability_ci_upper is None):
        interval_message = "Probability confidence interval is not available from this fitted Logit result."

    predicted_class = positive_class if predicted_probability >= threshold else negative_class
    return pd.DataFrame(
        [
            {
                "predicted_probability": predicted_probability,
                "probability_ci_lower": probability_ci_lower,
                "probability_ci_upper": probability_ci_upper,
                "predicted_class": predicted_class,
                "threshold": threshold,
                "alpha": alpha,
                "interval_message": interval_message,
            }
        ]
    )


def unsupported_interval_result(model_name: str) -> dict[str, Any]:
    """Return a clear unsupported interval result for future or unsupported models."""
    return {
        "interval_available": False,
        "interval_message": f"Classical prediction intervals are not available for {model_name}.",
    }


def _probability_ci_from_summary(summary: pd.DataFrame) -> tuple[float | None, float | None]:
    """Extract probability CI columns from a statsmodels prediction summary."""
    lower_candidates = ["ci_lower", "mean_ci_lower", "predicted_ci_lower"]
    upper_candidates = ["ci_upper", "mean_ci_upper", "predicted_ci_upper"]
    lower_column = _first_existing_column(summary, lower_candidates)
    upper_column = _first_existing_column(summary, upper_candidates)
    if lower_column is None or upper_column is None:
        return None, None
    return float(summary[lower_column].iloc[0]), float(summary[upper_column].iloc[0])


def _first_existing_column(frame: pd.DataFrame, columns: list[str]) -> str | None:
    """Return the first column name present in a DataFrame."""
    for column in columns:
        if column in frame.columns:
            return column
    return None
