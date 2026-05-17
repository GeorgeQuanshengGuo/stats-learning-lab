"""Empirical uncertainty intervals for machine learning predictions.

These helpers are intentionally separate from the regular prediction service
because bootstrap intervals can be slow. They are empirical intervals based on
re-fitting the full sklearn Pipeline many times, not classical confidence
intervals.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.prediction.input_builder import build_single_row_input_frame


PipelineFactory = Callable[..., Pipeline]


def bootstrap_prediction_interval(
    base_pipeline_factory: PipelineFactory,
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    input_row: dict[str, Any] | pd.Series | pd.DataFrame,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """Estimate an empirical bootstrap interval for one ML regression prediction.

    The factory must return a fresh unfitted sklearn Pipeline containing both
    preprocessing and the model. Each bootstrap sample fits the full Pipeline,
    so imputation, scaling, and encoding remain inside the resampled training
    workflow.
    """
    model_df = _prepare_bootstrap_data(df, target_column, feature_columns)
    input_frame = _prepare_input_frame(input_row, feature_columns)
    _validate_bootstrap_request(model_df, n_bootstrap, alpha)

    point_prediction = _point_prediction_from_available_model(
        base_pipeline_factory=base_pipeline_factory,
        model_df=model_df,
        target_column=target_column,
        feature_columns=feature_columns,
        input_frame=input_frame,
    )

    rng = np.random.default_rng(random_state)
    predictions: list[float] = []
    for _ in range(n_bootstrap):
        sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        bootstrap_sample = model_df.sample(
            n=len(model_df),
            replace=True,
            random_state=sample_seed,
        )
        pipeline = _build_pipeline(base_pipeline_factory, bootstrap_sample)
        _validate_pipeline(pipeline)
        pipeline.fit(bootstrap_sample[feature_columns], bootstrap_sample[target_column])
        prediction = pipeline.predict(input_frame)[0]
        predictions.append(float(prediction))

    prediction_array = np.asarray(predictions, dtype=float)
    lower_bound = float(np.quantile(prediction_array, alpha / 2))
    upper_bound = float(np.quantile(prediction_array, 1 - alpha / 2))

    return {
        "point_prediction": point_prediction,
        "bootstrap_mean": float(np.mean(prediction_array)),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_bootstrap": int(n_bootstrap),
        "alpha": float(alpha),
        "interval_type": "bootstrap_empirical_interval",
        "interval_explanation": (
            "This is an empirical bootstrap prediction interval from repeated "
            "re-fitting of the full sklearn Pipeline. It is not a classical "
            "statistical confidence interval."
        ),
    }


def _prepare_bootstrap_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return a modeling copy without modifying the source DataFrame."""
    required_columns = [target_column, *feature_columns]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Required columns were not found in the dataset: {missing_text}")

    model_df = df[required_columns].copy(deep=True)
    model_df[target_column] = pd.to_numeric(model_df[target_column], errors="coerce")
    model_df = model_df.dropna(subset=[target_column])
    return model_df


def _prepare_input_frame(
    input_row: dict[str, Any] | pd.Series | pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Normalize user input into one raw-feature DataFrame row."""
    if isinstance(input_row, pd.DataFrame):
        if len(input_row) != 1:
            raise ValueError("Bootstrap prediction input_row must contain exactly one row.")
        input_values = input_row.iloc[0].to_dict()
    elif isinstance(input_row, pd.Series):
        input_values = input_row.to_dict()
    else:
        input_values = dict(input_row)
    return build_single_row_input_frame(feature_columns, input_values)


def _validate_bootstrap_request(model_df: pd.DataFrame, n_bootstrap: int, alpha: float) -> None:
    """Raise clear errors for unsupported bootstrap requests."""
    if len(model_df) < 3:
        raise ValueError("At least 3 rows with a numeric target are needed for bootstrap intervals.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _point_prediction_from_available_model(
    base_pipeline_factory: PipelineFactory,
    model_df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    input_frame: pd.DataFrame,
) -> float:
    """Use the original fitted pipeline when attached, otherwise fit one full-data pipeline."""
    fitted_pipeline = getattr(base_pipeline_factory, "fitted_pipeline", None)
    if fitted_pipeline is not None:
        _validate_pipeline(fitted_pipeline)
        return float(fitted_pipeline.predict(input_frame)[0])

    pipeline = _build_pipeline(base_pipeline_factory, model_df)
    _validate_pipeline(pipeline)
    pipeline.fit(model_df[feature_columns], model_df[target_column])
    return float(pipeline.predict(input_frame)[0])


def _build_pipeline(base_pipeline_factory: PipelineFactory, training_df: pd.DataFrame) -> Pipeline:
    """Call a pipeline factory that may accept the current training DataFrame."""
    try:
        parameter_count = len(signature(base_pipeline_factory).parameters)
    except (TypeError, ValueError):
        parameter_count = 0

    if parameter_count == 0:
        return base_pipeline_factory()
    return base_pipeline_factory(training_df)


def _validate_pipeline(pipeline: Pipeline) -> None:
    """Ensure the bootstrap estimator is the full preprocessing + model Pipeline."""
    if not isinstance(pipeline, Pipeline):
        raise ValueError("base_pipeline_factory must return an sklearn Pipeline.")
    if "preprocessing" not in pipeline.named_steps:
        raise ValueError("The sklearn Pipeline must include a 'preprocessing' step.")
    if "model" not in pipeline.named_steps:
        raise ValueError("The sklearn Pipeline must include a 'model' step.")
