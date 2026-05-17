"""Statsmodels count regression helpers for Poisson and Negative Binomial models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import NegativeBinomial

from src.modeling.metrics import compute_regression_metrics
from src.reporting.formula_builder import (
    build_estimated_poisson_formula,
    build_symbolic_poisson_formula,
)


POISSON_MODEL_NAME = "Poisson Regression"
NEGATIVE_BINOMIAL_MODEL_NAME = "Negative Binomial Regression"
COUNT_MODEL_OPTIONS = [POISSON_MODEL_NAME, NEGATIVE_BINOMIAL_MODEL_NAME]


def run_count_regression(
    df: pd.DataFrame,
    y_column: str,
    x_columns: list[str],
    model_type: str = POISSON_MODEL_NAME,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit a Poisson or Negative Binomial count regression model."""
    _validate_inputs(df, y_column, x_columns, model_type)
    model_df = _prepare_model_data(df, y_column, x_columns)

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
    )
    x_train = _design_matrix(train_df[x_columns])
    x_test = _align_design_matrix(test_df[x_columns], x_train.columns)
    y_train = train_df[y_column]
    y_test = test_df[y_column]

    fitted = _fit_count_model(y_train, x_train, model_type)
    train_predictions = _predict_expected_count(fitted, x_train)
    test_predictions = _predict_expected_count(fitted, x_test)
    alpha = _alpha_parameter(fitted, model_type)
    coefficient_table = build_count_coefficient_table(fitted, model_type)
    overdispersion = build_overdispersion_summary(model_df[y_column], fitted, y_train, x_train, alpha)
    zero_warning = zero_inflation_warning(model_df[y_column])
    model_statistics = _model_statistics(fitted, model_type, overdispersion, zero_warning, alpha)

    return {
        "model": fitted,
        "model_type": model_type,
        "y_column": y_column,
        "x_columns": x_columns,
        "test_size": test_size,
        "random_state": random_state,
        "rows_used": len(model_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "design_columns": list(x_train.columns),
        "coefficient_table": coefficient_table,
        "model_statistics": model_statistics,
        "overdispersion": overdispersion,
        "zero_warning": zero_warning,
        "train_metrics": count_regression_metrics(y_train, train_predictions),
        "test_metrics": count_regression_metrics(y_test, test_predictions),
        "train_actual": y_train,
        "test_actual": y_test,
        "train_predictions": pd.Series(train_predictions, index=y_train.index),
        "test_predictions": pd.Series(test_predictions, index=y_test.index),
        "formula_latex": _count_formula(y_column, x_columns, coefficient_table),
    }


def build_count_coefficient_table(fitted_model, model_type: str) -> pd.DataFrame:
    """Return count-model coefficient rows with incidence rate ratios."""
    params = fitted_model.params
    standard_errors = fitted_model.bse
    z_values = params / standard_errors
    p_values = fitted_model.pvalues
    conf_int = fitted_model.conf_int()
    coefficient_terms = [term for term in params.index if term != "alpha"]

    rows = []
    for term in coefficient_terms:
        estimate = params.loc[term]
        ci_lower = conf_int.loc[term, 0]
        ci_upper = conf_int.loc[term, 1]
        rows.append(
            {
                "term": term,
                "estimate": float(estimate),
                "std_error": float(standard_errors.loc[term]),
                "z_value": float(z_values.loc[term]),
                "p_value": float(p_values.loc[term]),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "incidence_rate_ratio": float(np.exp(estimate)),
                "irr_ci_lower": float(np.exp(ci_lower)),
                "irr_ci_upper": float(np.exp(ci_upper)),
                "model_type": model_type,
            }
        )
    return pd.DataFrame(rows)


def build_overdispersion_summary(
    y_values: pd.Series,
    fitted_model=None,
    train_y: pd.Series | None = None,
    train_x: pd.DataFrame | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Return simple overdispersion diagnostics and recommendation text."""
    numeric_y = pd.to_numeric(y_values, errors="coerce").dropna()
    mean_value = float(numeric_y.mean()) if len(numeric_y) else np.nan
    variance_value = float(numeric_y.var(ddof=1)) if len(numeric_y) > 1 else np.nan
    variance_mean_ratio = variance_value / mean_value if mean_value > 0 else np.nan

    pearson_chi_square_per_df = None
    if fitted_model is not None and train_y is not None and train_x is not None:
        predictions = _predict_expected_count(fitted_model, train_x)
        denominator = predictions if not alpha else predictions + alpha * predictions**2
        denominator = np.maximum(denominator, 1e-12)
        pearson_chi_square = float(np.sum((np.asarray(train_y, dtype=float) - predictions) ** 2 / denominator))
        df_resid = max(len(train_y) - len([term for term in fitted_model.params.index if term != "alpha"]), 1)
        pearson_chi_square_per_df = pearson_chi_square / df_resid

    overdispersed = bool(
        (pd.notna(variance_mean_ratio) and variance_mean_ratio > 1.5)
        or (pearson_chi_square_per_df is not None and pearson_chi_square_per_df > 1.5)
    )
    recommendation = (
        "Overdispersion is detected. Negative Binomial Regression may be more appropriate than Poisson Regression."
        if overdispersed
        else "No strong overdispersion signal was detected by these simple diagnostics."
    )
    return {
        "target_mean": mean_value,
        "target_variance": variance_value,
        "variance_mean_ratio": variance_mean_ratio,
        "pearson_chi_square_per_df": pearson_chi_square_per_df,
        "overdispersion_detected": overdispersed,
        "recommendation": recommendation,
    }


def zero_inflation_warning(y_values: pd.Series, threshold: float = 0.3) -> dict[str, Any]:
    """Return a warning when many observed counts are zero."""
    numeric_y = pd.to_numeric(y_values, errors="coerce").dropna()
    zero_proportion = float((numeric_y == 0).mean()) if len(numeric_y) else np.nan
    warning = (
        "Many observations are zero. Zero-inflated count models may be appropriate in a future version."
        if pd.notna(zero_proportion) and zero_proportion >= threshold
        else ""
    )
    return {
        "zero_proportion": zero_proportion,
        "warning": warning,
    }


def count_regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    """Return RMSE, MAE, and mean Poisson deviance for count predictions."""
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["mean_deviance"] = _mean_poisson_deviance(y_true, y_pred)
    return metrics


def predict_count_with_mean_ci(
    fitted_model,
    new_data: pd.DataFrame,
    design_columns: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Predict expected count and a confidence interval for the expected mean."""
    design = _align_design_matrix(new_data, pd.Index(design_columns))
    coefficient_terms = [term for term in fitted_model.params.index if term != "alpha"]
    beta = fitted_model.params.loc[coefficient_terms].to_numpy(dtype=float)
    design_beta = design.reindex(columns=coefficient_terms, fill_value=0).to_numpy(dtype=float)

    eta = design_beta @ beta
    predicted_mean = np.exp(eta)
    try:
        covariance = fitted_model.cov_params().loc[coefficient_terms, coefficient_terms].to_numpy(dtype=float)
        se_eta = np.sqrt(np.maximum(np.sum((design_beta @ covariance) * design_beta, axis=1), 0))
        z_value = norm.ppf(1 - alpha / 2)
        lower = np.exp(eta - z_value * se_eta)
        upper = np.exp(eta + z_value * se_eta)
        explanation = (
            "This is a confidence interval for the expected count mean. "
            "It is not an individual count prediction interval."
        )
    except Exception:
        lower = np.repeat(np.nan, len(predicted_mean))
        upper = np.repeat(np.nan, len(predicted_mean))
        explanation = (
            "Expected count was predicted, but a confidence interval is not available "
            "because parameter covariance could not be computed."
        )
    return pd.DataFrame(
        {
            "predicted_expected_count": predicted_mean,
            "mean_ci_lower": lower,
            "mean_ci_upper": upper,
            "alpha": alpha,
            "interval_explanation": explanation,
        }
    )


def plot_observed_vs_predicted_counts(y_true, y_pred):
    """Return an observed-vs-predicted count scatter plot."""
    plot_df = pd.DataFrame(
        {
            "observed_count": y_true,
            "predicted_expected_count": y_pred,
        }
    )
    return px.scatter(
        plot_df,
        x="observed_count",
        y="predicted_expected_count",
        title="Observed vs Predicted Expected Counts",
        template="plotly_white",
    )


def _validate_inputs(df: pd.DataFrame, y_column: str, x_columns: list[str], model_type: str) -> None:
    """Validate a count-model request."""
    if y_column not in df.columns:
        raise ValueError(f"Target column was not found in the dataset: {y_column}")
    if not x_columns:
        raise ValueError("Choose at least one predictor variable.")
    if y_column in x_columns:
        raise ValueError("The target column cannot also be used as a predictor.")
    missing_columns = [column for column in x_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Predictor columns were not found in the dataset: {missing_text}")
    if model_type not in COUNT_MODEL_OPTIONS:
        raise ValueError(f"Unsupported count model: {model_type}")


def _prepare_model_data(df: pd.DataFrame, y_column: str, x_columns: list[str]) -> pd.DataFrame:
    """Return complete-case data with a validated count target."""
    model_df = df[[y_column, *x_columns]].copy(deep=True).dropna()
    numeric_y = pd.to_numeric(model_df[y_column], errors="coerce")
    if numeric_y.isna().any():
        raise ValueError("Count regression target must be numeric.")
    if (numeric_y < 0).any():
        raise ValueError("Count regression target must be nonnegative.")
    if not np.allclose(numeric_y, np.round(numeric_y)):
        raise ValueError("Count regression target must contain integer-like counts.")
    if len(model_df) < 3:
        raise ValueError("At least 3 complete rows are needed for count regression.")

    model_df[y_column] = numeric_y.astype(int)
    return model_df


def _fit_count_model(y_train: pd.Series, x_train: pd.DataFrame, model_type: str):
    """Fit one statsmodels count model."""
    if model_type == POISSON_MODEL_NAME:
        return sm.GLM(y_train, x_train, family=sm.families.Poisson()).fit()
    return NegativeBinomial(y_train, x_train).fit(disp=False, maxiter=200)


def _design_matrix(x_values: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode predictors and add a constant column."""
    encoded = pd.get_dummies(x_values, drop_first=True, dtype=float)
    return sm.add_constant(encoded, has_constant="add")


def _align_design_matrix(x_values: pd.DataFrame, design_columns: pd.Index) -> pd.DataFrame:
    """Encode and align predictors to training design columns."""
    encoded = _design_matrix(x_values)
    return encoded.reindex(columns=design_columns, fill_value=0).astype(float)


def _predict_expected_count(fitted_model, x_values: pd.DataFrame) -> np.ndarray:
    """Return expected count predictions."""
    predictions = fitted_model.predict(x_values)
    return np.asarray(predictions, dtype=float)


def _alpha_parameter(fitted_model, model_type: str) -> float | None:
    """Return Negative Binomial alpha when available."""
    if model_type != NEGATIVE_BINOMIAL_MODEL_NAME:
        return None
    if "alpha" in fitted_model.params.index:
        return float(fitted_model.params.loc["alpha"])
    return None


def _model_statistics(
    fitted_model,
    model_type: str,
    overdispersion: dict[str, Any],
    zero_warning: dict[str, Any],
    alpha: float | None,
) -> pd.DataFrame:
    """Return model-level statistics as rows."""
    rows = [
        {"statistic": "AIC", "value": float(getattr(fitted_model, "aic", np.nan))},
        {"statistic": "BIC", "value": _bic_value(fitted_model)},
        {"statistic": "Model", "value": model_type},
        {"statistic": "variance / mean", "value": overdispersion["variance_mean_ratio"]},
        {"statistic": "Pearson chi-square / df", "value": overdispersion["pearson_chi_square_per_df"]},
        {"statistic": "zero proportion", "value": zero_warning["zero_proportion"]},
    ]
    if hasattr(fitted_model, "deviance"):
        rows.append({"statistic": "Deviance", "value": float(fitted_model.deviance)})
    if hasattr(fitted_model, "pearson_chi2"):
        rows.append({"statistic": "Pearson chi-square", "value": float(fitted_model.pearson_chi2)})
    if alpha is not None:
        rows.append({"statistic": "alpha", "value": alpha})
    return pd.DataFrame(rows)


def _bic_value(fitted_model) -> float:
    """Return BIC while avoiding known GLM deviance-BIC warnings when possible."""
    if hasattr(fitted_model, "bic_llf"):
        return float(fitted_model.bic_llf)
    return float(getattr(fitted_model, "bic", np.nan))


def _mean_poisson_deviance(y_true, y_pred) -> float | None:
    """Return mean Poisson deviance for nonnegative predictions."""
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.maximum(np.asarray(y_pred, dtype=float), 1e-12)
    if len(true_values) == 0:
        return np.nan
    term = np.where(true_values == 0, 0, true_values * np.log(np.maximum(true_values, 1e-12) / predicted_values))
    deviance = 2 * (term - (true_values - predicted_values))
    return float(np.mean(deviance))


def _count_formula(y_column: str, x_columns: list[str], coefficient_table: pd.DataFrame) -> dict[str, str]:
    """Return symbolic and estimated log-mean count formulas."""
    symbolic = build_symbolic_poisson_formula(y_column, x_columns)
    estimated = build_estimated_poisson_formula(y_column, coefficient_table)
    formula = {**symbolic, **estimated}
    formula["note"] = "Count regression with log link. The model estimates the expected count, not an exact individual count."
    return formula
