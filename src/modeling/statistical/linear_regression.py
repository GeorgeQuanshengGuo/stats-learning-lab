"""Statsmodels-based linear regression helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats

from src.modeling.diagnostics.influence import compute_ols_influence_table, top_influential_rows


def run_linear_regression(
    df: pd.DataFrame,
    y_column: str,
    x_columns: list[str],
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """Prepare data, fit OLS, and return tables, metrics, and diagnostics data."""
    model_df = df[[y_column, *x_columns]].dropna().copy()
    train_df, test_df = train_test_split(model_df, test_size=test_size, random_state=random_state)
    y_train, x_train, y_test, x_test = build_design_matrices(train_df, test_df, y_column, x_columns)

    model = sm.OLS(y_train, x_train).fit()
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    influence_table = compute_ols_influence_table(model, original_index=y_train.index)

    return {
        "model": model,
        "rows_used": len(model_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "y_column": y_column,
        "x_columns": x_columns,
        "test_size": test_size,
        "random_state": random_state,
        "coefficient_table": build_coefficient_table(model),
        "model_statistics": build_model_statistics(model),
        "train_metrics": regression_metrics(y_train, train_predictions),
        "test_metrics": regression_metrics(y_test, test_predictions),
        "y_train": y_train,
        "y_test": y_test,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "fitted_values": model.fittedvalues,
        "residuals": model.resid,
        "influence": model.get_influence(),
        "influence_table": influence_table,
        "top_influential_rows": top_influential_rows(influence_table),
    }


def train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train and test sets without pulling in extra libraries."""
    if len(df) < 3:
        raise ValueError("At least 3 complete rows are needed for a train/test split.")

    test_size = min(max(test_size, 0.1), 0.9)
    shuffled = df.sample(frac=1, random_state=random_state)
    test_count = int(round(len(shuffled) * test_size))
    test_count = min(max(test_count, 1), len(shuffled) - 1)

    test_df = shuffled.iloc[:test_count].copy()
    train_df = shuffled.iloc[test_count:].copy()
    return train_df, test_df


def build_design_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_column: str,
    x_columns: list[str],
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create numeric y vectors and one-hot encoded X matrices."""
    y_train = pd.to_numeric(train_df[y_column], errors="coerce")
    y_test = pd.to_numeric(test_df[y_column], errors="coerce")

    x_train = encode_predictors(train_df[x_columns])
    x_test = encode_predictors(test_df[x_columns])
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    x_train = sm.add_constant(x_train, has_constant="add")
    x_test = sm.add_constant(x_test, has_constant="add")
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    return y_train, x_train, y_test, x_test


def encode_predictors(x: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical predictors and keep numeric predictors numeric."""
    encoded = pd.get_dummies(x, drop_first=True, dtype=float)
    return encoded.apply(pd.to_numeric, errors="coerce").astype(float)


def build_coefficient_table(model: Any) -> pd.DataFrame:
    """Return an R-style OLS coefficient table."""
    confidence_intervals = model.conf_int()

    return pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_lower": confidence_intervals[0].values,
            "ci_upper": confidence_intervals[1].values,
        }
    )


def build_model_statistics(model: Any) -> pd.DataFrame:
    """Return model-level OLS statistics."""
    return pd.DataFrame(
        [
            {"statistic": "R-squared", "value": float(model.rsquared)},
            {"statistic": "Adjusted R-squared", "value": float(model.rsquared_adj)},
            {"statistic": "F-statistic", "value": float(model.fvalue) if model.fvalue is not None else np.nan},
            {"statistic": "AIC", "value": float(model.aic)},
            {"statistic": "BIC", "value": float(model.bic)},
        ]
    )


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calculate RMSE, MAE, and R-squared for regression predictions."""
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    errors = true_values - predicted_values

    return {
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAE": float(np.mean(np.abs(errors))),
        "R-squared": _r_squared(true_values, predicted_values),
    }


def plot_residuals_vs_fitted(fitted_values: pd.Series, residuals: pd.Series) -> go.Figure:
    """Plot residuals against fitted values."""
    figure = px.scatter(
        x=fitted_values,
        y=residuals,
        labels={"x": "Fitted values", "y": "Residuals"},
        title="Residuals vs Fitted",
        template="plotly_white",
    )
    figure.add_hline(y=0, line_dash="dash", line_color="gray")
    return figure


def plot_normal_qq(residuals: pd.Series) -> go.Figure:
    """Create a normal Q-Q plot for residuals."""
    theoretical, ordered = stats.probplot(residuals, dist="norm", fit=False)
    figure = px.scatter(
        x=theoretical,
        y=ordered,
        labels={"x": "Theoretical quantiles", "y": "Sample quantiles"},
        title="Normal Q-Q",
        template="plotly_white",
    )
    if len(theoretical) > 1:
        line = stats.linregress(theoretical, ordered)
        x_line = np.asarray([min(theoretical), max(theoretical)])
        figure.add_trace(
            go.Scatter(
                x=x_line,
                y=line.intercept + line.slope * x_line,
                mode="lines",
                name="Reference line",
            )
        )
    return figure


def plot_scale_location(fitted_values: pd.Series, residuals: pd.Series) -> go.Figure:
    """Plot square-root standardized residuals against fitted values."""
    residual_std = residuals.std()
    standardized = residuals / residual_std if residual_std else residuals * np.nan
    sqrt_abs_standardized = np.sqrt(np.abs(standardized))

    return px.scatter(
        x=fitted_values,
        y=sqrt_abs_standardized,
        labels={"x": "Fitted values", "y": "sqrt(|standardized residuals|)"},
        title="Scale-Location",
        template="plotly_white",
    )


def plot_residuals_vs_leverage(influence: Any, residuals: pd.Series) -> go.Figure:
    """Plot residuals against leverage values when influence data is available."""
    leverage = influence.hat_matrix_diag
    standardized_residuals = influence.resid_studentized_internal

    figure = px.scatter(
        x=leverage,
        y=standardized_residuals,
        labels={"x": "Leverage", "y": "Standardized residuals"},
        title="Residuals vs Leverage",
        template="plotly_white",
    )
    figure.add_hline(y=0, line_dash="dash", line_color="gray")
    return figure


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate prediction R-squared, returning NaN when it is undefined."""
    total_sum_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total_sum_squares == 0:
        return np.nan

    residual_sum_squares = float(np.sum((y_true - y_pred) ** 2))
    return float(1 - residual_sum_squares / total_sum_squares)
