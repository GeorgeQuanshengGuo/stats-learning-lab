"""Statsmodels ordinal logistic regression helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.reporting.formula_builder import format_latex_number, sanitize_latex_variable_name


PROPORTIONAL_ODDS_WARNING = (
    "Ordinal logistic regression uses a proportional odds assumption. This app "
    "does not yet run a formal assumption test, so review results cautiously."
)


def run_ordinal_logistic_regression(
    df: pd.DataFrame,
    y_column: str,
    category_order: list[Any],
    x_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit a proportional-odds ordinal logistic regression model."""
    _validate_inputs(df, y_column, category_order, x_columns)
    model_df = _prepare_model_data(df, y_column, category_order, x_columns)

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
        stratify=model_df["__target_code"],
    )
    x_train = _design_matrix(train_df[x_columns])
    x_test = _align_design_matrix(test_df[x_columns], x_train.columns)
    y_train = train_df["__target_code"]
    y_test = test_df["__target_code"]

    model = OrderedModel(y_train, x_train, distr="logit")
    fitted = model.fit(method="bfgs", disp=False, maxiter=200)

    train_probabilities = _probability_frame(fitted.predict(x_train), category_order)
    test_probabilities = _probability_frame(fitted.predict(x_test), category_order)
    train_predictions = _predicted_categories(train_probabilities)
    test_predictions = _predicted_categories(test_probabilities)

    coefficient_table, threshold_table = build_parameter_tables(fitted)
    train_metrics = ordinal_classification_metrics(
        train_df[y_column],
        train_predictions,
        category_order,
    )
    test_metrics = ordinal_classification_metrics(
        test_df[y_column],
        test_predictions,
        category_order,
    )
    model_statistics = pd.DataFrame(
        [
            {"statistic": "AIC", "value": float(fitted.aic)},
            {"statistic": "BIC", "value": float(fitted.bic)},
            {"statistic": "Log-Likelihood", "value": float(fitted.llf)},
            {"statistic": "Link", "value": "logit"},
        ]
    )

    return {
        "model": fitted,
        "y_column": y_column,
        "x_columns": x_columns,
        "category_order": list(category_order),
        "test_size": test_size,
        "random_state": random_state,
        "rows_used": len(model_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "design_columns": list(x_train.columns),
        "coefficient_table": coefficient_table,
        "threshold_table": threshold_table,
        "model_statistics": model_statistics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_actual": train_df[y_column],
        "test_actual": test_df[y_column],
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "train_probabilities": train_probabilities,
        "test_probabilities": test_probabilities,
        "test_confusion_matrix": confusion_matrix_table(test_df[y_column], test_predictions, category_order),
        "test_probability_table": class_probability_table(test_probabilities, test_predictions),
        "formula_latex": build_ordinal_formula(y_column, x_columns, coefficient_table, threshold_table),
        "diagnostic_warning": PROPORTIONAL_ODDS_WARNING,
    }


def build_parameter_tables(fitted_model) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split OrderedModel parameters into predictor coefficients and thresholds."""
    params = fitted_model.params
    standard_errors = fitted_model.bse
    z_values = params / standard_errors
    p_values = fitted_model.pvalues
    conf_int = fitted_model.conf_int()
    predictor_count = int(getattr(fitted_model.model, "k_vars", 0))
    predictor_terms = list(params.index[:predictor_count])
    threshold_terms = list(params.index[predictor_count:])

    coefficient_rows = []
    for term in predictor_terms:
        estimate = params.loc[term]
        ci_lower = conf_int.loc[term, 0]
        ci_upper = conf_int.loc[term, 1]
        coefficient_rows.append(
            {
                "term": term,
                "estimate": float(estimate),
                "std_error": float(standard_errors.loc[term]),
                "z_value": float(z_values.loc[term]),
                "p_value": float(p_values.loc[term]),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "odds_ratio": float(np.exp(estimate)),
                "odds_ratio_ci_lower": float(np.exp(ci_lower)),
                "odds_ratio_ci_upper": float(np.exp(ci_upper)),
            }
        )

    threshold_rows = []
    for term in threshold_terms:
        threshold_rows.append(
            {
                "cutpoint": term,
                "estimate": float(params.loc[term]),
                "std_error": float(standard_errors.loc[term]),
                "z_value": float(z_values.loc[term]),
                "p_value": float(p_values.loc[term]),
                "ci_lower": float(conf_int.loc[term, 0]),
                "ci_upper": float(conf_int.loc[term, 1]),
            }
        )

    return pd.DataFrame(coefficient_rows), pd.DataFrame(threshold_rows)


def ordinal_classification_metrics(y_true, y_pred, category_order: list[Any]) -> dict[str, float | None]:
    """Return classification metrics plus mean absolute error on ordered codes."""
    true_series = pd.Series(y_true)
    predicted_series = pd.Series(y_pred)
    if len(true_series) == 0:
        return {"accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "ordered_mae": np.nan}

    true_codes = _category_codes(true_series, category_order)
    predicted_codes = _category_codes(predicted_series, category_order)
    return {
        "accuracy": float(np.mean(true_series.to_numpy() == predicted_series.to_numpy())),
        "macro_f1": float(f1_score(true_series, predicted_series, labels=category_order, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_series, predicted_series, labels=category_order, average="weighted", zero_division=0)),
        "ordered_mae": float(np.mean(np.abs(true_codes - predicted_codes))),
    }


def confusion_matrix_table(y_true, y_pred, category_order: list[Any]) -> pd.DataFrame:
    """Return a labeled confusion matrix in ordered-category order."""
    matrix = confusion_matrix(y_true, y_pred, labels=category_order)
    return pd.DataFrame(
        matrix,
        index=[f"actual_{value}" for value in category_order],
        columns=[f"predicted_{value}" for value in category_order],
    )


def class_probability_table(probabilities: pd.DataFrame, predicted_categories: pd.Series) -> pd.DataFrame:
    """Return predicted category probabilities and selected category."""
    table = probabilities.copy(deep=True).reset_index(drop=True)
    table.insert(0, "predicted_category", list(predicted_categories))
    table.insert(0, "row_id", list(range(len(table))))
    return table


def build_ordinal_formula(
    target: str,
    terms: list[str],
    coefficient_table: pd.DataFrame,
    threshold_table: pd.DataFrame,
) -> dict[str, str]:
    """Build symbolic and estimated cumulative logit formulas."""
    rhs_symbolic = " + ".join(
        [
            rf"\beta_{{{index}}} {sanitize_latex_variable_name(term)}_i"
            for index, term in enumerate(terms, start=1)
        ]
    )
    rhs_estimated = _estimated_linear_predictor(coefficient_table)
    thresholds = ", ".join(
        f"{row['cutpoint']}={format_latex_number(row['estimate'])}"
        for _, row in threshold_table.iterrows()
    )
    return {
        "symbolic": rf"\log\left(\frac{{P({sanitize_latex_variable_name(target)}_i \le k)}}{{P({sanitize_latex_variable_name(target)}_i > k)}}\right) = \theta_k - \eta_i",
        "linear_predictor": rf"\eta_i = {rhs_symbolic}",
        "estimated": rf"\hat{{\eta}}_i = {rhs_estimated}",
        "thresholds": rf"\hat{{\theta}}: {thresholds}",
        "note": "Cumulative logit / proportional odds formula. Thresholds separate adjacent ordered categories.",
    }


def predict_ordered_model(
    fitted_model,
    x_values: pd.DataFrame,
    design_columns: list[str],
    category_order: list[Any],
) -> dict[str, Any]:
    """Predict ordered-category probabilities for raw feature rows."""
    design = _align_design_matrix(x_values, pd.Index(design_columns))
    probabilities = _probability_frame(fitted_model.predict(design), category_order)
    predicted_categories = _predicted_categories(probabilities)
    return {
        "probabilities": probabilities,
        "predicted_categories": predicted_categories,
        "probability_table": class_probability_table(probabilities, predicted_categories),
    }


def _validate_inputs(
    df: pd.DataFrame,
    y_column: str,
    category_order: list[Any],
    x_columns: list[str],
) -> None:
    """Validate an ordinal regression request."""
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

    unique_values = df[y_column].dropna().unique().tolist()
    if len(unique_values) < 3:
        raise ValueError("Ordinal logistic regression requires at least three ordered categories.")
    if len(category_order) != len(unique_values) or set(category_order) != set(unique_values):
        raise ValueError("category_order must include every non-missing target category exactly once.")


def _prepare_model_data(
    df: pd.DataFrame,
    y_column: str,
    category_order: list[Any],
    x_columns: list[str],
) -> pd.DataFrame:
    """Return complete-case data with ordered target codes."""
    model_df = df[[y_column, *x_columns]].copy(deep=True).dropna()
    class_counts = model_df[y_column].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each ordered category needs at least two rows for a stratified train/test split.")

    model_df["__target_code"] = _category_codes(model_df[y_column], category_order)
    return model_df


def _design_matrix(x_values: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode predictors without adding an intercept."""
    return pd.get_dummies(x_values, drop_first=True, dtype=float)


def _align_design_matrix(x_values: pd.DataFrame, design_columns: pd.Index) -> pd.DataFrame:
    """Encode and align new predictors to training design columns."""
    encoded = _design_matrix(x_values)
    return encoded.reindex(columns=design_columns, fill_value=0).astype(float)


def _probability_frame(probabilities, category_order: list[Any]) -> pd.DataFrame:
    """Normalize OrderedModel probabilities into a labeled DataFrame."""
    table = pd.DataFrame(probabilities).copy(deep=True)
    table.columns = list(category_order)
    return table.reset_index(drop=True)


def _predicted_categories(probabilities: pd.DataFrame) -> pd.Series:
    """Return the highest-probability ordered category for each row."""
    probability_array = probabilities.to_numpy(dtype=float)
    predicted_indices = np.argmax(probability_array, axis=1)
    return pd.Series([probabilities.columns[index] for index in predicted_indices])


def _category_codes(values, category_order: list[Any]) -> np.ndarray:
    """Encode ordered categories as integer ranks."""
    lookup = {category: index for index, category in enumerate(category_order)}
    return pd.Series(values).map(lookup).astype(int).to_numpy()


def _estimated_linear_predictor(coefficient_table: pd.DataFrame) -> str:
    """Build an estimated eta expression from predictor coefficients."""
    if coefficient_table.empty:
        return "0"
    pieces = []
    for _, row in coefficient_table.iterrows():
        estimate = float(row["estimate"])
        sign = "+" if estimate >= 0 and pieces else "-" if estimate < 0 else ""
        coefficient = format_latex_number(abs(estimate) if estimate < 0 else estimate)
        term = rf"{sanitize_latex_variable_name(row['term'])}_i"
        pieces.append(f"{sign} {coefficient} {term}".strip())
    return " ".join(pieces)
