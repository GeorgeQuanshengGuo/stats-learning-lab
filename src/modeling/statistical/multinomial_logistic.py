"""Statsmodels multinomial logistic regression helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import MNLogit

from src.modeling.metrics import compute_multiclass_classification_metrics
from src.reporting.formula_builder import sanitize_latex_variable_name


def run_multinomial_logistic_regression(
    df: pd.DataFrame,
    y_column: str,
    reference_class: Any,
    x_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit a statsmodels multinomial logistic regression model."""
    _validate_inputs(df, y_column, reference_class, x_columns)
    model_df, target_classes = _prepare_model_data(df, y_column, reference_class, x_columns)

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
        stratify=model_df[y_column],
    )
    x_train = _design_matrix(train_df[x_columns])
    x_test = _align_design_matrix(test_df[x_columns], x_train.columns)
    y_train = train_df["__target_code"]
    y_test = test_df["__target_code"]

    model = MNLogit(y_train, x_train)
    fitted = model.fit(disp=False, maxiter=200)

    train_probabilities = fitted.predict(x_train)
    test_probabilities = fitted.predict(x_test)
    train_predictions = _predicted_labels(train_probabilities, target_classes)
    test_predictions = _predicted_labels(test_probabilities, target_classes)
    y_train_labels = train_df[y_column]
    y_test_labels = test_df[y_column]

    coefficient_table = build_coefficient_table(fitted, target_classes, reference_class)
    model_statistics = pd.DataFrame(
        [
            {"statistic": "AIC", "value": float(fitted.aic) if pd.notna(fitted.aic) else np.nan},
            {"statistic": "BIC", "value": float(fitted.bic) if pd.notna(fitted.bic) else np.nan},
            {"statistic": "Log-Likelihood", "value": float(fitted.llf) if pd.notna(fitted.llf) else np.nan},
            {"statistic": "Reference class", "value": reference_class},
        ]
    )
    train_metrics = compute_multiclass_classification_metrics(
        y_train_labels,
        train_predictions,
        y_proba=train_probabilities,
        labels=target_classes,
    )
    test_metrics = compute_multiclass_classification_metrics(
        y_test_labels,
        test_predictions,
        y_proba=test_probabilities,
        labels=target_classes,
    )

    return {
        "model": fitted,
        "y_column": y_column,
        "x_columns": x_columns,
        "reference_class": reference_class,
        "target_classes": target_classes,
        "test_size": test_size,
        "random_state": random_state,
        "rows_used": len(model_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "design_columns": list(x_train.columns),
        "coefficient_table": coefficient_table,
        "model_statistics": model_statistics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_actual": y_train_labels,
        "test_actual": y_test_labels,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "train_probabilities": train_probabilities,
        "test_probabilities": test_probabilities,
        "test_confusion_matrix": confusion_matrix_table(y_test_labels, test_predictions, target_classes),
        "test_class_metrics": class_wise_metrics_table(y_test_labels, test_predictions, target_classes),
        "formula_latex": build_multinomial_logit_formula(y_column, x_columns, target_classes, reference_class),
    }


def build_coefficient_table(fitted_model, target_classes: list[Any], reference_class: Any) -> pd.DataFrame:
    """Return class-specific coefficient rows from a fitted MNLogit model."""
    params = fitted_model.params
    standard_errors = fitted_model.bse
    z_values = params / standard_errors
    p_values = fitted_model.pvalues
    conf_int = fitted_model.conf_int()

    non_reference_classes = [class_value for class_value in target_classes if class_value != reference_class]
    rows = []
    for column_index, class_value in enumerate(non_reference_classes):
        parameter_column = params.columns[column_index]
        for term in params.index:
            estimate = params.loc[term, parameter_column]
            std_error = standard_errors.loc[term, parameter_column]
            ci_lower, ci_upper = _confidence_interval(conf_int, column_index + 1, term)
            rows.append(
                {
                    "class": class_value,
                    "reference_class": reference_class,
                    "term": term,
                    "estimate": float(estimate),
                    "std_error": float(std_error),
                    "z_value": float(z_values.loc[term, parameter_column]),
                    "p_value": float(p_values.loc[term, parameter_column]),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "odds_ratio": float(np.exp(estimate)),
                    "odds_ratio_ci_lower": float(np.exp(ci_lower)) if ci_lower is not None else None,
                    "odds_ratio_ci_upper": float(np.exp(ci_upper)) if ci_upper is not None else None,
                }
            )
    return pd.DataFrame(rows)


def confusion_matrix_table(y_true, y_pred, labels: list[Any]) -> pd.DataFrame:
    """Return a labeled multiclass confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )


def class_wise_metrics_table(y_true, y_pred, labels: list[Any]) -> pd.DataFrame:
    """Return class-wise precision, recall, F1, and support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def build_multinomial_logit_formula(
    target: str,
    terms: list[str],
    target_classes: list[Any],
    reference_class: Any,
) -> dict[str, str]:
    """Build a symbolic class-specific logit formula."""
    target_latex = sanitize_latex_variable_name(target)
    term_text = " + ".join(
        [r"\beta_{0,k}"]
        + [
            rf"\beta_{{{index},k}} {sanitize_latex_variable_name(term)}_i"
            for index, term in enumerate(terms, start=1)
        ]
    )
    logits = []
    for class_value in target_classes:
        if class_value == reference_class:
            continue
        class_latex = sanitize_latex_variable_name(class_value)
        reference_latex = sanitize_latex_variable_name(reference_class)
        logits.append(
            rf"\log\left(\frac{{P({target_latex}_i={class_latex})}}{{P({target_latex}_i={reference_latex})}}\right) = {term_text}"
        )
    return {
        "symbolic": r"\\ ".join(logits),
        "note": f"Class-specific logits are shown relative to reference class `{reference_class}`.",
    }


def _validate_inputs(
    df: pd.DataFrame,
    y_column: str,
    reference_class: Any,
    x_columns: list[str],
) -> None:
    """Validate a multinomial regression request."""
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

    classes = df[y_column].dropna().unique().tolist()
    if len(classes) <= 2:
        raise ValueError("Multinomial logistic regression requires more than two target classes.")
    if reference_class not in classes:
        raise ValueError("The reference class must be one of the target classes.")


def _prepare_model_data(
    df: pd.DataFrame,
    y_column: str,
    reference_class: Any,
    x_columns: list[str],
) -> tuple[pd.DataFrame, list[Any]]:
    """Return complete-case model data with reference class encoded as zero."""
    model_df = df[[y_column, *x_columns]].copy(deep=True).dropna()
    target_classes = [reference_class] + [
        class_value
        for class_value in sorted(model_df[y_column].dropna().unique().tolist(), key=lambda value: str(value))
        if class_value != reference_class
    ]
    if len(target_classes) <= 2:
        raise ValueError("Multinomial logistic regression requires more than two target classes.")
    class_counts = model_df[y_column].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each target class needs at least two rows for a stratified train/test split.")

    code_lookup = {class_value: index for index, class_value in enumerate(target_classes)}
    model_df["__target_code"] = model_df[y_column].map(code_lookup).astype(int)
    return model_df, target_classes


def _design_matrix(x_values: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode predictors and add a constant column."""
    encoded = pd.get_dummies(x_values, drop_first=True, dtype=float)
    return sm.add_constant(encoded, has_constant="add")


def _align_design_matrix(x_values: pd.DataFrame, design_columns: pd.Index) -> pd.DataFrame:
    """Encode and align test predictors to training design columns."""
    encoded = _design_matrix(x_values)
    return encoded.reindex(columns=design_columns, fill_value=0).astype(float)


def _predicted_labels(probabilities, target_classes: list[Any]) -> pd.Series:
    """Return class labels from a probability matrix."""
    probability_array = np.asarray(probabilities, dtype=float)
    predicted_indices = np.argmax(probability_array, axis=1)
    return pd.Series([target_classes[index] for index in predicted_indices])


def _confidence_interval(conf_int: pd.DataFrame, class_code: int, term: str) -> tuple[float | None, float | None]:
    """Return one confidence interval from statsmodels output."""
    for key in [(class_code, term), (str(class_code), term)]:
        try:
            row = conf_int.loc[key]
            return float(row["lower"]), float(row["upper"])
        except Exception:
            continue
    return None, None
