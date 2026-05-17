"""Statsmodels-based binary logistic regression helpers."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

from src.modeling.statistical.linear_regression import encode_predictors, train_test_split


def run_logistic_regression(
    df: pd.DataFrame,
    y_column: str,
    positive_class: Any,
    x_columns: list[str],
    test_size: float,
    random_state: int,
    threshold: float = 0.5,
    optimize_threshold: bool = False,
) -> dict[str, Any]:
    """Prepare data, fit binary Logit, and return inference and prediction results."""
    model_df = df[[y_column, *x_columns]].dropna().copy()
    classes = list(model_df[y_column].dropna().unique())
    if len(classes) != 2:
        raise ValueError("Binary logistic regression requires exactly two target classes.")

    model_df["_target_binary"] = (model_df[y_column] == positive_class).astype(int)
    if model_df["_target_binary"].nunique() != 2:
        raise ValueError("The selected positive class must be one of the two target classes.")

    train_df, test_df = train_test_split(model_df, test_size=test_size, random_state=random_state)
    y_train, x_train, y_test, x_test = build_logistic_design_matrices(
        train_df,
        test_df,
        "_target_binary",
        x_columns,
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        model = sm.Logit(y_train, x_train).fit(disp=False)
    warning_messages = _warning_messages(captured_warnings)
    train_probabilities = model.predict(x_train)
    test_probabilities = model.predict(x_test)

    chosen_threshold = (
        find_best_f1_threshold(y_train, train_probabilities)
        if optimize_threshold
        else threshold
    )

    return {
        "model": model,
        "rows_used": len(model_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "y_column": y_column,
        "positive_class": positive_class,
        "x_columns": x_columns,
        "test_size": test_size,
        "random_state": random_state,
        "threshold": float(chosen_threshold),
        "threshold_strategy": "optimized_for_f1" if optimize_threshold else "custom",
        "warnings": warning_messages,
        "coefficient_table": build_logistic_coefficient_table(model),
        "model_statistics": build_logistic_model_statistics(model),
        "train_metrics": classification_metrics(y_train, train_probabilities, chosen_threshold),
        "test_metrics": classification_metrics(y_test, test_probabilities, chosen_threshold),
        "train_confusion_matrix": confusion_matrix_table(y_train, train_probabilities, chosen_threshold),
        "test_confusion_matrix": confusion_matrix_table(y_test, test_probabilities, chosen_threshold),
        "test_roc_curve": roc_curve_table(y_test, test_probabilities),
        "test_pr_curve": precision_recall_curve_table(y_test, test_probabilities),
        "train_probabilities": pd.Series(train_probabilities, name="predicted_probability"),
        "train_actual": pd.Series(y_train, name="actual"),
        "test_probabilities": pd.Series(test_probabilities, name="predicted_probability"),
        "test_actual": pd.Series(y_test, name="actual"),
    }


def _warning_messages(captured_warnings: list[warnings.WarningMessage]) -> list[str]:
    """Return unique model-fitting warning messages for UI display."""
    messages: list[str] = []
    for warning in captured_warnings:
        message = str(warning.message)
        if message and message not in messages:
            messages.append(message)
    return messages


def build_logistic_design_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_column: str,
    x_columns: list[str],
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create binary y vectors and one-hot encoded X matrices."""
    y_train = train_df[y_column].astype(int)
    y_test = test_df[y_column].astype(int)

    x_train = encode_predictors(train_df[x_columns])
    x_test = encode_predictors(test_df[x_columns])
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    x_train = sm.add_constant(x_train, has_constant="add")
    x_test = sm.add_constant(x_test, has_constant="add")
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    return y_train, x_train, y_test, x_test


def build_logistic_coefficient_table(model: Any) -> pd.DataFrame:
    """Return a logistic regression coefficient table with odds ratios."""
    confidence_intervals = model.conf_int()
    with np.errstate(over="ignore", invalid="ignore"):
        odds_ratio = np.exp(model.params)
        odds_ci_lower = np.exp(confidence_intervals[0])
        odds_ci_upper = np.exp(confidence_intervals[1])

    return pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.values,
            "std_error": model.bse.values,
            "z_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "odds_ratio": odds_ratio.values,
            "odds_ratio_ci_lower": odds_ci_lower.values,
            "odds_ratio_ci_upper": odds_ci_upper.values,
        }
    )


def build_logistic_model_statistics(model: Any) -> pd.DataFrame:
    """Return model-level logistic regression statistics."""
    return pd.DataFrame(
        [
            {"statistic": "AIC", "value": float(model.aic)},
            {"statistic": "BIC", "value": float(model.bic)},
            {"statistic": "Pseudo R-squared", "value": float(model.prsquared)},
        ]
    )


def classification_metrics(y_true: Any, probabilities: Any, threshold: float) -> dict[str, float]:
    """Calculate binary classification metrics from predicted probabilities."""
    true_values = np.asarray(y_true, dtype=int)
    predicted_labels = probability_to_label(probabilities, threshold)

    tp, tn, fp, fn = confusion_counts(true_values, predicted_labels)
    accuracy = (tp + tn) / len(true_values) if len(true_values) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "ROC AUC": roc_auc_score(true_values, probabilities),
        "PR AUC": pr_auc_score(true_values, probabilities),
    }


def probability_to_label(probabilities: Any, threshold: float) -> np.ndarray:
    """Convert probabilities into binary class predictions."""
    return (np.asarray(probabilities, dtype=float) >= threshold).astype(int)


def confusion_counts(y_true: Any, y_pred: Any) -> tuple[int, int, int, int]:
    """Return true positives, true negatives, false positives, and false negatives."""
    true_values = np.asarray(y_true, dtype=int)
    predicted_values = np.asarray(y_pred, dtype=int)

    tp = int(((true_values == 1) & (predicted_values == 1)).sum())
    tn = int(((true_values == 0) & (predicted_values == 0)).sum())
    fp = int(((true_values == 0) & (predicted_values == 1)).sum())
    fn = int(((true_values == 1) & (predicted_values == 0)).sum())
    return tp, tn, fp, fn


def confusion_matrix_table(y_true: Any, probabilities: Any, threshold: float) -> pd.DataFrame:
    """Return a 2x2 confusion matrix table."""
    predicted_labels = probability_to_label(probabilities, threshold)
    tp, tn, fp, fn = confusion_counts(y_true, predicted_labels)

    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )


def find_best_f1_threshold(y_true: Any, probabilities: Any) -> float:
    """Find the threshold that maximizes F1 on the provided data."""
    probability_values = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.r_[0.0, probability_values, 1.0])
    best_threshold = 0.5
    best_f1 = -1.0

    for candidate in candidates:
        f1 = classification_metrics(y_true, probabilities, float(candidate))["F1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(candidate)

    return best_threshold


def roc_curve_table(y_true: Any, probabilities: Any) -> pd.DataFrame:
    """Build a ROC curve table across probability thresholds."""
    true_values = np.asarray(y_true, dtype=int)
    probability_values = np.asarray(probabilities, dtype=float)
    thresholds = np.unique(np.r_[1.0, probability_values, 0.0])[::-1]
    rows = []

    for threshold in thresholds:
        predicted = probability_to_label(probability_values, float(threshold))
        tp, tn, fp, fn = confusion_counts(true_values, predicted)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({"threshold": float(threshold), "fpr": float(fpr), "tpr": float(tpr)})

    return pd.DataFrame(rows).sort_values("fpr")


def precision_recall_curve_table(y_true: Any, probabilities: Any) -> pd.DataFrame:
    """Build a precision-recall curve table across probability thresholds."""
    true_values = np.asarray(y_true, dtype=int)
    probability_values = np.asarray(probabilities, dtype=float)
    thresholds = np.unique(np.r_[1.0, probability_values, 0.0])[::-1]
    rows = []

    for threshold in thresholds:
        predicted = probability_to_label(probability_values, float(threshold))
        tp, _, fp, fn = confusion_counts(true_values, predicted)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

    return pd.DataFrame(rows).sort_values("recall")


def roc_auc_score(y_true: Any, probabilities: Any) -> float:
    """Calculate ROC AUC from the ROC curve table."""
    true_values = np.asarray(y_true, dtype=int)
    if len(np.unique(true_values)) < 2:
        return np.nan

    curve = roc_curve_table(true_values, probabilities)
    return float(np.trapezoid(curve["tpr"], curve["fpr"]))


def pr_auc_score(y_true: Any, probabilities: Any) -> float:
    """Calculate area under the precision-recall curve when feasible."""
    true_values = np.asarray(y_true, dtype=int)
    if true_values.sum() == 0:
        return np.nan

    curve = precision_recall_curve_table(true_values, probabilities)
    return float(np.trapezoid(curve["precision"], curve["recall"]))


def plot_roc_curve(roc_curve: pd.DataFrame) -> go.Figure:
    """Plot a ROC curve."""
    figure = px.line(
        roc_curve,
        x="fpr",
        y="tpr",
        title="ROC Curve",
        template="plotly_white",
    )
    figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash"})
    )
    figure.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return figure


def plot_precision_recall_curve(pr_curve: pd.DataFrame) -> go.Figure:
    """Plot a precision-recall curve."""
    figure = px.line(
        pr_curve,
        x="recall",
        y="precision",
        title="Precision-Recall Curve",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return figure


def plot_confusion_matrix(confusion_matrix: pd.DataFrame) -> go.Figure:
    """Plot a confusion matrix heatmap."""
    figure = px.imshow(
        confusion_matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        template="plotly_white",
    )
    figure.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return figure


def plot_probability_distribution(probabilities: pd.Series, actual: pd.Series) -> go.Figure:
    """Plot predicted probability distributions by actual class."""
    plot_df = pd.DataFrame(
        {
            "predicted_probability": probabilities,
            "actual": actual.astype(str),
        }
    )
    return px.histogram(
        plot_df,
        x="predicted_probability",
        color="actual",
        barmode="overlay",
        opacity=0.7,
        nbins=20,
        title="Predicted Probability Distribution",
        template="plotly_white",
    )
