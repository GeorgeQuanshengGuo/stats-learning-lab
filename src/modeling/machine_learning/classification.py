"""Reusable machine learning binary classification baseline helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.core.model_artifacts import save_model_artifact
from src.core.model_run import create_model_run
from src.modeling.machine_learning.feature_importance import (
    combine_feature_importance_tables,
    compute_permutation_importance_table,
    compute_tree_feature_importance,
    get_transformed_feature_names,
    importance_type_label,
)
from src.modeling.interpretability.coefficients import (
    add_coefficient_interpretation_notes,
    extract_logistic_model_coefficients,
)
from src.modeling.interpretability.tree_explainer import (
    TREE_EXPLANATION_WARNING,
    extract_tree_feature_importance,
    extract_tree_rules,
)
from src.modeling.metrics import compute_binary_classification_metrics
from src.modeling.preprocessing import build_preprocessing_pipeline
from src.reporting.formula_builder import build_sklearn_linear_formula


SUPPORTED_CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "Gradient Boosting Classifier",
    "KNN Classifier",
]

TARGET_BINARY_COLUMN = "__target_binary"


def run_ml_binary_classification_models(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    selected_models: list[str],
    positive_class: Any,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = True,
    cv_folds: int | None = None,
    compute_permutation_importance: bool = False,
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int = 1,
    tree_min_samples_split: int = 2,
) -> list[dict[str, Any]]:
    """Fit selected sklearn binary classification baselines.

    The target is encoded as 0/1 based on the user-selected positive class.
    Preprocessing is fit only inside each sklearn Pipeline after a stratified
    train/test split.
    """
    _validate_inputs(df, target_column, feature_columns, selected_models)
    cv_folds = _normalize_cv_folds(cv_folds)
    model_df, target_classes = _prepare_model_data(df, target_column, feature_columns, positive_class)

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
        stratify=model_df[TARGET_BINARY_COLUMN],
    )
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_BINARY_COLUMN]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_BINARY_COLUMN]
    split_warnings = _class_balance_warnings(y_train, y_test)

    results = []
    for model_name in selected_models:
        pipeline = _build_classification_pipeline(
            model_name=model_name,
            train_df=train_df,
            feature_columns=feature_columns,
            random_state=random_state,
            scale_numeric=scale_numeric,
            tree_max_depth=tree_max_depth,
            tree_min_samples_leaf=tree_min_samples_leaf,
            tree_min_samples_split=tree_min_samples_split,
        )
        pipeline.fit(x_train, y_train)

        train_predictions = pipeline.predict(x_train)
        test_predictions = pipeline.predict(x_test)
        train_probabilities = _positive_probabilities(pipeline, x_train)
        test_probabilities = _positive_probabilities(pipeline, x_test)
        train_metrics = _prefixed_classification_metrics(
            compute_binary_classification_metrics(
                y_train,
                train_predictions,
                y_proba=train_probabilities,
                positive_label=1,
            ),
            prefix="train",
        )
        test_metrics = _prefixed_classification_metrics(
            compute_binary_classification_metrics(
                y_test,
                test_predictions,
                y_proba=test_probabilities,
                positive_label=1,
            ),
            prefix="test",
        )
        cv_metrics = _compute_classification_cv_metrics(
            model_name=model_name,
            model_df=model_df,
            feature_columns=feature_columns,
            random_state=random_state,
            scale_numeric=scale_numeric,
            cv_folds=cv_folds,
            tree_max_depth=tree_max_depth,
            tree_min_samples_leaf=tree_min_samples_leaf,
            tree_min_samples_split=tree_min_samples_split,
        )
        test_metrics.update(cv_metrics)
        feature_importance_table = _compute_feature_importance(
            pipeline=pipeline,
            x_test=x_test,
            y_test=y_test,
            random_state=random_state,
            compute_permutation=compute_permutation_importance,
        )
        importance_type = importance_type_label(feature_importance_table)
        coefficient_table = extract_logistic_model_coefficients(
            pipeline,
            target_name=target_column,
            positive_class=positive_class,
        )
        if coefficient_table is not None:
            coefficient_table["model_name"] = model_name
        formula_latex = (
            build_sklearn_linear_formula(target_column, coefficient_table, model_name)
            if coefficient_table is not None
            else None
        )
        if formula_latex is not None:
            formula_latex["note"] = (
                f"{formula_latex['note']} Positive class is `{positive_class}`."
            )
        tree_rules = _tree_rules_for_model(model_name, pipeline, tree_max_depth)
        tree_feature_importance_table = _tree_importance_for_model(model_name, pipeline)

        model_run = create_model_run(
            task_type="binary_classification",
            model_family="machine_learning",
            model_name=model_name,
            target=target_column,
            features=feature_columns,
            split_config={
                "test_size": test_size,
                "random_state": random_state,
                "rows_used": len(model_df),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "stratified": True,
                "cv_folds": cv_folds,
            },
            preprocessing={
                "positive_class": positive_class,
                "target_classes": target_classes,
                "target_encoding": f"{positive_class} encoded as 1; all other class values encoded as 0",
                "numeric_imputation": "median",
                "categorical_imputation": "most_frequent",
                "categorical_encoding": "one-hot handle_unknown=ignore",
                "scale_numeric": scale_numeric,
                "cross_validation": _cv_description(cv_folds, "StratifiedKFold"),
                "pipeline": "sklearn Pipeline with ColumnTransformer preprocessing",
                "decision_tree_settings": {
                    "max_depth": tree_max_depth,
                    "min_samples_leaf": tree_min_samples_leaf,
                    "min_samples_split": tree_min_samples_split,
                }
                if model_name == "Decision Tree Classifier"
                else None,
            },
            statistical_summary={},
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            coefficient_table=coefficient_table,
            diagnostic_plot_keys=[],
            feature_importance_table=feature_importance_table,
            importance_type=importance_type,
            notes=_model_notes(
                base_note="Machine learning binary classification baseline fitted with an sklearn Pipeline.",
                tree_rules=tree_rules,
                warnings=split_warnings,
            ),
        )
        model_run["formula_latex"] = formula_latex
        _save_ml_classification_artifact(
            model_run=model_run,
            pipeline=pipeline,
            positive_class=positive_class,
            target_classes=target_classes,
            feature_importance_table=feature_importance_table,
            tree_rules=tree_rules,
            tree_feature_importance_table=tree_feature_importance_table,
        )

        results.append(
            {
                "model_run": model_run,
                "pipeline": pipeline,
                "positive_class": positive_class,
                "train_actual": y_train,
                "test_actual": y_test,
                "train_predictions": train_predictions,
                "test_predictions": test_predictions,
                "train_probabilities": train_probabilities,
                "test_probabilities": test_probabilities,
                "feature_importance_table": feature_importance_table,
                "x_test": x_test,
                "tree_rules": tree_rules,
                "tree_feature_importance_table": tree_feature_importance_table,
            }
        )

    return results


def _save_ml_classification_artifact(
    model_run: dict[str, Any],
    pipeline: Pipeline,
    positive_class: Any,
    target_classes: list[Any],
    feature_importance_table: pd.DataFrame,
    tree_rules: str | None = None,
    tree_feature_importance_table: pd.DataFrame | None = None,
) -> None:
    """Save the fitted sklearn Pipeline for later prediction-related features."""
    save_model_artifact(
        model_run["run_id"],
        {
            "model_name": model_run["model_name"],
            "model_family": model_run["model_family"],
            "task_type": model_run["task_type"],
            "fitted_model": pipeline.named_steps.get("model"),
            "fitted_pipeline": pipeline,
            "target": model_run["target"],
            "features": model_run["features"],
            "transformed_feature_names": get_transformed_feature_names(pipeline),
            "preprocessing_summary": model_run.get("preprocessing", {}),
            "target_encoder": {
                "positive_class": positive_class,
                "target_classes": target_classes,
                "positive_class_encoded_as": 1,
            },
            "positive_class": positive_class,
            "formula_latex": model_run.get("formula_latex"),
            "coefficient_table": model_run.get("coefficient_table"),
            "coefficient_interpretation_notes": add_coefficient_interpretation_notes(
                model_run.get("coefficient_table"),
                scale_numeric=bool(model_run.get("preprocessing", {}).get("scale_numeric")),
            ),
            "feature_importance_table": feature_importance_table,
            "tree_rules": tree_rules,
            "tree_feature_importance_table": tree_feature_importance_table,
            "tree_explanation_warning": TREE_EXPLANATION_WARNING if tree_rules else None,
            "prediction_supported": True,
            "interval_supported": False,
            "interval_method": None,
        },
    )


def _validate_inputs(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    selected_models: list[str],
) -> None:
    """Validate model request before target encoding."""
    if target_column not in df.columns:
        raise ValueError(f"Target column was not found in the dataset: {target_column}")
    if not feature_columns:
        raise ValueError("Choose at least one feature column.")
    if target_column in feature_columns:
        raise ValueError("The target column cannot also be used as a feature.")
    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        missing_text = ", ".join(missing_features)
        raise ValueError(f"Feature columns were not found in the dataset: {missing_text}")
    if not selected_models:
        raise ValueError("Choose at least one classification model.")
    unknown_models = [model for model in selected_models if model not in SUPPORTED_CLASSIFICATION_MODELS]
    if unknown_models:
        unknown_text = ", ".join(unknown_models)
        raise ValueError(f"Unsupported classification model: {unknown_text}")


def _prepare_model_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    positive_class: Any,
) -> tuple[pd.DataFrame, list[Any]]:
    """Return a modeling copy with a validated 0/1 encoded target."""
    model_df = df[[target_column, *feature_columns]].copy(deep=True)
    model_df = model_df.dropna(subset=[target_column])
    target_classes = model_df[target_column].dropna().unique().tolist()

    if len(target_classes) != 2:
        raise ValueError("Binary classification requires exactly two non-missing target classes.")
    if positive_class not in target_classes:
        raise ValueError("The selected positive_class must be one of the two target classes.")

    class_counts = model_df[target_column].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each target class needs at least two rows for a stratified train/test split.")

    model_df[TARGET_BINARY_COLUMN] = (model_df[target_column] == positive_class).astype(int)
    return model_df, target_classes


def _build_classification_pipeline(
    model_name: str,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
    scale_numeric: bool,
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int = 1,
    tree_min_samples_split: int = 2,
) -> Pipeline:
    """Build one sklearn Pipeline with preprocessing and classifier."""
    preprocessing = build_preprocessing_pipeline(
        train_df,
        feature_columns,
        scale_numeric=scale_numeric,
    )
    model = _model_for_name(
        model_name,
        random_state=random_state,
        train_rows=len(train_df),
        tree_max_depth=tree_max_depth,
        tree_min_samples_leaf=tree_min_samples_leaf,
        tree_min_samples_split=tree_min_samples_split,
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def _model_for_name(
    model_name: str,
    random_state: int,
    train_rows: int,
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int = 1,
    tree_min_samples_split: int = 2,
):
    """Create a baseline sklearn classifier for a display model name."""
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_name == "Decision Tree Classifier":
        return DecisionTreeClassifier(
            random_state=random_state,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
            min_samples_split=tree_min_samples_split,
        )
    if model_name == "Random Forest Classifier":
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    if model_name == "Gradient Boosting Classifier":
        return GradientBoostingClassifier(random_state=random_state)
    if model_name == "KNN Classifier":
        return KNeighborsClassifier(n_neighbors=min(5, train_rows))

    raise ValueError(f"Unsupported classification model: {model_name}")


def _normalize_cv_folds(cv_folds: int | None) -> int | None:
    """Validate and normalize the optional cross-validation fold count."""
    if cv_folds in (None, 0):
        return None
    if cv_folds < 2:
        raise ValueError("Cross-validation needs at least 2 folds.")
    return int(cv_folds)


def _cv_description(cv_folds: int | None, splitter_name: str) -> str:
    """Return a short human-readable description of the CV setup."""
    if cv_folds is None:
        return "not used"
    return f"{cv_folds}-fold {splitter_name}; estimator is the full sklearn Pipeline"


def _compute_classification_cv_metrics(
    model_name: str,
    model_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
    scale_numeric: bool,
    cv_folds: int | None,
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int = 1,
    tree_min_samples_split: int = 2,
) -> dict[str, float | None]:
    """Compute optional StratifiedKFold CV metrics using the full Pipeline."""
    if cv_folds is None:
        return {}

    class_counts = model_df[TARGET_BINARY_COLUMN].value_counts()
    if len(model_df) < cv_folds or class_counts.min() < cv_folds:
        return _empty_classification_cv_metrics()

    pipeline = _build_classification_pipeline(
        model_name=model_name,
        train_df=model_df,
        feature_columns=feature_columns,
        random_state=random_state,
        scale_numeric=scale_numeric,
        tree_max_depth=tree_max_depth,
        tree_min_samples_leaf=tree_min_samples_leaf,
        tree_min_samples_split=tree_min_samples_split,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        pipeline,
        model_df[feature_columns],
        model_df[TARGET_BINARY_COLUMN],
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "roc_auc": "roc_auc",
        },
        error_score=np.nan,
    )

    accuracy_mean, accuracy_std = _mean_std(cv_results["test_accuracy"])
    precision_mean, precision_std = _mean_std(cv_results["test_precision"])
    recall_mean, recall_std = _mean_std(cv_results["test_recall"])
    f1_mean, f1_std = _mean_std(cv_results["test_f1"])
    roc_auc_mean, roc_auc_std = _mean_std(cv_results["test_roc_auc"])
    return {
        "cv_accuracy_mean": accuracy_mean,
        "cv_accuracy_std": accuracy_std,
        "cv_precision_mean": precision_mean,
        "cv_precision_std": precision_std,
        "cv_recall_mean": recall_mean,
        "cv_recall_std": recall_std,
        "cv_f1_mean": f1_mean,
        "cv_f1_std": f1_std,
        "cv_roc_auc_mean": roc_auc_mean,
        "cv_roc_auc_std": roc_auc_std,
    }


def _empty_classification_cv_metrics() -> dict[str, None]:
    """Return empty CV fields when the requested fold count is not feasible."""
    return {
        "cv_accuracy_mean": None,
        "cv_accuracy_std": None,
        "cv_precision_mean": None,
        "cv_precision_std": None,
        "cv_recall_mean": None,
        "cv_recall_std": None,
        "cv_f1_mean": None,
        "cv_f1_std": None,
        "cv_roc_auc_mean": None,
        "cv_roc_auc_std": None,
    }


def _mean_std(values) -> tuple[float | None, float | None]:
    """Return mean and standard deviation, ignoring failed folds."""
    values_array = np.asarray(values, dtype=float)
    valid_values = values_array[~np.isnan(values_array)]
    if len(valid_values) == 0:
        return None, None
    return float(np.mean(valid_values)), float(np.std(valid_values))


def _positive_probabilities(pipeline: Pipeline, x_values: pd.DataFrame):
    """Return predicted probabilities for the encoded positive class when available."""
    if not hasattr(pipeline, "predict_proba"):
        return None

    probabilities = pipeline.predict_proba(x_values)
    model = pipeline.named_steps["model"]
    positive_index = list(model.classes_).index(1)
    return probabilities[:, positive_index]


def _compute_feature_importance(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    compute_permutation: bool,
) -> pd.DataFrame:
    """Compute tree and optional permutation importance for a fitted model."""
    importance_tables = [compute_tree_feature_importance(pipeline)]
    if compute_permutation:
        importance_tables.append(
            compute_permutation_importance_table(
                pipeline,
                x_test,
                y_test,
                scoring="accuracy",
                random_state=random_state,
            )
        )
    return combine_feature_importance_tables(importance_tables)


def _prefixed_classification_metrics(metrics: dict[str, float | None], prefix: str) -> dict[str, float | None]:
    """Return prefixed metrics plus comparison-friendly aliases."""
    output = {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "F1": metrics["f1"],
    }

    if "roc_auc" in metrics:
        output[f"{prefix}_roc_auc"] = metrics["roc_auc"]
        output["ROC AUC"] = metrics["roc_auc"]
    if "pr_auc" in metrics:
        output[f"{prefix}_pr_auc"] = metrics["pr_auc"]
        output["PR AUC"] = metrics["pr_auc"]

    return output


def _tree_rules_for_model(model_name: str, pipeline: Pipeline, tree_max_depth: int | None) -> str | None:
    """Return text rules only for the single decision tree baseline."""
    if model_name != "Decision Tree Classifier":
        return None
    return extract_tree_rules(pipeline, max_depth=tree_max_depth or 5)


def _tree_importance_for_model(model_name: str, pipeline: Pipeline) -> pd.DataFrame | None:
    """Return tree-specific feature importance only for the single decision tree baseline."""
    if model_name != "Decision Tree Classifier":
        return None
    return extract_tree_feature_importance(pipeline)


def _class_balance_warnings(y_train: pd.Series, y_test: pd.Series) -> list[str]:
    """Return warnings for imbalanced splits that can make metrics unstable."""
    warnings = []
    for split_name, values in [("training", y_train), ("test", y_test)]:
        counts = values.value_counts()
        if len(counts) < 2:
            warnings.append(
                f"The {split_name} split contains only one target class; some classification metrics may be undefined."
            )
        elif counts.min() / counts.sum() < 0.1:
            warnings.append(
                f"The {split_name} split is severely imbalanced; review precision, recall, PR AUC, and class counts."
            )
    return warnings


def _model_notes(base_note: str, tree_rules: str | None, warnings: list[str] | None = None) -> str:
    """Add an explanation warning to decision tree model notes."""
    notes = base_note
    if tree_rules:
        notes = f"{notes} {TREE_EXPLANATION_WARNING}"
    if warnings:
        notes = f"{notes} {' '.join(warnings)}"
    return notes
