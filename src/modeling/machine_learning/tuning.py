"""Hyperparameter tuning helpers for machine learning baseline models."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.core.model_artifacts import save_model_artifact
from src.core.model_run import create_model_run
from src.modeling.interpretability.coefficients import (
    add_coefficient_interpretation_notes,
    extract_linear_model_coefficients,
    extract_logistic_model_coefficients,
)
from src.modeling.interpretability.tree_explainer import (
    TREE_EXPLANATION_WARNING,
    extract_tree_feature_importance,
    extract_tree_rules,
)
from src.modeling.machine_learning.classification import (
    TARGET_BINARY_COLUMN,
    _build_classification_pipeline,
    _prepare_model_data as _prepare_binary_model_data,
)
from src.modeling.machine_learning.feature_importance import (
    combine_feature_importance_tables,
    compute_tree_feature_importance,
    get_transformed_feature_names,
    importance_type_label,
)
from src.modeling.machine_learning.regression import (
    _build_regression_pipeline,
    _prepare_model_data as _prepare_regression_model_data,
)
from src.modeling.metrics import compute_binary_classification_metrics, compute_regression_metrics
from src.reporting.formula_builder import build_sklearn_linear_formula


SUPPORTED_TUNING_REGRESSION_MODELS = [
    "Ridge",
    "Lasso",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "KNN Regressor",
]

SUPPORTED_TUNING_CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "Gradient Boosting Classifier",
    "KNN Classifier",
]

REGRESSION_TUNING_SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R-squared": "r2",
}

BINARY_TUNING_SCORING = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score, zero_division=0),
    "Recall": make_scorer(recall_score, zero_division=0),
    "F1": "f1",
    "ROC AUC": "roc_auc",
}


def run_tuned_ml_regression_models(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    selected_models: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = False,
    search_type: str = "randomized",
    cv_folds: int = 5,
    scoring: str = "RMSE",
    n_iter: int = 10,
) -> list[dict[str, Any]]:
    """Tune selected regression models with GridSearchCV or RandomizedSearchCV.

    The estimator passed to the search object is the full sklearn Pipeline, so
    preprocessing is fit inside each cross-validation fold.
    """
    _validate_tuning_request(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        selected_models=selected_models,
        supported_models=SUPPORTED_TUNING_REGRESSION_MODELS,
        model_kind="regression",
    )
    cv_folds = _validate_cv_folds(cv_folds)
    model_df = _prepare_regression_model_data(df, target_column, feature_columns)

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
    )
    if len(train_df) < cv_folds:
        raise ValueError("The training set has fewer rows than the requested tuning CV folds.")

    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring_name, sklearn_scoring = _resolve_scoring(scoring, REGRESSION_TUNING_SCORING)

    results = []
    for model_name in selected_models:
        pipeline = _build_regression_pipeline(
            model_name=model_name,
            train_df=train_df,
            feature_columns=feature_columns,
            random_state=random_state,
            scale_numeric=scale_numeric,
        )
        search = _build_search(
            pipeline=pipeline,
            param_grid=_regression_param_grid(model_name, len(train_df), cv_folds),
            search_type=search_type,
            scoring=sklearn_scoring,
            cv=cv,
            random_state=random_state,
            n_iter=n_iter,
        )

        started_at = perf_counter()
        search.fit(x_train, y_train)
        runtime_seconds = perf_counter() - started_at

        best_pipeline = search.best_estimator_
        train_predictions = best_pipeline.predict(x_train)
        test_predictions = best_pipeline.predict(x_test)
        train_metrics = _prefixed_regression_metrics(compute_regression_metrics(y_train, train_predictions), "train")
        test_metrics = _prefixed_regression_metrics(compute_regression_metrics(y_test, test_predictions), "test")
        overfitting_warning = _regression_overfitting_warning(train_metrics, test_metrics)
        tuning_results_table = _tuning_results_table(search.cv_results_, scoring_name, sklearn_scoring)
        feature_importance_table = _feature_importance_table(best_pipeline)
        importance_type = importance_type_label(feature_importance_table)
        coefficient_table = extract_linear_model_coefficients(best_pipeline, target_column)
        if coefficient_table is not None:
            coefficient_table["model_name"] = f"{model_name} (Tuned)"
        formula_latex = (
            build_sklearn_linear_formula(target_column, coefficient_table, f"{model_name} (Tuned)")
            if coefficient_table is not None
            else None
        )
        tree_rules = _tree_rules_for_model(model_name, best_pipeline)
        tree_feature_importance_table = _tree_importance_for_model(model_name, best_pipeline)

        model_run = create_model_run(
            task_type="regression",
            model_family="machine_learning",
            model_name=f"{model_name} (Tuned)",
            target=target_column,
            features=feature_columns,
            split_config={
                "test_size": test_size,
                "random_state": random_state,
                "rows_used": len(model_df),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "cv_folds": cv_folds,
            },
            preprocessing=_preprocessing_metadata(
                scale_numeric=scale_numeric,
                search_type=search_type,
                cv_folds=cv_folds,
                scoring=scoring_name,
                best_score=_display_score(search.best_score_, sklearn_scoring),
                sklearn_best_score=search.best_score_,
                best_params=search.best_params_,
                runtime_seconds=runtime_seconds,
                overfitting_warning=overfitting_warning,
            ),
            statistical_summary={},
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            coefficient_table=coefficient_table,
            diagnostic_plot_keys=[],
            feature_importance_table=feature_importance_table,
            importance_type=importance_type,
            formula_latex=formula_latex,
            notes=_tuning_notes("regression", overfitting_warning, tree_rules),
        )
        _save_tuned_artifact(
            model_run=model_run,
            pipeline=best_pipeline,
            feature_importance_table=feature_importance_table,
            tuning_results_table=tuning_results_table,
            tree_rules=tree_rules,
            tree_feature_importance_table=tree_feature_importance_table,
        )
        results.append(
            {
                "model_run": model_run,
                "pipeline": best_pipeline,
                "search": search,
                "tuning_results_table": tuning_results_table,
                "train_predictions": train_predictions,
                "test_predictions": test_predictions,
                "feature_importance_table": feature_importance_table,
                "x_test": x_test,
                "tree_rules": tree_rules,
                "tree_feature_importance_table": tree_feature_importance_table,
            }
        )

    return results


def run_tuned_ml_binary_classification_models(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    selected_models: list[str],
    positive_class: Any,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = True,
    search_type: str = "randomized",
    cv_folds: int = 5,
    scoring: str = "F1",
    n_iter: int = 10,
) -> list[dict[str, Any]]:
    """Tune selected binary classifiers with the full Pipeline in CV."""
    _validate_tuning_request(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        selected_models=selected_models,
        supported_models=SUPPORTED_TUNING_CLASSIFICATION_MODELS,
        model_kind="binary classification",
    )
    cv_folds = _validate_cv_folds(cv_folds)
    model_df, target_classes = _prepare_binary_model_data(df, target_column, feature_columns, positive_class)

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
        stratify=model_df[TARGET_BINARY_COLUMN],
    )
    class_counts = train_df[TARGET_BINARY_COLUMN].value_counts()
    if class_counts.min() < cv_folds:
        raise ValueError("Each class in the training set needs at least as many rows as tuning CV folds.")

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_BINARY_COLUMN]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_BINARY_COLUMN]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring_name, sklearn_scoring = _resolve_scoring(scoring, BINARY_TUNING_SCORING)

    results = []
    for model_name in selected_models:
        pipeline = _build_classification_pipeline(
            model_name=model_name,
            train_df=train_df,
            feature_columns=feature_columns,
            random_state=random_state,
            scale_numeric=scale_numeric,
        )
        search = _build_search(
            pipeline=pipeline,
            param_grid=_classification_param_grid(model_name, len(train_df), cv_folds),
            search_type=search_type,
            scoring=sklearn_scoring,
            cv=cv,
            random_state=random_state,
            n_iter=n_iter,
        )

        started_at = perf_counter()
        search.fit(x_train, y_train)
        runtime_seconds = perf_counter() - started_at

        best_pipeline = search.best_estimator_
        train_predictions = best_pipeline.predict(x_train)
        test_predictions = best_pipeline.predict(x_test)
        train_probabilities = _positive_probabilities(best_pipeline, x_train)
        test_probabilities = _positive_probabilities(best_pipeline, x_test)
        train_metrics = _prefixed_classification_metrics(
            compute_binary_classification_metrics(
                y_train,
                train_predictions,
                y_proba=train_probabilities,
                positive_label=1,
            ),
            "train",
        )
        test_metrics = _prefixed_classification_metrics(
            compute_binary_classification_metrics(
                y_test,
                test_predictions,
                y_proba=test_probabilities,
                positive_label=1,
            ),
            "test",
        )
        overfitting_warning = _classification_overfitting_warning(train_metrics, test_metrics)
        tuning_results_table = _tuning_results_table(search.cv_results_, scoring_name, sklearn_scoring)
        feature_importance_table = _feature_importance_table(best_pipeline)
        importance_type = importance_type_label(feature_importance_table)
        coefficient_table = extract_logistic_model_coefficients(
            best_pipeline,
            target_name=target_column,
            positive_class=positive_class,
        )
        if coefficient_table is not None:
            coefficient_table["model_name"] = f"{model_name} (Tuned)"
        formula_latex = (
            build_sklearn_linear_formula(target_column, coefficient_table, f"{model_name} (Tuned)")
            if coefficient_table is not None
            else None
        )
        if formula_latex is not None:
            formula_latex["note"] = f"{formula_latex['note']} Positive class is `{positive_class}`."
        tree_rules = _tree_rules_for_model(model_name, best_pipeline)
        tree_feature_importance_table = _tree_importance_for_model(model_name, best_pipeline)

        model_run = create_model_run(
            task_type="binary_classification",
            model_family="machine_learning",
            model_name=f"{model_name} (Tuned)",
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
                **_preprocessing_metadata(
                    scale_numeric=scale_numeric,
                    search_type=search_type,
                    cv_folds=cv_folds,
                    scoring=scoring_name,
                    best_score=_display_score(search.best_score_, sklearn_scoring),
                    sklearn_best_score=search.best_score_,
                    best_params=search.best_params_,
                    runtime_seconds=runtime_seconds,
                    overfitting_warning=overfitting_warning,
                ),
                "positive_class": positive_class,
                "target_classes": target_classes,
                "target_encoding": f"{positive_class} encoded as 1; all other class values encoded as 0",
            },
            statistical_summary={},
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            coefficient_table=coefficient_table,
            diagnostic_plot_keys=[],
            feature_importance_table=feature_importance_table,
            importance_type=importance_type,
            formula_latex=formula_latex,
            notes=_tuning_notes("binary classification", overfitting_warning, tree_rules),
        )
        _save_tuned_artifact(
            model_run=model_run,
            pipeline=best_pipeline,
            feature_importance_table=feature_importance_table,
            tuning_results_table=tuning_results_table,
            positive_class=positive_class,
            target_classes=target_classes,
            tree_rules=tree_rules,
            tree_feature_importance_table=tree_feature_importance_table,
        )
        results.append(
            {
                "model_run": model_run,
                "pipeline": best_pipeline,
                "search": search,
                "positive_class": positive_class,
                "train_actual": y_train,
                "test_actual": y_test,
                "train_predictions": train_predictions,
                "test_predictions": test_predictions,
                "train_probabilities": train_probabilities,
                "test_probabilities": test_probabilities,
                "tuning_results_table": tuning_results_table,
                "feature_importance_table": feature_importance_table,
                "x_test": x_test,
                "tree_rules": tree_rules,
                "tree_feature_importance_table": tree_feature_importance_table,
            }
        )

    return results


def _build_search(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    search_type: str,
    scoring: str | Any,
    cv: KFold | StratifiedKFold,
    random_state: int,
    n_iter: int,
):
    """Build a sklearn search object around the full Pipeline."""
    search_type = search_type.lower()
    if search_type == "grid":
        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit=True,
            error_score=np.nan,
            return_train_score=True,
        )
    if search_type == "randomized":
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=min(max(1, int(n_iter)), _param_grid_size(param_grid)),
            scoring=scoring,
            cv=cv,
            refit=True,
            random_state=random_state,
            error_score=np.nan,
            return_train_score=True,
        )
    raise ValueError("search_type must be either 'grid' or 'randomized'.")


def _regression_param_grid(model_name: str, train_rows: int, cv_folds: int) -> dict[str, list[Any]]:
    """Return small beginner-friendly grids for supported regressors."""
    if model_name == "Ridge":
        return {"model__alpha": [0.1, 1.0, 10.0]}
    if model_name == "Lasso":
        return {"model__alpha": [0.001, 0.01, 0.1, 1.0]}
    if model_name == "Decision Tree Regressor":
        return {
            "model__max_depth": [None, 2, 4, 6],
            "model__min_samples_leaf": [1, 2, 5],
            "model__min_samples_split": [2, 5, 10],
        }
    if model_name == "Random Forest Regressor":
        return {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 2],
        }
    if model_name == "Gradient Boosting Regressor":
        return {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
    if model_name == "KNN Regressor":
        return {
            "model__n_neighbors": _knn_neighbor_options(train_rows, cv_folds),
            "model__weights": ["uniform", "distance"],
        }
    raise ValueError(f"Unsupported regression tuning model: {model_name}")


def _classification_param_grid(model_name: str, train_rows: int, cv_folds: int) -> dict[str, list[Any]]:
    """Return small beginner-friendly grids for supported classifiers."""
    if model_name == "Logistic Regression":
        return {"model__C": [0.1, 1.0, 10.0]}
    if model_name == "Decision Tree Classifier":
        return {
            "model__max_depth": [None, 2, 4, 6],
            "model__min_samples_leaf": [1, 2, 5],
            "model__min_samples_split": [2, 5, 10],
        }
    if model_name == "Random Forest Classifier":
        return {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 2],
        }
    if model_name == "Gradient Boosting Classifier":
        return {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
    if model_name == "KNN Classifier":
        return {
            "model__n_neighbors": _knn_neighbor_options(train_rows, cv_folds),
            "model__weights": ["uniform", "distance"],
        }
    raise ValueError(f"Unsupported binary classification tuning model: {model_name}")


def _validate_tuning_request(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    selected_models: list[str],
    supported_models: list[str],
    model_kind: str,
) -> None:
    """Validate common tuning inputs."""
    if target_column not in df.columns:
        raise ValueError(f"Target column was not found in the dataset: {target_column}")
    if not feature_columns:
        raise ValueError("Choose at least one feature column.")
    if target_column in feature_columns:
        raise ValueError("The target column cannot also be used as a feature.")
    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns were not found in the dataset: {', '.join(missing_features)}")
    if not selected_models:
        raise ValueError(f"Choose at least one {model_kind} model to tune.")
    unsupported = [model for model in selected_models if model not in supported_models]
    if unsupported:
        raise ValueError(f"Unsupported {model_kind} tuning model: {', '.join(unsupported)}")


def _validate_cv_folds(cv_folds: int) -> int:
    """Return a safe integer CV fold count for tuning."""
    folds = int(cv_folds)
    if folds < 2:
        raise ValueError("Tuning CV needs at least 2 folds.")
    return folds


def _resolve_scoring(scoring: str, scoring_options: dict[str, Any]) -> tuple[str, Any]:
    """Return display name and sklearn scoring object."""
    if scoring in scoring_options:
        return scoring, scoring_options[scoring]
    if scoring in scoring_options.values():
        for label, value in scoring_options.items():
            if value == scoring:
                return label, value
    raise ValueError(f"Unsupported tuning scoring metric: {scoring}")


def _param_grid_size(param_grid: dict[str, list[Any]]) -> int:
    """Return the number of possible parameter combinations."""
    sizes = [len(values) for values in param_grid.values()]
    total = 1
    for size in sizes:
        total *= size
    return total


def _knn_neighbor_options(train_rows: int, cv_folds: int) -> list[int]:
    """Choose KNN neighbor counts that remain feasible inside CV folds."""
    smallest_cv_train_size = max(1, int(np.floor(train_rows * (cv_folds - 1) / cv_folds)))
    return [value for value in [1, 3, 5, 7] if value <= smallest_cv_train_size] or [1]


def _tuning_results_table(cv_results: dict[str, Any], scoring_name: str, sklearn_scoring: Any) -> pd.DataFrame:
    """Return a compact table from sklearn search cv_results_."""
    table = pd.DataFrame(cv_results)
    keep_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "params",
    ]
    parameter_columns = [column for column in table.columns if column.startswith("param_")]
    keep_columns.extend(parameter_columns)
    keep_columns = [column for column in keep_columns if column in table.columns]
    table = table[keep_columns].copy()
    table["scoring"] = scoring_name
    table["display_test_score"] = table["mean_test_score"].apply(
        lambda value: _display_score(value, sklearn_scoring)
    )
    table = table.sort_values("rank_test_score").reset_index(drop=True)
    return table


def _display_score(value: Any, sklearn_scoring: Any) -> float | None:
    """Convert negative error scores into positive display values."""
    if pd.isna(value):
        return None
    if sklearn_scoring in {"neg_root_mean_squared_error", "neg_mean_absolute_error"}:
        return float(-value)
    return float(value)


def _preprocessing_metadata(
    scale_numeric: bool,
    search_type: str,
    cv_folds: int,
    scoring: str,
    best_score: float | None,
    sklearn_best_score: float,
    best_params: dict[str, Any],
    runtime_seconds: float,
    overfitting_warning: str | None,
) -> dict[str, Any]:
    """Return ModelRun preprocessing and tuning metadata."""
    return {
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "one-hot handle_unknown=ignore",
        "scale_numeric": scale_numeric,
        "pipeline": "sklearn Pipeline with ColumnTransformer preprocessing",
        "tuning": {
            "enabled": True,
            "search_type": search_type.lower(),
            "cv_folds": cv_folds,
            "scoring": scoring,
            "best_cv_score": float(best_score) if best_score is not None and not pd.isna(best_score) else None,
            "sklearn_best_score": float(sklearn_best_score) if not pd.isna(sklearn_best_score) else None,
            "best_params": dict(best_params),
            "runtime_seconds": float(runtime_seconds),
            "overfitting_warning": overfitting_warning,
        },
    }


def _feature_importance_table(pipeline: Pipeline) -> pd.DataFrame:
    """Return feature importance table for tuned tree-based models."""
    return combine_feature_importance_tables([compute_tree_feature_importance(pipeline)])


def _prefixed_regression_metrics(metrics: dict[str, float | None], prefix: str) -> dict[str, float | None]:
    """Return regression metrics with existing comparison-friendly aliases."""
    return {
        f"{prefix}_rmse": metrics["rmse"],
        f"{prefix}_mae": metrics["mae"],
        f"{prefix}_r2": metrics["r2"],
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "R-squared": metrics["r2"],
    }


def _prefixed_classification_metrics(metrics: dict[str, float | None], prefix: str) -> dict[str, float | None]:
    """Return binary classification metrics with existing aliases."""
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


def _positive_probabilities(pipeline: Pipeline, x_values: pd.DataFrame):
    """Return predicted probabilities for encoded positive class when available."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    probabilities = pipeline.predict_proba(x_values)
    model = pipeline.named_steps["model"]
    positive_index = list(model.classes_).index(1)
    return probabilities[:, positive_index]


def _regression_overfitting_warning(
    train_metrics: dict[str, float | None],
    test_metrics: dict[str, float | None],
) -> str | None:
    """Return a simple warning when train/test regression gap is large."""
    train_r2 = train_metrics.get("train_r2")
    test_r2 = test_metrics.get("test_r2")
    train_rmse = train_metrics.get("train_rmse")
    test_rmse = test_metrics.get("test_rmse")
    if train_r2 is not None and test_r2 is not None and train_r2 - test_r2 > 0.15:
        return "Train R-squared is much higher than test R-squared; possible overfitting."
    if train_rmse and test_rmse and test_rmse > 1.5 * train_rmse:
        return "Test RMSE is much larger than train RMSE; possible overfitting."
    return None


def _classification_overfitting_warning(
    train_metrics: dict[str, float | None],
    test_metrics: dict[str, float | None],
) -> str | None:
    """Return a simple warning when train/test classification gap is large."""
    train_f1 = train_metrics.get("train_f1")
    test_f1 = test_metrics.get("test_f1")
    train_accuracy = train_metrics.get("train_accuracy")
    test_accuracy = test_metrics.get("test_accuracy")
    if train_f1 is not None and test_f1 is not None and train_f1 - test_f1 > 0.15:
        return "Train F1 is much higher than test F1; possible overfitting."
    if train_accuracy is not None and test_accuracy is not None and train_accuracy - test_accuracy > 0.15:
        return "Train accuracy is much higher than test accuracy; possible overfitting."
    return None


def _tree_rules_for_model(model_name: str, pipeline: Pipeline) -> str | None:
    """Return text rules only for single decision tree tuned models."""
    if model_name not in {"Decision Tree Regressor", "Decision Tree Classifier"}:
        return None
    return extract_tree_rules(pipeline, max_depth=5)


def _tree_importance_for_model(model_name: str, pipeline: Pipeline) -> pd.DataFrame | None:
    """Return tree-specific feature importance for single decision tree tuned models."""
    if model_name not in {"Decision Tree Regressor", "Decision Tree Classifier"}:
        return None
    return extract_tree_feature_importance(pipeline)


def _tuning_notes(task_label: str, overfitting_warning: str | None, tree_rules: str | None) -> str:
    """Return concise notes for tuned ModelRuns."""
    notes = [f"Tuned machine learning {task_label} model fitted with an sklearn Pipeline."]
    if overfitting_warning:
        notes.append(overfitting_warning)
    if tree_rules:
        notes.append(TREE_EXPLANATION_WARNING)
    return " ".join(notes)


def _save_tuned_artifact(
    model_run: dict[str, Any],
    pipeline: Pipeline,
    feature_importance_table: pd.DataFrame,
    tuning_results_table: pd.DataFrame,
    positive_class: Any | None = None,
    target_classes: list[Any] | None = None,
    tree_rules: str | None = None,
    tree_feature_importance_table: pd.DataFrame | None = None,
) -> None:
    """Save a tuned fitted Pipeline for later prediction and interpretation."""
    save_model_artifact(
        model_run["run_id"],
        {
            "model_name": model_run["model_name"],
            "model_family": model_run["model_family"],
            "task_type": model_run["task_type"],
            "fitted_model": pipeline.named_steps.get("model"),
            "fitted_pipeline": pipeline,
            "best_estimator": pipeline,
            "target": model_run["target"],
            "features": model_run["features"],
            "transformed_feature_names": get_transformed_feature_names(pipeline),
            "preprocessing_summary": model_run.get("preprocessing", {}),
            "target_encoder": {
                "positive_class": positive_class,
                "target_classes": target_classes,
                "positive_class_encoded_as": 1,
            }
            if positive_class is not None
            else None,
            "positive_class": positive_class,
            "formula_latex": model_run.get("formula_latex"),
            "coefficient_table": model_run.get("coefficient_table"),
            "coefficient_interpretation_notes": add_coefficient_interpretation_notes(
                model_run.get("coefficient_table"),
                scale_numeric=bool(model_run.get("preprocessing", {}).get("scale_numeric")),
            ),
            "feature_importance_table": feature_importance_table,
            "tuning_results_table": tuning_results_table,
            "tree_rules": tree_rules,
            "tree_feature_importance_table": tree_feature_importance_table,
            "tree_explanation_warning": TREE_EXPLANATION_WARNING if tree_rules else None,
            "prediction_supported": True,
            "interval_supported": False,
            "interval_method": None,
        },
    )
