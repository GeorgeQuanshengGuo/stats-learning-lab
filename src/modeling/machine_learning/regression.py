"""Reusable machine learning regression baseline helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

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
    extract_linear_model_coefficients,
)
from src.modeling.interpretability.tree_explainer import (
    TREE_EXPLANATION_WARNING,
    extract_tree_feature_importance,
    extract_tree_rules,
)
from src.modeling.metrics import compute_regression_metrics
from src.modeling.preprocessing import build_preprocessing_pipeline
from src.reporting.formula_builder import build_sklearn_linear_formula


SUPPORTED_REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "KNN Regressor",
]


def run_ml_regression_models(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    selected_models: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = False,
    cv_folds: int | None = None,
    compute_permutation_importance: bool = False,
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int = 1,
    tree_min_samples_split: int = 2,
) -> list[dict[str, Any]]:
    """Fit selected sklearn regression baselines and return ModelRun results.

    The input DataFrame is copied before modeling. Preprocessing is built from
    the training data and fitted only inside each sklearn Pipeline.
    """
    _validate_inputs(df, target_column, feature_columns, selected_models)
    cv_folds = _normalize_cv_folds(cv_folds)
    model_df = _prepare_model_data(df, target_column, feature_columns)

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
    )
    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]

    results = []
    for model_name in selected_models:
        pipeline = _build_regression_pipeline(
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
        train_metrics = _prefixed_regression_metrics(
            compute_regression_metrics(y_train, train_predictions),
            prefix="train",
        )
        test_metrics = _prefixed_regression_metrics(
            compute_regression_metrics(y_test, test_predictions),
            prefix="test",
        )
        cv_metrics = _compute_regression_cv_metrics(
            model_name=model_name,
            model_df=model_df,
            target_column=target_column,
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
        coefficient_table = extract_linear_model_coefficients(pipeline, target_column)
        if coefficient_table is not None:
            coefficient_table["model_name"] = model_name
        formula_latex = (
            build_sklearn_linear_formula(target_column, coefficient_table, model_name)
            if coefficient_table is not None
            else None
        )
        tree_rules = _tree_rules_for_model(model_name, pipeline, tree_max_depth)
        tree_feature_importance_table = _tree_importance_for_model(model_name, pipeline)

        model_run = create_model_run(
            task_type="regression",
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
                "cv_folds": cv_folds,
            },
            preprocessing={
                "numeric_imputation": "median",
                "categorical_imputation": "most_frequent",
                "categorical_encoding": "one-hot handle_unknown=ignore",
                "scale_numeric": scale_numeric,
                "cross_validation": _cv_description(cv_folds, "KFold"),
                "pipeline": "sklearn Pipeline with ColumnTransformer preprocessing",
                "decision_tree_settings": {
                    "max_depth": tree_max_depth,
                    "min_samples_leaf": tree_min_samples_leaf,
                    "min_samples_split": tree_min_samples_split,
                }
                if model_name == "Decision Tree Regressor"
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
                base_note="Machine learning regression baseline fitted with an sklearn Pipeline.",
                tree_rules=tree_rules,
            ),
        )
        model_run["formula_latex"] = formula_latex
        _save_ml_regression_artifact(
            model_run=model_run,
            pipeline=pipeline,
            feature_importance_table=feature_importance_table,
            tree_rules=tree_rules,
            tree_feature_importance_table=tree_feature_importance_table,
        )

        results.append(
            {
                "model_run": model_run,
                "pipeline": pipeline,
                "train_predictions": train_predictions,
                "test_predictions": test_predictions,
                "feature_importance_table": feature_importance_table,
                "x_test": x_test,
                "tree_rules": tree_rules,
                "tree_feature_importance_table": tree_feature_importance_table,
            }
        )

    return results


def _save_ml_regression_artifact(
    model_run: dict[str, Any],
    pipeline: Pipeline,
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
            "target_encoder": None,
            "positive_class": None,
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
    """Validate model request before splitting data."""
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
        raise ValueError("Choose at least one regression model.")
    unknown_models = [model for model in selected_models if model not in SUPPORTED_REGRESSION_MODELS]
    if unknown_models:
        unknown_text = ", ".join(unknown_models)
        raise ValueError(f"Unsupported regression model: {unknown_text}")


def _prepare_model_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return a modeling copy with a numeric non-missing target."""
    model_df = df[[target_column, *feature_columns]].copy(deep=True)
    model_df[target_column] = pd.to_numeric(model_df[target_column], errors="coerce")
    model_df = model_df.dropna(subset=[target_column])

    if len(model_df) < 3:
        raise ValueError("At least 3 rows with a numeric target are needed for a train/test split.")

    return model_df


def _build_regression_pipeline(
    model_name: str,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
    scale_numeric: bool,
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int = 1,
    tree_min_samples_split: int = 2,
) -> Pipeline:
    """Build one sklearn Pipeline with preprocessing and estimator."""
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
    """Create a baseline sklearn regressor for a display model name."""
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Ridge":
        return Ridge()
    if model_name == "Lasso":
        return Lasso(max_iter=10000)
    if model_name == "Decision Tree Regressor":
        return DecisionTreeRegressor(
            random_state=random_state,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
            min_samples_split=tree_min_samples_split,
        )
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=100, random_state=random_state)
    if model_name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(random_state=random_state)
    if model_name == "KNN Regressor":
        return KNeighborsRegressor(n_neighbors=min(5, train_rows))

    raise ValueError(f"Unsupported regression model: {model_name}")


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


def _compute_regression_cv_metrics(
    model_name: str,
    model_df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    random_state: int,
    scale_numeric: bool,
    cv_folds: int | None,
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int = 1,
    tree_min_samples_split: int = 2,
) -> dict[str, float | None]:
    """Compute optional KFold CV metrics using the full sklearn Pipeline."""
    if cv_folds is None:
        return {}
    if len(model_df) < cv_folds:
        return _empty_regression_cv_metrics()

    pipeline = _build_regression_pipeline(
        model_name=model_name,
        train_df=model_df,
        feature_columns=feature_columns,
        random_state=random_state,
        scale_numeric=scale_numeric,
        tree_max_depth=tree_max_depth,
        tree_min_samples_leaf=tree_min_samples_leaf,
        tree_min_samples_split=tree_min_samples_split,
    )
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        pipeline,
        model_df[feature_columns],
        model_df[target_column],
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        error_score=np.nan,
    )

    rmse_mean, rmse_std = _mean_std(-cv_results["test_rmse"])
    mae_mean, mae_std = _mean_std(-cv_results["test_mae"])
    r2_mean, r2_std = _mean_std(cv_results["test_r2"])
    return {
        "cv_rmse_mean": rmse_mean,
        "cv_rmse_std": rmse_std,
        "cv_mae_mean": mae_mean,
        "cv_mae_std": mae_std,
        "cv_r2_mean": r2_mean,
        "cv_r2_std": r2_std,
    }


def _empty_regression_cv_metrics() -> dict[str, None]:
    """Return empty CV fields when the requested fold count is not feasible."""
    return {
        "cv_rmse_mean": None,
        "cv_rmse_std": None,
        "cv_mae_mean": None,
        "cv_mae_std": None,
        "cv_r2_mean": None,
        "cv_r2_std": None,
    }


def _mean_std(values) -> tuple[float | None, float | None]:
    """Return mean and standard deviation, ignoring failed folds."""
    values_array = np.asarray(values, dtype=float)
    valid_values = values_array[~np.isnan(values_array)]
    if len(valid_values) == 0:
        return None, None
    return float(np.mean(valid_values)), float(np.std(valid_values))


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
                scoring="neg_root_mean_squared_error",
                random_state=random_state,
            )
        )
    return combine_feature_importance_tables(importance_tables)


def _prefixed_regression_metrics(metrics: dict[str, float | None], prefix: str) -> dict[str, float | None]:
    """Return prefixed metrics plus comparison-friendly aliases."""
    return {
        f"{prefix}_rmse": metrics["rmse"],
        f"{prefix}_mae": metrics["mae"],
        f"{prefix}_r2": metrics["r2"],
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "R-squared": metrics["r2"],
    }


def _tree_rules_for_model(model_name: str, pipeline: Pipeline, tree_max_depth: int | None) -> str | None:
    """Return text rules only for the single decision tree baseline."""
    if model_name != "Decision Tree Regressor":
        return None
    return extract_tree_rules(pipeline, max_depth=tree_max_depth or 5)


def _tree_importance_for_model(model_name: str, pipeline: Pipeline) -> pd.DataFrame | None:
    """Return tree-specific feature importance only for the single decision tree baseline."""
    if model_name != "Decision Tree Regressor":
        return None
    return extract_tree_feature_importance(pipeline)


def _model_notes(base_note: str, tree_rules: str | None) -> str:
    """Add an explanation warning to decision tree model notes."""
    if tree_rules:
        return f"{base_note} {TREE_EXPLANATION_WARNING}"
    return base_note
