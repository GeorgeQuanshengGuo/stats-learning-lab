"""Reusable machine learning multiclass classification baseline helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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
from src.modeling.metrics import compute_multiclass_classification_metrics
from src.modeling.preprocessing import build_preprocessing_pipeline


SUPPORTED_MULTICLASS_MODELS = [
    "Multinomial Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "Gradient Boosting Classifier",
    "KNN Classifier",
]


def run_ml_multiclass_classification_models(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    selected_models: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = True,
    compute_permutation_importance: bool = False,
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int = 1,
    tree_min_samples_split: int = 2,
) -> list[dict[str, Any]]:
    """Fit selected sklearn multiclass classifiers and return ModelRun results."""
    _validate_inputs(df, target_column, feature_columns, selected_models)
    model_df, target_classes = _prepare_model_data(df, target_column, feature_columns)

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
        stratify=model_df[target_column],
    )
    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]

    results = []
    for model_name in selected_models:
        pipeline = _build_multiclass_pipeline(
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
        train_probabilities = _predict_probabilities(pipeline, x_train)
        test_probabilities = _predict_probabilities(pipeline, x_test)

        train_metrics = _prefixed_multiclass_metrics(
            compute_multiclass_classification_metrics(
                y_train,
                train_predictions,
                y_proba=train_probabilities,
                labels=target_classes,
            ),
            prefix="train",
        )
        test_metrics = _prefixed_multiclass_metrics(
            compute_multiclass_classification_metrics(
                y_test,
                test_predictions,
                y_proba=test_probabilities,
                labels=target_classes,
            ),
            prefix="test",
        )
        confusion_table = confusion_matrix_table(y_test, test_predictions, target_classes)
        class_metrics = class_wise_metrics_table(y_test, test_predictions, target_classes)
        probability_table = predicted_probability_table(
            x_test,
            test_predictions,
            test_probabilities,
            target_classes,
        )
        feature_importance_table = _compute_feature_importance(
            pipeline=pipeline,
            x_test=x_test,
            y_test=y_test,
            random_state=random_state,
            compute_permutation=compute_permutation_importance,
        )
        importance_type = importance_type_label(feature_importance_table)

        model_run = create_model_run(
            task_type="multiclass_classification",
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
            },
            preprocessing={
                "target_classes": target_classes,
                "numeric_imputation": "median",
                "categorical_imputation": "most_frequent",
                "categorical_encoding": "one-hot handle_unknown=ignore",
                "scale_numeric": scale_numeric,
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
            coefficient_table=None,
            diagnostic_plot_keys=["confusion_matrix", "class_wise_metrics", "predicted_probabilities"],
            feature_importance_table=feature_importance_table,
            importance_type=importance_type,
            notes="Machine learning multiclass classification baseline fitted with an sklearn Pipeline.",
        )
        _save_ml_multiclass_artifact(
            model_run=model_run,
            pipeline=pipeline,
            target_classes=target_classes,
            feature_importance_table=feature_importance_table,
        )

        results.append(
            {
                "model_run": model_run,
                "pipeline": pipeline,
                "target_classes": target_classes,
                "train_actual": y_train,
                "test_actual": y_test,
                "train_predictions": train_predictions,
                "test_predictions": test_predictions,
                "train_probabilities": train_probabilities,
                "test_probabilities": test_probabilities,
                "confusion_matrix": confusion_table,
                "class_wise_metrics": class_metrics,
                "probability_table": probability_table,
                "feature_importance_table": feature_importance_table,
                "x_test": x_test,
            }
        )

    return results


def confusion_matrix_table(y_true, y_pred, labels: list[Any]) -> pd.DataFrame:
    """Return a labeled confusion matrix table."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )


def class_wise_metrics_table(y_true, y_pred, labels: list[Any]) -> pd.DataFrame:
    """Return precision, recall, F1, and support for each class."""
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


def predicted_probability_table(
    x_values: pd.DataFrame,
    predicted_classes,
    probabilities,
    class_labels: list[Any],
) -> pd.DataFrame:
    """Return one probability row per observation when probabilities are available."""
    if probabilities is None:
        return pd.DataFrame()
    table = pd.DataFrame(probabilities, columns=[f"probability_{label}" for label in class_labels])
    table.insert(0, "predicted_class", list(predicted_classes))
    table.insert(0, "row_id", list(x_values.index))
    return table


def _save_ml_multiclass_artifact(
    model_run: dict[str, Any],
    pipeline: Pipeline,
    target_classes: list[Any],
    feature_importance_table: pd.DataFrame,
) -> None:
    """Save the fitted sklearn Pipeline for later prediction and interpretation."""
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
                "target_classes": target_classes,
            },
            "positive_class": None,
            "formula_latex": None,
            "coefficient_table": None,
            "feature_importance_table": feature_importance_table,
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
        raise ValueError("Choose at least one multiclass classification model.")
    unknown_models = [model for model in selected_models if model not in SUPPORTED_MULTICLASS_MODELS]
    if unknown_models:
        unknown_text = ", ".join(unknown_models)
        raise ValueError(f"Unsupported multiclass classification model: {unknown_text}")


def _prepare_model_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[Any]]:
    """Return a modeling copy with a validated multiclass target."""
    model_df = df[[target_column, *feature_columns]].copy(deep=True)
    model_df = model_df.dropna(subset=[target_column])
    target_classes = sorted(model_df[target_column].dropna().unique().tolist(), key=lambda value: str(value))

    if len(target_classes) <= 2:
        raise ValueError("Multiclass classification requires more than two target classes.")
    class_counts = model_df[target_column].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each target class needs at least two rows for a stratified train/test split.")

    return model_df, target_classes


def _build_multiclass_pipeline(
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
    """Create a baseline sklearn multiclass classifier."""
    if model_name == "Multinomial Logistic Regression":
        try:
            return LogisticRegression(max_iter=1000, random_state=random_state, multi_class="multinomial")
        except TypeError:
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

    raise ValueError(f"Unsupported multiclass classification model: {model_name}")


def _predict_probabilities(pipeline: Pipeline, x_values: pd.DataFrame):
    """Return predicted class probabilities when supported."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    return pipeline.predict_proba(x_values)


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


def _prefixed_multiclass_metrics(metrics: dict[str, float | None], prefix: str) -> dict[str, float | None]:
    """Return prefixed metrics plus comparison-friendly aliases."""
    output = {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_macro_precision": metrics["macro_precision"],
        f"{prefix}_macro_recall": metrics["macro_recall"],
        f"{prefix}_macro_f1": metrics["macro_f1"],
        f"{prefix}_weighted_precision": metrics["weighted_precision"],
        f"{prefix}_weighted_recall": metrics["weighted_recall"],
        f"{prefix}_weighted_f1": metrics["weighted_f1"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
    }
    if "log_loss" in metrics:
        output[f"{prefix}_log_loss"] = metrics["log_loss"]
        output["log_loss"] = metrics["log_loss"]
    return output
