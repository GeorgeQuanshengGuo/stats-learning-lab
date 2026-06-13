"""Learning-focused rigor checks and readiness summaries.

These helpers produce advisory warnings only. They do not block the user from
continuing and they never modify the uploaded or working dataset.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

import pandas as pd

from src.core.analysis_plan import analysis_plan_is_recorded, normalize_analysis_plan
from src.data.type_detector import detect_variable_types
from src.eda.missing import build_missing_table


HIGH_MISSING_THRESHOLD = 40.0
HIGH_CARDINALITY_UNIQUE_COUNT = 20
HIGH_CARDINALITY_RATIO = 0.5
RARE_CLASS_RATIO = 0.25
EVENTS_PER_VARIABLE_WARNING = 10


def build_data_readiness(
    df: pd.DataFrame | None,
    target_column: str | None = None,
    schema: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Summarize dataset issues that matter before modeling."""
    if df is None:
        return _empty_data_readiness()

    schema_table = schema if schema is not None and not schema.empty else detect_variable_types(df)
    missing_table = build_missing_table(df)
    high_missing = missing_table.loc[
        missing_table["missing_pct"] >= HIGH_MISSING_THRESHOLD,
        "variable",
    ].tolist()

    constant_columns = schema_table.loc[
        schema_table["detected_type"] == "constant",
        "variable",
    ].tolist()
    id_like_columns = schema_table.loc[
        schema_table["detected_type"] == "id_like",
        "variable",
    ].tolist()
    datetime_columns = schema_table.loc[
        schema_table["detected_type"] == "datetime",
        "variable",
    ].tolist()
    high_cardinality_columns = _high_cardinality_columns(df, schema_table)
    possible_grouping_columns = _possible_grouping_columns(df, schema_table)

    target_missing_count = None
    target_unique_count = None
    if target_column and target_column in df.columns:
        target_missing_count = int(df[target_column].isna().sum())
        target_unique_count = int(df[target_column].dropna().nunique())

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "duplicate_row_count": int(df.duplicated().sum()),
        "missing_summary": missing_table.where(pd.notna(missing_table), None).to_dict(orient="records"),
        "high_missing_columns": high_missing,
        "constant_columns": constant_columns,
        "id_like_columns": id_like_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "datetime_columns": datetime_columns,
        "target_missing_count": target_missing_count,
        "target_unique_count": target_unique_count,
        "possible_grouping_columns": possible_grouping_columns,
    }


def build_model_readiness(
    df: pd.DataFrame | None,
    target_column: str | None,
    feature_columns: list[str] | None,
    task_type: str,
    split_strategy: str = "random",
    schema: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return advisory model-readiness facts and warnings."""
    feature_columns = list(feature_columns or [])
    if df is None or not target_column or target_column not in df.columns:
        return {
            "selected_target": target_column,
            "selected_features": feature_columns,
            "target_detected_type": None,
            "task_type": task_type,
            "sample_size_after_drop_missing": 0,
            "feature_count": len(feature_columns),
            "events_per_variable": None,
            "warnings": ["Choose a valid target variable before interpreting model readiness."],
        }

    available_features = [column for column in feature_columns if column in df.columns and column != target_column]
    schema_table = schema if schema is not None and not schema.empty else detect_variable_types(df)
    target_detected_type = _detected_type(schema_table, target_column)
    model_df = df[[target_column, *available_features]].dropna()
    warnings = []

    if len(available_features) == 0:
        warnings.append("Choose at least one predictor before fitting a model.")
    if len(available_features) >= len(model_df) and len(model_df) > 0:
        warnings.append("There are at least as many predictors as usable rows; estimates may be unstable.")

    id_like_features = _features_with_type(schema_table, available_features, "id_like")
    if id_like_features:
        warnings.append(f"Possible ID-like predictors selected: {', '.join(id_like_features)}.")

    high_cardinality_features = [
        column
        for column in _high_cardinality_columns(df[available_features], schema_table)
        if column in available_features
    ]
    if high_cardinality_features:
        warnings.append(f"High-cardinality categorical predictors may create sparse features: {', '.join(high_cardinality_features)}.")

    datetime_features = _features_with_type(schema_table, available_features, "datetime")
    if datetime_features and "random" in split_strategy.lower():
        warnings.append("Datetime predictors are present with a random split; time-ordered validation may be more appropriate.")

    leakage_features = _possible_leakage_features(target_column, available_features)
    if leakage_features:
        warnings.append(f"Feature names may indicate leakage from the target: {', '.join(leakage_features)}.")

    events_per_variable = None
    rare_class_warning = None
    if "classification" in task_type or task_type in {"binary", "logistic"}:
        class_counts = model_df[target_column].value_counts(dropna=True)
        if not class_counts.empty:
            smallest_class = int(class_counts.min())
            largest_class = int(class_counts.max())
            if smallest_class / max(int(class_counts.sum()), 1) <= RARE_CLASS_RATIO:
                rare_class_warning = "A rare class is present; classification metrics and logistic estimates may be unstable."
                warnings.append(rare_class_warning)
            if "binary" in task_type or task_type == "logistic":
                events_per_variable = smallest_class / max(len(available_features), 1)
                if events_per_variable < EVENTS_PER_VARIABLE_WARNING:
                    warnings.append("Events per predictor are low for logistic regression; coefficients may be unstable.")
            if largest_class == int(class_counts.sum()):
                warnings.append("The target has only one observed class after dropping missing values.")

    if "count" in task_type:
        target_values = pd.to_numeric(model_df[target_column], errors="coerce").dropna()
        if target_values.empty:
            warnings.append("The count target has no usable numeric values after dropping missing values.")
        else:
            if (target_values < 0).any():
                warnings.append("Count regression expects a nonnegative target; negative target values were found.")
            if not (target_values.dropna() % 1 == 0).all():
                warnings.append("Count regression expects integer-like target values; non-integer values were found.")
            zero_ratio = float((target_values == 0).mean())
            if zero_ratio >= 0.3:
                warnings.append("Many zero counts are present; consider zero-inflation risk when interpreting count models.")

    return {
        "selected_target": target_column,
        "selected_features": available_features,
        "target_detected_type": target_detected_type,
        "task_type": task_type,
        "sample_size_after_drop_missing": int(model_df.shape[0]),
        "feature_count": len(available_features),
        "events_per_variable": events_per_variable,
        "predictors_more_than_rows_warning": len(available_features) >= len(model_df) and len(model_df) > 0,
        "rare_class_warning": rare_class_warning,
        "high_cardinality_warning": bool(high_cardinality_features),
        "time_split_warning": bool(datetime_features and "random" in split_strategy.lower()),
        "leakage_column_warning": bool(leakage_features or id_like_features),
        "warnings": warnings,
    }


def build_rigor_checklist(
    analysis_plan: dict[str, Any] | None = None,
    working_df: pd.DataFrame | None = None,
    cleaning_log: list[dict[str, Any]] | None = None,
    transformation_log: list[dict[str, Any]] | None = None,
    model_runs: list[dict[str, Any]] | None = None,
    prediction_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a compact checklist for learning-oriented rigor."""
    plan = normalize_analysis_plan(analysis_plan)
    has_plan = analysis_plan_is_recorded(plan)
    cleaning_log = cleaning_log or []
    transformation_log = transformation_log or []
    model_runs = model_runs or []
    prediction_log = prediction_log or []
    target = plan.get("target_variable")
    readiness = build_data_readiness(working_df, target_column=target)

    return [
        _check("Research Question", "Research question is written", "completed" if plan.get("research_question") else "not_started", "State the question before interpreting model results."),
        _check("Research Question", "Analysis goal is clear", "completed" if plan.get("analysis_goal") else "not_started", "Choose exploratory, prediction, inference, or learning demo."),
        _check("Research Question", "Target variable is planned", "completed" if target else "not_started", "Choose a target before modeling."),
        _check("Data Understanding", "Dataset has been uploaded", "completed" if working_df is not None else "not_started", "Upload data before analysis."),
        _check("Data Understanding", "ID/time/grouping columns reviewed", _id_time_status(plan, readiness), "Review ID-like, datetime, and grouping columns before modeling."),
        _check("Data Understanding", "High-risk columns reviewed", _risk_column_status(readiness), "Review constant, ID-like, high-missing, and high-cardinality columns."),
        _check("Missing Data", "Missing values inspected", "completed" if working_df is not None else "not_started", "Inspect missingness before cleaning or modeling."),
        _check("Missing Data", "Cleaning decisions are logged", "completed" if cleaning_log else "not_applicable", "No cleaning has been applied yet."),
        _check("Modeling Readiness", "Transformations are documented", "completed" if transformation_log else "not_applicable", "No transformations have been applied yet."),
        _check("Modeling Readiness", "Model validation strategy is planned", "completed" if plan.get("train_test_strategy") or model_runs else "needs_attention", "Record train/test split or cross-validation strategy."),
        _check("Diagnostics", "Model diagnostics have been reviewed", _diagnostics_status(model_runs), "Review residuals, overfitting, VIF, influence, or calibration as applicable."),
        _check("Interpretation", "Interpretation boundaries are stated", "completed" if plan.get("interpretation_boundaries") else "needs_attention", "State that exploratory findings, coefficients, and feature importance are not automatic causal claims."),
        _check("Interpretation", "Prediction examples are logged if used", "completed" if prediction_log else "not_applicable", "No predictions have been logged yet."),
        _check("Interpretation", "Analysis plan exists", "completed" if has_plan else "needs_attention", "No analysis plan was recorded before modeling."),
    ]


def build_reproducibility_manifest(
    working_df: pd.DataFrame | None,
    original_df: pd.DataFrame | None = None,
    uploaded_file_name: str | None = None,
    analysis_plan: dict[str, Any] | None = None,
    cleaning_log: list[dict[str, Any]] | None = None,
    transformation_log: list[dict[str, Any]] | None = None,
    model_runs: list[dict[str, Any]] | None = None,
    generated_at: str | None = None,
    app_version: str | None = None,
) -> dict[str, Any]:
    """Return report metadata that helps reproduce the analysis choices."""
    plan = normalize_analysis_plan(analysis_plan)
    model_runs = model_runs or []
    latest_run = model_runs[-1] if model_runs else {}
    split_config = latest_run.get("split_config") or {}
    preprocessing = latest_run.get("preprocessing") or {}

    return {
        "app_version": app_version or _read_app_version(),
        "generated_at": generated_at,
        "uploaded_file_name": uploaded_file_name,
        "original_rows": int(original_df.shape[0]) if original_df is not None else None,
        "original_columns": int(original_df.shape[1]) if original_df is not None else None,
        "working_rows": int(working_df.shape[0]) if working_df is not None else None,
        "working_columns": int(working_df.shape[1]) if working_df is not None else None,
        "analysis_goal": plan.get("analysis_goal"),
        "target": plan.get("target_variable") or latest_run.get("target"),
        "features": plan.get("candidate_features") or latest_run.get("features") or [],
        "excluded_columns": plan.get("excluded_columns") or [],
        "cleaning_steps": len(cleaning_log or []),
        "transformation_steps": len(transformation_log or []),
        "train_test_split": split_config,
        "random_state": split_config.get("random_state"),
        "cv_folds": _cv_folds(preprocessing),
        "tuning_method": _tuning_method(preprocessing),
        "model_name": latest_run.get("model_name"),
        "model_family": latest_run.get("model_family"),
        "preprocessing_summary": preprocessing,
        "package_versions": _package_versions(),
        "known_limitations": [
            "Coefficients are associations unless the study design supports causal interpretation.",
            "Feature importance is not causal importance.",
            "p-values are not effect sizes.",
            "Exploratory p-values should be interpreted cautiously.",
            "Prediction intervals depend on model assumptions or empirical interval methods.",
        ],
    }


def _empty_data_readiness() -> dict[str, Any]:
    return {
        "row_count": 0,
        "column_count": 0,
        "duplicate_row_count": 0,
        "missing_summary": [],
        "high_missing_columns": [],
        "constant_columns": [],
        "id_like_columns": [],
        "high_cardinality_columns": [],
        "datetime_columns": [],
        "target_missing_count": None,
        "target_unique_count": None,
        "possible_grouping_columns": [],
    }


def _check(group: str, item: str, status: str, note: str) -> dict[str, str]:
    return {"group": group, "item": item, "status": status, "note": note}


def _id_time_status(plan: dict[str, Any], readiness: dict[str, Any]) -> str:
    found_columns = bool(readiness.get("id_like_columns") or readiness.get("datetime_columns") or readiness.get("possible_grouping_columns"))
    planned_columns = bool(plan.get("known_id_columns") or plan.get("known_time_columns") or plan.get("grouping_or_cluster_columns"))
    if planned_columns:
        return "completed"
    if found_columns:
        return "needs_attention"
    return "not_applicable"


def _risk_column_status(readiness: dict[str, Any]) -> str:
    risky = bool(
        readiness.get("constant_columns")
        or readiness.get("high_missing_columns")
        or readiness.get("id_like_columns")
        or readiness.get("high_cardinality_columns")
    )
    return "needs_attention" if risky else "completed"


def _diagnostics_status(model_runs: list[dict[str, Any]]) -> str:
    if not model_runs:
        return "not_started"
    for run in model_runs:
        if run.get("diagnostic_plot_keys") or run.get("diagnostics_summary"):
            return "completed"
        if run.get("test_metrics") or run.get("train_metrics"):
            return "needs_attention"
    return "needs_attention"


def _high_cardinality_columns(df: pd.DataFrame, schema_table: pd.DataFrame) -> list[str]:
    columns = []
    if df.empty:
        return columns
    categorical_types = {"nominal_categorical", "ordinal_categorical_candidate", "text"}
    for column in df.columns:
        detected = _detected_type(schema_table, column)
        if detected not in categorical_types:
            continue
        unique_count = int(df[column].dropna().nunique())
        ratio = unique_count / max(len(df), 1)
        if unique_count >= HIGH_CARDINALITY_UNIQUE_COUNT or ratio >= HIGH_CARDINALITY_RATIO:
            columns.append(column)
    return columns


def _possible_grouping_columns(df: pd.DataFrame, schema_table: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        detected = _detected_type(schema_table, column)
        unique_count = int(df[column].dropna().nunique())
        if detected in {"binary", "nominal_categorical", "ordinal_categorical_candidate"} and 2 <= unique_count <= 20:
            columns.append(column)
    return columns


def _detected_type(schema_table: pd.DataFrame, column: str) -> str | None:
    if schema_table is None or schema_table.empty or "variable" not in schema_table:
        return None
    match = schema_table.loc[schema_table["variable"] == column, "detected_type"]
    if match.empty:
        return None
    return str(match.iloc[0])


def _features_with_type(schema_table: pd.DataFrame, features: list[str], detected_type: str) -> list[str]:
    return [feature for feature in features if _detected_type(schema_table, feature) == detected_type]


def _possible_leakage_features(target_column: str, features: list[str]) -> list[str]:
    target_lower = target_column.lower()
    risky_words = ["target", "outcome", "label", "response", "result"]
    leakage_features = []
    for feature in features:
        feature_lower = feature.lower()
        if target_lower and target_lower in feature_lower:
            leakage_features.append(feature)
        elif any(word in feature_lower for word in risky_words):
            leakage_features.append(feature)
    return leakage_features


def _cv_folds(preprocessing: dict[str, Any]) -> Any:
    cross_validation = preprocessing.get("cross_validation")
    if isinstance(cross_validation, dict):
        return cross_validation.get("folds")
    return None


def _tuning_method(preprocessing: dict[str, Any]) -> Any:
    tuning = preprocessing.get("tuning")
    if isinstance(tuning, dict) and tuning.get("enabled"):
        return tuning.get("search_type") or tuning.get("method")
    return None


def _read_app_version() -> str:
    try:
        with open("VERSION", encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return "unknown"


def _package_versions() -> dict[str, str]:
    packages = ["streamlit", "pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "plotly"]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions
