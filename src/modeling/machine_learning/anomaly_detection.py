"""Exploratory anomaly detection helpers for numeric feature sets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANOMALY_WARNING = (
    "Anomaly detection flags unusual observations according to selected features and algorithm. "
    "It does not prove that rows are errors."
)

ANOMALY_LOG_FIELDS = [
    "operation_type",
    "method",
    "source_columns",
    "new_column",
    "parameters",
    "rows_before",
    "rows_after",
    "missing_before",
    "missing_after",
    "notes",
]


def run_anomaly_detection(
    df: pd.DataFrame,
    feature_columns: list[str],
    method: str = "IQR rule",
    scale_numeric: bool = True,
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    contamination: float = 0.1,
    n_neighbors: int = 20,
    random_state: int = 42,
    top_n: int = 20,
) -> dict[str, Any]:
    """Run exploratory anomaly detection on selected numeric features."""
    feature_columns = list(feature_columns)
    _validate_anomaly_inputs(
        df=df,
        feature_columns=feature_columns,
        method=method,
        iqr_multiplier=iqr_multiplier,
        z_threshold=z_threshold,
        contamination=contamination,
        n_neighbors=n_neighbors,
    )
    model_df = df[feature_columns].copy(deep=True).apply(pd.to_numeric, errors="coerce")
    imputed_values = SimpleImputer(strategy="median").fit_transform(model_df)

    if method == "IQR rule":
        labels, scores = _detect_iqr(model_df, imputed_values, iqr_multiplier)
        model = None
        processed_values = imputed_values
    elif method == "Z-score":
        labels, scores = _detect_z_score(imputed_values, z_threshold)
        model = None
        processed_values = imputed_values
    else:
        preprocessing = _build_numeric_preprocessing(scale_numeric=scale_numeric)
        processed_values = preprocessing.fit_transform(model_df)
        if method == "Isolation Forest":
            model = IsolationForest(
                contamination=float(contamination),
                random_state=random_state,
            )
            raw_predictions = model.fit_predict(processed_values)
            scores = -model.decision_function(processed_values)
        elif method == "Local Outlier Factor":
            model = LocalOutlierFactor(
                n_neighbors=min(int(n_neighbors), len(df) - 1),
                contamination=float(contamination),
            )
            raw_predictions = model.fit_predict(processed_values)
            scores = -model.negative_outlier_factor_
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
        labels = raw_predictions == -1

    anomaly_flags = pd.Series(labels.astype(bool), index=df.index, name="is_anomaly")
    score_series = pd.Series(scores, index=df.index, name="anomaly_score")
    result_table = build_anomaly_result_table(anomaly_flags, score_series)
    top_anomalies = build_top_anomalies_table(df, result_table, top_n=top_n)

    return {
        "method": method,
        "feature_columns": feature_columns,
        "scale_numeric": scale_numeric,
        "parameters": _parameters_for_method(
            method=method,
            iqr_multiplier=iqr_multiplier,
            z_threshold=z_threshold,
            contamination=contamination,
            n_neighbors=n_neighbors,
            random_state=random_state,
        ),
        "model": model,
        "processed_values": processed_values,
        "anomaly_labels": anomaly_flags,
        "anomaly_scores": score_series,
        "result_table": result_table,
        "summary": {
            "number_of_anomalies": int(anomaly_flags.sum()),
            "anomaly_percentage": float(anomaly_flags.mean()) if len(anomaly_flags) else 0.0,
            "rows_evaluated": int(len(anomaly_flags)),
        },
        "top_anomalies": top_anomalies,
        "pca_scores": build_anomaly_pca_scores(processed_values, index=df.index),
        "feature_boxplot_data": build_feature_boxplot_data(model_df, anomaly_flags),
        "notes": ANOMALY_WARNING,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_anomaly_result_table(labels: pd.Series, scores: pd.Series | None = None) -> pd.DataFrame:
    """Return one row per observation with anomaly label and score."""
    table = pd.DataFrame(
        {
            "row_index": labels.index,
            "is_anomaly": labels.astype(bool).to_numpy(),
        },
        index=labels.index,
    )
    if scores is not None:
        table["anomaly_score"] = scores.reindex(labels.index).to_numpy()
    return table


def build_top_anomalies_table(df: pd.DataFrame, result_table: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the most anomalous flagged rows with original data values."""
    if result_table.empty:
        return pd.DataFrame()

    flagged = result_table.loc[result_table["is_anomaly"]].copy()
    if flagged.empty:
        return flagged
    if "anomaly_score" in flagged.columns:
        flagged = flagged.sort_values("anomaly_score", ascending=False)

    original_rows = df.loc[flagged.index].copy(deep=True)
    output = flagged.join(original_rows, how="left")
    return output.head(int(top_n)).reset_index(drop=True)


def build_anomaly_pca_scores(processed_values: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Project processed features to two PCA dimensions for anomaly plotting."""
    if processed_values.shape[0] < 2 or processed_values.shape[1] < 2:
        return pd.DataFrame(index=index, columns=["PC1", "PC2"])
    scores = PCA(n_components=2).fit_transform(processed_values)
    return pd.DataFrame(scores, columns=["PC1", "PC2"], index=index)


def build_feature_boxplot_data(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Return long-form data for per-feature boxplots grouped by anomaly flag."""
    output = features.copy(deep=True)
    output["anomaly_status"] = np.where(labels.to_numpy(), "Anomaly", "Typical")
    return output.melt(
        id_vars="anomaly_status",
        var_name="feature",
        value_name="value",
    )


def add_anomaly_flag_to_dataframe(
    df: pd.DataFrame,
    anomaly_result: dict[str, Any],
    new_column: str = "anomaly_flag",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a new DataFrame with anomaly flags and a transformation log."""
    if new_column in df.columns:
        raise ValueError(f"Column already exists: {new_column}")
    labels = anomaly_result.get("anomaly_labels")
    if not isinstance(labels, pd.Series) or labels.empty:
        raise ValueError("Anomaly result does not contain labels to add.")

    new_df = df.copy(deep=True)
    new_df[new_column] = labels.reindex(df.index).fillna(False).astype(bool).to_numpy()
    feature_columns = list(anomaly_result.get("feature_columns") or [])

    log_entry = {
        "operation_type": "transformation",
        "method": "anomaly_flag",
        "source_columns": feature_columns,
        "new_column": new_column,
        "parameters": {
            "anomaly_method": anomaly_result.get("method"),
            "scale_numeric": bool(anomaly_result.get("scale_numeric", True)),
            **dict(anomaly_result.get("parameters") or {}),
        },
        "rows_before": len(df),
        "rows_after": len(new_df),
        "missing_before": int(df[feature_columns].isna().sum().sum()) if feature_columns else 0,
        "missing_after": int(new_df[new_column].isna().sum()),
        "notes": f"Added anomaly flag column using {anomaly_result.get('method')}. {ANOMALY_WARNING}",
    }
    return new_df, log_entry


def save_anomaly_result_to_session(session_state: Any, anomaly_result: dict[str, Any]) -> None:
    """Store the latest anomaly result in Streamlit session state."""
    session_state["latest_anomaly_result"] = anomaly_result
    session_state.setdefault("anomaly_artifacts", [])
    session_state["anomaly_artifacts"].append(
        {
            "created_at": anomaly_result.get("created_at"),
            "method": anomaly_result.get("method"),
            "feature_columns": anomaly_result.get("feature_columns"),
            "scale_numeric": anomaly_result.get("scale_numeric"),
            "parameters": anomaly_result.get("parameters"),
            "summary": anomaly_result.get("summary"),
        }
    )


def _detect_iqr(
    features: pd.DataFrame,
    imputed_values: np.ndarray,
    iqr_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag rows outside IQR fences in any selected feature."""
    q1 = np.nanpercentile(imputed_values, 25, axis=0)
    q3 = np.nanpercentile(imputed_values, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    below = np.maximum(lower - imputed_values, 0)
    above = np.maximum(imputed_values - upper, 0)
    denominator = np.where(iqr == 0, 1, iqr)
    exceedance = np.maximum(below, above) / denominator
    labels = (exceedance > 0).any(axis=1)
    scores = exceedance.max(axis=1)
    return labels, scores


def _detect_z_score(imputed_values: np.ndarray, z_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Flag rows with large absolute z-scores in any selected feature."""
    means = imputed_values.mean(axis=0)
    stds = imputed_values.std(axis=0)
    stds = np.where(stds == 0, 1, stds)
    absolute_z = np.abs((imputed_values - means) / stds)
    scores = absolute_z.max(axis=1)
    labels = scores > z_threshold
    return labels, scores


def _build_numeric_preprocessing(scale_numeric: bool) -> Pipeline:
    """Build numeric preprocessing for model-based anomaly detection."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _validate_anomaly_inputs(
    df: pd.DataFrame,
    feature_columns: list[str],
    method: str,
    iqr_multiplier: float,
    z_threshold: float,
    contamination: float,
    n_neighbors: int,
) -> None:
    """Validate feature and method choices."""
    if method not in {"IQR rule", "Z-score", "Isolation Forest", "Local Outlier Factor"}:
        raise ValueError(f"Unsupported anomaly detection method: {method}")
    if len(feature_columns) < 1:
        raise ValueError("Choose at least one numeric feature for anomaly detection.")
    if len(df) < 2:
        raise ValueError("Anomaly detection needs at least two rows.")

    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns were not found in the dataset: {', '.join(missing_features)}")

    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"Anomaly detection requires numeric feature columns: {', '.join(non_numeric)}")

    all_missing = [column for column in feature_columns if df[column].notna().sum() == 0]
    if all_missing:
        raise ValueError(f"Anomaly detection cannot use all-missing numeric columns: {', '.join(all_missing)}")

    if iqr_multiplier <= 0:
        raise ValueError("IQR multiplier must be greater than 0.")
    if z_threshold <= 0:
        raise ValueError("Z-score threshold must be greater than 0.")
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be greater than 0 and less than 0.5.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if method == "Local Outlier Factor" and len(df) < 3:
        raise ValueError("Local Outlier Factor needs at least three rows.")


def _parameters_for_method(
    method: str,
    iqr_multiplier: float,
    z_threshold: float,
    contamination: float,
    n_neighbors: int,
    random_state: int,
) -> dict[str, Any]:
    """Return method-specific parameters for logs and artifacts."""
    if method == "IQR rule":
        return {"iqr_multiplier": float(iqr_multiplier)}
    if method == "Z-score":
        return {"z_threshold": float(z_threshold)}
    if method == "Isolation Forest":
        return {"contamination": float(contamination), "random_state": int(random_state)}
    return {"contamination": float(contamination), "n_neighbors": int(n_neighbors)}
