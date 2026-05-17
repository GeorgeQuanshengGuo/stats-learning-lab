"""Classic outlier detection helpers for EDA.

These helpers identify unusual rows but never delete or modify data in place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTLIER_WARNING = (
    "Outlier detection flags unusual observations under the selected method and variables. "
    "A flagged row is not automatically an error."
)

OUTLIER_METHODS = {
    "IQR rule",
    "Z-score",
    "Modified Z-score",
    "Mahalanobis distance",
    "Isolation Forest",
    "Local Outlier Factor",
}


def run_outlier_detection(
    df: pd.DataFrame,
    numeric_columns: list[str],
    method: str = "IQR rule",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    modified_z_threshold: float = 3.5,
    mahalanobis_quantile: float = 0.975,
    contamination: float = 0.1,
    n_neighbors: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the selected EDA outlier detection method."""
    numeric_columns = list(numeric_columns)
    _validate_inputs(
        df=df,
        numeric_columns=numeric_columns,
        method=method,
        iqr_multiplier=iqr_multiplier,
        z_threshold=z_threshold,
        modified_z_threshold=modified_z_threshold,
        mahalanobis_quantile=mahalanobis_quantile,
        contamination=contamination,
        n_neighbors=n_neighbors,
    )
    numeric_data = df[numeric_columns].copy(deep=True).apply(pd.to_numeric, errors="coerce")

    if method == "IQR rule":
        labels, scores, details = detect_iqr_outliers(numeric_data, multiplier=iqr_multiplier)
        processed_values = _impute_numeric(numeric_data)
    elif method == "Z-score":
        labels, scores, details = detect_zscore_outliers(numeric_data, threshold=z_threshold)
        processed_values = _impute_numeric(numeric_data)
    elif method == "Modified Z-score":
        labels, scores, details = detect_modified_zscore_outliers(numeric_data, threshold=modified_z_threshold)
        processed_values = _impute_numeric(numeric_data)
    elif method == "Mahalanobis distance":
        labels, scores, details = detect_mahalanobis_outliers(numeric_data, quantile=mahalanobis_quantile)
        processed_values = _scaled_imputed_numeric(numeric_data)
    elif method == "Isolation Forest":
        labels, scores, details, processed_values = detect_isolation_forest_outliers(
            numeric_data,
            contamination=contamination,
            random_state=random_state,
        )
    elif method == "Local Outlier Factor":
        labels, scores, details, processed_values = detect_lof_outliers(
            numeric_data,
            contamination=contamination,
            n_neighbors=n_neighbors,
        )
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")

    result_table = build_outlier_result_table(labels, scores)
    flagged_rows = build_flagged_rows_table(df, result_table)
    pca_scores = build_outlier_pca_scores(processed_values, index=df.index)
    return {
        "method": method,
        "numeric_columns": numeric_columns,
        "parameters": _parameters_for_method(
            method=method,
            iqr_multiplier=iqr_multiplier,
            z_threshold=z_threshold,
            modified_z_threshold=modified_z_threshold,
            mahalanobis_quantile=mahalanobis_quantile,
            contamination=contamination,
            n_neighbors=n_neighbors,
            random_state=random_state,
        ),
        "details": details,
        "labels": labels,
        "scores": scores,
        "result_table": result_table,
        "flagged_rows": flagged_rows,
        "summary": {
            "flagged_count": int(labels.sum()),
            "flagged_percentage": float(labels.mean()) if len(labels) else 0.0,
            "rows_evaluated": int(len(labels)),
        },
        "pca_scores": pca_scores,
        "notes": OUTLIER_WARNING,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def detect_iqr_outliers(
    numeric_data: pd.DataFrame,
    multiplier: float = 1.5,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Detect univariate IQR-rule outliers across selected columns."""
    if multiplier <= 0:
        raise ValueError("IQR multiplier must be greater than 0.")
    data = numeric_data.copy(deep=True).apply(pd.to_numeric, errors="coerce")
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    flags = data.lt(lower, axis=1) | data.gt(upper, axis=1)
    labels = flags.any(axis=1).fillna(False)
    denominator = iqr.replace(0, np.nan)
    lower_excess = data.rsub(lower, axis=1).clip(lower=0)
    upper_excess = data.sub(upper, axis=1).clip(lower=0)
    scores = pd.DataFrame(
        np.maximum(lower_excess.to_numpy(), upper_excess.to_numpy()),
        index=data.index,
        columns=data.columns,
    )
    scores = scores.divide(denominator.replace(0, np.nan), axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
    return (
        labels.astype(bool),
        scores.max(axis=1).rename("outlier_score"),
        {
            "lower_fences": lower.to_dict(),
            "upper_fences": upper.to_dict(),
            "multiplier": float(multiplier),
        },
    )


def detect_zscore_outliers(
    numeric_data: pd.DataFrame,
    threshold: float = 3.0,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Detect univariate z-score outliers across selected columns."""
    if threshold <= 0:
        raise ValueError("Z-score threshold must be greater than 0.")
    values = _impute_numeric(numeric_data)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds = np.where(stds == 0, 1, stds)
    absolute_z = np.abs((values - means) / stds)
    scores = pd.Series(absolute_z.max(axis=1), index=numeric_data.index, name="outlier_score")
    return (
        (scores > threshold).astype(bool),
        scores,
        {
            "threshold": float(threshold),
            "lower_thresholds": dict(zip(numeric_data.columns, means - threshold * stds)),
            "upper_thresholds": dict(zip(numeric_data.columns, means + threshold * stds)),
        },
    )


def detect_modified_zscore_outliers(
    numeric_data: pd.DataFrame,
    threshold: float = 3.5,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Detect robust modified z-score outliers using median and MAD."""
    if threshold <= 0:
        raise ValueError("Modified Z-score threshold must be greater than 0.")
    values = _impute_numeric(numeric_data)
    medians = np.median(values, axis=0)
    mad = np.median(np.abs(values - medians), axis=0)
    mad = np.where(mad == 0, 1, mad)
    modified_z = np.abs(0.6745 * (values - medians) / mad)
    scores = pd.Series(modified_z.max(axis=1), index=numeric_data.index, name="outlier_score")
    raw_distance = threshold * mad / 0.6745
    return (
        (scores > threshold).astype(bool),
        scores,
        {
            "threshold": float(threshold),
            "lower_thresholds": dict(zip(numeric_data.columns, medians - raw_distance)),
            "upper_thresholds": dict(zip(numeric_data.columns, medians + raw_distance)),
        },
    )


def detect_mahalanobis_outliers(
    numeric_data: pd.DataFrame,
    quantile: float = 0.975,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Detect multivariate outliers using squared Mahalanobis distance."""
    if not 0 < quantile < 1:
        raise ValueError("Mahalanobis quantile must be between 0 and 1.")
    values = _scaled_imputed_numeric(numeric_data)
    centered = values - values.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    covariance_inverse = np.linalg.pinv(np.atleast_2d(covariance))
    distances = np.einsum("ij,jk,ik->i", centered, covariance_inverse, centered)
    threshold = float(chi2.ppf(quantile, df=values.shape[1]))
    scores = pd.Series(distances, index=numeric_data.index, name="outlier_score")
    return (
        (scores > threshold).astype(bool),
        scores,
        {"threshold": threshold, "quantile": float(quantile), "score": "squared_mahalanobis_distance"},
    )


def detect_isolation_forest_outliers(
    numeric_data: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, dict[str, Any], np.ndarray]:
    """Detect multivariate outliers with Isolation Forest."""
    _validate_contamination(contamination)
    values = _scaled_imputed_numeric(numeric_data)
    model = IsolationForest(contamination=float(contamination), random_state=random_state)
    raw_labels = model.fit_predict(values)
    scores = pd.Series(-model.decision_function(values), index=numeric_data.index, name="outlier_score")
    return (
        pd.Series(raw_labels == -1, index=numeric_data.index, name="is_outlier"),
        scores,
        {"contamination": float(contamination), "random_state": int(random_state), "score": "negative_decision_function"},
        values,
    )


def detect_lof_outliers(
    numeric_data: pd.DataFrame,
    contamination: float = 0.1,
    n_neighbors: int = 20,
) -> tuple[pd.Series, pd.Series, dict[str, Any], np.ndarray]:
    """Detect multivariate outliers with Local Outlier Factor."""
    _validate_contamination(contamination)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    values = _scaled_imputed_numeric(numeric_data)
    effective_neighbors = min(int(n_neighbors), len(numeric_data) - 1)
    model = LocalOutlierFactor(n_neighbors=effective_neighbors, contamination=float(contamination))
    raw_labels = model.fit_predict(values)
    scores = pd.Series(-model.negative_outlier_factor_, index=numeric_data.index, name="outlier_score")
    return (
        pd.Series(raw_labels == -1, index=numeric_data.index, name="is_outlier"),
        scores,
        {"contamination": float(contamination), "n_neighbors": effective_neighbors, "score": "negative_outlier_factor"},
        values,
    )


def build_outlier_result_table(labels: pd.Series, scores: pd.Series | None = None) -> pd.DataFrame:
    """Return one row per observation with outlier label and optional score."""
    table = pd.DataFrame(
        {
            "row_index": labels.index,
            "is_outlier": labels.astype(bool).to_numpy(),
        },
        index=labels.index,
    )
    if scores is not None:
        table["outlier_score"] = scores.reindex(labels.index).to_numpy()
    return table


def build_flagged_rows_table(df: pd.DataFrame, result_table: pd.DataFrame) -> pd.DataFrame:
    """Return flagged rows joined with original data values."""
    if result_table.empty:
        return pd.DataFrame()
    flagged = result_table.loc[result_table["is_outlier"]].copy()
    if flagged.empty:
        return flagged.reset_index(drop=True)
    if "outlier_score" in flagged.columns:
        flagged = flagged.sort_values("outlier_score", ascending=False)
    return flagged.join(df.loc[flagged.index].copy(deep=True), how="left").reset_index(drop=True)


def build_outlier_pca_scores(processed_values: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Return a two-component PCA projection for multivariate outlier plots."""
    if processed_values.shape[0] < 2 or processed_values.shape[1] < 2:
        return pd.DataFrame(index=index, columns=["PC1", "PC2"])
    scores = PCA(n_components=2).fit_transform(processed_values)
    return pd.DataFrame(scores, columns=["PC1", "PC2"], index=index)


def add_outlier_flag_column(
    df: pd.DataFrame,
    outlier_result: dict[str, Any],
    new_column: str = "outlier_flag",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a new DataFrame with an outlier flag and a transformation log entry."""
    if new_column in df.columns:
        raise ValueError(f"Column already exists: {new_column}")
    labels = outlier_result.get("labels")
    if not isinstance(labels, pd.Series) or labels.empty:
        raise ValueError("Outlier result does not contain labels to add.")

    new_df = df.copy(deep=True)
    new_df[new_column] = labels.reindex(df.index).fillna(False).astype(bool).to_numpy()
    source_columns = list(outlier_result.get("numeric_columns") or [])
    log_entry = {
        "operation_type": "transformation",
        "method": "outlier_flag",
        "source_columns": source_columns,
        "new_column": new_column,
        "parameters": {
            "outlier_method": outlier_result.get("method"),
            **dict(outlier_result.get("parameters") or {}),
        },
        "rows_before": len(df),
        "rows_after": len(new_df),
        "missing_before": int(df[source_columns].isna().sum().sum()) if source_columns else 0,
        "missing_after": int(new_df[new_column].isna().sum()),
        "notes": f"Added outlier flag column using {outlier_result.get('method')}. {OUTLIER_WARNING}",
    }
    return new_df, log_entry


def _validate_inputs(
    df: pd.DataFrame,
    numeric_columns: list[str],
    method: str,
    iqr_multiplier: float,
    z_threshold: float,
    modified_z_threshold: float,
    mahalanobis_quantile: float,
    contamination: float,
    n_neighbors: int,
) -> None:
    """Validate outlier detection inputs."""
    if method not in OUTLIER_METHODS:
        raise ValueError(f"Unsupported outlier detection method: {method}")
    if not numeric_columns:
        raise ValueError("Choose at least one numeric variable for outlier detection.")
    if len(df) < 2:
        raise ValueError("Outlier detection needs at least two rows.")
    missing = [column for column in numeric_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Columns were not found in the dataset: {', '.join(missing)}")
    non_numeric = [column for column in numeric_columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"Outlier detection requires numeric variables: {', '.join(non_numeric)}")
    all_missing = [column for column in numeric_columns if df[column].notna().sum() == 0]
    if all_missing:
        raise ValueError(f"Cannot use all-missing numeric variables: {', '.join(all_missing)}")
    if iqr_multiplier <= 0:
        raise ValueError("IQR multiplier must be greater than 0.")
    if z_threshold <= 0:
        raise ValueError("Z-score threshold must be greater than 0.")
    if modified_z_threshold <= 0:
        raise ValueError("Modified Z-score threshold must be greater than 0.")
    if not 0 < mahalanobis_quantile < 1:
        raise ValueError("Mahalanobis quantile must be between 0 and 1.")
    _validate_contamination(contamination)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if method == "Local Outlier Factor" and len(df) < 3:
        raise ValueError("Local Outlier Factor needs at least three rows.")


def _validate_contamination(contamination: float) -> None:
    """Validate sklearn contamination values."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be greater than 0 and less than 0.5.")


def _impute_numeric(numeric_data: pd.DataFrame) -> np.ndarray:
    """Median-impute numeric values without modifying the input."""
    return SimpleImputer(strategy="median").fit_transform(numeric_data)


def _scaled_imputed_numeric(numeric_data: pd.DataFrame) -> np.ndarray:
    """Median-impute and scale numeric values for multivariate methods."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    ).fit_transform(numeric_data)


def _parameters_for_method(
    method: str,
    iqr_multiplier: float,
    z_threshold: float,
    modified_z_threshold: float,
    mahalanobis_quantile: float,
    contamination: float,
    n_neighbors: int,
    random_state: int,
) -> dict[str, Any]:
    """Return method-specific parameters for display and logs."""
    if method == "IQR rule":
        return {"iqr_multiplier": float(iqr_multiplier)}
    if method == "Z-score":
        return {"z_threshold": float(z_threshold)}
    if method == "Modified Z-score":
        return {"modified_z_threshold": float(modified_z_threshold)}
    if method == "Mahalanobis distance":
        return {"mahalanobis_quantile": float(mahalanobis_quantile)}
    if method == "Isolation Forest":
        return {"contamination": float(contamination), "random_state": int(random_state)}
    return {"contamination": float(contamination), "n_neighbors": int(n_neighbors)}
