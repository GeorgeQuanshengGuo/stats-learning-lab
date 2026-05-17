"""Relationship summaries for target-aware EDA."""

from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency


def numeric_target_numeric_feature_summary(
    df: pd.DataFrame,
    target_column: str,
    feature_column: str,
) -> pd.DataFrame:
    """Return correlation and row count for numeric y and numeric x."""
    values = _numeric_pair(df, target_column, feature_column)
    correlation = values[target_column].corr(values[feature_column]) if len(values) >= 2 else None
    return pd.DataFrame(
        [
            {
                "target": target_column,
                "feature": feature_column,
                "complete_rows": int(len(values)),
                "correlation": float(correlation) if pd.notna(correlation) else None,
            }
        ]
    )


def numeric_target_categorical_feature_summary(
    df: pd.DataFrame,
    target_column: str,
    feature_column: str,
) -> pd.DataFrame:
    """Return grouped y statistics for numeric y and categorical x."""
    values = _target_feature_frame(df, target_column, feature_column).dropna()
    if values.empty:
        return _empty_grouped_numeric_table(feature_column)

    values[target_column] = pd.to_numeric(values[target_column], errors="coerce")
    values = values.dropna(subset=[target_column])
    grouped = (
        values.groupby(feature_column, dropna=False)[target_column]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={feature_column: "feature_value", "count": "row_count"})
    )
    grouped["feature"] = feature_column
    return grouped[["feature", "feature_value", "mean", "median", "row_count"]]


def binary_target_numeric_feature_summary(
    df: pd.DataFrame,
    target_column: str,
    feature_column: str,
) -> pd.DataFrame:
    """Return grouped x statistics for binary y and numeric x."""
    values = _target_feature_frame(df, target_column, feature_column).dropna()
    if values.empty:
        return _empty_grouped_numeric_table(target_column, label_column="target_value")

    values[feature_column] = pd.to_numeric(values[feature_column], errors="coerce")
    values = values.dropna(subset=[feature_column])
    grouped = (
        values.groupby(target_column, dropna=False)[feature_column]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={target_column: "target_value", "count": "row_count"})
    )
    grouped["feature"] = feature_column
    return grouped[["feature", "target_value", "mean", "median", "row_count"]]


def binary_target_categorical_feature_summary(
    df: pd.DataFrame,
    target_column: str,
    feature_column: str,
    positive_class=None,
) -> pd.DataFrame:
    """Return event-rate table for binary y and categorical x."""
    values = _target_feature_frame(df, target_column, feature_column).dropna()
    if values.empty:
        return pd.DataFrame(columns=["feature", "feature_value", "positive_class", "row_count", "event_count", "event_rate"])

    if positive_class is None:
        positive_class = values[target_column].dropna().unique().tolist()[0]

    grouped = (
        values.assign(_event=values[target_column] == positive_class)
        .groupby(feature_column, dropna=False)["_event"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                feature_column: "feature_value",
                "count": "row_count",
                "sum": "event_count",
                "mean": "event_rate",
            }
        )
    )
    grouped["feature"] = feature_column
    grouped["positive_class"] = positive_class
    return grouped[["feature", "feature_value", "positive_class", "row_count", "event_count", "event_rate"]]


def categorical_target_numeric_feature_summary(
    df: pd.DataFrame,
    target_column: str,
    feature_column: str,
) -> pd.DataFrame:
    """Return grouped x statistics for categorical y and numeric x."""
    values = _target_feature_frame(df, target_column, feature_column).dropna()
    if values.empty:
        return _empty_grouped_numeric_table(target_column, label_column="target_value")

    values[feature_column] = pd.to_numeric(values[feature_column], errors="coerce")
    values = values.dropna(subset=[feature_column])
    grouped = (
        values.groupby(target_column, dropna=False)[feature_column]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={target_column: "target_value", "count": "row_count"})
    )
    grouped["feature"] = feature_column
    return grouped[["feature", "target_value", "mean", "median", "row_count"]]


def categorical_target_categorical_feature_tables(
    df: pd.DataFrame,
    target_column: str,
    feature_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return contingency and row-percentage tables for categorical y and x."""
    values = _target_feature_frame(df, target_column, feature_column).dropna()
    if values.empty:
        empty = pd.DataFrame()
        return empty, empty

    contingency = pd.crosstab(values[feature_column], values[target_column])
    row_percentages = contingency.div(contingency.sum(axis=1), axis=0) * 100
    return contingency.reset_index(), row_percentages.reset_index()


def numeric_numeric_relationship_summary(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
) -> pd.DataFrame:
    """Return Pearson/Spearman correlations and complete-row count."""
    values = _numeric_pair(df, column_a, column_b)
    pearson = values[column_a].corr(values[column_b], method="pearson") if len(values) >= 2 else None
    spearman = values[column_a].corr(values[column_b], method="spearman") if len(values) >= 2 else None
    return pd.DataFrame(
        [
            {
                "variable_a": column_a,
                "variable_b": column_b,
                "complete_rows": int(len(values)),
                "pearson_correlation": float(pearson) if pd.notna(pearson) else None,
                "spearman_correlation": float(spearman) if pd.notna(spearman) else None,
            }
        ]
    )


def numeric_categorical_relationship_summary(
    df: pd.DataFrame,
    numeric_column: str,
    categorical_column: str,
) -> pd.DataFrame:
    """Return grouped numeric statistics for numeric vs categorical exploration."""
    values = _target_feature_frame(df, numeric_column, categorical_column)
    values[numeric_column] = pd.to_numeric(values[numeric_column], errors="coerce")
    values = values.dropna(subset=[numeric_column, categorical_column])
    if values.empty:
        return pd.DataFrame(
            columns=["category", "count", "mean", "median", "std", "min", "max"]
        )

    grouped = (
        values.groupby(categorical_column, dropna=False)[numeric_column]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={categorical_column: "category"})
    )
    return grouped


def categorical_categorical_relationship_tables(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
) -> dict[str, pd.DataFrame]:
    """Return contingency, percentage tables, and chi-square summary."""
    values = _target_feature_frame(df, column_a, column_b).dropna()
    if values.empty:
        empty = pd.DataFrame()
        return {
            "contingency_table": empty,
            "row_percentage_table": empty,
            "column_percentage_table": empty,
            "chi_square_summary": pd.DataFrame(),
        }

    contingency = pd.crosstab(values[column_a], values[column_b])
    row_percentages = contingency.div(contingency.sum(axis=1), axis=0) * 100
    column_percentages = contingency.div(contingency.sum(axis=0), axis=1) * 100
    chi_square_summary = _chi_square_summary(contingency)
    return {
        "contingency_table": contingency.reset_index(),
        "row_percentage_table": row_percentages.reset_index(),
        "column_percentage_table": column_percentages.reset_index(),
        "chi_square_summary": chi_square_summary,
    }


def datetime_numeric_relationship_summary(
    df: pd.DataFrame,
    datetime_column: str,
    numeric_column: str,
) -> pd.DataFrame:
    """Return simple date range and sample-size facts for datetime vs numeric."""
    _validate_columns(df, [datetime_column, numeric_column])
    values = df[[datetime_column, numeric_column]].copy(deep=True)
    values[datetime_column] = pd.to_datetime(values[datetime_column], errors="coerce")
    values[numeric_column] = pd.to_numeric(values[numeric_column], errors="coerce")
    values = values.dropna()
    if values.empty:
        return pd.DataFrame(
            [
                {
                    "datetime_variable": datetime_column,
                    "numeric_variable": numeric_column,
                    "complete_rows": 0,
                    "start": None,
                    "end": None,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "datetime_variable": datetime_column,
                "numeric_variable": numeric_column,
                "complete_rows": int(len(values)),
                "start": values[datetime_column].min(),
                "end": values[datetime_column].max(),
            }
        ]
    )


def _numeric_pair(df: pd.DataFrame, column_a: str, column_b: str) -> pd.DataFrame:
    """Return complete numeric rows for two columns."""
    values = _target_feature_frame(df, column_a, column_b)
    values[column_a] = pd.to_numeric(values[column_a], errors="coerce")
    values[column_b] = pd.to_numeric(values[column_b], errors="coerce")
    return values.dropna()


def _target_feature_frame(df: pd.DataFrame, target_column: str, feature_column: str) -> pd.DataFrame:
    """Return a safe two-column copy for relationship summaries."""
    _validate_columns(df, [target_column, feature_column])
    return df[[target_column, feature_column]].copy(deep=True)


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if requested columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Columns were not found in the dataset: {', '.join(missing)}")


def _empty_grouped_numeric_table(group_column: str, label_column: str = "feature_value") -> pd.DataFrame:
    """Return an empty grouped numeric table with stable columns."""
    return pd.DataFrame(columns=["feature", label_column, "mean", "median", "row_count"])


def _chi_square_summary(contingency: pd.DataFrame) -> pd.DataFrame:
    """Return chi-square test output when the contingency table is usable."""
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return pd.DataFrame(
            [
                {
                    "chi_square": None,
                    "p_value": None,
                    "degrees_of_freedom": None,
                    "note": "Chi-square test requires at least a 2 by 2 contingency table.",
                }
            ]
        )

    try:
        chi_square, p_value, dof, _ = chi2_contingency(contingency)
    except ValueError as error:
        return pd.DataFrame(
            [
                {
                    "chi_square": None,
                    "p_value": None,
                    "degrees_of_freedom": None,
                    "note": str(error),
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "chi_square": float(chi_square),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "note": "Chi-square test describes association, not causation.",
            }
        ]
    )
