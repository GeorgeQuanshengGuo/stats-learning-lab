import pandas as pd

from src.data.cleaning import (
    drop_all_rows_with_any_missing,
    drop_missing_rows_for_column,
    fill_categorical_missing_label,
    fill_categorical_mode,
    fill_numeric_mean,
    fill_numeric_median,
)


def test_drop_missing_rows_for_column_returns_new_df_and_log():
    data = pd.DataFrame({"age": [20, None, 30], "city": ["A", "B", None]})

    new_df, log_entry = drop_missing_rows_for_column(data, "age")

    assert len(new_df) == 2
    assert data["age"].isna().sum() == 1
    assert log_entry["operation"] == "drop_missing_rows_for_column"
    assert log_entry["column"] == "age"
    assert log_entry["rows_before"] == 3
    assert log_entry["rows_after"] == 2
    assert log_entry["missing_before"] == 1
    assert log_entry["missing_after"] == 0


def test_drop_all_rows_with_any_missing_returns_new_df_and_log():
    data = pd.DataFrame({"age": [20, None, 30], "city": ["A", "B", None]})

    new_df, log_entry = drop_all_rows_with_any_missing(data)

    assert len(new_df) == 1
    assert len(data) == 3
    assert log_entry["operation"] == "drop_all_rows_with_any_missing"
    assert log_entry["column"] is None
    assert log_entry["missing_before"] == 2
    assert log_entry["missing_after"] == 0


def test_fill_numeric_mean_returns_new_df_and_log():
    data = pd.DataFrame({"score": [10.0, None, 20.0]})

    new_df, log_entry = fill_numeric_mean(data, "score")

    assert new_df.loc[1, "score"] == 15.0
    assert pd.isna(data.loc[1, "score"])
    assert log_entry["operation"] == "fill_numeric_mean"
    assert log_entry["missing_before"] == 1
    assert log_entry["missing_after"] == 0


def test_fill_numeric_median_returns_new_df_and_log():
    data = pd.DataFrame({"score": [10.0, None, 30.0, 100.0]})

    new_df, log_entry = fill_numeric_median(data, "score")

    assert new_df.loc[1, "score"] == 30.0
    assert pd.isna(data.loc[1, "score"])
    assert log_entry["operation"] == "fill_numeric_median"


def test_fill_categorical_mode_returns_new_df_and_log():
    data = pd.DataFrame({"city": ["A", "A", None, "B"]})

    new_df, log_entry = fill_categorical_mode(data, "city")

    assert new_df.loc[2, "city"] == "A"
    assert pd.isna(data.loc[2, "city"])
    assert log_entry["operation"] == "fill_categorical_mode"
    assert log_entry["missing_after"] == 0


def test_fill_categorical_missing_label_returns_new_df_and_log():
    data = pd.DataFrame({"city": ["A", None, "B"]})

    new_df, log_entry = fill_categorical_missing_label(data, "city", label="Unknown")

    assert new_df.loc[1, "city"] == "Unknown"
    assert pd.isna(data.loc[1, "city"])
    assert log_entry["operation"] == "fill_categorical_missing_label"
    assert log_entry["details"] == "Filled missing values in city with the label: Unknown."
