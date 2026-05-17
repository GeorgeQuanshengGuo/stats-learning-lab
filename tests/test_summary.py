import pandas as pd

from src.eda.missing import build_missing_table, build_row_missing_summary
from src.eda.summary import build_summary_table, dataset_overview


def test_dataset_overview_counts_rows_columns_and_missing_values():
    data = pd.DataFrame({"a": [1, None], "b": [3, 4]})

    summary = dataset_overview(data)

    assert summary["rows"] == 2
    assert summary["columns"] == 2
    assert summary["missing_values"] == 1


def test_build_summary_table_returns_one_row_per_column():
    data = pd.DataFrame({"age": [20, None, 30], "group": ["a", "b", "a"]})

    table = build_summary_table(data)

    assert list(table["variable"]) == ["age", "group"]
    assert table.loc[0, "detected_type"] == "binary"
    assert table.loc[0, "missing_count"] == 1
    assert table.loc[1, "unique_count"] == 2


def test_build_summary_table_numeric_column_stats():
    data = pd.DataFrame({"score": [1.5, 2.5, 3.5, 4.5, 5.5]})

    table = build_summary_table(data)
    row = table.iloc[0]

    assert row["detected_type"] == "continuous_numeric"
    assert row["mean"] == 3.5
    assert row["min"] == 1.5
    assert row["median"] == 3.5
    assert row["max"] == 5.5


def test_build_summary_table_categorical_top_value_stats():
    data = pd.DataFrame({"city": ["Toronto", "Montreal", "Toronto", None]})

    table = build_summary_table(data)
    row = table.iloc[0]

    assert row["top_value"] == "Toronto"
    assert row["top_count"] == 2
    assert row["top_pct"] == 50.0


def test_build_summary_table_all_missing_column():
    data = pd.DataFrame({"empty": [None, None, None]})

    table = build_summary_table(data)
    row = table.iloc[0]

    assert row["detected_type"] == "constant"
    assert row["missing_count"] == 3
    assert row["missing_pct"] == 100.0
    assert pd.isna(row["mean"])
    assert row["top_value"] is None


def test_build_summary_table_constant_column():
    data = pd.DataFrame({"status": ["active", "active", "active"]})

    table = build_summary_table(data)
    row = table.iloc[0]

    assert row["detected_type"] == "constant"
    assert row["unique_count"] == 1
    assert row["top_value"] == "active"
    assert row["top_count"] == 3
    assert row["top_pct"] == 100.0


def test_build_missing_table_counts_missing_values_by_variable():
    data = pd.DataFrame({"a": [1, None], "b": [None, None]})

    table = build_missing_table(data)

    assert list(table["variable"]) == ["a", "b"]
    assert list(table["missing_count"]) == [1, 2]
    assert list(table["missing_pct"]) == [50.0, 100.0]


def test_build_row_missing_summary_counts_missing_values_by_row():
    data = pd.DataFrame({"a": [1, None], "b": [None, None]})

    table = build_row_missing_summary(data)

    assert list(table["row_number"]) == [1, 2]
    assert list(table["missing_count"]) == [1, 2]
    assert list(table["missing_pct"]) == [50.0, 100.0]
