import pandas as pd

from src.data.type_detector import detect_column_type, detect_variable_types


def test_detect_variable_types_returns_required_columns():
    data = pd.DataFrame({"age": [21, 32, 43], "group": ["a", "b", "a"]})

    result = detect_variable_types(data)

    assert list(result.columns) == [
        "variable",
        "raw_dtype",
        "detected_type",
        "missing_count",
        "missing_pct",
        "unique_count",
        "example_values",
    ]


def test_detect_constant_column_type():
    assert detect_column_type(pd.Series(["same", "same", None])) == "constant"


def test_detect_binary_column_type():
    assert detect_column_type(pd.Series([True, False, True])) == "binary"


def test_detect_datetime_column_type():
    values = pd.Series(["2024-01-01", "2024-02-15", "not a date", "2024-03-20", "2024-04-10"])

    assert detect_column_type(values) == "datetime"


def test_detect_id_like_column_type():
    values = pd.Series([f"CUST-{number:03d}" for number in range(20)])

    assert detect_column_type(values, column_name="customer_id") == "id_like"


def test_detect_text_column_type():
    values = pd.Series(
        [
            f"This is a long free-text response from customer number {number}."
            for number in range(20)
        ]
    )

    assert detect_column_type(values) == "text"


def test_detect_continuous_numeric_column_type():
    values = pd.Series([number + 0.25 for number in range(30)])

    assert detect_column_type(values) == "continuous_numeric"


def test_detect_discrete_numeric_column_type():
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3)

    assert detect_column_type(values) == "discrete_numeric"


def test_detect_ordinal_text_column_type():
    values = pd.Series(["low", "medium", "high", "medium", "low"])

    assert detect_column_type(values) == "ordinal_categorical_candidate"


def test_detect_ordinal_rating_column_type():
    values = pd.Series([1, 2, 3, 4, 5, 4, 3])

    assert detect_column_type(values) == "ordinal_categorical_candidate"


def test_detect_nominal_categorical_column_type():
    values = pd.Series(["red", "blue", "green", "yellow", "purple"])

    assert detect_column_type(values) == "nominal_categorical"
