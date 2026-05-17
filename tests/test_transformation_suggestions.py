import pandas as pd

from src.data.transformation_suggestions import build_transformation_suggestions


def test_positive_right_skewed_variable_gets_positive_transform_suggestions():
    data = pd.DataFrame({"income": [1, 2, 2, 3, 4, 8, 16, 64, 256, 1024]})

    suggestions = build_transformation_suggestions(data)
    row = suggestions.iloc[0]

    assert row["variable"] == "income"
    assert row["detected_issue"] == "positive_high_skew"
    assert row["suggested_methods"] == ["Log", "Square root", "Box-Cox", "Yeo-Johnson"]


def test_zero_right_skewed_variable_gets_log1p_suggestion():
    data = pd.DataFrame({"visits": [0, 0, 1, 1, 2, 3, 5, 8, 20, 80]})

    suggestions = build_transformation_suggestions(data)
    row = suggestions.iloc[0]

    assert row["detected_issue"] == "count_like_right_skew"
    assert "Log1p" in row["suggested_methods"]
    assert "count models" in row["warning"]


def test_negative_skewed_variable_gets_yeojohnson_or_cube_root():
    data = pd.DataFrame({"change": [-100, -40, -20, -10, -5, -2, -1, 0, 1, 2]})

    suggestions = build_transformation_suggestions(data)
    row = suggestions.iloc[0]

    assert row["detected_issue"] == "skewed_with_negative_values"
    assert row["suggested_methods"] == ["Yeo-Johnson", "Cube root"]


def test_roughly_symmetric_variable_gets_no_strong_suggestion():
    data = pd.DataFrame({"score": [-4, -3, -2, -1, 0, 1, 2, 3, 4]})

    suggestions = build_transformation_suggestions(data)
    row = suggestions.iloc[0]

    assert row["detected_issue"] == "roughly_symmetric"
    assert row["suggested_methods"] == []
    assert "No transformation is strongly suggested" in row["reason"]


def test_too_few_unique_values_do_not_get_continuous_transformations():
    data = pd.DataFrame({"rating": [1, 1, 2, 2, 3, 3, 3]})

    suggestions = build_transformation_suggestions(data)
    row = suggestions.iloc[0]

    assert row["detected_issue"] == "too_few_unique_values"
    assert row["suggested_methods"] == []
    assert "discrete or categorical" in row["warning"]


def test_suggestions_include_required_columns():
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5, None]})

    suggestions = build_transformation_suggestions(data)

    assert list(suggestions.columns) == [
        "variable",
        "detected_issue",
        "min",
        "max",
        "skewness",
        "kurtosis",
        "zero_count",
        "negative_count",
        "missing_count",
        "suggested_methods",
        "reason",
        "warning",
    ]
