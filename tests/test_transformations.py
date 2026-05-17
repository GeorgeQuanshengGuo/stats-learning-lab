import numpy as np
import pandas as pd
import pytest

from src.data.transformations import (
    LOG_FIELDS,
    apply_boxcox_transform,
    apply_log_transform,
    apply_log1p_transform,
    apply_sqrt_transform,
    apply_yeojohnson_transform,
    apply_reciprocal_transform,
    create_interaction_term,
    create_polynomial_feature,
    create_ratio_feature,
    inverse_transform_available,
    inverse_transform_values,
)


def test_input_df_is_not_modified_in_place():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    original_columns = list(data.columns)

    new_df, _ = apply_log_transform(data, "x")

    assert list(data.columns) == original_columns
    assert "x_log" not in data.columns
    assert "x_log" in new_df.columns


def test_log_rejects_zero_or_negative_values():
    data = pd.DataFrame({"x": [1.0, 0.0, -2.0]})

    with pytest.raises(ValueError, match="> 0"):
        apply_log_transform(data, "x")


def test_log1p_rejects_values_less_than_or_equal_to_minus_one():
    data = pd.DataFrame({"x": [0.0, -1.0, 2.0]})

    with pytest.raises(ValueError, match="> -1"):
        apply_log1p_transform(data, "x")


def test_sqrt_rejects_negative_values():
    data = pd.DataFrame({"x": [0.0, -1.0, 4.0]})

    with pytest.raises(ValueError, match=">= 0"):
        apply_sqrt_transform(data, "x")


def test_reciprocal_rejects_zero_values():
    data = pd.DataFrame({"x": [1.0, 0.0, 2.0]})

    with pytest.raises(ValueError, match="nonzero"):
        apply_reciprocal_transform(data, "x")


def test_boxcox_rejects_zero_or_negative_values():
    data = pd.DataFrame({"x": [1.0, 0.0, 2.0]})

    with pytest.raises(ValueError, match="> 0"):
        apply_boxcox_transform(data, "x")


def test_yeojohnson_works_with_zero_and_negative_values():
    data = pd.DataFrame({"x": [-2.0, 0.0, 2.0, 4.0]})

    new_df, log_entry = apply_yeojohnson_transform(data, "x")

    assert "x_yeojohnson" in new_df.columns
    assert new_df["x_yeojohnson"].notna().all()
    assert log_entry["method"] == "yeojohnson"


def test_new_columns_are_created_correctly_for_feature_engineering():
    data = pd.DataFrame({"a": [2.0, 4.0], "b": [1.0, 2.0]})

    interaction_df, _ = create_interaction_term(data, "a", "b")
    ratio_df, _ = create_ratio_feature(data, "a", "b")
    polynomial_df, _ = create_polynomial_feature(data, "a", 3)

    assert list(interaction_df["a_x_b_interaction"]) == [2.0, 8.0]
    assert list(ratio_df["a_over_b_ratio"]) == [2.0, 2.0]
    assert list(polynomial_df["a_power_3"]) == [8.0, 64.0]


def test_log_entry_contains_required_fields():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    _, log_entry = apply_log_transform(data, "x", new_column="log_x")

    assert list(log_entry.keys()) == LOG_FIELDS
    assert log_entry["operation_type"] == "transformation"
    assert log_entry["source_columns"] == ["x"]
    assert log_entry["new_column"] == "log_x"


def test_inverse_transform_values_for_common_methods():
    data = pd.Series([1.0, 4.0, 9.0])

    assert inverse_transform_values("log", np.log(data)).round(8).equals(data)
    assert inverse_transform_values("log1p", np.log1p(data)).round(8).equals(data)
    assert inverse_transform_values("sqrt", np.sqrt(data)).round(8).equals(data)
    assert inverse_transform_values("cube_root", np.cbrt(data)).round(8).equals(data)


def test_inverse_transform_values_for_boxcox_and_yeojohnson():
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})

    boxcox_df, boxcox_log = apply_boxcox_transform(data, "x")
    boxcox_inverted = inverse_transform_values("boxcox", boxcox_df["x_boxcox"], boxcox_log["parameters"])

    yeojohnson_df, yeojohnson_log = apply_yeojohnson_transform(data, "x")
    yeojohnson_inverted = inverse_transform_values(
        "yeojohnson",
        yeojohnson_df["x_yeojohnson"],
        yeojohnson_log["parameters"],
    )

    assert boxcox_inverted.round(8).equals(data["x"])
    assert yeojohnson_inverted.round(8).equals(data["x"])


def test_inverse_transform_available_flags_supported_methods():
    assert inverse_transform_available("log")
    assert inverse_transform_available("boxcox")
    assert not inverse_transform_available("reciprocal")
