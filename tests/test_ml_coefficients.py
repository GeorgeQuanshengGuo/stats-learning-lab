import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from src.modeling.interpretability.coefficients import (
    PREDICTIVE_COEFFICIENT_NOTE,
    add_coefficient_interpretation_notes,
    build_ml_coefficient_table,
    extract_linear_model_coefficients,
    extract_logistic_model_coefficients,
    get_transformed_feature_names_from_pipeline,
)
from src.modeling.preprocessing import build_preprocessing_pipeline


def _fit_pipeline(data, features, target, model, scale_numeric=False):
    preprocessing = build_preprocessing_pipeline(data, features, scale_numeric=scale_numeric)
    pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
    pipeline.fit(data[features], data[target])
    return pipeline


def test_numeric_only_linear_regression_coefficients_have_expected_columns():
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 1.0, 2.0, 1.0, 2.0],
        }
    )
    pipeline = _fit_pipeline(data, ["x1", "x2"], "y", LinearRegression())

    table = extract_linear_model_coefficients(pipeline, target_name="y")

    assert {"term", "coefficient", "absolute_coefficient", "model_name", "coefficient_scale_note"}.issubset(
        table.columns
    )
    assert "p_value" not in table.columns
    assert "std_error" not in table.columns
    assert "intercept" in table["term"].tolist()
    assert {"x1", "x2"}.issubset(set(table["term"]))


def test_categorical_and_numeric_linear_regression_uses_one_hot_feature_names():
    data = pd.DataFrame(
        {
            "y": [10.0, 12.0, 13.0, 16.0, 18.0, 21.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "group": ["A", "B", "A", "B", "C", "C"],
        }
    )
    pipeline = _fit_pipeline(data, ["x", "group"], "y", LinearRegression())

    names = get_transformed_feature_names_from_pipeline(pipeline)
    table = extract_linear_model_coefficients(pipeline, target_name="y")

    assert "x" in names
    assert any(name.startswith("group_") for name in names)
    assert any(term.startswith("group_") for term in table["term"])


def test_logistic_regression_coefficients_include_positive_class_and_odds_ratios():
    data = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 1, 0, 1, 0],
            "x": [8, 1, 7, 2, 9, 1, 8, 2],
            "group": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    pipeline = _fit_pipeline(data, ["x", "group"], "target", LogisticRegression(max_iter=1000), scale_numeric=True)

    table = extract_logistic_model_coefficients(pipeline, target_name="target", positive_class="yes")

    assert {
        "term",
        "coefficient_log_odds",
        "odds_ratio",
        "absolute_coefficient",
        "positive_class",
        "coefficient_scale_note",
    }.issubset(table.columns)
    assert "p_value" not in table.columns
    assert set(table["positive_class"]) == {"yes"}
    assert np.isfinite(table["odds_ratio"]).all()


def test_scaled_numeric_features_have_scaled_note():
    data = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6], "x": [10, 20, 30, 40, 50, 60]})
    pipeline = _fit_pipeline(data, ["x"], "y", LinearRegression(), scale_numeric=True)

    table = extract_linear_model_coefficients(pipeline, target_name="y")
    notes = add_coefficient_interpretation_notes(table, scale_numeric=True)

    assert "standardized" in table["coefficient_scale_note"].iloc[0]
    assert any("scaled feature scale" in note for note in notes)


def test_build_ml_coefficient_table_adds_predictive_note():
    table = build_ml_coefficient_table(
        terms=["x"],
        coefficients=[2.0],
        model_name="Linear Regression",
        target_name="y",
        intercept=1.0,
    )
    notes = add_coefficient_interpretation_notes(table)

    assert PREDICTIVE_COEFFICIENT_NOTE in notes
    assert table.attrs["interpretation_notes"][0] == PREDICTIVE_COEFFICIENT_NOTE


def test_lasso_zero_coefficients_are_preserved():
    data = pd.DataFrame(
        {
            "y": [1.0, 1.1, 0.9, 1.05, 0.95, 1.0],
            "x1": [10, 20, 30, 40, 50, 60],
            "x2": [2, 4, 6, 8, 10, 12],
        }
    )
    pipeline = _fit_pipeline(data, ["x1", "x2"], "y", Lasso(alpha=1000, max_iter=10000), scale_numeric=True)

    table = extract_linear_model_coefficients(pipeline, target_name="y")
    non_intercept = table.loc[table["term"] != "intercept"]

    assert (non_intercept["coefficient"].abs() < 1e-12).any()
