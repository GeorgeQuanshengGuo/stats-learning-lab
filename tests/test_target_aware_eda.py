import pandas as pd

from src.eda.relationships import (
    binary_target_categorical_feature_summary,
    binary_target_numeric_feature_summary,
    categorical_target_categorical_feature_tables,
    categorical_target_numeric_feature_summary,
    numeric_target_categorical_feature_summary,
    numeric_target_numeric_feature_summary,
)
from src.eda.summary import build_summary_table
from src.eda.target_aware import (
    build_model_recommendations,
    build_target_profile,
    classify_target_type,
)
from src.visualization.relationship_plots import (
    plot_categorical_relationship_bar,
    plot_numeric_by_category_box,
    plot_numeric_vs_numeric_scatter,
)


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [100.5, 120.2, 140.7, 160.1, 180.8, 200.4],
            "visits": [0, 1, 2, 4, 7, 10],
            "churn": ["yes", "no", "yes", "no", "no", "yes"],
            "segment": ["A", "A", "B", "B", "C", "C"],
            "satisfaction": ["low", "medium", "high", "medium", "high", "low"],
            "spend": [10.0, 15.0, 20.0, 30.0, 35.0, 40.0],
        }
    )


def test_continuous_target_profile_and_recommendations():
    data = _sample_data()
    schema = build_summary_table(data)

    profile = build_target_profile(data, "price", schema)
    recommendations = build_model_recommendations(profile["target_type"])

    assert profile["target_type"] == "continuous_numeric"
    assert "Linear Regression" in recommendations["recommended_module"].tolist()
    assert "ML Regression" in recommendations["recommended_module"].tolist()


def test_count_like_target_gets_count_recommendations():
    data = _sample_data()
    schema = build_summary_table(data)

    target_type = classify_target_type(data, "visits", schema)
    recommendations = build_model_recommendations(target_type)

    assert target_type == "count_like_numeric"
    assert "Poisson / Negative Binomial" in recommendations["recommended_module"].tolist()
    assert "ML Regression" in recommendations["recommended_module"].tolist()


def test_binary_target_profile_and_recommendations():
    data = _sample_data()
    schema = build_summary_table(data)

    profile = build_target_profile(data, "churn", schema)
    recommendations = build_model_recommendations(profile["target_type"])

    assert profile["target_type"] == "binary"
    assert "Logistic Regression" in recommendations["recommended_module"].tolist()
    assert "ML Binary Classification" in recommendations["recommended_module"].tolist()


def test_categorical_target_profile_and_recommendations():
    data = _sample_data()
    schema = build_summary_table(data)

    profile = build_target_profile(data, "segment", schema)
    recommendations = build_model_recommendations(profile["target_type"])

    assert profile["target_type"] == "nominal_categorical"
    assert "Multinomial Logistic Regression" in recommendations["recommended_module"].tolist()


def test_ordinal_target_profile_and_recommendations():
    data = _sample_data()
    schema = build_summary_table(data)

    profile = build_target_profile(data, "satisfaction", schema)
    recommendations = build_model_recommendations(profile["target_type"])

    assert profile["target_type"] == "ordinal_categorical_candidate"
    assert "Ordinal Regression" in recommendations["recommended_module"].tolist()


def test_continuous_target_numeric_feature_summary_and_plot():
    data = _sample_data()

    table = numeric_target_numeric_feature_summary(data, "price", "spend")
    figure = plot_numeric_vs_numeric_scatter(data, "spend", "price")

    assert table.loc[0, "complete_rows"] == 6
    assert table.loc[0, "correlation"] > 0
    assert figure is not None


def test_continuous_target_categorical_feature_summary_and_plot():
    data = _sample_data()

    table = numeric_target_categorical_feature_summary(data, "price", "segment")
    figure = plot_numeric_by_category_box(data, "price", "segment")

    assert set(table["feature_value"]) == {"A", "B", "C"}
    assert {"mean", "median", "row_count"}.issubset(table.columns)
    assert figure is not None


def test_binary_target_numeric_feature_summary():
    data = _sample_data()

    table = binary_target_numeric_feature_summary(data, "churn", "spend")

    assert set(table["target_value"]) == {"yes", "no"}
    assert {"mean", "median", "row_count"}.issubset(table.columns)


def test_binary_target_categorical_feature_event_rate_and_plot():
    data = _sample_data()

    table = binary_target_categorical_feature_summary(data, "churn", "segment", positive_class="yes")
    figure = plot_categorical_relationship_bar(data, "segment", "churn")

    assert set(table["feature_value"]) == {"A", "B", "C"}
    assert set(table["positive_class"]) == {"yes"}
    assert table["event_rate"].between(0, 1).all()
    assert figure is not None


def test_categorical_target_numeric_feature_summary():
    data = _sample_data()

    table = categorical_target_numeric_feature_summary(data, "segment", "spend")

    assert set(table["target_value"]) == {"A", "B", "C"}
    assert {"mean", "median", "row_count"}.issubset(table.columns)


def test_categorical_target_categorical_feature_tables():
    data = _sample_data()

    contingency, row_percentages = categorical_target_categorical_feature_tables(data, "segment", "churn")

    assert not contingency.empty
    assert not row_percentages.empty
    assert "churn" in contingency.columns


def test_target_aware_helpers_do_not_modify_input_df():
    data = _sample_data()
    original = data.copy(deep=True)
    schema = build_summary_table(data)

    build_target_profile(data, "price", schema)
    numeric_target_numeric_feature_summary(data, "price", "spend")
    binary_target_categorical_feature_summary(data, "churn", "segment", positive_class="yes")
    categorical_target_categorical_feature_tables(data, "segment", "churn")

    pd.testing.assert_frame_equal(data, original)
