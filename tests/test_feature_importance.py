import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.modeling.machine_learning.classification import run_ml_binary_classification_models
from src.modeling.machine_learning.feature_importance import (
    compute_permutation_importance_table,
    compute_tree_feature_importance,
    get_transformed_feature_names,
)
from src.modeling.machine_learning.regression import run_ml_regression_models
from src.modeling.preprocessing import build_preprocessing_pipeline
from src.visualization.model_plots import plot_feature_importance_bar


def _regression_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [10, 12, 13, 15, 18, 20, 22, 24, 25, 27, 30, 32, 34, 35, 37, 40],
            "numeric_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "category": ["A", "B", "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B"],
        }
    )


def _classification_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"] * 3,
            "numeric_feature": list(range(1, 25)),
            "category": ["A", "B", "A", "B", "C", "A", "B", "C"] * 3,
        }
    )


def _fitted_random_forest_regression_pipeline() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    data = _regression_data()
    features = ["numeric_feature", "category"]
    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline(data, features)),
            ("model", RandomForestRegressor(n_estimators=20, random_state=7)),
        ]
    )
    pipeline.fit(data[features], data["target"])
    return pipeline, data[features], data["target"]


def test_transformed_feature_names_include_one_hot_encoded_categories():
    pipeline, _, _ = _fitted_random_forest_regression_pipeline()

    feature_names = get_transformed_feature_names(pipeline)

    assert "numeric_feature" in feature_names
    assert "category_A" in feature_names
    assert "category_B" in feature_names
    assert "category_C" in feature_names


def test_tree_feature_importance_aligns_with_transformed_feature_names():
    pipeline, _, _ = _fitted_random_forest_regression_pipeline()

    table = compute_tree_feature_importance(pipeline)

    assert not table.empty
    assert set(table.columns) == {"feature", "importance", "importance_std", "importance_type"}
    assert set(table["importance_type"]) == {"tree_based"}
    assert "numeric_feature" in table["feature"].tolist()
    assert any(feature.startswith("category_") for feature in table["feature"])


def test_tree_feature_importance_empty_for_non_tree_model():
    data = _regression_data()
    features = ["numeric_feature", "category"]
    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline(data, features)),
            ("model", Ridge()),
        ]
    )
    pipeline.fit(data[features], data["target"])

    table = compute_tree_feature_importance(pipeline)

    assert table.empty


def test_permutation_importance_uses_original_test_columns():
    pipeline, x_values, y_values = _fitted_random_forest_regression_pipeline()

    table = compute_permutation_importance_table(
        pipeline,
        x_values,
        y_values,
        scoring="neg_root_mean_squared_error",
        n_repeats=2,
        random_state=7,
    )

    assert not table.empty
    assert set(table["feature"]) == {"numeric_feature", "category"}
    assert set(table["importance_type"]) == {"permutation"}


def test_ml_regression_random_forest_saves_feature_importance():
    data = _regression_data()

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "category"],
        selected_models=["Random Forest Regressor"],
        random_state=7,
    )[0]
    model_run = result["model_run"]

    assert model_run["importance_type"] == "tree_based"
    assert model_run["feature_importance_table"]
    assert result["feature_importance_table"]["importance_type"].tolist() == ["tree_based"] * len(
        result["feature_importance_table"]
    )


def test_ml_classification_random_forest_saves_feature_importance():
    data = _classification_data()

    result = run_ml_binary_classification_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "category"],
        selected_models=["Random Forest Classifier"],
        positive_class="yes",
        random_state=7,
    )[0]
    model_run = result["model_run"]

    assert model_run["importance_type"] == "tree_based"
    assert model_run["feature_importance_table"]
    assert set(result["feature_importance_table"]["importance_type"]) == {"tree_based"}


def test_permutation_importance_can_be_saved_with_ml_result():
    data = _regression_data()

    result = run_ml_regression_models(
        data,
        target_column="target",
        feature_columns=["numeric_feature", "category"],
        selected_models=["Ridge"],
        random_state=7,
        compute_permutation_importance=True,
    )[0]

    assert result["model_run"]["importance_type"] == "permutation"
    assert set(result["feature_importance_table"]["importance_type"]) == {"permutation"}


def test_feature_importance_plot_returns_plotly_figure():
    pipeline, _, _ = _fitted_random_forest_regression_pipeline()
    table = compute_tree_feature_importance(pipeline)

    figure = plot_feature_importance_bar(table)

    assert figure is not None
