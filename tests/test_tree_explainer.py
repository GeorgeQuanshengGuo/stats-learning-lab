import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.modeling.interpretability.tree_explainer import (
    create_tree_plot_figure,
    extract_tree_feature_importance,
    extract_tree_rules,
    get_decision_path_for_observation,
    summarize_leaf_prediction,
)
from src.modeling.preprocessing import build_preprocessing_pipeline


def _fit_tree_pipeline(data, features, target, model):
    preprocessing = build_preprocessing_pipeline(data, features, scale_numeric=False)
    pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
    pipeline.fit(data[features], data[target])
    return pipeline


def test_tree_rules_are_generated_with_transformed_feature_names():
    data = pd.DataFrame(
        {
            "y": [1.0, 1.5, 3.0, 3.5, 6.0, 6.5],
            "x": [1, 2, 3, 4, 5, 6],
            "group": ["A", "A", "B", "B", "C", "C"],
        }
    )
    pipeline = _fit_tree_pipeline(
        data,
        ["x", "group"],
        "y",
        DecisionTreeRegressor(max_depth=3, random_state=7),
    )

    rules = extract_tree_rules(pipeline, max_depth=3)

    assert "x" in rules or "group_" in rules
    assert "|---" in rules


def test_tree_feature_importance_is_generated():
    data = pd.DataFrame({"y": [1, 1, 2, 2, 3, 3], "x": [1, 2, 3, 4, 5, 6]})
    pipeline = _fit_tree_pipeline(data, ["x"], "y", DecisionTreeRegressor(max_depth=2, random_state=7))

    importance = extract_tree_feature_importance(pipeline)

    assert list(importance.columns) == ["feature", "importance"]
    assert importance.loc[0, "feature"] == "x"
    assert importance.loc[0, "importance"] > 0


def test_decision_path_works_for_one_input_row():
    data = pd.DataFrame({"y": [1, 1, 2, 2, 3, 3], "x": [1, 2, 3, 4, 5, 6]})
    pipeline = _fit_tree_pipeline(data, ["x"], "y", DecisionTreeRegressor(max_depth=2, random_state=7))

    path = get_decision_path_for_observation(pipeline, {"x": 5})

    assert not path.empty
    assert {"step", "feature", "threshold", "value", "direction", "rule"}.issubset(path.columns)
    assert path.iloc[0]["feature"] == "x"


def test_classifier_probabilities_are_shown_when_available():
    data = pd.DataFrame(
        {
            "target": [0, 0, 0, 1, 1, 1],
            "x": [1, 2, 3, 8, 9, 10],
            "group": ["A", "A", "B", "B", "C", "C"],
        }
    )
    pipeline = _fit_tree_pipeline(
        data,
        ["x", "group"],
        "target",
        DecisionTreeClassifier(max_depth=2, random_state=7),
    )

    summary = summarize_leaf_prediction(pipeline, pd.DataFrame([{"x": 9, "group": "C"}]))

    assert summary["prediction_supported"] is True
    assert summary["predicted_class"] in {0, 1}
    assert "predicted_probabilities" in summary
    assert set(summary["predicted_probabilities"]) == {"0", "1"}
    assert summary["leaf_samples"] >= 1


def test_regression_prediction_is_shown_when_available():
    data = pd.DataFrame({"y": [1.0, 1.5, 3.0, 3.5, 6.0, 6.5], "x": [1, 2, 3, 4, 5, 6]})
    pipeline = _fit_tree_pipeline(data, ["x"], "y", DecisionTreeRegressor(max_depth=3, random_state=7))

    summary = summarize_leaf_prediction(pipeline, {"x": 5})

    assert summary["prediction_supported"] is True
    assert summary["task_type"] == "regression"
    assert isinstance(summary["predicted_value"], float)
    assert summary["leaf_samples"] >= 1


def test_tree_plot_figure_is_created():
    data = pd.DataFrame({"y": [1, 1, 2, 2, 3, 3], "x": [1, 2, 3, 4, 5, 6]})
    pipeline = _fit_tree_pipeline(data, ["x"], "y", DecisionTreeRegressor(max_depth=2, random_state=7))

    figure = create_tree_plot_figure(pipeline, max_depth=2)

    assert figure is not None
