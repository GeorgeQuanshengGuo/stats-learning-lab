import numpy as np
import pandas as pd
import pytest

from src.modeling.statistical.multinomial_logistic import (
    build_multinomial_logit_formula,
    run_multinomial_logistic_regression,
)


def test_multinomial_logistic_runs_and_returns_inference_tables():
    data = _stable_multiclass_data()

    result = run_multinomial_logistic_regression(
        data,
        y_column="target",
        reference_class="A",
        x_columns=["x1", "x2", "group"],
        random_state=7,
    )

    assert result["reference_class"] == "A"
    assert result["target_classes"][0] == "A"
    assert not result["coefficient_table"].empty
    assert {
        "class",
        "reference_class",
        "term",
        "estimate",
        "std_error",
        "z_value",
        "p_value",
        "ci_lower",
        "ci_upper",
        "odds_ratio",
    }.issubset(result["coefficient_table"].columns)
    assert {"accuracy", "macro_f1", "weighted_f1", "log_loss"}.issubset(result["test_metrics"])
    assert result["test_confusion_matrix"].shape == (3, 3)
    assert set(result["test_class_metrics"]["class"]) == {"A", "B", "C"}


def test_multinomial_logistic_rejects_binary_target():
    data = _stable_multiclass_data()
    data["target"] = data["target"].replace({"C": "B"})

    with pytest.raises(ValueError, match="more than two"):
        run_multinomial_logistic_regression(
            data,
            y_column="target",
            reference_class="A",
            x_columns=["x1", "x2"],
            random_state=7,
        )


def test_reference_class_must_exist():
    data = _stable_multiclass_data()

    with pytest.raises(ValueError, match="reference class"):
        run_multinomial_logistic_regression(
            data,
            y_column="target",
            reference_class="missing",
            x_columns=["x1", "x2"],
            random_state=7,
        )


def test_multinomial_formula_mentions_reference_class():
    formula = build_multinomial_logit_formula(
        target="target",
        terms=["x1", "group"],
        target_classes=["A", "B", "C"],
        reference_class="A",
    )

    assert "reference class `A`" in formula["note"]
    assert "frac" in formula["symbolic"]


def _stable_multiclass_data() -> pd.DataFrame:
    rng = np.random.default_rng(10)
    rows = []
    for class_label, x1_center, x2_center in [("A", 0.0, 0.2), ("B", 0.6, 0.0), ("C", 1.0, 0.5)]:
        for _ in range(30):
            rows.append(
                {
                    "target": class_label,
                    "x1": rng.normal(x1_center, 1.4),
                    "x2": rng.normal(x2_center, 1.4),
                    "group": rng.choice(["g1", "g2", "g3"]),
                }
            )
    return pd.DataFrame(rows)
