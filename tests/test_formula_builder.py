import pandas as pd

from src.reporting.formula_builder import (
    build_decision_tree_rule_summary_latex_or_text,
    build_estimated_linear_formula,
    build_estimated_logistic_formula,
    build_estimated_poisson_formula,
    build_sklearn_linear_formula,
    build_symbolic_linear_formula,
    build_symbolic_logistic_formula,
    build_symbolic_poisson_formula,
    format_latex_number,
    sanitize_latex_variable_name,
)


def test_sanitize_latex_variable_name_handles_continuous_names():
    assert sanitize_latex_variable_name("age") == r"\mathrm{age}"
    assert sanitize_latex_variable_name("income_usd") == r"\mathrm{income\_usd}"


def test_sanitize_latex_variable_name_handles_spaces_and_special_characters():
    sanitized = sanitize_latex_variable_name("total cost_%")

    assert sanitized == r"\mathrm{total cost\_\%}"


def test_format_latex_number_uses_compact_precision():
    assert format_latex_number(1.234567, digits=4) == "1.235"
    assert format_latex_number(-0.004321, digits=2) == "-0.0043"


def test_symbolic_linear_formula_contains_terms_and_error():
    formula = build_symbolic_linear_formula("price", ["size", "bedrooms"])

    assert formula["symbolic"] == (
        r"\mathrm{price}_i = \beta_0 + \beta_1 \mathrm{size}_i"
        r" + \beta_2 \mathrm{bedrooms}_i + \epsilon_i"
    )


def test_estimated_linear_formula_formats_coefficients():
    coefficients = pd.DataFrame(
        {
            "term": ["const", "size", "bedrooms"],
            "estimate": [10.0, 2.5, -1.25],
        }
    )

    formula = build_estimated_linear_formula("price", coefficients)

    assert formula["estimated"] == (
        r"\hat{\mathrm{price}}_i = 10 + 2.5 \mathrm{size}_i - 1.25 \mathrm{bedrooms}_i"
    )


def test_categorical_dummy_terms_are_displayed_clearly():
    coefficients = [
        {"term": "intercept", "estimate": 0.5},
        {"term": "education_Bachelor", "estimate": 1.2},
        {"term": "group_Treatment", "estimate": -0.4},
    ]

    formula = build_estimated_linear_formula("score", coefficients)

    assert r'I(\mathrm{education} = \text{Bachelor})' in formula["estimated"]
    assert r'I(\mathrm{group} = \text{Treatment})' in formula["estimated"]


def test_symbolic_logistic_formula_contains_logit_and_probability():
    formula = build_symbolic_logistic_formula("default", ["income"], positive_class="yes")

    assert r"\log\left(\frac{p_i}{1 - p_i}\right)" in formula["symbolic"]
    assert r"p_i = \frac{1}{1 + \exp(-\eta_i)}" == formula["probability"]
    assert "`default` = `yes`" in formula["note"]


def test_estimated_logistic_formula_formats_coefficients():
    coefficients = [
        {"term": "intercept", "estimate": -0.5},
        {"term": "income", "estimate": 0.25},
    ]

    formula = build_estimated_logistic_formula("default", coefficients, positive_class=1)

    assert formula["estimated"] == (
        r"\log\left(\frac{\hat{p}_i}{1 - \hat{p}_i}\right) = -0.5 + 0.25 \mathrm{income}_i"
    )


def test_poisson_formulas_are_built():
    symbolic = build_symbolic_poisson_formula("count", ["exposure"])
    estimated = build_estimated_poisson_formula(
        "count",
        [{"term": "intercept", "estimate": 0.1}, {"term": "exposure", "estimate": 0.8}],
    )

    assert r"\log(\lambda_i)" in symbolic["symbolic"]
    assert r"E[\mathrm{count}_i \mid x_i] = \lambda_i" == symbolic["mean"]
    assert estimated["estimated"] == r"\log(\hat{\lambda}_i) = 0.1 + 0.8 \mathrm{exposure}_i"


def test_sklearn_linear_formula_adds_transformed_scale_note():
    formula = build_sklearn_linear_formula(
        "target",
        [{"term": "intercept", "estimate": 1.0}, {"term": "x", "estimate": 2.0}],
        "Ridge",
    )

    assert formula["estimated"] == r"\hat{\mathrm{target}}_i = 1 + 2 \mathrm{x}_i"
    assert "transformed feature scale" in formula["note"]


def test_decision_tree_rule_summary_returns_text():
    summary = build_decision_tree_rule_summary_latex_or_text(["if x <= 1: predict 0"])

    assert summary["text"] == "if x <= 1: predict 0"
    assert "not naturally a compact equation" in summary["note"]
