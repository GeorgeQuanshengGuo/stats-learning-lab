"""Build concise LaTeX/text formulas for fitted model runs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def sanitize_latex_variable_name(name: Any) -> str:
    """Return a variable name that is safe to place inside LaTeX."""
    text = str(name)
    replacements = {
        "\\": r"\backslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return rf"\mathrm{{{text}}}"


def format_latex_number(value: Any, digits: int = 4) -> str:
    """Format a coefficient for a compact estimated formula."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if pd.isna(number):
        return "nan"
    if number == 0:
        return "0"
    formatted = f"{number:.{digits}g}"
    return formatted.replace("e", r"\times 10^{") + "}" if "e" in formatted else formatted


def build_symbolic_linear_formula(target: str, terms: list[str]) -> dict[str, str]:
    """Build a symbolic linear regression formula."""
    target_latex = sanitize_latex_variable_name(target)
    rhs = _symbolic_rhs(terms)
    return {
        "symbolic": rf"{target_latex}_i = {rhs} + \epsilon_i",
        "note": "Symbolic linear regression formula.",
    }


def build_estimated_linear_formula(target: str, coefficient_table: Any) -> dict[str, str]:
    """Build an estimated linear regression formula from fitted coefficients."""
    target_latex = sanitize_latex_variable_name(target)
    rows = _coefficient_rows(coefficient_table)
    return {
        "estimated": rf"\hat{{{target_latex}}}_i = {_estimated_rhs(rows)}",
        "note": "Estimated linear regression formula.",
    }


def build_symbolic_logistic_formula(
    target: str,
    terms: list[str],
    positive_class: Any | None = None,
) -> dict[str, str]:
    """Build symbolic logit and probability formulas."""
    rhs = _symbolic_rhs(terms)
    target_text = _target_positive_text(target, positive_class)
    return {
        "symbolic": rf"\log\left(\frac{{p_i}}{{1 - p_i}}\right) = {rhs}",
        "linear_predictor": rf"\eta_i = {rhs}",
        "probability": r"p_i = \frac{1}{1 + \exp(-\eta_i)}",
        "note": f"Here p_i is the probability of {target_text}.",
    }


def build_estimated_logistic_formula(
    target: str,
    coefficient_table: Any,
    positive_class: Any | None = None,
) -> dict[str, str]:
    """Build an estimated logistic regression formula."""
    rows = _coefficient_rows(coefficient_table)
    target_text = _target_positive_text(target, positive_class)
    return {
        "estimated": rf"\log\left(\frac{{\hat{{p}}_i}}{{1 - \hat{{p}}_i}}\right) = {_estimated_rhs(rows)}",
        "probability": r"\hat{p}_i = \frac{1}{1 + \exp(-\hat{\eta}_i)}",
        "note": f"Here \\hat{{p}}_i is the estimated probability of {target_text}.",
    }


def build_symbolic_poisson_formula(target: str, terms: list[str]) -> dict[str, str]:
    """Build a symbolic Poisson regression formula."""
    target_latex = sanitize_latex_variable_name(target)
    rhs = _symbolic_rhs(terms)
    return {
        "symbolic": rf"\log(\lambda_i) = {rhs}",
        "linear_predictor": rf"\eta_i = {rhs}",
        "mean": rf"E[{target_latex}_i \mid x_i] = \lambda_i",
        "note": "Symbolic Poisson regression formula.",
    }


def build_estimated_poisson_formula(target: str, coefficient_table: Any) -> dict[str, str]:
    """Build an estimated Poisson regression formula."""
    target_latex = sanitize_latex_variable_name(target)
    rows = _coefficient_rows(coefficient_table)
    return {
        "estimated": rf"\log(\hat{{\lambda}}_i) = {_estimated_rhs(rows)}",
        "mean": rf"\widehat{{E[{target_latex}_i \mid x_i]}} = \hat{{\lambda}}_i",
        "note": "Estimated Poisson regression formula.",
    }


def build_sklearn_linear_formula(target: str, coefficient_table: Any, model_name: str) -> dict[str, str]:
    """Build an estimated formula for sklearn linear or logistic estimators."""
    model_name_lower = model_name.lower()
    if "logistic" in model_name_lower:
        formula = build_estimated_logistic_formula(target, coefficient_table)
    else:
        formula = build_estimated_linear_formula(target, coefficient_table)

    formula["note"] = (
        f"{model_name} coefficients are shown on the transformed feature scale used by the sklearn Pipeline. "
        "If numeric scaling or one-hot encoding was used, these are not original-scale coefficients."
    )
    return formula


def build_decision_tree_rule_summary_latex_or_text(tree_rules: str | list[str]) -> dict[str, str]:
    """Return a readable rule summary for tree models."""
    if isinstance(tree_rules, list):
        rule_text = "\n".join(str(rule) for rule in tree_rules)
    else:
        rule_text = str(tree_rules or "")

    if not rule_text.strip():
        rule_text = "No decision tree rules are available."

    return {
        "text": rule_text,
        "note": "Decision tree rules are shown as text because a full tree is not naturally a compact equation.",
    }


def _symbolic_rhs(terms: list[str]) -> str:
    """Build the right-hand side of a symbolic model equation."""
    pieces = [r"\beta_0"]
    for index, term in enumerate(terms, start=1):
        pieces.append(rf"\beta_{index} {_term_to_latex(term)}")

    return " + ".join(pieces)


def _estimated_rhs(rows: list[dict[str, Any]]) -> str:
    """Build the right-hand side of an estimated model equation."""
    intercept = _intercept_value(rows)
    pieces = [format_latex_number(intercept)]

    for row in rows:
        term = str(row.get("term", ""))
        if _is_intercept(term):
            continue
        coefficient = row.get("estimate")
        pieces.append(_signed_term(coefficient, _term_to_latex(term)))

    return " ".join(pieces)


def _signed_term(coefficient: Any, term_latex: str) -> str:
    """Format one signed coefficient-times-term piece."""
    try:
        number = float(coefficient)
    except (TypeError, ValueError):
        return rf"+ {coefficient} {term_latex}"

    sign = "+" if number >= 0 else "-"
    return rf"{sign} {format_latex_number(abs(number))} {term_latex}"


def _term_to_latex(term: str) -> str:
    """Render continuous or dummy-coded terms in readable LaTeX."""
    term_text = str(term)
    if _is_intercept(term_text):
        return "1"
    dummy = _dummy_parts(term_text)
    if dummy is not None:
        variable, level = dummy
        return rf"I({sanitize_latex_variable_name(variable)} = \text{{{_escape_text(level)}}})"
    return rf"{sanitize_latex_variable_name(term_text)}_i"


def _dummy_parts(term: str) -> tuple[str, str] | None:
    """Return variable/category parts for common one-hot encoded names."""
    if "=" in term:
        variable, level = term.split("=", maxsplit=1)
        return variable.strip(), level.strip().strip('"')
    if "_" not in term:
        return None

    variable, level = term.rsplit("_", maxsplit=1)
    if not variable or not level:
        return None

    looks_categorical = level[:1].isupper() or " " in level or level.isdigit()
    if not looks_categorical:
        return None
    return variable, level


def _coefficient_rows(coefficient_table: Any) -> list[dict[str, Any]]:
    """Normalize coefficient tables into a list of dictionaries."""
    if isinstance(coefficient_table, pd.DataFrame):
        rows = coefficient_table.to_dict(orient="records")
    elif isinstance(coefficient_table, list):
        rows = [row for row in coefficient_table if isinstance(row, dict)]
    else:
        return []

    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        if "estimate" not in normalized:
            if "coefficient" in normalized:
                normalized["estimate"] = normalized["coefficient"]
            elif "coefficient_log_odds" in normalized:
                normalized["estimate"] = normalized["coefficient_log_odds"]
        normalized_rows.append(normalized)
    return normalized_rows


def _intercept_value(rows: list[dict[str, Any]]) -> Any:
    """Return the fitted intercept value, defaulting to zero."""
    for row in rows:
        if _is_intercept(str(row.get("term", ""))):
            return row.get("estimate", 0)
    return 0


def _is_intercept(term: str) -> bool:
    """Return True for common intercept term names."""
    return term.strip().lower() in {"const", "intercept", "(intercept)"}


def _target_positive_text(target: str, positive_class: Any | None) -> str:
    """Build human-readable target text for logistic probability notes."""
    target_text = str(target)
    if positive_class is None:
        return f"`{target_text}` being the positive class"
    return f"`{target_text}` = `{positive_class}`"


def _escape_text(value: Any) -> str:
    """Escape text used inside LaTeX text braces."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
