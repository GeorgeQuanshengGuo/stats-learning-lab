"""Influence diagnostics for statsmodels OLS models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import OLSInfluence


INFLUENCE_WARNING = (
    "Influential observations should be investigated. They are not automatically wrong "
    "and should not be removed without domain justification."
)


def compute_ols_influence_table(results: Any, original_index: Any | None = None) -> pd.DataFrame:
    """Return residual, leverage, Cook's distance, DFFITS, and DFBETAS diagnostics."""
    influence = OLSInfluence(results)
    n = int(results.nobs)
    p = int(len(results.params))
    row_index = _row_index(results, original_index, n)

    studentized = _safe_array(getattr(influence, "resid_studentized_external", None), n)
    if np.isnan(studentized).all():
        studentized = _safe_array(getattr(influence, "resid_studentized_internal", None), n)

    dffits = _dffits_values(influence, n)
    table = pd.DataFrame(
        {
            "row_index": row_index,
            "fitted_value": np.asarray(results.fittedvalues, dtype=float),
            "residual": np.asarray(results.resid, dtype=float),
            "standardized_residual": _safe_array(getattr(influence, "resid_studentized_internal", None), n),
            "studentized_residual": studentized,
            "leverage": _safe_array(getattr(influence, "hat_matrix_diag", None), n),
            "cooks_distance": _cooks_distance_values(influence, n),
            "dffits": dffits,
        }
    )

    dfbetas = _dfbetas_frame(influence, results, row_index)
    if not dfbetas.empty:
        table = pd.concat([table, dfbetas.reset_index(drop=True)], axis=1)
        dfbeta_columns = [column for column in dfbetas.columns if column.startswith("dfbeta_")]
        table["max_abs_dfbetas"] = table[dfbeta_columns].abs().max(axis=1)

    table["large_residual"] = table["studentized_residual"].abs() > 3
    table["high_leverage"] = table["leverage"] > (2 * p / n)
    table["strong_high_leverage"] = table["leverage"] > (3 * p / n)
    table["high_cooks_distance"] = table["cooks_distance"] > (4 / n)
    table["strong_cooks_distance"] = table["cooks_distance"] > 1
    table["large_dffits"] = table["dffits"].abs() > (2 * np.sqrt(p / n))
    table["influential_observation"] = (
        table["large_residual"]
        | table["high_leverage"]
        | table["high_cooks_distance"]
        | table["large_dffits"]
    )
    return table


def identify_large_studentized_residuals(
    influence_table: pd.DataFrame,
    threshold: float = 3,
) -> pd.DataFrame:
    """Return rows where absolute studentized residual exceeds threshold."""
    _validate_influence_table(influence_table, ["studentized_residual"])
    return influence_table.loc[influence_table["studentized_residual"].abs() > threshold].copy()


def identify_high_leverage_points(
    influence_table: pd.DataFrame,
    p: int,
    n: int,
    multiplier: float = 2,
) -> pd.DataFrame:
    """Return rows with leverage greater than multiplier * p / n."""
    _validate_influence_table(influence_table, ["leverage"])
    if p <= 0 or n <= 0:
        raise ValueError("p and n must be positive.")
    threshold = multiplier * p / n
    return influence_table.loc[influence_table["leverage"] > threshold].copy()


def identify_large_cooks_distance(
    influence_table: pd.DataFrame,
    n: int,
    threshold_rule: str = "4/n",
) -> pd.DataFrame:
    """Return rows with large Cook's distance."""
    _validate_influence_table(influence_table, ["cooks_distance"])
    threshold = _cooks_threshold(n, threshold_rule)
    return influence_table.loc[influence_table["cooks_distance"] > threshold].copy()


def identify_large_dffits(
    influence_table: pd.DataFrame,
    p: int,
    n: int,
) -> pd.DataFrame:
    """Return rows with large absolute DFFITS."""
    _validate_influence_table(influence_table, ["dffits"])
    if p <= 0 or n <= 0:
        raise ValueError("p and n must be positive.")
    threshold = 2 * np.sqrt(p / n)
    return influence_table.loc[influence_table["dffits"].abs() > threshold].copy()


def top_influential_rows(influence_table: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return rows ranked by Cook's distance for quick inspection."""
    if influence_table is None or influence_table.empty:
        return pd.DataFrame()
    return (
        influence_table.sort_values("cooks_distance", ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def _row_index(results: Any, original_index: Any | None, n: int) -> list[Any]:
    """Return original row labels when available."""
    if original_index is not None:
        labels = list(original_index)
        if len(labels) == n:
            return labels

    row_labels = getattr(getattr(results.model, "data", None), "row_labels", None)
    if row_labels is not None and len(row_labels) == n:
        return list(row_labels)
    if hasattr(results.resid, "index") and len(results.resid.index) == n:
        return list(results.resid.index)
    return list(range(n))


def _safe_array(values: Any, n: int) -> np.ndarray:
    """Return a numeric array, falling back to NaN values."""
    if values is None:
        return np.full(n, np.nan)
    try:
        array = np.asarray(values, dtype=float)
    except Exception:
        return np.full(n, np.nan)
    if len(array) != n:
        return np.full(n, np.nan)
    return array


def _cooks_distance_values(influence: OLSInfluence, n: int) -> np.ndarray:
    """Return Cook's distance values."""
    try:
        return np.asarray(influence.cooks_distance[0], dtype=float)
    except Exception:
        return np.full(n, np.nan)


def _dffits_values(influence: OLSInfluence, n: int) -> np.ndarray:
    """Return DFFITS values when available."""
    try:
        return np.asarray(influence.dffits[0], dtype=float)
    except Exception:
        try:
            return np.asarray(influence.dffits_internal[0], dtype=float)
        except Exception:
            return np.full(n, np.nan)


def _dfbetas_frame(influence: OLSInfluence, results: Any, row_index: list[Any]) -> pd.DataFrame:
    """Return DFBETAS columns when available."""
    try:
        dfbetas = np.asarray(influence.dfbetas, dtype=float)
    except Exception:
        return pd.DataFrame()
    terms = [f"dfbeta_{term}" for term in results.params.index]
    if dfbetas.ndim != 2 or dfbetas.shape[1] != len(terms):
        return pd.DataFrame()
    return pd.DataFrame(dfbetas, columns=terms, index=row_index)


def _cooks_threshold(n: int, threshold_rule: str) -> float:
    """Return a Cook's distance threshold."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if threshold_rule == "4/n":
        return 4 / n
    if threshold_rule == "1":
        return 1.0
    try:
        return float(threshold_rule)
    except (TypeError, ValueError):
        raise ValueError("threshold_rule must be '4/n', '1', or a numeric value.") from None


def _validate_influence_table(influence_table: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate influence table inputs."""
    if not isinstance(influence_table, pd.DataFrame) or influence_table.empty:
        raise ValueError("influence_table must be a non-empty pandas DataFrame.")
    missing = [column for column in required_columns if column not in influence_table.columns]
    if missing:
        raise ValueError(f"influence_table is missing required columns: {', '.join(missing)}")
