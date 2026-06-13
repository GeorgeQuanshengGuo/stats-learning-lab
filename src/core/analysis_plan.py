"""Helpers for recording a lightweight analysis plan.

The plan is intentionally simple. It does not make the analysis rigorous by
itself, but it gives the user a place to state the goal before interpreting
model outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


ANALYSIS_GOALS = [
    "exploratory",
    "prediction",
    "statistical inference",
    "learning demo",
]

TARGET_TYPES = [
    "continuous numeric",
    "binary",
    "multiclass categorical",
    "ordinal categorical",
    "count",
    "not sure yet",
]

PLAN_FIELDS = [
    "analysis_title",
    "research_question",
    "analysis_goal",
    "unit_of_analysis",
    "target_variable",
    "candidate_features",
    "excluded_columns",
    "known_id_columns",
    "known_time_columns",
    "grouping_or_cluster_columns",
    "expected_target_type",
    "planned_model_family",
    "primary_metric_or_statistic",
    "train_test_strategy",
    "missing_data_strategy",
    "transformation_plan",
    "assumptions_to_check",
    "interpretation_boundaries",
    "created_at",
    "last_updated_at",
]

LIST_FIELDS = {
    "candidate_features",
    "excluded_columns",
    "known_id_columns",
    "known_time_columns",
    "grouping_or_cluster_columns",
}


def create_empty_analysis_plan() -> dict[str, Any]:
    """Return a blank analysis plan with all expected fields."""
    plan = {field: "" for field in PLAN_FIELDS}
    for field in LIST_FIELDS:
        plan[field] = []
    return plan


def create_analysis_plan(**values: Any) -> dict[str, Any]:
    """Create a normalized analysis plan from user-provided values."""
    now = values.pop("now", None) or _utc_now()
    plan = create_empty_analysis_plan()
    for field in PLAN_FIELDS:
        if field in values:
            plan[field] = _normalize_value(field, values[field])
    plan["created_at"] = values.get("created_at") or now
    plan["last_updated_at"] = values.get("last_updated_at") or now
    return plan


def update_analysis_plan(
    current_plan: dict[str, Any] | None,
    updates: dict[str, Any],
    reason: str = "",
    now: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply updates and return the new plan plus decision-log entries."""
    timestamp = now or _utc_now()
    new_plan = dict(current_plan or create_empty_analysis_plan())
    if not new_plan.get("created_at"):
        new_plan["created_at"] = timestamp

    log_entries = []
    for field, raw_value in updates.items():
        if field not in PLAN_FIELDS or field in {"created_at", "last_updated_at"}:
            continue
        old_value = _normalize_value(field, new_plan.get(field, [] if field in LIST_FIELDS else ""))
        new_value = _normalize_value(field, raw_value)
        if old_value == new_value:
            continue
        new_plan[field] = new_value
        log_entries.append(
            build_decision_log_entry(
                section=field,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                timestamp=timestamp,
            )
        )

    if log_entries:
        new_plan["last_updated_at"] = timestamp
    return normalize_analysis_plan(new_plan), log_entries


def normalize_analysis_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Return a plan with all expected keys and normalized list fields."""
    normalized = create_empty_analysis_plan()
    for field in PLAN_FIELDS:
        if plan and field in plan:
            normalized[field] = _normalize_value(field, plan[field])
    return normalized


def analysis_plan_is_recorded(plan: dict[str, Any] | None) -> bool:
    """Return True when the plan has meaningful user-entered content."""
    if not plan:
        return False
    normalized = normalize_analysis_plan(plan)
    meaningful_fields = [
        "analysis_title",
        "research_question",
        "target_variable",
        "planned_model_family",
        "primary_metric_or_statistic",
    ]
    return any(bool(normalized.get(field)) for field in meaningful_fields)


def build_decision_log_entry(
    section: str,
    old_value: Any,
    new_value: Any,
    reason: str = "",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one human-readable analysis decision log entry."""
    return {
        "timestamp": timestamp or _utc_now(),
        "section": section,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason or "Analysis plan updated.",
    }


def _normalize_value(field: str, value: Any) -> Any:
    """Normalize text/list values for stable comparisons and report output."""
    if field in LIST_FIELDS:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        return [str(item) for item in value if str(item)]
    if value is None:
        return ""
    return str(value).strip()


def _utc_now() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()
