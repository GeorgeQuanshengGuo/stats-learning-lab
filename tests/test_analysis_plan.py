from src.core.analysis_plan import (
    analysis_plan_is_recorded,
    create_analysis_plan,
    create_empty_analysis_plan,
    update_analysis_plan,
)


def test_create_empty_analysis_plan_contains_expected_fields() -> None:
    plan = create_empty_analysis_plan()

    assert plan["analysis_title"] == ""
    assert plan["candidate_features"] == []
    assert plan["created_at"] == ""


def test_create_analysis_plan_normalizes_lists_and_timestamps() -> None:
    plan = create_analysis_plan(
        analysis_title="Demo",
        research_question="What predicts y?",
        candidate_features=["x1", "x2"],
        now="2026-06-13T00:00:00+00:00",
    )

    assert plan["analysis_title"] == "Demo"
    assert plan["candidate_features"] == ["x1", "x2"]
    assert plan["created_at"] == "2026-06-13T00:00:00+00:00"
    assert plan["last_updated_at"] == "2026-06-13T00:00:00+00:00"
    assert analysis_plan_is_recorded(plan) is True


def test_update_analysis_plan_returns_decision_log_entries() -> None:
    plan = create_analysis_plan(
        analysis_title="Old title",
        target_variable="old_y",
        now="2026-06-13T00:00:00+00:00",
    )

    updated, entries = update_analysis_plan(
        plan,
        {
            "analysis_title": "New title",
            "target_variable": "new_y",
            "candidate_features": ["x"],
        },
        reason="Refined after upload.",
        now="2026-06-13T01:00:00+00:00",
    )

    assert updated["analysis_title"] == "New title"
    assert updated["target_variable"] == "new_y"
    assert updated["candidate_features"] == ["x"]
    assert updated["created_at"] == "2026-06-13T00:00:00+00:00"
    assert updated["last_updated_at"] == "2026-06-13T01:00:00+00:00"
    assert len(entries) == 3
    assert entries[0]["reason"] == "Refined after upload."


def test_update_analysis_plan_ignores_unchanged_fields() -> None:
    plan = create_analysis_plan(analysis_title="Same", now="2026-06-13T00:00:00+00:00")

    updated, entries = update_analysis_plan(
        plan,
        {"analysis_title": "Same"},
        now="2026-06-13T01:00:00+00:00",
    )

    assert updated["last_updated_at"] == "2026-06-13T00:00:00+00:00"
    assert entries == []
