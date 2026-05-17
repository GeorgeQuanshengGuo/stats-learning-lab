"""Metadata helpers for professional chart cards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class ChartMetadata:
    """Descriptive context for one rendered chart."""

    chart_id: str
    title: str
    subtitle: str | None
    chart_type: str
    source_page: str
    variables_used: list[str]
    method: str | None
    sample_size: int | None
    missing_handling: str | None
    caption: str | None
    how_to_read: str | None
    warnings: list[str] = field(default_factory=list)
    next_step: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return metadata as a plain dictionary."""
        return asdict(self)


def create_chart_metadata(
    title: str,
    chart_type: str,
    source_page: str,
    variables_used: list[str] | tuple[str, ...] | None = None,
    subtitle: str | None = None,
    method: str | None = None,
    sample_size: int | None = None,
    missing_handling: str | None = None,
    caption: str | None = None,
    how_to_read: str | None = None,
    warnings: list[str] | tuple[str, ...] | str | None = None,
    next_step: str | None = None,
    chart_id: str | None = None,
    created_at: str | None = None,
) -> ChartMetadata:
    """Create validated chart metadata with sensible defaults."""
    if isinstance(warnings, str):
        warning_list = [warnings]
    else:
        warning_list = list(warnings or [])

    return ChartMetadata(
        chart_id=chart_id or f"chart_{uuid4().hex[:10]}",
        title=title,
        subtitle=subtitle,
        chart_type=chart_type,
        source_page=source_page,
        variables_used=list(variables_used or []),
        method=method,
        sample_size=sample_size,
        missing_handling=missing_handling,
        caption=caption,
        how_to_read=how_to_read,
        warnings=warning_list,
        next_step=next_step,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )
