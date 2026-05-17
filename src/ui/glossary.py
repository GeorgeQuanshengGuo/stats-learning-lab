"""Glossary data loading helpers for educational term explanations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.ui.theme import PROJECT_ROOT


GLOSSARY_PATH = PROJECT_ROOT / "assets" / "glossary_terms.json"

REQUIRED_ENTRY_FIELDS = {
    "key",
    "term",
    "short_definition",
    "long_definition",
    "example",
    "common_misunderstanding",
    "related_terms",
}

FALLBACK_ENTRY = {
    "key": "unknown",
    "term": "Unknown term",
    "category": "general",
    "short_definition": "No glossary entry is available for this term yet.",
    "long_definition": "This term has not been added to the workbench glossary.",
    "example": "No example is available.",
    "common_misunderstanding": "No common misunderstanding is documented yet.",
    "related_terms": [],
}


def _validate_entry(entry: dict[str, Any]) -> None:
    """Validate one glossary entry and raise a clear error when incomplete."""
    missing_fields = REQUIRED_ENTRY_FIELDS.difference(entry)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Glossary entry is missing required fields: {missing}")
    if not isinstance(entry.get("related_terms"), list):
        raise ValueError("Glossary entry related_terms must be a list.")


@lru_cache(maxsize=1)
def _load_default_glossary_terms() -> dict[str, dict[str, Any]]:
    """Load glossary terms from the default project asset."""
    return _load_glossary_terms_from_path(GLOSSARY_PATH)


def _load_glossary_terms_from_path(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate glossary entries from a JSON file."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Glossary terms file must contain a list of entries.")

    terms: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each glossary entry must be a JSON object.")
        _validate_entry(entry)
        key = str(entry["key"])
        if key in terms:
            raise ValueError(f"Duplicate glossary key: {key}")
        terms[key] = entry

    return terms


def load_glossary_terms(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return glossary entries keyed by their stable key."""
    if path is None:
        return dict(_load_default_glossary_terms())
    return _load_glossary_terms_from_path(Path(path))


def get_glossary_entry(key: str) -> dict[str, Any]:
    """Return a glossary entry, or a safe fallback if the key is unknown."""
    terms = load_glossary_terms()
    if key in terms:
        return terms[key]

    fallback = dict(FALLBACK_ENTRY)
    fallback["key"] = key
    fallback["term"] = key.replace("_", " ").title()
    return fallback


def list_glossary_categories() -> list[str]:
    """Return sorted glossary categories."""
    categories = {entry.get("category", "general") for entry in load_glossary_terms().values()}
    return sorted(categories)


def search_glossary_terms(
    query: str = "",
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Search glossary entries by term, definition, category, or key."""
    terms = load_glossary_terms().values()
    normalized_query = query.strip().lower()
    normalized_category = category.strip().lower() if category else None
    results = []

    for entry in terms:
        entry_category = str(entry.get("category", "general"))
        if normalized_category and normalized_category != "all":
            if entry_category.lower() != normalized_category:
                continue

        searchable_text = " ".join(
            [
                str(entry.get("key", "")),
                str(entry.get("term", "")),
                str(entry.get("short_definition", "")),
                str(entry.get("long_definition", "")),
                entry_category,
                " ".join(str(term) for term in entry.get("related_terms", [])),
            ]
        ).lower()
        if normalized_query and normalized_query not in searchable_text:
            continue
        results.append(entry)

    return sorted(results, key=lambda item: str(item.get("term", "")).lower())


def glossary_terms_as_table() -> list[dict[str, str]]:
    """Return a compact table-friendly glossary summary."""
    return [
        {
            "key": key,
            "term": entry["term"],
            "category": entry.get("category", "general"),
            "short_definition": entry["short_definition"],
        }
        for key, entry in sorted(load_glossary_terms().items())
    ]
