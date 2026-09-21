"""
Email classification logic — validates LLM output and merges it with parsed emails.
"""

from __future__ import annotations

import json
from typing import Any

from classifier.llm_client import classify_with_llm
from config.settings import ACTIONS, CATEGORIES, PRIORITIES


def _validate_classification(data: dict[str, Any]) -> dict[str, str | float]:
    """
    Validate and normalize a classification dict from the LLM.

    Falls back to safe defaults when fields are missing or invalid.
    """
    category = data.get("category", "other")
    if category not in CATEGORIES:
        category = "other"

    priority = data.get("priority", "medium")
    if priority not in PRIORITIES:
        priority = "medium"

    action = data.get("recommended_action", "review")
    if action not in ACTIONS:
        action = "review"

    confidence = data.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.5

    summary = str(data.get("summary", "Unable to summarize this email.")).strip()
    if not summary:
        summary = "Unable to summarize this email."

    return {
        "category": category,
        "priority": priority,
        "recommended_action": action,
        "confidence": confidence,
        "summary": summary,
    }


def _parse_llm_response(raw: str) -> dict[str, str | float]:
    """Parse raw LLM JSON text, returning fallback values on failure."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return _validate_classification(data)
    except json.JSONDecodeError:
        pass
    return _validate_classification({})


def classify_email(email: dict[str, str]) -> dict[str, str | float]:
    """
    Classify a single parsed email and return an enriched dictionary.

    Preserves all Phase 1 fields (``id``, ``sender``, ``subject``, ``snippet``)
    and adds classification metadata from the LLM.

    Args:
        email: Parsed email dict from ``fetch_unread_emails()``.

    Returns:
        Enriched dict with ``category``, ``priority``, ``recommended_action``,
        ``confidence``, and ``summary`` fields added.
    """
    raw = classify_with_llm(email)
    classification = _parse_llm_response(raw)
    return {**email, **classification}


def classify_emails(emails: list[dict[str, str]]) -> list[dict[str, str | float]]:
    """
    Classify a batch of parsed emails sequentially.

    Args:
        emails: List of parsed email dicts from ``fetch_unread_emails()``.

    Returns:
        List of enriched email dicts with classification metadata.
    """
    return [classify_email(email) for email in emails]
