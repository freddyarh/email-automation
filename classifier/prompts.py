"""
Prompt templates for LLM-based email classification.
"""

from __future__ import annotations

from config.settings import ACTIONS, CATEGORIES, PRIORITIES

SYSTEM_PROMPT = f"""You are an email triage assistant. Classify each email into exactly one category, assign a priority, recommend an action, and write a one-sentence summary.

Categories (pick one): {", ".join(CATEGORIES)}
Priorities (pick one): {", ".join(PRIORITIES)}
Recommended actions (pick one): {", ".join(ACTIONS)}

Guidelines:
- important: personal messages, deadlines, replies needed, work-critical mail
- newsletter: subscriptions, digests, regular content updates
- promotional: sales, discounts, marketing campaigns
- social: LinkedIn, GitHub, Twitter/X, community notifications
- notification: receipts, confirmations, shipping updates, system alerts
- spam: suspicious, irrelevant, or unwanted bulk mail
- other: anything that does not fit the categories above

Respond with valid JSON only, using this exact schema:
{{
  "category": "<one category>",
  "priority": "<one priority>",
  "recommended_action": "<one action>",
  "confidence": <float between 0 and 1>,
  "summary": "<one concise sentence>"
}}"""


def build_user_prompt(email: dict[str, str]) -> str:
    """
    Build the user message sent to the LLM for a single parsed email.

    Args:
        email: Parsed email dict with ``sender``, ``subject``, and ``snippet``.

    Returns:
        Formatted prompt string for the chat completion API.
    """
    return (
        f"From: {email.get('sender', 'Unknown')}\n"
        f"Subject: {email.get('subject', '(no subject)')}\n"
        f"Body preview:\n{email.get('snippet', '')}"
    )
