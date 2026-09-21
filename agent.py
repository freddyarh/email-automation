"""
AI Gmail Cleanser Agent — Phase 2: Fetch, classify, and review.

Fetches unread inbox emails via the Phase 1 pipeline, classifies each
with an LLM, and prints enriched results to the terminal.

Usage:
    python agent.py

Prerequisites:
    - Phase 1 setup (credentials.json, token.json)
    - .env file with OPENAI_API_KEY (see .env.example)
"""

from __future__ import annotations

from pprint import pprint

from classifier import classify_emails
from config.settings import MAX_EMAILS
from data_pipeline import fetch_unread_emails
from googleapiclient.errors import HttpError


def main() -> None:
    """Fetch unread emails, classify them, and print results."""
    try:
        emails = fetch_unread_emails(max_results=MAX_EMAILS)
    except FileNotFoundError as exc:
        print(f"Setup error: {exc}")
        return
    except HttpError as exc:
        print(f"Gmail API error: {exc}")
        return

    if not emails:
        print("No unread emails found in the inbox.")
        return

    print(f"\nFetched {len(emails)} unread email(s). Classifying...\n")

    try:
        classified = classify_emails(emails)
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return

    for index, email in enumerate(classified, start=1):
        print(f"--- Email {index} ---")
        pprint(email, sort_dicts=False)
        print()


if __name__ == "__main__":
    main()
