"""
Application settings loaded from environment variables.

Copy ``.env.example`` to ``.env`` and fill in your API key before running
``agent.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_EMAILS: int = int(os.getenv("MAX_EMAILS", "10"))

CATEGORIES: tuple[str, ...] = (
    "important",
    "newsletter",
    "promotional",
    "social",
    "notification",
    "spam",
    "other",
)
PRIORITIES: tuple[str, ...] = ("high", "medium", "low")
ACTIONS: tuple[str, ...] = ("keep", "archive", "delete", "review")
