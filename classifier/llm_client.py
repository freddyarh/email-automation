"""
Thin OpenAI client wrapper for structured email classification.
"""

from __future__ import annotations

from openai import OpenAI

from classifier.prompts import SYSTEM_PROMPT, build_user_prompt
from config.settings import MODEL_NAME, OPENAI_API_KEY


def get_client() -> OpenAI:
    """
    Return an authenticated OpenAI client.

    Raises:
        ValueError: If ``OPENAI_API_KEY`` is not set in the environment.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def classify_with_llm(email: dict[str, str]) -> str:
    """
    Send a parsed email to the LLM and return the raw JSON response text.

    Args:
        email: Parsed email dict from ``data_pipeline.parse_email()``.

    Returns:
        Raw JSON string from the model's response content field.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(email)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "{}"
