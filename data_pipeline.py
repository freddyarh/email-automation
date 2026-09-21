"""
AI Gmail Cleanser Agent — Phase 1: Secure Data Pipeline.

Connects to the Gmail API via OAuth2, fetches unread inbox messages,
and parses each message into a structured dictionary suitable for
downstream AI classification (future phases).

Usage:
    python data_pipeline.py

Prerequisites:
    - credentials.json from Google Cloud Console (OAuth 2.0 Desktop client)
    - Gmail API enabled on the Google Cloud project
"""

from __future__ import annotations

import base64
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from pprint import pprint
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gmail read-only scope is sufficient for fetching and parsing messages.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_PATH = Path("credentials.json")
TOKEN_PATH = Path("token.json")

SNIPPET_MIN_LEN = 200
SNIPPET_MAX_LEN = 300

# Common reply-chain and mobile-client markers stripped before snippet extraction.
_REPLY_DELIMITER = re.compile(
    r"\nOn .+ wrote:\s*\n", re.IGNORECASE | re.DOTALL)
_SIGNATURE_DELIMITER = re.compile(r"\n--\s*\n.*", re.DOTALL)
_MOBILE_FOOTER = re.compile(
    r"\n(Sent from my .+|Get Outlook for .+|Sent from Mail for Windows.+)$",
    re.IGNORECASE | re.DOTALL,
)
_WHITESPACE = re.compile(r"\s+")


class _HTMLStripper(HTMLParser):
    """Minimal HTML-to-text converter using the standard library."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def authenticate_gmail() -> Any:
    """
    Authenticate with Gmail using OAuth2 and return an authorized API client.

    Loads a saved token from ``token.json`` when available and refreshes it
    if expired. Otherwise, launches a local browser flow using
    ``credentials.json`` and persists the new token for future runs.

    Returns:
        Authorized Gmail API service resource.

    Raises:
        FileNotFoundError: If ``credentials.json`` is missing.
    """
    creds: Credentials | None = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_PATH}. Download OAuth credentials "
                    "from Google Cloud Console and place them in this directory."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)

        TOKEN_PATH.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def fetch_unread_emails(max_results: int = 10) -> list[dict[str, str]]:
    """
    Retrieve the most recent unread messages from the inbox.

    Args:
        max_results: Maximum number of unread messages to fetch (default: 10).

    Returns:
        A list of dictionaries, each containing ``id``, ``sender``,
        ``subject``, and ``snippet`` keys.

    Raises:
        googleapiclient.errors.HttpError: If the Gmail API request fails.
    """
    service = authenticate_gmail()

    list_response = (
        service.users()
        .messages()
        .list(userId="me", q="is:unread in:inbox", maxResults=max_results)
        .execute()
    )

    messages = list_response.get("messages", [])
    if not messages:
        return []

    parsed_emails: list[dict[str, str]] = []
    for item in messages:
        raw_message = (
            service.users()
            .messages()
            .get(userId="me", id=item["id"], format="full")
            .execute()
        )
        parsed_emails.append(parse_email(raw_message))

    return parsed_emails


def parse_email(message: dict[str, Any]) -> dict[str, str]:
    """
    Parse a Gmail API message resource into a clean dictionary.

    Args:
        message: Full Gmail message object returned by ``messages().get()``.

    Returns:
        Dictionary with keys: ``id``, ``sender``, ``subject``, ``snippet``.
    """
    headers = message.get("payload", {}).get("headers", [])
    body_text = _extract_body_text(message.get("payload", {}))

    return {
        "id": message["id"],
        "sender": _get_header(headers, "From"),
        "subject": _get_header(headers, "Subject"),
        "date": _get_header(headers, "Date"),
        "snippet": _build_snippet(body_text or message.get("snippet", "")),
    }


def _get_header(headers: list[dict[str, str]], name: str) -> str:
    """Return the value of a case-insensitive email header."""
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "").strip()
    return ""


def _extract_body_text(payload: dict[str, Any]) -> str:
    """
    Recursively extract plain text from a Gmail message payload.

    Prefers ``text/plain`` parts; falls back to ``text/html`` when needed.
    """
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if body_data:
        decoded = _decode_body(body_data)
        if mime_type == "text/html":
            return _strip_html(decoded)
        return decoded

    parts = payload.get("parts", [])
    plain_text = ""
    html_text = ""

    for part in parts:
        part_mime = part.get("mimeType", "")
        if part_mime.startswith("multipart/"):
            nested = _extract_body_text(part)
            if nested:
                return nested
            continue

        part_body = part.get("body", {}).get("data")
        if not part_body:
            continue

        decoded = _decode_body(part_body)
        if part_mime == "text/plain" and not plain_text:
            plain_text = decoded
        elif part_mime == "text/html" and not html_text:
            html_text = _strip_html(decoded)

    return plain_text or html_text


def _decode_body(data: str) -> str:
    """Decode a Gmail API base64url-encoded message body."""
    padding = "=" * (-len(data) % 4)
    raw_bytes = base64.urlsafe_b64decode(data + padding)
    return raw_bytes.decode("utf-8", errors="replace")


def _strip_html(html: str) -> str:
    """Remove HTML tags and unescape entities."""
    stripper = _HTMLStripper()
    stripper.feed(unescape(html))
    return stripper.get_text()


def _clean_text(text: str) -> str:
    """
    Normalize email body text by removing HTML remnants, signatures,
    reply chains, and excess whitespace.
    """
    cleaned = _strip_html(text) if "<" in text and ">" in text else text
    cleaned = _REPLY_DELIMITER.split(cleaned, maxsplit=1)[0]
    cleaned = _SIGNATURE_DELIMITER.sub("", cleaned)
    cleaned = _MOBILE_FOOTER.sub("", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned


def _build_snippet(
    text: str,
    min_len: int = SNIPPET_MIN_LEN,
    max_len: int = SNIPPET_MAX_LEN,
) -> str:
    """
    Build a concise snippet between ``min_len`` and ``max_len`` characters.

    When the cleaned body is shorter than ``min_len``, the full text is
    returned. Otherwise, the snippet is trimmed at a word boundary near
    ``max_len`` when possible.
    """
    cleaned = _clean_text(text)
    if len(cleaned) <= min_len:
        return cleaned

    if len(cleaned) <= max_len:
        return cleaned

    candidate = cleaned[:max_len]
    if " " in candidate:
        candidate = candidate.rsplit(" ", 1)[0]

    if len(candidate) < min_len:
        return cleaned[:max_len].rstrip()

    return candidate.rstrip()


def main() -> None:
    """Run the pipeline and print parsed unread emails."""
    try:
        emails = fetch_unread_emails(max_results=10)
    except FileNotFoundError as exc:
        print(f"Setup error: {exc}")
        return
    except HttpError as exc:
        print(f"Gmail API error: {exc}")
        return

    if not emails:
        print("No unread emails found in the inbox.")
        return

    print(f"\nFetched {len(emails)} unread email(s):\n")
    for index, email in enumerate(emails, start=1):
        print(f"--- Email {index} ---")
        pprint(email, sort_dicts=False)
        print()


if __name__ == "__main__":
    main()
