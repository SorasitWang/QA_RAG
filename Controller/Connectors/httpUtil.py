"""Shared HTTP and date helpers for connectors."""

from datetime import datetime

import requests

from Config import Config


def getJson(url: str, params: dict | None = None, headers: dict | None = None):
    """GET a JSON document with the poller's User-Agent and timeout; raises on HTTP errors."""
    response = requests.get(
        url,
        params=params,
        headers={'User-Agent': Config.Feeds.UserAgent, **(headers or {})},
        timeout=Config.Feeds.RequestTimeoutSeconds,
    )
    response.raise_for_status()
    return response.json()


def parseIsoTimestamp(value: str | None) -> float | None:
    """Convert an ISO-8601 timestamp (with 'Z' or offset) to epoch seconds."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def headlineFromText(text: str, maxChars: int = 80) -> str:
    """First line of a post, shortened at a word boundary, for sources without titles."""
    firstLine = text.strip().splitlines()[0] if text.strip() else ''
    if len(firstLine) <= maxChars:
        return firstLine
    return firstLine[:maxChars].rsplit(' ', 1)[0] + '…'
