"""
Source represents a single piece of raw content before it is chunked and embedded.
It carries enough metadata (title, sourceType, tags) for the retrieval layer to filter
and attribute results back to the original document.
"""

from dataclasses import dataclass
from enum import Enum


class SourceType(str, Enum):
    """Where a source came from. The string value is what every store persists."""
    File = 'file'   # local file in the watched source folder
    News = 'news'
    Blog = 'blog'
    Sns  = 'sns'

    @classmethod
    def values(cls) -> list[str]:
        """All persisted string values."""
        return [member.value for member in cls]


# Legacy free-form values from before SourceType was enforced.
_LegacySourceTypes: dict[str, str] = {'text': SourceType.File.value}


def normaliseSourceType(value: str | SourceType | None) -> str:
    """Return the persisted string for a source type; raises ValueError for unknown types."""
    if value is None:
        return SourceType.File.value
    if isinstance(value, SourceType):
        return value.value
    value = _LegacySourceTypes.get(value, value)
    if value not in SourceType.values():
        raise ValueError(f"unknown sourceType '{value}', expected one of {SourceType.values()}")
    return value


@dataclass
class Source:
    """
    Raw content container used during the ingestion pipeline.

    title      — unique identifier: the file path for local files, the canonical URL
                 for feed items; used to detect and replace already-indexed sources.
    sourceType — one of SourceType's values ('file', 'news', 'blog', 'sns').
    content    — full text.
    tags       — optional labels for metadata filtering at query time.
    optional   — extra scalar metadata (url, publishedAt, author, feedName, platform …)
                 merged into each chunk's stored metadata.
    """
    title: str
    sourceType: str
    content: str
    tags: list[str]
    optional: dict = None

    @classmethod
    def fromDict(cls, data: dict) -> "Source":
        """Build a Source from a JSON payload (e.g. the retrieval service's /ingest body)."""
        return cls(
            title=data["title"],
            # 'type' is accepted for payloads written before the rename.
            sourceType=normaliseSourceType(data.get("sourceType", data.get("type"))),
            content=data["content"],
            tags=list(data.get("tags") or []),
            optional=data.get("optional"),
        )
