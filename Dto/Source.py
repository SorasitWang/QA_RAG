"""
Source represents a single piece of raw content before it is chunked and embedded.
It carries enough metadata (title, type, tags) for the retrieval layer to filter
and attribute results back to the original document.
"""

from dataclasses import dataclass
from enum import Enum


class SourceType(Enum):
    TextFile = 'TextFile'
    WebPage  = 'WebPage'


@dataclass
class Source:
    """
    Raw content container used during the ingestion pipeline.

    title    — unique identifier (typically the file path); used by RagService
               to detect already-indexed sources.
    type     — free-form string describing the content origin (e.g. 'text', 'html').
    content  — full text; gets overwritten per-chunk inside updateData's loop.
    tags     — optional labels for metadata filtering at query time.
    optional — arbitrary extra metadata merged into the ChromaDB document metadata.
    """
    title: str
    type: str
    content: str
    tags: list[str]
    optional: dict = None
