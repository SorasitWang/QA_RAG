"""
VectorDocument is the internal transfer object between RagService and
VectorRepository.  It holds everything ChromaDB needs to store one chunk.
"""

from dataclasses import dataclass


@dataclass
class VectorDocument:
    """
    A single chunk ready for insertion into the vector store.

    id         — source title; combined with a short uuid in VectorRepository
                 to form the final ChromaDB document ID.
    document   — the raw chunk text stored alongside the embedding.
    embeddings — pre-computed embedding vector for this chunk.
    metadata   — arbitrary key/value pairs (type, title, tags, …) used for
                 filtered queries in RagService.
    """
    id: str
    document: str
    embeddings: list[float]
    metadata: dict[str, str]
