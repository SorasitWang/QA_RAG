"""
RetrievalResult is the transfer object returned by RagService.query().
All lists are parallel — index i across every field refers to the same chunk.

vectorDistances and tfidfDistances are populated only in hybrid mode;
they carry the original per-retriever distances before RRF fusion so the UI
can display dense and sparse scores independently. None for a given slot means
that chunk was found by only one of the two retrievers.
"""

from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """Ranked list of retrieved chunks produced by the RAG retrieval pipeline."""

    ids:        list[str]
    documents:  list[str]
    distances:  list[float]
    metadatas:  list[dict]
    embeddings: list[list[float]]

    # Hybrid-mode only; None in vector / sparse modes.
    vectorDistances: list[float | None] | None = field(default=None)
    tfidfDistances:  list[float | None] | None = field(default=None)

    def isEmpty(self) -> bool:
        return len(self.ids) == 0
