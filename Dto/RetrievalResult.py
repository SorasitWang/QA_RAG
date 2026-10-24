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

    def toDict(self, includeEmbeddings: bool = False) -> dict:
        """
        Serialise to a JSON-friendly dict for transport between services.
        Embeddings are omitted by default: MMR already ran inside the retrieval
        service, so callers rarely need them and they dominate payload size.
        """
        data: dict = {
            "ids":             list(self.ids),
            "documents":       list(self.documents),
            "distances":       [float(d) for d in self.distances],
            "metadatas":       list(self.metadatas),
            "vectorDistances": self.vectorDistances,
            "tfidfDistances":  self.tfidfDistances,
        }
        if includeEmbeddings:
            data["embeddings"] = [[float(x) for x in emb] for emb in self.embeddings]
        return data

    @classmethod
    def fromDict(cls, data: dict) -> "RetrievalResult":
        """Inverse of toDict(); missing embeddings become an empty list per chunk."""
        ids: list[str] = data.get("ids", [])
        return cls(
            ids=ids,
            documents=data.get("documents", []),
            distances=data.get("distances", []),
            metadatas=data.get("metadatas", []),
            embeddings=data.get("embeddings") or [[] for _ in ids],
            vectorDistances=data.get("vectorDistances"),
            tfidfDistances=data.get("tfidfDistances"),
        )

    @classmethod
    def empty(cls) -> "RetrievalResult":
        """Return a result with no chunks."""
        return cls(ids=[], documents=[], distances=[], metadatas=[], embeddings=[])
