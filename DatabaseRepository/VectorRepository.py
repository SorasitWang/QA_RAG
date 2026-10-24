"""
VectorRepository wraps ChromaDB collection operations.
It is the only layer that knows about ChromaDB's API; all callers work through
the RagRepository interface so the storage backend can be swapped later.
"""

from typing import Any
from uuid import uuid4

from Dto import VectorDocument
from Dto.RetrievalResult import RetrievalResult


class RagRepository:
    """Abstract interface for vector storage backends."""

    def search(self, query: str, filters: dict[str, Any] | None, topK: int) -> RetrievalResult:
        ...

    def getByIds(self, ids: list[str]) -> RetrievalResult:
        ...


class VectorRepository(RagRepository):
    """ChromaDB-backed implementation of RagRepository."""

    def __init__(self, collection):
        self.collection = collection

    def insert(self, documents: list[VectorDocument]) -> list[str]:  # type: ignore
        """
        Adds documents to the collection.
        IDs are generated as '<source_title>_<8-char uuid>' so they are
        traceable back to their source while remaining unique across chunks.
        Returns the list of generated IDs so callers can mirror them in other stores.
        """
        idList: list[str] = []
        textList: list[str] = []
        embeddingList: list[list[float]] = []
        metadataList: list[dict[str, str]] = []

        for document in documents:
            runningId = str(uuid4())[:8]
            idList.append(f"{document.id}_{runningId}")
            textList.append(document.document)
            embeddingList.append(document.embeddings)
            metadataList.append(document.metadata)

        self.collection.add(
            ids=idList,
            documents=textList,
            embeddings=embeddingList,
            metadatas=metadataList
        )
        return idList

    def existsByTitle(self, title: str) -> bool:
        """
        Checks whether any chunk from a given source title exists in the collection.
        Uses limit=1 so ChromaDB stops scanning after the first match.
        """
        result = self.collection.get(where={"title": title}, limit=1)
        return len(result["ids"]) > 0

    def backfillLegacyMetadata(self, defaultSourceType: str = "file", batchSize: int = 500) -> int:
        """
        Repair chunks indexed by older versions so filters and title-based replace work:
          - sourceType missing → defaultSourceType
          - title missing      → recovered from the chunk ID ('<title>_<8-char uuid>', see insert)
        Without a title, re-indexing a file could not find (and would duplicate) its old chunks.
        Returns the number of chunks updated.
        """
        updated = 0
        offset = 0
        while True:
            page = self.collection.get(include=["metadatas"], limit=batchSize, offset=offset)
            if not page["ids"]:
                break
            repairs: list[tuple[str, dict]] = []
            for chunkId, meta in zip(page["ids"], page["metadatas"]):
                meta = dict(meta or {})
                if meta.get("sourceType") and meta.get("title"):
                    continue
                meta.setdefault("sourceType", defaultSourceType)
                if not meta.get("title") and "_" in chunkId:
                    meta["title"] = chunkId.rsplit("_", 1)[0]
                repairs.append((chunkId, meta))
            if repairs:
                self.collection.update(ids=[r[0] for r in repairs], metadatas=[r[1] for r in repairs])
                updated += len(repairs)
            offset += batchSize
        return updated

    def deleteByTitle(self, title: str) -> int:
        """Delete every chunk belonging to a source title. Returns the number of chunks removed."""
        ids: list[str] = self.collection.get(where={"title": title}, include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def getByIds(self, ids: list[str]) -> RetrievalResult:
        """
        Fetch documents, embeddings, and metadatas for an explicit list of chunk IDs.
        Used by graph-mode retrieval to hydrate graph-retrieved chunk IDs with full content.
        Distances are set to 0.0 as placeholders; callers replace them with rank-based values.
        """
        if not ids:
            return RetrievalResult(ids=[], documents=[], distances=[], metadatas=[], embeddings=[])
        raw = self.collection.get(
            ids=ids,
            include=["embeddings", "documents", "metadatas"],
        )
        return RetrievalResult(
            ids=       raw["ids"],
            documents= raw["documents"],
            distances= [0.0] * len(raw["ids"]),
            metadatas= raw["metadatas"],
            embeddings=raw["embeddings"],
        )

    def search(self, query: str, filters: dict[str, Any] | None = None, topK: int = 1) -> RetrievalResult:
        """
        Queries the collection and returns a RetrievalResult including embeddings.
        Embeddings are requested explicitly so the caller can run MMR re-ranking.
        nResults is capped at the actual collection size to avoid a ChromaDB error
        when topK exceeds the number of stored documents.
        """
        if not filters:
            filters = None
        nResults = min(topK, self.collection.count())
        if nResults == 0:
            return RetrievalResult(ids=[], documents=[], distances=[], metadatas=[], embeddings=[])

        raw = self.collection.query(
            query_texts=[query],
            n_results=nResults,
            where=filters,
            include=["embeddings", "documents", "distances", "metadatas"]
        )
        return RetrievalResult(
            ids=       raw["ids"][0],
            documents= raw["documents"][0],
            distances= raw["distances"][0],
            metadatas= raw["metadatas"][0],
            embeddings=raw["embeddings"][0],
        )
