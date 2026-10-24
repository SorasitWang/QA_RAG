"""
RetrievalClient is the web service's HTTP adapter to the retrieval service.
It mirrors RagService.query so callers only swap the object they hold.

Failures (connection refused, timeout, 5xx) degrade to an empty RetrievalResult,
so chat keeps working without RAG context while the retrieval service is down.
"""

import logging

import requests

from Config import Config
from Dto.RetrievalResult import RetrievalResult
from Dto.Source import Source

logger = logging.getLogger(__name__)


class RetrievalClient:
    """Calls the retrieval service's /query and /ingest endpoints."""

    def __init__(
        self,
        baseUrl: str = Config.Services.RetrievalServiceUrl,
        timeoutSeconds: float = Config.Services.RetrievalTimeoutSeconds,
    ):
        self.baseUrl: str = baseUrl.rstrip("/")
        self.timeoutSeconds: float = timeoutSeconds
        self.session = requests.Session()

    def query(
        self,
        query: str,
        tags: list[str] | None = None,
        topK: int | None = None,
        searchMode: str = "vector",
        sourceTypes: list[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve the most relevant chunks; returns an empty result if the service is unreachable."""
        body: dict = {"query": query, "mode": searchMode}
        if topK is not None:
            body["topK"] = topK
        if tags:
            body["tags"] = tags
        if sourceTypes:
            body["sourceTypes"] = sourceTypes
        try:
            response = self.session.post(f"{self.baseUrl}/query", json=body, timeout=self.timeoutSeconds)
            response.raise_for_status()
            return RetrievalResult.fromDict(response.json())
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Retrieval service query failed, continuing without context: %s", exc)
            return RetrievalResult.empty()

    def ingest(self, sources: list[Source]) -> int:
        """Queue sources for indexing. Returns how many were accepted; raises on failure."""
        payload = {"sources": [
            {"title": s.title, "sourceType": s.sourceType, "content": s.content, "tags": s.tags, "optional": s.optional}
            for s in sources
        ]}
        response = self.session.post(f"{self.baseUrl}/ingest", json=payload, timeout=self.timeoutSeconds)
        response.raise_for_status()
        return response.json()["queued"]

    def deleteSources(self, titles: list[str]) -> int:
        """Queue removal of sources by title (e.g. expired feed items). Returns how many were queued."""
        response = self.session.post(
            f"{self.baseUrl}/ingest/delete", json={"titles": titles}, timeout=self.timeoutSeconds
        )
        response.raise_for_status()
        return response.json()["queued"]

    def rescan(self) -> None:
        """Ask the retrieval service to reconcile its indexes with the source folder."""
        response = self.session.post(f"{self.baseUrl}/ingest/rescan", timeout=self.timeoutSeconds)
        response.raise_for_status()

    def status(self) -> dict:
        """Return the ingestion worker status."""
        response = self.session.get(f"{self.baseUrl}/ingest/status", timeout=self.timeoutSeconds)
        response.raise_for_status()
        return response.json()
