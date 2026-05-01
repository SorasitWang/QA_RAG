"""
RagService implements a multi-mode retrieval pipeline:

  'vector'  — semantic similarity only (original behaviour)
              Stage 1: wide ChromaDB candidate pool (RetrievalTopK)
              Stage 2: L2-distance threshold filter → MMR re-rank

  'sparse'  — TF-IDF keyword search only
              Stage 1: TF-IDF candidate pool (TfidfTopK)
              Stage 2: distance threshold filter → MMR re-rank

  'hybrid'  — Reciprocal Rank Fusion of vector + TF-IDF, then MMR
              Stage 1: independent vector and TF-IDF searches
              Stage 2: RRF score merges both ranked lists
              Stage 3: MMR re-rank on the merged candidates

MMR (Maximal Marginal Relevance) prevents feeding the LLM redundant chunks
about the same sub-topic. Lambda balances relevance vs. diversity; lower λ → more diverse.
"""

import numpy as np

from Config import Config
from DatabaseRepository.TfidfRepository import TfidfRepository
from DatabaseRepository.VectorRepository import RagRepository
from DatabaseService.BaseService import BaseService
from Dto.RetrievalResult import RetrievalResult
from Dto.Source import SourceType
from Dto.VectorDocument import VectorDocument


def _emptyResult() -> RetrievalResult:
    return RetrievalResult(ids=[], documents=[], distances=[], metadatas=[], embeddings=[])


def _mmrSelect(
    distances: list[float],
    embeddings: list[list[float]],
    k: int,
    lambda_: float,
) -> list[int]:
    """
    Greedily select k indices from candidates using Maximal Marginal Relevance.

    Each iteration picks the candidate that maximises:
        score = lambda * relevance_to_query  -  (1 - lambda) * max_sim_to_selected

    Embeddings are L2-normalised before computing cosine similarity so the
    redundancy term is scale-independent.
    ChromaDB L2 distances are negated to act as a relevance proxy
    (lower distance = more relevant = higher score).
    """
    docs = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(docs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    docsNorm = docs / norms

    relevance = [-d for d in distances]

    selected: list[int] = []
    remaining = list(range(len(distances)))

    for _ in range(min(k, len(remaining))):
        bestIdx, bestScore = None, float('-inf')
        for idx in remaining:
            if not selected:
                redundancy = 0.0
            else:
                sims = docsNorm[idx] @ docsNorm[selected].T
                redundancy = float(np.max(sims))
            score = lambda_ * relevance[idx] - (1 - lambda_) * redundancy
            if score > bestScore:
                bestScore, bestIdx = score, idx
        selected.append(bestIdx)
        remaining.remove(bestIdx)

    return selected


class RagService(BaseService):
    """Provides high-level retrieval operations on top of the vector and sparse repositories."""

    def __init__(self, repository: RagRepository, tfidfRepository: TfidfRepository | None = None):
        self.repository: RagRepository = repository
        self.tfidfRepository: TfidfRepository | None = tfidfRepository

    def addSources(self, sources: list, embeddings: list[float]):
        """
        Converts Source objects to VectorDocuments and persists them in the vector store.
        The actual ChromaDB IDs returned by VectorRepository.insert are then mirrored into
        the TF-IDF store so both indexes share the same chunk IDs — a prerequisite for RRF
        in hybrid mode.
        """
        documents: list[VectorDocument] = []
        for source in sources:
            metadata: dict = {'type': source.type, 'title': source.title}
            if source.tags and len(source.tags) > 0:
                metadata['tags'] = source.tags
            if source.optional is not None:
                assert isinstance(source.optional, dict), "optional field must be a dictionary"
                metadata.update(source.optional)

            documents.append(VectorDocument(source.title, source.content, embeddings, metadata))

        actualIds = self.repository.insert(documents)

        if self.tfidfRepository is not None:
            # Re-create VectorDocument list with the actual ChromaDB IDs so TF-IDF and
            # vector stores share identical chunk identifiers for later RRF merging.
            tfidfDocs = [
                VectorDocument(actualId, doc.document, doc.embeddings, doc.metadata)
                for actualId, doc in zip(actualIds, documents)
            ]
            self.tfidfRepository.insert(tfidfDocs)

    def sourceExists(self, title: str) -> bool:
        """Return True if any chunk from this source title is already indexed."""
        return self.repository.existsByTitle(title)

    # ------------------------------------------------------------------
    # Public query entry point
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        tags: list[str] | None = None,
        topK: int | None = None,
        sourceType: SourceType | None = None,
        searchMode: str = "vector",
    ) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for *query*.

        searchMode options:
          'vector'  — semantic similarity via ChromaDB (default)
          'sparse'  — TF-IDF keyword search via SQLite
          'hybrid'  — Reciprocal Rank Fusion of vector + TF-IDF, then MMR

        *tags* and *sourceType* are applied as metadata filters for vector search.
        TF-IDF search does not support metadata filters (searches full corpus).

        Falls back to 'vector' mode when tfidfRepository is not configured.
        """
        if topK is None:
            topK = Config.Model.MMRTopK

        if searchMode == "sparse" and self.tfidfRepository is not None:
            return self._querySparse(query, topK)

        if searchMode == "hybrid" and self.tfidfRepository is not None:
            return self._queryHybrid(query, tags, topK, sourceType)

        return self._queryVector(query, tags, topK, sourceType)

    # ------------------------------------------------------------------
    # Internal retrieval implementations
    # ------------------------------------------------------------------

    def _buildFilters(self, tags: list[str] | None, sourceType: SourceType | None) -> dict:
        filters: dict = {}
        if tags:
            filters['tags'] = tags
        if sourceType is not None:
            filters['type'] = sourceType
        return filters

    def _queryVector(
        self,
        query: str,
        tags: list[str] | None,
        topK: int,
        sourceType: SourceType | None,
    ) -> RetrievalResult:
        """
        Original two-stage pipeline: wide ChromaDB search → threshold filter → MMR.
        """
        filters = self._buildFilters(tags, sourceType)
        raw: RetrievalResult = self.repository.search(query, filters, Config.Model.RetrievalTopK)

        if raw.isEmpty():
            return _emptyResult()

        valid: list[int] = [i for i, d in enumerate(raw.distances) if d <= Config.Model.SimilarityThreshold]
        if not valid:
            return _emptyResult()

        candidateIds:  list[str]         = [raw.ids[i]        for i in valid]
        candidateDocs: list[str]         = [raw.documents[i]  for i in valid]
        candidateDist: list[float]       = [raw.distances[i]  for i in valid]
        candidateMeta: list[dict]        = [raw.metadatas[i]  for i in valid]
        candidateEmb:  list[list[float]] = [raw.embeddings[i] for i in valid]

        selected: list[int] = _mmrSelect(candidateDist, candidateEmb, k=topK, lambda_=Config.Model.MMRLambda)

        return RetrievalResult(
            ids=       [candidateIds[i]  for i in selected],
            documents= [candidateDocs[i] for i in selected],
            distances= [candidateDist[i] for i in selected],
            metadatas= [candidateMeta[i] for i in selected],
            embeddings=[candidateEmb[i]  for i in selected],
        )

    def _querySparse(self, query: str, topK: int) -> RetrievalResult:
        """
        TF-IDF-only retrieval: threshold filter → MMR re-rank.
        Embeddings stored in SQLite enable MMR without an extra ChromaDB round-trip.
        """
        raw: RetrievalResult = self.tfidfRepository.search(query, Config.Model.TfidfTopK)

        if raw.isEmpty():
            return _emptyResult()

        valid: list[int] = [i for i, d in enumerate(raw.distances) if d <= Config.Model.TfidfSimilarityThreshold]
        if not valid:
            return _emptyResult()

        candidateIds:  list[str]         = [raw.ids[i]        for i in valid]
        candidateDocs: list[str]         = [raw.documents[i]  for i in valid]
        candidateDist: list[float]       = [raw.distances[i]  for i in valid]
        candidateMeta: list[dict]        = [raw.metadatas[i]  for i in valid]
        candidateEmb:  list[list[float]] = [raw.embeddings[i] for i in valid]

        selected: list[int] = _mmrSelect(candidateDist, candidateEmb, k=topK, lambda_=Config.Model.MMRLambda)

        return RetrievalResult(
            ids=       [candidateIds[i]  for i in selected],
            documents= [candidateDocs[i] for i in selected],
            distances= [candidateDist[i] for i in selected],
            metadatas= [candidateMeta[i] for i in selected],
            embeddings=[candidateEmb[i]  for i in selected],
        )

    def _queryHybrid(
        self,
        query: str,
        tags: list[str] | None,
        topK: int,
        sourceType: SourceType | None,
    ) -> RetrievalResult:
        """
        Hybrid retrieval: independent vector + TF-IDF searches fused with RRF, then MMR.

        Reciprocal Rank Fusion (RRF) formula for each chunk:
            rrf = 1/(k + rank_vector) + 1/(k + rank_tfidf)
        Chunks absent from one ranked list receive a penalty rank of RetrievalTopK + 1.

        The merged candidate set is then re-ranked with MMR. Since both stores hold
        embeddings for every chunk, no additional lookups are needed.
        """
        # --- Stage 1: independent searches ---
        filters = self._buildFilters(tags, sourceType)
        vectorRaw: RetrievalResult = self.repository.search(query, filters, Config.Model.RetrievalTopK)
        tfidfRaw:  RetrievalResult = self.tfidfRepository.search(query, Config.Model.TfidfTopK)

        # --- Stage 2: threshold filter on vector results ---
        vectorIds: list[str] = []
        vectorDataById: dict[str, dict] = {}
        for i, (docId, dist) in enumerate(zip(vectorRaw.ids, vectorRaw.distances)):
            if dist <= Config.Model.SimilarityThreshold:
                vectorIds.append(docId)
                vectorDataById[docId] = {
                    "document":  vectorRaw.documents[i],
                    "distance":  dist,
                    "metadata":  vectorRaw.metadatas[i],
                    "embedding": vectorRaw.embeddings[i],
                }

        tfidfIds: list[str] = list(tfidfRaw.ids)
        tfidfDataById: dict[str, dict] = {
            docId: {
                "document":  tfidfRaw.documents[i],
                "distance":  tfidfRaw.distances[i],
                "metadata":  tfidfRaw.metadatas[i],
                "embedding": tfidfRaw.embeddings[i],
            }
            for i, docId in enumerate(tfidfIds)
        }

        if not vectorDataById and not tfidfDataById:
            return _emptyResult()

        # --- Stage 3: Reciprocal Rank Fusion ---
        # Preserve order (vector IDs first, then TF-IDF-only IDs)
        allIds: list[str] = list(vectorIds)
        for docId in tfidfIds:
            if docId not in vectorDataById:
                allIds.append(docId)

        penaltyRank = Config.Model.RetrievalTopK + 1
        k = Config.Model.RRFk

        rrfScores: dict[str, float] = {}
        for docId in allIds:
            vRank = (vectorIds.index(docId) + 1) if docId in vectorDataById else penaltyRank
            tRank = (tfidfIds.index(docId) + 1) if docId in tfidfDataById else penaltyRank
            rrfScores[docId] = 1.0 / (k + vRank) + 1.0 / (k + tRank)

        # Sort descending by RRF score
        sortedIds = sorted(allIds, key=lambda d: rrfScores[d], reverse=True)

        # --- Stage 4: build merged candidate pool ---
        # Convert RRF scores into "distance-like" values so _mmrSelect treats
        # higher-RRF chunks as more relevant (negate score; MMR negates distances again).
        maxRrf = max(rrfScores.values())

        mergedIds, mergedDocs, mergedMeta, mergedEmb, mergedDist = [], [], [], [], []
        for docId in sortedIds:
            data = vectorDataById.get(docId) or tfidfDataById.get(docId)
            if data is None:
                continue
            mergedIds.append(docId)
            mergedDocs.append(data["document"])
            mergedMeta.append(data["metadata"])
            mergedEmb.append(data["embedding"])
            # distance = maxRrf - score → lower for higher-ranked docs → correct for MMR
            mergedDist.append(maxRrf - rrfScores[docId])

        if not mergedIds:
            return _emptyResult()

        # --- Stage 5: MMR re-ranking ---
        selected: list[int] = _mmrSelect(mergedDist, mergedEmb, k=topK, lambda_=Config.Model.MMRLambda)

        return RetrievalResult(
            ids=       [mergedIds[i]  for i in selected],
            documents= [mergedDocs[i] for i in selected],
            distances= [mergedDist[i] for i in selected],
            metadatas= [mergedMeta[i] for i in selected],
            embeddings=[mergedEmb[i]  for i in selected],
            # Per-retriever distances so the UI can show dense and sparse scores
            # independently. None when a chunk was found by only one retriever.
            vectorDistances=[vectorDataById.get(mergedIds[i], {}).get("distance") for i in selected],
            tfidfDistances= [tfidfDataById.get(mergedIds[i], {}).get("distance") for i in selected],
        )
