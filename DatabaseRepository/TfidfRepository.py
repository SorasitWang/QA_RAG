"""
TfidfRepository implements sparse keyword search backed by SQLite.
Stores term frequencies per chunk and document frequencies per term;
computes TF-IDF relevance scores at query time using smooth IDF.

Chunk embeddings are also stored (as raw bytes) so that hybrid-mode callers
can run MMR re-ranking across both vector and sparse results without an extra
round-trip to ChromaDB.
"""

import math
import os
import re
import sqlite3

import numpy as np

from Dto.RetrievalResult import RetrievalResult
from Dto.VectorDocument import VectorDocument


_STOP_WORDS: frozenset[str] = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'to', 'of', 'in', 'on', 'at', 'by',
    'for', 'with', 'about', 'as', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over',
    'under', 'again', 'then', 'once', 'and', 'but', 'or', 'nor', 'so',
    'yet', 'both', 'either', 'neither', 'not', 'only', 'own', 'same',
    'than', 'too', 'very', 'just', 'that', 'this', 'these', 'those',
    'it', 'its', 'he', 'she', 'they', 'we', 'i', 'you', 'my', 'your',
    'his', 'her', 'their', 'our', 'me', 'him', 'us', 'them', 'if', 'all',
})


class TfidfRepository:
    """SQLite-backed TF-IDF index for sparse keyword search."""

    def __init__(self, dbPath: str = "./tfidf_db/tfidf.sqlite3"):
        self.dbPath = dbPath
        os.makedirs(os.path.dirname(dbPath), exist_ok=True)
        self._initDb()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.dbPath)
        conn.row_factory = sqlite3.Row
        return conn

    def _initDb(self) -> None:
        """Create schema and seed corpus stats row on first run."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id        TEXT    PRIMARY KEY,
                    title     TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    docLength INTEGER NOT NULL,
                    embedding BLOB    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS termFrequencies (
                    docId TEXT NOT NULL,
                    term  TEXT NOT NULL,
                    tf    REAL NOT NULL,
                    PRIMARY KEY (docId, term),
                    FOREIGN KEY (docId) REFERENCES documents(id)
                );

                CREATE TABLE IF NOT EXISTS documentFrequencies (
                    term TEXT    PRIMARY KEY,
                    df   INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS corpusStats (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tf_term ON termFrequencies(term);
                CREATE INDEX IF NOT EXISTS idx_docs_title ON documents(title);
            """)
            if conn.execute(
                "SELECT value FROM corpusStats WHERE key = 'totalDocs'"
            ).fetchone() is None:
                conn.execute(
                    "INSERT INTO corpusStats (key, value) VALUES ('totalDocs', '0')"
                )

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, extract alpha tokens, strip stop words and single-char tokens."""
        tokens = re.findall(r'[a-z]+', text.lower())
        return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    def _computeTf(self, tokens: list[str]) -> dict[str, float]:
        """Raw term frequency: occurrences / total tokens in the document."""
        if not tokens:
            return {}
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        total = len(tokens)
        return {term: count / total for term, count in counts.items()}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def insert(self, documents: list[VectorDocument]) -> None:
        """
        Index a batch of chunks into the TF-IDF store.

        For each chunk:
        - Tokenize content and compute per-document TF values.
        - Increment global DF counts for every unique term.
        - Persist the embedding bytes so hybrid callers can run MMR.

        The total document count in corpusStats is updated atomically with
        the batch to keep IDF calculations consistent.
        """
        with self._connect() as conn:
            totalDocs = int(
                conn.execute(
                    "SELECT value FROM corpusStats WHERE key = 'totalDocs'"
                ).fetchone()["value"]
            )

            for doc in documents:
                tokens = self._tokenize(doc.document)
                tf = self._computeTf(tokens)
                uniqueTerms = set(tokens)
                embeddingBytes = np.array(doc.embeddings, dtype=np.float32).tobytes()

                conn.execute(
                    """INSERT OR REPLACE INTO documents
                       (id, title, content, docLength, embedding)
                       VALUES (?, ?, ?, ?, ?)""",
                    (doc.id, doc.metadata.get("title", ""), doc.document, len(tokens), embeddingBytes),
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO termFrequencies (docId, term, tf) VALUES (?, ?, ?)",
                    [(doc.id, term, tfVal) for term, tfVal in tf.items()],
                )
                for term in uniqueTerms:
                    conn.execute(
                        """INSERT INTO documentFrequencies (term, df) VALUES (?, 1)
                           ON CONFLICT(term) DO UPDATE SET df = df + 1""",
                        (term,),
                    )
                totalDocs += 1

            conn.execute(
                "UPDATE corpusStats SET value = ? WHERE key = 'totalDocs'",
                (str(totalDocs),),
            )

    def existsByTitle(self, title: str) -> bool:
        """Return True if any chunk from this source title is already indexed."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT id FROM documents WHERE title = ? LIMIT 1", (title,)
            ).fetchone() is not None

    def search(self, query: str, topK: int = 15) -> RetrievalResult:
        """
        Score all indexed chunks against *query* using TF-IDF.

        Smooth IDF formula: log((N+1) / (df+1)) + 1  — avoids zero for unseen terms.

        distances = 1 - normalised_score  (lower = more relevant, mirrors ChromaDB L2 convention)
        embeddings are stored per-chunk so hybrid-mode callers can run MMR without a
        round-trip to ChromaDB.
        """
        empty = RetrievalResult(ids=[], documents=[], distances=[], metadatas=[], embeddings=[])

        queryTokens = self._tokenize(query)
        if not queryTokens:
            return empty

        with self._connect() as conn:
            totalDocs = int(
                conn.execute(
                    "SELECT value FROM corpusStats WHERE key = 'totalDocs'"
                ).fetchone()["value"]
            )
            if totalDocs == 0:
                return empty

            placeholders = ",".join("?" for _ in queryTokens)

            dfMap: dict[str, int] = {
                row["term"]: row["df"]
                for row in conn.execute(
                    f"SELECT term, df FROM documentFrequencies WHERE term IN ({placeholders})",
                    queryTokens,
                ).fetchall()
            }

            idfMap: dict[str, float] = {
                term: math.log((totalDocs + 1) / (dfMap.get(term, 0) + 1)) + 1
                for term in queryTokens
            }

            # Accumulate TF-IDF score per document for every matching (doc, term) pair
            scores: dict[str, float] = {}
            for row in conn.execute(
                f"SELECT docId, term, tf FROM termFrequencies WHERE term IN ({placeholders})",
                queryTokens,
            ).fetchall():
                scores[row["docId"]] = scores.get(row["docId"], 0.0) + row["tf"] * idfMap[row["term"]]

            if not scores:
                return empty

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topK]
            topIds:    list[str]   = [docId for docId, _ in ranked]
            topScores: list[float] = [score for _, score in ranked]

            maxScore: float = topScores[0] if topScores[0] > 0 else 1.0
            normalizedScores: list[float] = [s / maxScore for s in topScores]

            idPlaceholders = ",".join("?" for _ in topIds)
            docMap = {
                row["id"]: row
                for row in conn.execute(
                    f"SELECT id, title, content, embedding FROM documents WHERE id IN ({idPlaceholders})",
                    topIds,
                ).fetchall()
            }

            resultIds:        list[str]         = []
            resultDocs:       list[str]         = []
            resultDistances:  list[float]       = []
            resultMetadatas:  list[dict]        = []
            resultEmbeddings: list[list[float]] = []

            for docId, normalizedScore in zip(topIds, normalizedScores):
                if docId not in docMap:
                    continue
                row = docMap[docId]
                resultIds.append(docId)
                resultDocs.append(row["content"])
                resultDistances.append(1.0 - normalizedScore)
                resultMetadatas.append({"title": row["title"]})
                resultEmbeddings.append(np.frombuffer(row["embedding"], dtype=np.float32).tolist())

            return RetrievalResult(
                ids=       resultIds,
                documents= resultDocs,
                distances= resultDistances,
                metadatas= resultMetadatas,
                embeddings=resultEmbeddings,
            )
