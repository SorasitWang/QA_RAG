"""
GraphRepository stores entity co-occurrence relationships in SQLite and provides
graph-based chunk retrieval.

At ingest time, spaCy NER extracts named entities from each chunk.  Entities become
nodes and pairs of entities that appear in the same chunk become weighted edges
(co-occurrence).  Every node and edge links back to the shared chunk ID space used
by VectorRepository and TfidfRepository.

At query time, entities are extracted from the natural-language query, matched
against stored entities, and the graph is traversed via BFS up to a configurable
hop depth.  Chunk IDs reachable through the traversal are scored and returned in
ranked order for the caller (RagService) to hydrate with full text.
"""

import os
import sqlite3

import networkx as nx
import spacy

from Config import Config
from Dto.VectorDocument import VectorDocument

_nlp = None


def _getNlp():
    """Lazy-load spaCy model; raises RuntimeError with install instructions if missing."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    entityType  TEXT    NOT NULL,
    chunkCount  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (name, entityType)
);
CREATE TABLE IF NOT EXISTS entityChunks (
    entityId    INTEGER NOT NULL,
    chunkId     TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    PRIMARY KEY (entityId, chunkId),
    FOREIGN KEY (entityId) REFERENCES entities(id)
);
CREATE TABLE IF NOT EXISTS coOccurrences (
    entityIdA   INTEGER NOT NULL,
    entityIdB   INTEGER NOT NULL,
    weight      INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (entityIdA, entityIdB),
    FOREIGN KEY (entityIdA) REFERENCES entities(id),
    FOREIGN KEY (entityIdB) REFERENCES entities(id)
);
CREATE INDEX IF NOT EXISTS idx_entityChunks_entityId ON entityChunks(entityId);
CREATE INDEX IF NOT EXISTS idx_entityChunks_chunkId  ON entityChunks(chunkId);
CREATE INDEX IF NOT EXISTS idx_entities_name         ON entities(name);
"""


class GraphRepository:
    """SQLite-backed entity co-occurrence graph for graph-mode retrieval."""

    def __init__(self, dbPath: str = "./graph_db/graph.sqlite3"):
        """
        Initialises the repository and creates the schema on first run.
        The parent directory is created automatically if it does not exist.
        """
        self.dbPath = dbPath
        dirName = os.path.dirname(dbPath)
        if dirName:
            os.makedirs(dirName, exist_ok=True)
        self._initDb()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with row_factory set for dict-style access."""
        conn = sqlite3.connect(self.dbPath)
        conn.row_factory = sqlite3.Row
        # WAL lets search() read while the ingestion worker writes.
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initDb(self) -> None:
        """Create tables and indexes; idempotent via IF NOT EXISTS."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _extractEntities(self, text: str) -> list[tuple[str, str]]:
        """
        Run spaCy NER on text and return (name, entityType) pairs for kept label types.
        Single-character spans are dropped.
        """
        doc = _getNlp()(text)
        return [
            (ent.text.strip(), ent.label_)
            for ent in doc.ents
            if ent.label_ in Config.Graph.GraphKeptEntityTypes and len(ent.text.strip()) > 1
        ]

    def _upsertEntity(self, conn: sqlite3.Connection, name: str, entityType: str) -> int:
        """Insert entity if absent and return its integer ID. chunkCount is managed by insert()."""
        conn.execute(
            "INSERT OR IGNORE INTO entities (name, entityType, chunkCount) VALUES (?, ?, 0)",
            (name, entityType),
        )
        return conn.execute(
            "SELECT id FROM entities WHERE name = ? AND entityType = ?", (name, entityType)
        ).fetchone()["id"]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def insert(self, documents: list[VectorDocument]) -> None:
        """
        Extract entities from each chunk and persist nodes, chunk links, and
        co-occurrence edges.

        For each chunk:
          1. Run NER to obtain (name, entityType) pairs.
          2. Upsert each entity into entities; collect their integer IDs.
          3. Insert (entityId, chunkId, title) rows into entityChunks.
          4. For every unordered unique pair of entity IDs within the chunk,
             upsert a coOccurrences row (entityIdA < entityIdB) and increment weight.

        Chunks that yield no kept entities are silently skipped.
        """
        with self._connect() as conn:
            for doc in documents:
                entityPairs = self._extractEntities(doc.document)
                if not entityPairs:
                    continue

                title = doc.metadata.get("title", "")
                entityIds: list[int] = []

                for name, entityType in entityPairs:
                    entityId = self._upsertEntity(conn, name, entityType)
                    entityIds.append(entityId)
                    conn.execute(
                        "INSERT OR IGNORE INTO entityChunks (entityId, chunkId, title) VALUES (?, ?, ?)",
                        (entityId, doc.id, title),
                    )
                    # Increment chunkCount only when this (entity, chunk) pair is new,
                    # so repeated entity mentions within the same chunk or re-indexing
                    # do not inflate the relevance score.
                    if conn.execute("SELECT changes()").fetchone()[0] > 0:
                        conn.execute(
                            "UPDATE entities SET chunkCount = chunkCount + 1 WHERE id = ?",
                            (entityId,),
                        )

                uniqueIds = list(set(entityIds))
                for i in range(len(uniqueIds)):
                    for j in range(i + 1, len(uniqueIds)):
                        a = min(uniqueIds[i], uniqueIds[j])
                        b = max(uniqueIds[i], uniqueIds[j])
                        conn.execute(
                            """INSERT INTO coOccurrences (entityIdA, entityIdB, weight) VALUES (?, ?, 1)
                               ON CONFLICT(entityIdA, entityIdB) DO UPDATE SET weight = weight + 1""",
                            (a, b),
                        )

    def existsByTitle(self, title: str) -> bool:
        """Return True if any entity chunk from this source title is already indexed."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT entityId FROM entityChunks WHERE title = ? LIMIT 1", (title,)
            ).fetchone() is not None

    def deleteByTitle(self, title: str) -> int:
        """
        Remove every entity-chunk link for a source title, then repair the graph:
          - decrement entities.chunkCount and drop entities no longer in any chunk;
          - rebuild coOccurrences from entityChunks, because edge weights are
            aggregated without per-chunk provenance and cannot be decremented directly.
        Returns the number of distinct chunks removed.
        """
        with self._connect() as conn:
            chunkCount: int = conn.execute(
                "SELECT COUNT(DISTINCT chunkId) FROM entityChunks WHERE title = ?", (title,)
            ).fetchone()[0]
            if chunkCount == 0:
                return 0

            entityCounts = conn.execute(
                "SELECT entityId, COUNT(*) AS n FROM entityChunks WHERE title = ? GROUP BY entityId",
                (title,),
            ).fetchall()
            conn.executemany(
                "UPDATE entities SET chunkCount = chunkCount - ? WHERE id = ?",
                [(row["n"], row["entityId"]) for row in entityCounts],
            )
            conn.execute("DELETE FROM entityChunks WHERE title = ?", (title,))
            conn.execute("DELETE FROM entities WHERE chunkCount <= 0")

            # insert() adds 1 to an edge per chunk in which the pair co-occurs, so the
            # weight equals the number of chunks shared by both entities.
            conn.execute("DELETE FROM coOccurrences")
            conn.execute(
                """INSERT INTO coOccurrences (entityIdA, entityIdB, weight)
                   SELECT a.entityId, b.entityId, COUNT(DISTINCT a.chunkId)
                   FROM entityChunks a
                   JOIN entityChunks b ON a.chunkId = b.chunkId AND a.entityId < b.entityId
                   GROUP BY a.entityId, b.entityId"""
            )
            return chunkCount

    def search(self, query: str, topK: int = 10, hopDepth: int = 1) -> list[str]:
        """
        Translate a natural-language query into a ranked list of chunk IDs.

        Algorithm:
          1. Extract entities from the query using the same NER pipeline as ingest.
          2. Match query entity names against stored entities (case-insensitive substring).
          3. Load all co-occurrence edges into a NetworkX graph in memory; BFS from seed
             entity IDs for hopDepth hops to obtain a reachable entity set.
          4. Score each chunk ID by summing chunkCount of all reachable entities that
             appear in that chunk.
          5. Return up to topK chunk IDs sorted by descending score.

        Returns an empty list when no query entities match anything in the graph.
        """
        queryEntities = self._extractEntities(query)
        if not queryEntities:
            return []

        queryNames = [name.lower() for name, _ in queryEntities]

        with self._connect() as conn:
            allEntityRows = conn.execute(
                "SELECT id, name, chunkCount FROM entities"
            ).fetchall()

            seedIds: list[int] = [
                row["id"] for row in allEntityRows
                if any(qn in row["name"].lower() for qn in queryNames)
            ]
            if not seedIds:
                return []

            allEdges = conn.execute(
                "SELECT entityIdA, entityIdB, weight FROM coOccurrences"
            ).fetchall()

            G: nx.Graph = nx.Graph()
            for edge in allEdges:
                G.add_edge(edge["entityIdA"], edge["entityIdB"], weight=edge["weight"])

            reachable: set[int] = set(seedIds)
            frontier: set[int] = set(seedIds)
            for _ in range(hopDepth):
                nextFrontier: set[int] = set()
                for node in frontier:
                    if node in G:
                        nextFrontier.update(G.neighbors(node))
                nextFrontier -= reachable
                reachable.update(nextFrontier)
                frontier = nextFrontier

            if not reachable:
                return []

            placeholders = ",".join("?" for _ in reachable)
            entityRows = conn.execute(
                f"SELECT id, chunkCount FROM entities WHERE id IN ({placeholders})",
                list(reachable),
            ).fetchall()
            chunkCountById: dict[int, int] = {row["id"]: row["chunkCount"] for row in entityRows}

            chunkRows = conn.execute(
                f"SELECT entityId, chunkId FROM entityChunks WHERE entityId IN ({placeholders})",
                list(reachable),
            ).fetchall()

            chunkScores: dict[str, float] = {}
            for row in chunkRows:
                score = chunkCountById.get(row["entityId"], 1)
                chunkScores[row["chunkId"]] = chunkScores.get(row["chunkId"], 0.0) + score

        ranked = sorted(chunkScores.items(), key=lambda x: x[1], reverse=True)
        return [chunkId for chunkId, _ in ranked[:topK]]
