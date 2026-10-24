"""
IngestManifestRepository records which source files have been indexed and the
content hash they were indexed with. The ingestion service compares a file's
current hash against this manifest to decide whether to skip, re-index, or
remove it, so edits and deletions are detected (not just new titles).
"""

import os
import sqlite3
import time


class IngestManifestRepository:
    """SQLite-backed table of indexed source files keyed by title."""

    def __init__(self, dbPath: str = "./tfidf_db/manifest.sqlite3"):
        self.dbPath = dbPath
        dirName = os.path.dirname(dbPath)
        if dirName:
            os.makedirs(dirName, exist_ok=True)
        self._initDb()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.dbPath)
        conn.row_factory = sqlite3.Row
        return conn

    def _initDb(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestedFiles (
                    title      TEXT PRIMARY KEY,
                    sha256     TEXT NOT NULL,
                    chunkCount INTEGER NOT NULL,
                    ingestedAt REAL NOT NULL
                )
            """)

    def getHash(self, title: str) -> str | None:
        """Return the stored content hash for a title, or None if it was never indexed."""
        with self._connect() as conn:
            row = conn.execute("SELECT sha256 FROM ingestedFiles WHERE title = ?", (title,)).fetchone()
            return row["sha256"] if row else None

    def upsert(self, title: str, sha256: str, chunkCount: int) -> None:
        """Record that a title was indexed with the given content hash."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO ingestedFiles (title, sha256, chunkCount, ingestedAt) VALUES (?, ?, ?, ?)
                   ON CONFLICT(title) DO UPDATE SET
                       sha256 = excluded.sha256,
                       chunkCount = excluded.chunkCount,
                       ingestedAt = excluded.ingestedAt""",
                (title, sha256, chunkCount, time.time()),
            )

    def delete(self, title: str) -> None:
        """Forget a title."""
        with self._connect() as conn:
            conn.execute("DELETE FROM ingestedFiles WHERE title = ?", (title,))

    def listTitles(self) -> list[str]:
        """Return every title currently recorded as indexed."""
        with self._connect() as conn:
            return [row["title"] for row in conn.execute("SELECT title FROM ingestedFiles")]
