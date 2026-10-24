"""
FeedStateRepository persists what the feed poller has already sent to the
retrieval service, so each poll only pushes new or changed items and old items
can be expired.

items — one row per ingested item (title = canonical URL) with its content hash.
feeds — per-feed connector state (e.g. RSS ETag) and scheduling/backoff info.
"""

import json
import os
import sqlite3
import time


class FeedStateRepository:
    """SQLite-backed poller state."""

    def __init__(self, dbPath: str = "./feed_db/state.sqlite3"):
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
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS items (
                    title       TEXT PRIMARY KEY,
                    feedName    TEXT NOT NULL,
                    sha256      TEXT NOT NULL,
                    publishedAt REAL,
                    ingestedAt  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_items_feed ON items(feedName);

                CREATE TABLE IF NOT EXISTS feeds (
                    name          TEXT PRIMARY KEY,
                    connectorState TEXT NOT NULL DEFAULT '{}',
                    lastPolledAt  REAL,
                    lastSuccessAt REAL,
                    lastError     TEXT,
                    failures      INTEGER NOT NULL DEFAULT 0,
                    nextPollAt    REAL NOT NULL DEFAULT 0
                );
            """)

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def getItemHashes(self, titles: list[str]) -> dict[str, str]:
        """Return {title: sha256} for the titles already ingested."""
        if not titles:
            return {}
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in titles)
            return {
                row["title"]: row["sha256"]
                for row in conn.execute(f"SELECT title, sha256 FROM items WHERE title IN ({placeholders})", titles)
            }

    def upsertItems(self, rows: list[tuple[str, str, str, float | None]]) -> None:
        """Record ingested items as (title, feedName, sha256, publishedAt)."""
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO items (title, feedName, sha256, publishedAt, ingestedAt) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(title) DO UPDATE SET
                       feedName = excluded.feedName, sha256 = excluded.sha256,
                       publishedAt = excluded.publishedAt, ingestedAt = excluded.ingestedAt""",
                [(title, feedName, sha, publishedAt, now) for title, feedName, sha, publishedAt in rows],
            )

    def expiredItems(self, feedName: str, cutoff: float) -> list[str]:
        """Titles of a feed's items published (or, if undated, ingested) before cutoff."""
        with self._connect() as conn:
            return [
                row["title"] for row in conn.execute(
                    "SELECT title FROM items WHERE feedName = ? AND COALESCE(publishedAt, ingestedAt) < ?",
                    (feedName, cutoff),
                )
            ]

    def deleteItems(self, titles: list[str]) -> None:
        """Forget items (after they were removed from the index)."""
        with self._connect() as conn:
            conn.executemany("DELETE FROM items WHERE title = ?", [(t,) for t in titles])

    def countItems(self, feedName: str) -> int:
        """Number of a feed's items currently in the index."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM items WHERE feedName = ?", (feedName,)).fetchone()[0]

    # ------------------------------------------------------------------
    # Feeds
    # ------------------------------------------------------------------

    def getFeed(self, name: str) -> dict:
        """Return a feed's state row as a dict (defaults if it was never polled)."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM feeds WHERE name = ?", (name,)).fetchone()
        if row is None:
            return {"name": name, "connectorState": {}, "lastPolledAt": None, "lastSuccessAt": None,
                    "lastError": None, "failures": 0, "nextPollAt": 0.0}
        feed = dict(row)
        feed["connectorState"] = json.loads(feed["connectorState"] or "{}")
        return feed

    def saveFeed(self, feed: dict) -> None:
        """Persist a feed's state row."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO feeds (name, connectorState, lastPolledAt, lastSuccessAt, lastError, failures, nextPollAt)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       connectorState = excluded.connectorState, lastPolledAt = excluded.lastPolledAt,
                       lastSuccessAt = excluded.lastSuccessAt, lastError = excluded.lastError,
                       failures = excluded.failures, nextPollAt = excluded.nextPollAt""",
                (
                    feed["name"], json.dumps(feed.get("connectorState") or {}), feed.get("lastPolledAt"),
                    feed.get("lastSuccessAt"), feed.get("lastError"), feed.get("failures", 0),
                    feed.get("nextPollAt", 0.0),
                ),
            )
