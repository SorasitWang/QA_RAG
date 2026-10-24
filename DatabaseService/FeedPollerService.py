"""
FeedPollerService fetches external feeds on a schedule and pushes their items
into the retrieval service, which indexes them like any other source.

Per poll of one feed:
  1. Call the feed's connector (RSS, Reddit, Mastodon, NewsAPI …)
  2. Drop items older than the feed's retention window
  3. Compare each item's content hash with the poller state; keep new/changed ones
  4. POST them to the retrieval service's /ingest in batches, then record them
  5. Remove items that aged out of the retention window via /ingest/delete

A failing feed backs off exponentially and never blocks the other feeds.
"""

import hashlib
import json
import logging
import threading
import time
from typing import Callable

from Config import Config
from Controller.Connectors import Connectors
from Controller.RetrievalClient import RetrievalClient
from DatabaseRepository.FeedStateRepository import FeedStateRepository
from Dto.FeedItem import FeedItem
from Dto.Source import Source, SourceType

logger = logging.getLogger(__name__)


def loadFeeds(path: str, connectors: dict = Connectors) -> list[dict]:
    """
    Read and validate the feeds config (a JSON list). Applies default interval and
    retention. Raises ValueError with a readable message on invalid entries.
    """
    with open(path, 'r', encoding='utf-8') as f:
        feeds = json.load(f)
    if not isinstance(feeds, list):
        raise ValueError(f"{path} must contain a JSON list of feeds")

    names: set[str] = set()
    allowedTypes = [t for t in SourceType.values() if t != SourceType.File.value]
    for feed in feeds:
        name = feed.get('name')
        if not name or name in names:
            raise ValueError(f"every feed needs a unique 'name' (got {name!r})")
        names.add(name)
        if feed.get('connector') not in connectors:
            raise ValueError(f"feed '{name}': unknown connector {feed.get('connector')!r}; expected {sorted(connectors)}")
        if feed.get('sourceType') not in allowedTypes:
            raise ValueError(f"feed '{name}': sourceType must be one of {allowedTypes}")
        feed.setdefault('intervalMinutes', Config.Feeds.DefaultIntervalMinutes)
        feed.setdefault('retentionDays', Config.Feeds.DefaultRetentionDays)
        feed.setdefault('enabled', True)
    return feeds


def _itemHash(item: FeedItem) -> str:
    return hashlib.sha256(f"{item.headline}\n{item.body}".encode('utf-8')).hexdigest()


def toSource(item: FeedItem, feed: dict) -> Source:
    """Map a FeedItem to a Source keyed by its URL so re-ingesting replaces, never duplicates."""
    return Source(
        title=item.url,
        sourceType=feed['sourceType'],
        content=f"{item.headline}\n{item.body}" if item.headline else item.body,
        tags=[],
        optional={
            'url':         item.url,
            'headline':    item.headline or None,
            'publishedAt': int(item.publishedAt) if item.publishedAt else None,
            'author':      item.author,
            'feedName':    feed['name'],
            'platform':    item.platform,
        },
    )


class FeedPollerService:
    """Schedules feed polls, de-duplicates items and enforces retention."""

    def __init__(
        self,
        feeds: list[dict],
        stateRepository: FeedStateRepository,
        retrievalClient: RetrievalClient,
        connectors: dict = Connectors,
        clock: Callable[[], float] = time.time,
    ):
        self.feeds: dict[str, dict] = {feed['name']: feed for feed in feeds}
        self.state: FeedStateRepository = stateRepository
        self.client: RetrievalClient = retrievalClient
        self.connectors: dict = connectors
        self.clock: Callable[[], float] = clock
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # One feed
    # ------------------------------------------------------------------

    def pollFeed(self, name: str) -> dict[str, int]:
        """
        Poll one feed now and push new/changed items. Returns counts of
        fetched, ingested, skipped and expired items. Raises on failure.
        """
        feed = self.feeds[name]
        feedState = self.state.getFeed(name)
        connectorState: dict = dict(feedState['connectorState'])

        items: list[FeedItem] = self.connectors[feed['connector']](feed, connectorState)
        cutoff = self.clock() - feed['retentionDays'] * 86400

        # Drop items already past retention so they are not re-ingested and expired every poll.
        fresh = [i for i in items if i.publishedAt is None or i.publishedAt >= cutoff]
        # Some feeds repeat an item; keep the last occurrence per URL.
        unique: dict[str, FeedItem] = {i.url: i for i in fresh}

        known = self.state.getItemHashes(list(unique))
        changed = [(item, _itemHash(item)) for item in unique.values() if known.get(item.url) != _itemHash(item)]

        batchSize = Config.Feeds.IngestBatchSize
        for start in range(0, len(changed), batchSize):
            batch = changed[start:start + batchSize]
            self.client.ingest([toSource(item, feed) for item, _ in batch])
            # Record only after the retrieval service accepted the batch, so failures retry.
            self.state.upsertItems([(item.url, name, sha, item.publishedAt) for item, sha in batch])

        expired = self.expireFeed(name, cutoff)

        feedState['connectorState'] = connectorState
        self.state.saveFeed(feedState)
        return {'fetched': len(items), 'ingested': len(changed),
                'skipped': len(unique) - len(changed), 'expired': expired}

    def expireFeed(self, name: str, cutoff: float) -> int:
        """Remove a feed's items older than cutoff from the index and the state. Returns count."""
        titles = self.state.expiredItems(name, cutoff)
        batchSize = Config.Feeds.IngestBatchSize
        for start in range(0, len(titles), batchSize):
            batch = titles[start:start + batchSize]
            self.client.deleteSources(batch)
            self.state.deleteItems(batch)
        return len(titles)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def runDueFeeds(self) -> None:
        """Poll every enabled feed whose next poll time has passed."""
        for name, feed in self.feeds.items():
            if not feed.get('enabled', True):
                continue
            feedState = self.state.getFeed(name)
            now = self.clock()
            if feedState['nextPollAt'] > now:
                continue
            self._pollWithBackoff(name, feed)

    def _pollWithBackoff(self, name: str, feed: dict) -> None:
        interval = feed['intervalMinutes'] * 60
        try:
            counts = self.pollFeed(name)
            feedState = self.state.getFeed(name)
            feedState.update(lastSuccessAt=self.clock(), lastError=None, failures=0)
            logger.info("Polled feed '%s': %s", name, counts)
        except Exception as exc:
            logger.exception("Polling feed '%s' failed", name)
            feedState = self.state.getFeed(name)
            feedState['failures'] += 1
            feedState['lastError'] = str(exc)
            # Double the wait per consecutive failure, capped (but never below the normal interval).
            interval = min(interval * 2 ** feedState['failures'], max(interval, Config.Feeds.MaxBackoffSeconds))
        feedState['lastPolledAt'] = self.clock()
        feedState['nextPollAt'] = self.clock() + interval
        self.state.saveFeed(feedState)

    def requestPoll(self, name: str) -> None:
        """Make a feed due immediately and wake the scheduler."""
        feedState = self.state.getFeed(name)
        feedState['nextPollAt'] = 0.0
        self.state.saveFeed(feedState)
        self._wake.set()

    def start(self, tickSeconds: float = 5.0) -> None:
        """Run the scheduler in a background daemon thread."""
        def loop():
            while not self._stop.is_set():
                self.runDueFeeds()
                self._wake.wait(tickSeconds)
                self._wake.clear()
        self._thread = threading.Thread(target=loop, name="feed-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler thread after its current poll."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def status(self) -> list[dict]:
        """Per-feed status for GET /feeds/status."""
        result: list[dict] = []
        for name, feed in self.feeds.items():
            feedState = self.state.getFeed(name)
            result.append({
                'name':          name,
                'connector':     feed['connector'],
                'sourceType':    feed['sourceType'],
                'enabled':       feed.get('enabled', True),
                'items':         self.state.countItems(name),
                'lastPolledAt':  feedState['lastPolledAt'],
                'lastSuccessAt': feedState['lastSuccessAt'],
                'nextPollAt':    feedState['nextPollAt'],
                'failures':      feedState['failures'],
                'lastError':     feedState['lastError'],
            })
        return result
