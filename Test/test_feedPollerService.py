import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DatabaseRepository.FeedStateRepository import FeedStateRepository
from DatabaseService.FeedPollerService import FeedPollerService, loadFeeds, toSource
from Dto.FeedItem import FeedItem
from feedPollerApp import createApp

_Now = 1_700_000_000.0
_Day = 86400


class FakeClock:
    def __init__(self, now: float = _Now):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _item(url: str, body: str = "body", ageDays: float = 0) -> FeedItem:
    return FeedItem(url=url, headline=f"headline {url}", body=body, publishedAt=_Now - ageDays * _Day, platform="rss")


def _feed(name: str = "news", **overrides) -> dict:
    feed = {"name": name, "connector": "fake", "sourceType": "news", "intervalMinutes": 10, "retentionDays": 7}
    feed.update(overrides)
    return feed


class _PollerCase(unittest.TestCase):

    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()
        self.state = FeedStateRepository(os.path.join(self.tmpDir, "state.sqlite3"))
        self.client = MagicMock()
        self.items: dict[str, list[FeedItem]] = {}
        self.clock = FakeClock()

    def tearDown(self):
        shutil.rmtree(self.tmpDir, ignore_errors=True)

    def _connector(self, feedConfig: dict, feedState: dict) -> list[FeedItem]:
        result = self.items[feedConfig["name"]]
        if isinstance(result, Exception):
            raise result
        return result

    def _service(self, *feeds: dict) -> FeedPollerService:
        return FeedPollerService(list(feeds) or [_feed()], self.state, self.client,
                                 connectors={"fake": self._connector}, clock=self.clock)

    def _ingestedUrls(self) -> list[str]:
        return [s.title for call in self.client.ingest.call_args_list for s in call.args[0]]


class TestPollFeed(_PollerCase):

    def test_new_items_are_ingested_with_source_type_and_metadata(self):
        self.items["news"] = [_item("https://a"), _item("https://b")]
        counts = self._service().pollFeed("news")
        self.assertEqual(counts["ingested"], 2)
        source = self.client.ingest.call_args.args[0][0]
        self.assertEqual(source.sourceType, "news")
        self.assertEqual(source.optional["url"], "https://a")
        self.assertEqual(source.optional["feedName"], "news")
        self.assertEqual(source.optional["publishedAt"], int(_Now))

    def test_unchanged_items_are_skipped(self):
        self.items["news"] = [_item("https://a")]
        service = self._service()
        service.pollFeed("news")
        counts = service.pollFeed("news")
        self.assertEqual((counts["ingested"], counts["skipped"]), (0, 1))
        self.assertEqual(self.client.ingest.call_count, 1)

    def test_edited_item_is_reingested(self):
        service = self._service()
        self.items["news"] = [_item("https://a", body="v1")]
        service.pollFeed("news")
        self.items["news"] = [_item("https://a", body="v2")]
        self.assertEqual(service.pollFeed("news")["ingested"], 1)
        self.assertEqual(self._ingestedUrls(), ["https://a", "https://a"])

    def test_items_older_than_retention_are_never_ingested(self):
        self.items["news"] = [_item("https://old", ageDays=30), _item("https://new")]
        self._service().pollFeed("news")
        self.assertEqual(self._ingestedUrls(), ["https://new"])

    def test_expired_items_are_deleted(self):
        service = self._service()
        self.items["news"] = [_item("https://a")]
        service.pollFeed("news")
        self.clock.now += 8 * _Day
        self.items["news"] = []
        self.assertEqual(service.pollFeed("news")["expired"], 1)
        self.client.deleteSources.assert_called_once_with(["https://a"])
        self.assertEqual(self.state.countItems("news"), 0)

    def test_duplicate_urls_in_one_fetch_are_ingested_once(self):
        self.items["news"] = [_item("https://a", body="first"), _item("https://a", body="second")]
        self._service().pollFeed("news")
        self.assertEqual(self._ingestedUrls(), ["https://a"])

    def test_failed_ingest_is_not_recorded_so_it_retries(self):
        self.items["news"] = [_item("https://a")]
        self.client.ingest.side_effect = ConnectionError("retrieval down")
        service = self._service()
        with self.assertRaises(ConnectionError):
            service.pollFeed("news")
        self.client.ingest.side_effect = None
        self.assertEqual(service.pollFeed("news")["ingested"], 1)


class TestScheduling(_PollerCase):

    def test_failing_feed_backs_off_without_blocking_others(self):
        self.items["bad"] = RuntimeError("feed down")
        self.items["good"] = [_item("https://g")]
        service = self._service(_feed("bad"), _feed("good"))

        service.runDueFeeds()

        bad, good = self.state.getFeed("bad"), self.state.getFeed("good")
        self.assertEqual(bad["failures"], 1)
        self.assertIn("feed down", bad["lastError"])
        self.assertEqual(bad["nextPollAt"], _Now + 2 * 600)    # interval doubled
        self.assertEqual(good["nextPollAt"], _Now + 600)
        self.assertEqual(self._ingestedUrls(), ["https://g"])

    def test_backoff_is_capped(self):
        self.items["bad"] = RuntimeError("down")
        service = self._service(_feed("bad"))
        for _ in range(10):
            self.clock.now = self.state.getFeed("bad")["nextPollAt"] or self.clock.now
            service.runDueFeeds()
        bad = self.state.getFeed("bad")
        self.assertLessEqual(bad["nextPollAt"] - self.clock.now, 3600)

    def test_success_resets_failures(self):
        self.items["news"] = RuntimeError("down")
        service = self._service()
        service.runDueFeeds()
        self.items["news"] = [_item("https://a")]
        self.clock.now = self.state.getFeed("news")["nextPollAt"]
        service.runDueFeeds()
        self.assertEqual(self.state.getFeed("news")["failures"], 0)

    def test_feed_not_polled_before_it_is_due(self):
        self.items["news"] = [_item("https://a")]
        service = self._service()
        service.runDueFeeds()
        self.items["news"] = [_item("https://b")]
        service.runDueFeeds()
        self.assertEqual(self._ingestedUrls(), ["https://a"])

    def test_request_poll_makes_feed_due(self):
        self.items["news"] = [_item("https://a")]
        service = self._service()
        service.runDueFeeds()
        self.items["news"] = [_item("https://b")]
        service.requestPoll("news")
        service.runDueFeeds()
        self.assertEqual(self._ingestedUrls(), ["https://a", "https://b"])

    def test_disabled_feed_is_skipped(self):
        self.items["news"] = [_item("https://a")]
        self._service(_feed(enabled=False)).runDueFeeds()
        self.client.ingest.assert_not_called()


class TestLoadFeeds(unittest.TestCase):

    def _load(self, feeds) -> list[dict]:
        path = os.path.join(tempfile.mkdtemp(), "feeds.json")
        with open(path, "w") as f:
            json.dump(feeds, f)
        return loadFeeds(path, connectors={"rss": None})

    def test_applies_defaults(self):
        feed = self._load([{"name": "a", "connector": "rss", "sourceType": "blog", "url": "u"}])[0]
        self.assertTrue(feed["enabled"])
        self.assertIn("intervalMinutes", feed)
        self.assertIn("retentionDays", feed)

    def test_rejects_invalid_entries(self):
        for bad in (
            [{"connector": "rss", "sourceType": "news"}],                                 # no name
            [{"name": "a", "connector": "ftp", "sourceType": "news"}],                    # unknown connector
            [{"name": "a", "connector": "rss", "sourceType": "file"}],                    # file is local-only
            [{"name": "a", "connector": "rss", "sourceType": "news"}] * 2,                # duplicate name
        ):
            with self.assertRaises(ValueError):
                self._load(bad)

    def test_example_config_is_valid(self):
        path = os.path.join(os.path.dirname(__file__), "..", "feeds.example.json")
        self.assertGreater(len(loadFeeds(path)), 0)


class TestToSource(unittest.TestCase):

    def test_undated_item_has_no_published_at(self):
        source = toSource(FeedItem(url="https://a", headline="", body="b"), _feed())
        self.assertEqual(source.content, "b")
        self.assertIsNone(source.optional["publishedAt"])


class TestFeedPollerApp(_PollerCase):

    def test_status_poll_and_health(self):
        self.items["news"] = [_item("https://a")]
        service = self._service()
        client = createApp(service).test_client()
        self.assertEqual(client.get("/health").get_json(), {"ok": True})
        self.assertEqual(client.post("/feeds/news/poll").status_code, 202)
        self.assertEqual(client.post("/feeds/missing/poll").status_code, 404)
        status = client.get("/feeds/status").get_json()
        self.assertEqual(status[0]["name"], "news")
        self.assertEqual(status[0]["sourceType"], "news")


if __name__ == '__main__':
    unittest.main()
