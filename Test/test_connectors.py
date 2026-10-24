import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Controller.Connectors import mastodonConnector, newsApiConnector, redditConnector, rssConnector

_Rss = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel><title>Sample</title>
  <item>
    <title>Rates rise again</title>
    <link>https://news.example/rates</link>
    <description>Short summary</description>
    <content:encoded><![CDATA[<p>Full <b>article</b> text</p>]]></content:encoded>
    <pubDate>Tue, 14 Nov 2023 22:13:20 GMT</pubDate>
    <author>jane@example.com (Jane)</author>
  </item>
  <item>
    <title>No date item</title>
    <link>https://news.example/nodate</link>
    <description>Body only</description>
  </item>
  <item><title>No link is skipped</title></item>
</channel></rss>"""


class TestRssConnector(unittest.TestCase):

    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpDir, "feed.xml")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(_Rss)

    def tearDown(self):
        shutil.rmtree(self.tmpDir, ignore_errors=True)

    def test_parses_items(self):
        items = rssConnector.fetch({"url": self.path}, {})
        self.assertEqual([i.url for i in items], ["https://news.example/rates", "https://news.example/nodate"])
        first = items[0]
        self.assertEqual(first.headline, "Rates rise again")
        self.assertIn("Full <b>article</b> text", first.body)  # content:encoded preferred over summary
        self.assertEqual(first.publishedAt, 1700000000.0)
        self.assertEqual(first.platform, "rss")
        self.assertIsNone(items[1].publishedAt)
        self.assertEqual(items[1].body, "Body only")

    def test_not_modified_returns_nothing(self):
        class NotModified(dict):
            status = 304
            bozo = False
            entries = []
        with patch.object(rssConnector.feedparser, "parse", return_value=NotModified()):
            self.assertEqual(rssConnector.fetch({"url": "https://x"}, {"etag": "abc"}), [])

    def test_unparseable_feed_raises(self):
        with open(self.path, "w") as f:
            f.write("this is not xml <<<")
        with self.assertRaises(RuntimeError):
            rssConnector.fetch({"url": self.path}, {})


class TestRedditConnector(unittest.TestCase):

    def test_maps_posts(self):
        payload = {"data": {"children": [
            {"data": {"permalink": "/r/economics/comments/1/x/", "title": "CPI up", "selftext": "Discuss",
                      "created_utc": 1700000000, "author": "bob"}},
            {"data": {"title": "missing permalink"}},
        ]}}
        with patch.object(redditConnector, "getJson", return_value=payload) as getJson:
            items = redditConnector.fetch({"subreddit": "economics"}, {})
        self.assertIn("/r/economics/new.json", getJson.call_args.args[0])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://www.reddit.com/r/economics/comments/1/x/")
        self.assertEqual((items[0].headline, items[0].body, items[0].author), ("CPI up", "Discuss", "bob"))


class TestMastodonConnector(unittest.TestCase):

    def test_maps_statuses_and_unwraps_boosts(self):
        payload = [
            {"url": "https://m.social/@a/1", "content": "<p>Inflation is cooling https://x.y</p>",
             "created_at": "2023-11-14T22:13:20.000Z", "account": {"acct": "a"}},
            {"reblog": {"url": "https://m.social/@b/2", "content": "<p>Original</p>",
                        "created_at": "2023-11-14T22:13:20Z", "account": {"acct": "b"}}},
        ]
        with patch.object(mastodonConnector, "getJson", return_value=payload):
            items = mastodonConnector.fetch({"instance": "https://m.social/", "hashtag": "economics"}, {})
        self.assertEqual([i.url for i in items], ["https://m.social/@a/1", "https://m.social/@b/2"])
        self.assertEqual(items[0].headline, "Inflation is cooling")
        self.assertEqual(items[0].publishedAt, 1700000000.0)
        self.assertEqual(items[1].author, "b")


class TestNewsApiConnector(unittest.TestCase):

    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                newsApiConnector.fetch({"query": "inflation"}, {})

    def test_maps_articles(self):
        payload = {"articles": [{"url": "https://n.example/a", "title": "T", "description": "D",
                                 "content": "C", "publishedAt": "2023-11-14T22:13:20Z",
                                 "author": None, "source": {"name": "Reuters"}}]}
        with patch.dict(os.environ, {"NEWSAPI_KEY": "k"}), \
             patch.object(newsApiConnector, "getJson", return_value=payload) as getJson:
            items = newsApiConnector.fetch({"query": "inflation"}, {})
        self.assertEqual(getJson.call_args.kwargs["headers"], {"X-Api-Key": "k"})
        self.assertEqual((items[0].body, items[0].author, items[0].publishedAt), ("D\nC", "Reuters", 1700000000.0))


if __name__ == '__main__':
    unittest.main()
