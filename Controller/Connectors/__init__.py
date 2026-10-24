"""
Connectors fetch items from external APIs and normalise them into FeedItems.

Contract: fetch(feedConfig: dict, feedState: dict) -> list[FeedItem]
  feedConfig — one entry from feeds.json (name, connector, url/params, …)
  feedState  — per-feed dict persisted between polls; a connector may read and
               update it (e.g. RSS ETag / Last-Modified) to skip unchanged feeds.
Connectors raise on failure; FeedPollerService handles retries and backoff.
"""

from Controller.Connectors import mastodonConnector, newsApiConnector, redditConnector, rssConnector

Connectors: dict = {
    'rss':      rssConnector.fetch,
    'reddit':   redditConnector.fetch,
    'mastodon': mastodonConnector.fetch,
    'newsapi':  newsApiConnector.fetch,
}
