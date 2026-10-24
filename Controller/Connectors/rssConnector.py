"""
RSS/Atom connector — covers most news sites and blogs without API keys.
Sends ETag / Last-Modified from the previous poll so unchanged feeds return 304.

feeds.json entry: {"connector": "rss", "url": "https://example.com/feed.xml", …}
"""

import calendar

import feedparser

from Config import Config
from Dto.FeedItem import FeedItem


def fetch(feedConfig: dict, feedState: dict) -> list[FeedItem]:
    """Fetch new entries from an RSS/Atom feed; returns [] when the feed is unchanged."""
    parsed = feedparser.parse(
        feedConfig['url'],
        etag=feedState.get('etag'),
        modified=feedState.get('modified'),
        agent=Config.Feeds.UserAgent,
    )
    if getattr(parsed, 'status', None) == 304:
        return []
    if parsed.bozo and not parsed.entries:
        raise RuntimeError(f"could not parse feed {feedConfig['url']}: {parsed.get('bozo_exception')}")
    if getattr(parsed, 'status', 200) >= 400:
        raise RuntimeError(f"feed {feedConfig['url']} returned HTTP {parsed.status}")

    feedState['etag'] = parsed.get('etag')
    feedState['modified'] = parsed.get('modified')
    return [item for item in (_toItem(entry) for entry in parsed.entries) if item is not None]


def _toItem(entry) -> FeedItem | None:
    url = entry.get('link') or entry.get('id')
    if not url:
        return None
    # Prefer full content (Atom <content>, RSS content:encoded) over the summary.
    body = entry['content'][0].get('value', '') if entry.get('content') else entry.get('summary', '')
    published = entry.get('published_parsed') or entry.get('updated_parsed')
    return FeedItem(
        url=url,
        headline=entry.get('title', ''),
        body=body,
        publishedAt=float(calendar.timegm(published)) if published else None,
        author=entry.get('author'),
        platform='rss',
    )
