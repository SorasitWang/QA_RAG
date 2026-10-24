"""
Mastodon connector — public posts for a hashtag on any instance (no key needed
on most instances).

feeds.json entry: {"connector": "mastodon", "instance": "https://mastodon.social",
                   "hashtag": "economics", "limit": 40, …}
"""

from Controller.Connectors.httpUtil import getJson, headlineFromText, parseIsoTimestamp
from Controller.SearchData import cleanFeedText
from Dto.FeedItem import FeedItem


def fetch(feedConfig: dict, feedState: dict) -> list[FeedItem]:
    """Fetch recent public posts tagged feedConfig['hashtag']."""
    instance = feedConfig['instance'].rstrip('/')
    statuses = getJson(
        f"{instance}/api/v1/timelines/tag/{feedConfig['hashtag']}",
        params={'limit': feedConfig.get('limit', 40)},
    )
    items: list[FeedItem] = []
    for status in statuses:
        # Boosts carry the original post in 'reblog'; index the original once.
        status = status.get('reblog') or status
        if not status.get('url'):
            continue
        content = status.get('content', '')
        items.append(FeedItem(
            url=status['url'],
            headline=headlineFromText(cleanFeedText(content)),
            body=content,
            publishedAt=parseIsoTimestamp(status.get('created_at')),
            author=(status.get('account') or {}).get('acct'),
            platform='mastodon',
        ))
    return items
