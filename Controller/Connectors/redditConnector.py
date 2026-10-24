"""
Reddit connector — newest posts of a subreddit via the public JSON endpoint (no key).

feeds.json entry: {"connector": "reddit", "subreddit": "economics", "limit": 25, …}
"""

from Controller.Connectors.httpUtil import getJson
from Dto.FeedItem import FeedItem


def fetch(feedConfig: dict, feedState: dict) -> list[FeedItem]:
    """Fetch the newest posts of feedConfig['subreddit']."""
    data = getJson(
        f"https://www.reddit.com/r/{feedConfig['subreddit']}/new.json",
        params={'limit': feedConfig.get('limit', 25)},
    )
    items: list[FeedItem] = []
    for child in data.get('data', {}).get('children', []):
        post = child.get('data', {})
        if not post.get('permalink'):
            continue
        items.append(FeedItem(
            url=f"https://www.reddit.com{post['permalink']}",
            headline=post.get('title', ''),
            # Link posts have no selftext; keep the linked URL's title as the content.
            body=post.get('selftext') or '',
            publishedAt=post.get('created_utc'),
            author=post.get('author'),
            platform='reddit',
        ))
    return items
