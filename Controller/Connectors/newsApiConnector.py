"""
NewsAPI connector — keyword news search via https://newsapi.org (needs NEWSAPI_KEY in .env).
The free tier returns truncated article bodies and delays results.

feeds.json entry: {"connector": "newsapi", "query": "inflation", "language": "en", "pageSize": 50, …}
"""

import os

from Controller.Connectors.httpUtil import getJson, parseIsoTimestamp
from Dto.FeedItem import FeedItem


def fetch(feedConfig: dict, feedState: dict) -> list[FeedItem]:
    """Fetch the newest articles matching feedConfig['query']."""
    apiKey = os.environ.get('NEWSAPI_KEY')
    if not apiKey:
        raise RuntimeError("NEWSAPI_KEY is not set")
    data = getJson(
        "https://newsapi.org/v2/everything",
        params={
            'q':        feedConfig['query'],
            'language': feedConfig.get('language', 'en'),
            'sortBy':   'publishedAt',
            'pageSize': feedConfig.get('pageSize', 50),
        },
        headers={'X-Api-Key': apiKey},
    )
    items: list[FeedItem] = []
    for article in data.get('articles', []):
        if not article.get('url'):
            continue
        body = '\n'.join(part for part in (article.get('description'), article.get('content')) if part)
        items.append(FeedItem(
            url=article['url'],
            headline=article.get('title') or '',
            body=body,
            publishedAt=parseIsoTimestamp(article.get('publishedAt')),
            author=article.get('author') or (article.get('source') or {}).get('name'),
            platform='newsapi',
        ))
    return items
