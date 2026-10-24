"""
FeedItem is what every connector returns: one news article, blog post or social
post, normalised so FeedPollerService can treat all external sources the same way.
"""

from dataclasses import dataclass


@dataclass
class FeedItem:
    """
    A single item fetched from an external source.

    url         — canonical link; becomes the Source title, so it must be stable and unique.
    headline    — title of the article/post (for SNS: the first words of the post).
    body        — main text; may contain HTML (cleaned later by cleanFeedText).
    publishedAt — epoch seconds, or None when the source does not say.
    author      — author or account name, if known.
    platform    — origin system, e.g. 'rss', 'reddit', 'mastodon', 'newsapi'.
    """
    url: str
    headline: str
    body: str
    publishedAt: float | None = None
    author: str | None = None
    platform: str | None = None
