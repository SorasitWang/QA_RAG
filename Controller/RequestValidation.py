"""
Shared validation for search/chat request bodies, used by both the web service
and the retrieval service so the rules stay identical at each boundary.
"""

SearchModes: tuple[str, ...] = ("vector", "sparse", "hybrid", "graph")


def validateSearchRequest(body: dict) -> tuple[str, str, int, str | None]:
    """Extract and validate common fields. Returns (query, mode, topK, error)."""
    query = (body.get("query") or "").strip()
    mode  = body.get("mode", "vector")
    topK  = body.get("topK", 5)

    if not query:
        return "", "", 0, "query is required"
    if mode not in SearchModes:
        return "", "", 0, "mode must be vector, sparse, hybrid, or graph"
    try:
        topK = int(topK)
        if not (1 <= topK <= 50):
            raise ValueError
    except (ValueError, TypeError):
        return "", "", 0, "topK must be an integer between 1 and 50"

    return query, mode, topK, None


def validateSourceTypes(body: dict) -> tuple[list[str] | None, str | None]:
    """
    Extract the optional sourceTypes filter. Returns (sourceTypes, error);
    sourceTypes is None when no filter was requested.
    """
    from Dto.Source import SourceType

    sourceTypes = body.get("sourceTypes")
    if sourceTypes is None:
        return None, None
    if not isinstance(sourceTypes, list) or not all(isinstance(t, str) for t in sourceTypes):
        return None, "sourceTypes must be a list of strings"
    unknown = [t for t in sourceTypes if t not in SourceType.values()]
    if unknown:
        return None, f"unknown sourceTypes {unknown}; expected {SourceType.values()}"
    # An empty list means "no filter" rather than "match nothing".
    return (sourceTypes or None), None
