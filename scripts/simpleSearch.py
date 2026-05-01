"""
simpleSearch — interactive document retrieval viewer.
Type a sentence to see which chunks matched, their relevance scores, source
titles, and full content.  Useful for inspecting retrieval quality without
running the full QA bot.

Commands:
  /mode vector        — semantic similarity search (default)
  /mode sparse        — TF-IDF keyword search
  /mode hybrid        — Reciprocal Rank Fusion of vector + TF-IDF
  /topk N             — change number of results shown (default 5)
  quit / exit         — stop
"""

import chromadb

from Config import Config
from DatabaseRepository.TfidfRepository import TfidfRepository
from DatabaseRepository.VectorRepository import VectorRepository
from DatabaseService.RagService import RagService
from Dto.RetrievalResult import RetrievalResult

# ANSI colours
Reset  = "\033[0m"
Bold   = "\033[1m"
Dim    = "\033[2m"
Cyan   = "\033[36m"
Green  = "\033[32m"
Yellow = "\033[33m"
Red    = "\033[31m"
White  = "\033[97m"

_VALID_MODES = ("vector", "sparse", "hybrid")


def _scoreBar(sim: float, width: int = 20) -> str:
    """Visual bar representing similarity score 0–1."""
    filled = round(sim * width)
    color = Green if sim >= 0.7 else (Yellow if sim >= 0.4 else Red)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{Reset}"


def _toSimilarity(dist: float, mode: str) -> float:
    """
    Convert stored distance to a 0–1 similarity score.
    Vector mode uses L2 distance on normalised embeddings (range 0–2).
    Sparse/hybrid distances are already normalised to [0, 1].
    """
    if mode == "vector":
        return max(0.0, 1.0 - dist / 2.0)
    return max(0.0, 1.0 - dist)


def _printResults(result: RetrievalResult, mode: str, query: str) -> None:
    ids       = result.ids
    docs      = result.documents
    distances = result.distances
    metadatas = result.metadatas

    width = 72
    print(f"\n{Bold}{Yellow}{'━' * width}{Reset}")
    print(f"{Bold}{White}  Query:{Reset} {Green}{query}{Reset}  "
          f"{Dim}[mode: {mode}]{Reset}")
    print(f"{Bold}{Yellow}{'━' * width}{Reset}")

    if not ids:
        print(f"\n  {Dim}No results passed the similarity threshold.{Reset}\n")
        return

    for rank, (docId, doc, dist, meta) in enumerate(
        zip(ids, docs, distances, metadatas), start=1
    ):
        sim   = _toSimilarity(dist, mode)
        title = meta.get("title", docId)
        bar   = _scoreBar(sim)

        print(f"\n  {Bold}#{rank}{Reset}  {bar}  "
              f"{Bold}{sim:.3f}{Reset}  {Cyan}{title}{Reset}")
        print(f"  {Dim}{'─' * (width - 2)}{Reset}")

        # Print full chunk content, word-wrapped at ~68 chars
        words, line = doc.split(), ""
        for word in words:
            if len(line) + len(word) + 1 > 68:
                print(f"    {line}")
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            print(f"    {line}")

    print(f"\n{Bold}{Yellow}{'━' * width}{Reset}")
    print(f"  {Dim}{len(ids)} chunk(s) returned{Reset}\n")


def main():
    client = chromadb.PersistentClient(path="./embeddings_db")
    collection = client.get_collection(
        name="my_docs",
        embedding_function=Config.Model.EmbeddingFunction,
    )
    vectorRepo = VectorRepository(collection)
    tfidfRepo  = TfidfRepository(Config.Model.TfidfDbPath)
    ragService = RagService(vectorRepo, tfidfRepo)

    searchMode = "vector"
    topK       = 5

    modeHelp = " / ".join(f"{Cyan}{m}{Reset}" for m in _VALID_MODES)
    print(
        f"\n{Bold}{White}Document Search Viewer{Reset}\n"
        f"  Mode : {Cyan}{searchMode}{Reset}   "
        f"TopK : {Cyan}{topK}{Reset}\n"
        f"  Commands: {Cyan}/mode{Reset} [{modeHelp}]  "
        f"{Cyan}/topk N{Reset}  {Dim}quit{Reset}\n"
    )

    while True:
        try:
            userInput = input(f"{Bold}{Green}Query:{Reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not userInput:
            continue

        if userInput.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if userInput.lower().startswith("/mode"):
            parts = userInput.split()
            if len(parts) == 2 and parts[1] in _VALID_MODES:
                searchMode = parts[1]
                print(f"  Mode → {Cyan}{searchMode}{Reset}\n")
            else:
                print(f"  Usage: /mode [{' | '.join(_VALID_MODES)}]\n")
            continue

        if userInput.lower().startswith("/topk"):
            parts = userInput.split()
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                topK = int(parts[1])
                print(f"  TopK → {Cyan}{topK}{Reset}\n")
            else:
                print(f"  Usage: /topk <positive integer>\n")
            continue

        result = ragService.query(userInput, topK=topK, searchMode=searchMode)
        _printResults(result, searchMode, userInput)


if __name__ == "__main__":
    main()
