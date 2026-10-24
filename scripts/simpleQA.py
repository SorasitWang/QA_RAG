"""
simpleQA — interactive RAG Q&A bot.
Each user message triggers retrieval and the chunks are injected into the LLM
prompt via LLMController.

Commands:
  /check              — inspect full conversation history with retrieved chunks
  /mode vector        — semantic similarity search (default)
  /mode sparse        — TF-IDF keyword search
  /mode hybrid        — Reciprocal Rank Fusion of vector + TF-IDF
  quit / exit         — stop
"""

###########################
# Standard library imports
###########################
import chromadb

###########################
# Local application imports
###########################
from Config import Config
from Controller.LLMController import LLMController
from DatabaseRepository.TfidfRepository import TfidfRepository
from DatabaseRepository.VectorRepository import VectorRepository
from DatabaseService.RagService import RagService

# ANSI escape codes for terminal colours
Reset   = "\033[0m"
Bold    = "\033[1m"
Dim     = "\033[2m"
Cyan    = "\033[36m"
Green   = "\033[32m"
Magenta = "\033[35m"
Yellow  = "\033[33m"
White   = "\033[97m"


def printHistory(controller: LLMController):
    """
    Prints a formatted debug view of the controller's exchange history.
    For each turn shows: user query, retrieved chunks with distance/similarity
    scores, and the assistant reply.  A progress bar at the bottom indicates
    how close the history is to the compaction threshold.
    """
    width = 72
    print(f"\n{Bold}{Yellow}{'━' * width}{Reset}")
    print(f"{Bold}{Yellow}  CHAT HISTORY DEBUG{Reset}")
    print(f"{Bold}{Yellow}{'━' * width}{Reset}\n")

    if controller.systemMessage:
        print(f"{Bold}{Cyan}[SYSTEM]{Reset}")
        print(f"{Cyan}{controller.systemMessage}{Reset}\n")
    else:
        print(f"{Dim}[SYSTEM]  (none){Reset}\n")

    if not controller.exchanges:
        print(f"{Dim}  No exchanges yet.{Reset}")
    else:
        for i, (userMsg, assistantMsg, chunks, distances) in enumerate(controller.exchanges, start=1):
            print(f"{Bold}{Yellow}── Turn {i} {'─' * (width - 9)}{Reset}")

            print(f"{Bold}{Green}[USER]{Reset}")
            print(f"{Green}{userMsg}{Reset}\n")

            if chunks:
                print(f"{Bold}{Cyan}[RETRIEVED — {len(chunks)} chunk(s)]{Reset}")
                for j, (chunk, dist) in enumerate(zip(chunks, distances), start=1):
                    preview = chunk.replace("\n", " ")[:80]
                    if len(chunk) > 80:
                        preview += "…"
                    # sim converts L2 distance to a 0-1 similarity score
                    # valid for normalised sentence-transformer embeddings
                    score = 1 - dist / 2
                    print(f"  {Cyan}{j}. dist={dist:.4f}  sim={score:.4f}  │  {preview}{Reset}")
                print()
            else:
                print(f"{Dim}  [RETRIEVED — none passed threshold]{Reset}\n")

            print(f"{Bold}{Magenta}[ASSISTANT]{Reset}")
            print(f"{Magenta}{assistantMsg}{Reset}\n")

    # History size progress bar — turns yellow when approaching compaction threshold
    totalChars = sum(len(u) + len(a) for u, a, *_ in controller.exchanges)
    threshold  = Config.Model.MaxChatHistoryChars
    barFilled  = min(int(totalChars / threshold * 20) if threshold else 0, 20)
    bar   = "█" * barFilled + "░" * (20 - barFilled)
    color = Yellow if barFilled >= 16 else Green
    print(f"{Bold}{Yellow}{'━' * width}{Reset}")
    print(f"  History size: {color}{bar}{Reset}  {totalChars}/{threshold} chars\n")


_VALID_MODES = ("vector", "sparse", "hybrid")


def main():
    client = chromadb.PersistentClient(path="./embeddings_db")
    collection = client.get_collection(
        name="my_docs",
        embedding_function=Config.Model.getEmbeddingFunction(),
    )
    vectorRepo = VectorRepository(collection)
    tfidfRepo  = TfidfRepository(Config.Model.TfidfDbPath)
    ragService = RagService(vectorRepo, tfidfRepo)

    controller  = LLMController()
    searchMode  = "vector"

    modeHelp = " / ".join(f"{Cyan}{m}{Reset}" for m in _VALID_MODES)
    print(
        f"{Bold}{White}RAG Q&A bot ready.{Reset}  "
        f"Mode: {Cyan}{searchMode}{Reset}  │  "
        f"Commands: {Cyan}/check{Reset}  {Cyan}/mode{Reset} [{modeHelp}]  {Dim}quit/exit{Reset}\n"
    )

    while True:
        try:
            userInput = input(f"{Bold}{Green}You:{Reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not userInput:
            continue

        if userInput.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if userInput.lower() == "/check":
            printHistory(controller)
            continue

        if userInput.lower().startswith("/mode"):
            parts = userInput.split()
            if len(parts) == 2 and parts[1] in _VALID_MODES:
                searchMode = parts[1]
                print(f"  Search mode → {Cyan}{searchMode}{Reset}\n")
            else:
                print(f"  Usage: /mode [{' | '.join(_VALID_MODES)}]\n")
            continue

        searchResult           = ragService.query(userInput, topK=5, searchMode=searchMode)
        chunks: list[str]      = searchResult.documents
        distances: list[float] = searchResult.distances

        reply = controller.generateResponse(userInput, chunks, distances)
        print(f"\n{Bold}{Magenta}Bot:{Reset} {reply}\n")


if __name__ == "__main__":
    main()
