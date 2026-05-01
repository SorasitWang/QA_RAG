"""
Flask web application — two-panel RAG interface.
  Left  : chat view (LLM conversation with RAG context)
  Right : debug view (retrieved documents + scores for the last turn)

Run:
    python3 app.py
Then open http://localhost:5000 in a browser.
"""

import os

import chromadb
from flask import Flask, jsonify, render_template, request

from Config import Config
from Controller.LLMController import LLMController
from DatabaseRepository.TfidfRepository import TfidfRepository
from DatabaseRepository.VectorRepository import VectorRepository
from DatabaseService.RagService import RagService
from Dto.RetrievalResult import RetrievalResult

app = Flask(__name__, template_folder="View/templates", static_folder="View/static")

# ---------------------------------------------------------------------------
# Shared retrieval infrastructure (initialised once at startup)
# ---------------------------------------------------------------------------
_client     = chromadb.PersistentClient(path="./embeddings_db")
_collection = _client.get_collection(
    name="my_docs",
    embedding_function=Config.Model.EmbeddingFunction,
)
_vectorRepo = VectorRepository(_collection)
_tfidfRepo  = TfidfRepository(Config.Model.TfidfDbPath)
_ragService = RagService(_vectorRepo, _tfidfRepo)

# Per-session LLMController instances keyed by client-supplied session ID
_sessions: dict[str, LLMController] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _formatResults(result: RetrievalResult, mode: str) -> list[dict]:
    """Convert a RetrievalResult into a JSON-serialisable list for the API response."""
    output: list[dict] = []
    for i, (docId, doc, dist, meta) in enumerate(
        zip(result.ids, result.documents, result.distances, result.metadatas)
    ):
        sim = (1.0 - dist / 2.0) if mode == "vector" else (1.0 - dist)
        sim = round(max(0.0, min(1.0, sim)), 4)
        title = os.path.basename(meta.get("title") or docId)
        entry: dict = {"rank": i + 1, "score": sim, "title": title, "content": doc}

        if result.vectorDistances is not None:
            vd = result.vectorDistances[i]
            entry["vectorScore"] = round(max(0.0, min(1.0, 1.0 - vd / 2.0)), 4) if vd is not None else None
        if result.tfidfDistances is not None:
            td = result.tfidfDistances[i]
            entry["tfidfScore"] = round(max(0.0, min(1.0, 1.0 - td)), 4) if td is not None else None

        output.append(entry)

    return output


def _validateRequest(body: dict) -> tuple[str, str, int, str | None]:
    """Extract and validate common fields. Returns (query, mode, topK, error)."""
    query = (body.get("query") or "").strip()
    mode  = body.get("mode", "vector")
    topK  = body.get("topK", 5)

    if not query:
        return "", "", 0, "query is required"
    if mode not in ("vector", "sparse", "hybrid"):
        return "", "", 0, "mode must be vector, sparse, or hybrid"
    try:
        topK = int(topK)
        if not (1 <= topK <= 50):
            raise ValueError
    except (ValueError, TypeError):
        return "", "", 0, "topK must be an integer between 1 and 50"

    return query, mode, topK, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    """
    Stand-alone document search (right panel can also call this directly).

    POST body  : {query, mode?, topK?}
    Response   : {query, mode, results: [{rank, score, title, content}]}
    """
    body = request.get_json(silent=True) or {}
    query, mode, topK, err = _validateRequest(body)
    if err:
        return jsonify({"error": err}), 400

    retrieved: RetrievalResult = _ragService.query(query, topK=topK, searchMode=mode)
    results = _formatResults(retrieved, mode)
    return jsonify({"query": query, "mode": mode, "results": results})


@app.route("/api/chat", methods=["POST"])
def chatEndpoint():
    """
    One turn of RAG-augmented conversation.

    POST body:
        sessionId : str  — client-generated ID; a new LLMController is created
                          on first use and reused for the life of the session.
        query     : str  — user message
        mode      : str  — 'vector' | 'sparse' | 'hybrid'
        topK      : int  — number of chunks to retrieve

    Response:
        reply   : str              — LLM response text
        results : [{rank, score, title, content}]  — retrieved chunks
        query   : str
        mode    : str
    """
    body = request.get_json(silent=True) or {}
    query, mode, topK, err = _validateRequest(body)
    if err:
        return jsonify({"error": err}), 400

    sessionId = (body.get("sessionId") or "default").strip()

    if sessionId not in _sessions:
        _sessions[sessionId] = LLMController()
    controller = _sessions[sessionId]

    retrieved: RetrievalResult = _ragService.query(query, topK=topK, searchMode=mode)

    # Fallback: no chunks on the first try — rewrite using history and retry once.
    rewrittenQuery: str | None = None
    if retrieved.isEmpty() and controller.exchanges:
        candidate = controller.rewriteQuery(query)
        if candidate.strip() and candidate.strip() != query:
            rewrittenQuery = candidate.strip()
            retrieved = _ragService.query(rewrittenQuery, topK=topK, searchMode=mode)

    results  = _formatResults(retrieved, mode)
    reply    = controller.generateResponse(query, retrieved.documents, retrieved.distances)

    response: dict = {
        "reply":   reply,
        "results": results,
        "query":   query,
        "mode":    mode,
        "promptLog": {
            "system":    controller.systemMessage,
            "user":      controller.generateQAPrompt(retrieved.documents, query),
            "assistant": reply,
            "compacted":        controller.lastCompacted,
            "compactedSummary": controller.lastCompactedSummary,
        },
    }
    if rewrittenQuery:
        response["rewrittenQuery"] = rewrittenQuery
    return jsonify(response)


@app.route("/api/chat/reset", methods=["POST"])
def chatReset():
    """Clear the conversation history for a given session."""
    body      = request.get_json(silent=True) or {}
    sessionId = (body.get("sessionId") or "default").strip()
    _sessions.pop(sessionId, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
