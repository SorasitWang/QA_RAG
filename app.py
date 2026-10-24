"""
Flask web application — two-panel RAG interface.
  Left  : chat view (LLM conversation with RAG context)
  Right : debug view (retrieved documents + scores for the last turn)

Retrieval is delegated to the retrieval service (retrievalApp.py) over HTTP;
this process only holds chat sessions and talks to Ollama.

Run:
    python3 retrievalApp.py   (in another terminal)
    python3 app.py
Then open http://localhost:5000 in a browser.
"""

import os

from flask import Flask, jsonify, render_template, request

from Config import Config
from Controller.LLMController import LLMController
from Controller.RequestValidation import validateSearchRequest, validateSourceTypes
from Controller.RetrievalClient import RetrievalClient
from Dto.RetrievalResult import RetrievalResult

app = Flask(__name__, template_folder="View/templates", static_folder="View/static")

# ---------------------------------------------------------------------------
# Retrieval service client (indexes live in the retrieval service)
# ---------------------------------------------------------------------------
_retrievalClient = RetrievalClient()

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
        # Feed items are keyed by URL, so show their headline instead of the URL's last segment.
        title = meta.get("headline") or os.path.basename(meta.get("title") or docId)
        entry: dict = {
            "rank":        i + 1,
            "score":       sim,
            "title":       title,
            "content":     doc,
            "sourceType":  meta.get("sourceType", "file"),
            "url":         meta.get("url"),
            "publishedAt": meta.get("publishedAt"),
        }

        if result.vectorDistances is not None:
            vd = result.vectorDistances[i]
            entry["vectorScore"] = round(max(0.0, min(1.0, 1.0 - vd / 2.0)), 4) if vd is not None else None
        if result.tfidfDistances is not None:
            td = result.tfidfDistances[i]
            entry["tfidfScore"] = round(max(0.0, min(1.0, 1.0 - td)), 4) if td is not None else None

        output.append(entry)

    return output


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

    POST body  : {query, mode?, topK?, sourceTypes?}
    Response   : {query, mode, results: [{rank, score, title, content}]}
    """
    body = request.get_json(silent=True) or {}
    query, mode, topK, err = validateSearchRequest(body)
    if not err:
        sourceTypes, err = validateSourceTypes(body)
    if err:
        return jsonify({"error": err}), 400

    retrieved: RetrievalResult = _retrievalClient.query(query, topK=topK, searchMode=mode, sourceTypes=sourceTypes)
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
        mode      : str  — 'vector' | 'sparse' | 'hybrid' | 'graph'
        topK      : int  — number of chunks to retrieve
        sourceTypes : list[str]? — restrict to 'file' | 'news' | 'blog' | 'sns' 

    Response:
        reply   : str              — LLM response text
        results : [{rank, score, title, content}]  — retrieved chunks
        query   : str
        mode    : str
    """
    body = request.get_json(silent=True) or {}
    query, mode, topK, err = validateSearchRequest(body)
    if not err:
        sourceTypes, err = validateSourceTypes(body)
    if err:
        return jsonify({"error": err}), 400

    sessionId = (body.get("sessionId") or "default").strip()

    if sessionId not in _sessions:
        _sessions[sessionId] = LLMController()
    controller = _sessions[sessionId]

    retrieved: RetrievalResult = _retrievalClient.query(query, topK=topK, searchMode=mode, sourceTypes=sourceTypes)

    # Fallback: no chunks on the first try — rewrite using history and retry once.
    rewrittenQuery: str | None = None
    if retrieved.isEmpty() and controller.exchanges:
        candidate = controller.rewriteQuery(query)
        if candidate.strip() and candidate.strip() != query:
            rewrittenQuery = candidate.strip()
            retrieved = _retrievalClient.query(rewrittenQuery, topK=topK, searchMode=mode, sourceTypes=sourceTypes)

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
    app.run(host="0.0.0.0", debug=True, port=Config.Services.WebPort)
