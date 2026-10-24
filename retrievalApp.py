"""
Retrieval service — owns every index (ChromaDB, TF-IDF, Graph) and the
embedding model, and keeps them in sync with the source folder.

Endpoints:
  POST /query           {query, mode?, topK?, tags?, sourceTypes?} → RetrievalResult as JSON
  POST /ingest          {sources: [{title, content, sourceType?, tags?, optional?}]} → queued
  POST /ingest/delete   {titles: [...]} → queued removal (used to expire feed items)
  POST /ingest/rescan   re-scan the source folder (index new/edited, drop deleted)
  GET  /ingest/status   ingestion worker progress
  GET  /health

On startup it reconciles the source folder with the indexes, then watches the
folder so new, edited and deleted files are re-indexed automatically.

Run:
    python3 retrievalApp.py          (listens on http://localhost:5001)
"""

import logging

from flask import Flask, jsonify, request

from Config import Config
from Controller.RequestValidation import validateSearchRequest, validateSourceTypes
from DatabaseService.IngestionService import IngestionWorker
from DatabaseService.RagService import RagService
from Dto.RetrievalResult import RetrievalResult
from Dto.Source import Source

logger = logging.getLogger(__name__)


def createApp(ragService: RagService, worker: IngestionWorker) -> Flask:
    """Build the Flask app around already-constructed services (lets tests inject mocks)."""
    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"ok": True})

    @app.route("/query", methods=["POST"])
    def query():
        body = request.get_json(silent=True) or {}
        queryText, mode, topK, err = validateSearchRequest(body)
        if err:
            return jsonify({"error": err}), 400
        sourceTypes, err = validateSourceTypes(body)
        if err:
            return jsonify({"error": err}), 400
        tags = body.get("tags") or None

        result: RetrievalResult = ragService.query(
            queryText, tags=tags, topK=topK, searchMode=mode, sourceTypes=sourceTypes
        )
        return jsonify(result.toDict())

    @app.route("/ingest", methods=["POST"])
    def ingest():
        body = request.get_json(silent=True) or {}
        rawSources = body.get("sources")
        if not isinstance(rawSources, list) or not rawSources:
            return jsonify({"error": "sources must be a non-empty list"}), 400
        try:
            sources: list[Source] = [Source.fromDict(s) for s in rawSources]
        except (KeyError, TypeError, AttributeError):
            return jsonify({"error": "each source needs title and content"}), 400
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        for source in sources:
            worker.submit("source", source)
        return jsonify({"queued": len(sources)}), 202

    @app.route("/ingest/delete", methods=["POST"])
    def ingestDelete():
        body = request.get_json(silent=True) or {}
        titles = body.get("titles")
        if not isinstance(titles, list) or not titles or not all(isinstance(t, str) and t for t in titles):
            return jsonify({"error": "titles must be a non-empty list of strings"}), 400
        for title in titles:
            worker.submit("delete", title)
        return jsonify({"queued": len(titles)}), 202

    @app.route("/ingest/rescan", methods=["POST"])
    def rescan():
        worker.submit("reconcile")
        return jsonify({"queued": True}), 202

    @app.route("/ingest/status")
    def ingestStatus():
        return jsonify(worker.snapshot())

    return app


def _buildServices() -> tuple[RagService, IngestionWorker]:
    """Wire the real repositories, embedding model and ingestion worker."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    from DatabaseRepository.GraphRepository import GraphRepository
    from DatabaseRepository.IngestManifestRepository import IngestManifestRepository
    from DatabaseRepository.TfidfRepository import TfidfRepository
    from DatabaseRepository.VectorRepository import VectorRepository
    from DatabaseService.IngestionService import IngestionService

    client = chromadb.PersistentClient(path=Config.Ingestion.ChromaDbPath)
    # get_or_create so a fresh setup starts with an empty index instead of crashing
    collection = client.get_or_create_collection(
        name=Config.Ingestion.CollectionName,
        embedding_function=Config.Model.getEmbeddingFunction(),
    )
    vectorRepo = VectorRepository(collection)
    repaired = vectorRepo.backfillLegacyMetadata()
    if repaired:
        logger.info("Repaired legacy metadata (sourceType/title) on %d existing chunks", repaired)
    ragService = RagService(
        vectorRepo,
        TfidfRepository(Config.Model.TfidfDbPath),
        GraphRepository(Config.Graph.GraphDbPath),
    )
    # Same model as the embedding function to keep the vector space consistent
    model = SentenceTransformer(Config.Model.EmbeddingModel)
    ingestionService = IngestionService(
        ragService,
        IngestManifestRepository(Config.Ingestion.ManifestDbPath),
        model,
        Config.Ingestion.SourceFolder,
    )
    return ragService, IngestionWorker(ingestionService)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ragService, worker = _buildServices()
    worker.start()

    if Config.Ingestion.WatchEnabled:
        from Controller.FolderWatcher import FolderWatcher

        # Catch up on changes made while the service was down, then follow live changes.
        worker.submit("reconcile")
        watcher = FolderWatcher(Config.Ingestion.SourceFolder, lambda path: worker.submit("file", path))
        watcher.start()

    app = createApp(ragService, worker)
    # use_reloader=False: the reloader would fork a second process with a second watcher.
    app.run(host="0.0.0.0", port=Config.Services.RetrievalPort, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
