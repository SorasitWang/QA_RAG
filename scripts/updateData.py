"""
updateData brings the indexes in line with the source folder in one pass:
new and edited files are (re)indexed, deleted files are removed, and unchanged
files are skipped (idempotent).

The retrieval service does this automatically on startup and while it watches
the folder, so you only need this script when the service is not running or its
watcher is disabled (WATCH_ENABLED=0).

Usage (from the project root):
    python -m scripts.updateData            # index locally, writing the stores directly
    python -m scripts.updateData --remote   # ask the running retrieval service to rescan

Do not run the local mode while the retrieval service is up: both would write
the same stores at the same time.
"""

import argparse
import logging

from Config import Config


def runLocal():
    """Reconcile the source folder against the stores in this process."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    from DatabaseRepository.GraphRepository import GraphRepository
    from DatabaseRepository.IngestManifestRepository import IngestManifestRepository
    from DatabaseRepository.TfidfRepository import TfidfRepository
    from DatabaseRepository.VectorRepository import VectorRepository
    from DatabaseService.IngestionService import IngestionService
    from DatabaseService.RagService import RagService

    client = chromadb.PersistentClient(path=Config.Ingestion.ChromaDbPath)
    collection = client.get_or_create_collection(
        name=Config.Ingestion.CollectionName,
        embedding_function=Config.Model.getEmbeddingFunction(),
    )
    ragService = RagService(
        VectorRepository(collection),
        TfidfRepository(Config.Model.TfidfDbPath),
        GraphRepository(Config.Graph.GraphDbPath),
    )
    # Use the same model as the embedding function to keep vector space consistent
    model = SentenceTransformer(Config.Model.EmbeddingModel)
    service = IngestionService(
        ragService,
        IngestManifestRepository(Config.Ingestion.ManifestDbPath),
        model,
        Config.Ingestion.SourceFolder,
    )
    print(service.reconcile())


def runRemote():
    """Trigger a rescan on the running retrieval service."""
    from Controller.RetrievalClient import RetrievalClient

    client = RetrievalClient()
    client.rescan()
    print(f"Rescan queued on {client.baseUrl}; check {client.baseUrl}/ingest/status for progress.")


def main():
    parser = argparse.ArgumentParser(description="Sync the source folder into the RAG indexes.")
    parser.add_argument("--remote", action="store_true", help="ask the running retrieval service to rescan")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.remote:
        runRemote()
    else:
        runLocal()


if __name__ == "__main__":
    main()
