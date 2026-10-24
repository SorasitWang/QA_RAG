"""
Central configuration for the RAG application.
All model names, tuning parameters, and thresholds live here so they can be
changed in one place without touching business logic.
"""

import os

_embeddingFunction = None


def _envBool(name: str, default: bool) -> bool:
    """Read a boolean flag from the environment ('1', 'true', 'yes' are truthy)."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes')


class Config:

    class Model:
        # --- Embedding ---
        EmbeddingModel    = 'all-MiniLM-L6-v2'

        @staticmethod
        def getEmbeddingFunction():
            """
            Return the shared ChromaDB embedding function, creating it on first use.
            Lazy so services that never touch ChromaDB (e.g. the web service) do not
            import chromadb or load the SentenceTransformer model.
            """
            global _embeddingFunction
            if _embeddingFunction is None:
                from chromadb.utils import embedding_functions
                _embeddingFunction = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=Config.Model.EmbeddingModel
                )
            return _embeddingFunction

        # --- Chunking ---
        MaxTextChunkLength = 500   # max characters per chunk fed into the embedder
        OverLapChunkLength = 50    # overlap keeps context across chunk boundaries

        # --- LLM ---
        LLMModel            = 'llama2'
        # Compact history when total accumulated chars exceed this to keep
        # the prompt within a manageable size for the LLM.
        MaxChatHistoryChars = 5000

        # --- Retrieval pipeline ---
        # Stage 1: fetch a wide candidate pool from ChromaDB
        RetrievalTopK       = 15
        # Stage 2: after MMR re-ranking, only this many chunks reach the LLM
        MMRTopK             = 5
        # MMR lambda: weight between relevance (1.0) and diversity (0.0)
        MMRLambda           = 0.3
        # L2 distance ceiling; chunks beyond this are too dissimilar to be useful.
        # Range: 0 (identical) → 2 (opposite) for normalised sentence embeddings.
        SimilarityThreshold = 0.75

        # --- Category tagging ---
        TagTopN: int = 4

        # --- Text cleaning ---
        # Lines with fewer characters than this (after stripping whitespace) are dropped.
        MinLineLength: int = 20

        # --- Sparse retrieval (TF-IDF) ---
        TfidfDbPath: str = './tfidf_db/tfidf.sqlite3'
        # Candidate pool fetched from the TF-IDF index before threshold filtering.
        TfidfTopK: int = 15
        # TF-IDF distance ceiling; chunks beyond this are discarded in sparse/hybrid mode.
        # TF-IDF distance = 1 - normalised_score, so 0.75 keeps results with score >= 0.25.
        TfidfSimilarityThreshold: float = 0.75
        # k constant for Reciprocal Rank Fusion (standard value is 60).
        RRFk: int = 60

    class Graph:
        GraphDbPath: str = './graph_db/graph.sqlite3'
        # BFS hop depth from seed entities during graph search.
        # 1 = direct co-occurrence neighbours only (fast, precise).
        GraphHopDepth: int = 1
        # Candidate chunk pool fetched from the graph before MMR re-ranking.
        GraphTopK: int = 20
        # With a sourceType filter the graph pool is widened by this factor before filtering.
        FilteredPoolMultiplier: int = 5
        # spaCy entity labels to index; numerics and temporals are excluded to reduce noise.
        GraphKeptEntityTypes: frozenset = frozenset({
            "PERSON", "ORG", "GPE", "EVENT", "LAW", "PRODUCT", "WORK_OF_ART"
        })

    class Ingestion:
        # Folder watched by the retrieval service; new/edited/deleted files are re-indexed.
        SourceFolder: str = os.environ.get('SOURCE_FOLDER', './Test/RawTextFiles')
        # Wait for this many seconds of quiet on a file before indexing it,
        # because editors often write a file several times per save.
        DebounceSeconds: float = float(os.environ.get('WATCH_DEBOUNCE_SECONDS', '2'))
        # inotify does not fire for /mnt/c under WSL2 or for Windows bind mounts in
        # Docker; polling works everywhere at the cost of a short delay.
        UsePolling: bool = _envBool('WATCH_POLLING', False)
        PollIntervalSeconds: float = float(os.environ.get('WATCH_POLL_INTERVAL_SECONDS', '5'))
        WatchEnabled: bool = _envBool('WATCH_ENABLED', True)
        # Tracks content hash per indexed file so edits can be detected.
        ManifestDbPath: str = './tfidf_db/manifest.sqlite3'
        ChromaDbPath: str = './embeddings_db'
        CollectionName: str = 'my_docs'

    class Services:
        RetrievalServiceUrl: str = os.environ.get('RETRIEVAL_SERVICE_URL', 'http://localhost:5001')
        RetrievalTimeoutSeconds: float = float(os.environ.get('RETRIEVAL_TIMEOUT_SECONDS', '30'))
        RetrievalPort: int = int(os.environ.get('RETRIEVAL_PORT', '5001'))
        WebPort: int = int(os.environ.get('WEB_PORT', '5000'))

    class Feeds:
        # JSON list of feeds the poller fetches; see feeds.example.json.
        ConfigPath: str = os.environ.get('FEEDS_CONFIG', './feeds.json')
        StateDbPath: str = './feed_db/state.sqlite3'
        PollerPort: int = int(os.environ.get('POLLER_PORT', '5002'))
        DefaultIntervalMinutes: float = 15
        # Items older than this are removed from the index so news does not drown out documents.
        DefaultRetentionDays: float = 30
        # Upper bound for the per-feed exponential backoff after failures.
        MaxBackoffSeconds: float = 3600
        RequestTimeoutSeconds: float = 20
        IngestBatchSize: int = 50
        UserAgent: str = 'rag-feed-poller/1.0'
