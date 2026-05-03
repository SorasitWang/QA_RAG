"""
Central configuration for the RAG application.
All model names, tuning parameters, and thresholds live here so they can be
changed in one place without touching business logic.
"""

from chromadb.utils import embedding_functions


class Config:

    class Model:
        # --- Embedding ---
        EmbeddingModel    = 'all-MiniLM-L6-v2'
        # Instantiated once at import time; shared across the app to avoid
        # reloading the model on every DB call.
        EmbeddingFunction = embedding_functions.SentenceTransformerEmbeddingFunction(
                                model_name=EmbeddingModel
                            )

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
