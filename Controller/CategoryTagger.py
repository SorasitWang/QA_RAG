import numpy as np
from sentence_transformers import SentenceTransformer


def precomputeLabelEmbeddings(
    model: SentenceTransformer,
    labels: list[str],
) -> np.ndarray:
    """Encodes labels and returns L2-normalised (N, D) matrix. Called once before the chunk loop."""
    raw = model.encode(labels)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return raw / norms


def assignTags(
    chunkEmbedding: list[float] | np.ndarray,
    labelEmbeddings: np.ndarray,
    labels: list[str],
    topN: int,
) -> list[str]:
    """Returns the topN category labels most similar (cosine) to chunkEmbedding."""
    vec = np.array(chunkEmbedding, dtype=float)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    similarities = labelEmbeddings @ vec
    topIndices = np.argsort(similarities)[-topN:][::-1]
    return [labels[i] for i in topIndices]
