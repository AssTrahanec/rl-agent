"""Sentence embeddings via all-MiniLM-L6-v2 for news texts."""
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading all-MiniLM-L6-v2 sentence-transformer...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_embeddings(
    texts: List[str],
    model: Optional[object] = None,
) -> np.ndarray:
    """Compute mean-pooled sentence embedding from list of texts.

    Args:
        texts: List of news article texts.
        model: Optional pre-loaded SentenceTransformer (for testing).

    Returns:
        np.ndarray of shape (384,). Returns zeros if texts is empty.
    """
    if not texts:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    st_model = model if model is not None else _get_model()
    embeddings = st_model.encode(texts, show_progress_bar=False)
    # Mean pooling across all texts for this day
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding.astype(np.float32)
