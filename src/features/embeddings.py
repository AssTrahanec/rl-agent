"""Sentence embeddings via FinLang/finance-embeddings-investopedia for news texts."""
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768  # finance-embeddings-investopedia outputs 768d
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        logger.info("Loading FinLang/finance-embeddings-investopedia...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer("FinLang/finance-embeddings-investopedia", device=device)
    return _model


def compute_embeddings(
    texts: List[str],
    model: Optional[object] = None,
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute weighted mean-pooled sentence embedding from list of texts.

    Args:
        texts: List of news article texts.
        model: Optional pre-loaded SentenceTransformer (for testing).
        weights: Optional list of weights per article. If None, uses mean pooling.
                 Weights are normalized to sum=1 internally.

    Returns:
        np.ndarray of shape (768,). Returns zeros if texts is empty.
    """
    if not texts:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    st_model = model if model is not None else _get_model()
    embeddings = st_model.encode(texts, show_progress_bar=False)

    if weights is None or len(weights) != len(texts):
        mean_embedding = np.mean(embeddings, axis=0)
    else:
        w = np.array(weights, dtype=np.float32)
        w_sum = w.sum()
        if w_sum < 1e-8:
            mean_embedding = np.mean(embeddings, axis=0)
        else:
            w = w / w_sum
            mean_embedding = np.average(embeddings, axis=0, weights=w)

    return mean_embedding.astype(np.float32)
