"""Sentence embeddings (FinLang/finance-embeddings-investopedia, 768d) + PCA compressor."""
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768
_MODEL_NAME = "FinLang/finance-embeddings-investopedia"

_model = None


def _get_model():
    """Lazy singleton for the FinLang sentence-transformer (loaded once)."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SentenceTransformer({_MODEL_NAME}) on {device}")
        _model = SentenceTransformer(_MODEL_NAME, device=device)
    return _model


def compute_embeddings(texts: List[str], weights: Optional[List[float]] = None) -> np.ndarray:
    """Mean-pooled FinLang embedding, optionally sentiment-weighted (shape: (EMBEDDING_DIM,))."""
    if not texts:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    embeddings = _get_model().encode(texts, show_progress_bar=False)
    if weights is None or len(weights) != len(texts):
        return np.mean(embeddings, axis=0).astype(np.float32)
    w = np.array(weights, dtype=np.float32)
    w_sum = w.sum()
    if w_sum < 1e-8:
        return np.mean(embeddings, axis=0).astype(np.float32)
    return np.average(embeddings, axis=0, weights=w / w_sum).astype(np.float32)


class EmbeddingCompressor:
    """PCA-based compressor: input_dim -> output_dim."""

    def __init__(self, input_dim: int = 768, output_dim: int = 64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._pca = PCA(n_components=output_dim)

    def fit(self, embeddings: np.ndarray) -> "EmbeddingCompressor":
        assert embeddings.shape[1] == self.input_dim, (
            f"Expected {self.input_dim}d embeddings, got {embeddings.shape[1]}d"
        )
        self._pca.fit(embeddings)
        logger.info(
            f"PCA fitted: {self.input_dim} -> {self.output_dim}, "
            f"explained var: {self.explained_variance_ratio():.3f}"
        )
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self._pca.transform(embeddings).astype(np.float32)

    def explained_variance_ratio(self) -> float:
        return float(np.sum(self._pca.explained_variance_ratio_))

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved compressor to {path}")

    @staticmethod
    def load(path: str) -> "EmbeddingCompressor":
        with open(path, "rb") as f:
            return pickle.load(f)
