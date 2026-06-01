"""Compress 768d sentence embeddings to 20d using PCA."""
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class EmbeddingCompressor:
    """PCA-based dimensionality reduction for sentence embeddings (768 -> 20)."""

    def __init__(self, input_dim: int = 768, output_dim: int = 20):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._pca = PCA(n_components=output_dim)

    def fit(self, embeddings: np.ndarray) -> "EmbeddingCompressor":
        """Fit PCA on training embeddings.

        Args:
            embeddings: (N, input_dim) array of embeddings.

        Returns:
            self
        """
        assert embeddings.shape[1] == self.input_dim
        self._pca.fit(embeddings)
        logger.info(
            f"PCA fitted: {self.input_dim}d -> {self.output_dim}d, "
            f"explained variance: {self.explained_variance_ratio():.3f}"
        )
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to compressed representation.

        Args:
            embeddings: (N, input_dim) array.

        Returns:
            (N, output_dim) array.
        """
        return self._pca.transform(embeddings).astype(np.float32)

    def explained_variance_ratio(self) -> float:
        """Total explained variance ratio of the PCA components."""
        return float(np.sum(self._pca.explained_variance_ratio_))

    def save(self, path: str):
        """Save compressor to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved compressor to {path}")

    @staticmethod
    def load(path: str) -> "EmbeddingCompressor":
        """Load compressor from file."""
        with open(path, "rb") as f:
            return pickle.load(f)
