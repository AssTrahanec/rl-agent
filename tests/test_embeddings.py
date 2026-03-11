import numpy as np
import pytest
from src.features.embeddings import compute_embeddings


@pytest.mark.integration
def test_embedding_shape():
    texts = ["Bitcoin surges to all-time high"]
    emb = compute_embeddings(texts)
    assert emb.shape == (384,)


@pytest.mark.integration
def test_embedding_nonzero():
    texts = ["Crypto market crashes hard"]
    emb = compute_embeddings(texts)
    assert np.any(emb != 0)


@pytest.mark.integration
def test_multiple_texts_mean_pooling():
    texts = ["Bitcoin up", "Ethereum down", "Market neutral"]
    emb = compute_embeddings(texts)
    assert emb.shape == (384,)


def test_empty_texts_returns_zeros():
    emb = compute_embeddings([])
    assert emb.shape == (384,)
    assert np.all(emb == 0)


def test_returns_ndarray():
    emb = compute_embeddings([])
    assert isinstance(emb, np.ndarray)
    assert emb.dtype == np.float32 or emb.dtype == np.float64
