import numpy as np
import pytest
from src.features.embeddings import compute_embeddings


@pytest.mark.integration
def test_embedding_shape():
    texts = ["Bitcoin surges to all-time high"]
    emb = compute_embeddings(texts)
    assert emb.shape == (768,)


@pytest.mark.integration
def test_embedding_nonzero():
    texts = ["Crypto market crashes hard"]
    emb = compute_embeddings(texts)
    assert np.any(emb != 0)


@pytest.mark.integration
def test_multiple_texts_mean_pooling():
    texts = ["Bitcoin up", "Ethereum down", "Market neutral"]
    emb = compute_embeddings(texts)
    assert emb.shape == (768,)


def test_empty_texts_returns_zeros():
    emb = compute_embeddings([])
    assert emb.shape == (768,)
    assert np.all(emb == 0)


def test_returns_ndarray():
    emb = compute_embeddings([])
    assert isinstance(emb, np.ndarray)
    assert emb.dtype == np.float32 or emb.dtype == np.float64


def test_weighted_pooling_differs_from_mean():
    """Weighted pooling with unequal weights differs from mean pooling."""
    class MockModel:
        def encode(self, texts, show_progress_bar=False):
            return np.array([
                [1.0, 0.0, 0.0] + [0.0] * 765,
                [0.0, 1.0, 0.0] + [0.0] * 765,
            ], dtype=np.float32)

    weights = [0.9, 0.1]
    result = compute_embeddings(["text1", "text2"], model=MockModel(), weights=weights)
    mean_result = compute_embeddings(["text1", "text2"], model=MockModel())
    assert result[0] > mean_result[0], "Weighted pooling should differ from mean"


def test_weighted_pooling_equal_weights_matches_mean():
    """Equal weights produce same result as mean pooling."""
    class MockModel:
        def encode(self, texts, show_progress_bar=False):
            return np.array([
                [1.0, 0.0] + [0.0] * 766,
                [0.0, 1.0] + [0.0] * 766,
            ], dtype=np.float32)

    weights = [0.5, 0.5]
    result_weighted = compute_embeddings(["t1", "t2"], model=MockModel(), weights=weights)
    result_mean = compute_embeddings(["t1", "t2"], model=MockModel())
    np.testing.assert_allclose(result_weighted, result_mean, atol=1e-6)


def test_empty_weights_falls_back_to_mean():
    """None weights falls back to mean pooling."""
    class MockModel:
        def encode(self, texts, show_progress_bar=False):
            return np.ones((len(texts), 768), dtype=np.float32)

    result = compute_embeddings(["text"], model=MockModel(), weights=None)
    assert result.shape == (768,)
