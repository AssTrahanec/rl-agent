import tempfile
from pathlib import Path

import numpy as np
from src.features.embedding_compressor import EmbeddingCompressor


def test_output_shape():
    np.random.seed(42)
    embeddings = np.random.randn(100, 384).astype(np.float32)
    compressor = EmbeddingCompressor(input_dim=384, output_dim=32)
    compressor.fit(embeddings)
    compressed = compressor.transform(embeddings)
    assert compressed.shape == (100, 32)


def test_transform_single_vector():
    np.random.seed(42)
    embeddings = np.random.randn(50, 384).astype(np.float32)
    compressor = EmbeddingCompressor(input_dim=384, output_dim=32)
    compressor.fit(embeddings)
    single = compressor.transform(embeddings[:1])
    assert single.shape == (1, 32)


def test_save_and_load():
    np.random.seed(42)
    embeddings = np.random.randn(50, 384).astype(np.float32)
    compressor = EmbeddingCompressor(input_dim=384, output_dim=32)
    compressor.fit(embeddings)
    original = compressor.transform(embeddings[:5])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "compressor.pkl"
        compressor.save(str(path))
        loaded = EmbeddingCompressor.load(str(path))
        reloaded = loaded.transform(embeddings[:5])

    np.testing.assert_array_almost_equal(original, reloaded)


def test_compressor_768_to_20():
    """EmbeddingCompressor works with 768d input and 20d output (new defaults)."""
    np.random.seed(42)
    data = np.random.randn(200, 768).astype(np.float32)
    compressor = EmbeddingCompressor(input_dim=768, output_dim=20)
    compressor.fit(data)
    result = compressor.transform(data[:5])
    assert result.shape == (5, 20)
    assert compressor.explained_variance_ratio() > 0


def test_default_dims_are_768_20():
    """Default constructor uses 768->20."""
    compressor = EmbeddingCompressor()
    assert compressor.input_dim == 768
    assert compressor.output_dim == 20


def test_explained_variance():
    """PCA compressor should capture reasonable variance."""
    np.random.seed(42)
    embeddings = np.random.randn(200, 384).astype(np.float32)
    compressor = EmbeddingCompressor(input_dim=384, output_dim=32)
    compressor.fit(embeddings)
    assert compressor.explained_variance_ratio() > 0.0
