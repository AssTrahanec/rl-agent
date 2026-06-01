"""Tests for Agent-4 fusion feature builder."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_fusion_features import build_fusion_features


def make_dummy_price_features(n=50):
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    data = np.random.randn(n, 20)
    cols = [f"feat_{i}" for i in range(20)]
    return pd.DataFrame(data, index=idx, columns=cols)


def make_dummy_news_by_day(n=30):
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "date": idx,
        "texts": [["bitcoin is doing well"] * 2] * n,
    })


def test_build_fusion_features_shape():
    np.random.seed(42)
    price_features = make_dummy_price_features(50)
    news_by_day = make_dummy_news_by_day(30)
    result = build_fusion_features(price_features, news_by_day, use_mock_nlp=True)
    # 20 price + 1 sentiment + 32 embeddings = 53
    assert result.shape[1] == 53


def test_build_fusion_has_sentiment_column():
    np.random.seed(42)
    price_features = make_dummy_price_features(50)
    news_by_day = make_dummy_news_by_day(30)
    result = build_fusion_features(price_features, news_by_day, use_mock_nlp=True)
    assert "sentiment" in result.columns


def test_build_fusion_has_embedding_columns():
    np.random.seed(42)
    price_features = make_dummy_price_features(50)
    news_by_day = make_dummy_news_by_day(30)
    result = build_fusion_features(price_features, news_by_day, use_mock_nlp=True)
    emb_cols = [f"emb_{i}" for i in range(32)]
    for col in emb_cols:
        assert col in result.columns


def test_build_fusion_fallback_zeros_for_missing_days():
    np.random.seed(42)
    price_features = make_dummy_price_features(50)
    # No news at all
    news_by_day = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns, UTC]"), "texts": []})
    result = build_fusion_features(price_features, news_by_day, use_mock_nlp=True)
    assert result["sentiment"].eq(0.0).all()
    emb_cols = [f"emb_{i}" for i in range(32)]
    assert result[emb_cols].eq(0.0).all().all()
