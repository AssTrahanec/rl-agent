"""Unit tests for src.data.load_features."""
import numpy as np
import pandas as pd
import pytest

from src.data.load_features import load_features_for_agent, _parquet_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 60, n_extra_features: int = 0) -> pd.DataFrame:
    """Return a DataFrame that mimics a processed parquet file."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    data = {
        "open":   np.random.rand(n) * 100 + 50,
        "high":   np.random.rand(n) * 100 + 60,
        "low":    np.random.rand(n) * 100 + 40,
        "close":  np.random.rand(n) * 100 + 50,
        "volume": np.random.rand(n) * 1e6,
    }
    for i in range(n_extra_features):
        data[f"feat_{i}"] = np.random.randn(n).astype(np.float32)
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# _parquet_path
# ---------------------------------------------------------------------------

def test_parquet_path_baseline():
    p = _parquet_path("baseline", "BTC/USDT", "data/processed")
    assert p.name == "btc_features.parquet"


def test_parquet_path_sentiment_eth():
    p = _parquet_path("sentiment", "ETH/USDT", "data/processed")
    assert p.name == "eth_sentiment_features.parquet"


def test_parquet_path_embeddings():
    p = _parquet_path("embeddings", "BTC/USDT", "data/processed")
    assert p.name == "btc_embedding_features.parquet"


def test_parquet_path_unknown_asset():
    with pytest.raises(ValueError, match="Unknown asset"):
        _parquet_path("baseline", "XYZ/USDT", "data/processed")


def test_parquet_path_unknown_agent():
    with pytest.raises(ValueError, match="Unknown agent_type"):
        _parquet_path("mystery", "BTC/USDT", "data/processed")


# ---------------------------------------------------------------------------
# load_features_for_agent — file-not-found
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Feature file not found"):
        load_features_for_agent(
            agent_type="baseline",
            asset="BTC/USDT",
            train_start="2020-01-01",
            train_end="2020-12-31",
            data_dir=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# load_features_for_agent — happy path
# ---------------------------------------------------------------------------

def test_load_baseline_returns_correct_shapes(tmp_path):
    df = _make_price_df(n=60, n_extra_features=15)
    path = tmp_path / "btc_features.parquet"
    df.to_parquet(path)

    features, prices = load_features_for_agent(
        agent_type="baseline",
        asset="BTC/USDT",
        train_start="2020-01-01",
        train_end="2020-02-29",
        data_dir=str(tmp_path),
    )

    assert features.ndim == 2
    assert prices.ndim == 1
    assert len(features) == len(prices)
    # feature cols = only the feat_* columns (OHLCV stripped)
    assert features.shape[1] == 15


def test_load_strips_ohlcv_columns(tmp_path):
    df = _make_price_df(n=60, n_extra_features=5)
    path = tmp_path / "btc_features.parquet"
    df.to_parquet(path)

    features, prices = load_features_for_agent(
        agent_type="baseline",
        asset="BTC/USDT",
        train_start="2020-01-01",
        train_end="2020-03-01",
        data_dir=str(tmp_path),
    )
    # prices = close column
    np.testing.assert_allclose(prices, df.loc["2020-01-01":"2020-03-01", "close"].values)
    # features = only feat_* (5 cols)
    assert features.shape[1] == 5


def test_load_nan_replaced_with_zero(tmp_path):
    df = _make_price_df(n=60, n_extra_features=3)
    # Inject NaN
    df.iloc[5, df.columns.get_loc("feat_0")] = float("nan")
    path = tmp_path / "btc_features.parquet"
    df.to_parquet(path)

    features, _ = load_features_for_agent(
        agent_type="baseline",
        asset="BTC/USDT",
        train_start="2020-01-01",
        train_end="2020-03-01",
        data_dir=str(tmp_path),
    )
    assert not np.isnan(features).any()


def test_load_empty_slice_raises(tmp_path):
    df = _make_price_df(n=60, n_extra_features=5)
    path = tmp_path / "btc_features.parquet"
    df.to_parquet(path)

    with pytest.raises(ValueError, match="No data"):
        load_features_for_agent(
            agent_type="baseline",
            asset="BTC/USDT",
            train_start="2025-01-01",  # out of range
            train_end="2025-12-31",
            data_dir=str(tmp_path),
        )
