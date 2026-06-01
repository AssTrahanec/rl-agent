import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DSR_EXP = Path(__file__).resolve().parent.parent / "dsr_experiment"
if str(_DSR_EXP) not in sys.path:
    sys.path.insert(0, str(_DSR_EXP))

from lib.features.price import add_technical_indicators_minimal  # noqa: E402

EXPECTED_INDICATORS = [
    "ema_26", "macd", "rsi_14", "bb_width", "atr_14", "obv", "stoch_k", "return_1d",
]


def _toy_ohlcv(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.RandomState(0)
    close = 100 + np.cumsum(rng.randn(n))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close,
         "volume": rng.rand(n) * 10},
        index=idx,
    )


def test_minimal_indicators_are_exactly_eight():
    df = add_technical_indicators_minimal(_toy_ohlcv())
    added = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    assert sorted(added) == sorted(EXPECTED_INDICATORS), added


# Integration: the actually-built train parquet must carry exactly the 41 minimal
# feature columns (8 indicators + sentiment_mean + 32 PCA embeddings).
_TRAIN_PARQUET = _DSR_EXP / "data" / "train" / "features.parquet"

EXPECTED_FEATURE_COLUMNS = (
    EXPECTED_INDICATORS
    + ["sentiment_mean"]
    + [f"emb_{i}" for i in range(32)]
)
_PRICE_COLS = {"open", "high", "low", "close", "volume", "raw_close"}


@pytest.mark.skipif(not _TRAIN_PARQUET.exists(), reason="run build_data --build train first")
def test_built_train_features_match_minimal_schema():
    df = pd.read_parquet(_TRAIN_PARQUET)
    feats = [c for c in df.columns if c.lower() not in _PRICE_COLS]
    assert sorted(feats) == sorted(EXPECTED_FEATURE_COLUMNS), feats
    assert len(feats) == 41
