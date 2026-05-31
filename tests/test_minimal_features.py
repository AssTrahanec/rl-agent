import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
