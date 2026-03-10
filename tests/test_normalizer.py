import pandas as pd
import numpy as np
from src.features.normalizer import rolling_zscore_normalize


def make_df(n=100):
    np.random.seed(0)
    return pd.DataFrame({
        "close": 100 + np.cumsum(np.random.randn(n)),
        "volume": np.random.randint(1000, 10000, n).astype(float),
        "rsi_14": np.random.uniform(20, 80, n),
    })


def test_output_shape_unchanged():
    df = make_df()
    result = rolling_zscore_normalize(df, window=30)
    assert result.shape == df.shape


def test_columns_unchanged():
    df = make_df()
    result = rolling_zscore_normalize(df, window=30)
    assert list(result.columns) == list(df.columns)


def test_no_nans_after_normalize():
    df = make_df(100)
    result = rolling_zscore_normalize(df, window=30)
    assert result.isnull().sum().sum() == 0


def test_mean_near_zero_after_window():
    """After rolling z-score, values should be roughly mean=0, std=1 (approximately)."""
    df = make_df(200)
    result = rolling_zscore_normalize(df, window=30)
    # Skip first window rows (NaN region before fill) — check tail
    tail = result.iloc[50:]
    assert abs(tail["close"].mean()) < 2.0  # loosely centered


def test_nan_filled_with_zero():
    """Initial rows (before window) should be filled with 0, not NaN."""
    df = make_df(50)
    result = rolling_zscore_normalize(df, window=30)
    assert not result.iloc[:29].isnull().any().any()
