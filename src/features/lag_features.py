"""Add lagged and rolling versions of features to capture delayed market reactions."""
from typing import List, Optional

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add time-lagged copies of specified columns.

    Args:
        df: DataFrame with DatetimeIndex.
        columns: Column names to create lags for.
        lags: List of lag periods (default: [1, 2] = yesterday and day before).

    Returns:
        DataFrame with added {col}_lag{n} columns. NaN filled with 0.0.
    """
    if lags is None:
        lags = [1, 2]
    result = df.copy()
    for col in columns:
        for lag in lags:
            result[f"{col}_lag{lag}"] = result[col].shift(lag).fillna(0.0)
    return result


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    window: int = 7,
) -> pd.DataFrame:
    """Add rolling mean of specified columns.

    Args:
        df: DataFrame with DatetimeIndex.
        columns: Column names to compute rolling mean for.
        window: Rolling window size in days (default: 7).

    Returns:
        DataFrame with added {col}_roll{window} columns. NaN filled with 0.0.
    """
    result = df.copy()
    for col in columns:
        result[f"{col}_roll{window}"] = (
            result[col].rolling(window=window, min_periods=window).mean().fillna(0.0)
        )
    return result
