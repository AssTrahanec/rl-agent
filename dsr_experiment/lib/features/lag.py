"""Lag and rolling helpers."""
from typing import List, Optional

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
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
    result = df.copy()
    for col in columns:
        result[f"{col}_roll{window}"] = (
            result[col].rolling(window=window, min_periods=window).mean().fillna(0.0)
        )
    return result
