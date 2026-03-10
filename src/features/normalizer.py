import pandas as pd
import logging

logger = logging.getLogger(__name__)


def rolling_zscore_normalize(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Apply rolling z-score normalization to all columns.

    For each value: z = (x - rolling_mean) / rolling_std
    NaN values (first `window-1` rows) are filled with 0.

    Input:  DataFrame with numeric columns
    Output: DataFrame of same shape with normalized values
    """
    rolling_mean = df.rolling(window=window, min_periods=1).mean()
    rolling_std = df.rolling(window=window, min_periods=1).std().replace(0, 1)

    normalized = (df - rolling_mean) / rolling_std
    normalized = normalized.fillna(0)

    logger.debug(f"Normalized {len(df.columns)} columns with window={window}")
    return normalized
