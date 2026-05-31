"""Load train / OOS feature parquet files."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRICE_COLUMNS_EXT = {"open", "high", "low", "close", "volume", "raw_close"}

# News-related feature column prefixes (used for ablation: --no-news).
# Columns whose name starts with any of these are considered news-derived
# and can be excluded from the feature matrix when exclude_news=True.
_NEWS_COLUMN_PREFIXES = ("sentiment_", "news_count", "emb_")


def _is_news_column(col: str) -> bool:
    """True if column name matches any news-feature prefix (case-insensitive)."""
    name = col.lower()
    return any(name.startswith(p) for p in _NEWS_COLUMN_PREFIXES)


def _load_features_parquet(
    path: str,
    period_start: str,
    period_end: str,
    exclude_news: bool = False,
):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Feature file not found: {p}\n"
            f"Run: python build_data.py --build <key>"
        )
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{p}: expected DatetimeIndex")
    start_ts = pd.Timestamp(period_start, tz="UTC") if df.index.tz else pd.Timestamp(period_start)
    end_ts = pd.Timestamp(period_end, tz="UTC") if df.index.tz else pd.Timestamp(period_end)
    df = df.loc[start_ts:end_ts]
    if df.empty:
        raise ValueError(f"{p}: empty for {period_start}..{period_end}")

    if "raw_close" in df.columns:
        prices = df["raw_close"].to_numpy(dtype=np.float64)
    else:
        prices = df["close"].to_numpy(dtype=np.float64)

    feature_cols = [c for c in df.columns if c.lower() not in _PRICE_COLUMNS_EXT]
    if exclude_news:
        dropped = [c for c in feature_cols if _is_news_column(c)]
        feature_cols = [c for c in feature_cols if not _is_news_column(c)]
        if dropped:
            logger.info(
                f"exclude_news=True: dropped {len(dropped)} news columns "
                f"(first: {dropped[:3]}{'...' if len(dropped) > 3 else ''})"
            )
    if not feature_cols:
        raise ValueError(f"{p}: no feature columns")

    features = df[feature_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if exclude_news:
        # Sentiment array is used as a separate signal in the reward (sentiment_lambda
        # bonus). For a clean no-news ablation we zero it out so the reward function
        # also loses access to news information.
        sentiment = np.zeros(len(df), dtype=np.float32)
    else:
        sentiment = (
            df["sentiment_mean"].to_numpy(dtype=np.float32)
            if "sentiment_mean" in df.columns
            else np.zeros(len(df), dtype=np.float32)
        )
        sentiment = np.nan_to_num(sentiment, nan=0.0)

    logger.info(
        f"Loaded {len(df)} rows × {features.shape[1]} features from {p.name}"
        f"{' [no-news]' if exclude_news else ''}"
    )
    return features, prices, sentiment


def load_train(cfg, exclude_news: bool = False) -> tuple:
    """Load train feature matrix + prices + sentiment from cfg.data.paths.train_features.

    Args:
        cfg: Config.
        exclude_news: If True, drop sentiment_*/news_count*/emb_* columns and zero out
            the sentiment array. Used for ablation: training/evaluating without news.
    """
    train = cfg.periods["train"]
    return _load_features_parquet(
        cfg.data.paths.train_features, train.start, train.end, exclude_news=exclude_news
    )


def load_oos(cfg, period_key: str, exclude_news: bool = False) -> tuple:
    """Load OOS feature matrix + prices + sentiment for a given period key.

    Args:
        cfg: Config.
        period_key: Key into cfg.periods (e.g. "oos_2024").
        exclude_news: If True, drop sentiment_*/news_count*/emb_* columns and zero out
            the sentiment array. Used for ablation: training/evaluating without news.
    """
    if period_key not in cfg.periods:
        raise ValueError(f"Unknown period: '{period_key}'")
    period = cfg.periods[period_key]
    path = cfg.oos_features_path(period_key)
    return _load_features_parquet(path, period.start, period.end, exclude_news=exclude_news)
