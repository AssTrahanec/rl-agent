"""Build raw + processed datasets.

Examples:
    python build_data.py --build train
    python build_data.py --build oos_2025
    python build_data.py --build all
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lib.config_loader import load_config, Config
from lib.features.price import (
    fetch_ohlcv, add_technical_indicators_minimal, rolling_zscore_normalize,
)
from lib.features.news import load_news_from_hf, preprocess_news_4h, deduplicate_embeddings
from lib.features.sentiment import compute_sentiment_scores
from lib.features.embeddings import compute_embeddings, EmbeddingCompressor, _get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _full_date_span(cfg: Config) -> tuple:
    """Earliest start, latest end across ALL periods."""
    starts = [pd.Timestamp(p.start) for p in cfg.periods.values()]
    ends = [pd.Timestamp(p.end) for p in cfg.periods.values()]
    return min(starts).strftime("%Y-%m-%d"), max(ends).strftime("%Y-%m-%d")


def ensure_raw_ohlcv(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.data.paths.raw_ohlcv)
    if path.exists():
        logger.info(f"Using cached OHLCV: {path}")
        return pd.read_parquet(path)
    start, end = _full_date_span(cfg)
    logger.info(f"Downloading OHLCV {cfg.data.asset} {cfg.data.timeframe} {start}..{end}")
    df = fetch_ohlcv(cfg.data.asset, start, end, timeframe=cfg.data.timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} candles to {path}")
    return df


def ensure_raw_news(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.data.paths.raw_news)
    if path.exists():
        logger.info(f"Using cached news: {path}")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df
    start, end = _full_date_span(cfg)
    logger.info(f"Downloading news {start}..{end}")
    df = load_news_from_hf(
        cfg.news.hf_dataset, start, end,
        text_col=cfg.news.text_col, date_col=cfg.news.date_col,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} articles to {path}")
    return df


def _slice_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        s = pd.Timestamp(start, tz="UTC") if df.index.tz else pd.Timestamp(start)
        e = pd.Timestamp(end, tz="UTC") if df.index.tz else pd.Timestamp(end)
        return df.loc[s:e]
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    return df[(df["date"] >= s) & (df["date"] <= e)]


def _build_baseline(ohlcv_slice: pd.DataFrame, normalize_window: int) -> pd.DataFrame:
    df = add_technical_indicators_minimal(ohlcv_slice)
    df["raw_close"] = df["close"].copy()
    cols = [c for c in df.columns if c != "raw_close"]
    df[cols] = rolling_zscore_normalize(df[cols], window=normalize_window)
    return df



def _attach_nlp_features_minimal(
    baseline: pd.DataFrame,
    news_slice: pd.DataFrame,
    cfg: Config,
    compressor: EmbeddingCompressor,
) -> pd.DataFrame:
    """Минимальная NLP-обработка: одна скалярная тональность + PCA-эмбеддинги.

    Сохраняет ключевые элементы методики работы — FinBERT-тональность
    и FinLang-эмбеддинги, — но удаляет избыточные признаки:
        - 5 sentiment-агрегатов (max, min, std, spread, news_count) → 1 (mean);
        - лаги (news_count_lag_1/2, emb_0/1/2_lag_1/2) → нет;
        - rolling-средние (news_count_rolling_7, sentiment_mean_rolling_7) → нет.

    Соответствует подходу FinRL (Liu et al., 2021) к sentiment + рекомендациям
    Delft TU (2025) о вреде избыточных признаков в DRL для трейдинга.

    Не используется в основном pipeline; оставлена для документации
    и сравнения с расширенной версией `_attach_nlp_features`.

    На вход:
        baseline    — таблица с ценовыми признаками;
        news_slice  — корпус новостей для периода;
        cfg         — конфигурация (нужна для размерности эмбеддингов);
        compressor  — обученный PCA-компрессор.

    На выход:
        DataFrame с колонками:
          - sentiment_mean — средняя тональность окна в [-1, +1];
          - emb_0 ... emb_63 — PCA-сжатый смысловой вектор новостей окна.
    """
    result = baseline.copy()
    emb_cols = [f"emb_{i}" for i in range(cfg.embeddings.compressed_dim)]
    for c in emb_cols:
        result[c] = 0.0
    result["sentiment_mean"] = 0.0

    news_4h = preprocess_news_4h(news_slice)
    st_model = _get_model(cfg.embeddings.model_name)

    for _, row in news_4h.iterrows():
        ws = row["date"]
        texts = row["texts"]
        if ws not in result.index:
            continue
        scores = compute_sentiment_scores(texts)
        if not scores:
            continue
        result.loc[ws, "sentiment_mean"] = float(np.mean(scores))

        # Эмбеддинги — взвешенное среднее по тональности, PCA-сжатие
        raw_embs = st_model.encode(texts, show_progress_bar=False)
        signed_w = np.array(
            [s + np.sign(s) * 0.1 if s != 0 else 0.1 for s in scores],
            dtype=np.float32,
        )
        denom = np.abs(signed_w).sum()
        signed_w = signed_w / denom if denom > 0 else np.ones_like(signed_w) / len(signed_w)
        agg_emb = (raw_embs * signed_w[:, None]).sum(axis=0).astype(np.float32)
        compressed = compressor.transform(agg_emb.reshape(1, -1))[0]
        result.loc[ws, emb_cols] = compressed

    return result


def build_train(cfg: Config):
    logger.info("=== BUILD TRAIN ===")
    ohlcv_full = ensure_raw_ohlcv(cfg)
    news_full = ensure_raw_news(cfg)

    train = cfg.periods["train"]
    ohlcv_train = _slice_by_period(ohlcv_full, train.start, train.end)
    news_train = _slice_by_period(news_full, train.start, train.end)

    baseline = _build_baseline(ohlcv_train, cfg.features.normalize_window)

    # Fit compressor on ALL train embeddings
    logger.info("Computing raw train embeddings for PCA fit...")
    news_4h = preprocess_news_4h(news_train)
    st_model = _get_model(cfg.embeddings.model_name)
    all_emb = []
    for idx, row in news_4h.iterrows():
        for text in row["texts"]:
            emb = st_model.encode([text], show_progress_bar=False)[0]
            all_emb.append(emb)
        if (idx + 1) % 500 == 0:
            logger.info(f"  PCA embeddings: {idx + 1}/{len(news_4h)} windows")
    all_emb = np.array(all_emb, dtype=np.float32)
    logger.info(f"  Total: {all_emb.shape}")

    compressor = EmbeddingCompressor(
        input_dim=cfg.embeddings.raw_dim,
        output_dim=cfg.embeddings.compressed_dim,
    )
    compressor.fit(all_emb)
    Path(cfg.data.paths.compressor).parent.mkdir(parents=True, exist_ok=True)
    compressor.save(cfg.data.paths.compressor)

    features = _attach_nlp_features_minimal(baseline, news_train, cfg, compressor)
    Path(cfg.data.paths.train_features).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cfg.data.paths.train_features)
    logger.info(f"Train features saved: {features.shape} -> {cfg.data.paths.train_features}")


def build_oos(cfg: Config, period_key: str):
    logger.info(f"=== BUILD OOS {period_key} ===")
    if not Path(cfg.data.paths.compressor).exists():
        raise FileNotFoundError(
            f"Compressor not found at {cfg.data.paths.compressor}. "
            f"Run: python build_data.py --build train"
        )

    period = cfg.periods[period_key]
    ohlcv_full = ensure_raw_ohlcv(cfg)
    news_full = ensure_raw_news(cfg)
    ohlcv_p = _slice_by_period(ohlcv_full, period.start, period.end)
    news_p = _slice_by_period(news_full, period.start, period.end)

    baseline = _build_baseline(ohlcv_p, cfg.features.normalize_window)
    compressor = EmbeddingCompressor.load(cfg.data.paths.compressor)
    features = _attach_nlp_features_minimal(baseline, news_p, cfg, compressor)

    out = Path(cfg.oos_features_path(period_key))
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out)
    logger.info(f"OOS {period_key}: {features.shape} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--build", required=True,
                    help="'train', 'all', or any oos period key from config.periods")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.build == "train":
        build_train(cfg)
    elif args.build == "all":
        build_train(cfg)
        for k in cfg.experiment.oos_periods:
            build_oos(cfg, k)
    elif args.build in cfg.periods:
        build_oos(cfg, args.build)
    else:
        valid = ["train", "all"] + [k for k in cfg.periods if k != "train"]
        raise SystemExit(f"Unknown --build target: {args.build}. Valid: {valid}")


if __name__ == "__main__":
    main()
