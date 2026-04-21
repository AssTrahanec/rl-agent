"""Run a selected model on each 4h news bucket and produce a decision.

For each bucket we:
1. Fetch OHLCV ending at the bucket's 4h boundary.
2. Build baseline features (tech indicators + z-score) like training.
3. Aggregate news embeddings (signed-weighted, same as build_data.py).
4. Compute sentiment features (max/min/mean/std/spread).
5. Insert into the last row of features matrix.
6. Build obs and run model.predict.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import ensure_lib_on_path
from dashboard.utils.features_live import (
    EMB_DIM, NORMALIZE_WINDOW, NEWS_LAGS, NEWS_ROLL, TOP_PCA_LAGS,
    expected_feature_columns,
)
from dashboard.utils.live_data import fetch_live_ohlcv
from dashboard.utils.model_loader import load_sb3_model, load_compressor
from dashboard.utils.news_live import get_embedder

ensure_lib_on_path()
from lib.data_loader import _PRICE_COLUMNS_EXT


@dataclass(frozen=True)
class BucketDecision:
    action_label: str     # "BUY" / "SELL" / "HOLD" / "LONG X%" / "CASH" etc
    action_color: str     # hex colour for the badge
    allocation: float     # 0..1 target allocation after this bar
    q_values: dict | None # for DQN, {"HOLD":..., "BUY":..., "SELL":...}


def _build_window_features(ohlcv_window: pd.DataFrame,
                           bucket_news: list[dict] | None) -> np.ndarray:
    """Build a single feature row matching training schema for the *last* bar.

    For news, we compute the same signed-weighted aggregation as build_data.py
    and apply the saved PCA compressor.
    """
    from lib.features.price import add_technical_indicators, rolling_zscore_normalize
    from lib.features.lag import add_lag_features, add_rolling_features

    df = add_technical_indicators(ohlcv_window)
    df["raw_close"] = df["close"].copy()
    cols_norm = [c for c in df.columns if c != "raw_close"]
    df[cols_norm] = rolling_zscore_normalize(df[cols_norm], window=NORMALIZE_WINDOW)

    # Embedding features: 64 zeros, filled on the last row if we have news
    for i in range(EMB_DIM):
        df[f"emb_{i}"] = 0.0
    df["news_count"] = 0
    df["sentiment_max"] = 0.0
    df["sentiment_min"] = 0.0
    df["sentiment_mean"] = 0.0
    df["sentiment_std"] = 0.0
    df["sentiment_spread"] = 0.0

    if bucket_news:
        scores = [r["sentiment_score"] for r in bucket_news]
        texts = [f"{r['title']}. {r['summary']}" for r in bucket_news]
        df.iloc[-1, df.columns.get_loc("news_count")] = len(bucket_news)
        df.iloc[-1, df.columns.get_loc("sentiment_mean")] = float(np.mean(scores))
        df.iloc[-1, df.columns.get_loc("sentiment_max")] = float(np.max(scores))
        df.iloc[-1, df.columns.get_loc("sentiment_min")] = float(np.min(scores))
        if len(scores) > 1:
            df.iloc[-1, df.columns.get_loc("sentiment_std")] = float(np.std(scores))
        df.iloc[-1, df.columns.get_loc("sentiment_spread")] = float(np.max(scores) - np.min(scores))

        embedder = get_embedder()
        raw = embedder.encode(texts, show_progress_bar=False)
        # Signed weights (build_data.py logic)
        signed_w = np.array([
            s + np.sign(s) * 0.1 if s != 0 else 0.1 for s in scores
        ], dtype=np.float32)
        denom = np.abs(signed_w).sum()
        if denom > 0:
            signed_w = signed_w / denom
        else:
            signed_w = np.ones_like(signed_w) / len(signed_w)
        agg_emb = (raw * signed_w[:, None]).sum(axis=0).astype(np.float32)

        compressor = load_compressor()
        compressed = compressor.transform(agg_emb.reshape(1, -1))[0]
        for i in range(EMB_DIM):
            df.iloc[-1, df.columns.get_loc(f"emb_{i}")] = float(compressed[i])

    # Lag + rolling same as training
    df = add_lag_features(df, columns=["news_count"], lags=NEWS_LAGS)
    df = add_rolling_features(
        df, columns=["news_count", "sentiment_mean"], window=NEWS_ROLL,
    )
    top_cols = [f"emb_{i}" for i in range(TOP_PCA_LAGS)]
    df = add_lag_features(df, columns=top_cols, lags=NEWS_LAGS)

    # Match training column order
    target_cols = expected_feature_columns()
    feat_df = df.reindex(columns=target_cols + ["raw_close"])
    features = feat_df[target_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


@st.cache_data(ttl=900, show_spinner=False)
def ohlcv_up_to(bucket_ts_iso: str, n_bars: int = 60) -> pd.DataFrame:
    """Get last n_bars OHLCV bars ending at or before `bucket_ts`."""
    df, _src = fetch_live_ohlcv(lookback_bars=n_bars * 2, use_live=True)
    cutoff = pd.Timestamp(bucket_ts_iso)
    if cutoff.tz is None:
        cutoff = cutoff.tz_localize("UTC")
    df = df[df.index <= cutoff]
    if len(df) > n_bars:
        df = df.tail(n_bars)
    return df


def decide(model, algo: str, features: np.ndarray,
           prev_alloc: float = 0.0, window: int = 30) -> BucketDecision:
    if len(features) < window:
        raise ValueError(f"need >= {window} bars, got {len(features)}")
    obs = np.append(features[-window:].flatten(), prev_alloc).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)

    if algo == "DQN":
        a = int(np.asarray(action).flatten()[0]) if not np.isscalar(action) else int(action)
        labels_map = {0: ("HOLD", "#6c757d", prev_alloc),
                      1: ("BUY", "#2ca02c", 1.0),
                      2: ("SELL", "#d62728", 0.0)}
        label, color, alloc = labels_map.get(a, (f"action {a}", "#6c757d", prev_alloc))
        q_dict = None
        try:
            import torch
            obs_t = torch.as_tensor(obs).float().unsqueeze(0)
            with torch.no_grad():
                q = model.q_net(obs_t).numpy().flatten()
            q_dict = {"HOLD": float(q[0]), "BUY": float(q[1]), "SELL": float(q[2])}
        except Exception:  # noqa: BLE001
            pass
        return BucketDecision(label, color, alloc, q_dict)

    # SAC / PPO continuous
    alloc = float(np.clip(np.asarray(action).flatten()[0], 0.0, 1.0))
    if alloc > 0.7:
        label, color = f"LONG {alloc*100:.0f}%", "#2ca02c"
    elif alloc > 0.3:
        label, color = f"PARTIAL {alloc*100:.0f}%", "#ffc107"
    else:
        label, color = "CASH", "#6c757d"
    return BucketDecision(label, color, alloc, None)


def decide_for_buckets(
    buckets: list[dict],
    model_path: str,
    algo: str,
) -> list[dict]:
    """Augment each bucket with an agent decision."""
    model = load_sb3_model(model_path, algo_hint=algo)
    out = []
    prev_alloc = 0.0
    for b in reversed(buckets):  # oldest first so prev_alloc threads correctly
        try:
            bar_ts = b["bucket_ts"].isoformat()
            ohlcv = ohlcv_up_to(bar_ts, n_bars=60)
            if len(ohlcv) < NORMALIZE_WINDOW + 1:
                continue
            feats = _build_window_features(ohlcv, b["items"])
            dec = decide(model, algo, feats, prev_alloc=prev_alloc)
            prev_alloc = dec.allocation
            out.append({**b, "decision": dec})
        except Exception as e:  # noqa: BLE001
            out.append({**b, "decision": None, "error": str(e)})
    out.reverse()
    return out
