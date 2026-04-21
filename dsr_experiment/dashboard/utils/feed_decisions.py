"""Run all seeds of a selected model on each 4h news bucket.

For each bucket we:
1. Fetch OHLCV up to the bucket's 4h boundary.
2. Build features (tech + zscore + signed-weighted embedding + sentiment).
3. Run EVERY seed of the selected model on this obs.
4. Aggregate results into an ensemble decision:
   - per-seed allocation
   - mean allocation
   - vote counts (for DQN)
5. Thread prev_allocation through buckets (oldest first).
"""
from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class EnsembleDecision:
    # Aggregated allocation (mean across seeds) in [0, 1]
    allocation: float
    # Previous allocation for reference (comes from threading)
    prev_allocation: float
    # Action direction vs previous position: "increase" / "decrease" / "hold"
    direction: str
    # Per-seed votes for DQN ("BUY" / "SELL" / "HOLD") or None for SAC/PPO
    votes: Optional[dict]        # {"BUY": 7, "HOLD": 2, "SELL": 1}
    total_seeds: int
    # Realized P&L on next bar if available
    next_bar_return: Optional[float]  # log-return to next close
    trade_pnl: Optional[float]        # allocation * next_bar_return (as daily return)


def _build_window_features(ohlcv_window: pd.DataFrame,
                           bucket_news: list[dict] | None) -> np.ndarray:
    from lib.features.price import add_technical_indicators, rolling_zscore_normalize
    from lib.features.lag import add_lag_features, add_rolling_features
    from lib.data_loader import _PRICE_COLUMNS_EXT

    df = add_technical_indicators(ohlcv_window)
    df["raw_close"] = df["close"].copy()
    cols_norm = [c for c in df.columns if c != "raw_close"]
    df[cols_norm] = rolling_zscore_normalize(df[cols_norm], window=NORMALIZE_WINDOW)

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

    df = add_lag_features(df, columns=["news_count"], lags=NEWS_LAGS)
    df = add_rolling_features(
        df, columns=["news_count", "sentiment_mean"], window=NEWS_ROLL,
    )
    top_cols = [f"emb_{i}" for i in range(TOP_PCA_LAGS)]
    df = add_lag_features(df, columns=top_cols, lags=NEWS_LAGS)

    target_cols = expected_feature_columns()
    feat_df = df.reindex(columns=target_cols + ["raw_close"])
    features = feat_df[target_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_ohlcv_cached() -> pd.DataFrame:
    df, _ = fetch_live_ohlcv(lookback_bars=300, use_live=True)
    return df


def _ensemble_on_obs(obs: np.ndarray, seeds_paths: list[tuple[int, str]],
                     algo: str, prev_alloc: float) -> dict:
    """Run every seed's model on the same obs, aggregate."""
    allocs = []
    labels = []  # For DQN: BUY/SELL/HOLD per seed
    for seed, path in seeds_paths:
        try:
            model = load_sb3_model(path, algo_hint=algo)
        except Exception:  # noqa: BLE001
            continue
        action, _ = model.predict(obs, deterministic=True)
        if algo == "DQN":
            a = int(np.asarray(action).flatten()[0]) if not np.isscalar(action) else int(action)
            if a == 1:
                allocs.append(1.0); labels.append("BUY")
            elif a == 2:
                allocs.append(0.0); labels.append("SELL")
            else:
                allocs.append(prev_alloc); labels.append("HOLD")
        else:
            alloc = float(np.clip(np.asarray(action).flatten()[0], 0.0, 1.0))
            allocs.append(alloc)
            labels.append(None)

    if not allocs:
        return {"allocation": prev_alloc, "votes": None, "total_seeds": 0}

    out = {
        "allocation": float(np.mean(allocs)),
        "total_seeds": len(allocs),
    }
    if algo == "DQN":
        votes = {"BUY": labels.count("BUY"),
                 "HOLD": labels.count("HOLD"),
                 "SELL": labels.count("SELL")}
        out["votes"] = votes
    else:
        out["votes"] = None
    return out


def _direction(prev: float, now: float) -> str:
    if now > prev + 0.05:
        return "increase"
    if now < prev - 0.05:
        return "decrease"
    return "hold"


def decide_ensemble_for_buckets(
    buckets: list[dict],
    seeds_paths: list[tuple[int, str]],
    algo: str,
) -> list[dict]:
    """For each bucket run all seeds, thread prev_alloc oldest->newest."""
    ohlcv_all = _fetch_ohlcv_cached()
    out = []
    prev_alloc = 0.0
    # Oldest first for prev_alloc threading
    for b in reversed(buckets):
        bucket_ts = b["bucket_ts"]
        cutoff = bucket_ts
        if cutoff.tz is None:
            cutoff = cutoff.tz_localize("UTC")

        ohlcv = ohlcv_all[ohlcv_all.index <= cutoff].tail(60)
        if len(ohlcv) < NORMALIZE_WINDOW + 2:
            out.append({**b, "decision": None,
                        "error": f"insufficient OHLCV bars ({len(ohlcv)})"})
            continue

        try:
            features = _build_window_features(ohlcv, b["items"])
            obs = np.append(features[-30:].flatten(), prev_alloc).astype(np.float32)
            agg = _ensemble_on_obs(obs, seeds_paths, algo, prev_alloc)

            new_alloc = agg["allocation"]
            direction = _direction(prev_alloc, new_alloc)

            # P&L: if we have bar AFTER bucket_ts — use it
            after_bars = ohlcv_all[ohlcv_all.index > cutoff]
            next_bar_return = None
            trade_pnl = None
            if not after_bars.empty and len(ohlcv) >= 1:
                last_close = float(ohlcv["close"].iloc[-1])
                next_close = float(after_bars.iloc[0]["close"])
                if last_close > 0:
                    next_bar_return = float(np.log(next_close / last_close))
                    trade_pnl = float(np.exp(new_alloc * next_bar_return) - 1)

            dec = EnsembleDecision(
                allocation=new_alloc,
                prev_allocation=prev_alloc,
                direction=direction,
                votes=agg["votes"],
                total_seeds=agg["total_seeds"],
                next_bar_return=next_bar_return,
                trade_pnl=trade_pnl,
            )
            out.append({**b, "decision": dec})
            prev_alloc = new_alloc
        except Exception as e:  # noqa: BLE001
            out.append({**b, "decision": None, "error": str(e)})

    out.reverse()
    return out


def compute_portfolio_trajectory(decorated: list[dict],
                                 initial_capital: float = 10000.0) -> list[dict]:
    """Compound portfolio value across all buckets from oldest to newest.

    Returns a copy of decorated with 'portfolio_value' added per bucket.
    """
    # Walk oldest -> newest
    ordered = list(reversed(decorated))
    equity = initial_capital
    for item in ordered:
        dec = item.get("decision")
        if dec is not None and dec.trade_pnl is not None:
            equity = equity * (1 + dec.trade_pnl)
        item["portfolio_value"] = equity
    # Return newest-first
    return list(reversed(ordered))
