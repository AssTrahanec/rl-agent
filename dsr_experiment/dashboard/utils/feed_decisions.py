"""Прогон ансамбля по 4h-окнам новостей.

Каждое окно: считаем фичи (цена + новости) → каждая из 10 сетей даёт ответ →
агрегируем (majority vote для DQN, медиана для SAC).
"""
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import ensure_lib_on_path
from dashboard.utils.features_live import (
    EMB_DIM, NORMALIZE_WINDOW, expected_feature_columns,
)
from dashboard.utils.live_data import fetch_live_ohlcv
from dashboard.utils.model_loader import load_sb3_model, load_compressor
from dashboard.utils.news_live import get_embedder

ensure_lib_on_path()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv() -> pd.DataFrame:
    """Свечи BTC/USDT 4h — Binance live, fallback на закешированный parquet."""
    df, _ = fetch_live_ohlcv(lookback_bars=300, use_live=True)
    return df


def _build_features(ohlcv_window: pd.DataFrame, news_items: list[dict] | None) -> np.ndarray:
    """8 индикаторов + sentiment_mean + сжатый эмбеддинг (схема обучения, 41 фича)."""
    from lib.features.price import add_technical_indicators_minimal, rolling_zscore_normalize

    df = add_technical_indicators_minimal(ohlcv_window)
    df["raw_close"] = df["close"]
    norm_cols = [c for c in df.columns if c != "raw_close"]
    df[norm_cols] = rolling_zscore_normalize(df[norm_cols], window=NORMALIZE_WINDOW)

    # News-фичи по умолчанию нули
    for i in range(EMB_DIM):
        df[f"emb_{i}"] = 0.0
    df["sentiment_mean"] = 0.0

    # Если в окне есть новости — впишем в последнюю строку
    if news_items:
        scores = [r["sentiment_score"] for r in news_items]
        texts = [f"{r['title']}. {r['summary']}" for r in news_items]
        last = df.index[-1]
        df.loc[last, "sentiment_mean"] = float(np.mean(scores))

        # Embedding всех новостей, взвешенный по силе тональности (как в обучении)
        raw = get_embedder().encode(texts, show_progress_bar=False)
        w = np.array([s + np.sign(s) * 0.1 if s != 0 else 0.1 for s in scores], dtype=np.float32)
        w = w / np.abs(w).sum() if np.abs(w).sum() else np.ones_like(w) / len(w)
        agg = (raw * w[:, None]).sum(axis=0)
        compressed = load_compressor().transform(agg.reshape(1, -1))[0]
        for i in range(EMB_DIM):
            df.loc[last, f"emb_{i}"] = float(compressed[i])

    target_cols = expected_feature_columns()
    features = df.reindex(columns=target_cols).to_numpy(dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _run_ensemble(obs: np.ndarray, seeds: list[tuple[int, str]],
                  algo: str, prev_alloc: float) -> dict:
    """Прогон obs через все сети ансамбля.

    DQN: majority vote по голосам BUY/HOLD/SELL → строго 0% или 100% капитала.
    SAC: медиана непрерывных аллокаций по сидам.
    """
    allocs, labels = [], []
    for _seed, path in seeds:
        try:
            model = load_sb3_model(path, algo_hint=algo)
        except Exception:
            continue
        action, _ = model.predict(obs, deterministic=True)
        if algo == "DQN":
            a = int(np.asarray(action).flatten()[0])
            if a == 1:
                allocs.append(1.0); labels.append("BUY")
            elif a == 2:
                allocs.append(0.0); labels.append("SELL")
            else:
                allocs.append(prev_alloc); labels.append("HOLD")
        else:
            allocs.append(float(np.clip(np.asarray(action).flatten()[0], 0.0, 1.0)))

    if not allocs:
        return {"allocation": prev_alloc, "votes": None, "winner": None, "total_seeds": 0}

    if algo == "DQN":
        votes = {"BUY": labels.count("BUY"), "HOLD": labels.count("HOLD"), "SELL": labels.count("SELL")}
        winner = _majority_winner(votes)
        allocation = 1.0 if winner == "BUY" else 0.0 if winner == "SELL" else float(prev_alloc)
        return {"allocation": allocation, "votes": votes, "winner": winner, "total_seeds": len(allocs)}

    # SAC — медиана
    return {"allocation": float(np.median(allocs)), "votes": None, "winner": None, "total_seeds": len(allocs)}


def _majority_winner(votes: dict) -> str:
    """Победитель в majority vote. При ничьей BUY=SELL → HOLD."""
    max_v = max(votes.values())
    top = [k for k, v in votes.items() if v == max_v]
    if "BUY" in top and "SELL" in top:
        return "HOLD"
    if len(top) == 1:
        return top[0]
    return "BUY" if "BUY" in top else "SELL" if "SELL" in top else "HOLD"


def decide_for_buckets(buckets: list[dict], seeds: list[tuple[int, str]],
                       algo: str) -> list[dict]:
    """Прогон ансамбля по каждому 4h окну, новейшие сверху.

    Каждый элемент списка — копия bucket плюс ключ `decision` с dict
    {allocation, prev_allocation, direction, votes, winner, trade_pnl, ...}
    или `decision=None` + `error` в случае проблем.
    """
    ohlcv = fetch_ohlcv()
    results = []
    prev_alloc = 0.0

    # Идём от старых к новым, чтобы корректно прокидывать prev_allocation
    for b in reversed(buckets):
        cutoff = b["bucket_ts"]
        if cutoff.tz is None:
            cutoff = cutoff.tz_localize("UTC")

        window = ohlcv[ohlcv.index <= cutoff].tail(60)
        if len(window) < NORMALIZE_WINDOW + 2:
            results.append({**b, "decision": None, "error": f"мало OHLCV-баров ({len(window)})"})
            continue

        try:
            features = _build_features(window, b["items"])
            obs = np.append(features[-30:].flatten(), prev_alloc).astype(np.float32)
            agg = _run_ensemble(obs, seeds, algo, prev_alloc)
            new_alloc = agg["allocation"]

            # P&L на следующем баре, если он есть
            after = ohlcv[ohlcv.index > cutoff]
            next_ret = trade_pnl = None
            if not after.empty:
                p1 = float(window["close"].iloc[-1])
                p2 = float(after.iloc[0]["close"])
                if p1 > 0:
                    next_ret = float(np.log(p2 / p1))
                    trade_pnl = float(np.exp(new_alloc * next_ret) - 1)

            results.append({**b, "decision": {
                "allocation": new_alloc,
                "prev_allocation": prev_alloc,
                "direction": _direction(prev_alloc, new_alloc),
                "votes": agg["votes"],
                "winner": agg["winner"],
                "total_seeds": agg["total_seeds"],
                "next_bar_return": next_ret,
                "trade_pnl": trade_pnl,
            }})
            prev_alloc = new_alloc
        except Exception as e:
            results.append({**b, "decision": None, "error": str(e)})

    results.reverse()  # newest first
    return results


def _direction(prev: float, now: float) -> str:
    if now > prev + 0.05:
        return "increase"
    if now < prev - 0.05:
        return "decrease"
    return "hold"


def compute_portfolio(results: list[dict], initial_capital: float = 1.0) -> list[dict]:
    """Накопить equity по trade_pnl от старых окон к новым."""
    equity = initial_capital
    for r in reversed(results):
        d = r.get("decision")
        if d and d.get("trade_pnl") is not None:
            equity *= (1 + d["trade_pnl"])
        r["portfolio_value"] = equity
    return results
