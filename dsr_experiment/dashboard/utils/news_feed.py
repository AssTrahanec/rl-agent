"""News feed — fetch RSS, bucket by 4h, score via FinBERT, run through model."""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import ROOT, ensure_lib_on_path


RSS_SOURCES = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
]

BTC_KEYWORDS = re.compile(
    r"\b(bitcoin|btc|crypto|cryptocurrenc(y|ies)|satoshi)\b",
    re.IGNORECASE,
)

CACHE_DIR = ROOT / "dashboard" / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------- Fetch ----------

@dataclass(frozen=True)
class NewsItem:
    ts: pd.Timestamp   # published time UTC
    title: str
    summary: str
    source: str
    link: str
    uid: str           # hash for dedup


def _uid(title: str, link: str) -> str:
    return hashlib.md5(f"{title}|{link}".encode()).hexdigest()[:16]


def _item_matches_btc(title: str, summary: str) -> bool:
    """Keep only articles that mention BTC/crypto keywords."""
    blob = f"{title} {summary}"
    return bool(BTC_KEYWORDS.search(blob))


def fetch_rss_once() -> list[NewsItem]:
    """Fetch all configured RSS feeds, return filtered+normalized items."""
    import feedparser

    out: list[NewsItem] = []
    for source_name, url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
        except Exception:  # noqa: BLE001
            continue
        for e in feed.entries:
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
            link = getattr(e, "link", "") or ""
            # Parse published time
            ts = None
            for key in ("published_parsed", "updated_parsed"):
                raw = getattr(e, key, None)
                if raw:
                    try:
                        ts = pd.Timestamp(*raw[:6], tz="UTC")
                        break
                    except Exception:  # noqa: BLE001
                        pass
            if ts is None:
                continue
            if not _item_matches_btc(title, summary):
                continue
            out.append(NewsItem(
                ts=ts,
                title=title.strip(),
                summary=re.sub(r"<[^>]+>", "", summary).strip()[:500],
                source=source_name,
                link=link,
                uid=_uid(title, link),
            ))
    return out


# ---------- Bucket by 4h ----------

def bucket_4h(items: list[NewsItem]) -> dict[pd.Timestamp, list[NewsItem]]:
    """Floor each item's ts to 4h boundary, group by bucket."""
    buckets: dict[pd.Timestamp, list[NewsItem]] = {}
    for it in items:
        b = it.ts.floor("4h")
        buckets.setdefault(b, []).append(it)
    return buckets


# ---------- Sentiment scoring ----------

@st.cache_resource
def _finbert():
    ensure_lib_on_path()
    from lib.features.sentiment import _get_pipeline
    return _get_pipeline()


def score_items(items: list[NewsItem]) -> list[dict]:
    """FinBERT sentiment for each item. Returns list of dicts with scores added."""
    if not items:
        return []
    pipe = _finbert()
    texts = [f"{it.title}. {it.summary}" for it in items]
    results = pipe(texts, truncation=True, max_length=512)
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    out = []
    for it, r in zip(items, results):
        label = r["label"].lower()
        signed = float(r["score"] * label_map.get(label, 0.0))
        out.append({
            "uid": it.uid,
            "ts": it.ts,
            "bucket_4h": it.ts.floor("4h"),
            "title": it.title,
            "summary": it.summary,
            "source": it.source,
            "link": it.link,
            "sentiment_label": label,
            "sentiment_score": signed,
        })
    return out


# ---------- Persistence ----------

FEED_CACHE = CACHE_DIR / "news_feed.parquet"


def load_cached_feed() -> pd.DataFrame:
    if not FEED_CACHE.exists():
        return pd.DataFrame(columns=[
            "uid", "ts", "bucket_4h", "title", "summary", "source", "link",
            "sentiment_label", "sentiment_score",
        ])
    return pd.read_parquet(FEED_CACHE)


def save_feed(df: pd.DataFrame) -> None:
    df.to_parquet(FEED_CACHE, index=False)


def merge_new(cached: pd.DataFrame, fresh: list[dict]) -> pd.DataFrame:
    """Append new unique items to cached feed."""
    if not fresh:
        return cached
    new_df = pd.DataFrame(fresh)
    if cached.empty:
        merged = new_df
    else:
        existing_uids = set(cached["uid"])
        new_df = new_df[~new_df["uid"].isin(existing_uids)]
        if new_df.empty:
            return cached
        merged = pd.concat([cached, new_df], ignore_index=True)
    merged = merged.sort_values("ts", ascending=False).reset_index(drop=True)
    return merged


def refresh_feed(force: bool = False) -> tuple[pd.DataFrame, int]:
    """Fetch RSS, score new items, merge into cache.

    Returns (updated_df, num_new_items).
    """
    cached = load_cached_feed()
    items = fetch_rss_once()
    if cached.empty:
        fresh_items = items
    else:
        seen = set(cached["uid"])
        fresh_items = [it for it in items if it.uid not in seen]
    if not fresh_items and not force:
        return cached, 0
    scored = score_items(fresh_items)
    merged = merge_new(cached, scored)
    save_feed(merged)
    return merged, len(scored)


# ---------- 4h aggregated view ----------

def group_by_bucket(df: pd.DataFrame, lookback_buckets: int = 20) -> list[dict]:
    """Return list of buckets with their news items, sorted newest first."""
    if df.empty:
        return []
    df = df.copy()
    df["bucket_4h"] = pd.to_datetime(df["bucket_4h"], utc=True)
    buckets = []
    for bucket_ts, group in df.groupby("bucket_4h", sort=False):
        items = group.sort_values("ts", ascending=False).to_dict("records")
        sentiments = [r["sentiment_score"] for r in items]
        buckets.append({
            "bucket_ts": bucket_ts,
            "n_news": len(items),
            "sentiment_mean": float(np.mean(sentiments)) if sentiments else 0.0,
            "sentiment_max": float(np.max(sentiments)) if sentiments else 0.0,
            "sentiment_min": float(np.min(sentiments)) if sentiments else 0.0,
            "items": items,
        })
    buckets.sort(key=lambda b: b["bucket_ts"], reverse=True)
    return buckets[:lookback_buckets]
