"""News scoring helpers — FinBERT sentiment + FinLang embeddings + PCA."""
import pandas as pd
import streamlit as st

from dashboard.utils.paths import RAW_NEWS, ensure_lib_on_path

ensure_lib_on_path()


@st.cache_resource
def get_finbert_pipeline():
    from lib.features.sentiment import _get_pipeline
    return _get_pipeline()


@st.cache_resource
def get_embedder():
    from lib.features.embeddings import _get_model
    return _get_model()


@st.cache_resource
def get_compressor():
    from dashboard.utils.model_loader import load_compressor
    return load_compressor()


def score_article(text: str) -> dict:
    """Return sentiment label + signed score + compressed 64d embedding."""
    if not text or not text.strip():
        return {"error": "empty text"}

    pipe = get_finbert_pipeline()
    result = pipe([text], truncation=True, max_length=512)[0]
    label = result["label"].lower()
    score = float(result["score"])
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    signed = score * label_map.get(label, 0.0)

    embedder = get_embedder()
    raw_emb = embedder.encode([text], show_progress_bar=False)[0]

    compressor = get_compressor()
    compressed = compressor.transform(raw_emb.reshape(1, -1))[0]

    return {
        "label": label,
        "signed_score": signed,
        "raw_score": score,
        "raw_embedding": raw_emb,
        "compressed_embedding": compressed,
    }


@st.cache_data(ttl=3600)
def sample_recent_news(n: int = 5) -> pd.DataFrame:
    """Return last N rows from cached news parquet."""
    if not RAW_NEWS.exists():
        return pd.DataFrame()
    df = pd.read_parquet(RAW_NEWS).sort_values("date").tail(n)
    return df[["date", "text"] + (["title"] if "title" in df.columns else [])]
