"""News Analyzer — FinBERT sentiment + PCA embedding inspection."""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.utils.paths import ensure_lib_on_path

ensure_lib_on_path()


st.set_page_config(page_title="News Analyzer", layout="wide")
st.title("News Analyzer")
st.caption(
    "FinBERT sentiment + FinLang embedding → PCA(64). "
    "Та же цепочка что и при обучении моделей."
)


# Lazy imports — heavy models, only load when user clicks "Analyze"
example_text = (
    "Bitcoin ETF outflows hit record $500 million as investor sentiment "
    "weakens amid regulatory uncertainty."
)

text = st.text_area("Новость / заголовок", value=example_text, height=120, max_chars=2000)


def render_result(result: dict):
    label = result["label"]
    signed = result["signed_score"]

    label_colors = {
        "positive": "#2ca02c",
        "negative": "#d62728",
        "neutral": "#6c757d",
    }
    color = label_colors.get(label, "#6c757d")

    st.markdown(
        f"""
        <div style="padding:20px;border-radius:8px;background:{color};
                    color:white;text-align:center;font-size:24px;font-weight:bold;">
            {label.upper()} &nbsp; {signed:+.3f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    emb = result["compressed_embedding"]
    st.subheader("Компрессированное представление (PCA 64d)")

    fig = go.Figure(go.Bar(
        x=list(range(len(emb))), y=emb,
        marker_color=["#d62728" if v < 0 else "#2ca02c" for v in emb],
    ))
    fig.update_layout(
        height=280,
        xaxis_title="Компонента PCA",
        yaxis_title="Значение",
        margin=dict(t=20, l=40, r=20, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("L2 норма", f"{float(np.linalg.norm(emb)):.3f}")
    c2.metric("Максимум", f"{float(emb.max()):+.3f}")
    c3.metric("Минимум", f"{float(emb.min()):+.3f}")


if st.button("Анализировать", type="primary"):
    try:
        from dashboard.utils.news_live import score_article
    except Exception as e:  # noqa: BLE001
        st.error(
            f"Не могу импортировать news_live: {e}. Убедись что установлены "
            "transformers, torch, sentence-transformers."
        )
        st.stop()

    with st.spinner("FinBERT + FinLang... (первый вызов качает модели)"):
        result = score_article(text)

    if "error" in result:
        st.warning(result["error"])
    else:
        render_result(result)

# Optional: recent news sample
with st.expander("Примеры из обучающего корпуса (кэш)"):
    try:
        from dashboard.utils.news_live import sample_recent_news
        sample = sample_recent_news(n=5)
        if sample.empty:
            st.caption("Корпус не найден (data/raw/news.parquet отсутствует).")
        else:
            for _, row in sample.iterrows():
                st.markdown(f"**{row['date'].strftime('%Y-%m-%d')}**")
                title = row.get("title", "")
                if isinstance(title, str) and title:
                    st.markdown(f"_{title}_")
                st.text(str(row["text"])[:400] + ("..." if len(str(row["text"])) > 400 else ""))
                st.divider()
    except Exception as e:  # noqa: BLE001
        st.caption(f"Cannot load corpus: {e}")
