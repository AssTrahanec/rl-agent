"""Анализатор новости — какой вес имеет конкретная новость для модели.

Сравниваем:
  · что модель решает на текущем 5-дневном фоне (decision_without),
  · что она решила бы, если бы единственным сигналом была эта новость (decision_with).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.utils import style
from dashboard.utils.news_feed import load_cached_feed, refresh_feed
from dashboard.utils.news_impact import analyze_news_impact
from dashboard.utils.paths import ensure_lib_on_path

ensure_lib_on_path()

st.set_page_config(page_title="Анализатор новости", layout="wide")
style.apply()

st.title("Какой вес у конкретной новости для модели")
st.caption(
    "Сравниваем решение ансамбля на текущем фоне последних 5 дней "
    "и на «только эта новость» — чтобы новость не растворилась в усреднении фона."
)

PRESETS = {
    "SEC одобрила биткоин-ETF": (
        "The SEC has approved the first spot Bitcoin ETF, opening the door "
        "for large institutional investment into cryptocurrency."
    ),
    "Китай запретил майнинг": (
        "China has banned all cryptocurrency mining operations nationwide, "
        "citing energy consumption and financial risk concerns."
    ),
    "Нейтральная новость": (
        "A blockchain technology conference is scheduled to take place "
        "in Europe next month."
    ),
}
BG_DAYS = 5


# ── Фоновые новости подтягиваются автоматически и не показываются ──
def auto_background() -> list[str]:
    """Заголовки за последние 5 дней. При необходимости подтягиваем свежие."""
    feed = load_cached_feed()
    now_utc = pd.Timestamp.now(tz="UTC")
    stale = feed.empty or (now_utc - pd.to_datetime(feed["ts"].max(), utc=True)) > pd.Timedelta(hours=4)
    if stale:
        try:
            feed, _, _ = refresh_feed(force=False)
        except Exception:
            return []
    if feed.empty:
        return []
    feed = feed.copy()
    feed["ts"] = pd.to_datetime(feed["ts"], utc=True)
    cutoff = now_utc - pd.Timedelta(days=BG_DAYS)
    recent = feed[feed["ts"] >= cutoff].sort_values("ts", ascending=False)
    if recent.empty:
        recent = feed.sort_values("ts", ascending=False).head(40)
    return [f"{r.title}. {r.summary}".strip() for r in recent.itertuples()]


if "bg_items" not in st.session_state:
    with st.spinner("Подтягиваю свежий новостной фон..."):
        st.session_state.bg_items = auto_background()
bg_items: list[str] = st.session_state.bg_items
if "tested_text" not in st.session_state:
    st.session_state.tested_text = ""


# ── Поле ввода ──
with st.container(border=True):
    head_l, head_r = st.columns([4, 2])
    head_l.markdown("### Проверяемая новость")
    head_l.caption("FinBERT — английский. Можно несколько строк.")
    head_r.caption(f"Фон: **{len(bg_items)}** новостей за {BG_DAYS} дней")

    preset_cols = st.columns(len(PRESETS))
    for col, (label, text) in zip(preset_cols, PRESETS.items()):
        if col.button(label, use_container_width=True, key=f"preset_{label}"):
            st.session_state.tested_text = text

    tested_text = st.text_area(
        "Проверяемая новость", key="tested_text", height=160,
        placeholder="Tesla announces $5B Bitcoin purchase added to its treasury.",
        label_visibility="collapsed",
    )

analyze = st.button("Анализировать", type="primary", use_container_width=True)


# ── Результат ──
if analyze:
    tested = [ln.strip() for ln in tested_text.splitlines() if ln.strip()]
    if not tested:
        st.warning("Введите проверяемую новость или нажмите один из пресетов.")
        st.stop()

    with st.spinner("FinBERT считает sentiment, ансамбль из 10 моделей пересчитывает решение..."):
        res = analyze_news_impact(bg_items, tested)

    v0 = res["decision_without"]["votes"] or {}
    v1 = res["decision_with"]["votes"] or {}
    total = res["decision_without"]["total_seeds"]
    diff_buy = v1.get("BUY", 0) - v0.get("BUY", 0)
    diff_sell = v1.get("SELL", 0) - v0.get("SELL", 0)

    # Вердикт
    if diff_buy >= 2 or diff_sell <= -2:
        st.success(f"**Сильный бычий сигнал.** Новость добавила +{diff_buy} голосов за BUY.")
    elif diff_buy <= -2 or diff_sell >= 2:
        st.error(f"**Сильный медвежий сигнал.** Новость отняла {abs(diff_buy)} голосов у BUY.")
    elif abs(diff_buy) >= 1:
        if diff_buy > 0:
            st.info(f"Лёгкий сдвиг к покупке (+{diff_buy} BUY).")
        else:
            st.info(f"Лёгкий сдвиг к продаже ({abs(diff_buy)} BUY → другое решение).")
    else:
        st.info("Нейтральный сигнал: новость не сдвинула голосование ансамбля.")

    # Stacked bar — голоса до/после
    st.subheader("Голоса ансамбля")
    st.caption("Слева — на текущем фоне 5 дней, справа — если бы единственным сигналом была эта новость.")
    fig = go.Figure()
    cats = ["Сейчас (фон 5 дней)", "Только эта новость"]
    fig.add_trace(go.Bar(x=cats, y=[v0.get("BUY", 0), v1.get("BUY", 0)],
                         name="BUY", marker_color=style.BUY))
    fig.add_trace(go.Bar(x=cats, y=[v0.get("HOLD", 0), v1.get("HOLD", 0)],
                         name="HOLD", marker_color=style.HOLD))
    fig.add_trace(go.Bar(x=cats, y=[v0.get("SELL", 0), v1.get("SELL", 0)],
                         name="SELL", marker_color=style.SELL))
    fig.update_layout(barmode="stack", height=320, plot_bgcolor="white",
                      yaxis_title="Число моделей", yaxis_range=[0, total],
                      margin=dict(t=20, l=50, r=20, b=40))
    st.plotly_chart(fig, width="stretch")

    # Дельты метриками
    m1, m2, m3 = st.columns(3)
    m1.metric("BUY", f"{v1.get('BUY', 0)} из {total}", delta=f"{diff_buy:+d}")
    m2.metric("HOLD", f"{v1.get('HOLD', 0)} из {total}",
              delta=f"{v1.get('HOLD', 0) - v0.get('HOLD', 0):+d}")
    m3.metric("SELL", f"{v1.get('SELL', 0)} из {total}", delta=f"{diff_sell:+d}")

    # Sentiment по каждой строке проверяемой новости
    st.subheader("Тональность проверяемой новости")
    with st.container(border=True):
        for text, sent in res["tested_per_news"]:
            preview = text if len(text) <= 130 else text[:130] + "…"
            st.markdown(f"{style.sentiment_badge(sent)} {preview}", unsafe_allow_html=True)
        st.caption(
            f"Контекст для левой колонки: {res['background_count']} новостей за {BG_DAYS} дней, "
            f"средний sentiment {res['background_mean']:+.2f}."
        )
