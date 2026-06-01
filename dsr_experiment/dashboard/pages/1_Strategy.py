"""Стратегия — текущее решение ансамбля, цена BTC и история за 5 дней."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.utils import snapshot, style
from dashboard.utils.feed_decisions import (
    decide_for_buckets, compute_portfolio, fetch_ohlcv,
)
from dashboard.utils.model_catalog import list_model_entries
from dashboard.utils.news_feed import (
    load_cached_feed, refresh_feed, group_by_bucket,
)
from dashboard.utils.paths import ensure_lib_on_path

ensure_lib_on_path()

TX_COST = 0.001   # совпадает с env.tx_cost

st.set_page_config(page_title="Стратегия", layout="wide")
style.apply()

st.title("Решение агента прямо сейчас")
st.caption(
    "Лента BTC-новостей разбита на 4-часовые окна. "
    "На каждом окне работают 10 обученных сетей. "
    "DQN: побеждает большинство (BUY / HOLD / SELL → 0% или 100% в BTC). "
    "SAC: итоговая аллокация = медиана непрерывных голосов."
)


# ── Выбор модели ──
entries = list_model_entries()
if not entries:
    st.error("Нет обученных моделей в `experiments/`.")
    st.stop()

default = next((e.label for e in entries if e.algo == "DQN"), entries[0].label)
labels = [e.label for e in entries]
picked = st.selectbox("Алгоритм", labels, index=labels.index(default))
entry = next(e for e in entries if e.label == picked)


# ── Автообновление новостей (раз в 4 часа) ──
feed = load_cached_feed()
now_utc = pd.Timestamp.now(tz="UTC")
stale = feed.empty or (now_utc - pd.to_datetime(feed["ts"].max(), utc=True)) > pd.Timedelta(hours=4)
if stale:
    with st.spinner("Подтягиваю свежие новости и считаю sentiment FinBERT..."):
        try:
            feed, n_new, _ = refresh_feed(force=False)
            if n_new > 0:
                st.toast(f"Добавлено {n_new} новостей", icon="✓")
        except Exception as e:
            st.error(f"Не удалось обновить ленту: {e}")

if feed.empty:
    st.warning("Лента пустая. Проверь интернет и API-ключи.")
    st.stop()


# ── Решения ансамбля по 30 последним 4h-окнам ──
buckets = group_by_bucket(feed, lookback_buckets=30)
seeds = [(m["seed"], m["path"]) for m in snapshot.discover_models(entry.snapshot).get(entry.algo, [])]

with st.spinner(f"{len(seeds)} моделей анализируют {len(buckets)} окон..."):
    decisions = decide_for_buckets(buckets, seeds, entry.algo)
    decisions = compute_portfolio(decisions, initial_capital=1.0)

if not decisions:
    st.warning("Нет данных для анализа.")
    st.stop()


# ──────────────── Секция 1 · текущее решение ────────────────
current = decisions[0]
dec = current["decision"]

st.divider()
st.subheader(f"Окно {current['bucket_ts']:%d.%m.%Y · %H:%M UTC}")

if dec is None:
    st.warning(f"Ошибка: {current.get('error', '—')}")
else:
    # Какое именно действие показать
    if dec["winner"]:
        verb = dec["winner"]
    elif dec["direction"] == "increase":
        verb = "BUY"
    elif dec["direction"] == "decrease":
        verb = "SELL"
    else:
        verb = "HOLD"

    left, right = st.columns([3, 2])
    with left:
        style.decision_badge(verb)
        prev_pct = int(dec["prev_allocation"] * 100)
        new_pct = int(dec["allocation"] * 100)
        st.markdown(f"**{prev_pct}% → {new_pct}%** капитала в BTC")
    with right:
        st.metric("Новостей в окне", current["n_news"])
        st.metric("Sentiment", f"{current['sentiment_mean']:+.2f}")
        if dec["votes"]:
            v = dec["votes"]
            st.markdown(f"Голоса: BUY **{v['BUY']}** · HOLD **{v['HOLD']}** · SELL **{v['SELL']}**")

    # Свежие заголовки окна — кликабельны, открываются в новой вкладке
    if current["items"]:
        with st.expander(f"Заголовки окна ({current['n_news']})", expanded=True):
            for item in current["items"][:8]:
                st.markdown(style.news_link(item), unsafe_allow_html=True)


# ──────────────── Секция 2 · BTC и аллокация ────────────────
st.divider()
days = max(1, int(len(decisions) * 4 / 24))
st.subheader(f"BTC и решения за последние {days} дней")

ordered = list(reversed(decisions))   # старые → новые
timestamps = [b["bucket_ts"] for b in ordered]
allocations = [b["decision"]["allocation"] if b["decision"] else 0.0 for b in ordered]
agent_eq = [b["portfolio_value"] for b in ordered]

# Buy & Hold baseline на тех же таймштампах
ohlcv = fetch_ohlcv()
btc_prices, bh_eq = [], [1.0]
prev_p = None
for ts in timestamps:
    cutoff = ts if ts.tz else ts.tz_localize("UTC")
    bars = ohlcv[ohlcv.index <= cutoff]
    p = float(bars.iloc[-1]["close"]) if not bars.empty else None
    btc_prices.append(p)
    if p is not None and prev_p is not None:
        bh_eq.append(bh_eq[-1] * p / prev_p)
    if p is not None:
        prev_p = p
bh_eq = bh_eq[:len(timestamps)]

# Сводные метрики
agent_ret = (agent_eq[-1] - 1) * 100
bh_ret = (bh_eq[-1] - 1) * 100
turnover = sum(abs(allocations[i] - (allocations[i - 1] if i else 0)) for i in range(len(allocations)))
n_switches = sum(abs(allocations[i] - (allocations[i - 1] if i else 0)) > 0.1 for i in range(len(allocations)))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Стратегия", f"{agent_ret:+.2f}%", f"{agent_ret - bh_ret:+.2f}% к HODL")
m2.metric("HODL", f"{bh_ret:+.2f}%")
m3.metric("Сделок", str(n_switches), help="Изменение позиции больше чем на 10 п.п.")
m4.metric("Комиссии", f"-{turnover * TX_COST * 100:.2f}%", help="0.1% за каждый Δallocation, уже учтено")

# График: цена сверху, аллокация снизу
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.65, 0.35], vertical_spacing=0.06,
    subplot_titles=("Цена BTC, $", "Доля капитала в BTC, %"),
)
valid_px = [(t, p) for t, p in zip(timestamps, btc_prices) if p is not None]
if valid_px:
    x, y = zip(*valid_px)
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=style.TEXT, width=2),
                             hovertemplate="$%{y:,.0f}<extra></extra>"), row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=timestamps, y=[a * 100 for a in allocations],
        line=dict(color=style.ACCENT, width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.15)",
        hovertemplate="%{y:.0f}%<extra></extra>",
    ),
    row=2, col=1,
)
fig.update_layout(height=460, showlegend=False, plot_bgcolor="white",
                  margin=dict(t=40, l=60, r=20, b=40))
fig.update_yaxes(range=[-5, 105], ticksuffix="%", row=2, col=1)
st.plotly_chart(fig, width="stretch")

# Equity vs HODL
st.subheader("Доходность vs Buy & Hold")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=timestamps, y=[(e - 1) * 100 for e in agent_eq],
                          line=dict(color=style.ACCENT, width=3), name="Стратегия"))
fig2.add_trace(go.Scatter(x=timestamps[:len(bh_eq)], y=[(e - 1) * 100 for e in bh_eq],
                          line=dict(color=style.TEXT, width=2, dash="dash"), name="HODL"))
fig2.update_layout(height=320, yaxis_ticksuffix="%", plot_bgcolor="white",
                   margin=dict(t=20, l=60, r=20, b=40),
                   legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center"))
st.plotly_chart(fig2, width="stretch")


# ──────────────── Секция 3 · история ────────────────
st.divider()
st.subheader("История решений")

for b in decisions[1:]:
    d = b["decision"]
    ts_str = b["bucket_ts"].strftime("%d.%m %H:%M UTC")
    if d is None:
        st.caption(f"{ts_str} — ошибка: {b.get('error', '—')}")
        continue

    if d["winner"]:
        verb = d["winner"]
    elif d["direction"] == "increase":
        verb = "BUY"
    elif d["direction"] == "decrease":
        verb = "SELL"
    else:
        verb = "HOLD"
    color = {"BUY": style.BUY, "SELL": style.SELL, "HOLD": style.HOLD}[verb]
    prev_pct = int(d["prev_allocation"] * 100)
    new_pct = int(d["allocation"] * 100)

    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{ts_str}** &nbsp; {prev_pct}% → {new_pct}%")
        c2.markdown(
            f"<div style='text-align:right;'><span style='background:{color};color:white;"
            f"padding:4px 14px;border-radius:6px;font-weight:700;'>{verb}</span></div>",
            unsafe_allow_html=True,
        )
        if d["votes"]:
            v = d["votes"]
            st.caption(f"BUY {v['BUY']} · HOLD {v['HOLD']} · SELL {v['SELL']} → {verb}")
        if d["trade_pnl"] is not None:
            ret = (np.exp(d["next_bar_return"]) - 1) * 100
            pnl = d["trade_pnl"] * 100
            st.caption(f"BTC: {ret:+.2f}% · стратегия: {pnl:+.2f}%")
        st.caption(f"{b['n_news']} новостей, sentiment {b['sentiment_mean']:+.2f}")

        # Топ-5 новостей по силе sentiment — кликабельные
        top_items = sorted(b["items"], key=lambda x: -abs(x["sentiment_score"]))[:5]
        for item in top_items:
            st.markdown(style.news_link(item), unsafe_allow_html=True)
        if len(b["items"]) > 5:
            st.caption(f"Ещё {len(b['items']) - 5} новостей")
