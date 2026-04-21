"""Strategy Live — main page.

Single story: what does the agent decide right now on the freshest news,
and how would this strategy have performed over the recent few days.
"""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.utils.paths import ensure_lib_on_path
from dashboard.utils import snapshot
from dashboard.utils.model_catalog import list_model_entries
from dashboard.utils.news_feed import (
    refresh_feed, load_cached_feed, group_by_bucket,
)
from dashboard.utils.feed_decisions import (
    decide_ensemble_for_buckets, compute_portfolio_trajectory,
)

ensure_lib_on_path()


st.set_page_config(page_title="Стратегия — live", layout="wide")
st.title("Стратегия в реальном времени")


# ---- Model selector ----
entries = list_model_entries()
if not entries:
    st.error("Нет обученных моделей.")
    st.stop()

default_label = next(
    (e.label for e in entries if "tuned + 10 seeds" in e.label and e.algo == "DQN"),
    entries[0].label,
)
labels = [e.label for e in entries]

top_cols = st.columns([4, 1])
with top_cols[0]:
    picked = st.selectbox("Модель (ансамбль всех сидов)", labels,
                          index=labels.index(default_label))
    entry = next(e for e in entries if e.label == picked)
with top_cols[1]:
    st.write("")
    st.write("")
    do_refresh = st.button("Обновить", use_container_width=True)


# ---- Refresh news if stale ----
cached = load_cached_feed()
now = pd.Timestamp.utcnow()
now_utc = now if now.tz is not None else now.tz_localize("UTC")

need_refresh = cached.empty or do_refresh
if not need_refresh and not cached.empty:
    latest_ts = pd.to_datetime(cached["ts"].max(), utc=True)
    need_refresh = (now_utc - latest_ts) > pd.Timedelta(hours=4)

if need_refresh:
    with st.spinner("Тяну RSS, считаю sentiment..."):
        try:
            cached, n_new = refresh_feed(force=do_refresh)
            if n_new > 0:
                st.toast(f"Добавлено {n_new} новостей.", icon="✓")
        except Exception as e:  # noqa: BLE001
            st.error(f"Не удалось обновить: {e}")

if cached.empty:
    st.warning("Лента пустая. Нажми «Обновить».")
    st.stop()


# ---- Run ensemble over last N buckets ----
buckets = group_by_bucket(cached, lookback_buckets=15)
models_avail = snapshot.discover_models(entry.snapshot).get(entry.algo, [])
seeds_paths = [(m["seed"], m["path"]) for m in models_avail]

with st.spinner(f"{len(seeds_paths)} моделей анализируют {len(buckets)} окон..."):
    decorated = decide_ensemble_for_buckets(buckets, seeds_paths, entry.algo)
    decorated = compute_portfolio_trajectory(decorated, initial_capital=10000.0)

if not decorated:
    st.warning("Нет данных для анализа.")
    st.stop()


# ========================================================================
# SECTION 1 — ТЕКУЩЕЕ РЕШЕНИЕ (самое свежее окно)
# ========================================================================

current = decorated[0]   # newest bucket
current_dec = current.get("decision")

st.divider()
hdr_cols = st.columns([3, 2])
with hdr_cols[0]:
    st.subheader("Решение агента сейчас")
    st.caption(f"Окно {current['bucket_ts'].strftime('%Y-%m-%d %H:%M UTC')}")

if current_dec is None:
    st.warning(f"Решение не получено: {current.get('error', '—')}")
else:
    # One large decision card
    alloc_pct = int(current_dec.allocation * 100)
    prev_pct = int(current_dec.prev_allocation * 100)

    if current_dec.direction == "increase":
        action_verb = "ПОКУПАТЬ"
        action_color = "#2ca02c"
        arrow = "▲"
    elif current_dec.direction == "decrease":
        action_verb = "ПРОДАВАТЬ"
        action_color = "#d62728"
        arrow = "▼"
    else:
        action_verb = "ДЕРЖАТЬ"
        action_color = "#6c757d"
        arrow = "●"

    votes_line = ""
    if current_dec.votes:
        # Dominant vote
        dom = max(current_dec.votes.items(), key=lambda kv: kv[1])
        votes_line = f"{dom[1]} из {current_dec.total_seeds} моделей согласны"
    else:
        votes_line = f"{current_dec.total_seeds} моделей, средняя доля {alloc_pct}%"

    st.markdown(
        f"""
        <div style="padding:28px;border-radius:12px;background:{action_color};color:white;">
            <div style="font-size:14px;opacity:0.8;margin-bottom:6px;">РЕКОМЕНДАЦИЯ</div>
            <div style="font-size:44px;font-weight:800;margin-bottom:8px;">
                {arrow} {action_verb}
            </div>
            <div style="font-size:20px;opacity:0.95;">
                позиция BTC: {prev_pct}% → <b>{alloc_pct}%</b>
            </div>
            <div style="font-size:14px;margin-top:10px;opacity:0.85;">
                {votes_line}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Why — input signals
    st.markdown("**Почему**")
    why_cols = st.columns(3)
    why_cols[0].metric("Новостей в окне", current["n_news"])
    why_cols[1].metric("Средний sentiment", f"{current['sentiment_mean']:+.2f}")
    if current_dec.votes:
        buy_n = current_dec.votes.get("BUY", 0)
        sell_n = current_dec.votes.get("SELL", 0)
        hold_n = current_dec.votes.get("HOLD", 0)
        why_cols[2].metric("Голоса BUY/HOLD/SELL", f"{buy_n}/{hold_n}/{sell_n}")
    else:
        why_cols[2].metric("Средняя доля ансамбля", f"{alloc_pct}%")

    # Show top news in the current window
    if current["items"]:
        with st.expander(f"Свежие новости этого окна ({current['n_news']})", expanded=True):
            for item in current["items"][:5]:
                s_color = (
                    "#2ca02c" if item["sentiment_label"] == "positive"
                    else "#d62728" if item["sentiment_label"] == "negative"
                    else "#6c757d"
                )
                ts_str = pd.Timestamp(item["ts"]).strftime("%H:%M")
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<span style="display:inline-block;padding:2px 6px;border-radius:3px;'
                    f'background:{s_color};color:white;font-size:11px;font-weight:600;'
                    f'margin-right:6px;">{item["sentiment_score"]:+.2f}</span>'
                    f'<a href="{item["link"]}" target="_blank">{item["title"]}</a>'
                    f'<span style="color:#888;font-size:12px;">  — {item["source"]}, {ts_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ========================================================================
# SECTION 2 — ЧТО БЫ БЫЛО (equity на последних окнах)
# ========================================================================

st.divider()
st.subheader("Результат стратегии за последние окна")
st.caption(
    "Капитал $10 000 в начале ленты. Агент торгует каждые 4 часа "
    "по тем же сигналам что выше. Сравнение с Buy & Hold."
)

# Build equity curves
ordered = list(reversed(decorated))   # oldest first
timestamps = [b["bucket_ts"] for b in ordered]
agent_equity = [b.get("portfolio_value", 10000.0) for b in ordered]

# Buy & Hold — using price from ohlcv near bucket timestamps
from dashboard.utils.feed_decisions import _fetch_ohlcv_cached
ohlcv_all = _fetch_ohlcv_cached()
bh_equity = [10000.0]
prev_price = None
for i, ts in enumerate(timestamps):
    cutoff = ts if ts.tz is not None else ts.tz_localize("UTC")
    window = ohlcv_all[ohlcv_all.index <= cutoff]
    if window.empty:
        bh_equity.append(bh_equity[-1])
        continue
    price = float(window.iloc[-1]["close"])
    if prev_price is None:
        prev_price = price
        continue
    ret = price / prev_price - 1
    bh_equity.append(bh_equity[-1] * (1 + ret))
    prev_price = price
# Align lengths
if len(bh_equity) > len(timestamps):
    bh_equity = bh_equity[:len(timestamps)]
elif len(bh_equity) < len(timestamps):
    pad = timestamps[:len(bh_equity)]
    timestamps_bh = pad
else:
    timestamps_bh = timestamps

final_agent = agent_equity[-1] if agent_equity else 10000.0
final_bh = bh_equity[-1] if bh_equity else 10000.0
agent_ret = (final_agent / 10000.0 - 1) * 100
bh_ret = (final_bh / 10000.0 - 1) * 100

k1, k2, k3 = st.columns(3)
k1.metric("Стратегия агента", f"${final_agent:,.0f}",
          delta=f"{agent_ret:+.2f}%")
k2.metric("Buy & Hold", f"${final_bh:,.0f}",
          delta=f"{bh_ret:+.2f}%")
k3.metric("Преимущество агента",
          f"{(agent_ret - bh_ret):+.2f}%",
          delta="от Buy & Hold", delta_color="off")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=timestamps, y=agent_equity, mode="lines",
    line=dict(color="#2ca02c", width=3),
    name="Агент",
))
fig.add_trace(go.Scatter(
    x=timestamps[:len(bh_equity)], y=bh_equity, mode="lines",
    line=dict(color="black", width=2, dash="dash"),
    name="Buy & Hold",
))
fig.update_layout(
    height=360,
    yaxis_title="Капитал, $",
    xaxis_title=None,
    hovermode="x unified",
    margin=dict(t=20, l=50, r=20, b=40),
    legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.5, xanchor="center"),
)
st.plotly_chart(fig, width="stretch")


# ========================================================================
# SECTION 3 — ЛЕНТА РЕШЕНИЙ (компактная)
# ========================================================================

st.divider()
st.subheader("История решений")

for b in decorated[1:]:   # skip current (already shown above)
    dec = b.get("decision")
    ts_str = b["bucket_ts"].strftime("%Y-%m-%d %H:%M UTC")

    if dec is None:
        with st.container(border=True):
            st.markdown(f"**{ts_str}** · ошибка: {b.get('error', '—')}")
        continue

    # Compact one-row layout
    if dec.direction == "increase":
        action_icon = "▲ BUY"
        action_color = "#2ca02c"
    elif dec.direction == "decrease":
        action_icon = "▼ SELL"
        action_color = "#d62728"
    else:
        action_icon = "● HOLD"
        action_color = "#6c757d"

    if dec.trade_pnl is not None:
        pnl_pct = dec.trade_pnl * 100
        pnl_color = "#2ca02c" if dec.trade_pnl > 0 else ("#d62728" if dec.trade_pnl < 0 else "#6c757d")
        pnl_text = f'<span style="color:{pnl_color};font-weight:700;">{pnl_pct:+.2f}%</span>'
    else:
        pnl_text = '<span style="color:#888;">—</span>'

    votes_short = ""
    if dec.votes:
        dom = max(dec.votes.items(), key=lambda kv: kv[1])
        votes_short = f"{dom[1]}/{dec.total_seeds}"
    else:
        votes_short = f"{int(dec.allocation*100)}%"

    with st.container(border=True):
        cols = st.columns([2, 1, 1, 1, 1])
        cols[0].markdown(f"**{ts_str}**")
        cols[1].markdown(
            f'<span style="background:{action_color};color:white;padding:4px 10px;'
            f'border-radius:4px;font-weight:600;">{action_icon}</span>',
            unsafe_allow_html=True,
        )
        cols[2].markdown(f"{b['n_news']} новостей · {b['sentiment_mean']:+.2f}")
        cols[3].markdown(f"согласие: **{votes_short}**")
        cols[4].markdown(f"P&L: {pnl_text}", unsafe_allow_html=True)

        with st.expander("Новости окна"):
            for item in b["items"]:
                s_color = (
                    "#2ca02c" if item["sentiment_label"] == "positive"
                    else "#d62728" if item["sentiment_label"] == "negative"
                    else "#6c757d"
                )
                ts_sub = pd.Timestamp(item["ts"]).strftime("%H:%M")
                st.markdown(
                    f'<span style="color:{s_color};font-weight:600;">[{item["sentiment_score"]:+.2f}]</span> '
                    f'<a href="{item["link"]}" target="_blank">{item["title"]}</a> '
                    f'<span style="color:#888;font-size:12px;">— {item["source"]} {ts_sub}</span>',
                    unsafe_allow_html=True,
                )
