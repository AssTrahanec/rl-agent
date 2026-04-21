"""Strategy Live — main page (percent-based, with explicit context)."""
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
    _fetch_ohlcv_cached,
)

ensure_lib_on_path()

TX_COST = 0.001   # matches env.tx_cost

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
    with st.spinner("Тяну новости (NewsAPI архив + RSS), считаю sentiment..."):
        try:
            cached, n_new, source_tag = refresh_feed(force=do_refresh)
            if n_new > 0:
                src_label = {"newsapi": "NewsAPI (5 дней архива)",
                             "rss": "RSS (только свежие)",
                             "newsapi+rss": "NewsAPI + RSS",
                             "none": "никаких источников"}.get(source_tag, source_tag)
                st.toast(f"Добавлено {n_new} новостей ({src_label}).", icon="✓")
        except Exception as e:  # noqa: BLE001
            st.error(f"Не удалось обновить: {e}")

if cached.empty:
    st.warning("Лента пустая. Нажми «Обновить».")
    st.stop()


# ---- Ensemble over last N buckets ----
buckets = group_by_bucket(cached, lookback_buckets=15)
models_avail = snapshot.discover_models(entry.snapshot).get(entry.algo, [])
seeds_paths = [(m["seed"], m["path"]) for m in models_avail]

with st.spinner(f"{len(seeds_paths)} моделей анализируют {len(buckets)} окон..."):
    decorated = decide_ensemble_for_buckets(buckets, seeds_paths, entry.algo)
    decorated = compute_portfolio_trajectory(decorated, initial_capital=1.0)

if not decorated:
    st.warning("Нет данных для анализа.")
    st.stop()


# ========================================================================
# SECTION 1 — CURRENT DECISION
# ========================================================================

current = decorated[0]
current_dec = current.get("decision")

st.divider()
st.subheader("Решение агента сейчас")
st.caption(f"Окно {current['bucket_ts'].strftime('%Y-%m-%d %H:%M UTC')}")

if current_dec is None:
    st.warning(f"Решение не получено: {current.get('error', '—')}")
else:
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

    if current_dec.votes:
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
                позиция в BTC: {prev_pct}% → <b>{alloc_pct}%</b>
            </div>
            <div style="font-size:14px;margin-top:10px;opacity:0.85;">
                {votes_line}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**На чём основано**")
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

    if current["items"]:
        with st.expander(f"Новости окна ({current['n_news']})", expanded=False):
            for item in current["items"][:8]:
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
# SECTION 2 — "КАК ЭТО ПОНИМАТЬ" (context + limitations)
# ========================================================================

st.divider()
with st.expander("Как это понимать — контекст и ограничения", expanded=False):
    st.markdown(
        """
        **Что делает агент**

        Каждые 4 часа модель смотрит на последние 30 4h-баров цены BTC,
        свежие новости этого окна и принимает решение о доле капитала в BTC.

        **Что такое ансамбль**

        Мы обучили 10 независимых моделей с разной инициализацией весов
        (стандарт RL-литературы, Henderson et al. 2018). Финальное решение —
        голосование ансамбля. Если 8 из 10 согласны — уверенное решение;
        если голоса разделились поровну — слабый сигнал.

        **SAC vs DQN**

        - **DQN** — discrete action: Hold / Buy (100% в BTC) / Sell (100% в cash).
          Каждая смена — полный переход, средних позиций нет.
        - **SAC** — continuous: любая доля 0..1. Плавные перекладывания.

        На наших данных **DQN обыгрывает SAC и Buy & Hold** по Sharpe на OOS 2025
        (см. вкладку Валидация). Причина: при одном активе дробные позиции
        создают больше transaction cost без выгоды — discrete решения чище.
        Это согласуется с литературой 2024-2025 (Jurnal Mandiri, MarketMind).

        **Ограничения**

        - **Long-only**: агент или в BTC, или в cash. Короткие позиции не
          моделируются — на спотовом рынке их нет, а на perpetual funding rate
          делает шорты невыгодными (проверено в экспериментах, провал PPO).
        - **Обучение до 2023-12**: модель не видела 2024-H2 и позже. OOS
          валидация проведена на 2024-2025.
        - **Транзакционные издержки 0.1% на смену позиции** учтены в обучении.
        - **Binance spot только**: без маржинальной торговли.
        """
    )


# ========================================================================
# SECTION 3 — RECENT PERFORMANCE (percent-based)
# ========================================================================

st.divider()
st.subheader("Результат стратегии за последние окна")

ordered = list(reversed(decorated))
timestamps = [b["bucket_ts"] for b in ordered]
agent_equity = [b.get("portfolio_value", 1.0) for b in ordered]

# Buy & Hold baseline
ohlcv_all = _fetch_ohlcv_cached()
bh_equity = [1.0]
prev_price = None
for ts in timestamps:
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
if len(bh_equity) > len(timestamps):
    bh_equity = bh_equity[:len(timestamps)]

# Turnover + tx cost calc
allocations = [b["decision"].allocation for b in ordered if b.get("decision")]
n_switches = 0
turnover_total = 0.0
prev = 0.0
for a in allocations:
    diff = abs(a - prev)
    turnover_total += diff
    if diff > 0.1:
        n_switches += 1
    prev = a
tx_total_pct = turnover_total * TX_COST * 100

# Percent-based final metrics
agent_ret_pct = (agent_equity[-1] - 1) * 100 if agent_equity else 0
bh_ret_pct = (bh_equity[-1] - 1) * 100 if bh_equity else 0
advantage_pct = agent_ret_pct - bh_ret_pct

period_hours = len(timestamps) * 4
period_days = period_hours / 24

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Стратегия агента",
    f"{agent_ret_pct:+.2f}%",
    delta=f"{advantage_pct:+.2f}% vs B&H",
)
k2.metric("Buy & Hold", f"{bh_ret_pct:+.2f}%")
k3.metric("Сделок (смен позиции)", f"{n_switches}")
k4.metric(
    "Издержки на комиссии",
    f"-{tx_total_pct:.2f}%",
    help="0.1% × суммарное изменение доли. Уже вычтено из доходности агента.",
)

# Equity chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=timestamps, y=[(e - 1) * 100 for e in agent_equity],
    mode="lines", line=dict(color="#2ca02c", width=3),
    name="Агент",
))
fig.add_trace(go.Scatter(
    x=timestamps[:len(bh_equity)], y=[(e - 1) * 100 for e in bh_equity],
    mode="lines", line=dict(color="black", width=2, dash="dash"),
    name="Buy & Hold",
))
fig.update_layout(
    height=340,
    yaxis_title="Доходность от старта, %",
    xaxis_title=None,
    hovermode="x unified",
    margin=dict(t=20, l=50, r=20, b=40),
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center"),
)
st.plotly_chart(fig, width="stretch")

# Caveat if period is too short
if period_days < 14:
    st.warning(
        f"⚠ Лента охватывает {period_days:.1f} дней — слишком короткий период "
        f"для статистического вывода. Для оценки стратегии смотри вкладку "
        f"**Валидация** (OOS 2024-2025)."
    )


# ========================================================================
# SECTION 4 — DECISION HISTORY (compact)
# ========================================================================

st.divider()
with st.expander(f"Лента решений ({len(decorated) - 1} предыдущих окон)", expanded=False):
    for b in decorated[1:]:
        dec = b.get("decision")
        ts_str = b["bucket_ts"].strftime("%Y-%m-%d %H:%M UTC")

        if dec is None:
            st.markdown(f"**{ts_str}** · ошибка: {b.get('error', '—')}")
            continue

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

        if dec.votes:
            dom = max(dec.votes.items(), key=lambda kv: kv[1])
            votes_short = f"{dom[1]}/{dec.total_seeds}"
        else:
            votes_short = f"{int(dec.allocation*100)}%"

        cols = st.columns([2, 1, 2, 1, 1])
        cols[0].markdown(f"**{ts_str}**")
        cols[1].markdown(
            f'<span style="background:{action_color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-weight:600;font-size:13px;">{action_icon}</span>',
            unsafe_allow_html=True,
        )
        cols[2].markdown(f"{b['n_news']} нов · sent {b['sentiment_mean']:+.2f}")
        cols[3].markdown(f"согл: **{votes_short}**")
        cols[4].markdown(f"P&L: {pnl_text}", unsafe_allow_html=True)
