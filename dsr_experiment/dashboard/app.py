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

st.set_page_config(page_title="Торговый ассистент", layout="wide")
st.title("Торговый ассистент для BTC")
st.caption(
    "Модель читает свежие новости и движения цены, "
    "и каждые 4 часа даёт совет: покупать, продавать или ничего не менять."
)


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
    picked = st.selectbox(
        "Какую модель использовать",
        labels,
        index=labels.index(default_label),
        help="Каждый вариант — это ансамбль из 10 обученных сетей. "
             "Финальный совет выбирается голосованием.",
    )
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
    with st.spinner("Забираю свежие новости и оцениваю настроение..."):
        try:
            cached, n_new, source_tag = refresh_feed(force=do_refresh)
            if n_new > 0:
                st.toast(f"Добавлено {n_new} новостей", icon="✓")
        except Exception as e:  # noqa: BLE001
            st.error(f"Не удалось обновить: {e}")

if cached.empty:
    st.warning("Лента пустая. Нажми «Обновить».")
    st.stop()


# ---- Ensemble over last N buckets ----
buckets = group_by_bucket(cached, lookback_buckets=30)   # up to 5 days of 4h buckets
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
st.subheader("Что модель советует прямо сейчас")
st.caption(f"По данным на {current['bucket_ts'].strftime('%d %B, %H:%M UTC')}")

if current_dec is None:
    st.warning(f"Модель не смогла дать ответ: {current.get('error', '—')}")
else:
    alloc_pct = int(current_dec.allocation * 100)
    prev_pct = int(current_dec.prev_allocation * 100)

    if current_dec.direction == "increase":
        action_verb = "BUY"
        action_phrase = "Сейчас хорошее время докупить BTC"
        action_color = "#2ca02c"
    elif current_dec.direction == "decrease":
        action_verb = "SELL"
        action_phrase = "Лучше выйти из BTC в кэш"
        action_color = "#d62728"
    else:
        action_verb = "HOLD"
        action_phrase = "Держим текущую позицию"
        action_color = "#6c757d"

    if current_dec.votes:
        buy_n = current_dec.votes.get("BUY", 0)
        hold_n = current_dec.votes.get("HOLD", 0)
        sell_n = current_dec.votes.get("SELL", 0)
        parts = []
        if buy_n: parts.append(f"{buy_n} за BUY")
        if hold_n: parts.append(f"{hold_n} за HOLD")
        if sell_n: parts.append(f"{sell_n} за SELL")
        votes_line = "Из 10 моделей: " + ", ".join(parts)
    else:
        votes_line = f"10 моделей в среднем рекомендуют держать {alloc_pct}% в BTC"

    st.markdown(
        f"""
        <div style="padding:28px;border-radius:12px;background:{action_color};color:white;">
            <div style="font-size:14px;opacity:0.8;margin-bottom:6px;">{action_phrase}</div>
            <div style="font-size:56px;font-weight:800;margin-bottom:12px;letter-spacing:2px;">
                {action_verb}
            </div>
            <div style="font-size:18px;opacity:0.95;">
                Было {prev_pct}% в BTC → стало <b>{alloc_pct}%</b>
            </div>
            <div style="font-size:14px;margin-top:10px;opacity:0.85;">
                {votes_line}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**На что модель смотрела**")
    why_cols = st.columns(3)
    why_cols[0].metric("Свежих новостей", current["n_news"])

    mean_s = current["sentiment_mean"]
    if mean_s > 0.15:
        mood = "хороший"
    elif mean_s < -0.15:
        mood = "тревожный"
    else:
        mood = "спокойный"
    why_cols[1].metric("Фон новостей", mood, delta=f"{mean_s:+.2f}")

    if current_dec.votes:
        buy_n = current_dec.votes.get("BUY", 0)
        sell_n = current_dec.votes.get("SELL", 0)
        hold_n = current_dec.votes.get("HOLD", 0)
        why_cols[2].metric(
            "Голоса 10 моделей",
            f"BUY {buy_n} · HOLD {hold_n} · SELL {sell_n}",
        )
    else:
        why_cols[2].metric("Средняя доля в BTC", f"{alloc_pct}%")

    if current["items"]:
        with st.expander(f"Новости ({current['n_news']})", expanded=False):
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
with st.expander("Как работает эта модель", expanded=False):
    st.markdown(
        """
        **Что она делает**

        Каждые 4 часа модель смотрит на цену BTC за последние 5 дней и
        свежие новости, и даёт совет — какую долю капитала держать в BTC
        (остальное — в наличных). Проще говоря: **сейчас покупать, продавать
        или ничего не менять**.

        **Почему 10 моделей, а не одна**

        Мы обучили 10 одинаковых сетей на одинаковых данных, но с разным
        случайным стартом. Это стандартный приём против «везения одного
        запуска» (Henderson et al. 2018). Финальный совет — **голосование
        ансамбля**: если 8 из 10 говорят «покупать» — решение уверенное,
        если голоса разделились — сигнал слабый.

        **Чем отличаются DQN и SAC**

        - **DQN** — может только полностью входить или выходить из BTC.
          Три варианта: купить всё, продать всё, ничего не делать.
        - **SAC** — может держать любую долю от 0% до 100%. Плавные переходы.

        На нашей валидации **DQN оказался лучше** — на одном активе (BTC)
        дробные доли создают лишние комиссии без выгоды, а резкие полные
        покупки/продажи работают чище.

        **Важные оговорки**

        - **Только покупка и кэш.** Коротких позиций (ставок на падение) нет —
          на споте их не бывает, а на фьючерсах ставки за удержание шорта
          делают их невыгодными (мы проверяли).
        - **Модель обучена на данных 2020-2023.** На новых данных 2024-2025
          она показала себя хорошо (смотри «Валидация»), но события последних
          месяцев напрямую она не видела.
        - **Комиссии биржи 0.1% за операцию** учтены при обучении.
        - **Spot-торговля на Binance.** Без плеча и маржинальной торговли.
        """
    )


# ========================================================================
# SECTION 3 — RECENT PERFORMANCE (percent-based)
# ========================================================================

st.divider()
days_covered = int(len(decorated) * 4 / 24)
st.subheader(f"Если бы торговали по советам модели последние {days_covered} дней")
st.caption("Сравнение с вариантом «просто купить BTC в начале и держать».")

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
    "Прибыль по советам модели",
    f"{agent_ret_pct:+.2f}%",
    delta=f"{advantage_pct:+.2f}% против HODL",
)
k2.metric("Если бы просто купили BTC", f"{bh_ret_pct:+.2f}%")
k3.metric("Операций на рынке", f"{n_switches}")
k4.metric(
    "Комиссии биржи",
    f"-{tx_total_pct:.2f}%",
    help="0.1% от объёма каждой операции. Уже вычтено из прибыли.",
)

# Equity chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=timestamps, y=[(e - 1) * 100 for e in agent_equity],
    mode="lines", line=dict(color="#2ca02c", width=3),
    name="По советам модели",
))
fig.add_trace(go.Scatter(
    x=timestamps[:len(bh_equity)], y=[(e - 1) * 100 for e in bh_equity],
    mode="lines", line=dict(color="black", width=2, dash="dash"),
    name="HODL (просто держать BTC)",
))
fig.update_layout(
    height=340,
    yaxis_title="Прибыль с начала периода, %",
    xaxis_title=None,
    hovermode="x unified",
    margin=dict(t=20, l=50, r=20, b=40),
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center"),
)
st.plotly_chart(fig, width="stretch")

# Caveat if period is too short
if period_days < 14:
    st.warning(
        f"Эти {period_days:.1f} дней — слишком короткий срок чтобы делать выводы. "
        f"Для серьёзной оценки модели смотри вкладку **Валидация** "
        f"(там — результаты за 2024 и 2025 год)."
    )


# ========================================================================
# SECTION 4 — DECISION HISTORY (compact)
# ========================================================================

st.divider()
st.subheader("Как модель торговала последние дни")

for b in decorated[1:]:
    dec = b.get("decision")
    ts_str = b["bucket_ts"].strftime("%d %B, %H:%M UTC")

    if dec is None:
        st.markdown(f"**{ts_str}** — модель не смогла дать ответ: {b.get('error', '—')}")
        continue

    prev_pct = int(dec.prev_allocation * 100)
    new_pct = int(dec.allocation * 100)
    change_pct = new_pct - prev_pct

    if dec.direction == "increase":
        action_verb = "BUY"
        action_color = "#2ca02c"
        action_line = f"Докупила BTC: {prev_pct}% → {new_pct}%"
    elif dec.direction == "decrease":
        action_verb = "SELL"
        action_color = "#d62728"
        action_line = f"Сократила BTC: {prev_pct}% → {new_pct}%"
    else:
        action_verb = "HOLD"
        action_color = "#6c757d"
        action_line = f"Оставила {new_pct}% в BTC"

    if dec.votes:
        buy_n = dec.votes.get("BUY", 0)
        hold_n = dec.votes.get("HOLD", 0)
        sell_n = dec.votes.get("SELL", 0)
        vote_parts = []
        if buy_n: vote_parts.append(f"{buy_n} за BUY")
        if hold_n: vote_parts.append(f"{hold_n} за HOLD")
        if sell_n: vote_parts.append(f"{sell_n} за SELL")
        vote_line = "Голоса 10 моделей: " + ", ".join(vote_parts)
    else:
        vote_line = f"В среднем 10 моделей рекомендуют {new_pct}% в BTC"

    if dec.trade_pnl is not None and dec.next_bar_return is not None:
        price_move_pct = (np.exp(dec.next_bar_return) - 1) * 100
        trade_pct = dec.trade_pnl * 100
        if dec.trade_pnl > 0:
            pnl_color = "#2ca02c"
            if price_move_pct > 0:
                outcome_line = (
                    f'BTC за 4 часа <b>+{price_move_pct:.2f}%</b>. '
                    f'Стратегия заработала <b style="color:{pnl_color}">+{trade_pct:.2f}%</b>.'
                )
            else:
                outcome_line = (
                    f'BTC за 4 часа <b>{price_move_pct:.2f}%</b>, но модель была в кэше — '
                    f'<b style="color:{pnl_color}">+{trade_pct:.2f}%</b>.'
                )
        elif dec.trade_pnl < 0:
            pnl_color = "#d62728"
            outcome_line = (
                f'BTC за 4 часа <b>{price_move_pct:+.2f}%</b>. '
                f'Стратегия <b style="color:{pnl_color}">{trade_pct:.2f}%</b> — сигнал не сработал.'
            )
        else:
            outcome_line = f'BTC почти не двигался ({price_move_pct:+.2f}%), без изменений.'
    else:
        outcome_line = "Следующий 4-часовой интервал ещё идёт — результат будет чуть позже."

    mean_s = b["sentiment_mean"]
    if mean_s > 0.15:
        mood_word = "фон хороший"
        mood_color = "#2ca02c"
    elif mean_s < -0.15:
        mood_word = "фон тревожный"
        mood_color = "#d62728"
    else:
        mood_word = "фон спокойный"
        mood_color = "#6c757d"

    with st.container(border=True):
        hdr_cols = st.columns([2, 2])
        hdr_cols[0].markdown(f"**{ts_str}**")
        hdr_cols[1].markdown(
            f'<div style="text-align:right;">'
            f'<span style="background:{action_color};color:white;padding:4px 16px;'
            f'border-radius:6px;font-weight:800;letter-spacing:1px;">{action_verb}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(action_line)
        st.markdown(vote_line)
        st.markdown(outcome_line, unsafe_allow_html=True)

        st.markdown(
            f"**Новости за эти 4 часа** · {b['n_news']} шт · "
            f'<span style="color:{mood_color};font-weight:700;">{mood_word}</span>',
            unsafe_allow_html=True,
        )

        items_to_show = sorted(b["items"], key=lambda x: -abs(x["sentiment_score"]))[:6]
        for item in items_to_show:
            s_label = item["sentiment_label"]
            if s_label == "positive":
                icon = "📈"
            elif s_label == "negative":
                icon = "📉"
            else:
                icon = "●"
            ts_sub = pd.Timestamp(item["ts"]).strftime("%H:%M")
            st.markdown(
                f'<div style="margin-left:10px;margin-bottom:6px;">'
                f'{icon} <a href="{item["link"]}" target="_blank">{item["title"]}</a> '
                f'<span style="color:#888;font-size:12px;">· {item["source"]} · {ts_sub}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if len(b["items"]) > 6:
            extra = len(b["items"]) - 6
            st.caption(f"Ещё {extra}")

