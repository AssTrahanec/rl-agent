"""News Feed — 4h buckets of RSS news + ensemble agent decision + realized P&L."""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import ensure_lib_on_path
from dashboard.utils import snapshot
from dashboard.utils.model_catalog import list_model_entries
from dashboard.utils.news_feed import refresh_feed, load_cached_feed, group_by_bucket
from dashboard.utils.feed_decisions import (
    decide_ensemble_for_buckets, compute_portfolio_trajectory,
)

ensure_lib_on_path()


st.set_page_config(page_title="News Feed", layout="wide")
st.title("Новостная лента")
st.caption(
    "Реальные заголовки BTC из RSS CoinDesk и Cointelegraph, сгруппированные по "
    "4-часовым окнам. Для каждого окна — решение ансамбля всех сидов выбранной "
    "модели и фактический результат сделки на следующем баре."
)


# ---- Controls ----
entries = list_model_entries()
if not entries:
    st.error("No trained models available.")
    st.stop()

default_label = next(
    (e.label for e in entries if "tuned + 10 seeds" in e.label and e.algo == "DQN"),
    entries[0].label,
)
labels = [e.label for e in entries]

c1, c2 = st.columns([4, 1])
with c1:
    picked = st.selectbox("Модель (ансамбль всех сидов)", labels,
                          index=labels.index(default_label))
    entry = next(e for e in entries if e.label == picked)
with c2:
    st.write("")
    st.write("")
    refresh = st.button("Обновить", use_container_width=True)


# ---- Load/refresh news cache ----
cached = load_cached_feed()

now = pd.Timestamp.utcnow()
if cached.empty or refresh:
    need_refresh = True
else:
    latest = pd.to_datetime(cached["ts"].max(), utc=True)
    now_utc = now if now.tz is not None else now.tz_localize("UTC")
    need_refresh = (now_utc - latest) > pd.Timedelta(hours=4)

if need_refresh:
    with st.spinner("Тяну RSS, считаю sentiment..."):
        try:
            cached, n_new = refresh_feed(force=refresh)
            if n_new > 0:
                st.toast(f"Добавлено {n_new} новостей.", icon="✓")
        except Exception as e:  # noqa: BLE001
            st.error(f"Не удалось обновить: {e}")

if cached.empty:
    st.warning("Лента пустая — нажми «Обновить».")
    st.stop()


# ---- Ensemble decisions on all 4h buckets ----
buckets = group_by_bucket(cached, lookback_buckets=15)
models_avail = snapshot.discover_models(entry.snapshot).get(entry.algo, [])
seeds_paths = [(m["seed"], m["path"]) for m in models_avail]

with st.spinner(f"Прогоняю {len(seeds_paths)} сидов модели {entry.algo} по {len(buckets)} окнам..."):
    decorated = decide_ensemble_for_buckets(buckets, seeds_paths, entry.algo)
    decorated = compute_portfolio_trajectory(decorated, initial_capital=10000.0)


# ---- Portfolio summary at top ----
final_equity = decorated[0].get("portfolio_value", 10000.0) if decorated else 10000.0
start_equity = 10000.0
total_return = (final_equity / start_equity - 1) * 100
n_closed = sum(1 for b in decorated if b.get("decision") and b["decision"].trade_pnl is not None)
profitable = sum(
    1 for b in decorated
    if b.get("decision") and b["decision"].trade_pnl is not None and b["decision"].trade_pnl > 0
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Окон в ленте", f"{len(decorated)}")
k2.metric("Капитал", f"${final_equity:,.0f}", delta=f"{total_return:+.2f}% от $10k")
k3.metric("Прибыльных окон", f"{profitable}/{n_closed}" if n_closed else "—")
k4.metric("Источников", f"{cached['source'].nunique()}")

st.divider()


# ---- Feed rendering ----

def _allocation_bar(pct: float, label: str = "", color: str = "#2ca02c") -> str:
    pct_clamped = max(0.0, min(1.0, pct))
    width = int(pct_clamped * 100)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="flex:1;height:18px;background:#e9ecef;border-radius:4px;overflow:hidden;">'
        f'<div style="width:{width}%;height:100%;background:{color};"></div>'
        f'</div>'
        f'<div style="min-width:60px;font-weight:600;color:#495057;">{label}</div>'
        f'</div>'
    )


def _direction_badge(direction: str, pct_change: float) -> str:
    if direction == "increase":
        return (f'<span style="color:#2ca02c;font-weight:700;">▲ '
                f'увеличил на {pct_change*100:.0f}%</span>')
    if direction == "decrease":
        return (f'<span style="color:#d62728;font-weight:700;">▼ '
                f'сократил на {abs(pct_change)*100:.0f}%</span>')
    return '<span style="color:#6c757d;font-weight:700;">● удержал позицию</span>'


def _pnl_label(trade_pnl: float | None, next_bar_return: float | None) -> str:
    if trade_pnl is None:
        return '<span style="color:#888;">ожидание следующего бара</span>'
    pct_move = (np.exp(next_bar_return) - 1) * 100 if next_bar_return is not None else 0
    trade_pct = trade_pnl * 100
    if trade_pnl > 0:
        col = "#2ca02c"
        sign = "+"
    elif trade_pnl < 0:
        col = "#d62728"
        sign = ""
    else:
        col = "#6c757d"
        sign = ""
    return (
        f'<span>цена на следующем баре: {pct_move:+.2f}% · '
        f'результат сделки: <span style="color:{col};font-weight:700;">'
        f'{sign}{trade_pct:.2f}%</span></span>'
    )


def _votes_strip(votes: dict | None, total: int) -> str:
    if not votes:
        return ""
    parts = []
    colors = {"BUY": "#2ca02c", "SELL": "#d62728", "HOLD": "#6c757d"}
    for label in ["BUY", "HOLD", "SELL"]:
        n = votes.get(label, 0)
        if n == 0:
            continue
        parts.append(
            f'<span style="display:inline-block;padding:2px 8px;margin-right:6px;'
            f'border-radius:4px;background:{colors[label]};color:white;font-size:12px;'
            f'font-weight:600;">{label}: {n}</span>'
        )
    return f'<div style="margin-top:4px;">{" ".join(parts)}</div>'


for b in decorated:
    dec = b.get("decision")
    bucket_ts_str = b["bucket_ts"].strftime("%Y-%m-%d %H:%M UTC")

    with st.container(border=True):
        top_col1, top_col2 = st.columns([3, 2])
        with top_col1:
            st.markdown(f"#### {bucket_ts_str}")
            st.caption(f"{b['n_news']} новостей · средний sentiment {b['sentiment_mean']:+.2f}")

        with top_col2:
            if dec is not None:
                value = b.get("portfolio_value", 10000.0)
                # Show portfolio to-date
                st.markdown(
                    f"<div style='text-align:right;font-size:14px;color:#6c757d;'>"
                    f"портфель после окна</div>"
                    f"<div style='text-align:right;font-size:22px;font-weight:700;'>"
                    f"${value:,.0f}</div>",
                    unsafe_allow_html=True,
                )

        if dec is None:
            st.warning(f"Решение не получено: {b.get('error', '—')}")
            continue

        # Allocation before / after bars
        st.markdown("**Позиция в BTC**")
        alloc_cols = st.columns([1, 1])
        with alloc_cols[0]:
            st.markdown(
                _allocation_bar(
                    dec.prev_allocation,
                    f"было {int(dec.prev_allocation * 100)}%",
                    color="#adb5bd",
                ),
                unsafe_allow_html=True,
            )
        with alloc_cols[1]:
            new_color = "#2ca02c" if dec.direction == "increase" else (
                "#d62728" if dec.direction == "decrease" else "#6c757d"
            )
            st.markdown(
                _allocation_bar(
                    dec.allocation,
                    f"стало {int(dec.allocation * 100)}%",
                    color=new_color,
                ),
                unsafe_allow_html=True,
            )

        # Direction + votes
        change = dec.allocation - dec.prev_allocation
        st.markdown(_direction_badge(dec.direction, change), unsafe_allow_html=True)

        if dec.votes:
            st.markdown(
                f"<div style='font-size:13px;color:#6c757d;margin-top:6px;'>"
                f"Ансамбль: {dec.total_seeds} сидов</div>",
                unsafe_allow_html=True,
            )
            st.markdown(_votes_strip(dec.votes, dec.total_seeds), unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='font-size:13px;color:#6c757d;margin-top:6px;'>"
                f"Ансамбль {dec.total_seeds} сидов · средняя доля {dec.allocation*100:.0f}%</div>",
                unsafe_allow_html=True,
            )

        # P&L
        st.markdown(
            f"<div style='margin-top:8px;font-size:14px;'>"
            f"{_pnl_label(dec.trade_pnl, dec.next_bar_return)}</div>",
            unsafe_allow_html=True,
        )

        # News expander
        with st.expander(f"Показать {b['n_news']} новостей"):
            for item in b["items"]:
                s_score = item["sentiment_score"]
                s_label = item["sentiment_label"]
                s_color = "#2ca02c" if s_label == "positive" else (
                    "#d62728" if s_label == "negative" else "#6c757d"
                )
                src = item["source"]
                ts_str = pd.Timestamp(item["ts"]).strftime("%H:%M")
                link = item["link"]
                title = item["title"]
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<span style="display:inline-block;padding:2px 6px;border-radius:3px;'
                    f'background:{s_color};color:white;font-size:11px;font-weight:600;'
                    f'margin-right:6px;">{s_score:+.2f}</span>'
                    f'<a href="{link}" target="_blank" style="font-weight:500;">{title}</a>'
                    f'<div style="color:#888;font-size:12px;margin-top:2px;">'
                    f'{src} · {ts_str} UTC</div></div>',
                    unsafe_allow_html=True,
                )
