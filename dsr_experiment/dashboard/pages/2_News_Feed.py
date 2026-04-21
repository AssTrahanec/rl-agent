"""News Feed — live RSS headlines every 4h with agent decision."""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.utils.paths import ensure_lib_on_path
from dashboard.utils import snapshot
from dashboard.utils.model_catalog import list_model_entries
from dashboard.utils.news_feed import refresh_feed, load_cached_feed, group_by_bucket
from dashboard.utils.feed_decisions import decide_for_buckets

ensure_lib_on_path()


st.set_page_config(page_title="News Feed", layout="wide")
st.title("News Feed")
st.caption(
    "Свежие заголовки BTC из RSS CoinDesk и Cointelegraph. "
    "Для каждого 4-часового окна показано решение выбранного агента."
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

c1, c2, c3 = st.columns([3, 2, 1])
with c1:
    picked = st.selectbox("Модель", labels, index=labels.index(default_label))
    entry = next(e for e in entries if e.label == picked)
with c2:
    models_avail = snapshot.discover_models(entry.snapshot).get(entry.algo, [])
    seeds = [m["seed"] for m in models_avail]
    default_seed = 7 if 7 in seeds else seeds[0]
    seed = st.selectbox("Seed", seeds, index=seeds.index(default_seed))
with c3:
    refresh = st.button("Обновить ленту", use_container_width=True)


# ---- Refresh / load feed ----
cached = load_cached_feed()

# Staleness: refresh if cache empty or newest bucket older than 4h
now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
need_refresh = refresh or cached.empty
if not need_refresh and not cached.empty:
    latest_ts = pd.to_datetime(cached["ts"].max(), utc=True)
    if (now - latest_ts) > pd.Timedelta(hours=4):
        need_refresh = True

if need_refresh:
    with st.spinner("Тяну RSS и считаю sentiment..."):
        try:
            cached, n_new = refresh_feed(force=refresh)
            if n_new > 0:
                st.success(f"Добавлено {n_new} новых новостей.")
            elif not cached.empty:
                st.info("Свежих новостей нет.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Не удалось обновить ленту: {e}")

if cached.empty:
    st.warning("Лента пустая — попробуй нажать «Обновить ленту».")
    st.stop()

# Status strip
s1, s2, s3 = st.columns(3)
s1.metric("Всего записей", f"{len(cached)}")
s2.metric("Источников", f"{cached['source'].nunique()}")
latest = pd.to_datetime(cached["ts"].max(), utc=True)
s3.metric("Последняя новость", latest.strftime("%Y-%m-%d %H:%M UTC"))


# ---- Bucket + decide ----
buckets = group_by_bucket(cached, lookback_buckets=15)

with st.spinner("Прогоняю модель по окнам..."):
    model_path = next(m["path"] for m in models_avail if m["seed"] == seed)
    decorated = decide_for_buckets(buckets, model_path, entry.algo)


# ---- Render feed ----
for b in decorated:
    ts = b["bucket_ts"].strftime("%Y-%m-%d %H:%M UTC")
    n = b["n_news"]
    mean_s = b["sentiment_mean"]

    dec = b.get("decision")
    if dec is not None:
        badge_html = (
            f'<span style="padding:4px 14px;border-radius:6px;background:{dec.action_color};'
            f'color:white;font-weight:bold;font-size:14px;">{dec.action_label}</span>'
        )
    else:
        badge_html = (
            '<span style="padding:4px 14px;border-radius:6px;background:#adb5bd;'
            'color:white;font-weight:bold;font-size:14px;">—</span>'
        )

    header_cols = st.columns([1.5, 1, 1, 1.5])
    header_cols[0].markdown(f"**{ts}**")
    header_cols[1].markdown(f"{n} новостей")
    sent_color = "#2ca02c" if mean_s > 0.05 else ("#d62728" if mean_s < -0.05 else "#6c757d")
    header_cols[2].markdown(
        f'<span style="color:{sent_color};font-weight:bold;">sentiment: {mean_s:+.2f}</span>',
        unsafe_allow_html=True,
    )
    header_cols[3].markdown(f"агент: {badge_html}", unsafe_allow_html=True)

    with st.expander(f"Открыть {n} новостей", expanded=False):
        # Q-values for DQN
        if dec is not None and dec.q_values:
            q = dec.q_values
            st.markdown(
                f"**Q-values**: Hold `{q['HOLD']:.3f}` · "
                f"Buy `{q['BUY']:.3f}` · Sell `{q['SELL']:.3f}`"
            )

        for item in b["items"]:
            s_score = item["sentiment_score"]
            s_label = item["sentiment_label"]
            s_color = "#2ca02c" if s_label == "positive" else ("#d62728" if s_label == "negative" else "#6c757d")
            score_badge = (
                f'<span style="color:{s_color};font-weight:bold;">[{s_score:+.2f}]</span>'
            )
            src = item["source"]
            ts_str = pd.Timestamp(item["ts"]).strftime("%H:%M")
            link = item["link"]
            title = item["title"]
            st.markdown(
                f'{score_badge} &nbsp; <a href="{link}" target="_blank">{title}</a>'
                f' &nbsp; <span style="color:#888;">— {src}, {ts_str}</span>',
                unsafe_allow_html=True,
            )
            if item.get("summary"):
                st.caption(item["summary"][:300] + ("…" if len(item["summary"]) > 300 else ""))
        if b.get("error"):
            st.warning(f"Decision failed: {b['error']}")

    st.divider()
