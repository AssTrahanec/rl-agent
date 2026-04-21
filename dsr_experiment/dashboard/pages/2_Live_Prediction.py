"""Live Prediction — fetch real BTC OHLCV and show model decision."""
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
from dashboard.utils import snapshot
from dashboard.utils.model_catalog import list_model_entries
from dashboard.utils.live_data import fetch_live_ohlcv
from dashboard.utils.features_live import build_live_features, build_live_obs
from dashboard.utils.model_loader import load_sb3_model

ensure_lib_on_path()


st.set_page_config(page_title="Live Prediction", layout="wide")
st.title("Live Prediction")


# ---- Controls ----
entries = list_model_entries()
if not entries:
    st.error("No trained models.")
    st.stop()

default_label = next(
    (e.label for e in entries if "tuned + 10 seeds" in e.label and e.algo == "DQN"),
    entries[0].label,
)
labels = [e.label for e in entries]

c1, c2, c3 = st.columns([3, 2, 1])
with c1:
    picked_label = st.selectbox("Модель", labels, index=labels.index(default_label))
with c2:
    entry = next(e for e in entries if e.label == picked_label)
    models_for_entry = snapshot.discover_models(entry.snapshot).get(entry.algo, [])
    seeds_avail = [m["seed"] for m in models_for_entry]
    default_seed = 7 if 7 in seeds_avail else seeds_avail[0]
    seed = st.selectbox("Seed", seeds_avail, index=seeds_avail.index(default_seed))
with c3:
    use_live = st.toggle("Live", value=True, help="Binance через ccxt, иначе кэш")


# ---- Fetch OHLCV ----
try:
    ohlcv, source = fetch_live_ohlcv(use_live=use_live, lookback_bars=120)
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot load OHLCV: {e}")
    st.stop()

last_price = float(ohlcv["close"].iloc[-1])
last_ts = ohlcv.index[-1]
price_24h_ago = float(ohlcv["close"].iloc[-7]) if len(ohlcv) >= 7 else last_price
change_24h = (last_price / price_24h_ago - 1) * 100

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("BTC/USDT", f"${last_price:,.0f}", f"{change_24h:+.1f}% (24h)")
kpi2.metric("Последний бар", str(last_ts)[:16])
kpi3.metric(
    "Источник",
    "Binance live" if source == "live" else "Кэш",
)

if source != "live" and use_live:
    st.info("Binance недоступна, используется локальный кэш.")


# ---- Candlestick chart ----
fig = go.Figure(data=go.Candlestick(
    x=ohlcv.index[-80:],
    open=ohlcv["open"].iloc[-80:],
    high=ohlcv["high"].iloc[-80:],
    low=ohlcv["low"].iloc[-80:],
    close=ohlcv["close"].iloc[-80:],
    increasing_line_color="#2ca02c",
    decreasing_line_color="#d62728",
))
fig.update_layout(
    height=380,
    xaxis_rangeslider_visible=False,
    margin=dict(t=20, l=40, r=20, b=40),
    yaxis_title="Цена USDT",
)
st.plotly_chart(fig, width="stretch")


# ---- Prediction ----
st.subheader("Решение модели")

if st.button("Получить решение", type="primary"):
    with st.spinner("Считаю features и прогоняю модель..."):
        try:
            features, prices, _ = build_live_features(ohlcv)
            obs = build_live_obs(features, prev_allocation=0.0, window=30)
        except Exception as e:  # noqa: BLE001
            st.error(f"Features build failed: {e}")
            st.stop()

        model_path = next(m["path"] for m in models_for_entry if m["seed"] == seed)
        try:
            model = load_sb3_model(model_path, algo_hint=entry.algo)
        except Exception as e:  # noqa: BLE001
            st.error(f"Model load failed: {e}")
            st.stop()

        action, _ = model.predict(obs, deterministic=True)

    if entry.algo == "DQN":
        a = int(np.asarray(action).flatten()[0]) if not np.isscalar(action) else int(action)
        labels_map = {0: ("HOLD", "#6c757d"), 1: ("BUY", "#2ca02c"), 2: ("SELL", "#d62728")}
        label_text, color = labels_map.get(a, (f"action {a}", "#6c757d"))

        st.markdown(
            f"""
            <div style="padding:24px;border-radius:8px;background:{color};
                        color:white;text-align:center;font-size:32px;font-weight:bold;">
                {label_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Q-values
        try:
            import torch
            obs_t = torch.as_tensor(obs).float().unsqueeze(0)
            with torch.no_grad():
                q = model.q_net(obs_t).numpy().flatten()
            q_df = pd.DataFrame({
                "Действие": ["HOLD", "BUY", "SELL"],
                "Q-value": q,
            })
            fig_q = go.Figure(go.Bar(
                x=q_df["Действие"], y=q_df["Q-value"],
                marker_color=["#6c757d", "#2ca02c", "#d62728"],
            ))
            fig_q.update_layout(
                height=260, margin=dict(t=10, l=40, r=20, b=40),
                yaxis_title="Q-value",
            )
            st.plotly_chart(fig_q, width="stretch")
        except Exception:  # noqa: BLE001
            pass

    else:
        alloc = float(np.clip(np.asarray(action).flatten()[0], 0.0, 1.0))
        if alloc > 0.7:
            label_text, color = f"LONG {alloc*100:.0f}%", "#2ca02c"
        elif alloc > 0.3:
            label_text, color = f"PARTIAL {alloc*100:.0f}%", "#ffc107"
        else:
            label_text, color = "CASH", "#6c757d"
        st.markdown(
            f"""
            <div style="padding:24px;border-radius:8px;background:{color};
                        color:white;text-align:center;font-size:32px;font-weight:bold;">
                {label_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(alloc, text=f"Доля в BTC: {alloc*100:.1f}%")

    st.caption(
        "Модель видит последние 30 баров цены и технических индикаторов. "
        "News/sentiment фичи обнулены — для полного inference нужен news pipeline."
    )

    if entry.algo == "SAC" and entry.snapshot == "run_2026-04-21_10seeds":
        st.info(
            "SAC в snapshot '10 seeds' обучался без сохранённого VecNormalize. "
            "Inference без нормализации — поведение совпадает с OOS-прогоном."
        )
