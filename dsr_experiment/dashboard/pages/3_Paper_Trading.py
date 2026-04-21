"""Paper Trading — step-by-step simulation of a trained agent on OOS data."""
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

from dashboard.utils.paths import OOS_DIR, ensure_lib_on_path
from dashboard.utils import snapshot
from dashboard.utils.model_catalog import list_model_entries
from dashboard.utils.model_loader import load_sb3_model
from dashboard.utils.theme import BH_COLOR

ensure_lib_on_path()
from lib.data_loader import _PRICE_COLUMNS_EXT
from lib.env import TradingEnv


st.set_page_config(page_title="Paper Trading", layout="wide")
st.title("Paper Trading")
st.caption("Демо-симуляция: агент принимает решения на исторических OOS-данных. Реальная торговля не выполняется.")


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

c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
with c1:
    picked_label = st.selectbox("Модель", labels, index=labels.index(default_label))
    entry = next(e for e in entries if e.label == picked_label)
with c2:
    models_for_entry = snapshot.discover_models(entry.snapshot).get(entry.algo, [])
    seeds_avail = [m["seed"] for m in models_for_entry]
    default_seed = 7 if 7 in seeds_avail else seeds_avail[0]
    seed = st.selectbox("Seed", seeds_avail, index=seeds_avail.index(default_seed))
with c3:
    periods = snapshot.list_periods(entry.snapshot)
    period = st.selectbox("Период", periods, index=len(periods) - 1)  # latest
with c4:
    capital = st.number_input("Капитал $", value=10000, step=1000, min_value=100)


# ---- Load OOS data ----
@st.cache_data
def load_oos_arrays(period_key: str):
    path = OOS_DIR / f"{period_key}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"OOS parquet missing: {path}")
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c.lower() not in _PRICE_COLUMNS_EXT]
    features = df[feature_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    prices = df["raw_close"].to_numpy(dtype=np.float64)
    sentiment = (df["sentiment_mean"].to_numpy(dtype=np.float32)
                 if "sentiment_mean" in df.columns
                 else np.zeros(len(df), dtype=np.float32))
    timestamps = df.index.to_list()
    return features, prices, sentiment, timestamps


def action_space_type_for(entry) -> str:
    """Infer discrete vs continuous from the snapshot's config."""
    import yaml
    snap_dir = Path(_PROJECT_ROOT) / "experiments" / entry.snapshot
    for cfg_path in snap_dir.glob("config*.yaml"):
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            algos_in_cfg = cfg.get("experiment", {}).get("algos", [])
            if entry.algo in algos_in_cfg:
                return cfg.get("env", {}).get("action_space_type", "continuous")
        except Exception:
            continue
    return "continuous"


# ---- Session state ----
SESSION_KEY = "paper"


def reset_session():
    features, prices, sentiment, timestamps = load_oos_arrays(period)
    model_path = next(m["path"] for m in models_for_entry if m["seed"] == seed)
    model = load_sb3_model(model_path, algo_hint=entry.algo)
    ast = action_space_type_for(entry)

    env = TradingEnv(
        features=features, prices=prices,
        window=30, tx_cost=0.001,
        reward_type="dsr", allow_short=False,
        sentiment_signal=sentiment,
        sentiment_lambda=0.3, dsr_eta=0.01,
        action_space_type=ast,
    )
    obs, _ = env.reset()
    st.session_state[SESSION_KEY] = {
        "env": env,
        "model": model,
        "obs": obs,
        "prices": prices,
        "timestamps": timestamps,
        "done": False,
        "equity": [1.0],
        "allocations": [],
        "steps_done": 0,
        "config_key": (entry.key, seed, period, capital),
    }


def step_n(n: int):
    s = st.session_state[SESSION_KEY]
    if s["done"]:
        return
    env = s["env"]
    model = s["model"]
    obs = s["obs"]
    for _ in range(n):
        if s["done"]:
            break
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        alloc = float(info.get("allocation", 0.0))
        log_r = float(info.get("log_return", 0.0))
        # Simple equity accumulation matching backtest.py
        daily = float(np.exp(log_r * alloc) - 1)
        new_eq = s["equity"][-1] * (1 + daily)
        s["equity"].append(new_eq)
        s["allocations"].append(alloc)
        s["steps_done"] += 1
        s["done"] = terminated or truncated
    s["obs"] = obs
    st.session_state[SESSION_KEY] = s


# Auto-reset on config change
cfg_key = (entry.key, seed, period, capital)
if SESSION_KEY not in st.session_state or st.session_state[SESSION_KEY].get("config_key") != cfg_key:
    reset_session()


# ---- Buttons ----
b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
with b1:
    if st.button("Сброс"):
        reset_session()
        st.rerun()
with b2:
    if st.button("+1 бар"):
        step_n(1)
with b3:
    if st.button("+10 баров"):
        step_n(10)
with b4:
    if st.button("До конца", type="primary"):
        with st.spinner("Гоню до конца..."):
            step_n(100000)  # env will terminate


# ---- Current state ----
s = st.session_state[SESSION_KEY]
total_bars = len(s["prices"]) - 30 - 1

k1, k2, k3, k4 = st.columns(4)
k1.metric("Шагов пройдено", f"{s['steps_done']} / {total_bars}")
current_value = capital * s["equity"][-1]
k2.metric(
    "Портфель",
    f"${current_value:,.0f}",
    delta=f"{(s['equity'][-1] - 1) * 100:+.1f}%",
)

if s["allocations"]:
    last_alloc = s["allocations"][-1]
    k3.metric("Текущая позиция", f"{last_alloc*100:.0f}% в BTC")
else:
    k3.metric("Текущая позиция", "—")

peak = max(s["equity"])
dd = (s["equity"][-1] / peak - 1) * 100 if peak > 0 else 0
k4.metric("Текущая просадка", f"{dd:+.1f}%")


# ---- Chart: equity + B&H reference ----
if s["steps_done"] > 0:
    x_range = list(range(len(s["equity"])))

    # B&H for the same prefix
    prices_prefix = s["prices"][30:30 + len(s["equity"])]
    bh_eq = (prices_prefix / prices_prefix[0]).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_range, y=bh_eq, mode="lines",
        line=dict(color=BH_COLOR, dash="dash", width=2),
        name="Buy & Hold",
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=s["equity"], mode="lines",
        line=dict(color="#2ca02c", width=3),
        name="Агент",
    ))
    fig.update_layout(
        height=400,
        xaxis_title="Шаг", yaxis_title="Капитал (× от начального)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(t=20, l=50, r=20, b=60),
    )
    st.plotly_chart(fig, width="stretch")

    # Allocation over time
    fig_alloc = go.Figure()
    fig_alloc.add_trace(go.Scatter(
        x=list(range(len(s["allocations"]))), y=s["allocations"],
        mode="lines", line=dict(color="#1f77b4", width=2),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.3)",
        name="Аллокация",
    ))
    fig_alloc.update_layout(
        height=200,
        xaxis_title="Шаг",
        yaxis_title="Доля в BTC",
        yaxis=dict(range=[-0.05, 1.05]),
        margin=dict(t=20, l=50, r=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_alloc, width="stretch")
else:
    st.info("Нажми «+10 баров» или «До конца» чтобы прогнать агента.")
