"""Backtest tab — simple equity comparison."""
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

from dashboard.utils.paths import DEFAULT_SNAPSHOT, OOS_DIR, ensure_lib_on_path
from dashboard.utils import snapshot
from dashboard.utils.theme import ALGO_COLORS, BH_COLOR

ensure_lib_on_path()
from lib.metrics import compute_metrics


st.set_page_config(page_title="Backtest", layout="wide")
st.title("Backtest")


# ---- Controls ----
snaps = snapshot.list_snapshots()
if not snaps:
    st.error("No snapshots found.")
    st.stop()

c1, c2 = st.columns([2, 1])
with c1:
    snap_name = st.selectbox(
        "Snapshot",
        snaps,
        index=snaps.index(DEFAULT_SNAPSHOT) if DEFAULT_SNAPSHOT in snaps else 0,
    )
with c2:
    periods = snapshot.list_periods(snap_name)
    if not periods:
        st.error(f"No OOS periods in `{snap_name}`.")
        st.stop()
    period = st.selectbox("Period", periods, index=0)


# ---- Data ----
@st.cache_data
def buy_hold(period_key, warmup=30, tx_cost=0.001):
    path = OOS_DIR / f"{period_key}_features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    prices = df["raw_close"].values[warmup:]
    log_r = np.log(prices[1:] / prices[:-1])
    bh = np.exp(log_r) - 1
    bh[0] -= tx_cost
    bh[-1] -= tx_cost
    eq = np.insert(np.cumprod(1 + bh), 0, 1.0)
    m = compute_metrics(bh)
    m["equity_curve"] = eq
    return m


def load_algo(snap, per, algo):
    rows = []
    for entry in snapshot.discover_models(snap).get(algo, []):
        try:
            d = snapshot.load_seed_npz(snap, per, algo, entry["seed"])
        except FileNotFoundError:
            continue
        d["metrics"] = compute_metrics(d["daily_returns"])
        rows.append(d)
    return rows


models = snapshot.discover_models(snap_name)
algos = sorted(models.keys())
data = {a: load_algo(snap_name, period, a) for a in algos}
bh = buy_hold(period)


# ---- Main chart: equity — 3 lines ----
fig = go.Figure()

if bh is not None:
    fig.add_trace(go.Scatter(
        y=bh["equity_curve"],
        mode="lines",
        line=dict(color=BH_COLOR, width=2, dash="dash"),
        name="Buy & Hold",
    ))

for algo in algos:
    rows = data[algo]
    if not rows:
        continue
    L = min(len(r["equity_curve"]) for r in rows)
    stack = np.array([r["equity_curve"][:L] for r in rows])
    median = np.percentile(stack, 50, axis=0)
    color = ALGO_COLORS.get(algo, "#808080")
    fig.add_trace(go.Scatter(
        y=median,
        mode="lines",
        line=dict(color=color, width=3),
        name=algo,
    ))

fig.update_layout(
    height=500,
    xaxis_title="Bars",
    yaxis_title="Капитал (× от начального)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.5, xanchor="center"),
    margin=dict(t=30, l=50, r=30, b=60),
)
st.plotly_chart(fig, width="stretch")


# ---- Compact metrics ----
def fmt_pct(x):
    return f"{x * 100:+.0f}%"


cols = st.columns(3)

if bh is not None:
    with cols[0]:
        st.markdown("**Buy & Hold**")
        st.metric("Return", fmt_pct(bh["total_return"]))
        st.metric("Макс. просадка", fmt_pct(-bh["max_drawdown"]))

col_idx = 1
for algo in algos:
    rows = data[algo]
    if not rows:
        continue
    if col_idx >= len(cols):
        break
    rets = np.array([r["metrics"]["total_return"] for r in rows])
    dds = np.array([r["metrics"]["max_drawdown"] for r in rows])
    delta_ret = rets.mean() - (bh["total_return"] if bh else 0)
    with cols[col_idx]:
        st.markdown(f"**{algo}**")
        st.metric(
            "Return",
            fmt_pct(rets.mean()),
            delta=f"{delta_ret * 100:+.0f}% vs B&H" if bh else None,
            delta_color="normal",
        )
        st.metric("Макс. просадка", fmt_pct(-dds.mean()))
    col_idx += 1
