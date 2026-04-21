"""Backtest tab — compare selected models vs Buy & Hold."""
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
from dashboard.utils.model_catalog import list_model_entries, find_entry
from dashboard.utils.theme import ALGO_COLORS, BH_COLOR, hex_to_rgba

ensure_lib_on_path()
from lib.metrics import compute_metrics


st.set_page_config(page_title="Backtest", layout="wide")
st.title("Backtest")


# ---- Controls ----
entries = list_model_entries()
if not entries:
    st.error("No trained models found under experiments/.")
    st.stop()

all_labels = [e.label for e in entries]

# Default selection: any entry mentioning "10 seeds" (primary snapshot)
default_labels = [lab for lab in all_labels if "10 seeds" in lab]
if not default_labels:
    default_labels = all_labels[:1]

c1, c2 = st.columns([3, 1])
with c1:
    selected_labels = st.multiselect(
        "Модели",
        all_labels,
        default=default_labels,
    )
with c2:
    # Periods present in all selected snapshots (intersection)
    selected_entries = [find_entry(entries, next(e.key for e in entries if e.label == lab))
                        for lab in selected_labels]
    selected_entries = [e for e in selected_entries if e is not None]
    period_sets = [set(snapshot.list_periods(e.snapshot)) for e in selected_entries]
    if period_sets:
        common_periods = sorted(set.intersection(*period_sets))
    else:
        common_periods = []
    if not common_periods:
        st.warning("Выбранные модели не имеют общих OOS периодов.")
        st.stop()
    period = st.selectbox("Период", common_periods, index=0)


# ---- Helpers ----
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


def load_entry_seeds(entry):
    rows = []
    for e in snapshot.discover_models(entry.snapshot).get(entry.algo, []):
        try:
            d = snapshot.load_seed_npz(entry.snapshot, period, entry.algo, e["seed"])
        except FileNotFoundError:
            continue
        d["metrics"] = compute_metrics(d["daily_returns"])
        rows.append(d)
    return rows


# A small palette so multiple entries with the same algo don't clash
ENTRY_PALETTE = [
    "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd",
    "#d62728", "#8c564b", "#e377c2", "#17becf",
]


def color_for_entry(entry, idx: int) -> str:
    """If only one entry per algo, use ALGO_COLORS. Otherwise use palette."""
    return ENTRY_PALETTE[idx % len(ENTRY_PALETTE)]


# ---- Main chart ----
bh = buy_hold(period)

fig = go.Figure()

if bh is not None:
    fig.add_trace(go.Scatter(
        y=bh["equity_curve"],
        mode="lines",
        line=dict(color=BH_COLOR, width=2, dash="dash"),
        name="Buy & Hold",
    ))

entry_results = []  # for metric cards below
for idx, entry in enumerate(selected_entries):
    rows = load_entry_seeds(entry)
    if not rows:
        continue
    L = min(len(r["equity_curve"]) for r in rows)
    stack = np.array([r["equity_curve"][:L] for r in rows])
    median = np.percentile(stack, 50, axis=0)
    color = color_for_entry(entry, idx)
    fig.add_trace(go.Scatter(
        y=median,
        mode="lines",
        line=dict(color=color, width=3),
        name=entry.label,
    ))
    entry_results.append({
        "entry": entry,
        "rows": rows,
        "color": color,
    })

fig.update_layout(
    height=500,
    xaxis_title="Bars (4h)",
    yaxis_title="Капитал (× от начального)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.5, xanchor="center"),
    margin=dict(t=30, l=50, r=30, b=80),
)
st.plotly_chart(fig, width="stretch")


# ---- Metric cards ----
def fmt_pct(x):
    return f"{x * 100:+.0f}%"


n_cards = 1 + len(entry_results) if bh is not None else len(entry_results)
cols = st.columns(n_cards) if n_cards else None

card_idx = 0

if bh is not None and cols is not None:
    with cols[card_idx]:
        st.markdown("**Buy & Hold**")
        st.metric("Return", fmt_pct(bh["total_return"]))
        st.metric("Макс. просадка", fmt_pct(-bh["max_drawdown"]))
    card_idx += 1

for res in entry_results:
    if cols is None or card_idx >= len(cols):
        break
    rows = res["rows"]
    entry = res["entry"]
    rets = np.array([r["metrics"]["total_return"] for r in rows])
    dds = np.array([r["metrics"]["max_drawdown"] for r in rows])
    delta = rets.mean() - (bh["total_return"] if bh else 0)
    with cols[card_idx]:
        st.markdown(f"**{entry.label}**")
        st.metric(
            "Return",
            fmt_pct(rets.mean()),
            delta=f"{delta * 100:+.0f}% vs B&H" if bh is not None else None,
        )
        st.metric("Макс. просадка", fmt_pct(-dds.mean()))
    card_idx += 1
