"""Backtest tab — interactive viewer for saved snapshot results."""
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
from dashboard.utils.theme import ALGO_COLORS, BH_COLOR, hex_to_rgba

ensure_lib_on_path()
from lib.metrics import compute_metrics
from lib.bootstrap import bootstrap_ci


st.set_page_config(page_title="Backtest", layout="wide")
st.title("Backtest")


# ---- Sidebar controls ----
snaps = snapshot.list_snapshots()
if not snaps:
    st.error("No snapshots found. Run training first via `python run.py`.")
    st.stop()

default_idx = snaps.index(DEFAULT_SNAPSHOT) if DEFAULT_SNAPSHOT in snaps else 0
snap_name = st.sidebar.selectbox("Snapshot", snaps, index=default_idx)

periods = snapshot.list_periods(snap_name)
if not periods:
    st.error(f"No OOS periods in `{snap_name}`.")
    st.stop()
period = st.sidebar.selectbox("OOS period", periods, index=0)

models = snapshot.discover_models(snap_name)
algos_avail = sorted(models.keys())
if not algos_avail:
    st.error(f"No models found in `{snap_name}/models/`.")
    st.stop()
algos_selected = st.sidebar.multiselect("Algorithms", algos_avail, default=algos_avail)


# ---- Data loading ----
@st.cache_data
def build_buy_and_hold(period_key: str, warmup: int = 30, tx_cost: float = 0.001):
    path = OOS_DIR / f"{period_key}_features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    prices = df["raw_close"].values[warmup:]
    if len(prices) < 2:
        return None
    log_r = np.log(prices[1:] / prices[:-1])
    bh = np.exp(log_r) - 1
    bh[0] -= tx_cost
    bh[-1] -= tx_cost
    eq = np.insert(np.cumprod(1 + bh), 0, 1.0)
    m = compute_metrics(bh)
    m["equity_curve"] = eq
    m["daily_returns"] = bh
    return m


def load_algo_seeds(snap: str, per: str, algo: str) -> list[dict]:
    rows = []
    for entry in snapshot.discover_models(snap).get(algo, []):
        try:
            d = snapshot.load_seed_npz(snap, per, algo, entry["seed"])
        except FileNotFoundError:
            continue
        d["seed"] = entry["seed"]
        d["metrics"] = compute_metrics(d["daily_returns"])
        rows.append(d)
    return rows


bh = build_buy_and_hold(period)
algo_data: dict[str, list[dict]] = {a: load_algo_seeds(snap_name, period, a) for a in algos_selected}


# ---- vs B&H classifier ----
def vs_bh_label(ci_lower: float, ci_upper: float, bh_sharpe: float) -> str:
    if bh is None or np.isnan(bh_sharpe):
        return "n/a"
    if ci_lower > bh_sharpe:
        return "higher"
    if ci_upper < bh_sharpe:
        return "lower"
    return "overlap"


VS_COLORS = {"higher": "#28a745", "overlap": "#ffc107", "lower": "#dc3545", "n/a": "#6c757d"}


# ---- Build summary table ----
table_rows = []
if bh is not None:
    table_rows.append({
        "Strategy": "Buy & Hold",
        "n": 1,
        "Sharpe": f"{bh['sharpe_ratio']:+.3f}",
        "95% CI": "—",
        "Return": f"{bh['total_return']*100:+.1f}%",
        "MaxDD": f"{bh['max_drawdown']*100:.1f}%",
        "vs B&H": "—",
    })
    bh_sh = bh["sharpe_ratio"]
else:
    bh_sh = float("nan")

algo_ci = {}  # cache CI for plot
for algo in algos_selected:
    rows = algo_data[algo]
    if not rows:
        continue
    returns_list = [r["daily_returns"] for r in rows]
    ci = bootstrap_ci(returns_list, n_bootstrap=2000, confidence=0.95, seed=42)
    sh = np.array([r["metrics"]["sharpe_ratio"] for r in rows])
    ret = np.array([r["metrics"]["total_return"] for r in rows])
    dd = np.array([r["metrics"]["max_drawdown"] for r in rows])
    lo, hi = ci["sharpe_ratio"]["ci_lower"], ci["sharpe_ratio"]["ci_upper"]
    vs = vs_bh_label(lo, hi, bh_sh)
    algo_ci[algo] = {"mean": sh.mean(), "lo": lo, "hi": hi, "vs": vs}
    table_rows.append({
        "Strategy": algo,
        "n": len(rows),
        "Sharpe": f"{sh.mean():+.3f} ± {sh.std():.3f}",
        "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
        "Return": f"{ret.mean()*100:+.1f}% ± {ret.std()*100:.1f}%",
        "MaxDD": f"{dd.mean()*100:.1f}% ± {dd.std()*100:.1f}%",
        "vs B&H": vs,
    })

df_summary = pd.DataFrame(table_rows)


def style_vs(v):
    return f"background-color: {VS_COLORS.get(v, 'transparent')}; color: white" if v in VS_COLORS else ""


st.dataframe(
    df_summary.style.map(style_vs, subset=["vs B&H"]),
    width="stretch",
    hide_index=True,
)


# ---- Figure 1: equity curves (median + IQR band, 3 lines total) ----
st.subheader("Equity curves")

fig_eq = go.Figure()

if bh is not None:
    fig_eq.add_trace(go.Scatter(
        y=bh["equity_curve"], mode="lines",
        line=dict(color=BH_COLOR, dash="dash", width=2),
        name="Buy & Hold",
    ))

for algo in algos_selected:
    rows = algo_data[algo]
    if not rows:
        continue
    # Stack equity curves into (n_seeds, T) for percentile aggregation
    L = min(len(r["equity_curve"]) for r in rows)
    stack = np.array([r["equity_curve"][:L] for r in rows])
    p25 = np.percentile(stack, 25, axis=0)
    p50 = np.percentile(stack, 50, axis=0)
    p75 = np.percentile(stack, 75, axis=0)
    color = ALGO_COLORS.get(algo, "#808080")
    rgba_fill = hex_to_rgba(color, 0.15)
    x = np.arange(L)
    # Upper band
    fig_eq.add_trace(go.Scatter(
        x=x, y=p75, mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))
    # Lower band with fill
    fig_eq.add_trace(go.Scatter(
        x=x, y=p25, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=rgba_fill,
        hoverinfo="skip", name=f"{algo} IQR",
        showlegend=False,
    ))
    # Median line
    fig_eq.add_trace(go.Scatter(
        x=x, y=p50, mode="lines",
        line=dict(color=color, width=2.5),
        name=f"{algo} median (n={len(rows)})",
    ))

fig_eq.update_layout(
    xaxis_title="Bar (4h)", yaxis_title="Cumulative value",
    height=440, hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.15),
)
st.plotly_chart(fig_eq, width="stretch")


# ---- Figure 2: Sharpe distribution per seed ----
st.subheader("Sharpe distribution (per seed)")

fig_box = go.Figure()

for algo in algos_selected:
    rows = algo_data[algo]
    if not rows:
        continue
    sharpes = [r["metrics"]["sharpe_ratio"] for r in rows]
    seeds = [r["seed"] for r in rows]
    color = ALGO_COLORS.get(algo, "gray")
    fig_box.add_trace(go.Box(
        y=sharpes, name=algo, boxpoints="all", jitter=0.4, pointpos=0,
        marker=dict(color=color, size=8),
        line=dict(color=color),
        hovertext=[f"seed={s}" for s in seeds],
        hovertemplate="%{hovertext}<br>Sharpe=%{y:.3f}<extra></extra>",
    ))

if bh is not None:
    fig_box.add_hline(
        y=bh["sharpe_ratio"], line_dash="dash", line_color="red",
        annotation_text=f"B&H {bh['sharpe_ratio']:+.3f}",
        annotation_position="right",
    )
fig_box.add_hline(y=0, line_color="gray", line_width=1)
fig_box.update_layout(
    yaxis_title="Sharpe ratio", height=380,
    showlegend=False,
)
st.plotly_chart(fig_box, width="stretch")


# ---- Figure 3: Bootstrap CI forest plot ----
st.subheader("Bootstrap 95% CI — Sharpe")

fig_ci = go.Figure()
labels, centers, lo_err, hi_err, colors_fp = [], [], [], [], []
for algo in algos_selected:
    if algo not in algo_ci:
        continue
    d = algo_ci[algo]
    labels.append(f"{algo} (n={len(algo_data[algo])})")
    centers.append(d["mean"])
    lo_err.append(d["mean"] - d["lo"])
    hi_err.append(d["hi"] - d["mean"])
    colors_fp.append(VS_COLORS.get(d["vs"], "#6c757d"))

if labels:
    fig_ci.add_trace(go.Scatter(
        x=centers, y=labels, mode="markers",
        marker=dict(size=16, color=colors_fp),
        error_x=dict(type="data", symmetric=False,
                     array=hi_err, arrayminus=lo_err, thickness=2, width=8),
        showlegend=False,
    ))
    if bh is not None:
        fig_ci.add_vline(x=bh_sh, line_dash="dash", line_color="red",
                         annotation_text=f"B&H {bh_sh:+.3f}", annotation_position="top")
    fig_ci.add_vline(x=0, line_color="gray", line_width=1)
    fig_ci.update_layout(
        xaxis_title="Sharpe", yaxis_title="",
        height=120 + 60 * len(labels), margin=dict(l=150),
    )
    st.plotly_chart(fig_ci, width="stretch")


# ---- Per-seed expanders ----
st.subheader("Individual seeds")

for algo in algos_selected:
    rows = algo_data[algo]
    if not rows:
        continue
    with st.expander(f"{algo} — {len(rows)} seeds"):
        table = []
        for r in rows:
            m = r["metrics"]
            table.append({
                "seed": r["seed"],
                "Sharpe": round(m["sharpe_ratio"], 3),
                "Return": round(m["total_return"] * 100, 1),
                "MaxDD": round(m["max_drawdown"] * 100, 1),
                "Sortino": round(m["sortino_ratio"], 3),
                "Calmar": round(m["calmar_ratio"], 3),
            })
        df_seeds = pd.DataFrame(table).sort_values("Sharpe", ascending=False)
        st.dataframe(df_seeds, width="stretch", hide_index=True)
