"""Backtest tab — interactive viewer for saved snapshot results."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.utils.paths import DEFAULT_SNAPSHOT, OOS_DIR, ensure_lib_on_path
from dashboard.utils import snapshot
from dashboard.utils.theme import ALGO_COLORS, BH_COLOR

ensure_lib_on_path()

from lib.metrics import compute_metrics
from lib.bootstrap import bootstrap_ci

st.set_page_config(page_title="Backtest", page_icon="📊", layout="wide")

st.title("📊 Backtest Results")
st.caption("Per-seed backtest artifacts from `experiments/` with bootstrap 95% CI.")

# ---- Sidebar controls ----
snaps = snapshot.list_snapshots()
if not snaps:
    st.error("No snapshots found. Run training first via `python run.py`.")
    st.stop()

default_idx = snaps.index(DEFAULT_SNAPSHOT) if DEFAULT_SNAPSHOT in snaps else 0
snap_name = st.sidebar.selectbox("Snapshot", snaps, index=default_idx)

periods = snapshot.list_periods(snap_name)
if not periods:
    st.error(f"No OOS periods in `{snap_name}`. Expected files like oos_oos_*.csv.")
    st.stop()

period = st.sidebar.selectbox("OOS period", periods, index=0)

models = snapshot.discover_models(snap_name)
algos_avail = sorted(models.keys())
if not algos_avail:
    st.error(f"No models found in `{snap_name}/models/`.")
    st.stop()

algos_selected = st.sidebar.multiselect(
    "Algorithms", algos_avail, default=algos_avail
)

# ---- Buy & Hold baseline ----
@st.cache_data
def build_buy_and_hold(period_key: str, warmup: int = 30,
                      tx_cost: float = 0.001) -> dict:
    """Compute B&H daily returns, equity curve, metrics for a given OOS period."""
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


bh = build_buy_and_hold(period)
if bh is None:
    st.warning(f"No OOS parquet for `{period}` — B&H baseline unavailable.")
    bh = {"sharpe_ratio": float("nan"), "total_return": float("nan"),
          "max_drawdown": float("nan"), "calmar_ratio": float("nan"),
          "sortino_ratio": float("nan"),
          "equity_curve": np.array([1.0]), "daily_returns": np.array([])}


# ---- Gather seed data ----
def load_algo_seeds(algo: str) -> list[dict]:
    """Load all available per-seed artifacts for given algo."""
    rows = []
    for entry in models.get(algo, []):
        seed = entry["seed"]
        try:
            d = snapshot.load_seed_npz(snap_name, period, algo, seed)
            d["seed"] = seed
            d["algo"] = algo
            d["metrics"] = compute_metrics(d["daily_returns"])
            rows.append(d)
        except FileNotFoundError:
            pass  # Skip silently; aggregated warnings shown below
    return rows


algo_data: dict[str, list[dict]] = {a: load_algo_seeds(a) for a in algos_selected}

# Warn about missing artifacts
for algo in algos_selected:
    expected = len(models.get(algo, []))
    got = len(algo_data[algo])
    if got < expected:
        st.warning(
            f"{algo}: found {got}/{expected} .npz artifacts. "
            f"Missing seeds excluded from analysis."
        )

# ---- KPI row ----
st.subheader(f"Summary — `{snap_name}` / `{period}`")

n_cols = 1 + len(algos_selected)
cols = st.columns(n_cols)

with cols[0]:
    st.markdown("**Buy & Hold**")
    st.metric("Sharpe", f"{bh['sharpe_ratio']:+.3f}")
    st.metric("Return", f"{bh['total_return']*100:+.1f}%")
    st.metric("MaxDD", f"{bh['max_drawdown']*100:.1f}%")

for i, algo in enumerate(algos_selected):
    with cols[i + 1]:
        rows = algo_data[algo]
        if not rows:
            st.markdown(f"**{algo}** — no data")
            continue
        sharpes = np.array([r["metrics"]["sharpe_ratio"] for r in rows])
        rets = np.array([r["metrics"]["total_return"] for r in rows])
        dds = np.array([r["metrics"]["max_drawdown"] for r in rows])
        st.markdown(f"**{algo}** (n={len(rows)})")
        st.metric("Sharpe", f"{sharpes.mean():+.3f} ± {sharpes.std():.3f}")
        st.metric("Return", f"{rets.mean()*100:+.1f}% ± {rets.std()*100:.1f}%")
        st.metric("MaxDD", f"{dds.mean()*100:.1f}% ± {dds.std()*100:.1f}%")


# ---- Bootstrap CI forest plot ----
st.subheader("Bootstrap 95% confidence intervals — Sharpe")

fig_ci = go.Figure()
labels, centers, lo_err, hi_err, colors_fp = [], [], [], [], []

for algo in algos_selected:
    rows = algo_data[algo]
    if not rows:
        continue
    returns_list = [r["daily_returns"] for r in rows]
    ci = bootstrap_ci(returns_list, n_bootstrap=5000, confidence=0.95, seed=42)
    mean_v = ci["sharpe_ratio"]["mean"]
    lo = ci["sharpe_ratio"]["ci_lower"]
    hi = ci["sharpe_ratio"]["ci_upper"]
    labels.append(f"{algo} (n={len(rows)})")
    centers.append(mean_v)
    lo_err.append(mean_v - lo)
    hi_err.append(hi - mean_v)
    colors_fp.append(ALGO_COLORS.get(algo, "gray"))

if labels:
    fig_ci.add_trace(go.Scatter(
        x=centers, y=labels, mode="markers",
        marker=dict(size=14, color=colors_fp),
        error_x=dict(type="data", symmetric=False,
                     array=hi_err, arrayminus=lo_err, thickness=2),
        showlegend=False,
    ))
    fig_ci.add_vline(x=bh["sharpe_ratio"], line_dash="dash", line_color="red",
                     annotation_text=f"B&H ({bh['sharpe_ratio']:.2f})",
                     annotation_position="top")
    fig_ci.add_vline(x=0, line_color="gray", line_width=1)
    fig_ci.update_layout(
        xaxis_title="Sharpe ratio (bootstrap 95% CI)",
        yaxis_title="",
        height=150 + 60 * len(labels),
        margin=dict(l=140),
    )
    st.plotly_chart(fig_ci, use_container_width=True)

# Interpretation messages
bh_sh = bh["sharpe_ratio"]
for algo in algos_selected:
    rows = algo_data[algo]
    if not rows:
        continue
    returns_list = [r["daily_returns"] for r in rows]
    ci = bootstrap_ci(returns_list, n_bootstrap=5000, confidence=0.95, seed=42)
    lo = ci["sharpe_ratio"]["ci_lower"]
    hi = ci["sharpe_ratio"]["ci_upper"]
    if not np.isnan(bh_sh) and lo > bh_sh:
        st.success(
            f"✅ **{algo}**: CI lower bound `{lo:+.3f}` > B&H `{bh_sh:+.3f}` "
            "— statistically beats Buy & Hold (p < 0.05)."
        )
    elif not np.isnan(bh_sh) and hi < bh_sh:
        st.error(
            f"❌ **{algo}**: CI upper bound `{hi:+.3f}` < B&H `{bh_sh:+.3f}` "
            "— statistically worse than Buy & Hold."
        )
    else:
        st.warning(
            f"⚠️ **{algo}**: CI `[{lo:+.3f}, {hi:+.3f}]` includes B&H `{bh_sh:+.3f}` "
            "— not statistically distinguishable from baseline."
        )

# ---- Equity curves ----
st.subheader("Equity curves")

fig_eq = go.Figure()
for algo in algos_selected:
    rows = algo_data[algo]
    if not rows:
        continue
    for i, r in enumerate(rows):
        fig_eq.add_trace(go.Scatter(
            y=r["equity_curve"], mode="lines",
            line=dict(color=ALGO_COLORS.get(algo, "gray"), width=1),
            opacity=0.35,
            legendgroup=algo,
            showlegend=(i == 0),
            name=algo if i == 0 else None,
            hovertemplate=f"{algo} seed={r['seed']}<br>bar=%{{x}}<br>equity=%{{y:.3f}}<extra></extra>",
        ))

fig_eq.add_trace(go.Scatter(
    y=bh["equity_curve"], mode="lines",
    line=dict(color=BH_COLOR, dash="dash", width=3),
    name="Buy & Hold",
))

fig_eq.update_layout(
    xaxis_title="Bar (4h)",
    yaxis_title="Cumulative value (start=1.0)",
    height=500,
    hovermode="x unified",
)
st.plotly_chart(fig_eq, use_container_width=True)

# ---- Drawdown curves ----
st.subheader("Drawdown curves")

def dd_pct(eq: np.ndarray) -> np.ndarray:
    rm = np.maximum.accumulate(eq)
    return (eq / rm - 1) * 100

fig_dd = go.Figure()
for algo in algos_selected:
    rows = algo_data[algo]
    if not rows:
        continue
    for i, r in enumerate(rows):
        fig_dd.add_trace(go.Scatter(
            y=dd_pct(r["equity_curve"]), mode="lines",
            line=dict(color=ALGO_COLORS.get(algo, "gray"), width=1),
            opacity=0.35,
            legendgroup=algo,
            showlegend=(i == 0),
            name=algo if i == 0 else None,
        ))
fig_dd.add_trace(go.Scatter(
    y=dd_pct(bh["equity_curve"]), mode="lines",
    line=dict(color=BH_COLOR, dash="dash", width=3),
    name="Buy & Hold",
))
fig_dd.add_hline(y=0, line_color="gray", line_width=1)
fig_dd.update_layout(
    xaxis_title="Bar (4h)",
    yaxis_title="Drawdown (%)",
    height=350,
    hovermode="x unified",
)
st.plotly_chart(fig_dd, use_container_width=True)

# ---- Per-seed table ----
st.subheader("Per-seed breakdown")

for algo in algos_selected:
    rows = algo_data[algo]
    if not rows:
        continue
    with st.expander(f"{algo} — {len(rows)} seeds", expanded=False):
        table_rows = []
        for r in rows:
            m = r["metrics"]
            table_rows.append({
                "seed": r["seed"],
                "Sharpe": f"{m['sharpe_ratio']:+.3f}",
                "Return": f"{m['total_return']*100:+.1f}%",
                "MaxDD": f"{m['max_drawdown']*100:.1f}%",
                "Sortino": f"{m['sortino_ratio']:+.3f}",
                "Calmar": f"{m['calmar_ratio']:+.3f}",
            })
        df_seeds = pd.DataFrame(table_rows)
        # Sort by Sharpe descending
        df_seeds["_sort"] = df_seeds["Sharpe"].str.replace("+", "", regex=False).astype(float)
        df_seeds = df_seeds.sort_values("_sort", ascending=False).drop(columns="_sort")
        st.dataframe(df_seeds, use_container_width=True, hide_index=True)
