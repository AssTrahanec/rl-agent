"""Entry page — comparison of experiment snapshots."""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import OOS_DIR, ensure_lib_on_path
from dashboard.utils import snapshot
from dashboard.utils.model_catalog import list_model_entries

ensure_lib_on_path()
from lib.metrics import compute_metrics
from lib.bootstrap import bootstrap_ci


st.set_page_config(page_title="RL Trading — Overview", layout="wide")

st.title("Overview")


@st.cache_data
def buy_hold_metrics(period: str, warmup: int = 30, tx_cost: float = 0.001):
    path = OOS_DIR / f"{period}_features.parquet"
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
    return compute_metrics(bh)


@st.cache_data
def snapshot_row(snap: str, period: str, algo: str) -> dict:
    """Aggregate Sharpe/Return/MaxDD mean±std and 95% CI for one (snap,period,algo)."""
    models = snapshot.discover_models(snap)
    seeds = [e["seed"] for e in models.get(algo, [])]
    if not seeds:
        return None
    returns_list = []
    metrics_list = []
    for s in seeds:
        try:
            d = snapshot.load_seed_npz(snap, period, algo, s)
        except FileNotFoundError:
            continue
        returns_list.append(d["daily_returns"])
        metrics_list.append(compute_metrics(d["daily_returns"]))
    if not metrics_list:
        return None
    sh = np.array([m["sharpe_ratio"] for m in metrics_list])
    ret = np.array([m["total_return"] for m in metrics_list])
    dd = np.array([m["max_drawdown"] for m in metrics_list])
    ci = bootstrap_ci(returns_list, n_bootstrap=2000, confidence=0.95, seed=42)
    return {
        "n": len(metrics_list),
        "sharpe_mean": float(sh.mean()),
        "sharpe_std": float(sh.std()),
        "ci_lower": ci["sharpe_ratio"]["ci_lower"],
        "ci_upper": ci["sharpe_ratio"]["ci_upper"],
        "return_mean": float(ret.mean()),
        "return_std": float(ret.std()),
        "maxdd_mean": float(dd.mean()),
    }


def vs_bh(ci_lower: float, ci_upper: float, bh_sharpe: float) -> str:
    """Compare CI to B&H: return 'higher' / 'overlap' / 'lower'."""
    if np.isnan(bh_sharpe):
        return "n/a"
    if ci_lower > bh_sharpe:
        return "higher"
    if ci_upper < bh_sharpe:
        return "lower"
    return "overlap"


entries = list_model_entries()
if not entries:
    st.error("No trained models in experiments/.")
    st.stop()

periods_filter = sorted({p for e in entries for p in snapshot.list_periods(e.snapshot)})

# ---- Build comparison rows ----
rows = []
for period in periods_filter:
    bh_m = buy_hold_metrics(period)
    bh_sh = bh_m["sharpe_ratio"] if bh_m else float("nan")
    rows.append({
        "Модель": "Buy & Hold",
        "Период": period,
        "n": 1,
        "Return": f"{bh_m['total_return']*100:+.1f}%" if bh_m else "—",
        "MaxDD": f"{bh_m['max_drawdown']*100:.1f}%" if bh_m else "—",
        "vs B&H": "—",
    })
    for entry in entries:
        if period not in snapshot.list_periods(entry.snapshot):
            continue
        agg = snapshot_row(entry.snapshot, period, entry.algo)
        if agg is None:
            continue
        rows.append({
            "Модель": entry.label,
            "Период": period,
            "n": agg["n"],
            "Return": f"{agg['return_mean']*100:+.1f}% ± {agg['return_std']*100:.1f}%",
            "MaxDD": f"{agg['maxdd_mean']*100:.1f}%",
            "vs B&H": vs_bh(agg["ci_lower"], agg["ci_upper"], bh_sh),
        })

df = pd.DataFrame(rows)


def style_vs(val):
    if val == "higher":
        return "background-color: #d4edda; color: #155724"
    if val == "lower":
        return "background-color: #f8d7da; color: #721c24"
    if val == "overlap":
        return "background-color: #fff3cd; color: #856404"
    return ""


st.dataframe(
    df.style.map(style_vs, subset=["vs B&H"]),
    width="stretch",
    hide_index=True,
)

st.caption(
    f"{len(entries)} моделей × {len(periods_filter)} OOS периодов. "
    "`vs B&H` — сравнение 95% CI Sharpe с Buy & Hold."
)
