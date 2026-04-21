"""Validation — historical OOS comparison of all trained models."""
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
from dashboard.utils.theme import BH_COLOR

ensure_lib_on_path()
from lib.metrics import compute_metrics


st.set_page_config(page_title="Валидация", layout="wide")
st.title("Валидация стратегий на истории")
st.caption(
    "Сравнение обученных моделей и Buy & Hold на out-of-sample периодах. "
    "Цифры усреднены по всем сидам каждой модели."
)


# ---- Controls ----
entries = list_model_entries()
if not entries:
    st.error("Нет обученных моделей.")
    st.stop()

all_periods = sorted({p for e in entries for p in snapshot.list_periods(e.snapshot)})
if not all_periods:
    st.error("Нет OOS периодов.")
    st.stop()

period = st.selectbox("Период", all_periods, index=len(all_periods) - 1)


# ---- Buy & Hold baseline ----
@st.cache_data
def buy_hold(period_key: str, warmup=30, tx_cost=0.001):
    path = OOS_DIR / f"{period_key}_features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    prices = df["raw_close"].values[warmup:]
    log_r = np.log(prices[1:] / prices[:-1])
    bh = np.exp(log_r) - 1
    bh[0] -= tx_cost
    bh[-1] -= tx_cost
    m = compute_metrics(bh)
    m["equity_curve"] = np.insert(np.cumprod(1 + bh), 0, 1.0)
    return m


bh = buy_hold(period)


# ---- Aggregate all models ----
def load_entry(entry):
    rows = []
    for e in snapshot.discover_models(entry.snapshot).get(entry.algo, []):
        try:
            d = snapshot.load_seed_npz(entry.snapshot, period, entry.algo, e["seed"])
        except FileNotFoundError:
            continue
        d["metrics"] = compute_metrics(d["daily_returns"])
        rows.append(d)
    return rows


aggregated = []
for entry in entries:
    if period not in snapshot.list_periods(entry.snapshot):
        continue
    rows = load_entry(entry)
    if not rows:
        continue
    rets = np.array([r["metrics"]["total_return"] for r in rows])
    dds = np.array([r["metrics"]["max_drawdown"] for r in rows])
    sharpes = np.array([r["metrics"]["sharpe_ratio"] for r in rows])
    aggregated.append({
        "entry": entry,
        "rows": rows,
        "n": len(rows),
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std()),
        "maxdd_mean": float(dds.mean()),
        "sharpe_mean": float(sharpes.mean()),
    })

# Sort by return mean descending
aggregated.sort(key=lambda x: -x["return_mean"])

# ---- Summary table ----
st.subheader("Итоги по моделям")

table_rows = []
if bh is not None:
    table_rows.append({
        "Стратегия": "Buy & Hold",
        "Сидов": 1,
        "Return": f"{bh['total_return']*100:+.1f}%",
        "MaxDD": f"{bh['max_drawdown']*100:.1f}%",
        "Sharpe": f"{bh['sharpe_ratio']:+.2f}",
        "vs B&H": "—",
    })

for agg in aggregated:
    diff = agg["return_mean"] - (bh["total_return"] if bh else 0)
    table_rows.append({
        "Стратегия": agg["entry"].label,
        "Сидов": agg["n"],
        "Return": f"{agg['return_mean']*100:+.1f}% ± {agg['return_std']*100:.1f}%",
        "MaxDD": f"{agg['maxdd_mean']*100:.1f}%",
        "Sharpe": f"{agg['sharpe_mean']:+.2f}",
        "vs B&H": f"{diff*100:+.1f}%",
    })

st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)


# ---- Equity chart — top K by return ----
st.subheader("Сравнение капитала")

top_k = min(4, len(aggregated))   # limit for readability
top = aggregated[:top_k]

fig = go.Figure()

if bh is not None:
    fig.add_trace(go.Scatter(
        y=bh["equity_curve"], mode="lines",
        line=dict(color=BH_COLOR, dash="dash", width=2),
        name="Buy & Hold",
    ))

palette = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#8c564b"]
for i, agg in enumerate(top):
    rows = agg["rows"]
    if not rows:
        continue
    L = min(len(r["equity_curve"]) for r in rows)
    stack = np.array([r["equity_curve"][:L] for r in rows])
    median = np.percentile(stack, 50, axis=0)
    fig.add_trace(go.Scatter(
        y=median, mode="lines",
        line=dict(color=palette[i % len(palette)], width=3),
        name=agg["entry"].label,
    ))

fig.update_layout(
    height=440,
    yaxis_title="Капитал × начального",
    xaxis_title=None,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.1, x=0.5, xanchor="center"),
    margin=dict(t=20, l=50, r=20, b=60),
)
st.plotly_chart(fig, width="stretch")


# ---- Key insight ----
if bh and aggregated:
    best = aggregated[0]
    best_ret = best["return_mean"]
    bh_ret = bh["total_return"]
    best_dd = best["maxdd_mean"]
    bh_dd = bh["max_drawdown"]

    lines = [f"**Лучшая модель**: {best['entry'].label}"]
    if best_ret > bh_ret:
        ratio = best_ret / bh_ret if bh_ret > 0.001 else float("inf")
        if ratio > 1.5:
            lines.append(
                f"Доходность {best_ret*100:+.1f}% — в {ratio:.1f}× лучше Buy & Hold ({bh_ret*100:+.1f}%)."
            )
        else:
            lines.append(
                f"Доходность {best_ret*100:+.1f}% vs Buy & Hold {bh_ret*100:+.1f}%."
            )
    else:
        lines.append(
            f"Доходность {best_ret*100:+.1f}% уступает Buy & Hold ({bh_ret*100:+.1f}%)."
        )
    if best_dd < bh_dd:
        lines.append(
            f"Макс. просадка {best_dd*100:.1f}% vs {bh_dd*100:.1f}% у Buy & Hold — "
            f"лучшее управление риском."
        )

    st.info("\n\n".join(lines))
