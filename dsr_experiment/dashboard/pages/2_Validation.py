"""Валидация — OOS-прогон моделей по сохранённым результатам."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.utils import snapshot, style
from dashboard.utils.model_catalog import list_model_entries
from dashboard.utils.paths import OOS_DIR, ensure_lib_on_path

ensure_lib_on_path()
from lib.metrics import compute_metrics

st.set_page_config(page_title="Валидация", layout="wide")
style.apply()

st.title("Валидация на OOS")
st.caption(
    "Все модели обучены только на 2020–2023. Ниже — детерминированный rollout "
    "на 2024 и 2025 годах. 10 сидов на алгоритм дают распределение."
)


# ── Загрузка данных ──
entries = list_model_entries(all_snapshots=True)
periods = sorted({p for e in entries for p in snapshot.list_periods(e.snapshot)})
if not entries or not periods:
    st.error("Нет данных. Запусти `run.py`.")
    st.stop()


@st.cache_data
def buy_hold(period: str, warmup: int = 30, tx_cost: float = 0.001):
    """Buy & Hold baseline на тех же ценах, что видел агент."""
    path = OOS_DIR / f"{period}_features.parquet"
    if not path.exists():
        return None
    prices = pd.read_parquet(path)["raw_close"].values[warmup:]
    log_r = np.log(prices[1:] / prices[:-1])
    bh = np.exp(log_r) - 1
    bh[0] -= tx_cost
    bh[-1] -= tx_cost
    m = compute_metrics(bh)
    m["equity_curve"] = np.insert(np.cumprod(1 + bh), 0, 1.0)
    return m


def load_seed_results(entry, period):
    """Загрузить .npz файлы по всем сидам для (entry, period)."""
    rows = []
    for m in snapshot.discover_models(entry.snapshot).get(entry.algo, []):
        try:
            d = snapshot.load_seed_npz(entry.snapshot, period, entry.algo, m["seed"])
        except FileNotFoundError:
            continue
        d["metrics"] = compute_metrics(d["daily_returns"])
        rows.append(d)
    return rows


def aggregate_period(period):
    """Собрать сводку по каждой (algo, snapshot) для данного периода."""
    out = []
    for entry in entries:
        if period not in snapshot.list_periods(entry.snapshot):
            continue
        rows = load_seed_results(entry, period)
        if not rows:
            continue
        rets = np.array([r["metrics"]["total_return"] for r in rows])
        sharpes = np.array([r["metrics"]["sharpe_ratio"] for r in rows])
        dds = np.array([r["metrics"]["max_drawdown"] for r in rows])
        out.append({
            "entry": entry,
            "rows": rows,
            "n": len(rows),
            "return_mean": rets.mean(),
            "return_std": rets.std(),
            "sharpe_mean": sharpes.mean(),
            "sharpe_std": sharpes.std(),
            "maxdd_mean": dds.mean(),
        })
    out.sort(key=lambda x: -x["sharpe_mean"])
    return out


def render_period(period: str):
    aggs = aggregate_period(period)
    if not aggs:
        st.warning("Нет данных для этого периода.")
        return

    bh = buy_hold(period)
    best = aggs[0]
    diff = best["return_mean"] - (bh["total_return"] if bh else 0)
    n_pos = sum(1 for r in best["rows"] if r["metrics"]["total_return"] > 0)

    # Главный вывод по периоду
    st.success(
        f"**{best['entry'].label}** — лучшая конфигурация. "
        f"Средняя доходность по {best['n']} сидам: "
        f"**{best['return_mean']*100:+.1f}%** (±{best['return_std']*100:.1f}%), "
        f"Sharpe **{best['sharpe_mean']:+.2f}** (±{best['sharpe_std']:.2f}). "
        f"Прибыльных сидов: **{n_pos} из {best['n']}**. "
        f"vs Buy&Hold: **{diff*100:+.1f}%**."
    )

    # Таблица
    table = []
    if bh:
        table.append({
            "Стратегия": "Buy & Hold",
            "Сидов": "—",
            "Return": f"{bh['total_return']*100:+.1f}%",
            "Sharpe": f"{bh['sharpe_ratio']:+.2f}",
            "MaxDD": f"-{bh['max_drawdown']*100:.1f}%",
            "vs B&H": "—",
        })
    for a in aggs:
        d_vs_bh = a["return_mean"] - (bh["total_return"] if bh else 0)
        table.append({
            "Стратегия": a["entry"].label,
            "Сидов": str(a["n"]),
            "Return": f"{a['return_mean']*100:+.1f}% ± {a['return_std']*100:.1f}%",
            "Sharpe": f"{a['sharpe_mean']:+.2f} ± {a['sharpe_std']:.2f}",
            "MaxDD": f"-{a['maxdd_mean']*100:.1f}%",
            "vs B&H": f"{d_vs_bh*100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

    # Equity curves — медиана по сидам + ribbon P25-P75
    st.subheader("Equity curves")
    st.caption("Жирная линия — медиана по сидам, затенение — P25–P75 (разброс по сидам).")
    fig = go.Figure()
    if bh:
        fig.add_trace(go.Scatter(
            y=(bh["equity_curve"] - 1) * 100, mode="lines",
            line=dict(color=style.TEXT, dash="dash", width=2), name="Buy & Hold",
        ))

    colors = [style.ALGO_COLOR.get(a["entry"].algo, style.ACCENT) for a in aggs]
    for color, a in zip(colors, aggs):
        L = min(len(r["equity_curve"]) for r in a["rows"])
        stack = np.array([r["equity_curve"][:L] for r in a["rows"]])
        p25 = np.percentile(stack, 25, axis=0)
        p50 = np.percentile(stack, 50, axis=0)
        p75 = np.percentile(stack, 75, axis=0)
        x = np.arange(L)
        rgba = f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}"

        # Ribbon (upper bound невидимый → fill до lower bound)
        fig.add_trace(go.Scatter(x=x, y=(p75 - 1) * 100,
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=(p25 - 1) * 100, line=dict(width=0),
                                 fill="tonexty", fillcolor=rgba,
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=(p50 - 1) * 100, mode="lines",
                                 line=dict(color=color, width=2.5), name=a["entry"].label))

    fig.update_layout(height=420, yaxis_ticksuffix="%", plot_bgcolor="white",
                      margin=dict(t=20, l=60, r=20, b=60),
                      legend=dict(orientation="h", yanchor="top", y=-0.1, x=0.5, xanchor="center"))
    st.plotly_chart(fig, width="stretch")


# Табы по периодам
tabs = st.tabs([p.replace("oos_", "OOS ") for p in periods])
for period, tab in zip(periods, tabs):
    with tab:
        render_period(period)
