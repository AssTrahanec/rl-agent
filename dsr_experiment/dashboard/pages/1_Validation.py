"""Validation — historical OOS comparison with explicit limitations."""
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
from lib.bootstrap import bootstrap_ci


st.set_page_config(page_title="Валидация", layout="wide")
st.title("Валидация стратегий на истории")


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
    returns_list = [r["daily_returns"] for r in rows]
    try:
        ci = bootstrap_ci(returns_list, n_bootstrap=2000, confidence=0.95, seed=42)
        ci_low = ci["sharpe_ratio"]["ci_lower"]
        ci_high = ci["sharpe_ratio"]["ci_upper"]
    except Exception:  # noqa: BLE001
        ci_low = ci_high = None
    aggregated.append({
        "entry": entry,
        "rows": rows,
        "n": len(rows),
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std()),
        "maxdd_mean": float(dds.mean()),
        "sharpe_mean": float(sharpes.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
    })

aggregated.sort(key=lambda x: -x["sharpe_mean"])


# ---- Main finding banner ----
best = aggregated[0] if aggregated else None
if best and bh:
    best_sh = best["sharpe_mean"]
    bh_sh = bh["sharpe_ratio"]
    ci_low = best["ci_low"]
    ci_high = best["ci_high"]

    if ci_low is not None and ci_low > bh_sh:
        verdict_color = "#2ca02c"
        verdict = "СТАТИСТИЧЕСКИ ЗНАЧИМО ЛУЧШЕ"
        detail = (
            f"Sharpe лучшей модели **{best['entry'].label}** = {best_sh:+.2f}, "
            f"95% CI [{ci_low:+.2f}, {ci_high:+.2f}]. "
            f"Нижняя граница CI ({ci_low:+.2f}) > B&H Sharpe ({bh_sh:+.2f}) — "
            f"различие значимо на уровне p < 0.05 (bootstrap, n={best['n']} сидов)."
        )
    elif ci_low is not None and ci_high < bh_sh:
        verdict_color = "#d62728"
        verdict = "ХУЖЕ BUY & HOLD"
        detail = (
            f"Sharpe {best_sh:+.2f}, 95% CI [{ci_low:+.2f}, {ci_high:+.2f}] — "
            f"весь CI ниже B&H ({bh_sh:+.2f})."
        )
    else:
        verdict_color = "#ffc107"
        verdict = "СРАВНИМО С BUY & HOLD"
        detail = (
            f"Sharpe {best_sh:+.2f}, CI [{ci_low:+.2f}, {ci_high:+.2f}] "
            f"включает B&H ({bh_sh:+.2f}) — статистической разницы нет."
        )

    st.markdown(
        f"""
        <div style="padding:20px;border-radius:10px;background:{verdict_color};color:white;">
            <div style="font-size:12px;opacity:0.85;margin-bottom:4px;">ГЛАВНЫЙ ВЫВОД — {period}</div>
            <div style="font-size:24px;font-weight:800;margin-bottom:8px;">
                {verdict}
            </div>
            <div style="font-size:14px;opacity:0.95;">
                {detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()
st.subheader("Итоги по моделям")

table_rows = []
if bh is not None:
    table_rows.append({
        "Стратегия": "Buy & Hold",
        "Сидов": 1,
        "Return": f"{bh['total_return']*100:+.1f}%",
        "MaxDD": f"{bh['max_drawdown']*100:.1f}%",
        "Sharpe": f"{bh['sharpe_ratio']:+.2f}",
        "95% CI": "—",
        "vs B&H": "—",
    })

for agg in aggregated:
    diff = agg["return_mean"] - (bh["total_return"] if bh else 0)
    if agg["ci_low"] is not None:
        ci_str = f"[{agg['ci_low']:+.2f}, {agg['ci_high']:+.2f}]"
    else:
        ci_str = "—"
    table_rows.append({
        "Стратегия": agg["entry"].label,
        "Сидов": agg["n"],
        "Return": f"{agg['return_mean']*100:+.1f}% ± {agg['return_std']*100:.1f}%",
        "MaxDD": f"{agg['maxdd_mean']*100:.1f}%",
        "Sharpe": f"{agg['sharpe_mean']:+.2f}",
        "95% CI": ci_str,
        "vs B&H": f"{diff*100:+.1f}%",
    })

st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)


# ---- Equity chart — top K ----
st.subheader("Сравнение капитала")

top_k = min(4, len(aggregated))
top = aggregated[:top_k]

fig = go.Figure()

if bh is not None:
    fig.add_trace(go.Scatter(
        y=(bh["equity_curve"] - 1) * 100, mode="lines",
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
        y=(median - 1) * 100, mode="lines",
        line=dict(color=palette[i % len(palette)], width=3),
        name=agg["entry"].label,
    ))

fig.update_layout(
    height=420,
    yaxis_title="Доходность от старта, %",
    xaxis_title=None,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.1, x=0.5, xanchor="center"),
    margin=dict(t=20, l=50, r=20, b=60),
)
st.plotly_chart(fig, width="stretch")


# ---- Limitations section ----
st.divider()
with st.expander("Ограничения и контекст валидации", expanded=False):
    st.markdown(
        f"""
        **Период**

        - OOS 2024: весь 2024 год (BTC рос +110% — сильный бычий тренд)
        - OOS 2025: январь–май 2025 (BTC около +5% — боковик)
        - Train: 2020-01 — 2023-12 (4 года)

        **Ансамбль и bootstrap**

        Каждая модель обучена с 10 независимыми сидами (инициализация весов).
        Bootstrap 95% CI построен по 2000 ресэмплов дневных returns —
        показывает **диапазон** Sharpe, в котором с вероятностью 95%
        лежит истинное значение.

        Если нижняя граница CI выше Sharpe Buy & Hold — различие
        статистически значимо (p < 0.05).

        **SAC vs DQN — почему DQN побеждает**

        - **DQN** (discrete): Hold / Buy (100%) / Sell (0%). Меньше мелких
          tx-cost потерь, чище входы/выходы.
        - **SAC** (continuous): любая доля 0..1. На одном активе дробные
          аллокации создают больше издержек без выгоды.

        Результат согласуется с литературой 2024-2025 (Jurnal Mandiri, MarketMind,
        Nature Sci Reports): **для single-asset crypto discrete action
        предпочтительнее**.

        **Общие ограничения эксперимента**

        - **Только BTC** — обобщение на другие активы не проверено.
        - **Long-only** — короткие позиции исключены. long-short вариант был
          протестирован (snapshot `long-short +funding`) — PPO провалился
          в боковике, SAC показал близкие метрики, подтверждая что шорты
          на спот-рынке с funding cost невыгодны.
        - **Binance spot**, 4h таймфрейм, tx_cost 0.001.
        - **Данные до июня 2025** — свежий OOS требует обновления news-корпуса.
        """
    )
