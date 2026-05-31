"""Главная страница — обзор и навигация."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard.utils import snapshot, style
from dashboard.utils.model_catalog import list_model_entries
from dashboard.utils.paths import ensure_lib_on_path

ensure_lib_on_path()

st.set_page_config(page_title="Торговый агент BTC", layout="wide")
style.apply()

st.title("Торговый агент BTC/USDT с анализом новостей")
st.caption(
    "Два RL-агента обучены на 4h-данных Binance с DSR-наградой и sentiment-бонусом "
    "от FinBERT. DQN голосует BUY / HOLD / SELL (всё или ничего). "
    "SAC выбирает непрерывную долю 0–100%. Каждый — ансамбль из 10 сидов."
)


# ── Сводка по сохранённым OOS-результатам ──
@st.cache_data(ttl=600)
def load_summary():
    rows = []
    for entry in list_model_entries():
        for period in snapshot.list_periods(entry.snapshot):
            df = snapshot.load_aggregate_csv(entry.snapshot, period)
            df = df[df["algorithm"] == entry.algo]
            if df.empty:
                continue
            rows.append({
                "algo": entry.algo,
                "period": period,
                "n_seeds": len(df),
                "return_mean": df["total_return"].mean(),
                "sharpe_mean": df["sharpe_ratio"].mean(),
                "best_sharpe": df["sharpe_ratio"].max(),
                "positive": (df["total_return"] > 0).sum(),
            })
    return rows


summary = load_summary()
if not summary:
    st.error("Нет данных OOS — запусти `run.py`.")
    st.stop()

total_runs = sum(r["n_seeds"] for r in summary)
positive = sum(r["positive"] for r in summary)
best = max(summary, key=lambda r: r["best_sharpe"])

c1, c2, c3 = st.columns(3)
c1.metric("Лучший Sharpe", f"{best['best_sharpe']:+.2f}",
          f"{best['algo']} · {best['period']}")
c2.metric("Прибыльных моделей", f"{positive} из {total_runs}")
c3.metric("OOS периодов", len({r["period"] for r in summary}))

st.divider()
st.subheader("Разделы")

nav = st.columns(3)
with nav[0]:
    st.page_link("pages/1_Strategy.py", label="**Стратегия в реальном времени**")
    st.caption(
        "Свежие новости BTC, решение ансамбля прямо сейчас, цена и история за 5 дней."
    )
with nav[1]:
    st.page_link("pages/2_Validation.py", label="**Валидация на истории**")
    st.caption(
        "OOS-прогон на 2024 и 2025, сравнение с Buy & Hold, equity-кривые по 10 сидам."
    )
with nav[2]:
    st.page_link("pages/3_News_Analyzer.py", label="**Анализатор новости**")
    st.caption(
        "Вставь любую новость — узнай, как изменится голосование ансамбля."
    )
