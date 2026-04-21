"""Streamlit dashboard — entry point and home screen."""
import sys
from pathlib import Path

# Add dsr_experiment/ to sys.path so `from dashboard.utils...` works when
# streamlit runs this script with `cwd=dsr_experiment/`.
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.utils.paths import DEFAULT_SNAPSHOT, ensure_lib_on_path

ensure_lib_on_path()

st.set_page_config(
    page_title="RL Trading Agents — Thesis Demo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 RL Trading Agents — Thesis Demo")
st.markdown(
    """
    **Тема диплома:** _Оптимизация стратегий торговли с помощью обучения
    с подкреплением и обработки текстовой информации из новостных источников._

    **Актив:** BTC/USDT 4h spot · **Алгоритмы:** SAC (continuous allocation) + DQN (discrete Hold/Buy/Sell)

    ### Страницы

    - **📊 Backtest** — интерактивный просмотр готовых snapshot'ов: equity curves,
      bootstrap 95% CI, per-seed breakdown.
    - **🔮 Live Prediction** — real-time BTC price через Binance + решение агента.
    - **📈 Paper Trading** — step-by-step симуляция агента на историческом OOS.
    - **📰 News Analyzer** — интерпретация новости через FinBERT + PCA.

    ---
    """
)

st.info(
    f"🔹 Primary snapshot: `{DEFAULT_SNAPSHOT}` (10 сидов SAC + 10 сидов DQN)\n\n"
    "🔹 Все операции read-only. Реальная торговля не выполняется."
)

st.caption(
    "Используйте боковое меню слева для навигации между страницами. "
    "Первая загрузка моделей и pipeline может занять ~30 секунд."
)
