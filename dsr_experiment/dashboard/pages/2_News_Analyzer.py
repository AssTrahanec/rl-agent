"""News Analyzer — paste news articles, see how they shift the model's recommendation."""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.utils.paths import ensure_lib_on_path
from dashboard.utils.news_impact import analyze_news_impact
from dashboard.utils.news_feed import load_cached_feed

ensure_lib_on_path()

st.set_page_config(page_title="Анализатор новости", layout="wide")
st.title("Анализатор новости")
st.caption(
    "Вставьте новости — по одной на строку. Модель покажет, как новостной фон "
    "меняет её решение. FinBERT обучен на английском — вставляйте англоязычный текст."
)

PRESETS = {
    "SEC одобрила биткоин-ETF": (
        "The SEC has approved the first spot Bitcoin ETF, opening the door "
        "for large institutional investment into cryptocurrency."
    ),
    "Китай запретил майнинг": (
        "China has banned all cryptocurrency mining operations nationwide, "
        "citing energy consumption and financial risk concerns."
    ),
    "Нейтральная новость": (
        "A blockchain technology conference is scheduled to take place "
        "in Europe next month."
    ),
}
_RECENT_LIMIT = 10

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

st.write("Готовые примеры (заменяют поле):")
preset_cols = st.columns(len(PRESETS))
for col, (label, text) in zip(preset_cols, PRESETS.items()):
    if col.button(label, use_container_width=True):
        st.session_state.news_text = text

if st.button("Взять последние новости из ленты", use_container_width=True):
    feed = load_cached_feed()
    if feed.empty:
        st.warning("Лента пуста — открой главную страницу дашборда, чтобы она загрузилась.")
    else:
        recent = feed.sort_values("ts", ascending=False).head(_RECENT_LIMIT)
        lines = [f"{r.title}. {r.summary}".strip() for r in recent.itertuples()]
        st.session_state.news_text = "\n".join(lines)

news_text = st.text_area("Новости (по одной на строку)", height=200, key="news_text")
analyze = st.button("Анализировать", type="primary")


if analyze:
    news_list = [ln.strip() for ln in news_text.splitlines() if ln.strip()]
    if not news_list:
        st.warning("Введите хотя бы одну новость или выберите пример.")
        st.stop()

    with st.spinner(
        f"FinBERT оценивает {len(news_list)} новостей, "
        f"ансамбль из 10 моделей считает решение..."
    ):
        result = analyze_news_impact(news_list)

    # Block 1 — sentiment of each news item
    st.divider()
    st.subheader(f"Оценка новостей ({len(news_list)})")
    for text, sent in result["per_news"]:
        if sent > 0.05:
            color = "#2ca02c"
        elif sent < -0.05:
            color = "#d62728"
        else:
            color = "#6c757d"
        preview = text if len(text) <= 110 else text[:110] + "…"
        st.markdown(
            f'<div style="margin-bottom:6px;">'
            f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-weight:700;font-size:12px;">{sent:+.2f}</span> '
            f'<span style="font-size:14px;">{preview}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Средняя тональность фона: {result['stats']['sentiment_mean']:+.2f}")

    # Block 2 — ensemble vote, before vs after the news
    votes_without = result["decision_without"].get("votes") or {}
    votes_with = result["decision_with"].get("votes") or {}
    total = result["decision_without"].get("total_seeds", 0)
    buy_without = votes_without.get("BUY", 0)
    buy_with = votes_with.get("BUY", 0)

    st.divider()
    st.subheader("Решение ансамбля — сколько моделей за покупку")
    st.caption(
        "Каждая из 10 моделей решает всё-или-ничего: купить на весь капитал "
        "или выйти в кэш. Ниже — сколько проголосовало за покупку."
    )
    c1, c2 = st.columns(2)
    c1.metric("Без новостей", f"{buy_without} из {total}")
    c2.metric("С новостями", f"{buy_with} из {total}",
              delta=f"{buy_with - buy_without:+d}")

    # Block 3 — plain-language verdict
    st.divider()
    diff = buy_with - buy_without
    if diff <= -1:
        st.warning(
            f"Новостной фон переубедил {abs(diff)} модель(и) уйти в кэш — "
            f"совет склонился в сторону продажи."
        )
    elif diff >= 1:
        st.success(
            f"Новостной фон склонил ещё {diff} модель(и) к покупке — "
            f"совет усилился в сторону покупки."
        )
    else:
        st.info(
            "Новостной фон не изменил решение ансамбля — ценовой тренд перевесил."
        )
