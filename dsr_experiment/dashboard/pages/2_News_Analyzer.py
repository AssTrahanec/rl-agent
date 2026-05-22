"""News Analyzer — paste a news article, see how it shifts the model's recommendation."""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.utils.paths import ensure_lib_on_path
from dashboard.utils.news_impact import analyze_news_impact

ensure_lib_on_path()

st.set_page_config(page_title="Анализатор новости", layout="wide")
st.title("Анализатор новости")
st.caption(
    "Вставьте новость — модель покажет, как она меняет рекомендуемую долю "
    "капитала в биткоине. FinBERT обучен на английском: вставляйте англоязычный текст."
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

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

st.write("Готовые примеры:")
preset_cols = st.columns(len(PRESETS))
for col, (label, text) in zip(preset_cols, PRESETS.items()):
    if col.button(label, use_container_width=True):
        st.session_state.news_text = text

news_text = st.text_area("Текст новости", height=140, key="news_text")
analyze = st.button("Анализировать", type="primary")


if analyze:
    if not news_text.strip():
        st.warning("Введите текст новости или выберите пример.")
        st.stop()

    with st.spinner("FinBERT оценивает новость, ансамбль из 10 моделей считает решение..."):
        result = analyze_news_impact(news_text)

    # Block 1 — how FinBERT read the news
    st.divider()
    st.subheader("Как FinBERT оценил новость")
    sent = result["sentiment"]
    if sent > 0.05:
        label, color = "Позитивная новость", "#2ca02c"
    elif sent < -0.05:
        label, color = "Негативная новость", "#d62728"
    else:
        label, color = "Нейтральная новость", "#6c757d"
    st.markdown(
        f'<span style="background:{color};color:white;padding:6px 14px;'
        f'border-radius:6px;font-weight:700;font-size:16px;">'
        f'{label} ({sent:+.2f})</span>',
        unsafe_allow_html=True,
    )

    # Block 2 — recommended allocation, before vs after the news
    pct_without = round(result["decision_without"]["allocation"] * 100)
    pct_with = round(result["decision_with"]["allocation"] * 100)
    delta = pct_with - pct_without

    st.divider()
    st.subheader("Рекомендуемая доля капитала в биткоине")
    c1, c2 = st.columns(2)
    c1.metric("Без этой новости", f"{pct_without}%")
    c2.metric("С этой новостью", f"{pct_with}%", delta=f"{delta:+d} п.п.")

    # Block 3 — plain-language verdict
    st.divider()
    if delta >= 5:
        st.success(
            f"Новость повышает рекомендуемую долю на {delta} процентных пункта: "
            f"модель советует увеличить позицию в биткоине."
        )
    elif delta <= -5:
        st.warning(
            f"Новость снижает рекомендуемую долю на {abs(delta)} процентных пункта: "
            f"модель советует сократить позицию в биткоине."
        )
    else:
        st.info(
            "Новость почти не меняет рекомендацию: на текущем рынке ценовой тренд "
            "перевешивает влияние одной новости."
        )
