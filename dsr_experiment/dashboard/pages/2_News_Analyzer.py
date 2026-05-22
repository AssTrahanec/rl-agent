"""News Analyzer — paste a news article, see how it shifts the model's decision."""
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

_ACTION_COLOR = {"BUY": "#2ca02c", "SELL": "#d62728", "HOLD": "#6c757d"}


def _dominant(votes):
    """Action with the most ensemble votes."""
    return max(votes, key=votes.get)


def _lean(votes):
    """Net bullish lean: BUY votes minus SELL votes."""
    return votes.get("BUY", 0) - votes.get("SELL", 0)


def _decision_card(title, decision):
    votes = decision.get("votes") or {"BUY": 0, "HOLD": 0, "SELL": 0}
    action = _dominant(votes)
    color = _ACTION_COLOR[action]
    st.markdown(
        f'<div style="padding:18px;border-radius:10px;background:{color};color:white;">'
        f'<div style="font-size:13px;opacity:0.9;">{title}</div>'
        f'<div style="font-size:40px;font-weight:800;">{action}</div>'
        f'<div style="font-size:13px;opacity:0.9;">'
        f'BUY {votes["BUY"]} · HOLD {votes["HOLD"]} · SELL {votes["SELL"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Анализатор новости", layout="wide")
st.title("Анализатор новости")
st.caption(
    "Вставьте текст новости — модель покажет, как она меняет торговое решение. "
    "FinBERT обучен на английском: для точной оценки вставляйте англоязычный текст."
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

    # Block 1 — sentiment
    st.divider()
    st.subheader("Оценка новости")
    sent = result["sentiment"]
    if sent > 0.05:
        label, scolor = "Позитивная", "#2ca02c"
    elif sent < -0.05:
        label, scolor = "Негативная", "#d62728"
    else:
        label, scolor = "Нейтральная", "#6c757d"
    st.markdown(
        f'<span style="background:{scolor};color:white;padding:6px 14px;'
        f'border-radius:6px;font-weight:700;">{label}: {sent:+.2f}</span>',
        unsafe_allow_html=True,
    )

    # Block 2 — decision before / after
    st.divider()
    st.subheader("Решение модели")
    cols = st.columns(2)
    with cols[0]:
        _decision_card("Без этой новости", result["decision_without"])
    with cols[1]:
        _decision_card("С этой новостью", result["decision_with"])

    # Block 3 — verdict (based on the BUY-minus-SELL lean shift)
    st.divider()
    votes_without = result["decision_without"].get("votes") or {}
    votes_with = result["decision_with"].get("votes") or {}
    lean_delta = _lean(votes_with) - _lean(votes_without)
    if lean_delta >= 1:
        st.success(
            f"Новость склоняет модель к покупке — перевес BUY вырос "
            f"на {lean_delta} голос(а) ансамбля."
        )
    elif lean_delta <= -1:
        st.warning(
            f"Новость склоняет модель к продаже — перевес BUY упал "
            f"на {abs(lean_delta)} голос(а) ансамбля."
        )
    else:
        st.info(
            "Новость почти не меняет решение — текущий рыночный фон перевешивает."
        )

    # Under the hood
    with st.expander("Под капотом — что подаётся в модель"):
        st.write(f"Тональность FinBERT: `{sent:+.4f}`")
        st.write("Признак news_count установлен в `1` для каждого бара окна")
        st.write(
            f"Эмбеддинг новости пересчитан (64 PCA-компоненты), "
            f"первые 5: `{[round(float(x), 3) for x in result['emb_64'][:5]]}`"
        )
        st.caption(
            "Новость подаётся в окно наблюдения агента (30 баров) как единый "
            "новостной фон. Лаговые новостные признаки оставлены нулевыми."
        )
