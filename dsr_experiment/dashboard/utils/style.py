"""Цвета и базовая стилизация Streamlit."""
import streamlit as st

# Цвета для решений
BUY = "#16A34A"     # зелёный
SELL = "#DC2626"    # красный
HOLD = "#64748B"    # серый

# Цвета для подсветок и графиков
ACCENT = "#2563EB"  # синий
TEXT = "#0F172A"    # тёмный текст
MUTED = "#94A3B8"   # приглушённый

ALGO_COLOR = {"DQN": "#16A34A", "SAC": "#2563EB", "PPO": "#D97706"}


def apply():
    """Минимальный CSS — увеличить контейнер и аккуратные метрики."""
    st.markdown(
        """
        <style>
            .main .block-container { max-width: 1100px; padding-top: 2rem; }
            div[data-testid="stMetric"] {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 12px 16px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def decision_badge(verb: str):
    """Большая цветная плашка BUY / HOLD / SELL."""
    color = {"BUY": BUY, "SELL": SELL, "HOLD": HOLD}.get(verb, HOLD)
    st.markdown(
        f"<div style='background:{color};color:white;padding:24px;"
        f"border-radius:12px;text-align:center;font-size:48px;"
        f"font-weight:700;letter-spacing:2px;'>{verb}</div>",
        unsafe_allow_html=True,
    )


def sentiment_badge(score: float) -> str:
    """HTML-тег с цветом по знаку sentiment. Возвращает строку для st.markdown."""
    color = BUY if score > 0.05 else SELL if score < -0.05 else HOLD
    return (
        f"<span style='background:{color};color:white;padding:1px 6px;"
        f"border-radius:3px;font-size:11px;font-weight:600;'>{score:+.2f}</span>"
    )


def news_link(item: dict) -> str:
    """HTML-строка: sentiment-бейдж + кликабельный заголовок (target=_blank) + источник + время.

    Используется как: `st.markdown(news_link(item), unsafe_allow_html=True)`.
    """
    import pandas as pd
    badge = sentiment_badge(item["sentiment_score"])
    ts = pd.Timestamp(item["ts"]).strftime("%H:%M")
    return (
        f"{badge} "
        f"<a href='{item['link']}' target='_blank' "
        f"style='color:{TEXT};text-decoration:none;"
        f"border-bottom:1px solid #E2E8F0;'>{item['title']}</a>"
        f"<span style='color:{MUTED};font-size:12px;'> · {item['source']} · {ts}</span>"
    )
