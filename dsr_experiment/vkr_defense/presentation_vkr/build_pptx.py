"""Generate skeleton PPTX for VKR defense.

Usage:
    python build_pptx.py

Writes VKR_Presentation.pptx in the same folder.
No styling; plain text blocks + placeholders for images.
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt


HERE = Path(__file__).parent
IMG = HERE / "images"
import os
_default_out = HERE / "VKR_Presentation.pptx"
OUT = _default_out
try:
    # if PowerPoint has the file open (lock file exists), write to a _v2 name
    if (HERE / "~$VKR_Presentation.pptx").exists():
        OUT = HERE / "VKR_Presentation_v2.pptx"
except Exception:
    pass

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def add_blank(prs):
    layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(layout)


def add_text(slide, left, top, width, height, text, size=18, bold=False):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
    return tb


def add_placeholder(slide, left, top, width, height, label):
    """Rectangle with dashed-look label to mark where image/chart goes."""
    from pptx.shapes.autoshape import Shape  # noqa
    from pptx.enum.shapes import MSO_SHAPE

    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = _rgb(0xF5, 0xF5, 0xF5)
    shp.line.color.rgb = _rgb(0x99, 0x99, 0x99)
    tf = shp.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"[ {label} ]"
    for r in p.runs:
        r.font.size = Pt(14)
        r.font.italic = True
        r.font.color.rgb = _rgb(0x55, 0x55, 0x55)
    return shp


def _image_size(path):
    """Return (width_px, height_px) of an image on disk."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def add_image_fit(slide, filename, left, top, width, height, subdir="images"):
    """Embed image preserving aspect ratio, centered inside (left, top, width, height).

    Returns True if inserted, False if file missing (then leaves a placeholder).
    """
    img_path = HERE / subdir / filename
    if not img_path.exists():
        add_placeholder(slide, left, top, width, height, f"MISSING: {subdir}/{filename}")
        return False
    iw, ih = _image_size(img_path)
    box_w = width
    box_h = height
    # scale uniformly
    scale = min(box_w / iw, box_h / ih)
    new_w = int(iw * scale)
    new_h = int(ih * scale)
    offset_x = left + (box_w - new_w) // 2
    offset_y = top + (box_h - new_h) // 2
    slide.shapes.add_picture(str(img_path), offset_x, offset_y, width=new_w, height=new_h)
    return True


# Backwards-compatible alias
def add_image_or_placeholder(slide, filename, left, top, width, height, label=None):
    return add_image_fit(slide, filename, left, top, width, height)


def add_table(slide, left, top, width, height, rows_data,
              header_fill=(0x40, 0x55, 0x74), highlight_row=None,
              highlight_fill=(0xFF, 0xE6, 0xB3)):
    """rows_data: list of lists (first row is header)."""
    rows = len(rows_data)
    cols = len(rows_data[0])
    tbl_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    tbl = tbl_shape.table
    for r, row in enumerate(rows_data):
        for c, cell_text in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(cell_text)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(13)
                    if r == 0:
                        run.font.bold = True
                        run.font.color.rgb = _rgb(0xFF, 0xFF, 0xFF)
                    elif highlight_row is not None and r == highlight_row:
                        run.font.bold = True
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = _rgb(*header_fill)
            elif highlight_row is not None and r == highlight_row:
                cell.fill.solid()
                cell.fill.fore_color.rgb = _rgb(*highlight_fill)
    return tbl_shape


def add_timeline(slide, left, top, width, height, segments):
    """segments: list of (label, color, weight). Renders horizontal bars with labels."""
    from pptx.enum.shapes import MSO_SHAPE
    total_weight = sum(w for _, _, w in segments)
    x = left
    bar_h = Inches(0.7)
    bar_top = top + Inches(0.2)
    for label, color, weight in segments:
        seg_w = int(width * weight / total_weight)
        shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, bar_top, seg_w, bar_h)
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(*color)
        shp.line.color.rgb = _rgb(0x33, 0x33, 0x33)
        tf = shp.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = label
        for r in p.runs:
            r.font.size = Pt(14)
            r.font.bold = True
            r.font.color.rgb = _rgb(0xFF, 0xFF, 0xFF)
        x += seg_w


def _rgb(r, g, b):
    from pptx.dml.color import RGBColor
    return RGBColor(r, g, b)


def footnote(slide, text):
    add_text(
        slide,
        Inches(0.3), Inches(7.0),
        Inches(12.7), Inches(0.4),
        text, size=10,
    )


def title(slide, text, num=None):
    if num is not None:
        text = f"Слайд {num}. {text}"
    add_text(
        slide,
        Inches(0.3), Inches(0.15),
        Inches(12.7), Inches(0.7),
        text, size=24, bold=True,
    )


def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # ---------- Слайд 1. Титул ----------
    s = add_blank(prs)
    add_text(
        s, Inches(0.8), Inches(1.2), Inches(11.7), Inches(0.8),
        "МАГИСТЕРСКАЯ ДИССЕРТАЦИЯ", size=16, bold=True,
    )
    add_text(
        s, Inches(0.8), Inches(2.0), Inches(11.7), Inches(2.0),
        "Оптимизация стратегий торговли с помощью\n"
        "обучения с подкреплением и обработки текстовой информации\n"
        "из новостных источников",
        size=28, bold=True,
    )
    add_text(
        s, Inches(0.8), Inches(5.0), Inches(11.7), Inches(1.5),
        "Автор: [ФИО]\nНаучный руководитель: [ФИО, степень, звание]\nКафедра: [название]\n2026 г.",
        size=16,
    )
    add_placeholder(
        s, Inches(10.8), Inches(0.3), Inches(2.2), Inches(1.2),
        "LOGO вуза",
    )

    # ---------- Слайд 2. Актуальность ----------
    s = add_blank(prs)
    title(s, "Актуальность: торговля с учётом новостей — стандарт индустрии", 2)
    add_text(
        s, Inches(0.3), Inches(1.0), Inches(6.5), Inches(5.5),
        "На классическом финансовом рынке крупнейшие\n"
        "инвестиционные фонды более десяти лет используют\n"
        "обработку новостей при принятии торговых решений:\n\n"
        "• Citadel, Two Sigma, Renaissance\n"
        "  применяют методы машинного обучения\n"
        "  к новостям и альтернативным данным.\n\n"
        "• J. P. Morgan публикует собственный\n"
        "  индекс настроения рынка, рассчитанный\n"
        "  по новостям и публикациям в сети.\n\n"
        "Научное направление сформировалось\n"
        "и продолжает активно развиваться.",
        size=15,
    )
    # 4 logos in a 2x2 grid on the right
    logo_w = Inches(2.7)
    logo_h = Inches(1.3)
    col1 = Inches(7.3)
    col2 = Inches(10.2)
    row1 = Inches(1.2)
    row2 = Inches(2.8)
    add_image_fit(s, "citadel.png",     col1, row1, logo_w, logo_h, subdir="images/logos")
    add_image_fit(s, "two_sigma.png",   col2, row1, logo_w, logo_h, subdir="images/logos")
    add_image_fit(s, "renaissance.png", col1, row2, logo_w, logo_h, subdir="images/logos")
    add_image_fit(s, "jpmorgan.png",    col2, row2, logo_w, logo_h, subdir="images/logos")
    # caption under logos
    add_text(
        s, Inches(7.2), Inches(4.3), Inches(5.8), Inches(0.4),
        "Citadel · Two Sigma · Renaissance · J. P. Morgan",
        size=12,
    )
    # volatility chart placeholder below (optional illustrative)
    add_image_fit(
        s, "fang_2022_fig1_growth.png",
        Inches(7.2), Inches(4.8), Inches(5.8), Inches(1.9),
    )
    footnote(
        s,
        "Источники: обзор индустрии J. P. Morgan (2024); "
        "аналитический обзор Peak Frameworks о стратегии Citadel (2024).",
    )

    # ---------- Слайд 3. Пробел на крипторынке ----------
    s = add_blank(prs)
    title(s, "Пробел: для крипторынка метод не адаптирован", 3)
    add_text(
        s, Inches(0.3), Inches(1.0), Inches(6.5), Inches(5.7),
        "Крипторынок принципиально отличается\n"
        "от классического:\n\n"
        "• торги круглосуточно, семь дней в неделю;\n"
        "• слабое регулирование;\n"
        "• волатильность в 3–5 раз выше, чем у акций;\n"
        "• цена резко реагирует на публикации в соцсетях\n"
        "  и новостных лентах.\n\n"
        "При этом системных работ, которые бы объединяли\n"
        "обучение с подкреплением и обработку новостей\n"
        "для торговли криптовалютами, практически нет.\n\n"
        "На этот пробел прямо указывают современные\n"
        "обзоры литературы 2022 и 2024 годов.",
        size=14,
    )
    add_image_or_placeholder(
        s, "btc_vs_spx_volatility.png",
        Inches(7.2), Inches(1.2), Inches(5.8), Inches(4.5),
        "Волатильность BTC vs S&P 500",
    )
    footnote(
        s,
        "Источники: Fang F. и др. Обзор подходов к торговле криптовалютами. "
        "Financial Innovation, 8(13), 2022; "
        "Unnikrishnan A. Торговля на основе новостей с применением RL, 2024.",
    )

    # ---------- Слайд 4. Цель и задачи ----------
    s = add_blank(prs)
    title(s, "Цель и задачи", 4)
    add_text(
        s, Inches(0.3), Inches(1.0), Inches(12.7), Inches(1.8),
        "Цель работы:",
        size=18, bold=True,
    )
    add_text(
        s, Inches(0.3), Inches(1.5), Inches(12.7), Inches(1.5),
        "Разработать метод оптимизации стратегии торговли на основе\n"
        "обучения с подкреплением и обработки текстовой информации\n"
        "из новостных источников.",
        size=18,
    )
    add_text(
        s, Inches(0.3), Inches(3.3), Inches(12.7), Inches(0.5),
        "Задачи:",
        size=18, bold=True,
    )
    add_text(
        s, Inches(0.3), Inches(3.8), Inches(12.7), Inches(3.2),
        "1. Проанализировать существующие стратегии и применение RL + NLP в финансовой индустрии.\n"
        "2. Обосновать архитектуру, объединяющую RL и обработку новостной текстовой информации.\n"
        "3. Собрать котировки и корпус новостей за 2020–2024 годы для выбранного рынка.\n"
        "4. Реализовать торговую среду и обучить агентов с учётом новостного сигнала.\n"
        "5. Провести сравнительный эксперимент на отложенных периодах со статистической проверкой.\n"
        "6. Сопоставить результаты с пассивной стратегией удержания и сформулировать выводы.",
        size=14,
    )
    add_text(
        s, Inches(0.3), Inches(6.7), Inches(12.7), Inches(0.4),
        "Тестовый рынок: BTC/USDT, 4-часовой таймфрейм, 2020–2025 гг.",
        size=12,
    )

    # ---------- Слайд 5. Объект и предмет ----------
    s = add_blank(prs)
    title(s, "Объект и предмет исследования", 5)
    add_text(
        s, Inches(0.5), Inches(2.0), Inches(6.0), Inches(0.6),
        "Объект", size=22, bold=True,
    )
    add_text(
        s, Inches(0.5), Inches(2.8), Inches(6.0), Inches(2.5),
        "Торговые стратегии на финансовом рынке.",
        size=18,
    )
    add_text(
        s, Inches(6.8), Inches(2.0), Inches(6.0), Inches(0.6),
        "Предмет", size=22, bold=True,
    )
    add_text(
        s, Inches(6.8), Inches(2.8), Inches(6.0), Inches(2.5),
        "Методы оптимизации стратегий на базе\n"
        "обучения с подкреплением и обработки\n"
        "текстовой информации из новостей.",
        size=18,
    )

    # ---------- Слайд 6a. Что известно из литературы ----------
    s = add_blank(prs)
    title(s, "Что известно из литературы", "6a")

    # Left column: 3 claim lines (big, bold)
    add_text(
        s, Inches(0.5), Inches(1.4), Inches(7.0), Inches(1.4),
        "На акциях новости улучшают\nторговые стратегии.",
        size=20, bold=True,
    )
    add_text(
        s, Inches(0.5), Inches(2.7), Inches(7.0), Inches(0.5),
        "Tetlock (2007). Journal of Finance, 62(3).",
        size=12,
    )

    add_text(
        s, Inches(0.5), Inches(3.3), Inches(7.0), Inches(1.4),
        "На крипте интеграция RL и новостей\nостаётся пробелом.",
        size=20, bold=True,
    )
    add_text(
        s, Inches(0.5), Inches(4.6), Inches(7.0), Inches(0.5),
        "Fang и др. (2022). Financial Innovation, 8(13) — обзор 146 работ.",
        size=12,
    )

    add_text(
        s, Inches(0.5), Inches(5.2), Inches(7.0), Inches(1.4),
        "Sharpe-reward устойчивее,\nчем return-reward.",
        size=20, bold=True,
    )
    add_text(
        s, Inches(0.5), Inches(6.5), Inches(7.0), Inches(0.5),
        "Moody & Saffell (2001). IEEE TNN, 12(4).",
        size=12,
    )

    # Right: one illustration from Fang 2022
    add_image_fit(
        s, "fang_2022_fig10_methods.png",
        Inches(7.8), Inches(1.3), Inches(5.2), Inches(5.5),
    )

    # ---------- Слайд 6b. Обоснование архитектуры ----------
    s = add_blank(prs)
    title(s, "Обоснование архитектуры", "6b")

    # Central arrow-header
    add_text(
        s, Inches(0.5), Inches(1.1), Inches(5.5), Inches(0.6),
        "Из литературы следует", size=16, bold=True,
    )
    add_text(
        s, Inches(6.1), Inches(1.1), Inches(0.9), Inches(0.6),
        "→", size=22, bold=True,
    )
    add_text(
        s, Inches(7.1), Inches(1.1), Inches(5.9), Inches(0.6),
        "Мы сделали", size=16, bold=True,
    )

    # three text pairs — just text, no cards
    rows = [
        (
            "Направление зрелое на акциях\n(Tetlock 2007).",
            "Переносим методологию\nна криптовалютный рынок — BTC.",
        ),
        (
            "На крипте RL + NLP — пробел\n(Fang 2022).",
            "Состояние: цены + FinBERT sentiment\n+ FinLang embeddings (PCA 768→64).",
        ),
        (
            "Sharpe-reward устойчивее\nreturn-reward (Moody 2001).",
            "Reward = Differential Sharpe Ratio\n+ sentiment-бонус (λ = 0.3).",
        ),
    ]
    row_top = Inches(1.9)
    row_h = Inches(1.4)
    for i, (left_txt, right_txt) in enumerate(rows):
        top = row_top + row_h * i
        add_text(
            s, Inches(0.5), top, Inches(5.5), Inches(1.2),
            left_txt, size=15,
        )
        add_text(
            s, Inches(6.1), top + Inches(0.25), Inches(0.9), Inches(0.6),
            "→", size=18, bold=True,
        )
        add_text(
            s, Inches(7.1), top, Inches(5.9), Inches(1.2),
            right_txt, size=15, bold=True,
        )

    # bottom conclusion
    add_text(
        s, Inches(0.3), Inches(6.4), Inches(12.7), Inches(0.5),
        "Задачи 1 и 2 решены: подходы проанализированы, архитектура обоснована.",
        size=13, bold=True,
    )

    footnote(
        s,
        "Araci D. (2019) FinBERT. arXiv:1908.10063; Moody J., Saffell M. (2001) IEEE TNN 12(4).",
    )

    # ---------- Слайд 7. Выбор инструментов ----------
    s = add_blank(prs)
    title(s, "Выбранные инструменты", 7)
    # Three columns: logo on top, bold name, then 2-3 lines of explanation
    col_w = Inches(4.1)
    col_gap = Inches(0.1)
    col1 = Inches(0.3)
    col2 = col1 + col_w + col_gap
    col3 = col2 + col_w + col_gap

    # Column 1: FinBERT
    add_image_fit(s, "huggingface.png",
                  col1 + Inches(1.2), Inches(1.2), Inches(1.7), Inches(1.7),
                  subdir="images/logos")
    add_text(
        s, col1, Inches(3.1), col_w, Inches(0.6),
        "FinBERT", size=22, bold=True,
    )
    add_text(
        s, col1, Inches(3.8), col_w, Inches(3.0),
        "Извлекает оценку тональности\n"
        "финансовой новости —\n"
        "число от −1 до +1.\n\n"
        "Источник:\nAraci (2019). arXiv:1908.10063.",
        size=13,
    )

    # Column 2: FinLang embeddings
    add_image_fit(s, "huggingface.png",
                  col2 + Inches(1.2), Inches(1.2), Inches(1.7), Inches(1.7),
                  subdir="images/logos")
    add_text(
        s, col2, Inches(3.1), col_w, Inches(0.6),
        "FinLang embeddings", size=22, bold=True,
    )
    add_text(
        s, col2, Inches(3.8), col_w, Inches(3.0),
        "Превращает заголовок\n"
        "в 768-мерный смысловой вектор.\n"
        "PCA сжимает до 64 признаков\n"
        "для стабильности обучения.\n\n"
        "Источник:\nFinLang (2024). HuggingFace.",
        size=13,
    )

    # Column 3: Stable-Baselines3
    add_image_fit(s, "pytorch.png",
                  col3 + Inches(0.4), Inches(1.2), Inches(1.5), Inches(1.7),
                  subdir="images/logos")
    add_image_fit(s, "sb3_logo.png",
                  col3 + Inches(2.2), Inches(1.2), Inches(1.5), Inches(1.7),
                  subdir="images/logos")
    add_text(
        s, col3, Inches(3.1), col_w, Inches(0.6),
        "Stable-Baselines3", size=22, bold=True,
    )
    add_text(
        s, col3, Inches(3.8), col_w, Inches(3.0),
        "Эталонная, широко цитируемая\n"
        "реализация DQN и SAC на PyTorch —\n"
        "воспроизводимый стандарт в RL.\n\n"
        "Источник:\nRaffin и др. (2021). JMLR 22(268).",
        size=13,
    )

    footnote(
        s,
        "Araci D. (2019) FinBERT. arXiv:1908.10063. Raffin A. et al. (2021) Stable-Baselines3. JMLR 22(268).",
    )

    # ---------- Слайд 8. Данные ----------
    s = add_blank(prs)
    title(s, "Данные", 8)
    add_image_fit(s, "bitcoin.png",
                  Inches(0.5), Inches(1.2), Inches(0.7), Inches(0.7),
                  subdir="images/logos")
    add_image_fit(s, "binance.png",
                  Inches(1.3), Inches(1.2), Inches(0.7), Inches(0.7),
                  subdir="images/logos")
    add_text(
        s, Inches(2.1), Inches(1.3), Inches(5.0), Inches(0.6),
        "Цены", size=20, bold=True,
    )
    add_text(
        s, Inches(0.5), Inches(2.0), Inches(5.8), Inches(3.5),
        "• BTC/USDT, биржа Binance\n"
        "• 4-часовой таймфрейм\n"
        "• 2020-01-01 — 2025-04\n"
        "• ~11 000 баров\n"
        "• 20+ технических индикаторов",
        size=16,
    )
    add_text(
        s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(0.6),
        "Новости", size=20, bold=True,
    )
    add_text(
        s, Inches(6.8), Inches(2.0), Inches(6.0), Inches(3.5),
        "• Корпус edaschau/bitcoin_news (HuggingFace)\n"
        "• ~200 000 публикаций\n"
        "• Дедупликация по cosine similarity (0.85)\n"
        "• Группировка по 4-часовым окнам\n"
        "• FinBERT sentiment + эмбеддинги FinLang",
        size=16,
    )
    add_image_or_placeholder(
        s, "news_histogram.png",
        Inches(0.5), Inches(5.4), Inches(12.3), Inches(1.5),
        "Гистограмма новостей по месяцам",
    )
    footnote(s, "Binance через CCXT; HuggingFace Datasets: edaschau/bitcoin_news.")

    # ---------- Слайд 9. Архитектура ----------
    s = add_blank(prs)
    title(s, "Архитектура метода", 9)
    add_image_or_placeholder(
        s, "pipeline_overview.png",
        Inches(0.5), Inches(1.1), Inches(12.3), Inches(5.5),
        "Схема конвейера (pipeline_overview.png)",
    )
    footnote(
        s,
        "pipeline_overview.png | FinLang. finance-embeddings-investopedia, HuggingFace, 2024.",
    )

    # ---------- Слайд 10. Функция награды ----------
    s = add_blank(prs)
    title(s, "Функция награды", 10)
    # one big formula line, centered
    add_text(
        s, Inches(0.5), Inches(2.0), Inches(12.3), Inches(1.0),
        "R_t  =  DSR_t  +  0.3 · sentiment_t · log_return_t",
        size=32, bold=True,
    )
    # three short explanations
    add_text(
        s, Inches(1.5), Inches(3.6), Inches(10.5), Inches(2.8),
        "•  DSR — дифференциальный коэффициент Шарпа:\n"
        "    онлайн-оптимизация доходности с поправкой на риск.\n\n"
        "•  sentiment · log_return — бонус за согласованность\n"
        "    тона новостей с движением цены.\n\n"
        "•  Транзакционные издержки: 0.1% при изменении позиции.",
        size=16,
    )
    footnote(
        s,
        "Moody J., Saffell M. (2001). Learning to trade via direct reinforcement. IEEE TNN 12(4). "
        "Развёрнутая формула DSR — в backup-слайде.",
    )

    # ---------- Слайд 11. Дизайн эксперимента ----------
    s = add_blank(prs)
    title(s, "Дизайн эксперимента", 11)
    # Timeline: Train 2020-2023 (4 years) | OOS 2024 (1 year) | OOS 2025 (~4 months)
    add_timeline(
        s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(1.2),
        [
            ("Обучение 2020–2023", (0x40, 0x55, 0x74), 4.0),
            ("OOS 2024", (0x2C, 0xA0, 0x2C), 1.0),
            ("OOS 2025", (0xD6, 0x27, 0x28), 0.3),
        ],
    )
    add_text(
        s, Inches(0.5), Inches(2.4), Inches(12.3), Inches(0.5),
        "4 года обучения  →  2 отложенных периода для валидации",
        size=13,
    )
    add_text(
        s, Inches(0.5), Inches(3.5), Inches(12.3), Inches(3.0),
        "• Алгоритмы: SAC (continuous) и DQN (discrete)\n"
        "• 10 случайных начальных состояний (сидов) на алгоритм\n"
        "• 200 000 шагов обучения\n"
        "• Награда: Differential Sharpe Ratio + sentiment-бонус\n"
        "• База сравнения: пассивная стратегия удержания (Buy & Hold)\n"
        "• Метрики: Sharpe, Sortino, Max Drawdown, Calmar, TotalReturn\n"
        "• Статистическая проверка: t-тест, Mann-Whitney, bootstrap 95% CI",
        size=16,
    )

    # ---------- Слайд 12. OOS 2024 ----------
    s = add_blank(prs)
    title(s, "Результаты: OOS 2024 (бычий рынок)", 12)
    add_image_or_placeholder(
        s, "barplot_oos2024.png",
        Inches(0.5), Inches(1.2), Inches(6.5), Inches(5.5),
        "Barplot Sharpe OOS 2024",
    )
    add_text(
        s, Inches(7.3), Inches(1.2), Inches(5.7), Inches(5.5),
        "OOS 2024 (BTC +118%)\n\n"
        "Buy & Hold:\n  Sharpe 1.51  |  MaxDD 30.0%\n\n"
        "DQN (10 сидов):\n  Sharpe 0.74 ± 0.32\n  MaxDD 31.5%\n\n"
        "SAC (10 сидов):\n  Sharpe 0.58 ± 0.17\n  MaxDD 22.9% ← снижение\n\n"
        "Вывод:\n"
        "В бычьей фазе BH лидирует по Sharpe;\n"
        "SAC снижает просадку на 7 п.п.",
        size=14,
    )

    # ---------- Слайд 13. OOS 2025 (главный слайд) ----------
    s = add_blank(prs)
    title(s, "Результаты OOS 2025", 13)

    # Big headline number on top
    add_text(
        s, Inches(0.3), Inches(1.0), Inches(12.7), Inches(0.9),
        "DQN  Sharpe 0.79   vs   Buy & Hold  0.50      Δ = +0.30,  p = 0.002",
        size=24, bold=True,
    )

    # Large equity chart taking 2/3 of the slide
    add_image_fit(
        s, "equity_oos2025.png",
        Inches(0.3), Inches(2.1), Inches(8.6), Inches(4.8),
    )

    # Right column: 3 key numbers
    add_text(
        s, Inches(9.1), Inches(2.3), Inches(4.0), Inches(0.6),
        "Ключевые цифры", size=16, bold=True,
    )
    add_text(
        s, Inches(9.1), Inches(3.1), Inches(4.0), Inches(1.0),
        "MaxDD\n16.5%   vs   30.6%",
        size=16, bold=True,
    )
    add_text(
        s, Inches(9.1), Inches(4.2), Inches(4.0), Inches(1.0),
        "95% CI DQN\n[0.66 ; 0.92]",
        size=16, bold=True,
    )
    add_text(
        s, Inches(9.1), Inches(5.3), Inches(4.0), Inches(1.0),
        "10 сидов,\n200 000 шагов обучения",
        size=16,
    )

    add_text(
        s, Inches(0.3), Inches(7.0), Inches(12.7), Inches(0.4),
        "Максимальная просадка снижена почти вдвое. Главный результат работы.",
        size=13, bold=True,
    )

    # ---------- Слайд 14. Статистическая значимость ----------
    s = add_blank(prs)
    title(s, "Статистическая значимость", 14)

    # Big centered one-liner
    add_text(
        s, Inches(0.5), Inches(1.2), Inches(8.0), Inches(0.8),
        "Результат устойчив к трём независимым тестам",
        size=22, bold=True,
    )

    # Three test lines stacked
    add_text(
        s, Inches(0.8), Inches(2.4), Inches(7.8), Inches(0.8),
        "t-тест (параметрический)", size=16, bold=True,
    )
    add_text(
        s, Inches(0.8), Inches(3.0), Inches(7.8), Inches(0.6),
        "p = 0.002", size=20, bold=True,
    )

    add_text(
        s, Inches(0.8), Inches(3.9), Inches(7.8), Inches(0.8),
        "Mann–Whitney U (непараметрический)", size=16, bold=True,
    )
    add_text(
        s, Inches(0.8), Inches(4.5), Inches(7.8), Inches(0.6),
        "p = 0.001", size=20, bold=True,
    )

    add_text(
        s, Inches(0.8), Inches(5.4), Inches(7.8), Inches(0.8),
        "Bootstrap 95% CI (10 000 итераций)", size=16, bold=True,
    )
    add_text(
        s, Inches(0.8), Inches(6.0), Inches(7.8), Inches(0.6),
        "[0.66 ; 0.92]   нижняя граница > BH = 0.50",
        size=18, bold=True,
    )

    # Forest plot on the right
    add_image_fit(
        s, "forest_plot_ci.png",
        Inches(8.8), Inches(1.5), Inches(4.3), Inches(5.3),
    )

    footnote(
        s,
        "Efron B., Tibshirani R. An Introduction to the Bootstrap. Chapman & Hall, 1994.",
    )

    # ---------- Слайд 15. Поведение агента ----------
    s = add_blank(prs)
    title(s, "Поведение обученного агента", 15)

    # Left: action timeline with caption
    add_image_fit(
        s, "action_timeline.png",
        Inches(0.3), Inches(1.1), Inches(6.3), Inches(4.4),
    )
    add_text(
        s, Inches(0.3), Inches(5.6), Inches(6.3), Inches(0.5),
        "Временная шкала действий DQN, OOS 2025",
        size=13, bold=True,
    )
    add_text(
        s, Inches(0.3), Inches(6.1), Inches(6.3), Inches(1.0),
        "Агент держит позицию ~38% времени;\n"
        "выходит из рынка в фазах коррекции —\n"
        "за счёт этого снижается просадка.",
        size=13,
    )

    # Right: radar with caption
    add_image_fit(
        s, "radar_profile.png",
        Inches(6.9), Inches(1.1), Inches(6.1), Inches(4.4),
    )
    add_text(
        s, Inches(6.9), Inches(5.6), Inches(6.1), Inches(0.5),
        "Профиль метрик DQN vs Buy & Hold",
        size=13, bold=True,
    )
    add_text(
        s, Inches(6.9), Inches(6.1), Inches(6.1), Inches(1.0),
        "По всем четырём метрикам — Sharpe,\n"
        "Sortino, Calmar и снижение MaxDD —\n"
        "DQN обыгрывает пассивное удержание.",
        size=13,
    )

    # ---------- Слайд 16. Live-дашборд ----------
    s = add_blank(prs)
    title(s, "Работающий прототип: live-дашборд на Streamlit", 16)
    add_placeholder(
        s, Inches(0.5), Inches(1.1), Inches(8.5), Inches(5.7),
        "СКРИНШОТ: dashboard_main.png\n\n"
        "Должно быть видно:\n"
        "  • крупная карточка BUY / HOLD / SELL\n"
        "  • голоса ансамбля 10 моделей\n"
        "  • панель «На что модель смотрела»\n"
        "  • equity-график агент vs HODL",
    )
    add_text(
        s, Inches(9.2), Inches(1.2), Inches(3.8), Inches(5.5),
        "Что делает:\n\n"
        "• Свежие новости → FinBERT + эмбеддинги\n"
        "• Ансамбль 10 моделей (DQN / SAC)\n"
        "• Голосование BUY / HOLD / SELL\n"
        "• Equity vs HODL за последние дни\n\n"
        "Подтверждает работоспособность\n"
        "метода на живых данных.",
        size=12,
    )
    footnote(s, "Streamlit 1.x; обновление раз в 4 часа; ансамблевое голосование.")

    # ---------- Слайд 17. Выводы ----------
    s = add_blank(prs)
    title(s, "Выводы", 17)
    add_text(
        s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.0),
        "1. Разработан и экспериментально валидирован метод оптимизации торговой стратегии,\n"
        "    объединяющий обучение с подкреплением, DSR-награду и обработку новостного текста.\n\n"
        "2. На OOS 2025 DQN статистически значимо превосходит Buy & Hold:\n"
        "    Sharpe 0.79 vs 0.50  (p = 0.002),  MaxDD снижен с 30.6% до 16.5%.\n\n"
        "3. На OOS 2024 (бычий рынок) SAC снижает максимальную просадку с 30% до 23%\n"
        "    при сопоставимой доходности.\n\n"
        "4. Реализован действующий прототип (Streamlit + ансамбль 10 моделей),\n"
        "    демонстрирующий работу метода на свежих новостях.",
        size=16,
    )
    add_text(
        s, Inches(0.5), Inches(6.5), Inches(12.3), Inches(0.5),
        "Цель работы достигнута.  Спасибо за внимание.",
        size=20, bold=True,
    )

    # ========== BACKUP-СЛАЙДЫ (не входят в 17) ==========

    # ---------- B1. Гиперпараметры SAC и DQN ----------
    s = add_blank(prs)
    title(s, "Backup: гиперпараметры агентов")
    hp = [
        ["Параметр", "SAC (непрерывный)", "DQN (дискретный)"],
        ["Learning rate", "7.3·10⁻⁴", "1.0·10⁻⁴"],
        ["Batch size", "256", "256"],
        ["Buffer size", "300 000", "200 000"],
        ["Learning starts", "10 000", "20 000"],
        ["γ (gamma)", "0.99", "0.99"],
        ["τ (tau)", "0.02", "1.0 (hard update)"],
        ["Train freq / grad steps", "8 / 8", "4 / 1"],
        ["Target update interval", "—", "5 000"],
        ["Exploration ε", "—", "1.0 → 0.05 (20% обуч.)"],
        ["Архитектура сети", "MLP [128, 128]", "MLP [128, 128]"],
        ["Число шагов обучения", "200 000", "200 000"],
        ["Число сидов", "10", "10"],
    ]
    add_table(
        s, Inches(1.5), Inches(1.1), Inches(10.3), Inches(5.8),
        hp,
    )

    # ---------- B2. Таблица результатов по 10 сидам ----------
    s = add_blank(prs)
    title(s, "Backup: результаты по всем сидам (Sharpe OOS)")
    import csv
    # Collect all 20 rows (DQN + SAC) from internal tables
    dqn_2024 = [0.897, 0.499, 1.297, 0.344, 0.443, 0.611, 1.034, 0.958, 0.426, 0.845]
    dqn_2025 = [0.852, 0.820, 0.918, 0.395, 0.716, 0.604, 1.100, 0.581, 1.005, 0.939]
    sac_2024 = [0.652, 0.702, 0.676, 0.679, 0.611, 0.209, 0.394, 0.708, 0.680, 0.465]
    sac_2025 = [0.357, 0.189, 0.672, 0.535, 0.070, -0.559, 0.238, 0.038, 0.249, 0.282]
    seeds = [42, 123, 7, 2024, 99, 1337, 555, 888, 2001, 31415]
    rows = [["Сид", "SAC 2024", "SAC 2025", "DQN 2024", "DQN 2025"]]
    for i, sd in enumerate(seeds):
        rows.append([
            str(sd),
            f"{sac_2024[i]:+.3f}",
            f"{sac_2025[i]:+.3f}",
            f"{dqn_2024[i]:+.3f}",
            f"{dqn_2025[i]:+.3f}",
        ])
    import numpy as np
    rows.append([
        "среднее",
        f"{np.mean(sac_2024):+.3f}",
        f"{np.mean(sac_2025):+.3f}",
        f"{np.mean(dqn_2024):+.3f}",
        f"{np.mean(dqn_2025):+.3f}",
    ])
    rows.append([
        "std",
        f"{np.std(sac_2024, ddof=1):.3f}",
        f"{np.std(sac_2025, ddof=1):.3f}",
        f"{np.std(dqn_2024, ddof=1):.3f}",
        f"{np.std(dqn_2025, ddof=1):.3f}",
    ])
    add_table(
        s, Inches(2.5), Inches(1.1), Inches(8.3), Inches(5.8),
        rows, highlight_row=len(rows) - 2,
    )

    # ---------- B3. Список литературы ----------
    s = add_blank(prs)
    title(s, "Backup: источники")
    add_text(
        s, Inches(0.3), Inches(0.9), Inches(12.7), Inches(6.3),
        "Работы-аналоги (слайд 6):\n\n"
        "1. Lucarelli G., Borrotti M. A Deep RL Approach for Automated Cryptocurrency Trading. IFIP AIAI, 2019.\n"
        "2. Ye Y. и др. RL-based Portfolio Management with Augmented Asset Movement Prediction States. AAAI, 2020.\n"
        "3. Koratamaddi P. и др. Market sentiment-aware deep RL for stock portfolio allocation. JESTech, 24(4), 2021.\n"
        "4. Avramelou L. и др. Deep RL for financial trading using multi-modal features. ESWA, 238, 2024.\n"
        "5. Unnikrishnan A. Financial News-Driven LLM RL for Portfolio Management. arXiv:2411.11059, 2024.\n\n"
        "Обзоры и индустрия (слайды 2, 3):\n\n"
        "6. Tetlock P. Giving Content to Investor Sentiment. Journal of Finance, 62(3), 2007.\n"
        "7. Gentzkow M., Kelly B., Taddy M. Text as Data. Journal of Economic Literature, 57(3), 2019.\n"
        "8. Sebastião H., Godinho P. Forecasting and trading cryptocurrencies with ML. Financial Innovation, 7(3), 2021.\n"
        "9. Fang F. и др. Cryptocurrency trading: a comprehensive survey. Financial Innovation, 8(13), 2022.\n"
        "10. J. P. Morgan Markets. All about alternative data. 2024.\n\n"
        "Методологическая база:\n\n"
        "11. Moody J., Saffell M. Learning to trade via direct reinforcement. IEEE TNN, 12(4), 2001.\n"
        "12. Efron B., Tibshirani R. An Introduction to the Bootstrap. Chapman & Hall, 1994.\n"
        "13. Araci D. FinBERT: Financial Sentiment Analysis. arXiv:1908.10063, 2019.\n"
        "14. Raffin A. и др. Stable-Baselines3. JMLR, 22(268), 2021.",
        size=11,
    )

    prs.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
