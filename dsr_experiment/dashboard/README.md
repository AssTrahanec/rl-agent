# Dashboard — Streamlit UI для защиты ВКР

Демо-панель для защиты магистерской работы
«Оптимизация стратегий торговли с помощью RL и обработки новостных источников».
Спроектирована как набор сцен для презентации: лендинг → live → валидация → интерактив → методика.

## Запуск

Из корня `dsr_experiment/`:

```bash
../venv/Scripts/streamlit run dashboard/app.py
```

Откроется http://localhost:8501.

## Страницы

| Файл                        | Что показывает | Когда показывать |
|-----------------------------|----------------|------------------|
| `app.py` — **Главная**      | Обзор работы, ключевые метрики, навигация | Старт защиты |
| `pages/1_Strategy.py` — **Стратегия онлайн** | Свежие новости BTC, ансамбль 10 моделей, текущее решение, цена + аллокация, история | Live-демонстрация |
| `pages/2_Validation.py` — **Валидация на OOS** | OOS 2024 и 2025, bootstrap CI, equity curves, распределение Sharpe по сидам | Научная часть |
| `pages/3_News_Analyzer.py` — **Анализатор новости** | Голосование ансамбля до/после произвольной новости, сентименты, дельты | Интерактив с комиссией |

## Требования

- Обученные модели в `experiments/run_2026-04-21_10seeds/` (primary snapshot)
- `data/train/compressor.pkl` (PCA для live embeddings)
- `data/oos/*.parquet` (для recompute B&H)
- `data/raw/ohlcv.parquet` (fallback если Binance недоступна)

## Стиль

Общая тема — `dashboard/utils/ui.py`:
- Палитра `PALETTE` и `CHART_PALETTE`
- `inject_global_style()` подкладывает CSS на каждой странице
- Компоненты: `hero`, `section_header`, `insight`, `decision_card`
- `plotly_layout_defaults()` для одинаковых чартов

## Важные оговорки

- **NewsAPI ключ** — если задан в `.env` как `NEWSAPI_KEY`, лента подтягивает 5-дневный архив.
  Без ключа — fallback на CoinDesk/Cointelegraph RSS (≤48 ч).
- **Read-only**: реальная торговля не выполняется. Все P&L — симуляция.
- **FinBERT — англоязычный**. На странице News Analyzer надо вставлять английский текст.
