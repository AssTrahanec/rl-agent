# Dashboard — Streamlit UI

Demo-панель для защиты магистерской (работа "Оптимизация стратегий торговли с помощью
RL и обработки новостных источников").

## Как запустить

Из корня `dsr_experiment/`:

```bash
../venv/Scripts/streamlit run dashboard/app.py
```

Открывается http://localhost:8501.

## Страницы

- **📊 Backtest** — просмотр сохранённых backtest'ов (equity curves, bootstrap CI)
- **🔮 Live Prediction** — live BTC price + решение RL-модели
- **📈 Paper Trading** — re-симуляция агента на OOS 2024/2025
- **📰 News Analyzer** — FinBERT + PCA анализ новости

## Требования

- Обученные модели в `experiments/run_2026-04-21_10seeds/` (primary snapshot)
- `data/train/compressor.pkl` (PCA для live embeddings)
- `data/oos/*.parquet` (для Paper Trading)
- `data/raw/ohlcv.parquet` (fallback если Binance недоступна)

## Важные оговорки

- **SAC без VecNormalize**: в snapshot `run_2026-04-21_10seeds` не сохранён
  `vecnormalize.pkl`. Live Prediction показывает info banner об этом.
- **NLP zero-fill в Live Prediction**: news/sentiment фичи = 0 для live инференса.
  Backtest и Paper Trading используют pre-computed features из parquet —
  там новости реальные.
- **Read-only**: панель ничего не пишет в проект, реальная торговля не выполняется.
