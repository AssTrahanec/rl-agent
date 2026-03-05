# RL + NLP Trading Agent

Оптимизация стратегий торговли BTC/ETH с помощью Reinforcement Learning и NLP из новостных источников.

**Магистерская работа** — 3-way ablation study: сравнение PPO-агентов с различными NLP-фичами.

## Агенты

| Агент | Описание |
|-------|----------|
| Agent-1 (Baseline) | OHLCV + технические индикаторы |
| Agent-2 (+Sentiment) | + FinBERT sentiment score |
| Agent-3 (+Embeddings) | + sentence-transformer embeddings (32d) |

## Установка

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Структура

```
src/
  data/       — сбор данных (цены, новости)
  features/   — NLP pipeline, технические индикаторы
  env/        — Gymnasium trading environment
  agents/     — конфигурации и обучение агентов
  eval/       — метрики, backtesting, визуализация
```

## Запуск тестов

```bash
pytest tests/ -v
```

## Документация

- [PRD](docs/PRD.md) — требования проекта
- [Plan](docs/PLAN.md) — план реализации
