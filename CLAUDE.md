# CLAUDE.md — Контекст проекта

Читай этот файл в начале каждой сессии.

---

## Тема

**Оптимизация стратегий торговли с помощью RL и NLP из новостных источников**
Магистерская работа. Рынок: BTC/ETH. Горизонт: среднесрочный (daily, дни/недели).

---

## Цель и научный вклад

3-way ablation study: сравнение трёх PPO-агентов:
1. **Agent-1 (Baseline)** — только ценовые данные (OHLCV + технические индикаторы)
2. **Agent-2 (+Sentiment)** — + FinBERT sentiment score (mean по новостям за день)
3. **Agent-3 (+Embeddings)** — + sentence-transformer embeddings (mean pooling → 32d)

Дополнительно:
- Сравнение PPO vs A2C vs SAC (на лучшем агенте из ablation)
- Buy & Hold как наивный baseline
- (Опционально) Fine-tune FinBERT на крипто-новостях, если базовый sentiment слабый
- (Опционально) Agent-4 (sentiment + embeddings) как feature fusion

Novelty: первый явный 3-way ablation PPO (SB3) + FinBERT + sentence-transformers на BTC/ETH с открытыми данными.

---

## Технический стек

```
Python 3.10+
├── RL:    stable-baselines3, gymnasium
├── NLP:   transformers (ProsusAI/finbert), sentence-transformers (all-MiniLM-L6-v2)
├── Data:  ccxt, yfinance, newsapi-python, tweepy, pandas
├── Misc:  numpy, scikit-learn, matplotlib, seaborn, mlflow
└── Env:   RTX 3060+ (локально) / Google Colab T4
```

---

## Структура папок

```
rl-agent/
├── data/
│   ├── raw/          # сырые данные (цены, новости)
│   └── processed/    # обработанные features (numpy/parquet)
├── src/
│   ├── data/         # сборщики данных (Binance, NewsAPI, Twitter)
│   ├── features/     # NLP pipeline, feature engineering
│   ├── env/          # Gymnasium trading environment
│   ├── agents/       # конфигурации PPO агентов
│   └── eval/         # метрики, backtesting, plots
├── experiments/      # результаты обучения (mlflow runs)
├── notebooks/        # анализ данных, визуализация
├── docs/             # PRD.md, PLAN.md
├── tests/            # unit тесты
├── CLAUDE.md         # этот файл
├── WORKFLOW.md       # статус фаз
└── requirements.txt
```

---

## Ключевые параметры

| Параметр | Значение |
|----------|----------|
| Активы | BTC/USDT, ETH/USDT |
| Timeframe | Daily (1d) |
| Train период | 2020-01-01 – 2023-12-31 |
| Test период | 2024-01-01 – 2024-12-31 |
| Observation window | 30 дней |
| Action space | Continuous [0, 1] (allocation) |
| Reward | log return × allocation |
| Transaction cost | 0.1% при изменении позиции |
| Policy network | MLP [256, 256], tanh |

---

## NLP Pipeline

```
Текст новости
  → FinBERT → sentiment score ∈ [-1, +1]     # для Agent-2
  → all-MiniLM-L6-v2 → 384d embedding          # для Agent-3
      → mean pooling по N новостям за день
      → Linear(384, 32) → 32d feature
```

Fallback (нет новостей за день): sentiment = 0.0, embedding = zeros(32).

---

## Соглашения по коду

- Все скрипты запускаются из корня репозитория
- Конфиги агентов в `src/agents/config.py` (не хардкодить гиперпараметры)
- Логирование через `logging` (не `print`)
- Сохранение моделей в `experiments/{agent_name}/{timestamp}/`
- Тесты в `tests/`, запуск через `pytest`
- Форматирование: `black`, линтер: `flake8` (длина строки 100)

---

## Уже реализовано

*(пусто — заполняется по мере реализации)*

---

## Текущий фокус

**Фаза 4: Реализация**
- Текущая задача: Slice 1, Task 1.1 (OHLCV price collector)

Следующий шаг: открой `prompts/04_implementation.md`.
