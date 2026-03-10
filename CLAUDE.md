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

- **Slice 1, Task 1.1** — `src/data/price_collector.py`: fetch_ohlcv() через ccxt (Binance spot), тест `tests/test_price_collector.py` (4 unit tests + 1 integration)
- **Slice 1, Task 1.2** — `src/features/technical.py`: add_technical_indicators() — SMA, EMA, MACD, RSI, Bollinger, ATR, OBV, Stochastic, returns. Тест `tests/test_technical.py` (5 tests)
- **Slice 1, Task 1.3** — `src/features/normalizer.py`: rolling_zscore_normalize() window=30, NaN→0. Тест `tests/test_normalizer.py` (5 tests)
- **Slice 1, Task 1.4** — `src/data/build_price_features.py`: CLI pipeline fetch→indicators→normalize→parquet
- **Slice 2, Task 2.1-2.2** — `src/env/trading_env.py`: Gymnasium TradingEnv (flat obs для SB3, check_env pass). Тест `tests/test_trading_env.py` (7 tests)
- **Slice 3, Task 3.1** — `src/agents/config.py`: AgentConfig dataclass с PPO гиперпараметрами. Тест `tests/test_config.py` (5 tests)
- **Slice 3, Task 3.2** — `src/agents/train.py`: train_agent() с PPO/A2C/SAC, smoke test на dummy data. Тест `tests/test_train.py` (2 tests)
- **Slice 3, Task 3.3** — `src/eval/metrics.py`: compute_metrics() — Sharpe, Sortino, Max Drawdown, Calmar, Total Return. Тест `tests/test_metrics.py` (7 tests)
- **Slice 3, Task 3.4** — `src/eval/backtest.py`: run_backtest() — прогон модели, equity curve plot. Тест `tests/test_backtest.py` (3 tests)
- **Slice 3, Task 3.5** — `src/eval/bootstrap.py`: bootstrap_ci() — Bootstrap CI (95%) для Sharpe и Total Return. Тест `tests/test_bootstrap.py` (4 tests)
- **Инфраструктура**: `venv/` (Python 3.9), `conftest.py`, `pytest.ini` с маркером `integration`

---

## Текущий фокус

**Фаза 4: Реализация**
- Slices 1-3 завершены (42 теста pass)
- Следующая задача: Slice 4, Task 4.1 (News Data Pipeline — `src/data/news_collector.py`)
