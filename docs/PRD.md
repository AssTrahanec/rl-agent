# Product Requirements Document (PRD)

## Оптимизация стратегий торговли с помощью RL и NLP из новостных источников

---

## 1. Цель

Исследовать, улучшает ли добавление NLP-фичей из крипто-новостей (sentiment, embeddings) качество PPO-агента для среднесрочной торговли BTC и ETH.

Базовый агент использует ценовые данные и технические индикаторы (RSI, MACD, Bollinger Bands и др.). Два NLP-расширенных агента добавляют к этой базе:
- **sentiment score** (настроение новостей) — через FinBERT
- **sentence embeddings** (смысловое представление новостей) — через sentence-transformers

Формат: **3-way ablation study** — прямое сравнение вклада каждой NLP-модальности.

---

## 2. Научный вклад (Novelty)

Первый открытый 3-way ablation study с использованием:
- **PPO** (Stable-Baselines3) как базовый RL-алгоритм
- **FinBERT** для sentiment scoring
- **all-MiniLM-L6-v2** для sentence embeddings

на рынке BTC/ETH с daily timeframe и bootstrap confidence intervals для статистической значимости.

**Отличие от существующих работ:**
- Большинство работ используют только sentiment ИЛИ только embeddings, без прямого сравнения
- Редко встречается ablation study с >2 вариантами на крипто-рынке
- Открытые данные и воспроизводимый pipeline (открытые датасеты, SB3, HuggingFace модели)

**Дополнительные эксперименты:**
- Сравнение PPO vs A2C vs SAC (на лучшем агенте из ablation)
- Buy & Hold как наивный baseline
- (Опционально) Fine-tune FinBERT на крипто-новостях
- (Опционально) Agent-4: sentiment + embeddings (feature fusion)

---

## 3. Архитектура

### 3.1 Общая схема

```
┌────────────────────────────────────────────────────────┐
│                    Data Pipeline                        │
│  OHLCV (ccxt) ──→ Tech Indicators ──→ ┐               │
│  News (HF/Kaggle) ──→ FinBERT ──→ sentiment ──→ ┤     │
│  News (HF/Kaggle) ──→ MiniLM ──→ 384d ──→ Linear(32) ─┤
│                                                  ↓     │
│                              Observation Space         │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│               Gymnasium Environment                     │
│  State: window 30 дней × features                      │
│  Action: continuous [0, 1] (доля вложения)             │
│  Reward: log(return) × allocation − tx_cost            │
│  Transaction cost: 0.1% при изменении позиции          │
└────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│                  PPO Agent (SB3)                        │
│  Policy: MlpPolicy [256, 256], tanh                    │
│  Training: ~500K-1M steps                              │
│  Seeds: 3-5 runs per configuration                     │
└────────────────────────────────────────────────────────┘
```

### 3.2 Агенты

| Агент | Observation | Описание |
|-------|-------------|----------|
| Agent-1 (Baseline) | OHLCV + tech indicators (~20 фичей × 30 дней) | Только ценовые данные |
| Agent-2 (+Sentiment) | Agent-1 + sentiment score (+1 фича/день) | + настроение новостей |
| Agent-3 (+Embeddings) | Agent-1 + 32d embedding (+32 фичи/день) | + смысловое представление |

Каждый агент тренируется **отдельно на BTC и ETH** → 6 конфигураций × 3-5 seeds.

### 3.3 NLP Pipeline

```
Текст новости
  ├──→ ProsusAI/finbert → softmax → sentiment ∈ [-1, +1]
  └──→ all-MiniLM-L6-v2 → 384d embedding
          → mean pooling по N новостям за день
          → Linear(384, 32) → 32d compressed feature

Fallback (нет новостей за день):
  sentiment = 0.0
  embedding = zeros(32)
```

---

## 4. Данные

### 4.1 Ценовые данные

| Параметр | Значение |
|----------|----------|
| Активы | BTC/USDT, ETH/USDT |
| Источник | CCXT (Binance) / yfinance (резерв) |
| Timeframe | Daily (1d) |
| Период | 2020-01-01 — 2024-12-31 |
| Формат | OHLCV (Open, High, Low, Close, Volume) |

### 4.2 Технические индикаторы

Вычисляются из OHLCV:
- **Трендовые:** SMA(7), SMA(25), SMA(99), EMA(12), EMA(26)
- **Моментум:** RSI(14), MACD(12,26,9), Stochastic %K/%D
- **Волатильность:** Bollinger Bands(20,2), ATR(14)
- **Объём:** OBV, VWAP
- **Нормализация:** все фичи нормализуются rolling z-score (window=30)

### 4.3 Новостные данные

| Датасет | Источник | Описание |
|---------|----------|----------|
| `edaschau/bitcoin_news` | HuggingFace | Основной. Bitcoin новости с Yahoo Finance, timestamps |
| `Sentiment Analysis of Bitcoin News 2021-2024` | Kaggle | Дополнительный. Новости с готовым sentiment |

**Препроцессинг:**
1. Фильтрация по дате (2020–2024)
2. Дедупликация по заголовкам
3. Группировка по торговым дням (UTC)
4. Агрегация: mean sentiment / mean embedding за день

### 4.4 Split

| Набор | Период | Назначение |
|-------|--------|------------|
| Train | 2020-01-01 — 2023-12-31 | Обучение агентов |
| Test | 2024-01-01 — 2024-12-31 | Финальная оценка |

---

## 5. Модели

### 5.1 RL-алгоритмы

| Алгоритм | Библиотека | Роль |
|----------|------------|------|
| **PPO** | stable-baselines3 | Основной (ablation study) |
| **A2C** | stable-baselines3 | Сравнение алгоритмов |
| **SAC** | stable-baselines3 | Сравнение алгоритмов |

**Гиперпараметры PPO (начальные):**

| Параметр | Значение |
|----------|----------|
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| policy_kwargs | dict(net_arch=[256, 256], activation_fn=tanh) |

### 5.2 NLP-модели

| Модель | Задача | Выход |
|--------|--------|-------|
| `ProsusAI/finbert` | Sentiment classification | score ∈ [-1, +1] |
| `all-MiniLM-L6-v2` | Sentence embedding | 384d → Linear → 32d |

### 5.3 Baselines

- **Buy & Hold:** покупаем в день 1, держим до конца теста
- **Agent-1 (Baseline):** PPO без NLP-фичей

---

## 6. Метрики

### 6.1 Финансовые метрики

| Метрика | Описание |
|---------|----------|
| **Total Return** | Общая доходность за тестовый период (%) |
| **Sharpe Ratio** | Доходность с учётом риска (risk-adjusted return) |
| **Sortino Ratio** | Как Sharpe, но учитывает только downside risk |
| **Max Drawdown** | Максимальная просадка от пика до дна (%) |
| **Calmar Ratio** | Annual return / Max Drawdown |

### 6.2 RL-метрики (мониторинг обучения)

| Метрика | Описание |
|---------|----------|
| **Cumulative Reward** | Суммарная награда за эпизод |
| **Episode Length** | Длина эпизода (должна быть стабильной) |
| **Explained Variance** | Качество value function (чем ближе к 1, тем лучше) |
| **Policy Loss / Value Loss** | Стабильность обучения |

### 6.3 Статистическая значимость

- **Bootstrap Confidence Intervals (95%)** для Sharpe Ratio и Total Return
- **3-5 random seeds** на каждую конфигурацию
- Результаты: mean ± std + CI

### 6.4 Визуализация

- Equity curves (кривые капитала) для всех агентов
- Comparison bar charts (Sharpe, Return, Drawdown)
- Training curves (reward, loss vs steps)
- Heatmap: агенты × метрики

---

## 7. Out-of-scope

- Live trading (реальная торговля на бирже)
- Production deployment (деплой в продакшн)

---

## 8. Технический стек

```
Python 3.10+
├── RL
│   ├── stable-baselines3    # PPO, A2C, SAC
│   └── gymnasium            # trading environment
├── NLP
│   ├── transformers         # ProsusAI/finbert
│   └── sentence-transformers # all-MiniLM-L6-v2
├── Data
│   ├── ccxt                 # Binance OHLCV
│   ├── yfinance             # резервный источник цен
│   ├── datasets (HF)        # загрузка новостных датасетов
│   └── pandas               # обработка данных
├── Feature Engineering
│   ├── ta / ta-lib          # технические индикаторы
│   ├── numpy                # массивы, вычисления
│   └── scikit-learn         # нормализация, метрики
├── Experiment Tracking
│   └── mlflow               # логирование экспериментов
├── Visualization
│   ├── matplotlib           # графики
│   └── seaborn              # статистические визуализации
└── Code Quality
    ├── black                # форматирование
    ├── flake8               # линтер
    └── pytest               # тесты
```

**Hardware:** RTX 3060+ (локально) / Google Colab T4 (резерв)

---

## 9. Порядок реализации (высокоуровневый)

1. **Data Pipeline:** сбор цен + новостей, feature engineering
2. **Environment:** Gymnasium trading environment
3. **Agent-1 (Baseline):** PPO на ценовых данных
4. **NLP Pipeline:** FinBERT sentiment + sentence embeddings
5. **Agent-2, Agent-3:** PPO с NLP-фичами
6. **Evaluation:** метрики, bootstrap CI, визуализация
7. **Algorithm Comparison:** PPO vs A2C vs SAC
8. **(Опционально)** Fine-tune FinBERT, Agent-4 (fusion)
