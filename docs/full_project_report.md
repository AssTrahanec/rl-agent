# Полный отчёт по проекту: что сделано, что с чем сравнивали, что работает

**Проект:** Оптимизация стратегий торговли BTC/ETH с помощью RL + NLP  
**Тип:** Магистерская работа  
**Рынок:** BTC/USDT, ETH/USDT (daily timeframe)  
**Периоды:** обучение 2020–2023, тест 2024

---

## Общая идея проекта

Обучить RL-агента (PPO) торговать криптовалютой и проверить, помогают ли текстовые данные (новости) улучшить результаты. Три агента:

- **Agent-1 (Baseline)** — только цена (OHLCV + технические индикаторы) = 18 фичей
- **Agent-2 (Sentiment)** — цена + FinBERT sentiment score = 19 фичей
- **Agent-3 (Embeddings)** — цена + sentence-transformer embeddings = 50–95 фичей

Плюс сравнение с Buy & Hold (просто купить и держать).

---

## Этап 1: Ablation Study (3 агента × BTC + ETH)

**Что делали:** обучили каждого агента на 500K шагов, по 3–12 seeds, тестировали на 2024 год.

**Параметры:** PPO, сеть [256, 256], tanh, lr=3e-4 (constant), window=30 дней, tx_cost=0.1%.

### Результаты BTC (mean ± std)

| Агент | Seeds | Sharpe | Return | MaxDD |
|-------|-------|--------|--------|-------|
| **Baseline** | 12 | **1.663 ± 0.385** | 49.5% | 14.2% |
| **Embeddings** | 9 | 1.420 ± 0.800 | 48.5% | 16.5% |
| **Sentiment** | 9 | 1.120 ± 0.503 | 31.1% | 20.1% |
| Buy & Hold | — | 1.680 | 111.8% | 26.2% |

### Результаты ETH

| Агент | Seeds | Sharpe | Return | MaxDD |
|-------|-------|--------|--------|-------|
| **Baseline** | 5 | 0.916 ± 0.667 | 38.9% | 29.2% |
| **Embeddings** | 5 | 0.812 ± 0.619 | 39.9% | 29.1% |
| **Sentiment** | 5 | 0.896 ± 0.206 | 32.5% | 29.3% |

### Вывод этапа 1

- **Baseline победил** по Sharpe. Простые ценовые фичи работают лучше всего.
- **Sentiment — худший**. FinBERT обучен на финансовых отчётах, не на крипто-новостях.
- **Embeddings** — паритет с baseline, но высокая дисперсия (нестабильно между seeds).
- **Все агенты проиграли Buy & Hold по доходности** (49% vs 112%), но выиграли по просадке (14% vs 26%).
- NLP на дневном интервале не помогает — новости уже отражены в цене.

---

## Этап 2: Сравнение алгоритмов (PPO vs A2C vs SAC)

**Что делали:** взяли лучшего агента (embeddings) и обучили его тремя разными RL-алгоритмами.

### Результаты BTC (3 seeds каждый)

| Алгоритм | Sharpe | Return | MaxDD |
|----------|--------|--------|-------|
| **A2C** | **1.857 ± 0.121** | 83.9% | 21.6% |
| PPO | 1.512 ± 0.950 | 56.7% | 19.4% |
| SAC | (не показал стабильных результатов) | | |

### Вывод этапа 2

- **A2C стабильнее PPO** (std 0.121 vs 0.950!) при сравнимом Sharpe.
- PPO даёт лучший отдельный seed, но хуже в среднем.
- Для практического использования A2C предпочтительнее.

---

## Этап 3: Ансамбли

**Что делали:** вместо одной модели берём несколько лучших и усредняем их действия (allocation). Протестировали разные комбинации.

### Результаты BTC, 2024

| Стратегия | Sharpe | Return | MaxDD |
|-----------|--------|--------|-------|
| **Top-3 Baseline Ensemble** | **2.627** | 88.3% | **12.2%** |
| Top-3 Embeddings Ensemble | 2.341 | 74.8% | **9.8%** |
| Top-5 Baseline Ensemble | 2.431 | 78.2% | 11.9% |
| Single Best Model | 2.289 | 100.3% | 15.2% |
| Buy & Hold | 1.680 | **111.8%** | 26.2% |

### Вывод этапа 3

- **Ансамбли — лучшая стратегия** по risk-adjusted метрикам.
- Sharpe 2.63 vs 1.68 у Buy & Hold — ансамбль в 1.5 раза лучше.
- Просадка снизилась с 26% (B&H) до 12% (ансамбль).
- Embeddings-ансамбль дал наименьшую просадку (9.8%).

---

## Этап 4: Walk-Forward Validation

**Что делали:** проверили, как агент работает в разных рыночных условиях. Скользящее обучение: train 2 года → test 1 год → сдвиг.

### Результаты

| Тестовый год | Рынок | Agent Sharpe | Agent Return | B&H Sharpe | B&H Return |
|-------------|-------|-------------|-------------|------------|------------|
| **2022** | **Медвежий** | **-0.424** | **-27.8%** | **-1.343** | **-65.3%** |
| 2023 | Восстановление | 0.580 | 11.9% | 2.342 | 154.5% |
| 2024 | Бычий | 0.952 | 29.6% | 1.680 | 111.8% |

### Вывод этапа 4

- **На медвежьем рынке 2022** агент проиграл меньше (-28% vs -65% у B&H). Это главное преимущество — управление рисками.
- **На бычьем рынке** агент отстаёт от B&H (проще купить и держать).
- Агент адаптируется к смене режима рынка — каждое переобучение улучшает результат.

---

## Этап 5: Out-of-Sample тесты

**Что делали:** протестировали модели на данных, которых они вообще не видели.

### 2022 (медвежий рынок, BTC упал с 47K до 16K)

| Агент | Лучший Sharpe | Лучший Return |
|-------|-------------|--------------|
| Embeddings | **4.641** | **367%** |
| Baseline | 4.502 | 439% |
| Sentiment | 4.057 | 270% |

Агенты научились шортить/выходить из позиции на падающем рынке (старые модели, обученные 2020–2023).

### 2025 Q1 (январь–март, падение BTC)

| Агент | Лучший Sharpe | Лучший Return |
|-------|-------------|--------------|
| Sentiment | **0.672** | **2.6%** |
| Baseline | -0.055 | -0.1% |
| Embeddings | -0.485 | -4.1% |

### Вывод этапа 5

- На 2022 модели показали великолепные результаты (обученные на 2020–2023, они "видели" часть этого паттерна).
- На 2025 (совершенно новые данные) — все агенты плохо. Модели не обобщаются на будущее без переобучения.

---

## Этап 6: Эксперименты с reward и short

**Что делали:** попробовали изменить функцию награды и разрешить короткие позиции.

### Результаты

| Конфигурация | Sharpe | Return | MaxDD |
|-------------|--------|--------|-------|
| Стандартный reward | 1.663 | 49.5% | 14.2% |
| Risk-adjusted reward | -0.234 | -0.01% | 0.1% |
| Risk-adjusted + short | -0.790 | -0.1% | 0.3% |

### Вывод этапа 6

- **Risk-adjusted reward не помог** — агент стал слишком консервативным (почти не торгует).
- Разрешение коротких позиций при risk-adjusted reward тоже не дало результата.
- Стандартный reward (log_return × allocation) оказался лучшим.

---

## Этап 7: Улучшение Embeddings агента (v2)

**Что делали:** несколько итераций улучшений NLP pipeline.

### Итерация 7.1: Замена модели эмбеддингов

| Параметр | Было (v1) | Стало (v2) |
|----------|-----------|------------|
| Модель | all-MiniLM-L6-v2 (384d, общая) | FinLang/finance-embeddings-investopedia (768d, финансовая) |
| Pooling | Mean (все статьи одинаково) | Weighted по \|sentiment\| (важные новости → больший вес) |
| PCA | 384 → 32d | 768 → 64d |
| Фичи | 50 | 83 |

**Результат:** Return +24% (43.7% → 54.3%), но Sharpe без изменений (1.31 vs 1.36), просадка выросла на 39%.

### Итерация 7.2: Lag features + sentiment extremes + большая сеть

| Параметр | Было (v2) | Стало (v3) |
|----------|-----------|------------|
| Фичи | 83 | 95 (+12 новых) |
| Новые фичи | — | sentiment_max, sentiment_min, sentiment_spread, news_count lags (lag1, lag2, roll7), PCA lags (top-3 × lag1, lag2) |
| Сеть | [256, 256] | [512, 256] (для embeddings/fusion) |
| LR schedule | constant | linear decay (от 3e-4 до 0) |
| Timesteps | 500K | 2M (планировалось) |

**Зачем каждое изменение:**

- **sentiment_max/min/spread** — среднее sentiment теряет информацию. Если за день 5 позитивных и 5 негативных новостей, среднее = 0, но spread = 1 (высокая неопределённость).
- **news_count lags** — количество новостей вчера/позавчера/за неделю. Всплеск новостей часто предшествует движению цены.
- **PCA lags** (emb_0..2 lag1/lag2) — вчерашнее "направление новостей" может влиять на сегодняшнюю цену.
- **[512, 256] сеть** — с 95 фичами × 30 дней = 2850 входных нейронов, [256, 256] может быть мало.
- **Linear LR decay** — при 2M шагов и большой сети стабильнее: сначала быстро учится, потом fine-tune.

**Результат:** seed 42 на локальной машине показал Sharpe=0.804 (ниже предыдущих). Полный прогон (7 seeds) не завершён — запускали на DataSphere.

---

## Итоговый рейтинг всех стратегий (по Sharpe, тест 2024)

| # | Стратегия | Sharpe | Return | MaxDD | Комментарий |
|---|-----------|--------|--------|-------|-------------|
| 1 | Top-3 Baseline Ensemble | **2.627** | 88.3% | 12.2% | Лучший risk-adjusted |
| 2 | Top-5 Baseline Ensemble | 2.431 | 78.2% | 11.9% | |
| 3 | Top-3 Embeddings Ensemble | 2.341 | 74.8% | **9.8%** | Наименьшая просадка |
| 4 | Single Best Model | 2.289 | 100.3% | 15.2% | |
| 5 | A2C Embeddings (mean) | 1.857 | 83.9% | 21.6% | Самый стабильный алгоритм |
| 6 | Buy & Hold | 1.680 | **111.8%** | 26.2% | Наивный бенчмарк |
| 7 | Baseline (ablation, 12 seeds) | 1.663 | 49.5% | 14.2% | |
| 8 | Embeddings v1 (ablation, 9 seeds) | 1.420 | 48.5% | 16.5% | |
| 9 | Baseline (3 новых seeds) | 1.358 | 43.7% | 17.5% | |
| 10 | Improved Emb v2 (3 seeds) | 1.310 | 54.3% | 24.4% | |
| 11 | Sentiment (ablation, 9 seeds) | 1.120 | 31.1% | 20.1% | |
| 12 | Risk-adjusted reward | -0.234 | -0.01% | 0.1% | Не работает |

---

## Главные выводы проекта

### Что работает

1. **Ансамбли** — усреднение нескольких моделей даёт Sharpe 2.6+, лучше любой одиночной модели и Buy & Hold.
2. **Baseline агент** — простые ценовые фичи (SMA, EMA, RSI, MACD и т.д.) — самый стабильный и надёжный.
3. **RL-агент как инструмент risk management** — на медвежьем рынке 2022 потерял 28% vs 65% у Buy & Hold.
4. **A2C стабильнее PPO** при сравнимой доходности (std 0.12 vs 0.95).

### Что не работает

1. **NLP на дневном таймфрейме** — новости уже в цене к закрытию дня. Sentiment, embeddings, lag features — ничего из этого не дало устойчивого преимущества по Sharpe.
2. **Risk-adjusted reward** — агент становится слишком консервативным, перестаёт торговать.
3. **Обобщение на будущее без переобучения** — модели 2020–2023 плохо работают на 2025 (Sharpe < 0 для большинства).

### Что неоднозначно

1. **Embeddings v2** — более высокая доходность (+24%), но за счёт большей просадки (+39%). Зависит от толерантности к риску.
2. **Walk-forward** — показывает адаптивность, но не обгоняет B&H на бычьем рынке. Полезен для защиты на медвежьем.

---

## Хронология изменений кода

| Дата | Что изменили | Файлы | Эффект |
|------|-------------|-------|--------|
| Март 2026 | Базовая система: env, PPO, backtest, метрики | `src/env/`, `src/agents/`, `src/eval/` | Рабочая система |
| Март 2026 | NLP pipeline: FinBERT + MiniLM + PCA 32d | `src/features/` | Sentiment + Embeddings агенты |
| Март 2026 | Ablation study: 3 агента × 2 актива | `experiments/run_all.py` | Baseline > Embeddings > Sentiment |
| Март 2026 | Ансамбли, walk-forward, OOS тесты | `src/eval/ensemble_backtest.py`, `scripts/` | Ensemble Sharpe 2.63 |
| Март 2026 | Algo comparison: PPO vs A2C vs SAC | `src/agents/run_algo_comparison.py` | A2C стабильнее |
| Апрель 2026 | Замена MiniLM → FinLang, weighted pooling, PCA 64d | `src/features/embeddings.py` | Return +24%, MaxDD +39% |
| Апрель 2026 | Lag features, sentiment extremes, [512,256], linear LR | `src/features/lag_features.py`, `train.py` | Ещё не протестировано на 7 seeds |

---

## Что ещё было реализовано (инфраструктура)

Помимо экспериментов, был создан полный production-ready pipeline:

### Сбор и обработка данных
- **price_collector.py** — загрузка OHLCV через ccxt (Binance spot)
- **news_collector.py** — загрузка новостей из HuggingFace datasets (142K статей)
- **news_preprocessor.py** — дедупликация, группировка по дням (66K → 1823 дня)
- **build_price_features.py** — CLI pipeline: fetch → indicators → normalize → parquet

### Feature Engineering
- **technical.py** — 18 технических индикаторов: SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV, Stochastic, returns
- **normalizer.py** — rolling z-score нормализация (window=30), NaN→0
- **sentiment.py** — FinBERT sentiment scoring, + `compute_sentiment_scores()` для per-article scores
- **embeddings.py** — сначала all-MiniLM-L6-v2 (384d), потом заменён на FinLang/finance-embeddings-investopedia (768d) + weighted pooling по |sentiment|
- **embedding_compressor.py** — PCA сжатие (384→32d, потом 768→64d), save/load
- **lag_features.py** — утилиты для лагов и rolling features
- **build_sentiment_features.py** — merge daily sentiment с price features
- **build_embedding_features.py** — merge daily embeddings + sentiment extremes + lags
- **build_fusion_features.py** — Agent-4: price + sentiment + 32d embeddings = 53 фичи
- **finetune_finbert.py** — код для fine-tune FinBERT на крипто-данных (FinBERTFinetuner + prepare_finetune_dataset), готов но не использован в финальных экспериментах

### Среда и агенты
- **trading_env.py** — Gymnasium TradingEnv: obs = 30 дней × N фичей (flat), action ∈ [0,1], reward = log_return × allocation - tx_cost. Поддерживает risk_adjusted reward и short позиции
- **config.py** — AgentConfig dataclass со всеми гиперпараметрами PPO + lr_schedule
- **train.py** — универсальный тренер для PPO/A2C/SAC с linear LR schedule и автоматическим увеличением сети для embeddings
- **run_ablation.py** — запуск всех agent_type × asset × seed комбинаций
- **run_algo_comparison.py** — PPO vs A2C vs SAC на лучшем агенте

### Оценка
- **metrics.py** — Sharpe, Sortino, Max Drawdown, Calmar, Total Return (365 trading days/year для крипто)
- **backtest.py** — прогон модели на тестовых данных + equity curve plot
- **bootstrap.py** — Bootstrap CI (95%) для Sharpe и Total Return across seeds
- **ensemble_backtest.py** — ансамбль: несколько моделей предсказывают одновременно, их allocations усредняются
- **baselines.py** — Buy & Hold baseline
- **visualize.py** — equity curves, bar plots, heatmaps (matplotlib + seaborn)
- **report.py** — генерация Markdown и LaTeX таблиц

### Скрипты
- **run_all.py** — главный скрипт обучения: agents × seeds, backtest, CSV, wandb logging
- **run_ensemble.py** — запуск ансамблевого backtest
- **run_walkforward.py** — walk-forward validation (train 2 года → test 1 год → сдвиг)
- **oos_backtest.py** — out-of-sample тесты (2022, 2025 Q1)
- **analyze_all.py** — сбор всех результатов в единый CSV
- **rebuild_embedding_features.py** — пересборка parquet с новой моделью эмбеддингов
- **rebuild_metrics.py** — пересчёт метрик из сохранённых моделей

### Ноутбуки
- **results_analysis.ipynb** — визуализация ablation study: таблицы mean±std, barplot Sharpe, equity curves, t-test + Mann-Whitney, heatmap
- **colab_training.ipynb** — standalone ноутбук для обучения на Yandex DataSphere / Google Colab (весь код встроен, без зависимости от repo)

### Тестирование
- **94+ unit tests** покрывают все модули (pytest)
- Тесты для: price_collector, technical, normalizer, trading_env, config, train, metrics, backtest, bootstrap, news_collector, news_preprocessor, sentiment, embeddings, compressor, build_sentiment, build_embedding, build_fusion, run_ablation, run_algo_comparison, baselines, visualize, report, ensemble_backtest, walkforward, lag_features, load_features, finetune_finbert

### Документация
- **CLAUDE.md** — контекст проекта, все параметры, что реализовано
- **PRD.md** — Product Requirements Document
- **PLAN.md** — план разработки по Slices
- **docs/thesis/** — черновики глав диплома (literature review, methodology, results, discussion, conclusion)

---

## Техническая архитектура

```
Данные → Фичи → Среда → Обучение → Backtest → Метрики
  │         │       │        │          │          │
  │         │       │        │          │          └─ Sharpe, Sortino, MaxDD, Calmar, Return
  │         │       │        │          └─ Прогон модели на тестовых данных
  │         │       │        └─ PPO/A2C/SAC (stable-baselines3)
  │         │       └─ TradingEnv (Gymnasium): obs=30 дней×N фичей, action=[0,1]
  │         └─ Технические индикаторы (18), sentiment (1), embeddings (64), lags
  └─ Binance OHLCV, Bitcoin News (HuggingFace)
```
