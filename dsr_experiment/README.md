# DSR Experiment

Автономная папка финального эксперимента магистерской работы:
**Оптимизация стратегий торговли с помощью RL и обработки новостных текстов.**

Обучает **DQN** (дискретные действия; основной агент) и опционально **SAC**
(непрерывная доля) с **DSR-reward** и sentiment-бонусом на BTC/USDT 4h OHLCV +
финансовые эмбеддинги новостей, затем гоняет OOS-бэктесты на произвольных периодах.

Все гиперпараметры и пути — в [config.yaml](config.yaml).
Быстрый smoke-прогон — [config.smoke.yaml](config.smoke.yaml).

Подробный построчный разбор пути новости и дашборда (для защиты) —
[vkr_defense/news_path_walkthrough.md](vkr_defense/news_path_walkthrough.md).

---

## Что именно реализовано

### 1. Данные и фичи (пайплайн [build_data.py](build_data.py))

| Этап | Файл | Суть |
|------|------|------|
| OHLCV через `ccxt` Binance (4h) | [lib/features/price.py](lib/features/price.py) → `fetch_ohlcv` | Скачивание свечей + кэш в `data/raw/ohlcv.parquet` |
| **8 технических индикаторов** | [lib/features/price.py](lib/features/price.py) → `add_technical_indicators_minimal` | EMA-26, MACD, RSI-14, BB-width, ATR-14, OBV, Stoch %K, return_1d |
| Rolling z-score нормализация (строго трейлинг) | [lib/features/price.py](lib/features/price.py) → `rolling_zscore_normalize` | window=30, NaN→0, сохраняет `raw_close` для бэктеста |
| Загрузка новостей из HF | [lib/features/news.py](lib/features/news.py) → `load_news_from_hf` | `edaschau/bitcoin_news`, фильтр по датам, кэш в `data/raw/news.parquet` |
| Группировка в 4h окна | [lib/features/news.py](lib/features/news.py) → `preprocess_news_4h` | `dt.floor("4h")` → список текстов на окно |
| FinBERT sentiment | [lib/features/sentiment.py](lib/features/sentiment.py) → `compute_sentiment_scores` | `ProsusAI/finbert`, score ∈ [-1, +1] |
| Sentence embeddings | [lib/features/embeddings.py](lib/features/embeddings.py) → `_get_model` | `FinLang/finance-embeddings-investopedia` (768d) |
| **PCA-компрессор 768→32** | [lib/features/embeddings.py](lib/features/embeddings.py) → `EmbeddingCompressor` | `fit` **только на train**, сохраняется `compressor.pkl`, применяется к OOS |
| Склейка фичей | [build_data.py](build_data.py) → `_attach_nlp_features_minimal` | 1 `sentiment_mean` + 32 `emb_*` (взвешенное по тональности среднее) |

**Итоговый набор: 41 фича** = 8 индикаторов + `sentiment_mean` + 32 эмбеддинга.
На выходе — один parquet на период: `data/train/features.parquet` и
`data/oos/{period_key}_features.parquet`.

### 2. Trading environment

[lib/env.py](lib/env.py) → `TradingEnv` (Gymnasium):

- **Action space**: `Discrete(3)` {0=Hold, 1=Buy all-in, 2=Sell all-out} для **DQN**;
  `Box([0,1])` (непрерывная доля) для **SAC**.
- **Observation**: сплющенное окно последних `window` 4h-баров (`window × n_features`)
  + текущая доля аллокации → `(window·n_features + 1,)` = 30×41+1 = **1231**.
- **Transaction cost**: `tx_cost · |Δallocation|`, вычитается из шаговой отдачи.
- **Reward = DSR + sentiment-бонус**:
  `dsr` — **Differential Sharpe Ratio** (Moody & Saffell): рекурсивно обновляются
  экспоненциальные моменты `A`, `B`, награда = `(B·ΔA − 0.5·A·ΔB) / (B − A²)^(3/2)`
  (`_compute_dsr`, η=0.01).
- **Sentiment-бонус**: `+ sentiment_lambda · sentiment_t · log_return_t` (λ=0.3).
  Сигнал — колонка `sentiment_mean`, подмешивается на каждом шаге.

### 3. Обучение

[lib/train.py](lib/train.py) → `train_agent(cfg, algo, seed, features, prices, sentiment)`:

- Stable-Baselines3: `DQN` и `SAC` (CUDA). PPO убран.
- Linear / constant learning-rate schedule, `net_arch=[128,128]` по умолчанию.
- `ProgressCallback` печатает прогресс каждые 50k шагов.
- Модель кладётся в `models/{algo}/{algo}_seed{seed}_{ts}/model.zip`.

Гиперпараметры — секции `agent_dqn` и `agent_sac` в [config.yaml](config.yaml).

### 4. Бэктест + метрики

[lib/backtest.py](lib/backtest.py) → `run_backtest`: детерминированный rollout
(`deterministic=True`), возвращает метрики + equity curve + лог аллокаций.

[lib/metrics.py](lib/metrics.py) → `compute_metrics`: Total return, Sharpe, Sortino,
Max Drawdown, Calmar, win_rate, profit_factor, time_in_market.
**Аннуализация — √2190** (24/7 крипта: 6 четырёхчасовых баров × 365 = 2190 в году).

[lib/bootstrap.py](lib/bootstrap.py) → `bootstrap_ci`: Bootstrap 5000 реплик по
seed-ам → 95% CI для Sharpe / total return.

### 5. Оркестрация

[run.py](run.py) — две фазы: `train_phase` (по `algos × seeds`) и `oos_phase`
(прогон последних `len(seeds)` моделей на каждом OOS-периоде, пишет
`results/oos_{period}.csv`, печатает mean±std).

Флаги: `--algo {SAC,DQN}`, `--seeds ...`, `--skip-train`, `--skip-oos`,
`--oos <key>`, `--no-news` (аблация без новостей).

[lib/config_loader.py](lib/config_loader.py) — `load_config(path)` → типизированные
dataclass'ы + валидация; терпим к устаревшим/лишним ключам в конфиге.

[lib/data_loader.py](lib/data_loader.py) — `load_train` / `load_oos`: режет parquet
по датам, отделяет `raw_close` (реальные цены), остальное → feature matrix,
`sentiment_mean` → отдельный сигнал для награды.

---

## Ключевые параметры по умолчанию

| Настройка | Значение | Где |
|-----------|----------|-----|
| Asset / TF | BTC/USDT, 4h | `data` |
| Train | 2020-01-01 .. 2023-12-31 | `periods.train` |
| OOS | 2024 (полный), 2025-01-01..06-01 | `periods.oos_*` |
| Algos × seeds | **DQN × 5** (основной); SAC опционально | `experiment` |
| Reward | **DSR** + sentiment bonus (λ=0.3, η=0.01) | `env` |
| Window | 30 × 4h баров (~5 дней) | `env.window` |
| TX cost | 0.1% на Δallocation | `env.tx_cost` |
| Признаки | **41** = 8 инд + sentiment_mean + 32 emb | — |
| Embeddings dim | 768 → **32** (PCA on train) | `embeddings` |
| Аннуализация | **√2190** (4h бары) | `lib/metrics.py` |

---

## Quick start

Используется общий проектный `venv` из родительской папки.

### 1. Построить данные
```bash
cd dsr_experiment
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build all
```

### 2. Обучить + OOS
```bash
PYTHONPATH=. ../venv/Scripts/python.exe run.py                          # DQN × seeds + OOS
PYTHONPATH=. ../venv/Scripts/python.exe run.py --algo DQN --seeds 42 123 7 11 99
PYTHONPATH=. ../venv/Scripts/python.exe run.py --skip-train             # OOS на уже обученных
```
Результаты: `models/DQN/{...}/model.zip`, `results/oos_{period}.csv`.

### 3. Дашборд (живая демонстрация)
```bash
../venv/Scripts/streamlit run dashboard/app.py
```
Показывает решение ансамбля DQN (5 сидов) BUY/HOLD/SELL на свежих новостях,
валидацию на 2024/2025 и анализатор отдельной новости.

---

## Структура

```
dsr_experiment/
├── config.yaml              # параметры эксперимента (DQN; терпимый загрузчик)
├── run.py                   # train × seeds + OOS
├── build_data.py            # OHLCV + news → 41 фича
├── lib/
│   ├── config_loader.py     # YAML → typed dataclass + validation
│   ├── data_loader.py       # parquet → (features, prices, sentiment_mean)
│   ├── env.py               # TradingEnv (DSR reward + sentiment bonus)
│   ├── train.py             # SB3 DQN / SAC
│   ├── backtest.py          # deterministic rollout
│   ├── metrics.py           # Sharpe/Sortino/MaxDD/Calmar/... (√2190)
│   ├── bootstrap.py         # bootstrap CI по сидам
│   └── features/
│       ├── price.py         # ccxt OHLCV + 8 минимальных индикаторов + z-score
│       ├── news.py          # HF loader + 4h grouping
│       ├── sentiment.py     # FinBERT pipeline
│       └── embeddings.py    # FinLang + PCA-компрессор (768→32)
├── dashboard/               # Streamlit live-демо (читает снапшот из experiments/)
├── vkr_defense/             # материалы защиты (walkthrough, фигуры)
├── data/, models/, results/, experiments/   # gitignored (артефакты)
```

---

## Главный результат (OOS, √2190, DQN × 5 сидов)

- **OOS 2025**: DQN Sharpe **1.41 ± 0.88** > Buy & Hold (~1.22), MaxDD **15.8%** vs 30.6%.
- **OOS 2024** (бычий): DQN Sharpe 1.64 ± 0.26 (ниже BH по Sharpe, но просадка 23% vs 30%).

Вывод — фазовая зависимость эффекта: на коррекции 2025 DQN с новостным фоном
обыгрывает Buy & Hold и заметно снижает просадку.
