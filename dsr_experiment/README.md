# DSR Experiment

Автономная папка финального эксперимента магистерской работы:
**Оптимизация стратегий торговли с помощью RL и обработки новостных текстов.**

Обучает SAC и PPO агенты с **DSR-reward** и sentiment-бонусом на
BTC/USDT 4h OHLCV + финансовые эмбеддинги новостей, затем гоняет OOS-бэктесты
на произвольных временных периодах.

Все гиперпараметры и пути — в [config.yaml](config.yaml).
Быстрый smoke-прогон — [config.smoke.yaml](config.smoke.yaml).

---

## Что именно реализовано

### 1. Данные и фичи (пайплайн [build_data.py](build_data.py))

| Этап | Файл | Суть |
|------|------|------|
| OHLCV через `ccxt` Binance (4h) | [lib/features/price.py](lib/features/price.py) → `fetch_ohlcv` | Скачивание свечей + кэш в `data/raw/ohlcv.parquet` |
| 20+ технических индикаторов | [lib/features/price.py](lib/features/price.py) → `add_technical_indicators` | SMA, EMA, MACD, RSI, Bollinger, ATR, OBV, Stochastic, returns |
| Rolling z-score нормализация | [lib/features/price.py](lib/features/price.py) → `rolling_zscore_normalize` | window=30, NaN→0, сохраняет `raw_close` для бэктеста |
| Загрузка новостей из HF | [lib/features/news.py](lib/features/news.py) → `load_news_from_hf` | `edaschau/bitcoin_news`, фильтр по датам, кэш в `data/raw/news.parquet` |
| Группировка в 4h окна | [lib/features/news.py](lib/features/news.py) → `preprocess_news_4h` | `dt.floor("4h")` → список текстов на окно |
| Дедупликация по cosine sim | [lib/features/news.py](lib/features/news.py) → `deduplicate_embeddings` | Порог 0.85 (настраивается в конфиге) |
| FinBERT sentiment | [lib/features/sentiment.py](lib/features/sentiment.py) → `compute_sentiment_scores` | `ProsusAI/finbert`, score ∈ [-1, +1], кэш pipeline |
| Sentence embeddings | [lib/features/embeddings.py](lib/features/embeddings.py) → `compute_embeddings` | `FinLang/finance-embeddings-investopedia` (768d), взвешенное среднее |
| PCA-компрессор 768→64 | [lib/features/embeddings.py](lib/features/embeddings.py) → `EmbeddingCompressor` | `fit` только на train, сохраняется `compressor.pkl`, применяется к OOS |
| Лаги и rolling по news_count / топ-PCA компонент | [lib/features/lag.py](lib/features/lag.py) → `add_lag_features`, `add_rolling_features` | лаги [1,2], rolling 7 |
| Склейка всех фичей | [build_data.py](build_data.py) → `_attach_nlp_features` | Добавляет 64 emb-колонки + `news_count` + `sentiment_{max,min,spread}` + лаги |

На выходе — один parquet на период: [`data/train/features.parquet`](data/train/) и
[`data/oos/{period_key}_features.parquet`](data/oos/).

### 2. Trading environment

[lib/env.py](lib/env.py) → `TradingEnv` (Gymnasium):

- **Action space**: `Box([0,1])` (или `[-1,1]`, если `allow_short=true`).
- **Observation**: сплющенное окно последних `window` 4h-баров (`window × n_features`) + текущая доля аллокации → `(window·n_features + 1,)`.
- **Transaction cost**: `tx_cost · |Δallocation|`, вычитается из шагового P&L.
- **Три типа награды** (выбирается в `env.reward_type`):
  - `basic` — `log_return · allocation − tx_penalty`;
  - `risk_adjusted` — та же базовая отдача минус штраф на волатильность оборота (`volatility_penalty · |Δallocation|`);
  - `dsr` — **Differential Sharpe Ratio** (Moody & Saffell): рекурсивно обновляются экспоненциальные моменты `A`, `B` с темпом `dsr_eta`, награда = `(B·ΔA − 0.5·A·ΔB) / (B − A²)^(3/2)`. Реализация в [lib/env.py:96-109](lib/env.py#L96-L109).
- **Sentiment-бонус** (только при `reward_type=dsr`): `+ sentiment_lambda · sentiment_t · log_return_t`. Сигнал — `sentiment_max` из фичей, подмешивается на каждом шаге ([lib/env.py:86-88](lib/env.py#L86-L88)).

### 3. Обучение

[lib/train.py](lib/train.py) → `train_agent(cfg, algo, seed, features, prices, sentiment)`:

- Стейбл-baselines3: `PPO` (CPU) и `SAC` (CUDA) ([lib/train.py:17](lib/train.py#L17)).
- Для PPO — `VecNormalize` (обс+reward), для SAC — сырой env ([lib/train.py:100-102](lib/train.py#L100-L102)). `vecnormalize.pkl` сохраняется рядом с моделью и загружается на OOS.
- Linear / constant learning-rate schedule, `use_sde=true`, `net_arch=[128,128]` по умолчанию.
- `ProgressCallback` печатает прогресс каждые 50k шагов.
- Модель кладётся в `models/{algo}/{algo}_seed{seed}_{ts}/model.zip`.

Гиперпараметры — секции `agent_sac` и `agent_ppo` в [config.yaml](config.yaml).

### 4. Бэктест + метрики

[lib/backtest.py](lib/backtest.py) → `run_backtest`:
- Детерминированный rollout (`deterministic=True`).
- Корректно подхватывает `VecNormalize` статистики с `training=False, norm_reward=False`.
- Возвращает метрики + equity curve + лог аллокаций.

[lib/metrics.py](lib/metrics.py) → `compute_metrics`:
Total return, Sharpe, Sortino, Max Drawdown, Calmar (аннуализация × √365, т.к. 4h/дневной подход принят как 365 периодов для консистентности).

[lib/bootstrap.py](lib/bootstrap.py) → `bootstrap_ci`:
Bootstrap 5000 реплик по seed-ам → 95% CI для Sharpe / total return (используется при необходимости из ноутбуков).

### 5. Оркестрация

[run.py](run.py) — две фазы:
1. `train_phase` — по всем `experiment.algos × experiment.seeds`.
2. `oos_phase` — для каждого периода из `experiment.oos_periods` (или `--oos`) прогоняет последние `len(seeds)` моделей каждого алгоритма, пишет `results/oos_{period}.csv`, печатает сводку mean±std.

Флаги: `--algo {SAC,PPO}`, `--skip-train`, `--skip-oos`, `--oos <key>` (повторяемый).
`find_latest_models` ([run.py:45-56](run.py#L45-L56)) берёт самые свежие обученные модели по mtime, если запуск идёт с `--skip-train`.

[build_data.py](build_data.py) — сборка датасетов:
- `--build train` — фиттит PCA-компрессор и пишет train parquet.
- `--build oos_YYYY` — использует уже сохранённый компрессор.
- `--build all` — всё сразу.

[lib/config_loader.py](lib/config_loader.py) — `load_config(path)` → типизированные dataclass'ы + валидация (обязательный `periods.train`, допустимые `reward_type`/`algos`, `compressed_dim ≤ raw_dim`).

[lib/data_loader.py](lib/data_loader.py) — `load_train` / `load_oos`: режет parquet по датам периода, отделяет `raw_close` для реальных цен, остальные колонки → feature matrix, `sentiment_max` → отдельный сигнал.

---

## Ключевые параметры по умолчанию

| Настройка | Значение | Где |
|-----------|----------|-----|
| Asset / TF | BTC/USDT, 4h | `data` |
| Train | 2020-01-01 .. 2023-12-31 | `periods.train` |
| OOS | 2024 (полный), 2025-01-01..04-10 | `periods.oos_*` |
| Seeds × algos | 5 × {SAC, PPO} = 10 моделей | `experiment` |
| Reward | **DSR** + sentiment bonus (λ=0.1, η=0.01) | `env` |
| Window | 30 × 4h баров (~5 дней) | `env.window` |
| TX cost | 0.1% на Δallocation | `env.tx_cost` |
| Embeddings dim | 768 → 64 (PCA on train) | `embeddings` |
| Dedup threshold | cosine 0.85 | `news.dedup_threshold` |

---

## Quick start

Используется общий проектный `venv` из родительской папки.

### 1. Построить данные
```bash
cd dsr_experiment
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build all
```
После:
```
data/raw/{ohlcv,news}.parquet
data/train/{features.parquet, compressor.pkl}
data/oos/{oos_2024,oos_2025}_features.parquet
```

### 2. Обучить + OOS
```bash
PYTHONPATH=. ../venv/Scripts/python.exe run.py                    # полный пайплайн
PYTHONPATH=. ../venv/Scripts/python.exe run.py --algo SAC          # только SAC
PYTHONPATH=. ../venv/Scripts/python.exe run.py --skip-train        # OOS на уже обученных
PYTHONPATH=. ../venv/Scripts/python.exe run.py --skip-train --oos oos_2025
PYTHONPATH=. ../venv/Scripts/python.exe run.py --skip-oos          # только обучение
```
Результаты:
```
models/{SAC,PPO}/{algo}_seed{seed}_{ts}/model.zip (+ vecnormalize.pkl для PPO)
results/oos_{period}.csv                 # по строке на (algo, seed)
```

### 3. Smoke-тест (несколько минут, CPU)
```bash
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --config config.smoke.yaml --build all
PYTHONPATH=. ../venv/Scripts/python.exe run.py --config config.smoke.yaml
```
`config.smoke.yaml` сокращает train до полугода, один seed, 5k шагов, только SAC — для проверки, что всё собирается end-to-end.

---

## Добавить новый OOS период

В [config.yaml](config.yaml):
```yaml
periods:
  oos_2026: { start: "2026-01-01", end: "2026-04-18" }

experiment:
  oos_periods: ["oos_2024", "oos_2025", "oos_2026"]
```
Затем:
```bash
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build oos_2026
PYTHONPATH=. ../venv/Scripts/python.exe run.py --skip-train --oos oos_2026
```

---

## Структура

```
dsr_experiment/
├── config.yaml              # все параметры эксперимента
├── config.smoke.yaml        # быстрый end-to-end прогон
├── run.py                   # train × seeds + OOS
├── build_data.py            # OHLCV + news → features
├── lib/
│   ├── config_loader.py     # YAML → typed dataclass + validation
│   ├── data_loader.py       # features parquet → (features, prices, sentiment)
│   ├── env.py               # TradingEnv (basic / risk_adjusted / DSR + sentiment bonus)
│   ├── train.py             # SB3 PPO / SAC с VecNormalize и LR-schedule
│   ├── backtest.py          # deterministic rollout + VecNormalize loading
│   ├── metrics.py           # Sharpe / Sortino / MaxDD / Calmar / total
│   ├── bootstrap.py         # bootstrap CI по сидам
│   └── features/
│       ├── price.py         # ccxt OHLCV + 20+ индикаторов + z-score
│       ├── news.py          # HF loader + 4h grouping + cosine dedup
│       ├── sentiment.py     # FinBERT pipeline
│       ├── embeddings.py    # SentenceTransformer + PCA compressor
│       └── lag.py           # add_lag_features / add_rolling_features
├── data/                    # gitignored (raw + processed)
├── models/                  # gitignored (обученные модели + VecNormalize)
└── results/                 # gitignored (oos_{period}.csv)
```

---

## Роль в общем проекте

Этот пакет — финализированная версия ключевого эксперимента:
- vs. базовый пайплайн в [../src/](../src/) (3-way ablation по агентам 1/2/3/fusion), здесь оставлен **один** лучший feature set (price + sentiment-extremes + 64d PCA embeddings + лаги) и сосредоточено сравнение по **reward'у** (DSR + sentiment-bonus) и **алгоритму** (SAC vs PPO) с множественными seed'ами и OOS на нескольких годах;
- используется финансовый эмбеддер `FinLang/finance-embeddings-investopedia` (768d) вместо универсального `all-MiniLM-L6-v2` из базового пайплайна;
- 4h таймфрейм вместо дневного — больше обучающих шагов и более тонкая реакция на новости;
- выходные `results/oos_*.csv` — агрегируются в ноутбуках/скриптах в `../notebooks/` и `../scripts/` для итоговых таблиц диплома.
