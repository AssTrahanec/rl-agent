# DSR Experiment

Полностью автономная папка для запуска DSR-эксперимента: обучение SAC/PPO
агентов с DSR + sentiment reward на BTC 4h embeddings, OOS backtest на
произвольных периодах.

Все параметры — в [config.yaml](config.yaml).

## Установка

Использует общий проектный `venv` (родительская папка). Зависимости уже стоят
в `../venv` (см. `../requirements.txt`).

```bash
# из корня репо:
cd dsr_experiment
```

## Quick start

### 1. Построить данные

```bash
# Полный цикл: скачать OHLCV+news, построить train, фитнуть PCA, построить все OOS:
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build all

# Или по частям:
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build train
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build oos_2024
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build oos_2025
```

После build данные лежат в:
```
data/raw/{ohlcv,news}.parquet
data/train/{features.parquet, compressor.pkl}
data/oos/{oos_2024,oos_2025}_features.parquet
```

### 2. Обучить + OOS

```bash
# Полный пайплайн (train SAC+PPO × 5 seeds, OOS на всех periods):
PYTHONPATH=. ../venv/Scripts/python.exe run.py

# Только SAC:
PYTHONPATH=. ../venv/Scripts/python.exe run.py --algo SAC

# Только OOS на ранее обученных моделях:
PYTHONPATH=. ../venv/Scripts/python.exe run.py --skip-train

# Только OOS на одном периоде:
PYTHONPATH=. ../venv/Scripts/python.exe run.py --skip-train --oos oos_2025

# Только обучение, без OOS:
PYTHONPATH=. ../venv/Scripts/python.exe run.py --skip-oos
```

Результаты:
```
models/{SAC,PPO}/{algo}_seed{seed}_{ts}/model.zip
results/oos_{period}.csv
```

## Добавить новый OOS период

В `config.yaml`:
```yaml
periods:
  ...
  oos_2026: { start: "2026-01-01", end: "2026-04-18" }

experiment:
  oos_periods: ["oos_2024", "oos_2025", "oos_2026"]
```

Затем:
```bash
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build oos_2026
PYTHONPATH=. ../venv/Scripts/python.exe run.py --skip-train --oos oos_2026
```

## Структура

```
dsr_experiment/
├── config.yaml         # ВСЕ параметры
├── run.py              # train + OOS
├── build_data.py       # скачивание + features
├── lib/
│   ├── config_loader.py
│   ├── env.py          # TradingEnv
│   ├── train.py        # train_agent
│   ├── backtest.py     # run_backtest
│   ├── metrics.py
│   ├── bootstrap.py
│   ├── data_loader.py
│   └── features/
│       ├── price.py
│       ├── news.py
│       ├── sentiment.py
│       ├── embeddings.py
│       └── lag.py
├── data/               # gitignored
├── models/             # gitignored
└── results/            # gitignored
```
