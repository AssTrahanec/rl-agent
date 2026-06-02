# Карта артефактов: что и куда сохраняется по пути новости

> Где на каждом этапе пишутся файлы, в каком формате и что внутри. Пути — из
> [`config.yaml`](../config.yaml) (`data.paths`) и дефолтов `run.py`.
> Все артефакты данных/моделей **gitignored** (в репозиторий не коммитятся — только код).

---

## Схема: путь новости + файлы на каждом шаге

```
HuggingFace: edaschau/bitcoin_news
   │  build_data.py → ensure_raw_news()  (качается ОДИН раз, потом кэш)
   ▼
📄 data/raw/news.parquet            [текст, дата, заголовок] весь корпус 2020–2025
   │
   │  preprocess_news_4h → FinBERT → FinLang(768d) → PCA.fit (ТОЛЬКО train)
   ▼
📄 data/train/compressor.pkl        обученный PCA 768→32 (матрица весов)
   │
   │  _attach_nlp_features_minimal:  sentiment_mean + взвеш.среднее эмб → PCA → emb_0..31
   │  + 8 индикаторов из ohlcv (см. ниже)
   ▼
📄 data/train/features.parquet      train: ~8760 строк × (41 фича + цены)
📄 data/oos/oos_2024_features.parquet   \  то же на тестовых периодах
📄 data/oos/oos_2025_features.parquet   /  (build_oos берёт ГОТОВЫЙ compressor.pkl)
   │
   │  run.py: train_agent (обучение) → oos_phase (бэктест)
   ▼
📄 models/DQN/DQN_seed{S}_{ts}/model.zip      веса обученной DQN-сети (по сиду)
📄 results/oos_{period}.csv                   по строке на (алгоритм, сид): все метрики
📄 results/oos_{period}_DQN_seed{S}.npz       daily_returns, allocations, equity_curve
   │
   │  промоут (копирование) для дашборда
   ▼
📁 experiments/run_2026-06-01_minimal_dqn5/{models/DQN/, results/}   ← отсюда читает дашборд
```

Цены идут параллельно: `data/raw/ohlcv.parquet` (кэш свечей Binance) → 8 индикаторов →
z-score → склеиваются с новостными фичами в тот же `features.parquet`.

---

## Таблица артефактов

| # | Что | Файл | Формат | Что внутри | Кто пишет (функция) |
|---|---|---|---|---|---|
| 1 | Сырые новости | `data/raw/news.parquet` | parquet | `text, date, title` — весь корпус | `build_data.ensure_raw_news` |
| 2 | Сырые цены | `data/raw/ohlcv.parquet` | parquet | OHLCV 4h-свечи | `build_data.ensure_raw_ohlcv` |
| 3 | PCA-компрессор | `data/train/compressor.pkl` | pickle | обученный PCA 768→32 | `build_data.build_train` → `EmbeddingCompressor.save` |
| 4 | Train-фичи | `data/train/features.parquet` | parquet | 41 фича + `raw_close`/OHLCV | `build_data.build_train` |
| 5 | OOS-фичи | `data/oos/oos_2024_features.parquet`, `…2025…` | parquet | то же, тестовые периоды | `build_data.build_oos` |
| 6 | Модели | `models/DQN/DQN_seed{S}_{TS}/model.zip` | zip (SB3) | веса DQN-сети | `lib/train.train_agent` → `model.save` |
| 7 | Метрики OOS | `results/oos_oos_2024.csv`, `…2025.csv` | csv | строка на (algo, seed): Sharpe/MaxDD/… | `run.oos_phase` |
| 8 | Ряды по сидам | `results/oos_{period}_DQN_seed{S}.npz` | npz | `daily_returns`, `allocations`, `equity_curve` | `run.oos_phase` → `np.savez` |
| 9 | Снапшот дашборда | `experiments/run_2026-06-01_minimal_dqn5/` | папка | копия #6,7,8 для UI | промоут (копирование) |

---

## Что НЕ сохраняется отдельно (считается «на лету»)

- **FinBERT-тональности** каждой статьи — не кэшируются; считаются в
  `_attach_nlp_features_minimal` и сразу пишутся в колонку `sentiment_mean` файла #4/#5.
- **FinLang 768d-эмбеддинги** — тоже не сохраняются сырыми: при сборке train их кодируют
  для обучения PCA, потом снова на каждое окно — и пишут уже **сжатые** `emb_0..31` в #4/#5.
  (Это, кстати, кандидат на кэширование, чтобы пересборка была быстрее.)

> **Зачем так.** В parquet (#4/#5) лежит уже готовый вход модели — 41 число на бар. Промежуточные
> сырые эмбеддинги (768d × десятки тысяч окон) заняли бы гигабайты и не нужны после сжатия.

---

## Дашборд (живой режим) — отдельный кэш

Дашборд не использует #1–#5 для инференса заново, а:
- **свежие** новости тянет с NewsAPI/RSS и кэширует в свой parquet (`dashboard/cache/…`),
- считает фичи тем же кодом (`lib/features/*`) + тем же `compressor.pkl` (#3),
- модели и сохранённые OOS-результаты берёт из снапшота #9.

---

## Как воспроизвести (команды)

```bash
cd dsr_experiment
# 1–5: собрать данные (пишет raw/train/oos)
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build all
# 6–8: обучить DQN + OOS (пишет models/ и results/)
PYTHONPATH=. ../venv/Scripts/python.exe run.py --algo DQN --seeds 42 123 7 11 99
```
