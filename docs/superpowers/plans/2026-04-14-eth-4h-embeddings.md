# ETH/USDT 4h Embeddings Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Обучить SAC, PPO, A2C × 5 seeds на ETH/USDT 4h данных с embeddings + DSR reward — полный аналог лучших BTC-моделей.

**Architecture:** Те же новости (`data/raw/bitcoin_news.parquet`) что и для BTC — это общие крипто-новости влияющие на оба актива. Новый скрипт `build_eth_4h_dataset.py` собирает ETH 4h фичи (101 колонка, идентичная схема с BTC): скачивает ETH/USDT цены, строит технические индикаторы, обучает новый PCA компрессор только на train-новостях (≤2023-12-31), собирает embeddings + sentiment + lag features. Три новые config-функции в `config.py` + оркестратор `train_eth_embeddings.py` обучают 3 алго × 5 seeds с reward_type="dsr", sentiment_lambda=0.1, и запускают OOS backtest на 2024.

**Tech Stack:** Python 3.10+, ccxt (Binance), sentence-transformers (all-MiniLM-L6-v2), FinBERT, SB3 (SAC/PPO/A2C), pandas/numpy, sklearn (PCA).

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `scripts/build_eth_4h_dataset.py` | Create | ETH 4h OHLCV + NLP features → parquet |
| `data/processed/eth_4h_embedding_features.parquet` | Output | 101-col feature file для ETH |
| `data/processed/eth_4h_compressor.pkl` | Output | PCA 384→64, fit только на train |
| `src/agents/config.py` | Modify | Добавить 3 ETH DSR factory-функции |
| `scripts/train_eth_embeddings.py` | Create | Обучение 3 алго × 5 seeds + OOS 2024 |
| `results/oos_2024_eth_embeddings.csv` | Output | OOS метрики 15 моделей |

---

## Task 1: Build ETH 4h OHLCV + baseline features

**Files:**
- Create: `scripts/build_eth_4h_dataset.py` (шаги 1–2)

- [ ] **Step 1: Create `scripts/build_eth_4h_dataset.py` с шагами download + baseline**

```python
"""Build ETH/USDT 4h embedding features dataset.

Usage:
    PYTHONPATH=. venv/Scripts/python.exe scripts/build_eth_4h_dataset.py
    PYTHONPATH=. venv/Scripts/python.exe scripts/build_eth_4h_dataset.py --skip-download
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.price_collector import fetch_ohlcv
from src.features.technical import add_technical_indicators
from src.features.normalizer import rolling_zscore_normalize
from src.data.news_preprocessor import preprocess_news_4h
from src.features.embeddings import compute_embeddings, EMBEDDING_DIM
from src.features.embedding_compressor import EmbeddingCompressor
from src.features.sentiment import compute_sentiment_scores
from src.features.lag_features import add_lag_features, add_rolling_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_OHLCV_PATH = "data/raw/eth_4h_ohlcv.parquet"
RAW_NEWS_PATH = "data/raw/bitcoin_news.parquet"   # same crypto news used for BTC
OUTPUT_BASELINE = "data/processed/eth_4h_features.parquet"
OUTPUT_EMBEDDINGS = "data/processed/eth_4h_embedding_features.parquet"
COMPRESSOR_PATH = "data/processed/eth_4h_compressor.pkl"

COMPRESSED_DIM = 64
TOP_PCA_LAGS = 3
TRAIN_END = "2023-12-31"


def step1_download_ohlcv():
    """Download ETH/USDT 4h OHLCV from Binance."""
    logger.info("Step 1: Downloading ETH/USDT 4h OHLCV from Binance...")
    df = fetch_ohlcv("ETH/USDT", "2020-01-01", "2024-12-31", timeframe="4h")
    Path(RAW_OHLCV_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_OHLCV_PATH)
    logger.info(f"  Saved {len(df)} candles to {RAW_OHLCV_PATH}")
    return df


def step2_baseline_features(ohlcv: pd.DataFrame):
    """Add technical indicators + normalize → baseline parquet."""
    logger.info("Step 2: Building baseline features...")
    df = add_technical_indicators(ohlcv)
    df["raw_close"] = df["close"].copy()
    cols_to_normalize = [c for c in df.columns if c != "raw_close"]
    df[cols_to_normalize] = rolling_zscore_normalize(df[cols_to_normalize], window=30)
    Path(OUTPUT_BASELINE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_BASELINE)
    logger.info(f"  Baseline: {df.shape} saved to {OUTPUT_BASELINE}")
    return df


def step3_embedding_features(baseline_df: pd.DataFrame):
    """Add NLP features (sentiment + embeddings + lags) → embedding parquet."""
    logger.info("Step 3: Loading and preprocessing news for 4h windows...")
    raw_news = pd.read_parquet(RAW_NEWS_PATH)

    col_map = {}
    if "article_text" in raw_news.columns:
        col_map["article_text"] = "text"
    if "date_time" in raw_news.columns:
        col_map["date_time"] = "date"
    if col_map:
        raw_news = raw_news.rename(columns=col_map)
    raw_news["date"] = pd.to_datetime(raw_news["date"], utc=True)

    news_4h = preprocess_news_4h(raw_news)
    logger.info(f"  {len(news_4h)} 4h windows with news")

    # 3a. Compute raw embeddings for PCA fitting — TRAIN ONLY (no leakage)
    logger.info("Step 3a: Computing raw embeddings for PCA (train period only)...")
    train_end_ts = pd.Timestamp(TRAIN_END, tz="UTC")
    news_4h_train = news_4h[news_4h["date"] <= train_end_ts]
    logger.info(f"  Train windows for PCA fit: {len(news_4h_train)}")

    all_embeddings = []
    for idx, row in news_4h_train.iterrows():
        for text in row["texts"]:
            emb = compute_embeddings([text])
            all_embeddings.append(emb)
        if (idx + 1) % 500 == 0:
            logger.info(f"  PCA embeddings: {idx + 1}/{len(news_4h_train)} windows...")

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"  Total embeddings for PCA: {all_embeddings.shape}")

    # 3b. Fit compressor on train data only
    logger.info("Step 3b: Fitting PCA compressor (train only)...")
    compressor = EmbeddingCompressor(input_dim=EMBEDDING_DIM, output_dim=COMPRESSED_DIM)
    compressor.fit(all_embeddings)
    compressor.save(COMPRESSOR_PATH)
    logger.info(f"  Explained variance: {compressor.explained_variance_ratio():.4f}")

    # 3c. Build features per 4h window (ALL windows — train + test)
    logger.info("Step 3c: Building embedding features per 4h window...")
    result = baseline_df.copy()

    emb_cols = [f"emb_{i}" for i in range(COMPRESSED_DIM)]
    for col in emb_cols:
        result[col] = 0.0
    result["news_count"] = 0
    result["sentiment_max"] = 0.0
    result["sentiment_min"] = 0.0
    result["sentiment_spread"] = 0.0

    matched = 0
    for _, row in news_4h.iterrows():
        window_start = row["date"]
        texts = row["texts"]
        if window_start not in result.index:
            continue

        matched += 1
        result.loc[window_start, "news_count"] = len(texts)

        sentiment_scores = compute_sentiment_scores(texts)
        if sentiment_scores:
            result.loc[window_start, "sentiment_max"] = max(sentiment_scores)
            result.loc[window_start, "sentiment_min"] = min(sentiment_scores)
            result.loc[window_start, "sentiment_spread"] = max(sentiment_scores) - min(sentiment_scores)

            weights = [abs(s) + 0.1 for s in sentiment_scores]
            raw_emb = compute_embeddings(texts, weights=weights)
            compressed = compressor.transform(raw_emb.reshape(1, -1))[0]
            result.loc[window_start, emb_cols] = compressed

    logger.info(f"  Matched {matched}/{len(news_4h)} windows to OHLCV index")

    # 3d. Add lag and rolling features
    result = add_lag_features(result, columns=["news_count"], lags=[1, 2])
    result = add_rolling_features(result, columns=["news_count"], window=7)
    top_pca_cols = [f"emb_{i}" for i in range(TOP_PCA_LAGS)]
    result = add_lag_features(result, columns=top_pca_cols, lags=[1, 2])

    result.to_parquet(OUTPUT_EMBEDDINGS)
    feat_cols = [c for c in result.columns if c.lower() not in {"open", "high", "low", "close", "volume", "raw_close"}]
    logger.info(f"  Embeddings: {result.shape}, {len(feat_cols)} features saved to {OUTPUT_EMBEDDINGS}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build ETH/USDT 4h dataset")
    parser.add_argument("--skip-download", action="store_true", help="Skip OHLCV download")
    args = parser.parse_args()

    if args.skip_download and Path(RAW_OHLCV_PATH).exists():
        logger.info("Using cached ETH OHLCV...")
        ohlcv = pd.read_parquet(RAW_OHLCV_PATH)
    else:
        ohlcv = step1_download_ohlcv()

    baseline = step2_baseline_features(ohlcv)
    step3_embedding_features(baseline)
    logger.info("Done! Files saved to data/processed/eth_4h_*.parquet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run OHLCV download + baseline only (быстрая проверка)**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -c "
from src.data.price_collector import fetch_ohlcv
from src.features.technical import add_technical_indicators
from src.features.normalizer import rolling_zscore_normalize
from pathlib import Path
import pandas as pd

df = fetch_ohlcv('ETH/USDT', '2020-01-01', '2024-12-31', timeframe='4h')
print('OHLCV shape:', df.shape)
print('Date range:', df.index.min(), '->', df.index.max())
assert df.shape[0] > 10000, 'Expected >10000 4h candles'
print('OK: ETH 4h OHLCV fetched')
"
```

Ожидается: `OHLCV shape: (~10956, 5)`, `OK: ETH 4h OHLCV fetched`

- [ ] **Step 3: Commit скрипт**

```bash
git add scripts/build_eth_4h_dataset.py
git commit -m "feat(scripts): add build_eth_4h_dataset.py for ETH/USDT 4h embeddings pipeline"
```

---

## Task 2: Run full ETH 4h dataset build (~2–4 часа, GPU/CPU)

**Files:**
- Output: `data/raw/eth_4h_ohlcv.parquet`
- Output: `data/processed/eth_4h_features.parquet`
- Output: `data/processed/eth_4h_embedding_features.parquet`
- Output: `data/processed/eth_4h_compressor.pkl`

- [ ] **Step 1: Run full build script**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe scripts/build_eth_4h_dataset.py 2>&1 | tee results/build_eth_4h_dataset.log
```

Ожидается:
```
Step 1: Downloading ETH/USDT 4h OHLCV from Binance...
  Saved ~10956 candles to data/raw/eth_4h_ohlcv.parquet
Step 2: Building baseline features...
  Baseline: (~10956, 25) saved to data/processed/eth_4h_features.parquet
Step 3: Loading and preprocessing news for 4h windows...
  ~8700 4h windows with news
Step 3a: Computing raw embeddings for PCA (train period only)...
Step 3b: Fitting PCA compressor (train only)...
Step 3c: Building embedding features per 4h window...
  Matched ~8700/8700 windows to OHLCV index
  Embeddings: (~10956, 101), 95 features saved to data/processed/eth_4h_embedding_features.parquet
Done!
```

- [ ] **Step 2: Verify output schema**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/eth_4h_embedding_features.parquet')
price_cols = {'open', 'high', 'low', 'close', 'volume', 'raw_close'}
feat_cols = [c for c in df.columns if c.lower() not in price_cols]

print('Shape:', df.shape)
print('Feature cols:', len(feat_cols))
print('Date range:', df.index.min(), '->', df.index.max())
print('NaN count:', df[feat_cols].isna().sum().sum())
print('news_count nonzero rows:', (df['news_count'] != 0).sum())
print('raw_close sample:', df['raw_close'].dropna().head(3).values)
print('emb_0 nonzero rows:', (df['emb_0'] != 0).sum())

assert df.shape[1] == 101, f'Expected 101 cols, got {df.shape[1]}'
assert len(feat_cols) == 95, f'Expected 95 feature cols, got {len(feat_cols)}'
assert df[feat_cols].isna().sum().sum() == 0, 'NaNs found in features'
assert (df['news_count'] != 0).sum() > 5000, 'Too few windows with news'
print('OK: schema verified')
"
```

Ожидается: `Shape: (~10956, 101)`, `Feature cols: 95`, `NaN count: 0`, `OK: schema verified`

- [ ] **Step 3: Verify PCA leakage — compressor fit только на train**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -c "
from src.features.embedding_compressor import EmbeddingCompressor
c = EmbeddingCompressor.load('data/processed/eth_4h_compressor.pkl')
print('n_components:', c.n_components)
print('input_dim:', c.input_dim)
assert c.n_components == 64, f'Expected 64, got {c.n_components}'
print('OK: compressor has 64 components')
"
```

Ожидается: `n_components: 64`, `OK: compressor has 64 components`

- [ ] **Step 4: Verify load_features_for_agent finds ETH 4h file**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -c "
from src.data.load_features import load_features_for_agent
features, prices, sentiment = load_features_for_agent(
    agent_type='embeddings',
    asset='ETH/USDT',
    train_start='2020-01-01',
    train_end='2023-12-31',
    data_dir='data/processed',
    timeframe='4h',
    return_sentiment=True,
)
print('features shape:', features.shape)
print('prices shape:', prices.shape)
print('sentiment shape:', sentiment.shape)
assert features.shape[1] == 95, f'Expected 95, got {features.shape[1]}'
print('OK: load_features_for_agent works for ETH 4h')
"
```

Ожидается: `features shape: (~10956, 95)`, `OK: load_features_for_agent works for ETH 4h`

- [ ] **Step 5: Commit outputs**

```bash
git add results/build_eth_4h_dataset.log
git commit -m "feat(data): build ETH/USDT 4h embedding features (101 cols, PCA fit on train only)"
```

Примечание: parquet-файлы (~100MB) коммитить не нужно если не используется git-lfs.

---

## Task 3: Add ETH DSR config factory functions

**Files:**
- Modify: `src/agents/config.py`

- [ ] **Step 1: Append три factory-функции в конец `src/agents/config.py`**

```python
def sac_eth_embeddings_dsr_config(seed: int) -> AgentConfig:
    """SAC with DSR reward + sentiment modifier for ETH/USDT 4h embeddings."""
    return AgentConfig(
        algorithm="SAC",
        agent_type="embeddings",
        seed=seed,
        learning_rate=7.3e-4,
        lr_schedule="constant",
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.02,
        gamma=0.99,
        ent_coef="auto",
        train_freq=8,
        gradient_steps=8,
        use_sde=True,
        net_arch=[128, 128],
        activation_fn="relu",
        total_timesteps=200_000,
        optimize_memory_usage=False,
        device="cuda",
        window=30,
        tx_cost=0.001,
        reward_type="dsr",
        sentiment_lambda=0.1,
        allow_short=False,
    )


def ppo_eth_embeddings_dsr_config(seed: int) -> AgentConfig:
    """PPO with DSR reward + sentiment modifier for ETH/USDT 4h embeddings."""
    return AgentConfig(
        algorithm="PPO",
        agent_type="embeddings",
        seed=seed,
        learning_rate=3e-4,
        lr_schedule="linear",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        use_sde=True,
        net_arch=[128, 128],
        activation_fn="tanh",
        total_timesteps=500_000,
        device="cpu",
        window=30,
        tx_cost=0.001,
        reward_type="dsr",
        sentiment_lambda=0.1,
        allow_short=False,
    )


def a2c_eth_embeddings_dsr_config(seed: int) -> AgentConfig:
    """A2C with DSR reward + sentiment modifier for ETH/USDT 4h embeddings."""
    return AgentConfig(
        algorithm="A2C",
        agent_type="embeddings",
        seed=seed,
        learning_rate=7e-4,
        lr_schedule="linear",
        n_steps=16,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        max_grad_norm=0.5,
        use_sde=True,
        normalize_advantage=True,
        net_arch=[128, 128],
        activation_fn="tanh",
        total_timesteps=500_000,
        device="cpu",
        window=30,
        tx_cost=0.001,
        reward_type="dsr",
        sentiment_lambda=0.1,
        allow_short=False,
    )
```

- [ ] **Step 2: Verify import и параметры**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -c "
from src.agents.config import (
    sac_eth_embeddings_dsr_config,
    ppo_eth_embeddings_dsr_config,
    a2c_eth_embeddings_dsr_config,
)
sac = sac_eth_embeddings_dsr_config(42)
ppo = ppo_eth_embeddings_dsr_config(42)
a2c = a2c_eth_embeddings_dsr_config(42)

assert sac.algorithm == 'SAC' and sac.reward_type == 'dsr' and sac.sentiment_lambda == 0.1
assert ppo.algorithm == 'PPO' and ppo.reward_type == 'dsr' and ppo.sentiment_lambda == 0.1
assert a2c.algorithm == 'A2C' and a2c.reward_type == 'dsr' and a2c.sentiment_lambda == 0.1
assert sac.device == 'cuda'
assert ppo.device == 'cpu'
assert a2c.device == 'cpu'
print('OK: all 3 ETH DSR configs import and validate correctly')
"
```

Ожидается: `OK: all 3 ETH DSR configs import and validate correctly`

- [ ] **Step 3: Commit**

```bash
git add src/agents/config.py
git commit -m "feat(config): add SAC/PPO/A2C ETH DSR embeddings config factory functions"
```

---

## Task 4: Create ETH training + OOS orchestrator

**Files:**
- Create: `scripts/train_eth_embeddings.py`

- [ ] **Step 1: Create `scripts/train_eth_embeddings.py`**

```python
"""Train SAC/PPO/A2C embeddings agents on ETH/USDT 4h with DSR reward, run OOS 2024.

Usage:
    # Train SAC only (GPU, ~15 min):
    PYTHONPATH=. venv/Scripts/python.exe scripts/train_eth_embeddings.py --algo SAC

    # Train PPO only (CPU, ~30 min):
    PYTHONPATH=. venv/Scripts/python.exe scripts/train_eth_embeddings.py --algo PPO

    # Train A2C only (CPU, ~30 min):
    PYTHONPATH=. venv/Scripts/python.exe scripts/train_eth_embeddings.py --algo A2C

    # Train all (default):
    PYTHONPATH=. venv/Scripts/python.exe scripts/train_eth_embeddings.py
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.config import (
    sac_eth_embeddings_dsr_config,
    ppo_eth_embeddings_dsr_config,
    a2c_eth_embeddings_dsr_config,
)
from src.agents.train import train_agent
from src.eval.backtest import run_backtest
from src.eval.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 7, 2024, 99]
RESULTS_DIR = Path("results")
ASSET = "ETH/USDT"
TIMEFRAME = "4h"
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
OOS_START = "2024-01-01"
OOS_END = "2024-12-31"
FEATURE_PATH = "data/processed/eth_4h_embedding_features.parquet"

CONFIG_MAP = {
    "SAC": sac_eth_embeddings_dsr_config,
    "PPO": ppo_eth_embeddings_dsr_config,
    "A2C": a2c_eth_embeddings_dsr_config,
}


def _load_oos_features():
    """Load OOS feature matrix and prices from ETH 4h parquet."""
    df = pd.read_parquet(FEATURE_PATH)
    df.index = pd.to_datetime(df.index, utc=True)
    start_ts = pd.Timestamp(OOS_START, tz="UTC")
    end_ts = pd.Timestamp(OOS_END, tz="UTC")
    df = df.loc[start_ts:end_ts]

    if df.empty:
        raise ValueError(f"No data for OOS period {OOS_START}–{OOS_END} in {FEATURE_PATH}")

    prices = df["raw_close"].to_numpy(dtype=np.float64)
    drop = {"open", "high", "low", "close", "volume", "raw_close"}
    feat_cols = [c for c in df.columns if c not in drop]
    features = df[feat_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"OOS features: {features.shape}, prices: {prices.shape}")
    return features, prices


def train_all(algo: str) -> list:
    """Train algo × 5 seeds on ETH/USDT 4h, return list of model paths."""
    config_fn = CONFIG_MAP[algo]
    model_paths = []
    for seed in SEEDS:
        logger.info(f"Training {algo} ETH DSR seed={seed}")
        cfg = config_fn(seed)
        path = train_agent(
            config=cfg,
            dummy=False,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            asset=ASSET,
            data_dir="data/processed",
            timeframe=TIMEFRAME,
        )
        model_paths.append(path)
        logger.info(f"  Saved: {path}")
    return model_paths


def run_oos(model_paths: list, algo: str) -> list:
    """Run OOS 2024 backtest for all models, return result rows."""
    features, prices = _load_oos_features()
    rows = []
    for path, seed in zip(model_paths, SEEDS):
        try:
            result = run_backtest(
                features=features,
                prices=prices,
                model_path=str(path),
                window=30,
                tx_cost=0.001,
            )
            m = result["metrics"]
            rows.append({
                "algorithm": algo,
                "asset": ASSET,
                "seed": seed,
                "reward_type": "dsr",
                **m,
            })
            logger.info(
                f"  {algo} seed={seed}: "
                f"Sharpe={m['sharpe_ratio']:.3f} "
                f"Return={m['total_return'] * 100:.1f}% "
                f"MaxDD={m['max_drawdown'] * 100:.1f}%"
            )
        except Exception as e:
            logger.error(f"  FAILED {algo} seed={seed}: {e}")
    return rows


def save_results(rows: list) -> Path:
    """Append or create OOS results CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "oos_2024_eth_embeddings.csv"
    if not rows:
        logger.warning("No rows to save")
        return csv_path
    fields = list(rows[0].keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows to {csv_path}")
    return csv_path


def print_summary(csv_path: Path):
    """Print mean metrics per algorithm."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    print(f"\n{'='*65}")
    print(f"ETH/USDT OOS 2024 — DSR+sentiment reward")
    print(f"{'='*65}")
    for algo in ["SAC", "PPO", "A2C"]:
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        sharpe = sub["sharpe_ratio"].mean()
        ret = sub["total_return"].mean()
        dd = sub["max_drawdown"].mean()
        print(
            f"  {algo:4s}  Sharpe={sharpe:.3f}  "
            f"Return={ret * 100:.1f}%  MaxDD={dd * 100:.1f}%"
        )
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", choices=["SAC", "PPO", "A2C", "all"], default="all",
        help="Which algorithm to train (default: all)"
    )
    args = parser.parse_args()

    algos = ["SAC", "PPO", "A2C"] if args.algo == "all" else [args.algo]

    for algo in algos:
        model_paths = train_all(algo)
        rows = run_oos(model_paths, algo)
        save_results(rows)

    csv_path = RESULTS_DIR / "oos_2024_eth_embeddings.csv"
    print_summary(csv_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script imports без ошибок**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -c "
import scripts.train_eth_embeddings as m
print('SEEDS:', m.SEEDS)
print('ASSET:', m.ASSET)
print('CONFIG_MAP keys:', list(m.CONFIG_MAP.keys()))
print('OK: script imports without error')
"
```

Ожидается:
```
SEEDS: [42, 123, 7, 2024, 99]
ASSET: ETH/USDT
CONFIG_MAP keys: ['SAC', 'PPO', 'A2C']
OK: script imports without error
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_eth_embeddings.py
git commit -m "feat(scripts): add ETH/USDT 4h DSR embeddings training + OOS orchestrator"
```

---

## Task 5: Train SAC on ETH (GPU, ~15 min)

**Files:**
- Output: `experiments/embeddings/<timestamps>/model.zip` × 5

- [ ] **Step 1: Run SAC training**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe scripts/train_eth_embeddings.py --algo SAC 2>&1 | tee results/train_eth_sac.log
```

Ожидается: 5 моделей сохранены в `experiments/embeddings/`, лог показывает прогресс каждые 50k шагов. OOS результаты для SAC × 5 seeds добавлены в `results/oos_2024_eth_embeddings.csv`.

- [ ] **Step 2: Verify 5 SAC models saved**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('results/oos_2024_eth_embeddings.csv')
sac_rows = df[df['algorithm'] == 'SAC']
print(sac_rows[['algorithm', 'seed', 'sharpe_ratio', 'total_return', 'max_drawdown']].to_string())
assert len(sac_rows) == 5, f'Expected 5 SAC rows, got {len(sac_rows)}'
print('OK: 5 SAC OOS results saved')
"
```

---

## Task 6: Train PPO on ETH (CPU, ~30 min)

**Files:**
- Output: `experiments/embeddings/<timestamps>/model.zip` × 5

- [ ] **Step 1: Run PPO training**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe scripts/train_eth_embeddings.py --algo PPO 2>&1 | tee results/train_eth_ppo.log
```

- [ ] **Step 2: Verify 5 PPO rows in CSV**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -c "
import pandas as pd
df = pd.read_csv('results/oos_2024_eth_embeddings.csv')
ppo_rows = df[df['algorithm'] == 'PPO']
print(ppo_rows[['algorithm', 'seed', 'sharpe_ratio', 'total_return', 'max_drawdown']].to_string())
assert len(ppo_rows) == 5, f'Expected 5 PPO rows, got {len(ppo_rows)}'
print('OK: 5 PPO OOS results saved')
"
```

---

## Task 7: Train A2C on ETH (CPU, ~30 min)

**Files:**
- Output: `experiments/embeddings/<timestamps>/model.zip` × 5

- [ ] **Step 1: Run A2C training**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe scripts/train_eth_embeddings.py --algo A2C 2>&1 | tee results/train_eth_a2c.log
```

- [ ] **Step 2: Verify final CSV has 15 rows**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -c "
import pandas as pd
df = pd.read_csv('results/oos_2024_eth_embeddings.csv')
print(df[['algorithm', 'seed', 'sharpe_ratio', 'total_return', 'max_drawdown']].to_string())
assert len(df) == 15, f'Expected 15 rows (SAC+PPO+A2C x5), got {len(df)}'
print()
print('Summary by algo:')
print(df.groupby('algorithm')[['sharpe_ratio', 'total_return', 'max_drawdown']].mean().round(4))
print('OK: all 15 ETH OOS results saved')
"
```

Ожидается: 15 строк (SAC×5, PPO×5, A2C×5), таблица mean метрик по алгоритмам.

- [ ] **Step 3: Commit results**

```bash
git add results/oos_2024_eth_embeddings.csv results/train_eth_sac.log results/train_eth_ppo.log results/train_eth_a2c.log
git commit -m "results(eth): OOS 2024 ETH/USDT embeddings DSR — SAC/PPO/A2C x5 seeds"
```

---

## Self-Review

**Spec coverage:**
- ✅ ETH 4h OHLCV download (Task 1)
- ✅ Те же крипто-новости что и BTC — `bitcoin_news.parquet` (Task 2)
- ✅ Новый PCA компрессор, fit только на train ≤2023-12-31 (Task 2)
- ✅ Schema 101 cols / 95 features — идентично BTC 4h (Task 2, verification step)
- ✅ `load_features_for_agent` ETH/USDT 4h проверяется (Task 2, step 4)
- ✅ SAC/PPO/A2C ETH DSR configs (Task 3)
- ✅ Оркестратор train + OOS (Task 4)
- ✅ SAC × 5 seeds обучение (Task 5)
- ✅ PPO × 5 seeds обучение (Task 6)
- ✅ A2C × 5 seeds обучение (Task 7)
- ✅ Итоговый CSV с 15 строками (Task 7, step 2)

**Placeholder scan:** нет TBD/TODO.

**Type consistency:**
- `sac_eth_embeddings_dsr_config` / `ppo_eth_embeddings_dsr_config` / `a2c_eth_embeddings_dsr_config` — определены в Task 3, импортированы в Task 4 скрипте
- `run_backtest(features, prices, model_path, window, tx_cost)` — та же сигнатура что используется в `run_dsr_experiment.py`
- `result["metrics"]` — тот же ключ что в существующем `run_backtest()` → `compute_metrics()` возвращает `sharpe_ratio, total_return, max_drawdown, sortino_ratio, calmar_ratio`
- `SEEDS = [42, 123, 7, 2024, 99]` — те же seeds что в BTC экспериментах
