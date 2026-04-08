# Embedding Agent Improvements — Design Spec

**Date**: 2026-04-08
**Goal**: Improve embeddings agent stability, absolute performance, and statistical significance vs baseline.

## Problem Statement

Current embeddings agent (BTC/USDT, 3 seeds) shows high variance:
- Seed 42: +42.9% return, Sharpe 1.43
- Seed 43: -0.7% return, Sharpe 0.00
- Seed 44: +5.2% return, Sharpe 0.15
- Mean Sharpe: ~0.53 with std ~0.80

Baseline is more stable (mean Sharpe ~0.36, std ~0.33). The embeddings agent does not statistically significantly outperform baseline.

**Root causes identified:**
1. PCA 768->64 overfits on ~1400 training days
2. Net arch [512, 256] too large for obs dimension
3. No entropy bonus -> poor exploration
4. Noisy basic reward (raw log_return)
5. Agent doesn't know its own position (prev_allocation missing from obs)
6. No running normalization of obs/rewards
7. Duplicate news inflate embedding mean pooling
8. No temporal weighting of news

---

## Section 1: Embedding Pipeline Fixes

### 1.1 PCA dimension: 64 -> 20

Reduce `EmbeddingCompressor.output_dim` from 64 to 20. Rule of thumb: n_components < sqrt(n_samples) ~ 37, but 20 is safer given noise level. Reduces obs space from 95 to ~51 features.

**Files**: `src/features/embedding_compressor.py`, `src/features/build_embedding_features.py`

### 1.2 Cosine similarity dedup (threshold=0.85)

Before mean pooling: compute pairwise cosine similarity between article embeddings for each day. If sim > 0.85, keep only one. Removes reposts and rewrites that duplicate signal.

**Files**: `src/features/build_embedding_features.py` (new helper function)

### 1.3 Temporal decay weighting

Replace equal mean pooling with exponential decay: `weight_i = exp(-alpha * hours_since_close)` where alpha=0.1. News closer to candle close weighs more. The existing `weights` parameter in `compute_embeddings()` already supports this — only need to compute weights in `build_embedding_features.py`.

**Files**: `src/features/build_embedding_features.py`

### 1.4 Net arch: [512, 256] -> [128, 64]

For ~1530d obs vector (51 features x 30 window), [128, 64] = ~20K params is proportional to data. Current [512, 256] = ~400K params overfits.

**Files**: `src/agents/train.py` (EMBEDDINGS_NET_ARCH)

---

## Section 2: Environment & Reward Upgrades

### 2.1 Differential Sharpe Ratio (DSR) reward

New `reward_type="dsr"` in `TradingEnv`. Incremental Sharpe at each step:

```
A_t = A_{t-1} + eta * (R_t - A_{t-1})         # EMA of mean return
B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})       # EMA of squared return
DSR_t = (B_{t-1} * dA - 0.5 * A_{t-1} * dB) / (B_{t-1} - A_{t-1}^2)^{3/2}
```

Where `eta=0.01`, `R_t = log_return * allocation - tx_cost * |delta|`. Fallback to basic reward in first steps while statistics accumulate.

**Files**: `src/env/trading_env.py`

### 2.2 prev_allocation in observation

Append `self.prev_allocation` to flattened window in `_get_obs()`. Agent needs to know current position to evaluate cost of position change. Obs space grows by 1.

**Files**: `src/env/trading_env.py`

### 2.3 VecNormalize wrapper

Wrap env in `DummyVecEnv` + `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=5.0)` in `train.py`. Save normalization stats alongside model for backtest loading.

**Files**: `src/agents/train.py`, `src/eval/backtest.py`

---

## Section 3: Sentiment-Weighted Reward (SAPPO-lite)

### 3.1 Sentiment as reward modifier

```
reward_final = reward_dsr + lambda * sentiment_signal * R_t
```

Where `lambda=0.1`, `sentiment_signal` = mean sentiment for current day, `R_t` = log_return. Positive sentiment + price up = bonus. Negative sentiment + price down = bonus (correct "understanding" of market).

### 3.2 Implementation

- `TradingEnv.__init__()` accepts optional `sentiment_signal: np.ndarray` (length = len(prices))
- `step()` for DSR reward: add `lambda * sentiment[step] * log_return`
- For baseline/non-embedding agents: sentiment_signal=None, modifier=0
- `lambda` as `AgentConfig.sentiment_lambda` (default=0.1)

**Files**: `src/env/trading_env.py`, `src/agents/config.py`, `src/agents/train.py`

---

## Section 4: Hyperparameters & Training

### 4.1 New defaults for embeddings agent

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| gamma | 0.99 | 0.95 | Shorter horizon, less noise from far future |
| ent_coef | 0.0 | 0.01 | Exploration bonus |
| net_arch | [512, 256] | [128, 64] | Proportional to data |
| pca_dim | 64 | 20 | Less overfit |
| total_timesteps | 500K | 1M | More steps for DSR convergence |
| clip_range | 0.2 | 0.15 | More conservative updates |

Applied only for `agent_type in ("embeddings", "fusion")`. Baseline/sentiment keep current params for fair ablation.

### 4.2 Seeds: 3 -> 5

Add seeds 45, 46. Minimum for meaningful t-test at p<0.05.

### 4.3 Comparison with past results

New script `compare_results.py`:
- Save old metrics as `results/metrics_v1.csv` before new run
- Load old + new metrics
- Table: delta (Sharpe_new - Sharpe_old) per seed
- t-test: embeddings_new vs embeddings_old, embeddings_new vs baseline
- Bar plot: old vs new side-by-side

**Files**: new `experiments/compare_results.py`

---

## Feature count update

```
FEATURE_COUNTS = {
    "baseline": 18,
    "sentiment": 19,
    "embeddings": 47,   # 18 base + 20 emb + news_count(3) + roll7 + 3 sent extremes + 2 PCA lags (top-1 x 2) + 1 prev_alloc
    "fusion": 50,        # embeddings(47) + 3 sentiment (raw + lag1 + lag2)
}
```

Note: exact count will be validated during implementation after PCA dim change and lag adjustment.

---

## Out of scope

- Optuna hyperparameter sweep (Approach C)
- Ensemble PPO+SAC
- Walk-forward validation
- Attention-based fusion
- Fine-tuning FinBERT (already implemented, can be enabled separately)
