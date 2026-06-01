# Design: 3 Strong Embeddings Agents (PPO / A2C / SAC) with 2024 OOS Backtest

**Date:** 2026-04-11
**Scope:** Single design covering tuned hyperparameters, code changes, training orchestration, and out-of-sample backtest for three RL algorithms on the `embeddings` agent type. Phase 2 extension to 2025–2026 deferred.

## Goal

Produce three production-quality embeddings agents (PPO, A2C, SAC) tuned per SB3 / RL Baselines3 Zoo best practices, trained on BTC 4h 2020–2023, and compared out-of-sample on 2024. `baseline` and `sentiment` agent types are out of scope.

## Current State (from code + data inspection)

- **Train data:** `data/processed/btc_4h_embedding_features.parquet`, 10956 rows, 2020-01-01 → 2024-12-31, 95 feature columns (price + embedding PCA + lags).
- **Train split:** 2020-01-01 – 2023-12-31 = 8765 rows. One episode ≈ 8735 transitions.
- **Test split (Phase 1):** 2024-01-01 – 2024-12-31 = 2191 rows.
- **No 2025/2026 data yet.** Deferred to Phase 2.
- **No ETH 4h embedding features.** BTC only in this design.
- **Embedding encoder:** `FinLang/finance-embeddings-investopedia` (768d, domain-specific). Kept as-is — already a strong choice vs generic MiniLM.
- **PCA:** 768 → 20 components, compressed in `build_embedding_features.py`. **Leakage risk must be verified:** compressor must be fit on train-period news only.
- **Current hyperparams** (see `src/agents/train.py:206-215`): SAC has `buffer_size=10_000` (far below SB3 default 1M, Zoo Mujoco default 1M, Zoo BipedalWalker 300k), `learning_starts=1000`, `train_freq=4`, `gradient_steps=2`. All three algos hardcoded to `device="cpu"`.

## Architectural Approach

Single architecture decision: **hyperparameters live as three factory functions in `src/agents/config.py`**, not in YAML or CLI flags. Rationale: minimum moving parts, full reproducibility via git, direct comparison in diffs. YAML / CLI alternatives rejected as overkill for a thesis project.

## Hyperparameters

All three configs share: `agent_type="embeddings"`, `window=30`, `tx_cost=0.001`, `reward_type="basic"`, `allow_short=False`.

Sources: `rl-baselines3-zoo/hyperparams/{ppo,a2c,sac}.yml`, SB3 `rl_tips.html`, SB3 PPO doc ("meant to be run primarily on the CPU"), araffin comments in SB3 issues #264, #682, #314.

### PPO (device = cpu)

| Param | Value | Source / rationale |
|---|---|---|
| learning_rate | 3e-4 | Zoo PPO default |
| lr_schedule | linear | Zoo best practice for PPO |
| n_steps | 2048 | Zoo Mujoco default |
| batch_size | 64 | Zoo default |
| n_epochs | 10 | Zoo default |
| gamma | 0.99 | standard |
| gae_lambda | 0.95 | Zoo default |
| clip_range | 0.2 | Zoo default |
| ent_coef | 0.01 | continuous exploration (was 0.0 — too deterministic) |
| max_grad_norm | 0.5 | Zoo default |
| use_sde | True | gSDE for continuous action, Zoo recommendation |
| net_arch | [128, 128] | matches 95-feature obs dim, prevents overfit (default [256,256] too big) |
| activation_fn | tanh | PPO canon |
| total_timesteps | 500_000 | ~57 episodes |

### A2C (device = cpu)

| Param | Value | Source |
|---|---|---|
| learning_rate | 7e-4 | Zoo A2C default |
| lr_schedule | linear | Zoo |
| n_steps | 16 | A2C canon (short rollout, not PPO 2048) |
| gamma | 0.99 | standard |
| gae_lambda | 1.0 | Zoo default for A2C |
| ent_coef | 0.01 | continuous exploration |
| max_grad_norm | 0.5 | Zoo default |
| use_sde | True | gSDE continuous |
| normalize_advantage | True | Zoo recommendation |
| net_arch | [128, 128] | same as PPO |
| activation_fn | tanh | A2C canon |
| total_timesteps | 500_000 | same on-policy budget as PPO |

### SAC (device = cuda)

| Param | Value | Source / rationale |
|---|---|---|
| learning_rate | 7.3e-4 | Zoo BipedalWalker (best for short episodes) |
| lr_schedule | constant | SAC does not use lr decay |
| buffer_size | 300_000 | Zoo BipedalWalker. Was 10_000 — critical fix (×30) |
| learning_starts | 10_000 | Zoo Mujoco canonical. Was 1000 (×10) |
| batch_size | 256 | SAC standard |
| tau | 0.02 | Zoo BipedalWalker (short-episode envs). SB3 default 0.005 is for long Mujoco |
| gamma | 0.99 | standard |
| ent_coef | "auto" | automatic entropy temperature tuning |
| train_freq | 8 | was 4. Batched for GPU efficiency |
| gradient_steps | 8 | was 2. Update-to-data ratio = 1.0 |
| use_sde | True | gSDE continuous |
| net_arch | [128, 128] | same as PPO/A2C |
| activation_fn | relu | SAC canon (not tanh) |
| total_timesteps | 200_000 | off-policy is sample-efficient |
| optimize_memory_usage | True | halves replay buffer memory (stores obs_{t+1} implicitly) |

### Device assignment rationale

- **PPO / A2C → CPU.** Official SB3 doc states PPO is meant for CPU. araffin (SB3 author) confirms in issues #264, #682, #314: small MLP policies on vector observations are memory-bound, GPU transfers dominate over matmul. User benchmarks (CartPole) show CPU 13s vs GPU 21s.
- **SAC → CUDA.** Off-policy SAC runs `gradient_steps` × `batch_size=256` gradient updates per env step — this is compute-bound, GPU provides 1.5–3× speedup. Not 10×, but meaningful. `train_freq=8, gradient_steps=8` further batches updates for GPU efficiency.

## Code Changes

### `src/agents/config.py`

1. Add fields to `AgentConfig`: `max_grad_norm: float = 0.5`, `use_sde: bool = False`, `buffer_size: int = 1_000_000`, `tau: float = 0.005`, `train_freq: int = 1`, `gradient_steps: int = 1`, `learning_starts: int = 100`, `device: str = "cpu"`, `normalize_advantage: bool = True`, `optimize_memory_usage: bool = False`.
2. Add three factory functions: `ppo_embeddings_config(seed: int) -> AgentConfig`, `a2c_embeddings_config(seed: int)`, `sac_embeddings_config(seed: int)` returning configs with values from the tables above.

### `src/agents/train.py`

1. Change `device="cpu"` (line 160) to `device=config.device`.
2. Rewrite `_algo_specific_kwargs` (lines 187-215) to read `buffer_size / tau / train_freq / gradient_steps / learning_starts / max_grad_norm / use_sde / normalize_advantage` from `config` instead of hardcoding.
3. Remove hardcoded `FEATURE_COUNTS["embeddings"] = 51` (line 49) or gate it by timeframe: for real (non-dummy) runs, read `features.shape[1]` from the loaded parquet.
4. Change `EMBEDDINGS_NET_ARCH = [128, 64]` (line 54) to `[128, 128]` — 95 features × window 30 = 2850-dim flattened obs, [128,64] is too bottlenecked. Still overridable via `config.net_arch`.
5. `VecNormalize` wrapping stays: PPO/A2C yes, SAC no (replay buffer incompatibility, already handled at line 139).

### `src/eval/backtest.py`

1. Add `vecnormalize_path` optional parameter. When set, load with `VecNormalize.load(path, vec_env)`, then set `vec_env.training = False; vec_env.norm_reward = False`. This restores training-time obs normalization statistics without updating them during test.
2. For SAC models (saved without VecNormalize), skip VecNormalize loading.

### New: `src/agents/run_embeddings_experiments.py`

Orchestrator. For each `(algo, seed)` in `{PPO, A2C, SAC} × {42, 123, 7, 2024, 99}`:
1. Build config via appropriate factory.
2. Call `train_agent(config, asset="BTC/USDT", timeframe="4h", train_start="2020-01-01", train_end="2023-12-31")`.
3. Record (algo, seed, model_path, vecnorm_path, timestamp) to `results/embeddings_runs_v4.csv`.

Sequential execution (no multiprocessing — SAC on GPU would contend). Total estimated time: ~7–10 hours on RTX 3060.

### New: `src/eval/run_oos_backtest.py`

For each row in `results/embeddings_runs_v4.csv`:
1. Load model + VecNormalize stats (if PPO/A2C).
2. Load test features: `btc_4h_embedding_features.parquet` sliced to `2024-01-01 – 2024-12-31`.
3. Run `run_backtest()` → equity curve, allocations, daily returns.
4. Compute metrics via `src/eval/metrics.py` → Sharpe, Sortino, MaxDD, Calmar, Total Return.
5. Compute Bootstrap 95% CI for Sharpe + Total Return via `src/eval/bootstrap.py`.
6. Append row to `results/oos_2024_embeddings.csv`.

Also:
- Compute Buy & Hold baseline via `src/eval/baselines.py` for the same 2024 slice.
- Aggregate: mean ± std per algorithm, Mann-Whitney U test pairwise (PPO vs A2C, PPO vs SAC, A2C vs SAC) on Sharpe.
- Save figures to `results/figures/embeddings_v4/`: (a) equity curves (median seed per algo + B&H, 4 lines total), (b) Sharpe bar chart with error bars.
- Save markdown table to `results/oos_2024_embeddings_report.md`.

## Data Pipeline

### PCA leakage verification

Before training, verify that `EmbeddingCompressor.fit()` was called on news filtered to `2020-01-01 – 2023-12-31` only, not on the full dataset. Check the script that built `btc_4h_embedding_features.parquet`. If leakage exists, rebuild with correct train-period fit.

This is a non-negotiable correctness check — without it, 2024 OOS results are biased.

### Feature count adaptation

`btc_4h_embedding_features.parquet` has 95 feature columns (vs 51 for daily). The design does not hardcode this; `train.py` reads `features.shape[1]` dynamically.

## Execution Phases

### Phase 1 — 2024 backtest (scope of this design)

1. Verify + fix PCA leakage if present.
2. Implement code changes in `config.py`, `train.py`, `backtest.py`.
3. Create `run_embeddings_experiments.py` orchestrator.
4. Run training: 3 algos × 5 seeds = 15 models. Expected time ~7–10 hours.
5. Create `run_oos_backtest.py`.
6. Run OOS backtest on 2024. Generate CSV + figures + markdown report.
7. Git commit Phase 1 complete.
8. User review of results.

### Phase 2 — 2025–2026 extension (deferred, separate sprint)

Out of scope for this design doc. Will be a new spec after Phase 1 review. Key steps noted for context: fetch OHLCV + news through 2026-04-11, rebuild features using **the same PCA compressor from Phase 1** (no refit), re-run OOS backtest (no retraining). Walk-forward split optional.

## Success Criteria

- All 15 models train without crashing.
- `results/oos_2024_embeddings.csv` contains 15 rows with complete metrics.
- At least one algorithm achieves Sharpe > 0 on 2024 OOS (weak bar — we want to see which algo wins, not a profitability guarantee).
- Mann-Whitney U test produces p-values for all three pairwise comparisons.
- Equity curves plot saved to `results/figures/embeddings_v4/`.
- No VecNormalize training-time statistics leak into test (verified by `vec_env.training == False`).

## Risks

1. **PCA leakage in existing parquet.** Mitigation: verify before training, rebuild if needed.
2. **SAC GPU OOM.** Replay buffer stores `obs_t` and `obs_{t+1}`, action, reward, done. Dominant term: 300k × 2 × (30 × 95 + 1) × 4 bytes ≈ 6.8 GB. Fits RTX 3060 (12 GB) but leaves little headroom. Mitigation if OOM: drop `buffer_size` to 150_000 (~3.4 GB), or enable SB3 `optimize_memory_usage=True` (stores obs_{t+1} implicitly, halving memory).
3. **Poor 2024 performance across all algos.** If all Sharpes negative, Phase 2 makes no sense. Gate Phase 2 on Phase 1 review.
4. **Training time overrun.** If >12 hours, reduce SAC `total_timesteps` to 100k (off-policy usually converges early) or PPO/A2C to 300k.

## Out of Scope

- ETH (BTC only).
- Daily timeframe (4h only).
- `baseline`, `sentiment`, `fusion` agent types (embeddings only).
- Switching embedding encoder from `FinLang/finance-embeddings-investopedia`.
- Optuna hyperparameter tuning (using Zoo-derived values directly).
- 2025/2026 data (Phase 2).
