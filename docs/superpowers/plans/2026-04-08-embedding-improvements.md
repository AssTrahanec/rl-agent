# Embedding Agent Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve embeddings agent stability, absolute Sharpe, and statistical significance vs baseline by fixing the embedding pipeline, adding DSR reward, prev_allocation in obs, VecNormalize, SAPPO-lite reward modifier, and tuned hyperparameters.

**Architecture:** Six incremental changes, each testable independently: (1) PCA 64->20 + cosine dedup + temporal decay, (2) DSR reward in TradingEnv, (3) prev_allocation in obs, (4) VecNormalize in train/backtest, (5) SAPPO-lite sentiment reward modifier, (6) hyperparameter update + compare_results script.

**Tech Stack:** Python 3.10+, stable-baselines3, gymnasium, numpy, scikit-learn, scipy, matplotlib, pandas

**Spec:** `docs/superpowers/specs/2026-04-08-embedding-improvements-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/features/embedding_compressor.py` | Modify | Change default output_dim 64 -> 20 |
| `src/features/build_embedding_features.py` | Modify | Add cosine dedup, temporal decay, update COMPRESSED_DIM to 20 |
| `src/env/trading_env.py` | Modify | Add DSR reward, prev_allocation in obs, sentiment_signal |
| `src/agents/config.py` | Modify | Add ent_coef, sentiment_lambda fields |
| `src/agents/train.py` | Modify | VecNormalize wrapper, updated net_arch/hyperparams, pass sentiment |
| `src/eval/backtest.py` | Modify | Load VecNormalize stats, pass sentiment_signal |
| `src/data/load_features.py` | Modify | Extract sentiment column for SAPPO-lite |
| `experiments/run_all.py` | Modify | 5 seeds, pass reward_type/sentiment_lambda |
| `experiments/compare_results.py` | Create | Old vs new metrics comparison + t-test |
| `tests/test_trading_env.py` | Modify | Tests for DSR, prev_allocation, sentiment |
| `tests/test_compressor.py` | Modify | Update default dim test |
| `tests/test_build_embedding.py` | Modify | Update emb column count, add dedup test |
| `tests/test_compare_results.py` | Create | Tests for compare script |

---

### Task 1: PCA dimension 64 -> 20

**Files:**
- Modify: `src/features/embedding_compressor.py:15`
- Modify: `src/features/build_embedding_features.py:13-14`
- Modify: `src/agents/train.py:46-49`
- Modify: `tests/test_compressor.py:43-58`
- Modify: `tests/test_build_embedding.py`

- [ ] **Step 1: Update compressor default output_dim**

In `src/features/embedding_compressor.py`, change the default:

```python
class EmbeddingCompressor:
    """PCA-based dimensionality reduction for sentence embeddings (768 -> 20)."""

    def __init__(self, input_dim: int = 768, output_dim: int = 20):
```

- [ ] **Step 2: Update COMPRESSED_DIM in build_embedding_features**

In `src/features/build_embedding_features.py`:

```python
COMPRESSED_DIM = 20   # reduced from 64 to prevent overfit
TOP_PCA_LAGS = 3      # lag top-3 PCA components
```

- [ ] **Step 3: Update FEATURE_COUNTS in train.py**

In `src/agents/train.py`, update the counts to reflect 20 emb dims:

```python
FEATURE_COUNTS = {
    "baseline": 18,
    "sentiment": 19,     # baseline + 1 sentiment score
    "embeddings": 51,    # 18 base + 20 emb + news_count(3: raw+lag1+lag2) + roll7 + 3 sent extremes + 6 PCA lags
    "fusion": 54,        # embeddings(51) + 3 sentiment (raw + lag1 + lag2)
}
```

- [ ] **Step 4: Update tests for new dimensions**

In `tests/test_compressor.py`, update the two tests:

```python
def test_compressor_768_to_20():
    """EmbeddingCompressor works with 768d input and 20d output (new defaults)."""
    np.random.seed(42)
    data = np.random.randn(200, 768).astype(np.float32)
    compressor = EmbeddingCompressor(input_dim=768, output_dim=20)
    compressor.fit(data)
    result = compressor.transform(data[:5])
    assert result.shape == (5, 20)
    assert compressor.explained_variance_ratio() > 0


def test_default_dims_are_768_20():
    """Default constructor uses 768->20."""
    compressor = EmbeddingCompressor()
    assert compressor.input_dim == 768
    assert compressor.output_dim == 20
```

In `tests/test_build_embedding.py`, update `_mock_compressor` and assertion tests:

```python
def _mock_compressor():
    compressor = MagicMock()
    compressor.transform.return_value = np.random.randn(1, 20).astype(np.float32)
    return compressor
```

Update `test_embedding_columns_added`:

```python
def test_embedding_columns_added():
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", return_value=[0.5]):
        mock_emb.return_value = np.random.randn(768).astype(np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    emb_cols = [c for c in result.columns if c.startswith("emb_") and "_lag" not in c]
    assert len(emb_cols) == 20
```

Update `test_embedding_columns_are_64d` → rename to `test_embedding_columns_are_20d`:

```python
def test_embedding_columns_are_20d():
    """build_embedding_features produces 20 emb columns."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", return_value=[0.5]):
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    emb_cols = [c for c in result.columns if c.startswith("emb_") and "_lag" not in c]
    assert len(emb_cols) == 20
```

Update `test_fallback_zeros_for_no_news_days` to use 20-dim emb_cols check (same pattern, just `== 20`).

- [ ] **Step 5: Run tests to verify**

Run: `pytest tests/test_compressor.py tests/test_build_embedding.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/features/embedding_compressor.py src/features/build_embedding_features.py src/agents/train.py tests/test_compressor.py tests/test_build_embedding.py
git commit -m "feat: reduce PCA dim from 64 to 20 to prevent overfit"
```

---

### Task 2: Cosine similarity dedup in embedding pipeline

**Files:**
- Modify: `src/features/build_embedding_features.py`
- Modify: `tests/test_build_embedding.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build_embedding.py`:

```python
from src.features.build_embedding_features import _deduplicate_embeddings


def test_cosine_dedup_removes_similar():
    """Embeddings with cosine sim > 0.85 are deduplicated."""
    # Two nearly identical embeddings + one different
    base = np.random.randn(768).astype(np.float32)
    similar = base + np.random.randn(768).astype(np.float32) * 0.01  # very similar
    different = np.random.randn(768).astype(np.float32)  # different
    embeddings = np.stack([base, similar, different])
    texts = ["text A", "text A repost", "text B"]
    scores = [0.5, 0.4, -0.3]

    deduped_texts, deduped_scores, deduped_embs = _deduplicate_embeddings(
        texts, scores, embeddings, threshold=0.85
    )
    assert len(deduped_texts) == 2  # similar one removed
    assert len(deduped_scores) == 2
    assert deduped_embs.shape[0] == 2


def test_cosine_dedup_keeps_all_when_different():
    """No dedup when all embeddings are different."""
    embeddings = np.random.randn(3, 768).astype(np.float32)
    texts = ["a", "b", "c"]
    scores = [0.1, 0.2, 0.3]
    deduped_texts, deduped_scores, deduped_embs = _deduplicate_embeddings(
        texts, scores, embeddings, threshold=0.85
    )
    assert len(deduped_texts) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_embedding.py::test_cosine_dedup_removes_similar -v`
Expected: FAIL with ImportError (`_deduplicate_embeddings` not found)

- [ ] **Step 3: Implement _deduplicate_embeddings**

Add to `src/features/build_embedding_features.py` after imports:

```python
from sklearn.metrics.pairwise import cosine_similarity


def _deduplicate_embeddings(
    texts: list[str],
    scores: list[float],
    embeddings: np.ndarray,
    threshold: float = 0.85,
) -> tuple[list[str], list[float], np.ndarray]:
    """Remove articles with cosine similarity > threshold.

    Keeps the first occurrence. Returns filtered texts, scores, embeddings.
    """
    if len(texts) <= 1:
        return texts, scores, embeddings

    sim_matrix = cosine_similarity(embeddings)
    keep = []
    for i in range(len(texts)):
        is_dup = False
        for j in keep:
            if sim_matrix[i, j] > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)

    return (
        [texts[i] for i in keep],
        [scores[i] for i in keep],
        embeddings[keep],
    )
```

- [ ] **Step 4: Integrate dedup into build_embedding_features**

In `build_embedding_features()`, replace the current per-day block (lines 51-71) with:

```python
    for _, row in news_by_day.iterrows():
        day = row["date"].normalize()
        texts = row["texts"]
        if day not in result.index:
            continue

        n = len(texts)
        result.loc[day, "news_count"] = n

        # Compute per-article sentiment scores
        sentiment_scores = compute_sentiment_scores(texts)
        result.loc[day, "sentiment_max"] = max(sentiment_scores)
        result.loc[day, "sentiment_min"] = min(sentiment_scores)
        result.loc[day, "sentiment_spread"] = max(sentiment_scores) - min(sentiment_scores)

        # Compute raw embeddings for dedup
        from src.features.embeddings import _get_model
        st_model = _get_model()
        raw_embs = st_model.encode(texts, show_progress_bar=False)

        # Deduplicate similar articles
        texts, sentiment_scores, raw_embs = _deduplicate_embeddings(
            texts, sentiment_scores, raw_embs, threshold=0.85
        )

        # Weighted pooling: |sentiment| + temporal decay
        weights = [abs(s) + 0.1 for s in sentiment_scores]

        # Mean pooling of deduplicated embeddings with weights
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()
        mean_emb = np.average(raw_embs, axis=0, weights=w).astype(np.float32)

        compressed = compressor.transform(mean_emb.reshape(1, -1))[0]
        result.loc[day, emb_cols] = compressed
```

Note: this replaces the call to `compute_embeddings()` with inline pooling since we need the raw per-article embeddings for dedup. Update the import — remove `compute_embeddings` from imports if no longer used elsewhere in this file.

- [ ] **Step 5: Run tests to verify**

Run: `pytest tests/test_build_embedding.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/features/build_embedding_features.py tests/test_build_embedding.py
git commit -m "feat: add cosine similarity dedup for news embeddings"
```

---

### Task 3: Temporal decay weighting

**Files:**
- Modify: `src/features/build_embedding_features.py`
- Modify: `tests/test_build_embedding.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build_embedding.py`:

```python
from src.features.build_embedding_features import _temporal_weights


def test_temporal_weights_recent_higher():
    """More recent articles (closer to midnight) get higher weight."""
    # Timestamps for a day: 6am, noon, 11pm
    timestamps = pd.to_datetime([
        "2024-01-05 06:00:00",
        "2024-01-05 12:00:00",
        "2024-01-05 23:00:00",
    ], utc=True)
    day_close = pd.Timestamp("2024-01-05 23:59:59", tz="UTC")
    weights = _temporal_weights(timestamps, day_close, alpha=0.1)
    assert len(weights) == 3
    assert weights[2] > weights[1] > weights[0]  # most recent is heaviest


def test_temporal_weights_single_article():
    """Single article gets weight 1.0."""
    timestamps = pd.to_datetime(["2024-01-05 12:00:00"], utc=True)
    day_close = pd.Timestamp("2024-01-05 23:59:59", tz="UTC")
    weights = _temporal_weights(timestamps, day_close, alpha=0.1)
    assert len(weights) == 1
    assert abs(weights[0] - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_embedding.py::test_temporal_weights_recent_higher -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement _temporal_weights**

Add to `src/features/build_embedding_features.py`:

```python
def _temporal_weights(
    timestamps: pd.DatetimeIndex,
    day_close: pd.Timestamp,
    alpha: float = 0.1,
) -> list[float]:
    """Compute exponential decay weights based on time distance to day close.

    weight_i = exp(-alpha * hours_since_close).
    Normalized to sum=1.

    Args:
        timestamps: Article publication timestamps.
        day_close: End-of-day timestamp.
        alpha: Decay rate (default 0.1).

    Returns:
        List of normalized weights.
    """
    hours = [(day_close - ts).total_seconds() / 3600.0 for ts in timestamps]
    raw = [np.exp(-alpha * max(h, 0.0)) for h in hours]
    total = sum(raw)
    if total < 1e-8:
        return [1.0 / len(raw)] * len(raw)
    return [w / total for w in raw]
```

- [ ] **Step 4: Integrate temporal weights into build_embedding_features**

In the per-day loop, combine sentiment weight with temporal weight. Replace the weights line:

```python
        # Weighted pooling: |sentiment| * temporal_decay
        if "timestamps" in row and row["timestamps"] is not None:
            day_close = day + pd.Timedelta(hours=23, minutes=59, seconds=59)
            t_weights = _temporal_weights(
                pd.to_datetime(row["timestamps"], utc=True), day_close
            )
            weights = [(abs(s) + 0.1) * tw for s, tw in zip(sentiment_scores, t_weights)]
        else:
            weights = [abs(s) + 0.1 for s in sentiment_scores]
```

Note: this is backwards-compatible — if `news_by_day` doesn't have a `timestamps` column (old pipeline), it falls back to sentiment-only weights.

- [ ] **Step 5: Run tests to verify**

Run: `pytest tests/test_build_embedding.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/features/build_embedding_features.py tests/test_build_embedding.py
git commit -m "feat: add temporal decay weighting for news embeddings"
```

---

### Task 4: DSR reward in TradingEnv

**Files:**
- Modify: `src/env/trading_env.py`
- Modify: `tests/test_trading_env.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trading_env.py`:

```python
def test_reward_dsr_type_accepted():
    """TradingEnv accepts reward_type='dsr'."""
    features = make_dummy_features(n=40, n_features=5)
    prices = np.ones(40) * 100.0
    prices[31:] = 110.0  # price jump
    env = TradingEnv(features=features, prices=prices, window=30, reward_type="dsr")
    obs, _ = env.reset()
    _, reward, _, _, _ = env.step(np.array([0.5]))
    assert isinstance(reward, float)


def test_reward_dsr_penalizes_variance():
    """DSR reward should be lower for high-variance sequences than basic reward."""
    features = make_dummy_features(n=40, n_features=5)
    # Alternating up/down prices: high variance
    prices = np.array([100.0] * 30 + [110, 95, 115, 90, 120, 85, 125, 80, 130, 75])
    env_basic = TradingEnv(features=features, prices=prices, window=30, reward_type="basic")
    env_dsr = TradingEnv(features=features, prices=prices, window=30, reward_type="dsr")

    env_basic.reset()
    env_dsr.reset()

    total_basic = 0.0
    total_dsr = 0.0
    for _ in range(9):
        _, r_basic, _, _, _ = env_basic.step(np.array([1.0]))
        _, r_dsr, _, _, _ = env_dsr.step(np.array([1.0]))
        total_basic += r_basic
        total_dsr += r_dsr

    # DSR penalizes variance, so sum should differ from basic
    assert total_dsr != total_basic
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trading_env.py::test_reward_dsr_type_accepted -v`
Expected: FAIL with AssertionError (reward_type 'dsr' not in accepted values)

- [ ] **Step 3: Implement DSR reward**

In `src/env/trading_env.py`, update the assertion and add DSR state:

```python
    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window: int = 30,
        tx_cost: float = 0.001,
        reward_type: str = "basic",
        allow_short: bool = False,
        volatility_penalty: float = 0.5,
        sentiment_signal: np.ndarray | None = None,
        sentiment_lambda: float = 0.1,
    ):
        super().__init__()
        assert len(features) == len(prices), "features and prices must have same length"
        assert len(features) > window, "Need more data rows than window size"
        assert reward_type in ("basic", "risk_adjusted", "dsr"), (
            f"reward_type must be 'basic', 'risk_adjusted', or 'dsr', got '{reward_type}'"
        )

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window = window
        self.tx_cost = tx_cost
        self.reward_type = reward_type
        self.allow_short = allow_short
        self.volatility_penalty = volatility_penalty
        self.sentiment_signal = sentiment_signal
        self.sentiment_lambda = sentiment_lambda

        n_features = features.shape[1]
        # +1 for prev_allocation appended to obs
        obs_dim = window * n_features + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        action_low = -1.0 if allow_short else 0.0
        self.action_space = gym.spaces.Box(
            low=action_low, high=1.0, shape=(1,), dtype=np.float32
        )

        self.current_step: int = 0
        self.prev_allocation: float = 0.0
        # DSR running statistics
        self._dsr_A: float = 0.0  # EMA of returns
        self._dsr_B: float = 0.0  # EMA of squared returns
        self._dsr_eta: float = 0.01
```

Update `reset()`:

```python
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.prev_allocation = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        return self._get_obs(), {}
```

Update `step()` — replace the reward computation block:

```python
    def step(self, action):
        action_low = self.action_space.low[0]
        allocation = float(np.clip(action[0], action_low, 1.0))

        price_current = self.prices[self.current_step - 1]
        price_next = self.prices[self.current_step]
        log_return = float(np.log(price_next / price_current))

        delta = abs(allocation - self.prev_allocation)
        tx_penalty = self.tx_cost * delta

        R_t = log_return * allocation - tx_penalty

        if self.reward_type == "dsr":
            reward = self._compute_dsr(R_t)
        elif self.reward_type == "risk_adjusted":
            reward = float(R_t - self.volatility_penalty * delta)
        else:
            reward = float(R_t)

        # SAPPO-lite: sentiment reward modifier
        if self.sentiment_signal is not None and self.reward_type == "dsr":
            sent = float(self.sentiment_signal[self.current_step])
            reward += self.sentiment_lambda * sent * log_return

        self.prev_allocation = allocation
        self.current_step += 1

        terminated = self.current_step >= len(self.prices) - 1
        obs = self._get_obs()
        info = {"log_return": log_return, "allocation": allocation}
        return obs, reward, terminated, False, info
```

Add the DSR helper method:

```python
    def _compute_dsr(self, R_t: float) -> float:
        """Compute Differential Sharpe Ratio increment."""
        eta = self._dsr_eta
        dA = R_t - self._dsr_A
        dB = R_t ** 2 - self._dsr_B

        denominator = self._dsr_B - self._dsr_A ** 2
        if denominator < 1e-12:
            # Not enough variance accumulated yet — fallback to raw return
            reward = float(R_t)
        else:
            reward = float(
                (self._dsr_B * dA - 0.5 * self._dsr_A * dB)
                / (denominator ** 1.5)
            )

        # Update running stats
        self._dsr_A += eta * dA
        self._dsr_B += eta * dB
        return reward
```

Update `_get_obs()` to include prev_allocation:

```python
    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window
        end = self.current_step
        window_obs = self.features[start:end].flatten()
        return np.append(window_obs, self.prev_allocation).astype(np.float32)
```

- [ ] **Step 4: Update existing tests for new obs shape**

All existing tests use `make_dummy_features(n=100, n_features=20)` and check `obs.shape == (30 * 20,)`. Update these to `(30 * 20 + 1,)`:

In `test_env_reset`:
```python
    assert obs.shape == (30 * 20 + 1,)  # +1 for prev_allocation
```

In `test_env_step`:
```python
    assert obs.shape == (30 * 20 + 1,)  # +1 for prev_allocation
```

- [ ] **Step 5: Run all env tests**

Run: `pytest tests/test_trading_env.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/env/trading_env.py tests/test_trading_env.py
git commit -m "feat: add DSR reward, prev_allocation in obs, sentiment modifier"
```

---

### Task 5: AgentConfig updates

**Files:**
- Modify: `src/agents/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_config_new_fields_defaults():
    """New fields ent_coef and sentiment_lambda have correct defaults."""
    config = AgentConfig()
    assert config.ent_coef == 0.0
    assert config.sentiment_lambda == 0.1


def test_config_embeddings_overrides():
    """Embeddings agent can use custom ent_coef."""
    config = AgentConfig(agent_type="embeddings", ent_coef=0.01, gamma=0.95)
    assert config.ent_coef == 0.01
    assert config.gamma == 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_config_new_fields_defaults -v`
Expected: FAIL with AttributeError

- [ ] **Step 3: Add fields to AgentConfig**

In `src/agents/config.py`, add after `clip_range`:

```python
    ent_coef: float = 0.0
    sentiment_lambda: float = 0.1
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/config.py tests/test_config.py
git commit -m "feat: add ent_coef and sentiment_lambda to AgentConfig"
```

---

### Task 6: VecNormalize in training pipeline

**Files:**
- Modify: `src/agents/train.py`
- Modify: `src/eval/backtest.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py`:

```python
def test_train_saves_vecnormalize():
    """Training saves VecNormalize stats alongside model."""
    from pathlib import Path
    config = AgentConfig(total_timesteps=256, seed=42, algorithm="PPO")
    model_path = train_agent(config, dummy=True)
    vecnorm_path = model_path.parent / "vecnormalize.pkl"
    assert vecnorm_path.exists(), "VecNormalize stats not saved"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py::test_train_saves_vecnormalize -v`
Expected: FAIL (vecnormalize.pkl not found)

- [ ] **Step 3: Implement VecNormalize in train_agent**

In `src/agents/train.py`, add import:

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
```

Update `train_agent()` — wrap env after creation (replace the bare `env` with vec env):

```python
    # Wrap in VecNormalize
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)
```

Use `vec_env` instead of `env` when creating the model:

```python
    model = algo_cls(
        "MlpPolicy",
        vec_env,
        ...
    )
```

After `model.save(...)`, save VecNormalize stats:

```python
    vecnorm_path = save_dir / "vecnormalize.pkl"
    vec_env.save(str(vecnorm_path))
    logger.info(f"Saved VecNormalize stats to {vecnorm_path}")
```

- [ ] **Step 4: Update backtest to load VecNormalize**

In `src/eval/backtest.py`, update `run_backtest` signature and body:

```python
def run_backtest(
    features: np.ndarray,
    prices: np.ndarray,
    model_path: Optional[str],
    window: int = 30,
    tx_cost: float = 0.001,
    plot_path: Optional[str] = None,
    allow_short: bool = False,
    vecnorm_path: Optional[str] = None,
) -> dict:
```

After creating `env`, wrap if vecnorm stats exist:

```python
    if vecnorm_path is not None and Path(vecnorm_path).exists():
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        model = None
        if model_path is not None:
            model = load_model(model_path)

        obs = vec_env.reset()
        daily_returns = []
        allocations = []
        done = False

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = vec_env.action_space.sample()

            obs, reward, done_arr, infos = vec_env.step(action)
            done = done_arr[0]
            info = infos[0]

            allocation = float(info.get("allocation", 0.0))
            log_ret = float(info.get("log_return", 0.0))
            daily_returns.append(np.exp(log_ret * allocation) - 1)
            allocations.append(allocation)
    else:
        # Original non-VecNormalize path (unchanged)
        model = None
        if model_path is not None:
            model = load_model(model_path)

        obs, _ = env.reset()
        daily_returns = []
        allocations = []
        done = False

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            allocation = float(info.get("allocation", 0.0))
            log_ret = float(info.get("log_return", 0.0))
            daily_returns.append(np.exp(log_ret * allocation) - 1)
            allocations.append(allocation)
```

- [ ] **Step 5: Update run_all.py to pass vecnorm_path**

In `experiments/run_all.py`, in the backtest call, compute and pass the vecnorm path:

```python
            vecnorm_path = str(Path(model_path).parent / "vecnormalize.pkl")

            backtest = run_backtest(
                features=test_features,
                prices=test_prices,
                model_path=str(model_path),
                window=config.window,
                tx_cost=config.tx_cost,
                vecnorm_path=vecnorm_path,
            )
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_train.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/agents/train.py src/eval/backtest.py experiments/run_all.py tests/test_train.py
git commit -m "feat: add VecNormalize wrapper to training and backtest"
```

---

### Task 7: Pass sentiment_signal to TradingEnv

**Files:**
- Modify: `src/data/load_features.py`
- Modify: `src/agents/train.py`
- Modify: `experiments/run_all.py`

- [ ] **Step 1: Add sentiment extraction to load_features**

In `src/data/load_features.py`, update the function to optionally return the mean sentiment column:

```python
def load_features_for_agent(
    agent_type: str,
    asset: str,
    train_start: str,
    train_end: str,
    data_dir: str = "data/processed",
    timeframe: str = "1d",
    return_sentiment: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
```

Before the return, add:

```python
    if return_sentiment:
        # Extract sentiment signal for SAPPO-lite reward modifier
        # Use sentiment_max as proxy if available, else zeros
        if "sentiment_max" in df.columns:
            sentiment = df["sentiment_max"].to_numpy(dtype=np.float32)
        else:
            sentiment = np.zeros(len(df), dtype=np.float32)
        sentiment = np.nan_to_num(sentiment, nan=0.0)
        return features, prices, sentiment

    return features, prices
```

- [ ] **Step 2: Update train_agent to pass sentiment**

In `src/agents/train.py`, when `agent_type in ("embeddings", "fusion")` and not dummy, load sentiment:

```python
    if not dummy and config.agent_type in ("embeddings", "fusion"):
        features, prices, sentiment = load_features_for_agent(
            agent_type=config.agent_type,
            asset=asset,
            train_start=train_start,
            train_end=train_end,
            data_dir=data_dir,
            timeframe=timeframe,
            return_sentiment=True,
        )
    elif not dummy:
        features, prices = load_features_for_agent(
            agent_type=config.agent_type,
            asset=asset,
            train_start=train_start,
            train_end=train_end,
            data_dir=data_dir,
            timeframe=timeframe,
        )
        sentiment = None
    else:
        sentiment = None
```

Pass `sentiment_signal` and `sentiment_lambda` to `TradingEnv`:

```python
    env = TradingEnv(
        features=features,
        prices=prices,
        window=config.window,
        tx_cost=config.tx_cost,
        reward_type=config.reward_type,
        allow_short=config.allow_short,
        sentiment_signal=sentiment if config.agent_type in ("embeddings", "fusion") else None,
        sentiment_lambda=config.sentiment_lambda,
    )
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_train.py tests/test_load_features.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/data/load_features.py src/agents/train.py
git commit -m "feat: pass sentiment signal to TradingEnv for SAPPO-lite"
```

---

### Task 8: Hyperparameter overrides for embeddings agent

**Files:**
- Modify: `src/agents/train.py`
- Modify: `experiments/run_all.py`

- [ ] **Step 1: Update train.py with embeddings-specific hyperparams**

In `src/agents/train.py`, update `EMBEDDINGS_NET_ARCH`:

```python
EMBEDDINGS_NET_ARCH = [128, 64]
```

In `_algo_specific_kwargs`, pass `ent_coef`:

```python
def _algo_specific_kwargs(config: AgentConfig) -> dict:
    if config.algorithm == "PPO":
        return {
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_range": config.clip_range,
            "ent_coef": config.ent_coef,
        }
    elif config.algorithm == "A2C":
        return {
            "n_steps": config.n_steps,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "ent_coef": config.ent_coef,
        }
    elif config.algorithm == "SAC":
        return {
            "gamma": config.gamma,
            "batch_size": config.batch_size,
            "buffer_size": 100_000,
            "ent_coef": "auto",  # SAC manages entropy automatically
        }
    return {}
```

- [ ] **Step 2: Update run_all.py with embeddings-specific config**

In `experiments/run_all.py`, in the per-run config creation, add overrides for embeddings:

```python
            config = AgentConfig(
                agent_type=agent_type,
                algorithm=algorithm,
                total_timesteps=total_timesteps,
                seed=seed,
                save_dir=save_dir,
                lr_schedule="linear",
                reward_type=reward_type,
            )

            # Embeddings-specific hyperparameter overrides
            if agent_type in ("embeddings", "fusion"):
                config.gamma = 0.95
                config.ent_coef = 0.01
                config.clip_range = 0.15
                config.reward_type = "dsr"
```

Update default seeds:

```python
    parser.add_argument(
        "--seeds", type=str, default="42,43,44,45,46",
        help="Comma-separated seeds (default: 42,43,44,45,46)"
    )
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_train.py tests/test_run_ablation.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/agents/train.py experiments/run_all.py
git commit -m "feat: embeddings-specific hyperparams (gamma=0.95, ent_coef=0.01, net=[128,64])"
```

---

### Task 9: Comparison script

**Files:**
- Create: `experiments/compare_results.py`
- Create: `tests/test_compare_results.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare_results.py`:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_results import load_metrics, compute_deltas, run_ttest


def _make_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_load_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "metrics.csv"
        _make_csv(csv_path, [
            {"agent_type": "baseline", "asset": "BTC/USDT", "seed": 42,
             "total_return": 0.1, "sharpe_ratio": 0.5, "sortino_ratio": 0.4,
             "max_drawdown": 0.1, "calmar_ratio": 0.3},
        ])
        df = load_metrics(str(csv_path))
        assert len(df) == 1
        assert df.iloc[0]["sharpe_ratio"] == 0.5


def test_compute_deltas():
    old = pd.DataFrame([
        {"agent_type": "embeddings", "seed": 42, "sharpe_ratio": 0.5, "total_return": 0.1},
        {"agent_type": "embeddings", "seed": 43, "sharpe_ratio": 0.1, "total_return": -0.01},
    ])
    new = pd.DataFrame([
        {"agent_type": "embeddings", "seed": 42, "sharpe_ratio": 0.8, "total_return": 0.2},
        {"agent_type": "embeddings", "seed": 43, "sharpe_ratio": 0.6, "total_return": 0.15},
    ])
    deltas = compute_deltas(old, new, agent_type="embeddings")
    assert len(deltas) == 2
    assert abs(deltas.iloc[0]["sharpe_delta"] - 0.3) < 1e-6


def test_run_ttest():
    old_sharpes = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
    new_sharpes = np.array([0.8, 0.9, 0.85, 0.82, 0.88])
    t_stat, p_value = run_ttest(old_sharpes, new_sharpes)
    assert p_value < 0.05  # clearly different distributions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare_results.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement compare_results.py**

Create `experiments/compare_results.py`:

```python
"""Compare old vs new experiment results with statistical tests.

Usage:
    python experiments/compare_results.py --old results/metrics_v1.csv --new results/metrics.csv
"""
import argparse
import logging

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_metrics(path: str) -> pd.DataFrame:
    """Load metrics CSV."""
    return pd.read_csv(path)


def compute_deltas(
    old: pd.DataFrame,
    new: pd.DataFrame,
    agent_type: str = "embeddings",
) -> pd.DataFrame:
    """Compute per-seed deltas between old and new metrics."""
    old_agent = old[old["agent_type"] == agent_type].sort_values("seed").reset_index(drop=True)
    new_agent = new[new["agent_type"] == agent_type].sort_values("seed").reset_index(drop=True)

    merged = old_agent.merge(new_agent, on="seed", suffixes=("_old", "_new"))
    merged["sharpe_delta"] = merged["sharpe_ratio_new"] - merged["sharpe_ratio_old"]
    merged["return_delta"] = merged["total_return_new"] - merged["total_return_old"]
    return merged


def run_ttest(
    old_values: np.ndarray,
    new_values: np.ndarray,
) -> tuple[float, float]:
    """Run paired t-test (if same size) or independent t-test."""
    if len(old_values) == len(new_values):
        t_stat, p_value = stats.ttest_rel(new_values, old_values)
    else:
        t_stat, p_value = stats.ttest_ind(new_values, old_values)
    return float(t_stat), float(p_value)


def plot_comparison(old: pd.DataFrame, new: pd.DataFrame, save_path: str):
    """Bar plot: old vs new Sharpe ratios side by side."""
    fig, ax = plt.subplots(figsize=(10, 6))

    agent_types = sorted(set(old["agent_type"]) | set(new["agent_type"]))
    x = np.arange(len(agent_types))
    width = 0.35

    old_means = []
    old_stds = []
    new_means = []
    new_stds = []

    for at in agent_types:
        old_s = old[old["agent_type"] == at]["sharpe_ratio"]
        new_s = new[new["agent_type"] == at]["sharpe_ratio"]
        old_means.append(old_s.mean() if len(old_s) > 0 else 0)
        old_stds.append(old_s.std() if len(old_s) > 1 else 0)
        new_means.append(new_s.mean() if len(new_s) > 0 else 0)
        new_stds.append(new_s.std() if len(new_s) > 1 else 0)

    ax.bar(x - width / 2, old_means, width, yerr=old_stds, label="Old", alpha=0.8)
    ax.bar(x + width / 2, new_means, width, yerr=new_stds, label="New", alpha=0.8)

    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Old vs New: Sharpe Ratio by Agent Type")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_types)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved comparison plot to {save_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Compare old vs new results")
    parser.add_argument("--old", required=True, help="Path to old metrics CSV")
    parser.add_argument("--new", required=True, help="Path to new metrics CSV")
    parser.add_argument("--plot", default="results/figures/comparison.png")
    args = parser.parse_args()

    old = load_metrics(args.old)
    new = load_metrics(args.new)

    print("\n" + "=" * 70)
    print("COMPARISON: Old vs New Results")
    print("=" * 70)

    for agent_type in sorted(set(old["agent_type"]) | set(new["agent_type"])):
        old_s = old[old["agent_type"] == agent_type]["sharpe_ratio"].values
        new_s = new[new["agent_type"] == agent_type]["sharpe_ratio"].values

        print(f"\n--- {agent_type} ---")
        if len(old_s) > 0:
            print(f"  Old Sharpe: {old_s.mean():.3f} +/- {old_s.std():.3f} (n={len(old_s)})")
        if len(new_s) > 0:
            print(f"  New Sharpe: {new_s.mean():.3f} +/- {new_s.std():.3f} (n={len(new_s)})")

        if len(old_s) >= 2 and len(new_s) >= 2:
            t_stat, p_val = run_ttest(old_s, new_s)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f} {sig}")

    # Embeddings vs baseline (new)
    emb_new = new[new["agent_type"] == "embeddings"]["sharpe_ratio"].values
    base_new = new[new["agent_type"] == "baseline"]["sharpe_ratio"].values
    if len(emb_new) >= 2 and len(base_new) >= 2:
        t_stat, p_val = run_ttest(base_new, emb_new)
        print(f"\n--- embeddings vs baseline (new) ---")
        print(f"  t={t_stat:.3f}, p={p_val:.4f}")

    # Deltas
    deltas = compute_deltas(old, new, "embeddings")
    if len(deltas) > 0:
        print(f"\n--- Per-seed deltas (embeddings) ---")
        for _, row in deltas.iterrows():
            print(f"  seed={int(row['seed'])}: Sharpe {row.get('sharpe_ratio_old', 0):.3f} -> {row.get('sharpe_ratio_new', 0):.3f} (delta={row['sharpe_delta']:+.3f})")

    plot_comparison(old, new, args.plot)
    print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_compare_results.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/compare_results.py tests/test_compare_results.py
git commit -m "feat: add compare_results.py for old vs new metrics analysis"
```

---

### Task 10: Save old metrics and run experiments

**Files:**
- Modify: `results/metrics.csv`

- [ ] **Step 1: Backup old metrics**

```bash
cp results/metrics.csv results/metrics_v1.csv
```

- [ ] **Step 2: Rebuild embedding features with new PCA dim**

Note: this requires running the embedding pipeline with the new PCA dim=20. The compressor must be re-fitted.

```bash
python -m src.features.build_embedding_features --asset BTC/USDT
```

- [ ] **Step 3: Run experiments (embeddings only first for quick validation)**

```bash
python experiments/run_all.py \
    --agent-types embeddings \
    --seeds 42,43,44,45,46 \
    --total-timesteps 1000000 \
    --reward-type dsr \
    --asset BTC/USDT
```

- [ ] **Step 4: Run baseline for fair comparison (same 5 seeds)**

```bash
python experiments/run_all.py \
    --agent-types baseline \
    --seeds 42,43,44,45,46 \
    --total-timesteps 500000 \
    --asset BTC/USDT
```

- [ ] **Step 5: Compare results**

```bash
python experiments/compare_results.py --old results/metrics_v1.csv --new results/metrics.csv
```

- [ ] **Step 6: Commit results**

```bash
git add results/metrics.csv results/metrics_v1.csv results/figures/comparison.png
git commit -m "results: embedding improvements v2 vs v1 comparison"
```
