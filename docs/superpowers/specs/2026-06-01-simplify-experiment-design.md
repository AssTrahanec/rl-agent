# Simplify Experiment — Design Spec

**Date:** 2026-06-01
**Status:** approved (design), pending implementation plan
**Supersedes:** [`docs/superpowers/plans/2026-06-01-simplify-code.md`](../plans/2026-06-01-simplify-code.md) —
that earlier plan was **code-only** and explicitly *did not* touch features or the
dashboard. This spec overrides that boundary: we **do** reduce the feature set
(and retrain) and **do** simplify the dashboard. The earlier plan's code-cleanup
tasks are absorbed here as **Phase 2**.

---

## 1. Goal

Make `dsr_experiment/` small enough and clean enough that the thesis advisor (and
the defense committee) can read it in ~30 minutes without asking "а это зачем?".
Two levers:

1. **Cut the feature set to a literature-grounded minimum** and retrain, so the
   observation is small and every feature is explainable.
2. **Simplify the code** (kill dead branches, never-varied knobs, drift) and the
   **dashboard** (keep all functionality, shrink and de-duplicate the code).

This is driven by an exa literature review (FinRL, SAPPO, SSAI, LSEG-PCA): the
current **100-feature × 30-window = 3001-dim** observation is excessive by every
comparable benchmark. FinRL single-asset states use ~4–10 indicators; the closest
work (SAPPO, sentiment-augmented PPO on SB3) uses a *single scalar* sentiment, not
a dense embedding; SSAI shows news has "effective dimensionality ≈ 1"; the LSEG
study shows accuracy *drops* when too many PCA components are added (noise/overfit).
DQN-discrete for a single crypto is, by contrast, *well* justified (FinRL-Meta:
small dataset → DQN trains faster, overfits less) — so the algorithm choice stays.

---

## 2. Decisions (locked with user)

| # | Decision | Value |
|---|---|---|
| D1 | Touch features? | **Yes — cut to minimum and retrain** |
| D2 | Embedding size | **32** PCA dims (768 → 32) |
| D3 | Retrain scope | **DQN only, 5 seeds** (fast "does it still hold?" read) |
| D4 | Sentiment | **1 scalar** (`sentiment_mean`) |
| D5 | Lags / rolling / news_count | **Drop all** (window=30 already carries temporal context) |
| D6 | Window | **Keep 30** (change exactly one factor → clean attribution) |
| D7 | Metrics set | **Keep all** (total/annualized return, Sharpe, Sortino, MaxDD, Calmar, win_rate, profit_factor, time_in_market) — user wants win_rate & co. kept. Only a light readability pass, **no metric is dropped**. |
| D8 | Bootstrap | Keep as-is (already 37 lines); only re-point to the kept metrics |
| D9 | Dashboard | Keep all 3 pages' **functionality**; simplify the **code** (remove fragile/duplicate bits), re-align to the new feature schema |
| D10 | SAC | **Not** retrained — leave on old features (out of scope for this pass) |
| D11 | Sharpe annualization | **Fix** to √2190 (per-4h-bar returns, 24/7 crypto). Unify the two inconsistent annualization conventions in `metrics.py` onto one (2190). Keeps all metric outputs; corrects a ~2.45× under-statement. |

---

## 3. The minimal feature set: 100 → 41

**Key enabler:** the minimal pipeline already exists as dormant functions — it was
written earlier as documentation and never wired in. We *activate* it (+ set
`compressed_dim: 32`), then delete the bloated full versions.

| Group | Now | After | Source after |
|---|---|---|---|
| Technical indicators | 20 | **8** | [`add_technical_indicators_minimal`](../../../dsr_experiment/lib/features/price.py) (price.py:49) — `ema_26, macd, rsi_14, bb_width, atr_14, obv, stoch_k, return_1d` |
| News embeddings | 64 | **32** | PCA 768→32 (`embeddings.compressed_dim: 32`) |
| Sentiment | 5 aggregates | **1** (`sentiment_mean`) | [`_attach_nlp_features_minimal`](../../../dsr_experiment/build_data.py) (build_data.py:145) |
| news_count + lags + rolling | 11 | **0** | dropped |
| **Total** | **100** | **41** | observation = 30×41 + 1 = **1231** (was 3001, −59%) |

Notes:
- [`data_loader.py:71`](../../../dsr_experiment/lib/data_loader.py) already reads
  `sentiment_mean` as the reward signal, so the DSR sentiment-bonus needs **no
  change** — switching to a single `sentiment_mean` is already consistent.
- `_NEWS_COLUMN_PREFIXES = ("sentiment_", "news_count", "emb_")` still matches the
  minimal schema (`sentiment_mean`, `emb_*`) — the `--no-news` ablation keeps working.
- The minimal functions are **untested** → Phase 1 adds a schema test (asserts the
  built train parquet has exactly the 41 expected feature columns).

---

## 4. Metrics & bootstrap (keep all metrics; fix one correctness bug)

This came out of the exa research pass (QuantStart, ml4trading.io, Lo 2002).

- **P1 — annualization fix.** `run_backtest` produces **per-4h-bar** returns, but
  `lib/metrics.py` annualizes Sharpe/Sortino with `√365` while annualizing return
  with `√2190` in the *same* file — internally inconsistent, and the Sharpe is
  **~2.45× too low** (√(2190/365)=√6). Fix: **unify on the 2190 convention** (24/7
  crypto = 6 four-hour bars/day × 365):
  - `sharpe`, `sortino`: `× √2190` (was √365).
  - `annual_return` (used by Calmar): exponent `2190/n` (was `365/n`).
  - `calmar = annualized_return / max_dd`, reusing the single 2190-based annual
    return — **delete the duplicate 365-based `annual_return`** and the
    `TRADING_DAYS_PER_YEAR` constant. One annualization constant remains.
  - **All metric outputs kept** (total/annualized return, Sharpe, Sortino, MaxDD,
    Calmar, win_rate, profit_factor, time_in_market). This is correctness +
    simplification, not a cut.
  - Effect: absolute Sharpe/Sortino rise ~2.45× (DQN 2025 ~0.79→~1.93, BH
    ~0.50→~1.22). **Relative ranking, t-tests, bootstrap CIs are unchanged** (same
    factor on every strategy). All slide/thesis Sharpe figures must be updated to
    the new convention (Phase 5).
- **P2 — rename.** In `lib/backtest.py`, `daily_returns` → `step_returns` (the series
  is per-4h-bar, not daily) — removes a "why daily?" confusion.
- **P3 — no-leakage note.** The rolling z-score is strictly trailing (causal, window
  [t−29, t]) and the PCA compressor is fit **train-only** → no look-ahead. This is
  *already correct*; we just state it explicitly in `code_walkthrough.md` to preempt
  the #1 reviewer question (forward-contaminated normalization is the top leakage
  channel in the literature).
- `lib/bootstrap.py`: unchanged in logic (Sharpe + total_return CI across seeds);
  benefits automatically from the corrected Sharpe.

---

## 5. File-by-file change map

### Phase 1 — Minimal features + retrain (the risk gate)
| File | Change |
|---|---|
| `dsr_experiment/config.yaml` | `embeddings.compressed_dim: 64 → 32`. Mark `top_pca_lags`, `features.news_count_lags`, `features.news_count_roll` as now-unused (removed in Phase 2 config cleanup). |
| `dsr_experiment/build_data.py` | `_build_baseline` → call `add_technical_indicators_minimal`. `build_train`/`build_oos` → call `_attach_nlp_features_minimal`. Delete the full `_attach_nlp_features` (build_data is its only consumer — safe) and its lag/rolling imports. |
| `dsr_experiment/lib/features/price.py` | Stop using full `add_technical_indicators` in build_data. **Defer deleting it to Phase 4**, after the dashboard's `features_live.py` migrates to the minimal indicators (grep confirms the dashboard is the only other consumer) — avoids a broken intermediate dashboard. |
| `tests/` | New test: minimal train build → 41 feature columns, expected names. |
| (run) | `build_data.py --build all` (rebuild train+OOS), then `run.py --algo DQN` with `seeds:[5]`. |

### Phase 2 — Code cleanup (absorbed from old plan)
| File | Change |
|---|---|
| `lib/env.py` | Remove `reward_type` basic/risk_adjusted branches, `allow_short`, `volatility_penalty`, continuous-action path, `dsr_eta` kwarg (hardcode constant). Keep: DSR + sentiment bonus + tx_cost, Discrete(3). |
| `lib/train.py` | Drop PPO (import, ALGO_MAP, `_algo_kwargs`, dispatch). Drop dead env kwargs in `_build_env`. |
| `run.py` | Default algos `["DQN"]`; remove PPO from choices/defaults. |
| `lib/features/embeddings.py`, `lib/features/news.py` | Hardcode never-varied knobs (model name, text/date cols). |
| `lib/interpretability.py`, `scripts/run_permutation.py` | Drop unused `action_feature_correlation`, `action_distribution_by_sentiment_regime`. |
| `config.yaml`, `lib/config_loader.py` | Remove `agent_ppo`, `env.reward_type/allow_short/volatility_penalty/dsr_eta`, `agent_sac.use_sde`, `news.text_col/date_col`, `embeddings.top_pca_lags`, `features.news_count_lags/news_count_roll`; matching dataclass cleanup. |

### Phase 3 — Metrics correctness + readability (no metric dropped)
| File | Change |
|---|---|
| `lib/metrics.py` | **Fix annualization (P1):** unify on √2190 — `sharpe`/`sortino` ×√2190; `annual_return` exponent 2190/n; `calmar = annualized_return/max_dd`; delete the duplicate 365-based annual_return and `TRADING_DAYS_PER_YEAR`. Keep all metric outputs. |
| `lib/backtest.py` | **Rename (P2):** `daily_returns` → `step_returns`. (Also loses `allow_short`/`action_space_type` defaults — see Phase 2.) |
| `lib/bootstrap.py` | No logic change; verify it still reads `sharpe_ratio` + `total_return`. |
| `tests/test_dsr_metrics.py` (or equiv.) | Update expected Sharpe/Sortino to √2190 scaling; add an assertion that a known per-bar series annualizes with √2190. |

### Phase 4 — Dashboard (functionality kept, code simplified)
| File | Change |
|---|---|
| `dashboard/utils/features_live.py` | Re-align to new 41-feature schema: `EMB_DIM 64→32`, drop lag/rolling/sentiment-aggregate construction, switch to the minimal indicators. **Net simplification.** Then delete the now-orphaned full `add_technical_indicators` from `lib/features/price.py`. |
| `dashboard/utils/feed_decisions.py`, `news_impact.py` | Consolidate the duplicated ensemble/vote logic into one helper; update to new schema. |
| `dashboard/utils/news_feed.py` | Remove the rarely-used RSS fallback (~80 lines); keep NewsAPI + cached parquet. |
| `dashboard/utils/paths.py` / `model_catalog.py` | Point `DEFAULT_SNAPSHOT` at the new retrained DQN model set. |
| `dashboard/pages/2_Validation.py` | Drop P25/P75 ribbon clutter (median only); display the kept metric set. |
| `dashboard/pages/1_Strategy.py`, `3_News_Analyzer.py` | Trim per-decision rendering / vote-count clutter; no functional loss. |

### Phase 5 — Docs + final verification
| File | Change |
|---|---|
| `dsr_experiment/vkr_defense/code_walkthrough.md`, `README.md`, `dashboard/README.md` | Sync to: DQN-only, DSR-only, 41-feature minimal set. Fix the SAC/PPO/DQN drift. **Add the no-leakage note (P3)** (trailing z-score + train-only PCA). **Update every Sharpe/Sortino figure to the √2190 convention** (P1). |
| Defense artifacts (slides/thesis text, `results/figures/`) | Out of code scope, but flag for the user: all Sharpe/Sortino numbers and any regenerated figures must move to the √2190 convention; relative claims/CIs are unchanged. |
| (run) | Full test suite green; smoke-build; dashboard launches against new snapshot. |

---

## 6. Phase plan & the retrain gate

```
Phase 0  Baseline: full test suite green; record CURRENT OOS metrics
         (results/oos_oos_2024.csv, oos_oos_2025.csv) for before/after compare.
Phase 1  🚪 GATE: wire in minimal features → compressed_dim 32 → rebuild data →
         retrain DQN×5 → run OOS 2024 + 2025 → VERIFY (§7).
         If the result does NOT hold → STOP, reconsider feature set,
         touch nothing else.
Phase 2  Code cleanup (PPO, dead env branches, knobs, interpretability, config).
Phase 3  Metrics/bootstrap readability.
Phase 4  Dashboard simplification + re-align to new schema/snapshot.
Phase 5  Docs sync + full verification.
```

Risk is front-loaded: the only uncertain part (does the smaller feature set keep
the result?) is Phase 1 and gates everything after it.

---

## 7. Verification criteria ("does it still hold?")

After the Phase-1 DQN×5 retrain, compute OOS metrics and compare to the defended
baseline (memory `vkr_state.md`):

- **Primary (must hold):** on **OOS 2025**, DQN mean Sharpe is **> the Buy & Hold
  computed in the same run** with a positive margin, and MaxDD stays materially
  below BH's (~30.6%). Compare against the **recomputed** BH, not the historical
  √365-era figure (0.50): after the P1 fix everything is in √2190 units, so the
  comparison is unit-independent (both scale by the same √6). The defended result
  was Sharpe 0.79 ± 0.22, CI [0.66; 0.92] in √365 units (≈1.93 in √2190). We accept
  "holds" if mean Sharpe stays clearly above BH and the drawdown advantage persists
  (5-seed CI will be wider than the 10-seed one).
- **Secondary (report, not gate):** OOS 2024 behaves as before (active strategies
  ≤ BH Sharpe in the bull phase, drawdown controlled) — i.e. the phase-dependence
  story is intact.
- **Decision:** holds → continue to Phase 2. Does not hold → stop; options are
  (a) bump embeddings 32→64-but-drop-lags, (b) re-add news_count, (c) revert to old
  features and keep this a code-only simplification.

If Phase 1 holds, an **optional** follow-up (for the *final* defense numbers, not
part of this gate): scale DQN back to 10 seeds and re-run the `--no-news` ablation
on the minimal features so the "news helps" comparison is consistent.

---

## 8. What we explicitly do NOT touch

- The DSR reward algorithm, window=30, discrete DQN actions, env/backtest structure.
- Trained **SAC** models (stay on old features).
- `data/raw/*` (raw OHLCV + news cache) — only `data/train/` + `data/oos/` are rebuilt.
- The statistical methodology (bootstrap CI, t-test) — kept, only re-pointed.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Smaller features kill the defended result | Phase-1 gate before any other work; explicit fallback options (§7). |
| 5-seed CI too wide to claim significance | Honest framing: 5 seeds = directional check; optional 10-seed final run. |
| Dashboard breaks on new feature schema | features_live.py rewritten to the exact 41-col schema in the same pass; schema test in Phase 1 is the contract. |
| Old `experiments/run_2026-04-21_10seeds` still referenced | Phase 4 re-points `DEFAULT_SNAPSHOT`; old snapshot left intact as fallback. |
| Dead full functions still imported somewhere | Grep before deleting `add_technical_indicators` / `_attach_nlp_features`; remove imports in the same commit. |

---

## 10. Open items

- Exact path/naming for the new retrained DQN snapshot — resolved in the
  implementation plan by inspecting how `run.py` writes models and how the dashboard
  discovers snapshots.
- Whether to keep `config.smoke.yaml` aligned to the minimal pipeline (yes — update
  it so the smoke test exercises the same code path).
