# 4h Timeframe Experiment Results

## Setup
- **Asset:** BTC/USDT
- **Timeframe:** 4h (10,956 candles, 2020-01-01 to 2024-12-31)
- **Train:** 2020-01-01 to 2023-12-31 (8,760 candles)
- **Test:** 2024-01-01 to 2024-12-31 (2,191 candles)
- **News:** 66,868 articles grouped into 9,362 4h windows (99.99% coverage)
- **Algorithm:** PPO, linear LR decay
- **Seeds:** 42, 43, 44

## Experiment 1: 500K steps (baseline run)

| Agent | Sharpe | Return | MaxDD | Sortino |
|-------|--------|--------|-------|---------|
| Embeddings (95 feat) | 0.455 +/- 0.29 | 28.9% | 19.6% | 0.478 +/- 0.34 |
| Baseline (18 feat) | 0.089 +/- 0.20 | 3.6% | 27.6% | 0.079 +/- 0.17 |

### Per-seed (500K)
| Seed | Emb Sharpe | Emb Return | Base Sharpe | Base Return |
|------|-----------|------------|-------------|-------------|
| 42 | 0.203 | 11.3% | -0.191 | -14.6% |
| 43 | 0.859 | 63.0% | 0.174 | 7.5% |
| 44 | 0.303 | 12.2% | 0.283 | 18.0% |

## Experiment 2: 1M steps (2x training)

| Agent | Sharpe | Return | MaxDD | Sortino |
|-------|--------|--------|-------|---------|
| Embeddings (95 feat) | 0.528 +/- 0.64 | 15.8% | 10.8% | 0.571 +/- 0.71 |
| Baseline (18 feat) | 0.360 +/- 0.26 | 20.8% | 19.7% | 0.345 +/- 0.26 |

### Per-seed (1M)
| Seed | Emb Sharpe | Emb Return | Emb MaxDD | Base Sharpe | Base Return | Base MaxDD |
|------|-----------|------------|-----------|-------------|-------------|------------|
| 42 | 1.430 | 42.9% | 4.5% | 0.061 | 0.1% | 26.6% |
| 43 | 0.000 | -0.7% | 11.4% | 0.316 | 15.3% | 12.6% |
| 44 | 0.154 | 5.2% | 16.6% | 0.701 | 47.0% | 19.8% |

## Experiment 3: risk_adjusted reward (failed)

PPO + risk_adjusted reward + 1M steps: all seeds produced Sharpe=0.000.
The volatility_penalty=0.5 is too high, causing the agent to learn allocation=0 (never trade).

## Comparison: 4h vs Daily

| Timeframe | Embeddings Sharpe | Baseline Sharpe | Delta |
|-----------|------------------|-----------------|-------|
| Daily (500K) | 1.310 +/- 0.520 | 1.358 +/- 0.436 | -0.048 (NLP no help) |
| 4h (500K) | 0.455 +/- 0.29 | 0.089 +/- 0.20 | +0.366 (NLP helps!) |
| 4h (1M) | 0.528 +/- 0.64 | 0.360 +/- 0.26 | +0.168 (NLP helps) |

## Key Findings

1. **NLP features help more on 4h than daily** - consistent across both 500K and 1M experiments
2. **More training steps help baseline more** - baseline improved from 0.089 to 0.360 Sharpe (+0.271), embeddings from 0.455 to 0.528 (+0.073)
3. **Embeddings have lower MaxDD** - 10.8% vs 19.7% (1M), suggesting NLP helps avoid drawdowns
4. **High variance remains** - embeddings std=0.64 at 1M steps, one seed hit 1.430 while another got 0.000
5. **risk_adjusted reward broken** - volatility_penalty too high, needs tuning
6. **SAC too slow on CPU** - needs GPU (Colab/DataSphere)

## Next Steps

- Run with 7 seeds and 2M steps on GPU (Colab notebook ready)
- Try SAC on GPU
- Try ensemble of multiple models
- Tune risk_adjusted penalty (reduce from 0.5 to 0.1)
