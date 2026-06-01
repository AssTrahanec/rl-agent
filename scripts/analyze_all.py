"""Analyze all experiment results: BTC + ETH ablation, algo comparison.

Usage:
    python -m scripts.analyze_all
"""
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.load_features import load_features_for_agent
from src.eval.backtest import run_backtest
from src.eval.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

AGENT_TYPES = ["baseline", "sentiment", "embeddings"]
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"


def find_models(base_dir: str, agent_type: str) -> list[Path]:
    agent_dir = Path(base_dir) / agent_type
    if not agent_dir.exists():
        return []
    return sorted([
        d / "model.zip"
        for d in agent_dir.iterdir()
        if d.is_dir() and (d / "model.zip").exists()
    ])


def backtest_models(models: list[Path], features, prices, label: str) -> list[dict]:
    results = []
    for mp in models:
        ts = mp.parent.name
        try:
            r = run_backtest(features=features, prices=prices,
                             model_path=str(mp), window=30, tx_cost=0.001)
            m = r["metrics"]
            results.append({"label": f"{label}/{ts}", **m})
        except Exception as e:
            logger.warning(f"  SKIP {label}/{ts}: {e}")
    return results


def buy_and_hold(prices):
    dr = np.diff(prices) / prices[:-1]
    return compute_metrics(dr)


def print_section(title, grouped, keys=None):
    keys = keys or list(grouped.keys())
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Agent':<20} {'N':>3} {'Sharpe':>14} {'Return':>14} {'MaxDD':>14}")
    print(f"  {'-'*65}")
    for k in keys:
        rows = grouped.get(k, [])
        if not rows:
            continue
        sh = [r["sharpe_ratio"] for r in rows]
        ret = [r["total_return"] for r in rows]
        dd = [r["max_drawdown"] for r in rows]
        print(
            f"  {k:<20} {len(rows):>3} "
            f"{np.mean(sh):>6.3f} +/- {np.std(sh):.3f} "
            f"{np.mean(ret)*100:>5.1f} +/- {np.std(ret)*100:.1f}% "
            f"{np.mean(dd)*100:>5.1f} +/- {np.std(dd)*100:.1f}%"
        )


def main():
    all_csv_rows = []

    # ============================================================
    # 1. BTC Ablation (all models in experiments/)
    # ============================================================
    print("\n" + "#" * 80)
    print("#  BTC/USDT ABLATION (test: 2024)")
    print("#" * 80)

    btc_grouped = defaultdict(list)
    for at in AGENT_TYPES:
        models = find_models("experiments", at)
        if not models:
            continue
        try:
            feat, prices = load_features_for_agent(at, "BTC/USDT", TEST_START, TEST_END)
        except Exception as e:
            logger.error(f"Cannot load BTC {at}: {e}")
            continue

        results = backtest_models(models, feat, prices, f"btc/{at}")
        for r in results:
            btc_grouped[at].append(r)
            all_csv_rows.append({"asset": "BTC", "experiment": "ablation", "agent_type": at, **r})

    # BTC Buy & Hold
    feat_bl, prices_bl = load_features_for_agent("baseline", "BTC/USDT", TEST_START, TEST_END)
    bh = buy_and_hold(prices_bl)
    print(f"\n  Buy & Hold BTC: Sharpe={bh['sharpe_ratio']:.3f}  "
          f"Return={bh['total_return']*100:.1f}%  MaxDD={bh['max_drawdown']*100:.1f}%")

    # Filter out short-run models (return < 10% likely interrupted)
    btc_filtered = defaultdict(list)
    for at, rows in btc_grouped.items():
        good = [r for r in rows if abs(r["total_return"]) > 0.10]
        btc_filtered[at] = good

    print_section("BTC ALL MODELS", btc_grouped, AGENT_TYPES)
    print_section("BTC FILTERED (|return| > 10%)", btc_filtered, AGENT_TYPES)

    # ============================================================
    # 2. ETH Ablation (experiments_eth/)
    # ============================================================
    print("\n" + "#" * 80)
    print("#  ETH/USDT ABLATION (test: 2024)")
    print("#" * 80)

    eth_grouped = defaultdict(list)
    for at in AGENT_TYPES:
        models = find_models("experiments_eth", at)
        if not models:
            continue
        try:
            feat, prices = load_features_for_agent(at, "ETH/USDT", TEST_START, TEST_END)
        except Exception as e:
            logger.error(f"Cannot load ETH {at}: {e}")
            continue

        results = backtest_models(models, feat, prices, f"eth/{at}")
        for r in results:
            eth_grouped[at].append(r)
            all_csv_rows.append({"asset": "ETH", "experiment": "ablation", "agent_type": at, **r})

    if eth_grouped:
        feat_bl_eth, prices_bl_eth = load_features_for_agent("baseline", "ETH/USDT", TEST_START, TEST_END)
        bh_eth = buy_and_hold(prices_bl_eth)
        print(f"\n  Buy & Hold ETH: Sharpe={bh_eth['sharpe_ratio']:.3f}  "
              f"Return={bh_eth['total_return']*100:.1f}%  MaxDD={bh_eth['max_drawdown']*100:.1f}%")
        print_section("ETH ALL MODELS", eth_grouped, AGENT_TYPES)

    # ============================================================
    # 3. Algo Comparison (experiments/algo_comparison/)
    # ============================================================
    print("\n" + "#" * 80)
    print("#  ALGO COMPARISON: PPO vs A2C vs SAC (embeddings, BTC, test: 2024)")
    print("#" * 80)

    algo_dir = Path("experiments/algo_comparison/embeddings")
    if algo_dir.exists():
        algo_models = sorted([
            d / "model.zip" for d in algo_dir.iterdir()
            if d.is_dir() and (d / "model.zip").exists()
        ])

        feat_emb, prices_emb = load_features_for_agent("embeddings", "BTC/USDT", TEST_START, TEST_END)

        # PPO = first 3, A2C = next 3, SAC = next 3 (by timestamp order)
        algo_names = ["PPO", "A2C", "SAC"]
        algo_grouped = defaultdict(list)

        for i, mp in enumerate(algo_models):
            algo_idx = i // 3
            if algo_idx >= len(algo_names):
                break
            algo = algo_names[algo_idx]
            ts = mp.parent.name
            try:
                r = run_backtest(features=feat_emb, prices=prices_emb,
                                 model_path=str(mp), window=30, tx_cost=0.001)
                m = r["metrics"]
                algo_grouped[algo].append(m)
                all_csv_rows.append({
                    "asset": "BTC", "experiment": "algo_comparison",
                    "agent_type": f"embeddings_{algo}", **m
                })
                print(f"  {algo} seed {i%3}: Sharpe={m['sharpe_ratio']:.3f}  "
                      f"Return={m['total_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%")
            except Exception as e:
                logger.warning(f"  SKIP {algo}/{ts}: {e}")

        print_section("ALGO COMPARISON SUMMARY", algo_grouped, algo_names)

    # ============================================================
    # 4. Save all to CSV
    # ============================================================
    csv_path = Path("results/all_experiments.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if all_csv_rows:
        fields = list(all_csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\nAll results saved to {csv_path} ({len(all_csv_rows)} rows)")


if __name__ == "__main__":
    main()
