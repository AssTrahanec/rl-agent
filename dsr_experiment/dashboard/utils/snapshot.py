"""Snapshot discovery and artifact loading.

All functions are cached with st.cache_data (filesystem listings are cheap
but invoked on every rerun).
"""
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import SNAPSHOTS_DIR

# Folder names like "DQN_seed42_20260421_004100"
_MODEL_DIR_RE = re.compile(r"^(?P<algo>SAC|DQN|PPO)_seed(?P<seed>-?\d+)_(?P<ts>\d{8}_\d{6})$")


@st.cache_data(ttl=300)
def list_snapshots() -> list[str]:
    """Return snapshot folder names sorted newest-first.

    A valid snapshot has both models/ and results/ subfolders.
    """
    if not SNAPSHOTS_DIR.exists():
        return []
    out = []
    for p in SNAPSHOTS_DIR.iterdir():
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        if (p / "models").exists() and (p / "results").exists():
            out.append(p.name)
    return sorted(out, reverse=True)


@st.cache_data(ttl=300)
def discover_models(snapshot: str) -> dict[str, list[dict]]:
    """Scan experiments/{snapshot}/models/{ALGO}/{ALGO}_seed{S}_{TS}/model.zip.

    Returns {"DQN": [{"seed": 42, "path": ".../model.zip", "run_id": "..."}, ...],
             "SAC": [...]}
    Sorted by seed ascending.
    """
    out: dict[str, list[dict]] = {}
    models_root = SNAPSHOTS_DIR / snapshot / "models"
    if not models_root.exists():
        return out
    for algo_dir in models_root.iterdir():
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name
        if algo not in {"SAC", "DQN", "PPO"}:
            continue
        entries = []
        for run_dir in algo_dir.iterdir():
            if not run_dir.is_dir():
                continue
            m = _MODEL_DIR_RE.match(run_dir.name)
            if not m:
                continue
            zip_path = run_dir / "model.zip"
            if not zip_path.exists():
                continue
            entries.append({
                "seed": int(m.group("seed")),
                "path": str(zip_path),
                "run_id": run_dir.name,
                "dir": str(run_dir),
            })
        entries.sort(key=lambda x: x["seed"])
        if entries:
            out[algo] = entries
    return out


@st.cache_data(ttl=300)
def list_periods(snapshot: str) -> list[str]:
    """Return periods present as oos_{period}.csv in results/."""
    results_dir = SNAPSHOTS_DIR / snapshot / "results"
    if not results_dir.exists():
        return []
    periods = set()
    # Filename format from run.py: oos_{period_key}.csv, so filename like "oos_oos_2024.csv"
    # Strip leading "oos_" once.
    for f in results_dir.glob("oos_*.csv"):
        stem = f.stem  # "oos_oos_2024"
        if stem.startswith("oos_"):
            periods.add(stem[4:])  # "oos_2024"
    return sorted(periods)


@st.cache_data
def load_aggregate_csv(snapshot: str, period: str) -> pd.DataFrame:
    """Load results/oos_{period}.csv (which contains per-seed rows)."""
    path = SNAPSHOTS_DIR / snapshot / "results" / f"oos_{period}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_seed_npz(snapshot: str, period: str, algo: str, seed: int) -> dict:
    """Load daily_returns/allocations/equity_curve arrays for a single seed."""
    path = SNAPSHOTS_DIR / snapshot / "results" / f"oos_{period}_{algo}_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"npz artifact missing: {path}")
    d = np.load(path)
    return {
        "daily_returns": d["daily_returns"],
        "allocations": d["allocations"],
        "equity_curve": d["equity_curve"],
    }


def vecnorm_for(model_dir: str) -> Optional[str]:
    """Return path to sibling vecnormalize.pkl if it exists, else None."""
    p = Path(model_dir) / "vecnormalize.pkl"
    return str(p) if p.exists() else None
