"""Cached SB3 model loader and PCA compressor loader."""
from typing import Optional

import streamlit as st

from dashboard.utils.paths import TRAIN_COMPRESSOR, ensure_lib_on_path

ensure_lib_on_path()


@st.cache_resource
def load_sb3_model(model_path: str, algo_hint: Optional[str] = None):
    """Load a Stable-Baselines3 model from model.zip.

    Mirrors lib/backtest.py::load_model but cached via st.cache_resource.
    algo_hint ∈ {"SAC","DQN","PPO"} speeds up loading by trying the right
    class first.
    """
    from stable_baselines3 import PPO, SAC, DQN

    order = [PPO, SAC, DQN]
    if algo_hint == "SAC":
        order = [SAC, DQN, PPO]
    elif algo_hint == "DQN":
        order = [DQN, SAC, PPO]
    elif algo_hint == "PPO":
        order = [PPO, SAC, DQN]

    last_err = None
    for cls in order:
        try:
            return cls.load(model_path, device="cpu")
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise ValueError(f"Cannot load model from {model_path}: {last_err}")


@st.cache_resource
def load_compressor():
    """Load fitted PCA EmbeddingCompressor from data/train/compressor.pkl."""
    from lib.features.embeddings import EmbeddingCompressor

    if not TRAIN_COMPRESSOR.exists():
        raise FileNotFoundError(f"Compressor not found at {TRAIN_COMPRESSOR}")
    return EmbeddingCompressor.load(str(TRAIN_COMPRESSOR))
