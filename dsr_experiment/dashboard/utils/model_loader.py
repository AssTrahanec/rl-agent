"""Cached SB3 model loader and PCA compressor loader."""
from typing import Optional

import streamlit as st

from dashboard.utils.paths import TRAIN_COMPRESSOR, ensure_lib_on_path

ensure_lib_on_path()


@st.cache_resource
def load_sb3_model(model_path: str, algo_hint: Optional[str] = None):
    """Load a Stable-Baselines3 model from model.zip for inference.

    Off-policy models (SAC, DQN) serialize their replay buffer shape in the
    zip. Naive .load() tries to re-allocate the training-size buffer
    (300k × obs_dim ≈ 3 GB per model). For inference we don't need the
    buffer at all — override with buffer_size=1 via custom_objects.
    """
    from stable_baselines3 import PPO, SAC, DQN

    # custom_objects lets us replace serialized hyperparameters at load time.
    # For off-policy algos, we shrink the replay buffer to stop the 3GB alloc.
    custom = {"buffer_size": 1}

    hint_map = {"SAC": SAC, "DQN": DQN, "PPO": PPO}
    if algo_hint in hint_map:
        cls = hint_map[algo_hint]
        if cls is PPO:
            return cls.load(model_path, device="cpu")
        return cls.load(model_path, device="cpu", custom_objects=custom)

    # Fallback: try in order, independent attempts.
    last_err = None
    for cls in (DQN, SAC, PPO):
        try:
            if cls is PPO:
                return cls.load(model_path, device="cpu")
            return cls.load(model_path, device="cpu", custom_objects=custom)
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
