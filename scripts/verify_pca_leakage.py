"""Verify PCA compressor was fit on train-period news only.

Exits 0 if safe, 1 if leakage detected.
"""
import sys
from pathlib import Path
import pandas as pd

COMPRESSOR_PATH = Path("data/processed/embedding_compressor.pkl")
FEATURES_PATH = Path("data/processed/btc_4h_embedding_features.parquet")
TRAIN_END = "2023-12-31"


def main() -> int:
    if not COMPRESSOR_PATH.exists():
        print(f"FAIL: compressor not found at {COMPRESSOR_PATH}")
        return 1
    if not FEATURES_PATH.exists():
        print(f"FAIL: features not found at {FEATURES_PATH}")
        return 1

    from src.features.embedding_compressor import EmbeddingCompressor
    comp = EmbeddingCompressor.load(str(COMPRESSOR_PATH))
    meta = getattr(comp, "fit_metadata", None)
    if meta is None:
        print("WARN: compressor has no fit_metadata. Checking manually.")
        print(f"  n_components={comp.n_components}")
        print("  Unable to verify train cutoff automatically. "
              "Inspect the build script and rebuild if fit used full period.")
        return 1

    fit_end = pd.Timestamp(meta.get("train_end", ""))
    if fit_end > pd.Timestamp(TRAIN_END):
        print(f"FAIL: compressor fit on news up to {fit_end}, "
              f"must be <= {TRAIN_END}")
        return 1
    print(f"OK: compressor fit_end={fit_end}, <= {TRAIN_END}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
