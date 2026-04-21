"""Project path constants and sys.path helper for importing lib.*"""
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent.parent
SNAPSHOTS_DIR: Path = ROOT / "experiments"
DEFAULT_SNAPSHOT: str = "run_2026-04-21_10seeds"
DATA_DIR: Path = ROOT / "data"
TRAIN_COMPRESSOR: Path = DATA_DIR / "train" / "compressor.pkl"
TRAIN_FEATURES: Path = DATA_DIR / "train" / "features.parquet"
RAW_OHLCV: Path = DATA_DIR / "raw" / "ohlcv.parquet"
RAW_NEWS: Path = DATA_DIR / "raw" / "news.parquet"
OOS_DIR: Path = DATA_DIR / "oos"


def ensure_lib_on_path() -> None:
    """Prepend project root to sys.path so dashboard can import lib.* modules."""
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
