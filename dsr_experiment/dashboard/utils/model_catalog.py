"""Human-readable catalog of trained models across all snapshots.

A 'model' here = (snapshot, algo) — one configuration trained with N seeds.
"""
from dataclasses import dataclass

from dashboard.utils import snapshot


# Per-(snapshot, algo) labels where it matters (e.g. 10seeds has different
# algo-specific tweaks). Fallback to SNAPSHOT_LABELS by snapshot alone.
SNAPSHOT_ALGO_LABELS: dict[tuple[str, str], str] = {
    ("run_2026-04-21_10seeds", "SAC"): "long-only, 10 seeds",
    ("run_2026-04-21_10seeds", "DQN"): "discrete all-in/out, tuned + 10 seeds",
}

SNAPSHOT_LABELS: dict[str, str] = {
    "run_2026-04-20_cautious_agents": "long-only, 3 seeds",
    "run_2026-04-20_long_short": "long-short +funding, 3 seeds",
    "run_2026-04-20_discrete_trader": "discrete all-in/out, 3 seeds",
    "run_2026-04-21_10seeds": "10 seeds",
}


@dataclass(frozen=True)
class ModelEntry:
    snapshot: str
    algo: str
    n_seeds: int

    @property
    def label(self) -> str:
        specific = SNAPSHOT_ALGO_LABELS.get((self.snapshot, self.algo))
        if specific is not None:
            return f"{self.algo} — {specific}"
        snap_label = SNAPSHOT_LABELS.get(self.snapshot, self.snapshot)
        return f"{self.algo} — {snap_label}"

    @property
    def key(self) -> str:
        return f"{self.snapshot}::{self.algo}"


# Only these (snapshot, algo) pairs are exposed to the dashboard UI.
# Validation page iterates more broadly via an internal helper.
_VISIBLE_ENTRIES: set[tuple[str, str]] = {
    ("run_2026-04-21_10seeds", "DQN"),
    ("run_2026-04-21_10seeds", "SAC"),
}


def list_model_entries(all_snapshots: bool = False) -> list[ModelEntry]:
    """Enumerate (snapshot, algo) combos that have model.zip files.

    By default returns only the two primary 10-seed configurations
    (DQN + SAC from run_2026-04-21_10seeds). Pass all_snapshots=True
    to get everything (used by the Validation page).
    """
    out: list[ModelEntry] = []
    for snap in snapshot.list_snapshots():
        models = snapshot.discover_models(snap)
        for algo, entries in models.items():
            if not entries:
                continue
            if not all_snapshots and (snap, algo) not in _VISIBLE_ENTRIES:
                continue
            out.append(ModelEntry(snapshot=snap, algo=algo, n_seeds=len(entries)))
    algo_order = {"DQN": 0, "SAC": 1, "PPO": 2}
    snap_order = {s: i for i, s in enumerate(snapshot.list_snapshots())}
    out.sort(key=lambda e: (snap_order.get(e.snapshot, 99), algo_order.get(e.algo, 99)))
    return out


def find_entry(entries: list[ModelEntry], key: str) -> ModelEntry | None:
    for e in entries:
        if e.key == key:
            return e
    return None
