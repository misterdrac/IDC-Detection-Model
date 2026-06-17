"""Directory layout for reproducible experiment artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Heavy artifacts (logs, full stdout) — gitignored
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"

# Small thesis-ready summaries — committed to git
SUMMARIES_ROOT = REPO_ROOT / "reports" / "experiments"

# Append-only master log (one JSON object per line)
MANIFEST_PATH = REPO_ROOT / "experiments" / "manifest.jsonl"


def make_run_id(phase: str, backbone: str, train_balance: str, class_weight: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{phase}_{backbone}_{train_balance}_cw{class_weight}_{ts}"


def run_dir(phase: str, run_id: str) -> Path:
    return RESULTS_ROOT / phase / run_id


def ensure_phase_dirs(phase: str) -> None:
    (RESULTS_ROOT / phase).mkdir(parents=True, exist_ok=True)
    (SUMMARIES_ROOT / phase).mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
