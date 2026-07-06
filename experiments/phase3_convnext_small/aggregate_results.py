#!/usr/bin/env python3
"""Aggregate phase-3 ConvNeXt-Small FT runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.experiment_paths import SUMMARIES_ROOT

PHASE = "phase3_convnext_small"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / PHASE
OUT_DIR = SUMMARIES_ROOT / PHASE

_EPOCH_COLS = [
    "phase", "epoch", "train_loss",
    "pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1",
]


def _load_metrics_json(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _comparison_row(data: dict, run_dir: Path) -> dict:
    s = data.get("metrics_mean_std", {})
    cfg = data.get("config", {})
    return {
        "run_id": data.get("run_id", run_dir.name),
        "run_tag": data.get("run_tag", cfg.get("run_tag")),
        "model": data.get("model", "convnext_small_binary"),
        "image_size": cfg.get("image_size"),
        "head_epochs": cfg.get("head_epochs"),
        "ft_epochs": cfg.get("ft_epochs"),
        "ft_unfreeze_stages": cfg.get("ft_unfreeze_stages"),
        "lr_head": cfg.get("lr_head"),
        "lr_backbone": cfg.get("lr_backbone"),
        "batch_size": cfg.get("batch_size"),
        "grad_accum_steps": cfg.get("grad_accum_steps"),
        "acc_mean": s.get("accuracy", {}).get("mean"),
        "acc_std": s.get("accuracy", {}).get("std"),
        "roc_auc_mean": s.get("roc_auc", {}).get("mean"),
        "roc_auc_std": s.get("roc_auc", {}).get("std"),
        "pr_auc_mean": s.get("pr_auc", {}).get("mean"),
        "pr_auc_std": s.get("pr_auc", {}).get("std"),
        "runtime_min": data.get("runtime_min"),
        "artifacts_dir": str(run_dir.relative_to(REPO_ROOT)),
    }


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"No results dir: {RESULTS_DIR}")

    cmp_rows: list[dict] = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        data = _load_metrics_json(run_dir)
        if data is None:
            continue
        cmp_rows.append(_comparison_row(data, run_dir))

    if not cmp_rows:
        raise SystemExit(f"No metrics.json under {RESULTS_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmp_df = pd.DataFrame(cmp_rows).sort_values("pr_auc_mean", ascending=False)
    cmp_path = OUT_DIR / "comparison_latest.csv"
    cmp_df.to_csv(cmp_path, index=False)

    print("=" * 88)
    print("ConvNeXt-Small — usporedba runova")
    print("=" * 88)
    print(cmp_df.to_string(index=False))
    print(f"\nWrote {cmp_path} ({len(cmp_df)} run(s))")


if __name__ == "__main__":
    main()
