#!/usr/bin/env python3
"""
Rebuild phase 1 comparison tables from VM run folders.

Run on VM (or anywhere with experiments/results/phase1_balance/):

  python3 experiments/phase1_balance/aggregate_results.py

Writes:
  reports/experiments/phase1_balance/comparison_latest.csv
  reports/experiments/phase1_balance/comparison_latest.json
  reports/experiments/phase1_balance/per_run_summary.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.experiment_paths import SUMMARIES_ROOT

PHASE = "phase1_balance"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / PHASE
OUT_DIR = SUMMARIES_ROOT / PHASE


def _load_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    with metrics_path.open(encoding="utf-8") as f:
        data = json.load(f)
    s = data.get("summary_mean_std", {})
    cfg = data.get("config", {})
    return {
        "run_id": run_dir.name,
        "backbone": data.get("backbone"),
        "train_balance": data.get("train_balance"),
        "class_weight": data.get("class_weight"),
        "classifier_backend": data.get("classifier_backend"),
        "acc_mean": s.get("accuracy", {}).get("mean"),
        "acc_std": s.get("accuracy", {}).get("std"),
        "roc_auc_mean": s.get("roc_auc", {}).get("mean"),
        "roc_auc_std": s.get("roc_auc", {}).get("std"),
        "pr_auc_mean": s.get("pr_auc", {}).get("mean"),
        "pr_auc_std": s.get("pr_auc", {}).get("std"),
        "image_size": cfg.get("image_size"),
        "subset_per_fold": cfg.get("subset_per_fold"),
        "finished_at_utc": data.get("finished_at_utc"),
        "stdout_log": str((run_dir / "stdout.log").relative_to(REPO_ROOT)),
        "metrics_json": str(metrics_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"No results dir: {RESULTS_DIR}")

    rows: list[dict] = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        row = _load_run(run_dir)
        if row:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No metrics.json under {RESULTS_DIR}")

    # If duplicate backbone+train_balance+class_weight, keep latest by run_id (timestamp suffix)
    df = pd.DataFrame(rows)
    df = df.sort_values("run_id").drop_duplicates(
        subset=["backbone", "train_balance", "class_weight"],
        keep="last",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_run = OUT_DIR / "per_run_summary.csv"
    cmp_path = OUT_DIR / "comparison_latest.csv"
    cmp_json = OUT_DIR / "comparison_latest.json"

    df.to_csv(per_run, index=False)
    cmp_cols = [
        "backbone", "train_balance", "class_weight",
        "acc_mean", "acc_std", "roc_auc_mean", "roc_auc_std",
        "pr_auc_mean", "pr_auc_std",
        "image_size", "subset_per_fold", "finished_at_utc",
    ]
    df[cmp_cols].sort_values("pr_auc_mean", ascending=False).to_csv(cmp_path, index=False)
    with cmp_json.open("w", encoding="utf-8") as f:
        json.dump({"phase": PHASE, "runs": df.to_dict(orient="records")}, f, indent=2)

    print(f"Runs aggregated: {len(df)}")
    print(f"Wrote {per_run}")
    print(f"Wrote {cmp_path}")
    print(f"Wrote {cmp_json}")
    print(df[cmp_cols].sort_values("pr_auc_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
