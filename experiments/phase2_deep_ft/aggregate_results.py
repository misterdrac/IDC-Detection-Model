#!/usr/bin/env python3
"""Build phase 2 CNN FT comparison table from experiments/results/phase2_deep_ft/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.experiment_paths import SUMMARIES_ROOT

PHASE = "phase2_deep_ft"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / PHASE
OUT_DIR = SUMMARIES_ROOT / PHASE


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"No results dir: {RESULTS_DIR}")

    rows: list[dict] = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        with metrics_path.open(encoding="utf-8") as f:
            data = json.load(f)
        s = data.get("metrics_mean_std", {})
        cfg = data.get("config", {})
        rows.append({
            "run_id": data.get("run_id", run_dir.name),
            "run_tag": data.get("run_tag", cfg.get("run_tag")),
            "image_size": cfg.get("image_size"),
            "head_epochs": cfg.get("head_epochs"),
            "ft_epochs": cfg.get("ft_epochs"),
            "finetune_last_stage": cfg.get("finetune_last_stage"),
            "lr_head": cfg.get("lr_head"),
            "lr_backbone": cfg.get("lr_backbone"),
            "weight_decay": cfg.get("weight_decay"),
            "batch_size": cfg.get("batch_size"),
            "acc_mean": s.get("accuracy", {}).get("mean"),
            "acc_std": s.get("accuracy", {}).get("std"),
            "roc_auc_mean": s.get("roc_auc", {}).get("mean"),
            "roc_auc_std": s.get("roc_auc", {}).get("std"),
            "pr_auc_mean": s.get("pr_auc", {}).get("mean"),
            "pr_auc_std": s.get("pr_auc", {}).get("std"),
            "runtime_min": data.get("runtime_min"),
            "finished_at_utc": data.get("config", {}).get("finished_at_utc"),
            "artifacts_dir": str(run_dir.relative_to(REPO_ROOT)),
        })

    if not rows:
        raise SystemExit(f"No metrics.json under {RESULTS_DIR}")

    df = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmp_path = OUT_DIR / "comparison_latest.csv"
    df.to_csv(cmp_path, index=False)
    with (OUT_DIR / "comparison_latest.json").open("w", encoding="utf-8") as f:
        json.dump({"phase": PHASE, "runs": df.to_dict(orient="records")}, f, indent=2)

    print(f"Wrote {cmp_path} ({len(df)} run(s))")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
