#!/usr/bin/env python3
"""
Phase 1 (mentor): compare all linear backbones on natural vs balanced TRAIN data.

Validation always uses the natural (~28% IDC) fold distribution.

Runs:
  6 backbones × 2 train_balance modes (natural, balanced)
  default: class_weight=none (clean data comparison)

Artifacts:
  experiments/results/phase1_balance/<run_id>/  — full logs + metrics (gitignored)
  reports/experiments/phase1_balance/             — comparison CSV (local / thesis; not in git)
  experiments/manifest.jsonl                      — append-only audit log

Usage (repo root):
  python experiments/phase1_balance/run_all.py
  python experiments/phase1_balance/run_all.py --dry-run
  python experiments/phase1_balance/run_all.py --backbone convnext_tiny --train-balance balanced
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.experiment_paths import (
    MANIFEST_PATH,
    SUMMARIES_ROOT,
    ensure_phase_dirs,
    make_run_id,
    run_dir,
)
from experiments.lib.log_run import append_manifest, save_json, utc_now_iso

PHASE = "phase1_balance"
MAIN_SCRIPT = REPO_ROOT / "src" / "linear" / "embedding_svc.py"

BACKBONES = [
    "mobilenetv2",
    "resnet18",
    "efficientnet_b0",
    "convnext_tiny",
    "resnet_50",
    "inception_v3",
]

# (train_balance, class_weight) — mentor step 1: data regime comparison
DEFAULT_CONDITIONS = [
    ("natural", "none"),
    ("balanced", "none"),
]

# Optional: prior VM baseline (natural data + sklearn balanced weights)
LEGACY_CONDITION = ("natural", "balanced")


def _import_shared_config():
    from runners.linear_vm import shared_config as sc
    return sc


def run_one(
    backbone: str,
    train_balance: str,
    class_weight: str,
    dry_run: bool,
) -> dict | None:
    sc = _import_shared_config()
    run_id = make_run_id(PHASE, backbone, train_balance, class_weight)
    out = run_dir(PHASE, run_id)
    log_path = out / "stdout.log"

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--backbone",
        backbone,
        "--train-balance",
        train_balance,
        "--class-weight",
        class_weight,
        "--output-dir",
        str(out),
        "--splits-file",
        sc.SPLITS_FILE,
        "--image-size",
        str(sc.IMAGE_SIZE),
        "--batch-size",
        str(sc.BATCH_SIZE),
        "--num-workers",
        str(sc.NUM_WORKERS),
        "--subset-per-fold",
        str(sc.SUBSET_PER_FOLD),
        "--classifier-tag",
        f"linearsvc_{train_balance}_cw{class_weight}",
        "--emb-root",
        sc.EMB_ROOT,
        "--svm-c",
        str(sc.SVM_C),
        "--calibration-cv",
        str(sc.CALIBRATION_CV),
        "--seed",
        str(sc.RANDOM_SEED),
    ]
    if sc.PIN_MEMORY:
        cmd.append("--pin-memory")
    else:
        cmd.append("--no-pin-memory")

    print("\n" + "=" * 72)
    print("RUN", run_id)
    print(" ".join(cmd))
    print("=" * 72)

    if dry_run:
        return None

    out.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed ({proc.returncode}): {run_id} — see {log_path}")

    metrics_path = out / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"Missing metrics.json for {run_id}")

    with metrics_path.open(encoding="utf-8") as f:
        result = json.load(f)

    manifest_entry = {
        "phase": PHASE,
        "run_id": run_id,
        "started_via": "experiments/phase1_balance/run_all.py",
        "finished_at_utc": utc_now_iso(),
        "backbone": backbone,
        "train_balance": train_balance,
        "class_weight": class_weight,
        "summary": result.get("summary_mean_std", {}),
        "artifacts_dir": str(out.relative_to(REPO_ROOT)),
    }
    append_manifest(manifest_entry, MANIFEST_PATH)
    return result


def build_comparison_row(result: dict) -> dict:
    s = result.get("summary_mean_std", {})
    cfg = result.get("config", {})
    return {
        "backbone": result.get("backbone"),
        "train_balance": result.get("train_balance"),
        "class_weight": result.get("class_weight"),
        "acc_mean": s.get("accuracy", {}).get("mean"),
        "acc_std": s.get("accuracy", {}).get("std"),
        "roc_auc_mean": s.get("roc_auc", {}).get("mean"),
        "roc_auc_std": s.get("roc_auc", {}).get("std"),
        "pr_auc_mean": s.get("pr_auc", {}).get("mean"),
        "pr_auc_std": s.get("pr_auc", {}).get("std"),
        "image_size": cfg.get("image_size"),
        "subset_per_fold": cfg.get("subset_per_fold"),
        "finished_at_utc": result.get("finished_at_utc"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1: all backbones × natural/balanced train.")
    p.add_argument("--backbone", action="append", help="Limit to backbone(s); repeatable.")
    p.add_argument(
        "--train-balance",
        choices=("natural", "balanced"),
        help="Run only one train_balance mode (default: both).",
    )
    p.add_argument("--include-legacy", action="store_true", help="Also run natural+class_weight=balanced (old VM setup).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ensure_phase_dirs(PHASE)
    backbones = args.backbone or BACKBONES
    conditions = list(DEFAULT_CONDITIONS)
    if args.train_balance:
        conditions = [(args.train_balance, "none")]
    if args.include_legacy and LEGACY_CONDITION not in conditions:
        conditions.append(LEGACY_CONDITION)

    rows: list[dict] = []
    for backbone in backbones:
        for train_balance, class_weight in conditions:
            result = run_one(
                backbone,
                train_balance,
                class_weight,
                dry_run=args.dry_run,
            )
            if result:
                rows.append(build_comparison_row(result))

    if args.dry_run:
        print(f"\nDry run: would execute {len(backbones) * len(conditions)} jobs.")
        return

    if rows:
        summary_dir = SUMMARIES_ROOT / PHASE
        summary_dir.mkdir(parents=True, exist_ok=True)
        cmp_path = summary_dir / "comparison_latest.csv"
        df = pd.DataFrame(rows)
        df.to_csv(cmp_path, index=False)
        save_json(
            summary_dir / "comparison_latest.json",
            {"phase": PHASE, "finished_at_utc": utc_now_iso(), "runs": rows},
        )
        print(f"\nSaved comparison: {cmp_path}")
        print(df.sort_values("pr_auc_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
