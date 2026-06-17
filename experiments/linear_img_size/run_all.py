#!/usr/bin/env python3
"""
Linear track — image size sweep for the Phase 1 winner (ConvNeXt-Tiny).

Fixed protocol (best from phase 1):
  backbone=convnext_tiny | train_balance=natural | class_weight=none

Varies only --image-size (separate embedding cache per size).

Artifacts:
  experiments/results/linear_img_size/<run_id>/
  reports/experiments/linear_img_size/comparison_latest.csv

Usage (repo root):
  python3 experiments/linear_img_size/run_all.py
  python3 experiments/linear_img_size/run_all.py --image-size 224
  python3 experiments/linear_img_size/run_all.py --dry-run
  python3 runners/linear_img_size.py --image-size 224
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.experiment_paths import (
    MANIFEST_PATH,
    SUMMARIES_ROOT,
    ensure_phase_dirs,
    run_dir,
)
from experiments.lib.log_run import append_manifest, save_json, utc_now_iso
from experiments.lib.progress_pct import report_pct

PHASE = "linear_img_size"
MAIN_SCRIPT = REPO_ROOT / "src" / "linear" / "embedding_svc.py"
BACKBONE = "convnext_tiny"
TRAIN_BALANCE = "natural"
CLASS_WEIGHT = "none"

# Default sweep: 128 = phase-1 reference; 224 = standard ImageNet eval size for ConvNeXt
DEFAULT_IMAGE_SIZES = [128, 224]

# Embedding batch size per resolution (avoid OOM on 16 GB VRAM)
BATCH_SIZE_BY_IMAGE_SIZE: dict[int, int] = {
    96: 256,
    128: 256,
    160: 192,
    192: 128,
    224: 128,
    256: 64,
    299: 64,
}


def make_img_run_id(image_size: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{PHASE}_{BACKBONE}_img{image_size}_{TRAIN_BALANCE}_cw{CLASS_WEIGHT}_{ts}"


def batch_size_for(image_size: int) -> int:
    if image_size in BATCH_SIZE_BY_IMAGE_SIZE:
        return BATCH_SIZE_BY_IMAGE_SIZE[image_size]
    # scale roughly with pixel area vs 128 baseline
    scale = (128 / image_size) ** 2
    return max(8, int(256 * scale))


def _stream_subprocess(cmd: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        logf.flush()
        return proc.wait()


def _import_shared_config():
    from runners.linear_vm import shared_config as sc
    return sc


def run_one(image_size: int, dry_run: bool) -> dict | None:
    sc = _import_shared_config()
    batch_size = batch_size_for(image_size)
    run_id = make_img_run_id(image_size)
    out = run_dir(PHASE, run_id)
    log_path = out / "stdout.log"

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--backbone",
        BACKBONE,
        "--train-balance",
        TRAIN_BALANCE,
        "--class-weight",
        CLASS_WEIGHT,
        "--output-dir",
        str(out),
        "--splits-file",
        sc.SPLITS_FILE,
        "--image-size",
        str(image_size),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(sc.NUM_WORKERS),
        "--subset-per-fold",
        str(sc.SUBSET_PER_FOLD),
        "--classifier-tag",
        f"linearsvc_{TRAIN_BALANCE}_cw{CLASS_WEIGHT}",
        "--classifier-backend",
        sc.CLASSIFIER_BACKEND,
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
    rc = _stream_subprocess(cmd, REPO_ROOT, log_path)
    if rc != 0:
        raise RuntimeError(f"Run failed ({rc}): {run_id} — see {log_path}")

    metrics_path = out / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"Missing metrics.json for {run_id}")

    with metrics_path.open(encoding="utf-8") as f:
        result = json.load(f)

    append_manifest(
        {
            "phase": PHASE,
            "run_id": run_id,
            "started_via": "experiments/linear_img_size/run_all.py",
            "finished_at_utc": utc_now_iso(),
            "backbone": BACKBONE,
            "train_balance": TRAIN_BALANCE,
            "class_weight": CLASS_WEIGHT,
            "image_size": image_size,
            "batch_size": batch_size,
            "summary": result.get("summary_mean_std", {}),
            "artifacts_dir": str(out.relative_to(REPO_ROOT)),
        },
        MANIFEST_PATH,
    )
    return result


def build_comparison_row(result: dict) -> dict:
    s = result.get("summary_mean_std", {})
    cfg = result.get("config", {})
    return {
        "backbone": result.get("backbone"),
        "image_size": cfg.get("image_size"),
        "train_balance": result.get("train_balance"),
        "class_weight": result.get("class_weight"),
        "acc_mean": s.get("accuracy", {}).get("mean"),
        "acc_std": s.get("accuracy", {}).get("std"),
        "roc_auc_mean": s.get("roc_auc", {}).get("mean"),
        "roc_auc_std": s.get("roc_auc", {}).get("std"),
        "pr_auc_mean": s.get("pr_auc", {}).get("mean"),
        "pr_auc_std": s.get("pr_auc", {}).get("std"),
        "subset_per_fold": cfg.get("subset_per_fold"),
        "batch_size": cfg.get("batch_size"),
        "finished_at_utc": result.get("finished_at_utc"),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Linear img-size sweep: ConvNeXt-Tiny, natural train, class_weight=none."
    )
    p.add_argument(
        "--image-size",
        type=int,
        action="append",
        dest="image_sizes",
        help="Resolution(s) to run; repeatable. Default: 128 and 224.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    image_sizes = args.image_sizes or DEFAULT_IMAGE_SIZES
    ensure_phase_dirs(PHASE)

    rows: list[dict] = []
    total = len(image_sizes)
    phase_state: dict = {}

    for idx, image_size in enumerate(image_sizes, start=1):
        label = f"ImgSize {image_size}"
        print(f"\nJob {idx}/{total}: ConvNeXt-Tiny img={image_size}", flush=True)
        report_pct(label, 0, 100, phase_state)
        result = run_one(image_size, dry_run=args.dry_run)
        if result:
            rows.append(build_comparison_row(result))
        if not args.dry_run:
            report_pct(label, 100, 100, phase_state)
            report_pct("ImgSize sweep total", idx, total, phase_state)

    if args.dry_run:
        print(f"\nDry run: would execute {total} job(s): {image_sizes}")
        return

    if rows:
        summary_dir = SUMMARIES_ROOT / PHASE
        summary_dir.mkdir(parents=True, exist_ok=True)
        cmp_path = summary_dir / "comparison_latest.csv"
        df = pd.DataFrame(rows).sort_values("image_size")
        df.to_csv(cmp_path, index=False)
        save_json(
            summary_dir / "comparison_latest.json",
            {"phase": PHASE, "finished_at_utc": utc_now_iso(), "runs": rows},
        )
        print(f"\nSaved comparison: {cmp_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
