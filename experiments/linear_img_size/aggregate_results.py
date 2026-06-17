#!/usr/bin/env python3
"""Rebuild image-size sweep comparison; merge Phase 1 img128 reference if missing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.experiment_paths import SUMMARIES_ROOT

PHASE = "linear_img_size"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / PHASE
OUT_DIR = SUMMARIES_ROOT / PHASE
PHASE1_CMP = SUMMARIES_ROOT / "phase1_balance" / "comparison_latest.csv"


def _phase1_img128_row() -> dict | None:
    if not PHASE1_CMP.is_file():
        return None
    df = pd.read_csv(PHASE1_CMP)
    mask = (
        (df["backbone"] == "convnext_tiny")
        & (df["train_balance"] == "natural")
        & (df["class_weight"] == "none")
        & (df["image_size"] == 128)
    )
    hit = df[mask]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {
        "run_id": "phase1_balance_convnext_tiny_natural_cwnone_ref",
        "backbone": "convnext_tiny",
        "image_size": 128,
        "train_balance": "natural",
        "class_weight": "none",
        "acc_mean": r["acc_mean"],
        "acc_std": r["acc_std"],
        "roc_auc_mean": r["roc_auc_mean"],
        "roc_auc_std": r["roc_auc_std"],
        "pr_auc_mean": r["pr_auc_mean"],
        "pr_auc_std": r["pr_auc_std"],
        "batch_size": 256,
        "finished_at_utc": r.get("finished_at_utc", ""),
        "source": "phase1_balance",
    }


def main() -> None:
    rows: list[dict] = []

    if RESULTS_DIR.is_dir():
        for run_dir in sorted(RESULTS_DIR.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.is_file():
                continue
            with metrics_path.open(encoding="utf-8") as f:
                data = json.load(f)
            s = data.get("summary_mean_std", {})
            cfg = data.get("config", {})
            rows.append({
                "run_id": run_dir.name,
                "backbone": data.get("backbone"),
                "image_size": cfg.get("image_size"),
                "train_balance": data.get("train_balance"),
                "class_weight": data.get("class_weight"),
                "acc_mean": s.get("accuracy", {}).get("mean"),
                "acc_std": s.get("accuracy", {}).get("std"),
                "roc_auc_mean": s.get("roc_auc", {}).get("mean"),
                "roc_auc_std": s.get("roc_auc", {}).get("std"),
                "pr_auc_mean": s.get("pr_auc", {}).get("mean"),
                "pr_auc_std": s.get("pr_auc", {}).get("std"),
                "batch_size": cfg.get("batch_size"),
                "finished_at_utc": data.get("finished_at_utc"),
                "source": PHASE,
            })

    if not rows and _phase1_img128_row() is None:
        raise SystemExit(f"No metrics under {RESULTS_DIR} and no phase1 reference.")

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("finished_at_utc").drop_duplicates(subset=["image_size"], keep="last")

    if 128 not in df.get("image_size", pd.Series(dtype=int)).tolist():
        ref = _phase1_img128_row()
        if ref:
            df = pd.concat([pd.DataFrame([ref]), df], ignore_index=True)

    df = df.sort_values("image_size")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmp_path = OUT_DIR / "comparison_latest.csv"
    df.to_csv(cmp_path, index=False)

    payload: dict = {"phase": PHASE, "runs": df.to_dict(orient="records")}
    if len(df) >= 2 and 128 in df["image_size"].values and 224 in df["image_size"].values:
        r128 = df[df["image_size"] == 128].iloc[0]
        r224 = df[df["image_size"] == 224].iloc[0]
        payload["delta_224_vs_128"] = {
            "accuracy": float(r224["acc_mean"] - r128["acc_mean"]),
            "roc_auc": float(r224["roc_auc_mean"] - r128["roc_auc_mean"]),
            "pr_auc": float(r224["pr_auc_mean"] - r128["pr_auc_mean"]),
        }

    with (OUT_DIR / "comparison_latest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {cmp_path} ({len(df)} image size(s))")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
