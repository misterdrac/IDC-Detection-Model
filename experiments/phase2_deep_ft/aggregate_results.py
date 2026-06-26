#!/usr/bin/env python3
"""Aggregate phase-2 CNN FT runs: comparison table + per-fold per-epoch detail."""
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
        "image_size": cfg.get("image_size"),
        "head_epochs": cfg.get("head_epochs"),
        "ft_epochs": cfg.get("ft_epochs"),
        "finetune_last_stage": cfg.get("finetune_last_stage"),
        "ft_unfreeze_stages": cfg.get("ft_unfreeze_stages"),
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
        "artifacts_dir": str(run_dir.relative_to(REPO_ROOT)),
    }


def _epoch_rows_from_run(data: dict, run_id: str, run_tag: str) -> list[dict]:
    rows: list[dict] = []
    for fold_row in data.get("fold_rows", []):
        fold = fold_row.get("fold")
        best_tag = fold_row.get("best_tag", "")
        for ep in fold_row.get("epoch_history") or []:
            rows.append({
                "run_id": run_id,
                "run_tag": run_tag,
                "fold": fold,
                "best_tag": best_tag,
                **{k: ep.get(k) for k in _EPOCH_COLS},
            })
    return rows


def _format_epoch_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(nema podataka)"
    view = df.copy()
    if "train_loss" in view.columns:
        view["train_loss"] = view["train_loss"].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
        )
    for c in ("pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1"):
        if c in view.columns:
            view[c] = view[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    cols = [c for c in ["phase", "epoch", "train_loss", "pr_auc", "roc_auc",
                        "accuracy", "precision", "recall", "f1"] if c in view.columns]
    return view[cols].to_string(index=False)


def _mean_epochs_across_folds(epoch_df: pd.DataFrame) -> pd.DataFrame:
    if epoch_df.empty:
        return epoch_df
    numeric = [c for c in (
        "train_loss", "pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1",
    ) if c in epoch_df.columns]
    mean = epoch_df.groupby(["phase", "epoch"], as_index=False)[numeric].mean(numeric_only=True)
    for c in ("pr_auc", "roc_auc", "accuracy"):
        if c in epoch_df.columns:
            std_df = epoch_df.groupby(["phase", "epoch"], as_index=False)[c].std(ddof=0)
            mean = mean.merge(
                std_df.rename(columns={c: f"{c}_std"}),
                on=["phase", "epoch"],
                how="left",
            )
    phase_order = {"Init": 0, "HEAD": 1, "FT": 2, "Final": 3}
    mean["_ord"] = mean["phase"].map(lambda p: phase_order.get(p, 9))
    return mean.sort_values(["_ord", "epoch"]).drop(columns="_ord")


def print_epoch_report(run_id: str, run_tag: str, epoch_df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print(f"RUN: {run_tag or run_id}")
    print(f"     {run_id}")
    print("=" * 88)

    if epoch_df.empty:
        print("  (nema epoch_history — pokreni trening s novijom verzijom skripte)")
        return

    phase_order = {"Init": 0, "HEAD": 1, "FT": 2, "Final": 3}
    for fold in sorted(epoch_df["fold"].unique(), key=lambda x: int(x)):
        fdf = epoch_df[epoch_df["fold"] == fold].copy()
        fdf["_ord"] = fdf["phase"].map(lambda p: phase_order.get(p, 9))
        fdf = fdf.sort_values(["_ord", "epoch"]).drop(columns="_ord")
        best_tag = fdf["best_tag"].iloc[0] if "best_tag" in fdf.columns else ""
        print(f"\n--- Fold {fold} (best_tag={best_tag}) ---")
        print(_format_epoch_table(fdf))

    print("\n--- Mean ± std po checkpointu (svi foldovi) ---")
    mean_df = _mean_epochs_across_folds(epoch_df)
    view = mean_df.copy()
    for c in view.columns:
        if c != "phase" and c != "epoch" and view[c].dtype.kind == "f":
            view[c] = view[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    print(view.to_string(index=False))


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"No results dir: {RESULTS_DIR}")

    cmp_rows: list[dict] = []
    all_epoch_rows: list[dict] = []
    runs_with_epochs: list[tuple[str, str, pd.DataFrame]] = []

    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        data = _load_metrics_json(run_dir)
        if data is None:
            continue

        run_id = data.get("run_id", run_dir.name)
        run_tag = data.get("run_tag", data.get("config", {}).get("run_tag", ""))
        cmp_rows.append(_comparison_row(data, run_dir))

        ep_rows = _epoch_rows_from_run(data, run_id, run_tag)
        all_epoch_rows.extend(ep_rows)
        if ep_rows:
            runs_with_epochs.append((run_id, run_tag, pd.DataFrame(ep_rows)))

    if not cmp_rows:
        raise SystemExit(f"No metrics.json under {RESULTS_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmp_df = pd.DataFrame(cmp_rows).sort_values("pr_auc_mean", ascending=False)
    cmp_path = OUT_DIR / "comparison_latest.csv"
    cmp_df.to_csv(cmp_path, index=False)
    with (OUT_DIR / "comparison_latest.json").open("w", encoding="utf-8") as f:
        json.dump({"phase": PHASE, "runs": cmp_df.to_dict(orient="records")}, f, indent=2)

    print("=" * 88)
    print("USPOREDBA RUNOVA (mean ± std, 5-fold)")
    print("=" * 88)
    print(cmp_df.to_string(index=False))
    print(f"\nWrote {cmp_path} ({len(cmp_df)} run(s))")

    if all_epoch_rows:
        ep_df = pd.DataFrame(all_epoch_rows)
        ep_path = OUT_DIR / "epochs_detail_latest.csv"
        ep_df.to_csv(ep_path, index=False)

        mean_rows = []
        for run_id, run_tag, rdf in runs_with_epochs:
            mdf = _mean_epochs_across_folds(rdf)
            mdf.insert(0, "run_tag", run_tag)
            mdf.insert(0, "run_id", run_id)
            mean_rows.append(mdf)
        mean_df = pd.concat(mean_rows, ignore_index=True) if mean_rows else pd.DataFrame()
        mean_path = OUT_DIR / "epochs_mean_latest.csv"
        if not mean_df.empty:
            mean_df.to_csv(mean_path, index=False)

        print("\n" + "#" * 88)
        print("DETALJ PO FOLDU I EPOHI")
        print("#" * 88)
        for run_id, run_tag, rdf in runs_with_epochs:
            print_epoch_report(run_id, run_tag, rdf)

        print(f"\nWrote {ep_path} ({len(ep_df)} redova)")
        if not mean_df.empty:
            print(f"Wrote {mean_path}")
    else:
        print("\nNema epoch_history u metrics.json — novi runovi će imati detalj po epohi.")


if __name__ == "__main__":
    main()
