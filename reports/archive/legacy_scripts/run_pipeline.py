"""
Run patient-level 5-fold split, then training/eval scripts, from repo root.

Reference copy in reports/archive/legacy_scripts/ (2026-06-17).
Active entry point: runners/phase1_balance.py

Usage (from repo root, if restored to runners/):
  python runners/run_pipeline.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def run_step(script: str, cwd: Path) -> None:
    cmd = [sys.executable, script]
    print(f"\n{'=' * 60}\n>>> {' '.join(cmd)}\n{'=' * 60}\n", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Split CSV then linear + CNN scripts.")
    p.add_argument("--skip-split", action="store_true")
    p.add_argument("--with-baseline-embedding", action="store_true")
    p.add_argument("--only-split", action="store_true")
    p.add_argument("--only-linear", action="store_true")
    p.add_argument("--only-cnn", action="store_true")
    args = p.parse_args()

    only_flags = [args.only_split, args.only_linear, args.only_cnn]
    if sum(only_flags) > 1:
        p.error("Use at most one of --only-split, --only-linear, --only-cnn.")

    baseline_script = REPO_ROOT / "reports" / "archive" / "legacy_scripts" / "model_gpu_embbedding.py"

    skip_split = args.skip_split or args.only_linear or args.only_cnn

    if args.only_split:
        if args.skip_split:
            print(
                "Nothing to run: --only-split conflicts with --skip-split.",
                file=sys.stderr,
            )
            sys.exit(1)
        run_step("src/data/split.py", REPO_ROOT)
        return

    if not skip_split:
        run_step("src/data/split.py", REPO_ROOT)

    if args.only_linear:
        run_step("src/linear/embedding_svc.py", REPO_ROOT)
        return

    if args.only_cnn:
        run_step("src/cnn/convnext_5fold_ft.py", REPO_ROOT)
        return

    run_step("src/linear/embedding_svc.py", REPO_ROOT)
    run_step("src/cnn/convnext_5fold_ft.py", REPO_ROOT)

    if args.with_baseline_embedding and baseline_script.is_file():
        run_step("model_gpu_embbedding.py", REPO_ROOT)


if __name__ == "__main__":
    main()
