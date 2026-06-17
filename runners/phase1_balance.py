"""Launch phase 1 balance study (all linear backbones × natural/balanced train)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
script = REPO_ROOT / "experiments" / "phase1_balance" / "run_all.py"  # noqa: E501
extra = sys.argv[1:]
cmd = [sys.executable, str(script), *extra]
raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT)))
