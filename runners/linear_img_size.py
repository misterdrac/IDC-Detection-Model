"""Launch linear image-size sweep (ConvNeXt-Tiny, natural train)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
script = REPO_ROOT / "experiments" / "linear_img_size" / "run_all.py"
cmd = [sys.executable, str(script), *sys.argv[1:]]
raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT)))
