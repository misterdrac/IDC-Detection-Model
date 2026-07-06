"""Launch ConvNeXt-Small 5-fold fine-tune (phase 3)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
script = REPO_ROOT / "src" / "cnn" / "convnext_small_5fold_ft_unfreeze.py"
raise SystemExit(subprocess.call([sys.executable, str(script)], cwd=str(REPO_ROOT)))
