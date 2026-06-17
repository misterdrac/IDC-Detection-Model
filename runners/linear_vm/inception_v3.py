"""Linear VM runner: Inception v3 (same image size / batch / subset as other VM runners)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runners.linear_vm.shared_config import vm_cli_args

cmd = [
    sys.executable,
    str(ROOT / "src/linear/embedding_svc.py"),
    "--backbone",
    "inception_v3",
    *vm_cli_args(),
]
raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
