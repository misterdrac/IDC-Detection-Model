"""Linear VM runner: MobileNetV2 (hyperparameters from runners/linear_vm/shared_config.py)."""
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
    "mobilenetv2",
    *vm_cli_args(),
]
raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
