"""Linear VM runner: ConvNeXt-Tiny."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from linear_vm.shared_config import vm_cli_args

cmd = [
    sys.executable,
    str(ROOT / "model_gpu_embedding_CNN_changer.py"),
    "--backbone",
    "convnext_tiny",
    *vm_cli_args(),
]
raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
