#!/usr/bin/env bash
# Run all linear backbone jobs sequentially (identical flags from linear_vm/shared_config.py).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

python3 linear_vm/mobilenetv2.py
python3 linear_vm/resnet18.py
python3 linear_vm/efficientnet_b0.py
python3 linear_vm/convnext_tiny.py
python3 linear_vm/resnet_50.py
python3 linear_vm/inception_v3.py

echo "All linear_vm runners finished."
