#!/usr/bin/env bash
# ConvNeXt-Tiny linear track — image size sweep (natural train, class_weight=none)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export IDC_DATASET_PATH="${IDC_DATASET_PATH:-$HOME/Desktop/IDC_Detection_Model/IDC_Dataset/IDC_regular_ps50_idx5}"
python3 runners/linear_img_size.py "$@"
