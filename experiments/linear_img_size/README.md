# Linear image-size sweep (end of linear track)

**Model:** ConvNeXt-Tiny (Phase 1 winner)  
**Fixed:** `train_balance=natural`, `class_weight=none`, GPU linear head, 5-fold patient CV

**Varies:** `image_size` only (default **128** and **224**).

## Run on VM

```bash
source .venv/bin/activate
export IDC_DATASET_PATH="$HOME/Desktop/IDC_Detection_Model/IDC_Dataset/IDC_regular_ps50_idx5"

# Both sizes (128 reuses cache if phase 1 already ran)
python3 runners/linear_img_size.py

# Only 224 (faster if 128 is already in phase1 comparison)
python3 runners/linear_img_size.py --image-size 224

# Dry-run
python3 runners/linear_img_size.py --dry-run
```

After runs:

```bash
python3 experiments/linear_img_size/aggregate_results.py
```

## Outputs

| Path | Contents |
|------|----------|
| `experiments/results/linear_img_size/<run_id>/` | stdout.log, metrics.json |
| `reports/experiments/linear_img_size/comparison_latest.csv` | summary for thesis |

## Notes

- **128×128** baseline PR-AUC from Phase 1: **0.8267** (same protocol).
- **224×224** needs new embedding cache (~2–4× slower first run than 128).
- Batch size auto-reduced at 224 (128) to fit 16 GB VRAM.
