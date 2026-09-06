# Phase 2 — deep fine-tuning (ConvNeXt-Tiny)

**Status:** completed (2a–2d) · winner **U4** PR **0.900**

**Notes:** [`PHASE2B.md`](../../progress_tracker/PHASE2B.md) · [`PHASE2C.md`](../../progress_tracker/PHASE2C.md) · [`PHASE2D.md`](../../progress_tracker/PHASE2D.md)  
**VM paths:** [`DOCUMENTATION.md`](../../progress_tracker/DOCUMENTATION.md) § Paths

```bash
# 2a–2c (last-stage FT)
python3 src/cnn/convnext_5fold_ft.py

# 2d (extended / full unfreeze)
python3 src/cnn/convnext_5fold_ft_unfreeze.py

python3 experiments/phase2_deep_ft/aggregate_results.py
```

**One run → folder:**

```text
experiments/results/phase2_deep_ft/phase2_deep_ft_convnext_tiny_<run_tag>_<TIMESTAMP>Z/
  metrics.json    ← paste for analysis
  metrics.csv
  checkpoints/
```

**Aggregate:** `reports/experiments/phase2_deep_ft/comparison_latest.csv`
