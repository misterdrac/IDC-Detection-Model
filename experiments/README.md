# Experiments

Study orchestration (phase 1 balance, linear resolution, phase 2 deep FT, phase 3 ConvNeXt-Small).

**Document map:** [`../progress_tracker/DOCUMENTATION.md`](../progress_tracker/DOCUMENTATION.md)

| Phase | Folder | Notes / CFG |
|-------|--------|-------------|
| 1 | [`phase1_balance/`](phase1_balance/) | natural vs balanced |
| 1.5 | [`linear_img_size/`](linear_img_size/) | 128 / 224 / 256 |
| 2a–2d | [`phase2_deep_ft/`](phase2_deep_ft/) | [`PHASE2B.md`](../progress_tracker/PHASE2B.md) · [`PHASE2C.md`](../progress_tracker/PHASE2C.md) · [`PHASE2D.md`](../progress_tracker/PHASE2D.md) |
| 3 | [`phase3_convnext_small/`](phase3_convnext_small/) | [`PHASE3_SMALL.md`](../progress_tracker/PHASE3_SMALL.md) |

```bash
python3 experiments/<phase>/aggregate_results.py
```

Outputs: `experiments/results/<phase>/<run_id>/` (gitignored) and `reports/experiments/<phase>/comparison_latest.csv` (gitignored; VM only).
