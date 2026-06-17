# Reports

Committed experiment exports — baseline reference for comparing new phase 1 / phase 2 runs.

## Layout

```
reports/
├── baselines/                    # completed VM runs (reference)
│   ├── linear_vm_img128_subset100k_krk/
│   │   ├── comparison.csv        # all backbones
│   │   └── <backbone>/
│   │       ├── metrics.json
│   │       ├── metrics.csv
│   │       └── run.log.txt
│   └── cnn_convnext_tiny_5fold_img128_krk/
│       └── metrics.csv
├── experiments/                  # copy phase 1 summaries here after VM runs
│   └── phase1_balance/
│       └── comparison_latest.csv
└── archive/
    └── legacy_scripts/           # old runners (reference only)
```

## Baselines (img128, subset 100k, VM krk)

Linear track used natural data + `class_weight=balanced` (legacy setup). See `baselines/linear_vm_img128_subset100k_krk/comparison.csv`.

CNN baseline: light ConvNeXt-Tiny fine-tune (`baselines/cnn_convnext_tiny_5fold_img128_krk/metrics.csv`).

## After new runs

1. Full logs stay on VM: `experiments/results/` (gitignored).
2. Copy comparison CSV to `reports/experiments/phase1_balance/comparison_latest.csv` when ready for git / thesis.
