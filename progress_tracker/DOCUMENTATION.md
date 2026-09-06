# Documentation — what to open where

**Problem:** many `.md` files and cryptic run tags.  
**Solution:** one CFG / notes file per phase + `experiments/` for running and aggregation.

---

## Paths on the VM (krk) — copy-paste

**Repo (working directory):**

```text
~/Desktop/IDC_Detection_Model/IDC_Gitrepo/IDC-Detection-Model
```

**Dataset:**

```text
~/Desktop/IDC_Detection_Model/IDC_Dataset/IDC_regular_ps50_idx5
```

### One CNN fine-tune run (2a / 2b / 2c / 2d)

Each run creates a **new folder** (timestamp at the end):

```text
experiments/results/phase2_deep_ft/phase2_deep_ft_convnext_tiny_<run_tag>_<YYYYMMDD>T<HHMMSS>Z/
```

| File in `RUN_DIR/` | Purpose |
|--------------------|---------|
| **`metrics.json`** | **main** — CFG + all folds + epoch_history → paste into chat |
| `metrics.csv` | 5 rows, one per fold (final metrics) |
| `summary.csv` | mean PR/ROC/Acc only |
| `checkpoints/` | `.pth` per fold |
| `convnext_tiny_5fold_img256.csv` | same as metrics.csv |

**Phase 3 (ConvNeXt-Small):**

```text
experiments/results/phase3_convnext_small/phase3_convnext_small_convnext_small_<run_tag>_<YYYYMMDD>T<HHMMSS>Z/
```

**Aggregate (after all runs in a phase):**

```text
reports/experiments/phase2_deep_ft/comparison_latest.csv
reports/experiments/phase2_deep_ft/epochs_detail_latest.csv
reports/experiments/phase2_deep_ft/epochs_mean_latest.csv
reports/experiments/phase3_convnext_small/comparison_latest.csv
```

> Note: `experiments/results/` and `comparison_latest.*` are in `.gitignore` — they stay on the VM, not on GitHub.

**Manifest (audit log):**

```text
experiments/manifest.jsonl
```

### Commands — find the latest run

```bash
cd ~/Desktop/IDC_Detection_Model/IDC_Gitrepo/IDC-Detection-Model

# latest Tiny run (2a–2d)
ls -td experiments/results/phase2_deep_ft/*/ | head -1

# latest Small run (phase 3)
ls -td experiments/results/phase3_convnext_small/*/ | head -1

# latest run with a given run_tag (e.g. U4)
ls -td experiments/results/phase2_deep_ft/*phase2d_u4_unfreeze4* | head -1

# print path + mean metrics
RUN_DIR=$(ls -td experiments/results/phase2_deep_ft/*phase2d_u4_unfreeze4* | head -1)
echo "RUN_DIR=$RUN_DIR"
python3 -c "import json; d=json.load(open('$RUN_DIR/metrics.json')); print(json.dumps(d['metrics_mean_std'], indent=2))"
```

### Known folders (key runs)

| Run | `run_tag` | `RUN_DIR` (relative to repo) | PR-AUC |
|-----|-----------|------------------------------|--------|
| FINAL 2a | `korak0_img224` * | `…/phase2_deep_ft_convnext_tiny_korak0_img224_20260618T202030Z/` | **0.871** |
| U4 (Tiny) | `phase2d_u4_unfreeze4` | `…/phase2d_u4_unfreeze4_20260706T165305Z/` | **0.900** |
| Small | `small_full_ft_u4` | `…/phase3_…_small_full_ft_u4_20260706T214118Z/` | **0.903** |

\* On the VM the CFG `run_tag` was often wrong — folder timestamp `20260618T202030Z` = **img256, head3, ft5, PR 0.871**.

### tmux + SSH

A run in `tmux` writes to disk under `RUN_DIR/` — **an SSH drop does not delete the result** if the run finished.  
Live log: `tmux attach -t p2b_f2` or `tmux ls`.

---

## What to open first

| What you need | Open |
|---------------|------|
| **Document map** (this file) | [`DOCUMENTATION.md`](DOCUMENTATION.md) |
| **Experiment orchestration** | [`experiments/README.md`](../experiments/README.md) |
| **CFG + results 2b** | [`PHASE2B.md`](PHASE2B.md) |
| **CFG + results 2c** | [`PHASE2C.md`](PHASE2C.md) |
| **CFG + results 2d (Tiny U4)** | [`PHASE2D.md`](PHASE2D.md) |
| **CFG + results phase 3 (Small)** | [`PHASE3_SMALL.md`](PHASE3_SMALL.md) |
| **Local workbook** (gitignored) | `THESIS_WORKBOOK.md` §7 |
| **Phase 1 / 2a archive** (gitignored) | `PHASE1_SUMMARY.md`, `PHASE2_FT_GUIDE.md` |

---

## Project phases

| Phase | What | Status | CFG / notes | Experiments |
|-------|------|--------|-------------|-------------|
| **1** | backbones, natural vs balanced | ✓ | `PHASE1_SUMMARY.md` *(local)* | [`experiments/phase1_balance/`](../experiments/phase1_balance/) |
| **1.5** | resolution on linear head | ✓ | workbook §7 *(local)* | [`experiments/linear_img_size/`](../experiments/linear_img_size/) |
| **2a** | resolution, epochs, LR/WD/pw | ✓ PR **0.871** | `PHASE2_FT_GUIDE.md` *(local)* | [`experiments/phase2_deep_ft/`](../experiments/phase2_deep_ft/) |
| **2b** | batch, wider LR | ✓ no gain vs FINAL | **[`PHASE2B.md`](PHASE2B.md)** | same `phase2_deep_ft` |
| **2c** | exponential epochs | ✓ H1 no gain, H2 skipped | **[`PHASE2C.md`](PHASE2C.md)** | same `phase2_deep_ft` |
| **2d** | broader backbone unfreezing | ✓ **U4 PR 0.900** | **[`PHASE2D.md`](PHASE2D.md)** | same `phase2_deep_ft` |
| **3** | ConvNeXt-Small, same protocol as U4 | ✓ **PR 0.903** | **[`PHASE3_SMALL.md`](PHASE3_SMALL.md)** | [`experiments/phase3_convnext_small/`](../experiments/phase3_convnext_small/) |

---

## Experiments — how to run

Overview: [`experiments/README.md`](../experiments/README.md)

| Phase | Train script | Aggregate |
|-------|--------------|-----------|
| 2a–2c | `src/cnn/convnext_5fold_ft.py` | `experiments/phase2_deep_ft/aggregate_results.py` |
| 2d | `src/cnn/convnext_5fold_ft_unfreeze.py` | `experiments/phase2_deep_ft/aggregate_results.py` |
| 3 | `src/cnn/convnext_small_5fold_ft_unfreeze.py` | `experiments/phase3_convnext_small/aggregate_results.py` |

```bash
python3 src/cnn/convnext_5fold_ft.py
python3 experiments/phase2_deep_ft/aggregate_results.py

python3 src/cnn/convnext_5fold_ft_unfreeze.py
python3 experiments/phase2_deep_ft/aggregate_results.py

python3 src/cnn/convnext_small_5fold_ft_unfreeze.py
python3 experiments/phase3_convnext_small/aggregate_results.py
```

---

## Logging rule (after each run)

1. **Matching PHASE*.md** — numbers in that phase’s master table (`PHASE2B`, `PHASE2C`, `PHASE2D`, `PHASE3_SMALL`)
2. **`THESIS_WORKBOOK.md` §7** *(local)* — paragraph: *What we did* → *Evidence* → *Conclusion*

**Do not duplicate** long prose across files — PHASE files are the source of numbers; the workbook is the diary.

---

## Other documents

| File | Role | On GitHub? |
|------|------|------------|
| [`PHASE2B.md`](PHASE2B.md) … [`PHASE3_SMALL.md`](PHASE3_SMALL.md) | CFG + result tables per phase | yes |
| [`experiments/`](../experiments/) | run / aggregate scripts per phase | yes |
| `THESIS_WORKBOOK.md` | Work diary | **no** (`.gitignore`) |
| `PHASE2_FT_GUIDE.md` | Phase 2a archive | **no** (`.gitignore`) |
| `PHASE1_SUMMARY.md` | Phase 1 archive | **no** (`.gitignore`) |

---

## Quick link — where to paste results

After a run, send in chat:

1. Output of `echo $RUN_DIR`
2. Contents of **`$RUN_DIR/metrics.json`** (or at least `metrics_mean_std` + `fold_rows`)

See **§ Paths on the VM** above for `ls -td` commands by `run_tag`.
