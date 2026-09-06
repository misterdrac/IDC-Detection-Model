# Phase 2d — extended backbone unfreezing

**Status:** **U4 completed** · **Phase 2d winner = U4** (full backbone)

**Script:** `src/cnn/convnext_5fold_ft_unfreeze.py`

> Paths: [`DOCUMENTATION.md`](DOCUMENTATION.md) § VM paths · log: `THESIS_WORKBOOK.md` §7

---

## Results table

| Run | unfreeze | PR-AUC | ROC | Acc | Prec | Rec | F1 | min | Δ PR vs FINAL | `RUN_DIR` |
|-----|----------|--------|-----|-----|------|-----|-----|-----|---------------|-----------|
| FINAL | 1 block | **0,8712** | 0,941 | 0,872 | 0,734 | 0,859 | 0,792 | 128 | — | `…korak0_img224_20260618T202030Z/` |
| U2 | 2 blocks | 0,8737 | 0,942 | 0,873 | 0,739 | 0,859 | 0,793 | 132 | +0,003 | `…phase2d_u2_unfreeze2_20260627T203828Z/` |
| **U4** | **full** (`features` 0–7) | **0,8997** | **0,954** | **0,891** | **0,785** | **0,855** | **0,817** | **228** | **+0,029** | `…phase2d_u4_unfreeze4_20260706T165305Z/` |

**PR per fold (Final ckpt):**

| Run | F0 | F1 | F2 | F3 | F4 |
|-----|-----|-----|-----|-----|-----|
| FINAL | 0,877 | 0,844 | 0,876 | 0,882 | 0,877 |
| U2 | 0,879 | 0,846 | 0,879 | **0,886** | 0,879 |
| **U4** | **0,902** | 0,871 | **0,905** | **0,917** | **0,903** |

---

## Runs

| Run | `ft_unfreeze_stages` | `run_tag` | batch / accum | Status |
|-----|----------------------|-----------|---------------|--------|
| U1 | 1 (= FINAL) | `phase2d_u1_unfreeze1` | 32 / 1 | do not run |
| **U2** | **2** | `phase2d_u2_unfreeze2` | 32 / 1 | ✓ |
| **U4** | **8** (full backbone) | `phase2d_u4_unfreeze4` | **16 / 2** | ✓ |

Rest as FINAL/U2: img 256, head 3, ft 5, `lr_head=1e-3`, `lr_backbone=2e-5`, `wd=1e-4`.

**Note:** ConvNeXt-Tiny has **8 blocks** in `model.features` (stem + stages). `ft_unfreeze_stages=8` unfreezes **all** — not just the last 4.

---

## Epoch analysis (U2, mean ± std per fold)

| checkpoint | PR-AUC | ROC | Acc | F1 |
|------------|--------|-----|-----|-----|
| Init | 0,242 | 0,422 | 0,467 | 0,305 |
| HEAD ep3 | 0,846 | 0,930 | 0,859 | 0,773 |
| FT ep1 | 0,869 | 0,940 | 0,872 | 0,789 |
| **FT ep2** | **0,872** | **0,941** | 0,871 | **0,792** |
| FT ep3 | 0,872 | 0,941 | 0,871 | 0,789 |
| FT ep4 | 0,866 | 0,939 | 0,876 | 0,794 |
| FT ep5 | 0,861 | 0,937 | 0,871 | 0,790 |
| **Final (best ckpt)** | **0,874** | **0,942** | **0,873** | **0,793** |

**Best_tag per fold (U2):** FT ep1 (F2), FT ep2 (F0, F3, F4), FT ep3 (F1).

---

## Epoch analysis (U4, mean ± std per fold)

| checkpoint | PR-AUC | ROC | Acc | F1 |
|------------|--------|-----|-----|-----|
| Init | 0,241 | 0,422 | 0,467 | 0,305 |
| HEAD ep3 | 0,845 | 0,933 | 0,857 | 0,771 |
| FT ep1 | **0,897** | 0,953 | 0,891 | 0,814 |
| FT ep2 | **0,898** | 0,955 | 0,894 | 0,819 |
| FT ep3 | 0,894 | 0,953 | 0,889 | 0,816 |
| FT ep4 | 0,889 | 0,949 | 0,892 | 0,815 |
| FT ep5 | 0,879 | 0,947 | 0,889 | 0,811 |
| **Final (best ckpt)** | **0,900** | **0,954** | **0,891** | **0,817** |

**Best_tag per fold (U4):** FT ep1 (F0, F1, F2), FT ep2 (F3), FT ep3 (F4).

**U4 conclusion:** unfreezing the full backbone yields **+0.026 PR-AUC** vs U2 and **+0.029** vs FINAL. Best checkpoints are mostly in **FT ep1–ep2**; later epochs (ep4–ep5) show PR drop on most folds → signal of mild overfitting. Fold 1 remains weakest (0.871), but notably better than U2 (0.846).

---

## Running U4 (VM) — completed 2026-07-06

```bash
cd ~/Desktop/IDC_Detection_Model/IDC_Gitrepo/IDC-Detection-Model
source .venv/bin/activate
export IDC_DATASET_PATH="$HOME/Desktop/IDC_Detection_Model/IDC_Dataset/IDC_regular_ps50_idx5"

# sync CFG with repo (git pull) — ft_unfreeze_stages=8, batch 16, accum 2

tmux new -s p2d_u4
python3 src/cnn/convnext_5fold_ft_unfreeze.py
python3 experiments/phase2_deep_ft/aggregate_results.py
```

**OOM:** script automatically reduces batch (min 8). If it still fails, manually in CFG: `batch_size=8`, `grad_accum_steps=4`.

---

## Where results live

```bash
RUN_DIR=$(ls -td experiments/results/phase2_deep_ft/*phase2d_u4_unfreeze4* | head -1)
echo "$RUN_DIR"
cat "$RUN_DIR/metrics.json"
```

```text
RUN_DIR/metrics.json
RUN_DIR/metrics.csv
RUN_DIR/checkpoints/
reports/experiments/phase2_deep_ft/comparison_latest.csv
```

---

*Previous phase: `PHASE2C.md` · **best Phase 2 model: U4** (PR **0.900 ± 0.015**, ROC **0.954**) — use for thesis and further comparisons*
