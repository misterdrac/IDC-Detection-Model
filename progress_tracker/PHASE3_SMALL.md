# Phase 3 — ConvNeXt-Small (full backbone FT)

**Status:** **completed (2026-07-06)** · **best overall model = ConvNeXt-Small**

**Script:** `src/cnn/convnext_small_5fold_ft_unfreeze.py`

**Reference (Tiny):** U4 PR **0.900** · `PHASE2D.md`

---

## Goal

Same protocol as U4 (Tiny), but on **ConvNeXt-Small** — larger architecture capacity with the same patient-level 5-fold CV, to probe the upper bound of performance.

---

## CFG (default in script)

| Parameter | Value | Note |
|-----------|-------|------|
| `image_size` | 256 | as U4 |
| `head_epochs` / `ft_epochs` | 3 / 5 | as U4 |
| `ft_unfreeze_stages` | **8** | full backbone (Small = 8 blocks) |
| `batch_size` | **8** | smaller than Tiny U4 (16) due to VRAM |
| `grad_accum_steps` | **4** | effective batch **32** |
| `min_batch_size` | 4 | auto-reduce on OOM |
| `val_batch_size` | 32 | eval without OOM |
| `lr_head` / `lr_backbone` | 1e-3 / 2e-5 | as U4 |
| `run_tag` | `small_full_ft_u4` | |

---

## Results (completed run)

**RUN_DIR:** `…phase3_convnext_small_convnext_small_small_full_ft_u4_20260706T214118Z/`

| Metric | ConvNeXt-Small | Tiny U4 (reference) | Δ Small - Tiny |
|--------|----------------|---------------------|----------------|
| PR-AUC | **0,9027 ± 0,0147** | 0,8997 ± 0,0153 | **+0,0030** |
| ROC-AUC | **0,9563 ± 0,0075** | 0,9543 ± 0,0089 | **+0,0020** |
| Accuracy | 0,8828 ± 0,0133 | **0,8915 ± 0,0113** | −0,0087 |
| Balanced accuracy | **0,8889 ± 0,0113** | 0,8806 ± 0,0164 | +0,0083 |
| Precision | 0,7425 ± 0,0318 | **0,7852 ± 0,0323** | −0,0427 |
| Recall | **0,9029 ± 0,0236** | 0,8554 ± 0,0459 | +0,0475 |
| F1 | 0,8142 ± 0,0174 | **0,8173 ± 0,0180** | −0,0031 |
| Runtime | **541,9 min** | 228,4 min | **+313,5 min (~2.4x)** |

**PR per fold (Small):** 0,900 · 0,876 · 0,914 · 0,916 · 0,908

**best_tag per fold (Small):** FT ep1, FT ep2, FT ep4, FT ep3, FT ep1

---

## Running (VM)

```bash
cd ~/Desktop/IDC_Detection_Model/IDC_Gitrepo/IDC-Detection-Model
source .venv/bin/activate
export IDC_DATASET_PATH="$HOME/Desktop/IDC_Detection_Model/IDC_Dataset/IDC_regular_ps50_idx5"

git pull   # pull new script

tmux new -s p3_small
python3 src/cnn/convnext_small_5fold_ft_unfreeze.py
python3 experiments/phase3_convnext_small/aggregate_results.py
```

**OOM:** script reduces batch down to `min_batch_size=4`. If it still fails:

```python
batch_size = 4
grad_accum_steps = 8   # effective batch 32
val_batch_size = 16
```

---

## Where results live

```text
experiments/results/phase3_convnext_small/phase3_convnext_small_convnext_small_small_full_ft_u4_<TIMESTAMP>Z/
  metrics.json
  convnext_small_5fold_img256.csv
  checkpoints/
```

```bash
RUN_DIR=$(ls -td experiments/results/phase3_convnext_small/*small_full_ft_u4* | head -1)
cat "$RUN_DIR/metrics.json" | python3 -c "import json,sys; s=json.load(sys.stdin); m=s['metrics_mean_std']; print('PR:', round(m['pr_auc']['mean'],4), '±', round(m['pr_auc']['std'],4))"
```

---

## Discussion and conclusion

1. **Small is the new SOTA in this project by PR-AUC.** Gain exists (+0.003), but is modest.
2. **Gain comes mainly from recall** (0.903 vs 0.855), with lower precision; balanced accuracy rises, but accuracy and F1 do not.
3. **Time is the main cost**: ~542 min vs ~228 min for Tiny U4 (about 2.4x slower).
4. **No indication this leads toward PR 0.95** without further changes (augmentation, ensembling, different protocol).
5. **Thesis recommendation:**
   - if the priority is the best number: **ConvNeXt-Small (PR 0.903)**;
   - if the priority is efficiency and simplicity: **Tiny U4 (PR 0.900)**.

---

## Comparison with Tiny U4 (final)

| Model | PR-AUC (mean) | Note |
|-------|---------------|------|
| ConvNeXt-Tiny U4 | 0,8997 | faster, better precision/F1 |
| **ConvNeXt-Small** | **0,9027** | best PR-AUC, slower |

---

*Previous phase: `PHASE2D.md` (Tiny U4)*
