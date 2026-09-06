# Phase 2c — exponential epoch scaling

**Status:** **completed** — H1 done, H2 skipped (2026-06-27)  
**Script:** `src/cnn/convnext_5fold_ft.py`

> Log: `THESIS_WORKBOOK.md` §7 · **Paths:** [`DOCUMENTATION.md`](DOCUMENTATION.md) § Paths

---

## 1. Research question

In Phase 2a the schedule grew **linearly** (head 2→3, ft 1→3→5) and yielded a small gain at the end. Phase **2c** asks: if we **exponentially** increase head and fine-tuning epochs (e.g. doubling), does PR-AUC grow proportionally, do we hit a **plateau** or **overfitting**, and what is the time cost?

---

## 2. Reference (entry into 2c)

Take CFG from the **2b** winner (batch, LR). Typical starting point if 2b did not move metrics:

| Parameter | Value (2a FINAL) |
|-----------|------------------|
| `image_size` | 256 |
| `head_epochs` | 3 |
| `ft_epochs` | 5 |
| `lr_head` / `lr_backbone` | 1e-3 / 2e-5 |
| `batch_size` | 32 (or 2b winner) |
| PR-AUC | **0.871** |

**Starting-point label:** **H0** = (head 3, ft 5).

---

## 3. Protocol — doubling epochs

Each run **doubles** both head and ft epochs relative to the previous step in the chain:

| Run | `head_epochs` | `ft_epochs` | Ratio vs H0 | `run_tag` (template) |
|-----|---------------|-------------|-------------|----------------------|
| H0 | 3 | 5 | ×1 | *(reference, do not run)* |
| **H1** | **6** | **10** | ×2 | `phase2c_h1_h6_ft10` |
| **H2** | **12** | **20** | ×4 | `phase2c_h2_h12_ft20` |
| H3 *(opt.)* | 24 | 40 | ×8 | `phase2c_h3_h24_ft40` |

**Rule:** one run = one table row; all other hyperparameters **fixed** from 2b.

**Chain stop** *(H1 2026-06-27):* PR **−0.002** vs H0, no gain → **H2 skipped**. Next: **Phase 2d** (wider unfreezing) or keep **FINAL** for the thesis.

---

## 4. What to measure (besides final metrics)

- **PR-AUC / ROC-AUC** (mean ± std, 5-fold) — primary
- **Runtime** (min) — trade-off curve
- **`best_tag` per fold** — which FT epoch is best (overfitting signal)
- **Per-epoch evolution** — stdout log or §6.6.6 in workbook (Init → HEAD → FT)

---

## 5. Results table

```text
experiments/results/phase2_deep_ft/phase2_deep_ft_convnext_tiny_<run_tag>_<TIMESTAMP>Z/metrics.json
```

| Run | head | ft | PR-AUC | ROC | Acc | min | Δ PR vs H0 | `RUN_DIR` |
|-----|------|-----|--------|-----|-----|-----|------------|-----------|
| H0 | 3 | 5 | 0,871 | 0,941 | 0,872 | 128 | — | `…/korak0_img224_20260618T202030Z/` |
| H1 | 6 | 10 | **0,8692** | 0,940 | 0,868 | 129 | **−0,002** | `…phase2b_h1_h6_ft10_20260627T102339Z/` * |
| H1@224 | 6 | 10 | **0,8690** | 0,940 | 0,870 | **208** | **−0,002** | `…phase2b_h1_h6_ft10_img224_20260627T154528Z/` |
| H2 | 12 | 20 | — | — | — | — | — | **skipped** → 2d |

\* H1@256: `run_tag` `phase2b_h1_h6_ft10`.

**PR per fold (Final ckpt):**

| Run | F0 | F1 | F2 | F3 | F4 |
|-----|-----|-----|-----|-----|-----|
| H0 (FINAL) | 0,877 | 0,844 | 0,876 | 0,882 | 0,877 |
| H1@256 | 0,875 | 0,837 | 0,879 | 0,879 | 0,876 |
| H1@224 | 0,874 | 0,838 | 0,879 | 0,878 | 0,877 |

```bash
RUN_DIR=$(ls -td experiments/results/phase2_deep_ft/*phase2b_h1_h6_ft10_img224* | head -1)
echo "$RUN_DIR" && cat "$RUN_DIR/metrics.json"
```

---

## 6. CFG example — H1

```python
image_size = 256          # from 2b winner
head_epochs = 6
ft_epochs = 10
finetune_last_stage = True
lr_head = 1e-3            # from 2b
lr_backbone = 2e-5
weight_decay = 1e-4
batch_size = 32           # from 2b
grad_accum_steps = 1
run_tag = "phase2c_h1_h6_ft10"
```

---

## 7. Thesis text (draft)

> After optimizing batch size and learning rate (Phase 2b), exponential scaling of training duration was tested: head and fine-tuning epochs were doubled relative to the previous configuration (3/5 → 6/10 → 12/20). The goal was to estimate the lower bound of gain from longer training versus compute cost and overfitting risk on a patient-level split.

---

*Map: `DOCUMENTATION.md` · previous phase: `PHASE2B.md`*
