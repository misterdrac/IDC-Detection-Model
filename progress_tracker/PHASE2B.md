# Phase 2b — batch and learning rate

**Start date:** 18 June 2026.  
**Base:** FINAL from Phase 2a (PR-AUC **0.871**, ROC **0.941**)  
**Script:** `src/cnn/convnext_5fold_ft.py` (CFG at top, no CLI)

> Single document for all CFG values in this phase. **Log:** `THESIS_WORKBOOK.md` §7.  
> **VM paths:** [`DOCUMENTATION.md`](DOCUMENTATION.md) § Paths.

---

## 1. What we are doing and why

Phase **2a** tested resolution, epoch schedule, and **small** LR/WD/pw changes → plateau **~0.871**.

Phase **2b** tests **larger** changes we did not touch in 2a:

1. **`batch_size`** (+ `grad_accum_steps`)
2. **Wider LR** (`lr_head`, `lr_backbone`)

Everything else stays as in the FINAL run.

---

## 2. Reference — FINAL (2a, do not change in 2b unless explicitly stated)

| Parameter | Value |
|-----------|-------|
| `image_size` | **256** |
| `head_epochs` | **3** |
| `ft_epochs` | **5** |
| `finetune_last_stage` | **True** |
| `lr_head` | **1e-3** |
| `lr_backbone` | **2e-5** |
| `weight_decay` | **1e-4** |
| `pos_weight` | **auto** (~2.5, from class ratio in train fold) |
| `batch_size` | **32** |
| `grad_accum_steps` | **1** |
| `threshold` | 0.5 |
| `run_tag` (2a) | `korak0_img224` (CFG typo) |

**RUN_DIR (2a FINAL, known):**

```text
experiments/results/phase2_deep_ft/phase2_deep_ft_convnext_tiny_korak0_img224_20260618T202030Z/
```

**FINAL result:** PR **0.871 ± 0.014** · ROC **0.941** · Acc **0.872** · Prec **0.734** · Rec **0.859** · F1 **0.792** · ~128 min

---

## 3. Run order

```
FINAL (2a)  →  F2  →  [F3 skipped]  →  F4  →  FINAL base  →  G2  →  G3  →  G4
```

**Block F completed (2026-06-26):** F2 and F4 with no gain vs FINAL → **base for Block G = FINAL** (batch 32).

**Rule:** one run = **one** change + new **`run_tag`**.

---

## 4. Block F — batch *(completed)*

| Run | What we change | `batch_size` | `grad_accum_steps` | Effective batch | `run_tag` | Status |
|-----|----------------|--------------|--------------------|-----------------|-----------|--------|
| — | FINAL (reference) | 32 | 1 | 32 | `…202030Z` | reference |
| **F2** | smaller batch | **16** | 1 | 16 | `phase2b_f2_batch16` | ✓ PR −0.001 |
| F3 | even smaller | **8** | 1 | 8 | `phase2b_f3_batch8` | skipped |
| F4 | same batch, accumulation | 16 | **2** | 32 | `phase2b_f4_accum2` | ✓ PR −0.001 |

### Full CFG for each F run

All columns except those listed = as FINAL (§2).

#### F2 — completed

```python
batch_size = 16
grad_accum_steps = 1
run_tag = "phase2b_f2_batch16"
# rest as FINAL
```

#### F3 — skipped

#### F4 — completed

```python
batch_size = 16
grad_accum_steps = 2
run_tag = "phase2b_f4_accum2"
# rest as FINAL
```

---

## 5. Block G — learning rate *(completed)*

Base = **FINAL** (batch 32, accum 1). **Winner = FINAL** — no G run reliably beat it.

| Run | `lr_head` | `lr_backbone` | `run_tag` | PR vs FINAL | Status |
|-----|-----------|---------------|-----------|-------------|--------|
| G1 | 1e-3 | 2e-5 | = FINAL | — | reference |
| G2 | 5e-4 | 2e-5 | `phase2b_g2_lrhead5e4` | ≈0 | ✓ |
| G3 | 1e-3 | 5e-6 | `phase2b_g3_lrbb5e6` | **−0.002** | ✓ |
| G4 | 5e-4 | 5e-6 | `phase2b_g4_lr_conserv` | — | **skipped** → 2c |

### Full CFG for G runs

For G2/G3/G4 use **batch 32, accum 1** (FINAL).

#### G4 — skipped (see §10)

Combination of G2+G3 conservative LRs; both individually with no gain → low chance. Next: **Phase 2c**.

#### G3 — completed

```python
lr_head = 1e-3
lr_backbone = 5e-6
run_tag = "phase2b_g3_lrbb5e6"
```

#### G2 — completed

```python
lr_head = 5e-4
lr_backbone = 2e-5
run_tag = "phase2b_g2_lrhead5e4"
```

---

## 6. Results table (fill in after each run)

**Path template** (new run — timestamp changes):

```text
experiments/results/phase2_deep_ft/phase2_deep_ft_convnext_tiny_<run_tag>_<TIMESTAMP>Z/metrics.json
```

| Run | `run_tag` | batch | accum | lr_h | lr_bb | PR-AUC | ROC | Acc | Prec | Rec | F1 | min | Δ PR vs FINAL | `RUN_DIR` (fill in after run) |
|-----|-----------|-------|-------|------|-------|--------|-----|-----|------|-----|-----|-----|---------------|------------------------------|
| FINAL | `korak0_img224`* | 32 | 1 | 1e-3 | 2e-5 | **0,871** | 0,941 | 0,872 | 0,734 | 0,859 | 0,792 | 128 | — | `…/korak0_img224_20260618T202030Z/` |
| F2 | `phase2b_f2_batch16` | 16 | 1 | 1e-3 | 2e-5 | **0,8705** | 0,941 | 0,871 | 0,742 | 0,844 | 0,788 | 159 | **−0,001** | `…phase2b_f2_batch16_20260626T080222Z/` |
| F3 | `phase2b_f3_batch8` | 8 | 1 | 1e-3 | 2e-5 | — | — | — | — | — | — | — | — | **skipped** |
| F4 | `phase2b_f4_accum2` | 16 | 2 | 1e-3 | 2e-5 | **0,8699** | 0,941 | 0,870 | 0,732 | 0,858 | 0,789 | 155 | **−0,001** | `…phase2b_f4_accum2_20260626T111027Z/` |
| G? | `phase2b_g2_lrhead5e4`† | 32 | 1 | **1e-5** | 2e-5 | **0,8722** | 0,941 | 0,873 | 0,740 | 0,855 | 0,793 | 129 | **+0,001** | `…g2_lrhead5e4_20260626T140024Z/` |
| G2 | `phase2b_g2_lrhead5e4` | 32 | 1 | **5e-4** | 2e-5 | **0,8715** | 0,941 | 0,871 | 0,732 | 0,861 | 0,791 | 129 | **≈0** | `…g2_lrhead5e4_20260626T161501Z/` |
| G3 | `phase2b_g3_lrbb5e6` | 32 | 1 | 1e-3 | **5e-6** | **0,8688** | 0,940 | 0,871 | 0,734 | 0,857 | 0,790 | 128 | **−0,002** | `…g3_lrbb5e6_20260627T075823Z/` |
| G4 | `phase2b_g4_lr_conserv` | 32 | 1 | 5e-4 | 5e-6 | — | — | — | — | — | — | — | — | **skipped → 2c** |

\* FINAL folder name contains `korak0_img224` although this is an img256 run.

† **G?** — off-plan typo: `lr_head=1e-5` (not planned G2). Same `run_tag`; distinguish by `lr_head` in `metrics.json`.

**PR per fold:**

| Run | F0 | F1 | F2 | F3 | F4 |
|-----|-----|-----|-----|-----|-----|
| FINAL | 0,877 | 0,844 | 0,876 | 0,882 | 0,877 |
| G3 | 0,875 | 0,841 | 0,876 | 0,876 | 0,875 |
| G2 | 0,878 | 0,844 | 0,876 | 0,883 | 0,877 |
| G? | 0,877 | 0,845 | 0,878 | **0,884** | 0,877 |
| F2 | 0,878 | 0,842 | 0,873 | 0,883 | 0,876 |
| F4 | 0,877 | 0,841 | 0,873 | 0,882 | 0,876 |

---

## 7. Running on the VM

```bash
export IDC_DATASET_PATH="$HOME/Desktop/IDC_Detection_Model/IDC_Dataset/IDC_regular_ps50_idx5"
cd ~/Desktop/IDC_Detection_Model/IDC_Gitrepo/IDC-Detection-Model
git pull
source .venv/bin/activate

# Edit CFG — Phase 2c H1 (see `PHASE2C.md`)
tmux new -s p2c_h1
python3 src/cnn/convnext_5fold_ft.py
python3 experiments/phase2_deep_ft/aggregate_results.py
```

### After a run — find folder and copy-paste

```bash
RUN_DIR=$(ls -td experiments/results/phase2_deep_ft/*phase2c_h1_h6_ft10* | head -1)
echo "RUN_DIR=$RUN_DIR"
cat "$RUN_DIR/metrics.json"
```

Send to chat: **`RUN_DIR=`** line + **`metrics.json`** (or last ~50 lines of stdout with MEAN ± STD).

Phase aggregate:

```text
reports/experiments/phase2_deep_ft/comparison_latest.csv
reports/experiments/phase2_deep_ft/epochs_detail_latest.csv
```

**Pre-run checklist**

- [ ] Only **one** parameter changed vs previous run
- [ ] `run_tag` unique and matches the run (e.g. `phase2b_f2_batch16`)
- [ ] `tmux` for long runs
- [ ] After completion: fill in §6 here + **paragraph in `THESIS_WORKBOOK.md` §7**

---

## 8. Chronology

**Log is in `THESIS_WORKBOOK.md` §7** (one paragraph per run). Numbers only here in §6.

---

## 10. Phase 2b conclusion *(2026-06-27)*

**Winner:** **FINAL** (2a) — PR **0.8712**, batch 32, `lr_head=1e-3`, `lr_backbone=2e-5`.

| Block | Tested | Outcome |
|-------|--------|---------|
| F | batch 16, accum 2 | no gain |
| G | G2 head 5e-4, G3 bb 5e-6 | G2 ≈0, G3 **−0.002** |
| G4 | skipped | G2+G3 both poor/neutral |

**Next:** **Phase 2c** — more epochs on FINAL CFG (`PHASE2C.md`, first run **H1**: head 6, ft 10).

---

## 9. Thesis text (when 2b is done)

> After the first fine-tuning phase (2a) reached a plateau around PR-AUC 0.871, the second phase (2b) tested the effect of batch size (and gradient accumulation) and a wider learning-rate range for the classification head and backbone, with a fixed training architecture (256 px, three head epochs, five fine-tuning epochs).

---

*Document map: `DOCUMENTATION.md`*
