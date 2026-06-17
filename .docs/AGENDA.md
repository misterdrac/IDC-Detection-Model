# Agenda (current)

Last updated: 2026-06-17

Single document in `.docs/` — what we are doing, how we deploy to the VM, and what belongs in git.

---

## Project goal

IDC detection from histopathology images: compare a **linear** pipeline (CNN embeddings + Linear SVM) with **CNN fine-tuning** (ConvNeXt-Tiny), with focus on class imbalance, reproducibility, and thesis-ready evidence.

---

## Repository layout

```
src/           core scripts (data split, linear, CNN)
runners/       entry points (phase1, per-backbone linear_vm)
experiments/   phase orchestration + shared lib
reports/       baseline results + experiment summaries (in git)
```

---

## Workflow

```
Local (Cursor)  →  git commit + push  →  VM: git pull  →  one run command
```

One-time VM setup:

```bash
cd ~/Desktop/IDC_Detection_Model/IDC_Gitrepo/IDC-Detection-Model
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision pandas scikit-learn tqdm pillow numpy

export IDC_DATASET_PATH="/home/dgolubovic/Desktop/IDC_Detection_Model/IDC_Dataset/IDC_regular_ps50_idx5"
python3 src/data/split.py   # once (or when you need a fresh split)
```

---

## What to commit to git (VM runtime only)

### Required — code to run experiments

| Path | Purpose |
|------|---------|
| `src/data/split.py` | patient-level 5-fold split |
| `src/linear/embedding_svc.py` | linear track (embeddings + SVM) |
| `src/cnn/convnext_5fold_ft.py` | CNN fine-tuning |
| `runners/linear_vm/` | per-backbone runners + `shared_config.py` |
| `runners/phase1_balance.py` | phase 1 entry point |
| `experiments/lib/` | train balancing, logging |
| `experiments/phase1_balance/run_all.py` | phase 1 orchestrator |
| `.gitignore` | keep heavy artifacts out of git |
| `.docs/AGENDA.md` | this document |
| `reports/` | **baseline reference results** |

### Do not commit (generated on VM / locally)

| Path | Reason |
|------|--------|
| `embeddings/` | embedding cache |
| `checkpoints_convnext/`, `results_convnext/` | CNN checkpoints and metrics |
| `experiments/results/` | full per-run logs from new experiment batches |
| `breast_cancer_5fold_patient_splits.csv` | paths are machine-specific |
| `.venv/`, `__pycache__/`, `.idea/` | environment / IDE |

New phase 1 outputs land under `experiments/results/` on the VM; copy summaries into `reports/experiments/phase1_balance/` when you want them in git.

---

## Phase 1 — now (mentor: balanced vs unbalanced)

**Question:** How does each backbone perform on **natural** training data (~28% IDC) vs **balanced** training data (50/50), while **validation** stays natural?

| Regime | Train | SVM `class_weight` |
|--------|--------|---------------------|
| `natural` | full fold | `none` |
| `balanced` | undersample to 50/50 | `none` |

Backbones: mobilenetv2, resnet18, efficientnet_b0, convnext_tiny, resnet_50, inception_v3.

**Run on VM:**

```bash
source .venv/bin/activate
python3 runners/phase1_balance.py
# or: bash runners/phase1_balance.sh
```

**Outputs (on VM, not in git):**

- `experiments/results/phase1_balance/<run_id>/` — metrics.json, stdout.log
- `reports/experiments/phase1_balance/comparison_latest.csv` — copy for thesis

Optional comparison with earlier VM runs: `--include-legacy` (natural + `class_weight=balanced`).

---

## Phase 2 — next (deep fine-tune)

1. From phase 1, pick the **best backbone** for `natural` and for `balanced`.
2. Run **deep FT** on ConvNeXt (grid: lr, epochs, pos_weight, batch, image_size) — see `experiments/phase2_deep_ft/README.md`.
3. Goal: beat previous light FT and linear baselines in `reports/baselines/`.

---

## Useful commands

```bash
# Single backbone / dry-run
python3 experiments/phase1_balance/run_all.py --backbone convnext_tiny --train-balance balanced
python3 experiments/phase1_balance/run_all.py --dry-run

# Single backbone via runner
python3 runners/linear_vm/convnext_tiny.py

# ConvNeXt CNN fine-tune only
python3 src/cnn/convnext_5fold_ft.py
```

---

## Thesis activity log

After each batch of runs, add a short entry below:

```md
### YYYY-MM-DD — short title
- Command: ...
- Best result: backbone / regime / ROC-AUC / PR-AUC
- Next: ...
```

---

## Change log

### 2026-06-17
- Reorganized repo: `src/`, `runners/`, `reports/baselines/`.
- Added `experiments/` framework and phase 1 (natural vs balanced train).
- Git: commit VM code + `reports/` baselines; ignore `experiments/results/` and caches.
- All project docs in English.
