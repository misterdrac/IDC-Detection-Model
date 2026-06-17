# IDC Detection Model

Breast histopathology pipeline for **invasive ductal carcinoma (IDC)** detection: patient-level 5-fold CV, linear embeddings + SVM, and ConvNeXt-Tiny fine-tuning.

**Plan, VM workflow, git policy:** [`.docs/AGENDA.md`](.docs/AGENDA.md)

---

## Quick start (VM)

```bash
git pull
source .venv/bin/activate
export IDC_DATASET_PATH="/path/to/IDC_regular_ps50_idx5"
python3 src/data/split.py                         # if split CSV missing
python3 runners/phase1_balance.py                 # phase 1: all backbones × natural/balanced
# or: bash runners/phase1_balance.sh
```

---

## Repository layout

```
├── src/                          # core implementation
│   ├── data/split.py             # patient-level 5-fold split CSV
│   ├── linear/embedding_svc.py   # frozen CNN → embeddings → Linear SVM
│   └── cnn/convnext_5fold_ft.py  # ConvNeXt-Tiny GPU fine-tuning
├── runners/                      # entry points (thin wrappers)
│   ├── phase1_balance.py         # → experiments/phase1_balance/run_all.py
│   ├── phase1_balance.sh
│   └── linear_vm/                # one script per backbone + shared_config.py
├── experiments/                  # study orchestration (phase 1, phase 2, lib)
├── reports/                      # committed baseline results + experiment summaries
└── .docs/AGENDA.md
```

| Path | Role |
|------|------|
| `src/data/split.py` | Build `breast_cancer_5fold_patient_splits.csv` (`IDC_DATASET_PATH`) |
| `src/linear/embedding_svc.py` | Linear track main script |
| `runners/linear_vm/<backbone>.py` | Run one backbone with VM defaults |
| `src/cnn/convnext_5fold_ft.py` | CNN fine-tuning only |
| `experiments/phase1_balance/` | Natural vs balanced train comparison |
| `reports/baselines/` | Prior VM metrics (reference) |

---

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `tqdm`, `pillow`, `torch`, `torchvision`
- NVIDIA GPU recommended (required for CNN script)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset

```
IDC_regular_ps50_idx5/
  <patient_id>/
    .../class0.png   → label 0
    .../class1.png   → label 1
```

```bash
export IDC_DATASET_PATH="/path/to/IDC_regular_ps50_idx5"
python3 src/data/split.py
```

---

## Outputs

| Location | In git? | Purpose |
|----------|---------|---------|
| `reports/baselines/` | **Yes** | Reference VM results |
| `reports/experiments/` | **Yes** | Phase 1 comparison copies |
| `experiments/results/` | No | Full per-run logs on VM |
| `embeddings/`, `checkpoints_convnext/`, `results_convnext/` | No | Training caches |

See `reports/README.md` and `.gitignore`.
