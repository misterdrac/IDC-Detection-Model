# IDC (Invasive Ductal Carcinoma) Detection — How to Run

This repo contains **4 scripts**:

- `split_script.py` — scans the dataset and creates **patient-level** 5-fold splits CSV  
- `model_gpu_embbedding.py` — **GPU embeddings (PyTorch)** + **CPU Linear SVM** (5-fold CV), with embedding cache  
- `model_gpu_embedding_CNN_changer.py` — same idea, but lets you switch the CNN backbone (ResNet/EfficientNet/ConvNeXt/Inception)  
- `model_ConvNeXt_Tiny_5fold_FineTuned.py` — **GPU-only** ConvNeXt-Tiny fine-tuning with AMP, OOM-safe batch reduction, and 5-fold results

The general flow is:

1) Download dataset → put it on disk  
2) Run `split_script.py` → generates `breast_cancer_5fold_patient_splits.csv`  
3) Run **one** training script (SVM-embeddings OR ConvNeXt fine-tune)

---

## 1) Requirements

### System
- Python **3.10 / 3.11**
- (Recommended) NVIDIA GPU with CUDA for PyTorch (especially for ConvNeXt finetuning)

### Python packages
Install these packages in a virtual environment:

- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `pillow`
- `torch`
- `torchvision`

> Note: `model_gpu_embbedding.py` and `model_gpu_embedding_CNN_changer.py` set `TF_CPP_MIN_LOG_LEVEL`, but they do **not** require TensorFlow.

---

## 2) Setup (venv + dependencies)

### Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install numpy pandas scikit-learn tqdm pillow
```

Install PyTorch (+ CUDA) from official selector (recommended).  
Or CPU-only (slower) via pip.

### Linux/macOS
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install numpy pandas scikit-learn tqdm pillow
```

Then install PyTorch/torchvision appropriately for your OS/GPU.

---

## 3) Download dataset

These scripts expect the IDC dataset in a structure like:

```
IDC_regular_ps50_idx5/
  <patient_id>/
    0/  (or filenames containing class0.png)
    1/  (or filenames containing class1.png)
    *.png
```

Your `split_script.py` currently detects labels by filename:
- contains `class0.png` → label 0  
- contains `class1.png` → label 1  

So make sure your dataset filenames match that pattern (or modify the script).

---

## 4) Step 1 — Create patient-level 5-fold splits CSV

Open `split_script.py` and **change**:

```python
DATASET_PATH = 'D:/.Programming/datasets/IDC/data/IDC_regular_ps50_idx5'
```

to the path where you extracted the dataset, e.g.:

```python
DATASET_PATH = r"C:\datasets\IDC_regular_ps50_idx5"
# or on Linux:
# DATASET_PATH = "/home/user/datasets/IDC_regular_ps50_idx5"
```

Then run:

```bash
python split_script.py
```

Output:
- `breast_cancer_5fold_patient_splits.csv`

That CSV must contain columns:
- `path`, `patient_id`, `target`, `fold`

---

## 5) Step 2A — Option A (Recommended for quick experiments)
### GPU embeddings + CPU Linear SVM (5-fold CV)

This option extracts embeddings once (cached to disk) and then trains a Linear SVM with calibration.

#### Run (simple, fixed extractor in script)
```bash
python model_gpu_embbedding.py
```

Key config inside the file:
- `SPLITS_FILE = "breast_cancer_5fold_patient_splits.csv"`  
- `IMAGE_SIZE = 96`
- `SUBSET_PER_FOLD = 55000` (reduces size per fold)
- Embeddings cache folder: `embeddings/`

Outputs:
- Embeddings cache: `embeddings/<backbone>/<classifier_tag>/subset_<N>/...npz`
- Prints per-fold Accuracy / ROC-AUC / PR-AUC + confusion matrix + report

> Note: In `model_gpu_embbedding.py`, the extractor is **MobileNetV2**, even if `BACKBONE_NAME` is set differently.
> If you want true backbone switching, use the next script.

---

## 6) Step 2B — Option B (Backbone switcher)
### GPU embeddings + CPU Linear SVM with selectable CNN backbone

Run:
```bash
python model_gpu_embedding_CNN_changer.py
```

Change these settings in the file to test different CNNs:
- `BACKBONE_NAME = "convnext_tiny"` (options: mobilenetv2, efficientnet_b0, resnet18, resnet_50, convnext_tiny, inception_v3)
- `SUBSET_PER_FOLD = 35000` (set 10000/15000/25000/45000/55000 as needed)
- `IMAGE_SIZE = 96`

Outputs:
- same as above: cached embeddings + printed fold metrics

---

## 7) Step 2C — Option C (Heavier, best accuracy potential)
### ConvNeXt-Tiny fine-tuning (GPU only, 5-fold)

This script is **GPU-only** and will exit if CUDA is not available.

Run:
```bash
python model_ConvNeXt_Tiny_5fold_FineTuned.py
```

Important config at the top of the script (edit inside `CFG` dataclass):
- `csv_path = "breast_cancer_5fold_patient_splits.csv"`
- `image_size = 96` (safer for smaller VRAM)
- `batch_size = 16` (auto-reduces on OOM)
- `head_epochs = 2`, `ft_epochs = 1`
- imbalance handled via `pos_weight` in `BCEWithLogitsLoss`

Outputs:
- checkpoints in `checkpoints_convnext/`
- results CSV/JSON in `results_convnext/`
- printed per-fold metrics + final mean±std

---

## 8) Expected project files after running

After Step 1:
- `breast_cancer_5fold_patient_splits.csv`

After embeddings scripts:
- `embeddings/` folder with `.npz` caches

After ConvNeXt finetune:
- `checkpoints_convnext/` (model weights)
- `results_convnext/` (metrics CSV/JSON)

---

## 9) Troubleshooting

### “No images found! Please check your DATASET_PATH.”
- Your `DATASET_PATH` in `split_script.py` is wrong, or dataset not extracted properly.

### CUDA not available / GPU not detected
- Install CUDA-enabled PyTorch build.
- Check:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

### Out of memory (OOM)
- For ConvNeXt script: it auto-reduces batch size, but you can also lower `image_size` to 96 or 64.
- For embeddings scripts: reduce `BATCH_SIZE` or `SUBSET_PER_FOLD`.

---

## 10) Minimal “do everything” commands

```bash
# 1) Create splits CSV (after editing DATASET_PATH in split_script.py)
python split_script.py

# 2) Choose ONE of the following:

# A) embeddings + SVM (quick)
python model_gpu_embbedding.py

# B) embeddings + SVM (choose CNN backbone)
python model_gpu_embedding_CNN_changer.py

# C) ConvNeXt fine-tuning (GPU-only)
python model_ConvNeXt_Tiny_5fold_FineTuned.py
```
