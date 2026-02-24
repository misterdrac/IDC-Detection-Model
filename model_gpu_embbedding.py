# model_gpu_embbedding.py
# GPU embeddings (PyTorch backbone) + CPU LinearSVM (scikit-learn) + 5-fold CV
# Embeddings are cached per: backbone / classifier_tag / subset_per_fold / image_size
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# CONFIG

SPLITS_FILE = "breast_cancer_5fold_patient_splits.csv"

IMAGE_SIZE = 96
BATCH_SIZE = 128
NUM_WORKERS = 6
PIN_MEMORY = True

N_FOLDS = 5
RANDOM_SEED = 42

SUBSET_PER_FOLD = 55000

# --- RUN TAGS (for folder + filename naming)
BACKBONE_NAME = "convnext_tiny"          # kasnije: efficientnet_b0, resnet18, vgg16
CLASSIFIER_TAG = "linearsvc_balanced"
EMB_ROOT = "embeddings"                # root folder for all embeddings

# --- Classifier params
SVM_C = 1.0
CALIBRATION_CV = 2


# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Device:", DEVICE)
if DEVICE.type == "cuda":
    print(" GPU:", torch.cuda.get_device_name(0))
else:
    print("️ Nema CUDA, vrtim na CPU (provjeri torch CUDA build).")


# PATH HELPERS

def get_emb_path() -> str:
    """
    Cache path example:
    embeddings/mobilenetv2/linearsvc_balanced/subset_55000/mobilenetv2_linearsvc_balanced_subset55000_img96.npz
    """
    subdir = os.path.join(
        EMB_ROOT,
        BACKBONE_NAME,
        CLASSIFIER_TAG,
        f"subset_{SUBSET_PER_FOLD}"
    )
    os.makedirs(subdir, exist_ok=True)

    fname = f"{BACKBONE_NAME}_{CLASSIFIER_TAG}_subset{SUBSET_PER_FOLD}_img{IMAGE_SIZE}.npz"
    return os.path.join(subdir, fname)


# DATASET

class PatchDataset(Dataset):
    def __init__(self, paths, labels, tfm):
        self.paths = paths
        self.labels = labels
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = int(self.labels[idx])

        # PIL is fastest/most stable on Windows for pngs
        img = Image.open(p).convert("RGB")
        img = self.tfm(img)
        return img, y

# ImageNet normalization (works for MobileNetV2 / EfficientNet / ResNet / VGG)
IMG_TFM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# FEATURE EXTRACTOR

@torch.no_grad()
def build_feature_extractor():
    # MobileNetV2 -> 1280-dim embedding
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    backbone = m.features
    model = nn.Sequential(
        backbone,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()   # -> (N, 1280)
    )
    model.eval().to(DEVICE)
    return model


# EMBEDDING CACHE

@torch.no_grad()
def compute_embeddings(paths, labels, extractor: nn.Module):
    ds = PatchDataset(paths, labels, IMG_TFM)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and (DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    X_list = []
    y_list = []

    for imgs, ys in tqdm(dl, desc="Embedding (GPU)" if DEVICE.type == "cuda" else "Embedding (CPU)"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        feats = extractor(imgs)               # (B, D)
        X_list.append(feats.cpu().numpy())
        y_list.append(ys.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(np.int32)
    return X, y

def make_subset_per_fold(df: pd.DataFrame):
    parts = []
    for f in sorted(df["fold"].unique()):
        g = df[df["fold"] == f]
        parts.append(g.sample(n=min(SUBSET_PER_FOLD, len(g)), random_state=RANDOM_SEED))
    out = pd.concat(parts, ignore_index=True)
    return out

def precompute_and_save_embeddings(df: pd.DataFrame, out_file: str):
    paths = df["path"].astype(str).values
    labels = df["target"].astype(np.int32).values
    folds = df["fold"].astype(np.int32).values

    extractor = build_feature_extractor()

    print(f"🚀 Precomputing embeddings for subset: {len(paths)} images")
    X, y = compute_embeddings(paths, labels, extractor)

    np.savez_compressed(out_file, X=X, y=y, fold=folds)
    print(f" Saved embeddings: {out_file}")
    print(f"   X shape: {X.shape} | y shape: {y.shape} | fold shape: {folds.shape}")


# CV TRAIN/EVAL

def train_and_evaluate_cv():
    df = pd.read_csv(SPLITS_FILE)

    required_cols = {"path", "patient_id", "target", "fold"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df = df[df["fold"].notna()].copy()
    df["fold"] = df["fold"].astype(int)

    print(f"Loaded {len(df)} rows, {df['patient_id'].nunique()} unique patients.")
    print("Original fold counts:")
    print(df["fold"].value_counts().sort_index())

    # subset-per-fold (keeps imbalance, only reduces size)
    df = make_subset_per_fold(df)

    print(f"\nRun config -> backbone={BACKBONE_NAME} | classifier={CLASSIFIER_TAG} | subset_per_fold={SUBSET_PER_FOLD} | img={IMAGE_SIZE}")
    print(f"Embeddings path -> {get_emb_path()}")

    print(f"\nUsing subset-per-fold: {SUBSET_PER_FOLD} (or max available).")
    print("Subset fold counts:")
    print(df["fold"].value_counts().sort_index())
    print(f"Subset total images: {len(df)}")
    print(f"Subset overall pos ratio: {df['target'].mean():.2%}")

    # embeddings cache
    emb_path = get_emb_path()
    if os.path.exists(emb_path):
        print(f"\n Loading cached embeddings from {emb_path} ...")
        data = np.load(emb_path)
        X_all = data["X"]
        y_all = data["y"]
        fold_all = data["fold"]
        print(f" Loaded X={X_all.shape}, y={y_all.shape}")
    else:
        print(f"\n Cache not found. Creating: {emb_path}")
        precompute_and_save_embeddings(df, emb_path)
        data = np.load(emb_path)
        X_all = data["X"]
        y_all = data["y"]
        fold_all = data["fold"]

    fold_metrics = []

    for fold in range(N_FOLDS):
        print(f"\n======== Fold {fold}/{N_FOLDS-1} ========")

        train_mask = fold_all != fold
        val_mask = fold_all == fold

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]

        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        print(f"Train: {len(train_df)} images | pos ratio={train_df['target'].mean():.2%}")
        print(f"Val:   {len(val_df)} images | pos ratio={val_df['target'].mean():.2%}")

        # Linear SVM + calibration (probabilities)
        print("\n--- Training Linear SVM + calibration (class_weight='balanced') ---")
        base_svm = LinearSVC(C=SVM_C, class_weight="balanced", random_state=RANDOM_SEED)
        print(f"    SVM params: C={SVM_C}, class_weight='balanced', calibration_cv={CALIBRATION_CV}")

        svm = CalibratedClassifierCV(
            base_svm,
            method="sigmoid",
            cv=CALIBRATION_CV
        )

        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_val)
        y_prob = svm.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)

        roc_auc = None
        pr_auc = None
        if len(np.unique(y_val)) == 2:
            roc_auc = roc_auc_score(y_val, y_prob)
            pr_auc = average_precision_score(y_val, y_prob)

        print(f"\n--- Fold {fold} results ---")
        print(f"Accuracy: {acc:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC:  {roc_auc:.4f}")
            print(f"PR-AUC:   {pr_auc:.4f}")
        else:
            print("ROC-AUC / PR-AUC skipped (only one class in fold).")

        print("\nConfusion matrix:")
        print(confusion_matrix(y_val, y_pred))

        print("\nClassification report:")
        print(classification_report(
            y_val, y_pred,
            target_names=["No IDC (0)", "IDC (1)"],
            digits=4
        ))

        fold_metrics.append({"fold": fold, "accuracy": acc, "roc_auc": roc_auc, "pr_auc": pr_auc})

    print("\n=====================================")
    print("      FINAL CROSS-VALIDATION SUMMARY ")
    print("=====================================")
    accs = [m["accuracy"] for m in fold_metrics]
    print("Accuracy per fold:", [f"{a:.4f}" for a in accs])
    print(f"Mean Acc: {np.mean(accs):.4f} | Std: {np.std(accs):.4f}")

    roc_aucs = [m["roc_auc"] for m in fold_metrics if m["roc_auc"] is not None]
    pr_aucs = [m["pr_auc"] for m in fold_metrics if m["pr_auc"] is not None]
    if roc_aucs:
        print(f"Mean ROC-AUC: {np.mean(roc_aucs):.4f} | Std: {np.std(roc_aucs):.4f}")
    if pr_aucs:
        print(f"Mean PR-AUC:  {np.mean(pr_aucs):.4f} | Std: {np.std(pr_aucs):.4f}")
    print("=====================================")

if __name__ == "__main__":
    train_and_evaluate_cv()
