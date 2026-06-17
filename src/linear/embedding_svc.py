# Linear track: GPU embeddings + calibrated LinearSVM + 5-fold CV.
# Embeddings cached per backbone / subset_per_fold / image_size.
#
#   python src/linear/embedding_svc.py --backbone mobilenetv2
#   python runners/linear_vm/mobilenetv2.py
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from runners.linear_vm.shared_config import (
        SPLITS_FILE,
        IMAGE_SIZE,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        N_FOLDS,
        RANDOM_SEED,
        SUBSET_PER_FOLD,
        CLASSIFIER_TAG,
        EMB_ROOT,
        SVM_C,
        CALIBRATION_CV,
    )
except ImportError:
    SPLITS_FILE = "breast_cancer_5fold_patient_splits.csv"
    IMAGE_SIZE = 128
    BATCH_SIZE = 256
    NUM_WORKERS = 8
    PIN_MEMORY = True
    N_FOLDS = 5
    RANDOM_SEED = 42
    SUBSET_PER_FOLD = 100000
    CLASSIFIER_TAG = "linearsvc_balanced"
    EMB_ROOT = "embeddings"
    SVM_C = 1.0
    CALIBRATION_CV = 2

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


# Default backbone if CLI omitted; other knobs from linear_vm/shared_config.py or CLI.
BACKBONE_NAME = "convnext_tiny"
TRAIN_BALANCE = "natural"  # natural | balanced (train undersampling; val stays natural)
CLASS_WEIGHT_MODE = "balanced"  # none | balanced (LinearSVC class_weight)
OUTPUT_DIR = ""


# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("no cuda running on cpu")


# PATH HELPERS

def get_emb_path() -> str:
    """
    Cache path (independent of classifier / balance — same images, same backbone).
    embeddings/{backbone}/subset_{n}/img{size}.npz
    Legacy paths with classifier_tag are checked when loading.
    """
    subdir = os.path.join(
        EMB_ROOT,
        BACKBONE_NAME,
        f"subset_{SUBSET_PER_FOLD}",
    )
    os.makedirs(subdir, exist_ok=True)
    fname = f"{BACKBONE_NAME}_subset{SUBSET_PER_FOLD}_img{IMAGE_SIZE}.npz"
    return os.path.join(subdir, fname)


def _legacy_emb_paths() -> list[str]:
    """Older caches that included classifier_tag in the folder name."""
    subdir = os.path.join(
        EMB_ROOT,
        BACKBONE_NAME,
        CLASSIFIER_TAG,
        f"subset_{SUBSET_PER_FOLD}",
    )
    fname = f"{BACKBONE_NAME}_{CLASSIFIER_TAG}_subset{SUBSET_PER_FOLD}_img{IMAGE_SIZE}.npz"
    return [os.path.join(subdir, fname)]


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

# ImageNet normalization (works for MobileNetV2 / EfficientNet / ResNet / VGG / ConvNeXt)
def get_img_tfm():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# FEATURE EXTRACTOR

@torch.no_grad()
def build_feature_extractor(backbone_name: str):
    name = backbone_name.lower().strip()

    if name in ["mobilenetv2", "mobilenet_v2"]:
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model = nn.Sequential(
            m.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # -> (N, 1280)
        )

    elif name in ["efficientnet_b0", "efficientnetb0", "effnet_b0"]:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model = nn.Sequential(
            m.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # -> (N, 1280)
        )

    elif name in ["resnet18", "resnet_18"]:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove final FC -> output is (N, 512, 1, 1)
        model = nn.Sequential(
            *list(m.children())[:-1],
            nn.Flatten()  # -> (N, 512)
        )

    elif name in ["resnet50", "resnet_50"]:
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # -> (N, 2048, 1, 1)
        model = nn.Sequential(
            *list(m.children())[:-1],
            nn.Flatten()  # -> (N, 2048)
        )

    elif name in ["convnext_tiny", "convnexttiny"]:

        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        model = nn.Sequential(

            m.features,

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten()  # -> (N, 768)

        )

    elif name in ["inception_v3", "inceptionv3"]:

        m = models.inception_v3(

            weights=models.Inception_V3_Weights.IMAGENET1K_V1,

            aux_logits=True  # torchvision weights force this

        )

        class InceptionEmbedding(nn.Module):

            def __init__(self, net: nn.Module):

                super().__init__()

                self.net = net



                self.net.fc = nn.Identity()

            def forward(self, x):

                out = self.net(x)

                if isinstance(out, tuple):
                    out = out[0]

                if hasattr(out, "logits"):  # InceptionOutputs

                    out = out.logits

                return out  # -> (N, 2048)

        model = InceptionEmbedding(m)


    else:
        raise ValueError(
            f"Unknown BACKBONE_NAME='{backbone_name}'. "
            "Use: mobilenetv2, efficientnet_b0, resnet18, resnet_50, convnext_tiny, inception_v3."
        )

    model.eval().to(DEVICE)
    return model


# EMBEDDING CACHE

@torch.no_grad()
def compute_embeddings(paths, labels, extractor: nn.Module):
    ds = PatchDataset(paths, labels, get_img_tfm())
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

    extractor = build_feature_extractor(BACKBONE_NAME)

    print(f"🚀 Precomputing embeddings for subset: {len(paths)} images")
    X, y = compute_embeddings(paths, labels, extractor)

    np.savez_compressed(out_file, X=X, y=y, fold=folds)
    print(f"✅ Saved embeddings: {out_file}")
    print(f"   X shape: {X.shape} | y shape: {y.shape} | fold shape: {folds.shape}")


# CV TRAIN/EVAL

def _resolve_emb_cache() -> str:
    primary = get_emb_path()
    if os.path.exists(primary):
        return primary
    for legacy in _legacy_emb_paths():
        if os.path.exists(legacy):
            print(f" Using legacy embedding cache: {legacy}")
            return legacy
    return primary


def _save_run_artifacts(result: dict) -> None:
    if not OUTPUT_DIR:
        return
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    rows = result.get("folds", [])
    if rows:
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    summary = result.get("summary_mean_std", {})
    if summary:
        flat = {
            f"{k}_mean": v.get("mean") for k, v in summary.items()
        }
        flat.update({f"{k}_std": v.get("std") for k, v in summary.items()})
        pd.DataFrame([flat]).to_csv(out / "summary.csv", index=False)


def train_and_evaluate_cv() -> dict:
    try:
        from experiments.lib.balance import balance_train_arrays
    except ImportError:
        balance_train_arrays = None  # type: ignore

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

    print(f"\nRun config -> backbone={BACKBONE_NAME} | classifier_tag={CLASSIFIER_TAG}")
    print(f"  train_balance={TRAIN_BALANCE} | class_weight={CLASS_WEIGHT_MODE}")
    print(f"  subset_per_fold={SUBSET_PER_FOLD} | img={IMAGE_SIZE}")
    print(f"Embeddings path -> {get_emb_path()}")

    print(f"\nUsing subset-per-fold: {SUBSET_PER_FOLD} (or max available).")
    print("Subset fold counts:")
    print(df["fold"].value_counts().sort_index())
    print(f"Subset total images: {len(df)}")
    print(f"Subset overall pos ratio: {df['target'].mean():.2%}")

    # embeddings cache
    emb_path = _resolve_emb_cache()
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

    cw = None if CLASS_WEIGHT_MODE == "none" else "balanced"
    fold_metrics = []

    for fold in range(N_FOLDS):
        print(f"\n======== Fold {fold}/{N_FOLDS-1} ========")

        train_mask = fold_all != fold
        val_mask = fold_all == fold

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]

        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]

        if TRAIN_BALANCE == "balanced":
            if balance_train_arrays is None:
                raise RuntimeError("experiments.lib.balance required for --train-balance balanced")
            n_before = len(y_train)
            X_train, y_train = balance_train_arrays(X_train, y_train, seed=RANDOM_SEED + fold)
            print(f"Balanced train: {n_before} -> {len(y_train)} samples (50/50 in train only)")

        print(f"Train: {len(train_df)} images | pos ratio={train_df['target'].mean():.2%}"
              + (f" | SVM train vectors={len(y_train)} pos={y_train.mean():.2%}" if TRAIN_BALANCE == "balanced" else ""))
        print(f"Val:   {len(val_df)} images | pos ratio={val_df['target'].mean():.2%}")

        print(f"\n--- Training Linear SVM + calibration (class_weight={cw!r}) ---")
        base_svm = LinearSVC(C=SVM_C, class_weight=cw, random_state=RANDOM_SEED)
        print(f"    SVM params: C={SVM_C}, class_weight={cw!r}, calibration_cv={CALIBRATION_CV}")

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

        fold_metrics.append({
            "fold": fold,
            "accuracy": acc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "train_size": int(len(y_train)),
            "val_size": int(len(y_val)),
            "train_pos_ratio": float(y_train.mean()),
            "val_pos_ratio": float(y_val.mean()),
        })

    print("\n=====================================")
    print("      FINAL CROSS-VALIDATION SUMMARY ")
    print("=====================================")
    accs = [m["accuracy"] for m in fold_metrics]
    print("Accuracy per fold:", [f"{a:.4f}" for a in accs])
    print(f"Mean Acc: {np.mean(accs):.4f} | Std: {np.std(accs):.4f}")

    roc_aucs = [m["roc_auc"] for m in fold_metrics if m["roc_auc"] is not None]
    pr_aucs = [m["pr_auc"] for m in fold_metrics if m["pr_auc"] is not None]
    summary = {}
    if roc_aucs:
        print(f"Mean ROC-AUC: {np.mean(roc_aucs):.4f} | Std: {np.std(roc_aucs):.4f}")
        summary["roc_auc"] = {"mean": float(np.mean(roc_aucs)), "std": float(np.std(roc_aucs))}
    if pr_aucs:
        print(f"Mean PR-AUC:  {np.mean(pr_aucs):.4f} | Std: {np.std(pr_aucs):.4f}")
        summary["pr_auc"] = {"mean": float(np.mean(pr_aucs)), "std": float(np.std(pr_aucs))}
    summary["accuracy"] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    print("=====================================")

    result = {
        "track": "linear_embedding_svm",
        "backbone": BACKBONE_NAME,
        "train_balance": TRAIN_BALANCE,
        "class_weight": CLASS_WEIGHT_MODE,
        "classifier_tag": CLASSIFIER_TAG,
        "config": {
            "splits_file": SPLITS_FILE,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "subset_per_fold": SUBSET_PER_FOLD,
            "svm_c": SVM_C,
            "calibration_cv": CALIBRATION_CV,
            "random_seed": RANDOM_SEED,
            "n_folds": N_FOLDS,
        },
        "embedding_cache": emb_path,
        "embedding_shape": list(X_all.shape),
        "finished_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "folds": fold_metrics,
        "summary_mean_std": summary,
    }
    _save_run_artifacts(result)
    return result

def _apply_cli_to_globals(args: argparse.Namespace) -> None:
    global SPLITS_FILE, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    global N_FOLDS, RANDOM_SEED, SUBSET_PER_FOLD, BACKBONE_NAME, CLASSIFIER_TAG, EMB_ROOT
    global SVM_C, CALIBRATION_CV, TRAIN_BALANCE, CLASS_WEIGHT_MODE, OUTPUT_DIR

    SPLITS_FILE = args.splits_file
    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    PIN_MEMORY = args.pin_memory
    N_FOLDS = args.n_folds
    RANDOM_SEED = args.seed
    SUBSET_PER_FOLD = args.subset_per_fold
    BACKBONE_NAME = args.backbone.strip().lower()
    CLASSIFIER_TAG = args.classifier_tag
    EMB_ROOT = args.emb_root
    SVM_C = args.svm_c
    CALIBRATION_CV = args.calibration_cv
    TRAIN_BALANCE = args.train_balance
    CLASS_WEIGHT_MODE = args.class_weight
    OUTPUT_DIR = args.output_dir or ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Linear track: backbone embeddings + calibrated LinearSVC, 5-fold."
    )
    p.add_argument(
        "--backbone",
        default="convnext_tiny",
        help="mobilenetv2 | efficientnet_b0 | resnet18 | resnet_50 | convnext_tiny | inception_v3",
    )
    p.add_argument("--splits-file", default=SPLITS_FILE)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=PIN_MEMORY)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--subset-per-fold", type=int, default=SUBSET_PER_FOLD)
    p.add_argument("--classifier-tag", default=CLASSIFIER_TAG)
    p.add_argument("--emb-root", default=EMB_ROOT)
    p.add_argument("--svm-c", type=float, default=SVM_C)
    p.add_argument("--calibration-cv", type=int, default=CALIBRATION_CV)
    p.add_argument(
        "--train-balance",
        choices=("natural", "balanced"),
        default=TRAIN_BALANCE,
        help="natural=full train fold (~28%% IDC); balanced=undersample train to 50/50 (val stays natural).",
    )
    p.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default=CLASS_WEIGHT_MODE,
        help="LinearSVC class_weight (independent of train-balance).",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="If set, write metrics.json, metrics.csv, summary.csv here.",
    )
    return p.parse_args()


if __name__ == "__main__":
    _apply_cli_to_globals(parse_args())
    print(
        f"\nLinear run -> backbone={BACKBONE_NAME} | img={IMAGE_SIZE} | batch={BATCH_SIZE} "
        f"| workers={NUM_WORKERS} | subset_per_fold={SUBSET_PER_FOLD}"
        f"\n  train_balance={TRAIN_BALANCE} | class_weight={CLASS_WEIGHT_MODE}\n"
    )
    train_and_evaluate_cv()
