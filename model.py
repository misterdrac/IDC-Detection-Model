# model.py (CPU-friendly, IMBALANCED, MobileNetV2 + cached embeddings + subset-per-fold)
# Uses Linear SVM (LinearSVC) + calibration for probabilities (ROC-AUC / PR-AUC)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from tqdm import tqdm

# --------------------
# CONFIG
# --------------------
SPLITS_FILE = "breast_cancer_5fold_patient_splits.csv"
EMB_FILE = "mobilenetv2_embeddings_subset.npz"

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
INPUT_SHAPE = IMAGE_SIZE + (3,)

N_FOLDS = 5
RANDOM_SEED = 42

# SUBSET per fold (keeps imbalance, just fewer samples)
SUBSET_PER_FOLD = 15000  # try 10000 if still slow

# IMPORTANT: keep dataset imbalanced -> NO class_weight, NO resampling
SVM_C = 1.0

# Calibration CV for probabilities (2 is faster than 3)
CALIBRATION_CV = 2


# --------------------
# FEATURE EXTRACTOR
# --------------------
def build_feature_extractor():
    print(f"Loading MobileNetV2 (frozen) with input shape {INPUT_SHAPE}...")

    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    base.trainable = False

    extractor = models.Sequential([
        base,
        layers.GlobalAveragePooling2D()
    ])

    return extractor


# --------------------
# DATASET PIPELINE
# --------------------
def build_image_dataset(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_and_preprocess(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------
# PRECOMPUTE + CACHE
# --------------------
def precompute_and_save_embeddings(df, feature_extractor, out_file):
    paths = df["path"].astype(str).values
    labels = df["target"].astype(np.int32).values
    folds = df["fold"].astype(np.int32).values

    ds = build_image_dataset(paths, labels, BATCH_SIZE)

    feats, labs = [], []

    print(f"🚀 Precomputing embeddings for subset: {len(paths)} images (done once)")
    for batch_images, batch_labels in tqdm(ds, desc="Embedding images"):
        batch_features = feature_extractor(batch_images, training=False)
        feats.append(batch_features.numpy())
        labs.append(batch_labels.numpy())

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labs, axis=0)

    np.savez_compressed(out_file, X=X, y=y, fold=folds)
    print(f"✅ Saved embeddings: {out_file}")
    print(f"   X shape: {X.shape} | y shape: {y.shape} | fold shape: {folds.shape}")


# --------------------
# MAIN CV
# --------------------
def train_and_evaluate_cv():
    df = pd.read_csv(SPLITS_FILE)

    required_cols = {"path", "patient_id", "target", "fold"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df = df[df["fold"].notna()].copy()
    df["fold"] = df["fold"].astype(int)

    print(f"Loaded {len(df)} rows, {df['patient_id'].nunique()} unique patients.")
    print("Original fold counts:")
    print(df["fold"].value_counts().sort_index())

    # ---- SUBSET per fold (keeps imbalance, just fewer samples) ----
    parts = []
    for f in sorted(df["fold"].unique()):
        g = df[df["fold"] == f]
        parts.append(g.sample(n=min(SUBSET_PER_FOLD, len(g)), random_state=RANDOM_SEED))
    df = pd.concat(parts, ignore_index=True)

    print(f"\nUsing subset-per-fold: {SUBSET_PER_FOLD} (or max available).")
    print("Subset fold counts:")
    print(df["fold"].value_counts().sort_index())
    print(f"Subset total images: {len(df)}")
    print(f"Subset overall pos ratio: {df['target'].mean():.2%}")

    # Build extractor once
    feature_extractor = build_feature_extractor()

    # Load or compute embeddings
    if os.path.exists(EMB_FILE):
        print(f"\n📦 Loading cached embeddings from {EMB_FILE} ...")
        data = np.load(EMB_FILE)
        X_all = data["X"]
        y_all = data["y"]
        fold_all = data["fold"]
        print(f"✅ Loaded X={X_all.shape}, y={y_all.shape}")
    else:
        print(f"\n📦 Cache not found. Creating: {EMB_FILE}")
        precompute_and_save_embeddings(df, feature_extractor, EMB_FILE)
        data = np.load(EMB_FILE)
        X_all = data["X"]
        y_all = data["y"]
        fold_all = data["fold"]

    fold_metrics = []

    # CV loop (fast after embeddings exist)
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

        # --- Linear SVM + calibration ---
        print("\n--- Training Linear SVM (fast) + calibration (still imbalanced) ---")
        base_svm = LinearSVC(C=SVM_C, random_state=RANDOM_SEED)

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
            print(f"PR-AUC:   {pr_auc:.4f}  (Average Precision)")
        else:
            print("ROC-AUC / PR-AUC skipped (validation fold had only one class).")

        print("\nConfusion matrix:")
        print(confusion_matrix(y_val, y_pred))

        print("\nClassification report (watch recall for IDC=1):")
        print(classification_report(
            y_val, y_pred,
            target_names=["No IDC (0)", "IDC (1)"],
            digits=4
        ))

        fold_metrics.append({
            "fold": fold,
            "accuracy": acc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })

    # Summary
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
    try:
        train_and_evaluate_cv()
    except FileNotFoundError:
        print(f"ERROR: Splits file not found: {SPLITS_FILE}")
        print("Run split_script.py first.")
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nNajčešće: krivi path u CSV-u ili problem s čitanjem slika.")
