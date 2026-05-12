"""
Single source of truth for VM linear-track hyperparameters.
All linear_vm/*.py runners pass these identical CLI flags to model_gpu_embedding_CNN_changer.py
(in addition to --backbone).
"""
from __future__ import annotations

SPLITS_FILE = "breast_cancer_5fold_patient_splits.csv"
IMAGE_SIZE = 128
BATCH_SIZE = 256
NUM_WORKERS = 8
PIN_MEMORY = True
N_FOLDS = 5
RANDOM_SEED = 42
SUBSET_PER_FOLD = 100_000
CLASSIFIER_TAG = "linearsvc_balanced"
EMB_ROOT = "embeddings"
SVM_C = 1.0
CALIBRATION_CV = 2


def vm_cli_args() -> list[str]:
    return [
        "--splits-file",
        SPLITS_FILE,
        "--image-size",
        str(IMAGE_SIZE),
        "--batch-size",
        str(BATCH_SIZE),
        "--num-workers",
        str(NUM_WORKERS),
        "--pin-memory" if PIN_MEMORY else "--no-pin-memory",
        "--n-folds",
        str(N_FOLDS),
        "--seed",
        str(RANDOM_SEED),
        "--subset-per-fold",
        str(SUBSET_PER_FOLD),
        "--classifier-tag",
        CLASSIFIER_TAG,
        "--emb-root",
        EMB_ROOT,
        "--svm-c",
        str(SVM_C),
        "--calibration-cv",
        str(CALIBRATION_CV),
    ]
