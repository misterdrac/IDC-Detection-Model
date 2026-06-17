"""Train-set balancing for fold-wise CV (validation stays natural)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def balance_train_dataframe(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Undersample majority class in train to match minority count (image-level)."""
    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        raise ValueError("Cannot balance: one class is missing from train fold.")
    pos_s = pos.sample(n=n, random_state=seed)
    neg_s = neg.sample(n=n, random_state=seed)
    out = pd.concat([pos_s, neg_s], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def balance_train_arrays(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Undersample majority class in embedding arrays."""
    y = y.astype(np.int32)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n = min(len(idx_pos), len(idx_neg))
    if n == 0:
        raise ValueError("Cannot balance: one class is missing from train fold.")
    rng = np.random.default_rng(seed)
    pick_pos = rng.choice(idx_pos, size=n, replace=False)
    pick_neg = rng.choice(idx_neg, size=n, replace=False)
    idx = np.concatenate([pick_pos, pick_neg])
    rng.shuffle(idx)
    return X[idx], y[idx]
