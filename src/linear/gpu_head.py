"""GPU linear classifier on frozen embeddings (logistic regression via BCE)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from experiments.lib.progress_pct import report_pct
except ImportError:
    from progress_pct import report_pct  # type: ignore[no-redef]


def fit_predict_gpu_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    *,
    device: torch.device,
    C: float = 1.0,
    class_weight: str | None = None,
    epochs: int = 50,
    batch_size: int = 65_536,
    lr: float = 0.01,
    seed: int = 42,
    progress_label: str = "classifier",
    progress_state: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train nn.Linear on GPU; return (y_pred, y_prob_pos) for validation."""
    torch.manual_seed(seed)
    if device.type != "cuda":
        raise RuntimeError("fit_predict_gpu_linear requires a CUDA device.")

    state = progress_state if progress_state is not None else {}
    report_pct(progress_label, 0, 100, state)

    in_dim = X_train.shape[1]
    model = nn.Linear(in_dim, 1, device=device)
    weight_decay = 1.0 / (C * max(len(y_train), 1))

    if class_weight == "balanced":
        n_pos = float(y_train.sum())
        n_neg = float(len(y_train) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        report_pct(progress_label, epoch + 1, epochs, state)

    model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).float().to(device, non_blocking=True)
        logits = model(X_val_t).squeeze(-1)
        prob = torch.sigmoid(logits).cpu().numpy()
        pred = (prob >= 0.5).astype(np.int32)

    return pred, prob
