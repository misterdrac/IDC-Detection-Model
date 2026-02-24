# train_convnext_tiny_5fold_lite.py
# ConvNeXt Tiny 5-fold finetune (GPU-only) with:
# - visible progress (tqdm) + heartbeat logs
# - adaptive batch size on CUDA OOM
# - AMP (new torch.amp API)
# - class imbalance handling via pos_weight (BCEWithLogitsLoss)
# - per-fold metrics + final summary
# Run:  python train_convnext_tiny_5fold_lite.py

import os
import gc
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# CONFIG (edit here - no CLI args)

@dataclass
class CFG:
    csv_path: str = "breast_cancer_5fold_patient_splits.csv"
    n_folds: int = 5

    # Speed/VRAM knobs
    image_size: int = 96            # safer than 128 on 4GB VRAM
    batch_size: int = 16            # starting batch (auto-reduces on OOM)
    min_batch_size: int = 4
    grad_accum_steps: int = 2       # effective batch = batch_size * accum (roughly)

    num_workers: int = 2            # Windows stability
    pin_memory: bool = True

    # Training
    seed: int = 42
    amp: bool = True

    # Quick finetune plan:
    # - train head for head_epochs
    # - optionally finetune last stage for ft_epochs
    head_epochs: int = 2
    ft_epochs: int = 1
    finetune_last_stage: bool = True  # last stage only (still light)

    # Optimizer
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 1e-4

    # Evaluation
    threshold: float = 0.5
    eval_every: int = 1             # evaluate every epoch (keep 1 for clear logs)

    # Output
    out_dir: str = "results_convnext"
    ckpt_dir: str = "checkpoints_convnext"

    # Heartbeat
    heartbeat_sec: int = 15


cfg = CFG()


# FORCE GPU

if not torch.cuda.is_available():
    raise SystemExit(" CUDA not available. This is GPU only script")

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print(" Device:", DEVICE)
print(" GPU:", torch.cuda.get_device_name(0))


# UTILS

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def is_cuda_oom(e: RuntimeError) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) and ("cuda" in msg)

def safe_div(a, b):
    return float(a) / float(b + 1e-12)

seed_everything(cfg.seed)
torch.set_num_threads(4)


# DATASET

class PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tfm):
        self.paths = df["path"].astype(str).values
        self.targets = df["target"].astype(np.float32).values
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.tfm(img)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, y

def build_transforms(img_size: int):
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfm, val_tfm

def make_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=False,  # safer on Windows
    )


# MODEL

def build_convnext_tiny_binary() -> nn.Module:
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_features, 1)  # binary logit
    return m

def set_trainable(model: nn.Module, finetune_last_stage: bool):
    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # head always trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    if finetune_last_stage:
        # unfreeze last stage of ConvNeXt features
        for p in model.features[-1].parameters():
            p.requires_grad = True

def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(p)
        else:
            backbone_params.append(p)

    if backbone_params:
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.lr_backbone},
                {"params": head_params, "lr": cfg.lr_head},
            ],
            weight_decay=cfg.weight_decay
        )
    return torch.optim.AdamW(head_params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)


# METRICS

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, threshold: float) -> Dict:
    model.eval()
    ys, ps = [], []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits)

        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(np.int32)
    y_prob = np.concatenate(ps).astype(np.float32)

    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    pr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan

    y_pred = (y_prob >= threshold).astype(np.int32)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    spec = safe_div(tn, tn + fp)
    sens = safe_div(tp, tp + fn)
    bal_acc = 0.5 * (spec + sens)

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "sensitivity": float(sens),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold)
    }

def fmt(m: dict) -> str:
    return (
        f"ROC={m['roc_auc']:.4f} | PR={m['pr_auc']:.4f} | "
        f"ACC={m['accuracy']:.4f} | BalACC={m['balanced_accuracy']:.4f} | "
        f"Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | F1={m['f1']:.4f} | "
        f"Spec={m['specificity']:.4f} | Sens={m['sensitivity']:.4f} | thr={m['threshold']:.2f}"
    )

def fmt_cm(m: dict) -> str:
    return f"CM: [[TN={m['tn']}, FP={m['fp']}], [FN={m['fn']}, TP={m['tp']}]]"


# TRAIN LOOP (adaptive bs + grad accumulation + visible progress)

def train_one_epoch_adaptive(
    model: nn.Module,
    train_ds: Dataset,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    fold: int,
    phase: str,
    epoch: int,
) -> Tuple[float, int]:
    model.train()
    loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)

    running = 0.0
    seen = 0
    optimizer.zero_grad(set_to_none=True)

    start_t = time.time()
    last_heartbeat = start_t

    pbar = tqdm(
        loader,
        desc=f"[Fold {fold}] {phase} ep {epoch} | bs={batch_size}",
        leave=True,
        mininterval=0.2,
        smoothing=0.05,
        dynamic_ncols=True
    )

    for step, (x, y) in enumerate(pbar, 1):
        try:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cfg.amp):
                logits = model(x).squeeze(1)
                loss = criterion(logits, y) / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if step % cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_item = float(loss.item()) * cfg.grad_accum_steps
            running += loss_item * x.size(0)
            seen += x.size(0)

            lr = optimizer.param_groups[0]["lr"]
            mem_gb = torch.cuda.memory_allocated() / (1024**3)
            max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

            pbar.set_postfix(
                loss=f"{loss_item:.4f}",
                lr=f"{lr:.1e}",
                seen=f"{seen}",
                mem=f"{mem_gb:.2f}GB",
                max=f"{max_mem_gb:.2f}GB"
            )

            now = time.time()
            if now - last_heartbeat >= cfg.heartbeat_sec:
                elapsed = now - start_t
                print(f" Heartbeat | fold={fold} {phase} ep={epoch} step={step}/{len(loader)} "
                      f"elapsed={elapsed:.1f}s bs={batch_size} loss={loss_item:.4f}")
                last_heartbeat = now

        except RuntimeError as e:
            if not is_cuda_oom(e):
                raise
            print(f"\n⚠ CUDA OOM at bs={batch_size}. Reducing batch size...")
            cleanup_cuda()
            new_bs = batch_size // 2
            if new_bs < cfg.min_batch_size:
                raise RuntimeError(
                    f"OOM even at batch_size={batch_size}. min_batch_size={cfg.min_batch_size}. "
                    f"Try smaller image_size or higher grad_accum_steps."
                ) from e
            print(f" Retrying epoch with batch_size={new_bs}")
            return train_one_epoch_adaptive(
                model, train_ds, new_bs, optimizer, scaler, criterion, fold, phase, epoch
            )

    avg_loss = running / max(seen, 1)
    return avg_loss, batch_size


# SINGLE FOLD

def run_fold(df: pd.DataFrame, fold: int) -> Dict:
    train_tfm, val_tfm = build_transforms(cfg.image_size)

    train_df = df[df["fold"] != fold].copy()
    val_df = df[df["fold"] == fold].copy()

    # pos_weight = Nneg / Npos
    n_pos = int((train_df["target"] == 1).sum())
    n_neg = int((train_df["target"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE, dtype=torch.float32)

    train_ds = PatchDataset(train_df, train_tfm)
    val_ds = PatchDataset(val_df, val_tfm)

    # Val loader can be bigger
    val_loader = make_loader(val_ds, batch_size=max(64, cfg.batch_size), shuffle=False)

    model = build_convnext_tiny_binary().to(DEVICE)

    # Phase 1: head only
    set_trainable(model, finetune_last_stage=False)
    optimizer = build_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    bs = cfg.batch_size

    print("\n" + "=" * 90)
    print(f"START FOLD {fold}/{cfg.n_folds - 1}")
    print("=" * 90)
    print(f"Train={len(train_df)} (pos={train_df['target'].mean():.2%}) | "
          f"Val={len(val_df)} (pos={val_df['target'].mean():.2%})")
    print(f"img={cfg.image_size} | start_bs={cfg.batch_size} | min_bs={cfg.min_batch_size} | "
          f"accum={cfg.grad_accum_steps} | amp={cfg.amp}")
    print(f"pos_weight={pos_weight.item():.3f} | thr={cfg.threshold}")

    # Initial eval
    init_m = evaluate(model, val_loader, cfg.threshold)
    print("[Init] " + fmt(init_m))
    print("       " + fmt_cm(init_m))

    best_pr = -1.0
    best_state = None
    best_tag = ""

    
    # HEAD TRAIN
    
    for ep in range(1, cfg.head_epochs + 1):
        loss, bs = train_one_epoch_adaptive(
            model, train_ds, bs, optimizer, scaler, criterion,
            fold=fold, phase="HEAD", epoch=ep
        )
        print(f"[Fold {fold}] HEAD ep {ep}/{cfg.head_epochs} | bs={bs} | loss={loss:.4f}")

        if cfg.eval_every and (ep % cfg.eval_every == 0):
            m = evaluate(model, val_loader, cfg.threshold)
            print("   " + fmt(m))
            print("   " + fmt_cm(m))
            if m["pr_auc"] > best_pr:
                best_pr = m["pr_auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_tag = f"HEAD_ep{ep}"

    
    # FINETUNE LAST STAGE
    
    if cfg.finetune_last_stage and cfg.ft_epochs > 0:
        set_trainable(model, finetune_last_stage=True)
        optimizer = build_optimizer(model)  # rebuild with backbone lr
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

        for ep in range(1, cfg.ft_epochs + 1):
            loss, bs = train_one_epoch_adaptive(
                model, train_ds, bs, optimizer, scaler, criterion,
                fold=fold, phase="FT", epoch=ep
            )
            print(f"[Fold {fold}] FT   ep {ep}/{cfg.ft_epochs} | bs={bs} | loss={loss:.4f}")

            if cfg.eval_every and (ep % cfg.eval_every == 0):
                m = evaluate(model, val_loader, cfg.threshold)
                print("   " + fmt(m))
                print("   " + fmt_cm(m))
                if m["pr_auc"] > best_pr:
                    best_pr = m["pr_auc"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_tag = f"FT_ep{ep}"

    if best_state is not None:
        model.load_state_dict(best_state)

    final_m = evaluate(model, val_loader, cfg.threshold)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        cfg.ckpt_dir,
        f"convnext_tiny_fold{fold}_img{cfg.image_size}_bs{bs}_{best_tag}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)

    print(f" Fold {fold} DONE | best_tag={best_tag} | best_PR={best_pr:.4f}")
    print("   Final " + fmt(final_m))
    print("   " + fmt_cm(final_m))
    print(f"   ckpt={ckpt_path}")

    return {
        "fold": int(fold),
        "image_size": int(cfg.image_size),
        "start_batch_size": int(cfg.batch_size),
        "final_batch_size": int(bs),
        "min_batch_size": int(cfg.min_batch_size),
        "num_workers": int(cfg.num_workers),
        "amp": bool(cfg.amp),
        "grad_accum_steps": int(cfg.grad_accum_steps),
        "head_epochs": int(cfg.head_epochs),
        "ft_epochs": int(cfg.ft_epochs),
        "finetune_last_stage": bool(cfg.finetune_last_stage),
        "lr_head": float(cfg.lr_head),
        "lr_backbone": float(cfg.lr_backbone),
        "weight_decay": float(cfg.weight_decay),
        "threshold": float(cfg.threshold),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "train_pos_ratio": float(train_df["target"].mean()),
        "val_pos_ratio": float(val_df["target"].mean()),
        "pos_weight": float(pos_weight.item()),
        "best_tag": best_tag,
        "best_pr": float(best_pr),
        "checkpoint": ckpt_path,
        **final_m
    }


# MAIN

def main():
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    df = pd.read_csv(cfg.csv_path)
    required_cols = {"path", "patient_id", "target", "fold"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df = df[df["fold"].notna()].copy()
    df["fold"] = df["fold"].astype(int)

    print("\nLoaded:", len(df), "rows | unique patients:", df["patient_id"].nunique())
    print("Fold counts:")
    print(df["fold"].value_counts().sort_index().to_string())

    print("\nRUN CONFIG:")
    print(f"image_size={cfg.image_size} | start_bs={cfg.batch_size} | min_bs={cfg.min_batch_size} | accum={cfg.grad_accum_steps}")
    print(f"head_epochs={cfg.head_epochs} | ft_epochs={cfg.ft_epochs} | finetune_last_stage={cfg.finetune_last_stage}")
    print(f"lr_head={cfg.lr_head} | lr_backbone={cfg.lr_backbone} | wd={cfg.weight_decay}")
    print(f"amp={cfg.amp} | workers={cfg.num_workers} | threshold={cfg.threshold}")
    print("Output:", cfg.out_dir, "| CKPT:", cfg.ckpt_dir)

    start = time.time()
    rows: List[Dict] = []

    for fold in range(cfg.n_folds):
        cleanup_cuda()
        rows.append(run_fold(df, fold))

    out_df = pd.DataFrame(rows)

    cols = [
        "roc_auc", "pr_auc", "accuracy", "balanced_accuracy",
        "precision", "recall", "f1", "specificity", "sensitivity"
    ]

    print("\n" + "=" * 90)
    print("FINAL 5-FOLD SUMMARY (per fold)")
    print("=" * 90)
    print(out_df[["fold"] + cols + ["final_batch_size", "best_tag"]].to_string(index=False))

    print("\n" + "-" * 90)
    print("MEAN ± STD")
    print("-" * 90)
    for c in cols:
        mean = out_df[c].mean()
        std = out_df[c].std(ddof=0)
        print(f"{c.upper():<18}: {mean:.4f} ± {std:.4f}")

    runtime_min = (time.time() - start) / 60.0
    print(f"\nRuntime: {runtime_min:.2f} min")

    csv_out = os.path.join(cfg.out_dir, f"convnext_tiny_5fold_img{cfg.image_size}.csv")
    out_df.to_csv(csv_out, index=False)

    summary_out = os.path.join(cfg.out_dir, f"convnext_tiny_5fold_img{cfg.image_size}_summary.json")
    summary = {
        "model": "convnext_tiny_binary",
        "config": cfg.__dict__,
        "results_csv": csv_out,
        "metrics_mean_std": {c: {"mean": float(out_df[c].mean()), "std": float(out_df[c].std(ddof=0))} for c in cols},
        "runtime_min": float(runtime_min),
        "fold_rows": rows
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved CSV :", csv_out)
    print("Saved JSON:", summary_out)
    print("Done ")

if __name__ == "__main__":
    main()