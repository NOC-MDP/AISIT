"""
train.py — Training loop with early stopping, LR scheduling, and checkpointing.
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from model import TracerMLP

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(nn.functional.mse_loss(pred, target)).item()


def r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
):
    model.train(train)
    total_loss = 0.0
    all_pred, all_true = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            pred = model(X_batch)
            loss = nn.functional.mse_loss(pred, y_batch)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(X_batch)
            all_pred.append(pred.detach())
            all_true.append(y_batch.detach())

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    n = len(all_true)
    return {
        "loss": total_loss / n,
        "rmse": rmse(all_pred, all_true),
        "r2":   r2(all_pred, all_true),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    cfg: Config,
    model: TracerMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> TracerMLP:
    """
    Train the model with early stopping and cosine / plateau LR scheduling.

    Returns the best model (loaded from checkpoint).
    """
    tc = cfg.train
    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")
    log.info(f"Model: {model}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.lr,
        weight_decay=tc.weight_decay,
    )

    if tc.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc.epochs, eta_min=tc.lr * 1e-2
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

    ckpt_path = Path(tc.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = math.inf
    patience_ctr  = 0
    history       = []

    log.info(f"{'Epoch':>6} | {'Train RMSE':>10} | {'Val RMSE':>10} | {'Val R²':>8} | LR")
    log.info("-" * 60)

    for epoch in range(1, tc.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics   = _run_epoch(model, val_loader,   optimizer, device, train=False)

        # Scheduler step
        if tc.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()}})

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            log.info(
                f"{epoch:>6} | {train_metrics['rmse']:>10.4f} | "
                f"{val_metrics['rmse']:>10.4f} | {val_metrics['r2']:>8.4f} | {current_lr:.2e}"
            )

        # Checkpoint & early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_ctr  = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_rmse":    val_metrics["rmse"],
                "val_r2":      val_metrics["r2"],
            }, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= tc.patience:
                log.info(f"Early stopping at epoch {epoch}.")
                break

    # Reload best weights
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info(
        f"Best model from epoch {ckpt['epoch']}: "
        f"val RMSE={ckpt['val_rmse']:.4f}, val R²={ckpt['val_r2']:.4f}"
    )

    return model, history, device
