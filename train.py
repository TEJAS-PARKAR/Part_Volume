# -*- coding: utf-8 -*-
"""
train.py
--------
Training pipeline for PointNet / PointNet++ volume regressors.

Features:
  - 80/20 train/test split via DataLoaders
  - MSE loss, Adam optimiser (weight_decay=1e-5)
  - Gradient clipping (max_norm=1.0)
  - ReduceLROnPlateau scheduler
  - Best-model checkpointing -> models/best_model.pth
  - Per-epoch console logging with timing
  - Supports model_type: "pointnet" or "pointnet2"
"""

import os
import time
import torch
import torch.nn as nn

from utils.preprocessing  import get_dataloaders
from models.pointnet      import PointNetRegressor,  count_parameters as count_pn
from models.pointnet2     import PointNet2Regressor, count_parameters as count_pn2


# ─────────────────────────────────────────────────────────────
# Default training configuration
# ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "labels_csv"   : os.path.join("data", "labels.csv"),
    "pc_dir"       : os.path.join("data", "pointclouds"),
    "model_save"   : os.path.join("models", "best_model.pth"),
    "model_type"   : "pointnet2",    # "pointnet" or "pointnet2"
    "batch_size"   : 16,
    "num_epochs"   : 75,
    "lr"           : 1e-3,
    "weight_decay" : 1e-5,
    "grad_clip"    : 1.0,
    "dropout"      : 0.3,
    "test_split"   : 0.2,
    "seed"         : 42,
}


# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────

def build_model(model_type: str, dropout: float):
    """
    Instantiate the selected model architecture.

    Parameters
    ----------
    model_type : "pointnet" or "pointnet2"
    dropout    : float dropout rate

    Returns
    -------
    nn.Module
    """
    if model_type == "pointnet2":
        model = PointNet2Regressor(dropout_rate=dropout)
        params = count_pn2(model)
    elif model_type == "pointnet":
        model = PointNetRegressor(dropout_rate=dropout)
        params = count_pn(model)
    else:
        raise ValueError("Unknown model_type '%s'. Use 'pointnet' or 'pointnet2'." % model_type)

    print("  Model         : %s" % model_type)
    print("  Trainable params: {:,}".format(params))
    return model


# ─────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    """Run one training epoch, return average loss over dataset."""
    model.train()
    total_loss = 0.0

    for pc, vol in loader:
        pc  = pc.to(device)                 # (B, 6, N)
        vol = vol.to(device).view(-1, 1)    # (B, 1)

        optimizer.zero_grad()
        pred = model(pc)                    # (B, 1)
        loss = criterion(pred, vol)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += loss.item() * pc.size(0)

    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────
# Evaluate (no gradient)
# ─────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    """Run validation pass, return average loss over dataset."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for pc, vol in loader:
            pc  = pc.to(device)
            vol = vol.to(device).view(-1, 1)
            pred = model(pc)
            loss = criterion(pred, vol)
            total_loss += loss.item() * pc.size(0)

    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────

def train(config: dict = None):
    """
    Full training pipeline.

    Parameters
    ----------
    config : dict (optional) -- override DEFAULT_CONFIG keys

    Returns
    -------
    train_losses : list
    val_losses   : list
    cfg          : dict  -- resolved config used
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(cfg["seed"])

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n  Device        : %s" % device)

    # DataLoaders (80/20 split)
    train_loader, test_loader, _, _ = get_dataloaders(
        labels_csv = cfg["labels_csv"],
        pc_dir     = cfg["pc_dir"],
        test_split = cfg["test_split"],
        batch_size = cfg["batch_size"],
        seed       = cfg["seed"],
    )

    # Build model
    model = build_model(cfg["model_type"], cfg["dropout"]).to(device)

    # MSE loss for regression
    criterion = nn.MSELoss()

    # Adam optimiser with L2 regularisation
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # Reduce LR when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    os.makedirs(os.path.dirname(cfg["model_save"]), exist_ok=True)

    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []

    # Table header
    print("\n  %6s  %12s  %12s  %10s  %7s" % (
        "Epoch", "Train MSE", "Val MSE", "LR", "Time"))
    print("  %s  %s  %s  %s  %s" % (
        "-" * 6, "-" * 12, "-" * 12, "-" * 10, "-" * 7))

    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, cfg["grad_clip"])
        val_loss   = evaluate(model, test_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        print("  %6d  %12.6f  %12.6f  %10.2e  %6.1fs" % (
            epoch, train_loss, val_loss, current_lr, elapsed))

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_loss"   : best_val_loss,
                "config"     : cfg,
            }, cfg["model_save"])

    print("\n  Training complete. Best val MSE: %.6f" % best_val_loss)
    print("  Model saved -> %s\n" % cfg["model_save"])

    return train_losses, val_losses, cfg


if __name__ == "__main__":
    train()
