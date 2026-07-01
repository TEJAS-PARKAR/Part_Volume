# -*- coding: utf-8 -*-
"""
evaluate.py
-----------
Evaluation pipeline for trained PointNet / PointNet++ volume regressors.

Metrics computed:
  - MSE   (Mean Squared Error)
  - RMSE  (Root Mean Squared Error)
  - MAE   (Mean Absolute Error)
  - R2    (Coefficient of Determination)

Usage (standalone):
    python evaluate.py
"""

import os
import torch
import numpy as np

from models.pointnet      import PointNetRegressor
from models.pointnet2     import PointNet2Regressor
from utils.preprocessing  import get_dataloaders


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def mean_squared_error(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


# ─────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────

def predict(model, loader, device):
    """
    Run the model on every batch in `loader`.

    Returns
    -------
    y_true, y_pred : np.ndarray  shape (N,)
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for pc, vol in loader:
            pc   = pc.to(device)
            pred = model(pc).squeeze(1).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(vol.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


# ─────────────────────────────────────────────────────────────
# Load correct model class from checkpoint
# ─────────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt, device):
    """
    Read model_type from the checkpoint config and reconstruct the model.

    Parameters
    ----------
    ckpt   : dict  -- loaded torch checkpoint
    device : torch.device

    Returns
    -------
    model : nn.Module (eval mode, weights loaded)
    """
    config     = ckpt.get("config", {})
    model_type = config.get("model_type", "pointnet")   # default for old checkpoints
    dropout    = config.get("dropout", 0.3)

    if model_type == "pointnet2":
        model = PointNet2Regressor(dropout_rate=dropout)
    else:
        model = PointNetRegressor(dropout_rate=dropout)

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, model_type


# ─────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────

def evaluate(model_path: str = None,
             labels_csv : str = None,
             pc_dir     : str = None,
             batch_size : int = 16,
             seed       : int = 42):
    """
    Load the best-saved model and compute evaluation metrics on the test split.

    Parameters
    ----------
    model_path  : path to .pth checkpoint
    labels_csv  : path to labels.csv
    pc_dir      : directory with .txt point cloud files
    batch_size  : inference batch size
    seed        : same seed as training (reproduces the same split)

    Returns
    -------
    metrics       : dict  with keys mse, rmse, mae, r2
    y_true, y_pred: np.ndarray
    """
    model_path = model_path or os.path.join("models", "best_model.pth")
    labels_csv = labels_csv or os.path.join("data",   "labels.csv")
    pc_dir     = pc_dir     or os.path.join("data",   "pointclouds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No checkpoint found at '%s'. Please run train.py first." % model_path
        )

    ckpt   = torch.load(model_path, map_location=device)
    config = ckpt.get("config", {})

    # Reconstruct correct model architecture from checkpoint metadata
    model, model_type = load_model_from_checkpoint(ckpt, device)

    print("\n  Loaded model  : %s" % model_type)
    print("  Checkpoint    : '%s'" % model_path)
    print("  Trained epoch : %s  |  Best val MSE : %.6f" % (
        ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan"))))

    # DataLoaders (same seed -> same train/test split)
    _, test_loader, _, _ = get_dataloaders(
        labels_csv = labels_csv,
        pc_dir     = pc_dir,
        test_split = config.get("test_split", 0.2),
        batch_size = batch_size,
        seed       = seed,
    )

    # Collect predictions
    y_true, y_pred = predict(model, test_loader, device)

    # Compute metrics
    mse  = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    print("\n  " + "-" * 40)
    print("  Evaluation Results on Test Split")
    print("  " + "-" * 40)
    print("  MSE   : %12.6f" % mse)
    print("  RMSE  : %12.6f" % rmse)
    print("  MAE   : %12.6f" % mae)
    print("  R2    : %12.6f" % r2)
    print("  " + "-" * 40 + "\n")

    return metrics, y_true, y_pred


if __name__ == "__main__":
    evaluate()
