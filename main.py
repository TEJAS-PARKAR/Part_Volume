# -*- coding: utf-8 -*-
"""
main.py
-------
Entry point for the PointNet++ 3D Part Volume Prediction pipeline.

Running:
    python main.py

will automatically:
  1. Generate a synthetic dataset (if not already present)
  2. Train the PointNet++ regressor
  3. Evaluate on the test split and print metrics
  4. Run an inference demo on a freshly generated shape
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.generate_data  import generate_dataset
from utils.preprocessing  import normalize_pointcloud, shuffle_points
from models.pointnet2     import PointNet2Regressor
from models.pointnet      import PointNetRegressor
from train                import train
from evaluate             import evaluate, load_model_from_checkpoint


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

CONFIG = {
    # Data
    "data_dir"   : "data",
    "labels_csv" : os.path.join("data", "labels.csv"),
    "pc_dir"     : os.path.join("data", "pointclouds"),
    "num_samples": 200,

    # Model
    "model_save" : os.path.join("models", "best_model.pth"),
    "model_type" : "pointnet2",   # Switch to "pointnet" to use simplified version
    "dropout"    : 0.3,

    # Training
    "batch_size"   : 16,
    "num_epochs"   : 75,
    "lr"           : 1e-3,
    "weight_decay" : 1e-5,
    "grad_clip"    : 1.0,
    "test_split"   : 0.2,
    "seed"         : 42,
}


# ─────────────────────────────────────────────────────────────
# Step 1: Data generation
# ─────────────────────────────────────────────────────────────

def step_generate_data():
    if os.path.exists(CONFIG["labels_csv"]):
        print("\n[Step 1] Dataset already exists -- skipping generation.")
        return
    print("\n[Step 1] Generating synthetic dataset...")
    generate_dataset(data_dir=CONFIG["data_dir"], num_samples=CONFIG["num_samples"])


# ─────────────────────────────────────────────────────────────
# Step 2: Training
# ─────────────────────────────────────────────────────────────

def step_train():
    print("\n[Step 2] Training %s regressor..." % CONFIG["model_type"].upper())
    train_losses, val_losses, cfg = train(CONFIG)
    return train_losses, val_losses


# ─────────────────────────────────────────────────────────────
# Step 3: Evaluation
# ─────────────────────────────────────────────────────────────

def step_evaluate():
    print("\n[Step 3] Evaluating on test split...")
    metrics, y_true, y_pred = evaluate(
        model_path = CONFIG["model_save"],
        labels_csv = CONFIG["labels_csv"],
        pc_dir     = CONFIG["pc_dir"],
        batch_size = CONFIG["batch_size"],
        seed       = CONFIG["seed"],
    )
    return metrics, y_true, y_pred


# ─────────────────────────────────────────────────────────────
# Step 4: Inference demo
# ─────────────────────────────────────────────────────────────

def step_inference():
    print("\n[Step 4] Running inference demo...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and reconstruct model
    ckpt  = torch.load(CONFIG["model_save"], map_location=device)
    model, model_type = load_model_from_checkpoint(ckpt, device)

    # Generate a fresh test shape (unseen during training)
    from utils.generate_data import generate_cylinder, generate_sphere, generate_box

    np.random.seed(999)
    demo_choice = np.random.choice(["box", "sphere", "cylinder"])

    if demo_choice == "box":
        w, h, d = np.random.uniform(1.0, 3.0, 3)
        pc_raw, true_vol = generate_box(w, h, d)
        shape_desc = "Box  (w=%.3f, h=%.3f, d=%.3f)" % (w, h, d)
    elif demo_choice == "sphere":
        r = np.random.uniform(0.8, 2.0)
        pc_raw, true_vol = generate_sphere(r)
        shape_desc = "Sphere  (r=%.3f)" % r
    else:
        r = np.random.uniform(0.5, 1.5)
        h = np.random.uniform(1.0, 3.0)
        pc_raw, true_vol = generate_cylinder(r, h)
        shape_desc = "Cylinder  (r=%.3f, h=%.3f)" % (r, h)

    # Preprocess
    pc = shuffle_points(pc_raw)
    pc = normalize_pointcloud(pc)
    pc_tensor = torch.from_numpy(pc.T).unsqueeze(0).to(device)   # (1, 6, N)

    # Predict
    with torch.no_grad():
        pred_vol = model(pc_tensor).item()

    error_pct = abs(pred_vol - true_vol) / max(true_vol, 1e-8) * 100.0

    print("\n  " + "-" * 50)
    print("  Inference Demo  [model: %s]" % model_type)
    print("  " + "-" * 50)
    print("  Shape          : %s" % shape_desc)
    print("  True Volume    : %12.6f" % true_vol)
    print("  Predicted Vol  : %12.6f" % pred_vol)
    print("  Abs Error (%%)  : %11.2f%%" % error_pct)
    print("  " + "-" * 50 + "\n")


# ─────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  PointNet++ 3D Part Volume Prediction Pipeline")
    print("=" * 55)

    step_generate_data()
    step_train()
    step_evaluate()
    step_inference()

    print("\n" + "=" * 55)
    print("  Pipeline complete.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
