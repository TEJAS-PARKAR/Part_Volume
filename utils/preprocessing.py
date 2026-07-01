# -*- coding: utf-8 -*-
"""
preprocessing.py
----------------
Custom PyTorch Dataset and preprocessing utilities for point cloud data.

Point cloud format per file: (2048, 6)  [x, y, z, nx, ny, nz]

Preprocessing steps:
  1. Normalise XYZ: zero-mean, then scale to unit sphere
  2. Normalise normals to unit vectors
  3. Randomly shuffle points (data augmentation)
  4. Return tensor shape (6, 2048) ready for Conv1d-based PointNet
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────

def normalize_pointcloud(pc: np.ndarray) -> np.ndarray:
    """
    Normalise a single point cloud array of shape (N, 6).

    XYZ:
      - Subtract centroid  -> zero mean
      - Divide by max L2 norm of any point -> unit sphere

    Normals:
      - Re-normalise rows to unit length

    Parameters
    ----------
    pc : np.ndarray  shape (N, 6)

    Returns
    -------
    np.ndarray  shape (N, 6), dtype float32
    """
    pc = pc.copy().astype(np.float32)

    # XYZ normalisation: ZERO-MEAN ONLY
    # We DO NOT scale to unit sphere, because absolute scale is required to predict volume!
    xyz = pc[:, :3]
    centroid = xyz.mean(axis=0)          # (3,)
    xyz -= centroid                      # zero-mean
    pc[:, :3] = xyz

    # Normal normalisation
    normals = pc[:, 3:]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pc[:, 3:] = normals / norms

    return pc


def augment_pointcloud(pc: np.ndarray, volume: float):
    """
    Apply random data augmentations for volume regression.
    1. Random Rotation (Y-axis): does not change volume.
    2. Random Scaling: scales points by s, volume by s^3.
    
    Parameters
    ----------
    pc     : np.ndarray shape (N, 6)
    volume : float
    
    Returns
    -------
    pc, volume
    """
    pc = pc.copy()
    xyz = pc[:, :3]
    normals = pc[:, 3:]

    # 1. Random Rotation (around Y-axis, common for 3D objects)
    theta = np.random.uniform(0, 2 * np.pi)
    cosval = np.cos(theta)
    sinval = np.sin(theta)
    R = np.array([
        [cosval, 0, sinval],
        [0, 1, 0],
        [-sinval, 0, cosval]
    ], dtype=np.float32)

    xyz = np.dot(xyz, R)
    normals = np.dot(normals, R)

    # 2. Random Scaling (factor between 0.8 and 1.25)
    s = np.random.uniform(0.8, 1.25)
    xyz *= s
    volume *= (s ** 3)

    pc[:, :3] = xyz
    pc[:, 3:] = normals
    return pc, volume


def shuffle_points(pc: np.ndarray) -> np.ndarray:
    """Randomly permute the N points along axis-0."""
    idx = np.random.permutation(pc.shape[0])
    return pc[idx]


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class PointCloudDataset(Dataset):
    """
    PyTorch Dataset for point-cloud volume regression.

    Parameters
    ----------
    labels_csv  : str   Path to labels.csv (columns: filename, volume, ...)
    pc_dir      : str   Directory containing .txt point cloud files
    shuffle_pts : bool  If True, randomly shuffle points each access
    """

    def __init__(self, labels_csv: str, pc_dir: str, shuffle_pts: bool = True, augment: bool = False):
        self.labels_df   = pd.read_csv(labels_csv)
        self.pc_dir      = pc_dir
        self.shuffle_pts = shuffle_pts
        self.augment     = augment

        # Filter out rows where the pointcloud file doesn't exist
        valid_rows = []
        for idx, row in self.labels_df.iterrows():
            filename = row["filename"]
            filepath = os.path.join(self.pc_dir, filename)
            
            # If not found, try alternative extension (.txt <-> .csv)
            if not os.path.exists(filepath):
                alt_filename = filename.replace('.txt', '.csv') if '.txt' in filename else filename.replace('.csv', '.txt')
                alt_filepath = os.path.join(self.pc_dir, alt_filename)
                if os.path.exists(alt_filepath):
                    row["filename"] = alt_filename
                    valid_rows.append(row)
            else:
                valid_rows.append(row)
                
        if len(valid_rows) < len(self.labels_df):
            print(f"  Warning: {len(self.labels_df) - len(valid_rows)} missing point cloud files ignored.")
            
        self.labels_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        row      = self.labels_df.iloc[idx]
        filename = row["filename"]
        volume   = float(row["volume"])

        # Load raw point cloud (N x 6)
        filepath = os.path.join(self.pc_dir, filename)
        
        # Handle both space-separated .txt and comma-separated .csv (with headers)
        if filepath.endswith('.csv'):
            pc = np.loadtxt(filepath, dtype=np.float32, delimiter=',', skiprows=1)
        else:
            pc = np.loadtxt(filepath, dtype=np.float32)  # (2048, 6)

        # Augment: shuffle points
        if self.shuffle_pts:
            pc = shuffle_points(pc)

        # Normalise XYZ to zero mean (and unit normal)
        pc = normalize_pointcloud(pc)
        
        # Apply volume-specific dynamic augmentations
        if self.augment:
            pc, volume = augment_pointcloud(pc, volume)

        # Transpose to (6, 2048) for Conv1d  [channels-first]
        pc_tensor  = torch.from_numpy(pc.T)           # (6, 2048)
        vol_tensor = torch.tensor(volume, dtype=torch.float32)

        return pc_tensor, vol_tensor


# ─────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────

def get_dataloaders(labels_csv: str, pc_dir: str,
                    test_split: float = 0.2,
                    batch_size: int = 16,
                    num_workers: int = 0,
                    seed: int = 42):
    """
    Build train and test DataLoaders with an 80/20 split.

    Parameters
    ----------
    labels_csv  : path to labels.csv
    pc_dir      : directory with .txt point cloud files
    test_split  : fraction of samples for test set  (default 0.20)
    batch_size  : samples per mini-batch            (default 16)
    num_workers : DataLoader worker processes
    seed        : random seed for reproducibility

    Returns
    -------
    (train_loader, test_loader, train_dataset, test_dataset)
    """
    from torch.utils.data import random_split

    # Full dataset (shuffle enabled for training, augment enabled)
    full_dataset = PointCloudDataset(labels_csv, pc_dir, shuffle_pts=True, augment=True)

    total    = len(full_dataset)
    n_test   = int(total * test_split)
    n_train  = total - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(full_dataset, [n_train, n_test],
                                     generator=generator)

    # For test set, disable point shuffling and augmentation for deterministic evaluation
    test_ds.dataset = PointCloudDataset(labels_csv, pc_dir, shuffle_pts=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)

    print("  Dataset split -> train: %d (augmented)  |  test: %d (deterministic)" % (n_train, n_test))
    return train_loader, test_loader, train_ds, test_ds
