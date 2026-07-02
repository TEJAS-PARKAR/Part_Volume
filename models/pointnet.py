# -*- coding: utf-8 -*-
"""
pointnet.py
-----------
Simplified PointNet regressor for 3D volume prediction.

Architecture:
  Input  : (B, 6, 2048)  -- 6 channels [x,y,z,nx,ny,nz], 2048 points

  Shared MLP (Conv1d per-point operations):
    Conv1d(6   -> 64)  + BatchNorm + ReLU
    Conv1d(64  -> 128) + BatchNorm + ReLU
    Conv1d(128 -> 256) + BatchNorm

  Global Feature Extraction:
    Max-pool across point dimension -> (B, 256)

  Fully-Connected Regressor:
    Linear(256 -> 128) + BatchNorm + ReLU + Dropout
    Linear(128 ->  64) + BatchNorm + ReLU + Dropout
    Linear( 64 ->   1) + Softplus    [ensures strictly positive output]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Shared MLP (per-point feature extraction)
# ─────────────────────────────────────────────────────────────

class SharedMLP(nn.Module):
    """
    Per-point MLP implemented as Conv1d (kernel_size=1).

    Input shape:  (B, C_in, N)
    Output shape: (B, 256, N)
    """

    def __init__(self):
        super().__init__()

        # Layer 1: 6 -> 64
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(64)

        # Layer 2: 64 -> 128
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm1d(128)

        # Layer 3: 128 -> 256
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        x      : (B, 6, N)
        returns: (B, 256, N)
        """
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 64,  N)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 128, N)
        x = self.bn3(self.conv3(x))            # (B, 256, N)  -- no ReLU before pool
        return x


# ─────────────────────────────────────────────────────────────
# PointNet Regressor
# ─────────────────────────────────────────────────────────────

class PointNetRegressor(nn.Module):
    """
    PointNet-based volume regressor.

    Input  -> (B, 6, 2048)
    Output -> (B, 1)  strictly positive scalar via Softplus
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        # Per-point shared MLP
        self.shared_mlp = SharedMLP()

        # Fully-connected regressor head
        self.fc1    = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.drop1  = nn.Dropout(p=dropout_rate)

        self.fc2    = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop2  = nn.Dropout(p=dropout_rate)

        self.fc3      = nn.Linear(64, 1)

        # Softplus guarantees strictly positive volume output
        self.softplus = nn.Softplus()

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) initialisation for Conv1d and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 6, N)

        Returns
        -------
        torch.Tensor  shape (B, 1)  -- predicted volume (always > 0)
        """
        # Split coordinates and normals
        xyz = x[:, :3, :]
        normals = x[:, 3:, :]

        # Normalize xyz to unit sphere for scale invariance
        max_norm = torch.sqrt(torch.max(torch.sum(xyz**2, dim=1), dim=-1)[0])
        max_norm = torch.clamp(max_norm, min=1e-8)
        xyz_norm = xyz / max_norm.view(-1, 1, 1)

        # Recombine normalized coordinates and normals
        x_norm = torch.cat([xyz_norm, normals], dim=1)

        # Per-point feature extraction on normalized data
        feat = self.shared_mlp(x_norm)              # (B, 256, N)

        # Global max-pooling over all points
        global_feat = feat.max(dim=2).values   # (B, 256)

        # Regressor head
        out = F.relu(self.bn_fc1(self.fc1(global_feat)))   # (B, 128)
        out = self.drop1(out)

        out = F.relu(self.bn_fc2(self.fc2(out)))           # (B, 64)
        out = self.drop2(out)

        out = self.fc3(out)                                # (B, 1)
        norm_vol = self.softplus(out)                      # strictly positive

        # Scale back up to absolute volume
        return norm_vol * (max_norm.view(-1, 1) ** 3)


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity-check
    model = PointNetRegressor()
    dummy = torch.randn(4, 6, 2048)  # batch of 4
    out   = model(dummy)
    print("Input shape  : %s" % str(tuple(dummy.shape)))
    print("Output shape : %s" % str(tuple(out.shape)))
    print("Output values: %s" % str(out.squeeze().detach().numpy()))
    print("Trainable params: {:,}".format(count_parameters(model)))
    assert (out > 0).all(), "All outputs must be positive!"
    print("Sanity check passed.")
