# -*- coding: utf-8 -*-
"""
pointnet2.py
------------
PointNet++ regressor for 3D volume prediction on complex part geometries.

Optimizations over naive implementation:
  - FPS uses fully batched torch ops (no Python inner loops over points)
  - Ball query vectorised with a single pairwise distance matrix
  - All ops stay on the target device (CPU or CUDA GPU) throughout
  - No data movement between device and host during forward pass

Architecture:
  Input: (B, 6, N)  -- [x, y, z, nx, ny, nz]

  Set Abstraction 1  (fine-scale, local)
    FPS : 2048 -> 512 centroids
    Ball: radius=0.2, 32 neighbours
    MLP : [6 -> 64 -> 128]
    Out : (B, 512, 128)

  Set Abstraction 2  (medium-scale)
    FPS : 512  -> 128 centroids
    Ball: radius=0.4, 64 neighbours
    MLP : [131 -> 128 -> 256]
    Out : (B, 128, 256)

  Set Abstraction 3  (global)
    All 128 points -> single group
    MLP : [259 -> 256 -> 512 -> 1024]
    Out : (B, 1024)

  FC Regression Head:
    Linear(1024->512) + BN + ReLU + Dropout
    Linear(512 ->256) + BN + ReLU + Dropout
    Linear(256 ->  1) + Softplus   [strictly positive]

GPU Usage:
  The model automatically runs on GPU if CUDA is available.
  Install CUDA PyTorch: https://pytorch.org/get-started/locally/
  Then simply run: python main.py   -- device selection is automatic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Geometry primitives  (fully vectorised, device-agnostic)
# ─────────────────────────────────────────────────────────────

def square_distance(src, dst):
    """
    Pairwise squared Euclidean distances.

    Parameters
    ----------
    src : (B, N, C)
    dst : (B, M, C)

    Returns
    -------
    (B, N, M)
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    dist  = -2.0 * torch.bmm(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1, keepdim=True).permute(0, 2, 1)
    return dist.clamp(min=0.0)   # numerical safety: avoid tiny negatives


def index_points(points, idx):
    """
    Gather points by an index tensor (supports 1-D and 2-D idx).

    Parameters
    ----------
    points : (B, N, C)
    idx    : (B, S)  or  (B, S, K)

    Returns
    -------
    (B, S, C)  or  (B, S, K, C)
    """
    B      = points.shape[0]
    device = points.device

    # Build batch indices that broadcast over idx shape
    view   = [B] + [1] * (idx.dim() - 1)
    repeat = [1] + list(idx.shape[1:])
    b_idx  = (torch.arange(B, dtype=torch.long, device=device)
               .view(view).repeat(repeat))
    return points[b_idx, idx, :]


# ─────────────────────────────────────────────────────────────
# Optimised Farthest Point Sampling
# ─────────────────────────────────────────────────────────────

def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS).

    Iteratively selects the point farthest from the already-chosen set,
    giving maximally spread centroid coverage of the point cloud.

    Optimisations vs naive implementation
    --------------------------------------
    - Distance computation is a single batched tensor op per iteration
      (no Python-level loops over points or batches).
    - All tensors stay on `xyz.device` throughout -- zero host/device transfers.
    - Works identically on CPU and CUDA GPU; on GPU each distance step is
      fully parallelised across all N points simultaneously.

    Time complexity : O(npoint * N)  tensor ops, O(npoint) Python iterations
    CPU estimate    : ~5-8 s per epoch for N=2048, npoint=512  (batch=16)
    GPU estimate    : ~0.05-0.1 s per epoch (100x faster on CUDA)

    Parameters
    ----------
    xyz    : (B, N, 3)
    npoint : int

    Returns
    -------
    centroids : (B, npoint)  long tensor of selected indices
    """
    B, N, _  = xyz.shape
    device   = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    # Track minimum distance of every point to the chosen set
    min_dists = torch.full((B, N), float('inf'), device=device)
    # Start from a random point per batch element
    current   = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = current

        # Current centroid coords: (B, 1, 3)
        c_xyz = xyz[torch.arange(B, device=device), current, :].unsqueeze(1)

        # Squared distance from this centroid to ALL points: (B, N)
        d = torch.sum((xyz - c_xyz) ** 2, dim=-1)

        # Update minimum distance to the growing centroid set
        min_dists = torch.minimum(min_dists, d)

        # Next centroid = point farthest from the set
        current = min_dists.argmax(dim=-1)

    return centroids


# ─────────────────────────────────────────────────────────────
# Optimised Ball Query
# ─────────────────────────────────────────────────────────────

def ball_query(radius, nsample, xyz, new_xyz):
    """
    For each centroid in `new_xyz`, find up to `nsample` neighbours
    within `radius` in `xyz`.  Fully vectorised -- one pairwise
    distance matrix, then threshold + sort.

    Parameters
    ----------
    radius  : float
    nsample : int
    xyz     : (B, N, 3)
    new_xyz : (B, S, 3)

    Returns
    -------
    group_idx : (B, S, nsample) long tensor of indices into xyz
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device  = xyz.device

    # Full pairwise squared distances: (B, S, N)
    sq_dists  = square_distance(new_xyz, xyz)

    # Mask out-of-radius points with sentinel N
    group_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, S, N).clone()
    group_idx[sq_dists > radius ** 2] = N      # sentinel

    # Sort so valid indices (< N) come first; take first nsample
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # (B, S, nsample)

    # If a ball had < nsample valid points, repeat the first valid one
    group_first         = group_idx[:, :, 0:1].expand_as(group_idx)
    mask                = group_idx == N
    group_idx           = group_idx.clone()
    group_idx[mask]     = group_first[mask]

    return group_idx


# ─────────────────────────────────────────────────────────────
# Set Abstraction Layer
# ─────────────────────────────────────────────────────────────

class SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction (SA) module.

    Per forward pass:
      1. FPS  -- select `npoint` spread-out centroids
      2. Ball query -- group `nsample` neighbours per centroid
      3. Subtract centroid -> local relative coordinates
      4. Shared MLP (Conv2d) per grouped point
      5. Max-pool within each group -> one descriptor per centroid

    Parameters
    ----------
    npoint     : int   -- centroids to sample
    radius     : float -- ball radius (normalised unit-sphere coords)
    nsample    : int   -- max neighbours per ball
    in_channel : int   -- input feature channels fed into the MLP
    mlp        : list  -- MLP output channel sizes  e.g. [64, 128]
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        prev = in_channel
        for ch in mlp:
            self.convs.append(nn.Conv2d(prev, ch, kernel_size=1, bias=False))
            self.bns.append(nn.BatchNorm2d(ch))
            prev = ch

    def forward(self, xyz, features):
        """
        Parameters
        ----------
        xyz      : (B, N, 3)
        features : (B, N, C) or None

        Returns
        -------
        new_xyz      : (B, npoint, 3)
        new_features : (B, npoint, mlp[-1])
        """
        # 1. Sample centroids
        fps_idx  = farthest_point_sample(xyz, self.npoint)   # (B, npoint)
        new_xyz  = index_points(xyz, fps_idx)                # (B, npoint, 3)

        # 2. Find neighbours for each centroid
        idx      = ball_query(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, K)

        # 3. Gather and compute relative coordinates
        grp_xyz  = index_points(xyz, idx)                   # (B, npoint, K, 3)
        grp_xyz  = grp_xyz - new_xyz.unsqueeze(2)           # local coords

        # 4. Concatenate relative xyz with point features
        if features is not None:
            grp_feat = index_points(features, idx)           # (B, npoint, K, C)
            grouped  = torch.cat([grp_xyz, grp_feat], dim=-1)
        else:
            grouped  = grp_xyz                               # (B, npoint, K, 3)

        # 5. Conv2d shared MLP: expects (B, C, npoint, K)
        grouped = grouped.permute(0, 3, 1, 2).contiguous()

        for conv, bn in zip(self.convs, self.bns):
            grouped = F.relu(bn(conv(grouped)))

        # 6. Max-pool within each group
        new_feat = grouped.max(dim=-1)[0]                    # (B, C_out, npoint)
        new_feat = new_feat.permute(0, 2, 1)                 # (B, npoint, C_out)

        return new_xyz, new_feat


class GlobalSetAbstraction(nn.Module):
    """
    Global SA: treats ALL remaining points as one group.
    Produces a single global feature vector via max-pooling.

    Parameters
    ----------
    in_channel : int   -- input channels (xyz concatenated with features)
    mlp        : list  -- MLP output channel sizes
    """

    def __init__(self, in_channel, mlp):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        prev = in_channel
        for ch in mlp:
            self.convs.append(nn.Conv1d(prev, ch, kernel_size=1, bias=False))
            self.bns.append(nn.BatchNorm1d(ch))
            prev = ch

    def forward(self, xyz, features):
        """
        Parameters
        ----------
        xyz      : (B, N, 3)
        features : (B, N, C)

        Returns
        -------
        (B, mlp[-1])
        """
        if features is not None:
            combined = torch.cat([xyz, features], dim=-1)  # (B, N, 3+C)
        else:
            combined = xyz

        x = combined.permute(0, 2, 1).contiguous()         # (B, 3+C, N)

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))

        return x.max(dim=-1)[0]                            # (B, mlp[-1])


# ─────────────────────────────────────────────────────────────
# PointNet++ Regressor
# ─────────────────────────────────────────────────────────────

class PointNet2Regressor(nn.Module):
    """
    PointNet++ volume regressor.

    Input  : (B, 6, N)   [x, y, z, nx, ny, nz]
    Output : (B, 1)      strictly positive volume (Softplus activation)

    GPU Usage
    ---------
    The model and DataLoader are device-agnostic.
    To use GPU, install CUDA-enabled PyTorch:
        https://pytorch.org/get-started/locally/
    Then run normally -- device is auto-detected in train.py / main.py.
    """

    def __init__(self, dropout_rate=0.3):
        super().__init__()

        # SA1: 2048 -> 512 pts, fine local features
        # in_channel = relative_xyz(3) + normals(3) = 6
        self.sa1 = SetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=6, mlp=[64, 64, 128]
        )

        # SA2: 512 -> 128 pts, medium features
        # in_channel = relative_xyz(3) + sa1_features(128) = 131
        self.sa2 = SetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )

        # SA3: global, all 128 pts -> 1024-d vector
        # in_channel = xyz(3) + sa2_features(256) = 259
        self.sa3 = GlobalSetAbstraction(
            in_channel=256 + 3, mlp=[256, 512, 1024]
        )

        # FC regression head
        self.fc1   = nn.Linear(1024, 512)
        self.bn1   = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=dropout_rate)

        self.fc2   = nn.Linear(512, 256)
        self.bn2   = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=dropout_rate)

        self.fc3      = nn.Linear(256, 1)
        self.softplus = nn.Softplus()

        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) normal initialisation for all conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, 6, N)   DataLoader output [channels-first]

        Returns
        -------
        (B, 1)  -- predicted volume, always > 0
        """
        # Transpose to (B, N, 6) for neighbourhood operations
        x       = x.permute(0, 2, 1).contiguous()
        xyz     = x[:, :, :3]                          # (B, N, 3)
        normals = x[:, :, 3:]                          # (B, N, 3)

        # Normalize xyz to unit sphere so fixed radii (0.2, 0.4) work at any scale
        max_norm = torch.sqrt(torch.max(torch.sum(xyz**2, dim=-1), dim=-1)[0])
        max_norm = torch.clamp(max_norm, min=1e-8)
        xyz_norm = xyz / max_norm.view(-1, 1, 1)

        # Hierarchical local feature learning (on normalized coordinates!)
        xyz1, feat1 = self.sa1(xyz_norm, normals)            # (B, 512, 128)
        xyz2, feat2 = self.sa2(xyz1, feat1)             # (B, 128, 256)
        global_feat = self.sa3(xyz2, feat2)             # (B, 1024)

        # Regression to normalized volume
        out = F.relu(self.bn1(self.fc1(global_feat)))
        out = self.drop1(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.drop2(out)
        norm_vol = self.softplus(self.fc3(out))         # (B, 1), always > 0
        
        # Scale back up to absolute volume
        return norm_vol * (max_norm.view(-1, 1) ** 3)


def count_parameters(model):
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model = PointNet2Regressor().to(device)
    dummy = torch.randn(2, 6, 2048).to(device)

    t0  = time.time()
    out = model(dummy)
    print("Forward pass   : %.2f s" % (time.time() - t0))
    print("Output shape   : %s" % str(tuple(out.shape)))
    print("Output values  : %s" % str(out.squeeze().detach().cpu().numpy()))
    print("Params         : {:,}".format(count_parameters(model)))
    assert (out > 0).all()
    print("Sanity check PASSED.")
