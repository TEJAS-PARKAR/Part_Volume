# -*- coding: utf-8 -*-
"""
generate_data.py
----------------
Synthetic dataset generator for 3D part volume prediction.

Generates point clouds for three shape types:
  - Box (rectangular cuboid)
  - Sphere
  - Cylinder

Each point cloud has shape (2048, 6): [x, y, z, nx, ny, nz]
Volumes are computed analytically and saved to labels.csv.
"""

import os
import math
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Shape Generators
# ─────────────────────────────────────────────────────────────

def generate_sphere(radius: float, num_points: int = 2048):
    """
    Sample `num_points` uniformly on a sphere surface.
    Outward unit normals point radially from the centre.
    Volume = (4/3) * pi * r^3
    """
    # Uniform points on sphere via normalised Gaussian
    coords = np.random.randn(num_points, 3).astype(np.float32)
    norms  = np.linalg.norm(coords, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1.0, norms)

    normals = coords / norms           # outward unit normals
    points  = normals * radius         # scale to radius

    volume = (4.0 / 3.0) * math.pi * radius ** 3
    return np.hstack((points, normals)), volume


def generate_box(w: float, h: float, d: float, num_points: int = 2048):
    """
    Sample `num_points` uniformly on the surface of an axis-aligned box.
    Points distributed across 6 faces proportional to face area.
    Normals are outward axis-aligned unit vectors.
    Volume = w * h * d
    """
    area_yz = h * d    # +/-x faces
    area_xz = w * d    # +/-y faces
    area_xy = w * h    # +/-z faces

    areas = np.array([area_yz, area_yz, area_xz, area_xz, area_xy, area_xy])
    probs = areas / areas.sum()

    face_ids = np.random.choice(6, size=num_points, p=probs)

    points  = np.zeros((num_points, 3), dtype=np.float32)
    normals = np.zeros((num_points, 3), dtype=np.float32)

    u = np.random.uniform(-0.5, 0.5, num_points)
    v = np.random.uniform(-0.5, 0.5, num_points)

    # (face_id, normal_sign, normal_axis, u_axis, v_axis, u_scale, v_scale, fixed)
    face_defs = [
        (0, -1, 0, 1, 2, h, d, -w / 2),   # left  (-x)
        (1,  1, 0, 1, 2, h, d,  w / 2),   # right (+x)
        (2, -1, 1, 0, 2, w, d, -h / 2),   # bottom(-y)
        (3,  1, 1, 0, 2, w, d,  h / 2),   # top   (+y)
        (4, -1, 2, 0, 1, w, h, -d / 2),   # back  (-z)
        (5,  1, 2, 0, 1, w, h,  d / 2),   # front (+z)
    ]

    for face, sign, axis, ua, ub, ua_scale, ub_scale, fixed_val in face_defs:
        mask = face_ids == face
        n    = mask.sum()
        if n == 0:
            continue
        pts = np.zeros((n, 3), dtype=np.float32)
        pts[:, axis] = fixed_val
        pts[:, ua]   = u[mask] * ua_scale
        pts[:, ub]   = v[mask] * ub_scale
        points[mask]  = pts
        normals[mask, axis] = sign

    volume = w * h * d
    return np.hstack((points, normals)), volume


def generate_cylinder(radius: float, height: float, num_points: int = 2048):
    """
    Sample `num_points` uniformly on a cylinder surface (side + two caps).
    Axis is along Z.
    Volume = pi * r^2 * h
    """
    area_side = 2.0 * math.pi * radius * height
    area_cap  = math.pi * radius ** 2

    areas    = np.array([area_side, area_cap, area_cap])
    probs    = areas / areas.sum()
    part_ids = np.random.choice(3, size=num_points, p=probs)
    theta    = np.random.uniform(0, 2.0 * math.pi, num_points)

    points  = np.zeros((num_points, 3), dtype=np.float32)
    normals = np.zeros((num_points, 3), dtype=np.float32)

    # Side surface
    mask = part_ids == 0
    n = mask.sum()
    if n:
        t = theta[mask]
        points[mask]  = np.column_stack([
            radius * np.cos(t),
            radius * np.sin(t),
            np.random.uniform(-height / 2, height / 2, n)
        ])
        normals[mask] = np.column_stack([np.cos(t), np.sin(t), np.zeros(n)])

    # Top cap (z = +h/2)
    mask = part_ids == 1
    n = mask.sum()
    if n:
        t   = theta[mask]
        r_s = radius * np.sqrt(np.random.uniform(0, 1, n))
        points[mask]     = np.column_stack(
            [r_s * np.cos(t), r_s * np.sin(t), np.full(n, height / 2)])
        normals[mask, 2] = 1.0

    # Bottom cap (z = -h/2)
    mask = part_ids == 2
    n = mask.sum()
    if n:
        t   = theta[mask]
        r_s = radius * np.sqrt(np.random.uniform(0, 1, n))
        points[mask]     = np.column_stack(
            [r_s * np.cos(t), r_s * np.sin(t), np.full(n, -height / 2)])
        normals[mask, 2] = -1.0

    volume = math.pi * radius ** 2 * height
    return np.hstack((points, normals)), volume


# ─────────────────────────────────────────────────────────────
# Dataset Generator
# ─────────────────────────────────────────────────────────────

def generate_dataset(data_dir: str = "data", num_samples: int = 200,
                     num_points: int = 2048, seed: int = 42):
    """
    Generate a synthetic dataset of 3D shape point clouds.

    Saves:
      - Point clouds  -> data_dir/pointclouds/part_XXXX.txt
      - Labels CSV    -> data_dir/labels.csv
    """
    np.random.seed(seed)

    pc_dir = os.path.join(data_dir, "pointclouds")
    os.makedirs(pc_dir, exist_ok=True)

    records = []
    shapes  = ["box", "sphere", "cylinder"]

    print("\n" + "=" * 55)
    print("  Generating %d synthetic 3D shapes..." % num_samples)
    print("=" * 55)

    for i in range(num_samples):
        shape_type = np.random.choice(shapes)

        if shape_type == "box":
            w, h, d = np.random.uniform(0.5, 4.0, 3)
            data, vol = generate_box(w, h, d, num_points)

        elif shape_type == "sphere":
            r = np.random.uniform(0.4, 2.5)
            data, vol = generate_sphere(r, num_points)

        else:  # cylinder
            r = np.random.uniform(0.4, 2.0)
            h = np.random.uniform(0.5, 4.0)
            data, vol = generate_cylinder(r, h, num_points)

        filename = "part_%04d.txt" % i
        np.savetxt(os.path.join(pc_dir, filename), data,
                   fmt="%.6f", delimiter=" ")

        records.append({"filename": filename, "volume": round(vol, 6),
                        "shape_type": shape_type})

        if (i + 1) % 50 == 0:
            print("  [%d/%d] shapes generated..." % (i + 1, num_samples))

    df = pd.DataFrame(records)
    csv_path = os.path.join(data_dir, "labels.csv")
    df.to_csv(csv_path, index=False)

    print("\n  Labels saved  -> %s" % csv_path)
    print("  Pointclouds  -> %s" % pc_dir)
    print("\n  Shape distribution:")
    print(df["shape_type"].value_counts().to_string())
    print("\n  Volume stats:")
    print("  min=%.4f  max=%.4f  mean=%.4f" % (
        df["volume"].min(), df["volume"].max(), df["volume"].mean()))
    print("=" * 55 + "\n")


if __name__ == "__main__":
    generate_dataset()
