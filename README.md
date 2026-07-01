# PointNet++ 3D Part Volume Prediction

A robust, GPU-accelerated PyTorch deep-learning pipeline that predicts the **absolute volume of complex 3D parts** purely from their surface point clouds using the **PointNet++ (PointNet2)** architecture.

---

## Project Structure

```
part_volume/
│
├── data/
│   ├── pointclouds/              ← Mixed .txt and .csv files (2048 × 6: x y z nx ny nz)
│   └── labels.csv                ← filename, volume, shape_type
│
├── models/
│   ├── pointnet.py               ← Original Simple PointNet (Deprecated)
│   ├── pointnet2.py              ← PointNet++ Regressor (Active)
│   └── best_model.pth            ← Saved best checkpoint
│
├── utils/
│   ├── generate_data.py          ← Synthetic dataset generator
│   └── preprocessing.py          ← Dataset loader, Normalisation, & Augmentations
│
├── check_gpu.py                  ← GPU capability & speed benchmarking tool
├── train.py                      ← Training pipeline
├── evaluate.py                   ← Evaluation + metrics
├── main.py                       ← Full pipeline entry point
└── README.md
```

---

## Quick Start

### 1. Install dependencies
Ensure you have CUDA 12.8 installed (or compatible), then install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Verify GPU Acceleration
Run the diagnostic tool to benchmark your CPU vs GPU performance:
```bash
python check_gpu.py
```

### 3. Run the full pipeline
```bash
python main.py
```

This will:
1. **Load/Generate** point clouds (Supports both `.txt` and `.csv` drops).
2. **Train** the PointNet++ model for 75 epochs (using dynamic data augmentation).
3. **Evaluate** on the test split (MSE, RMSE, MAE, R²).
4. **Inference demo** on a fresh shape.

---

## Architecture: PointNet++ (PointNet2)

The pipeline uses an optimized **PointNet++** architecture (1.46M parameters) to learn complex geometric features (ribs, cavities, flanges) which simple PointNet cannot capture. 

It utilizes **Farthest Point Sampling (FPS)** and **Ball Query** grouping to build local hierarchical features before passing them to the global regression head.

---

## Input Format

The pipeline is highly robust and seamlessly supports both space-separated `.txt` and comma-separated `.csv` (with headers) formats.

Each file is a matrix of shape **(2048, 6)**:

| x | y | z | nx | ny | nz |
|---|---|---|----|----|----|
| float | float | float | float | float | float |

*Note: If `labels.csv` specifies a `.txt` file but only a `.csv` exists in the folder, the dataloader will automatically find and parse the `.csv`.*

---

## Preprocessing & Data Augmentation

Predicting **absolute volume** requires preserving the physical scale of the object. Therefore, point clouds are **NOT** scaled to a unit sphere. 

| Step | Operation |
|------|-----------|
| XYZ centering | Subtract centroid → **zero-mean (Translation Invariance)** |
| Scale Preservation | Absolute coordinates are preserved to maintain volume data |
| Normal enforcement | Re-normalise normals to unit vectors |
| Augmentation: Rotation | Random 3D rotation around Y-axis per epoch |
| Augmentation: Scaling | Random scaling by factor $s \in [0.8, 1.25]$, volume target adjusted by $s^3$ |

**Dynamic Augmentation:** The random rotation and scaling act as an infinite dataset multiplier, preventing overfitting and dramatically improving the R² score on small datasets.

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Architecture | PointNet++ (pointnet2) |
| Loss | MSELoss |
| Optimiser | Adam |
| Learning Rate | 1e-3 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=8) |
| Epochs | 75 |
| Batch Size | 16 |
| Train/Test Split | 80 / 20 |
| Device | Auto-detect (CUDA preferred) |

---

## Inference on a Custom Point Cloud

```python
import torch
import numpy as np
from models.pointnet2 import PointNet2Regressor
from utils.preprocessing import normalize_pointcloud

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt  = torch.load("models/best_model.pth", map_location=device)
model = PointNet2Regressor().to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Load your point cloud (2048, 6)
pc = np.loadtxt("path/to/your_part.csv", dtype=np.float32, delimiter=',', skiprows=1)

# Preprocess (Zero-mean ONLY)
pc = normalize_pointcloud(pc)
pc_tensor = torch.from_numpy(pc.T).unsqueeze(0).to(device)  # (1, 6, 2048)

# Predict
with torch.no_grad():
    pred_volume = model(pc_tensor).item()

print(f"Predicted Volume: {pred_volume:.6f}")
```
