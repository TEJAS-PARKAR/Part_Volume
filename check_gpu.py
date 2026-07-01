# -*- coding: utf-8 -*-
"""
check_gpu.py
------------
GPU / CUDA diagnostic and setup guide for the PointNet++ pipeline.

Run this script to:
  - Check if CUDA is available
  - Get GPU name, VRAM, and CUDA version
  - Get the exact pip install command to switch to GPU PyTorch
  - Benchmark a small PointNet++ forward pass on CPU vs GPU

Usage:
    python check_gpu.py
"""

import time
import torch
import sys


def separator(title=""):
    line = "=" * 55
    if title:
        print("\n" + line)
        print("  " + title)
        print(line)
    else:
        print(line)


def check_environment():
    separator("Environment")
    print("  Python version : %s" % sys.version.split()[0])
    print("  PyTorch version: %s" % torch.__version__)
    print("  CUDA available : %s" % torch.cuda.is_available())

    if torch.cuda.is_available():
        print("  CUDA version   : %s" % torch.version.cuda)
        print("  cuDNN version  : %s" % torch.backends.cudnn.version())
        n = torch.cuda.device_count()
        print("  GPU count      : %d" % n)
        for i in range(n):
            p    = torch.cuda.get_device_properties(i)
            vram = p.total_memory / 1024 ** 3
            print("  GPU [%d]        : %s  (%.1f GB VRAM)" % (i, p.name, vram))
    else:
        _print_install_guide()


def _print_install_guide():
    separator("How to Enable GPU (CUDA) Support")

    print("""
  Your current PyTorch build is CPU-only.
  To enable GPU training (100x faster for PointNet++):

  STEP 1 -- Check your NVIDIA driver
  ------------------------------------
  Open Command Prompt and run:
      nvidia-smi

  Look for the "CUDA Version" number in the top-right corner.
  If nvidia-smi is not found, install NVIDIA drivers first:
      https://www.nvidia.com/Download/index.aspx

  STEP 2 -- Uninstall CPU PyTorch
  ---------------------------------
      pip uninstall torch torchvision torchaudio -y

  STEP 3 -- Install CUDA PyTorch
  --------------------------------
  Go to: https://pytorch.org/get-started/locally/
  Select:  PyTorch Build = Stable
           OS            = Windows
           Package       = Pip
           Language      = Python
           Compute       = CUDA 12.x  (or match your nvidia-smi version)

  The generated command will look like:
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

  For CUDA 11.8 (older GPU):
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

  STEP 4 -- Verify
  ------------------
      python check_gpu.py

  MINIMUM GPU REQUIREMENTS for this project:
      VRAM   : 4 GB  (8+ GB recommended for batch_size=32)
      Driver : 520+  (for CUDA 12.x)

  RECOMMENDED GPUs (common for workstations):
      NVIDIA RTX 3060 / 3070 / 3080 / 3090
      NVIDIA RTX 4060 / 4070 / 4080 / 4090
      NVIDIA Quadro / Tesla (if available)
    """)


def benchmark():
    separator("Speed Benchmark")

    from models.pointnet2 import PointNet2Regressor

    BATCH = 4
    N     = 2048
    dummy = torch.randn(BATCH, 6, N)

    # CPU benchmark
    model_cpu = PointNet2Regressor()
    model_cpu.eval()
    t0 = time.time()
    with torch.no_grad():
        _ = model_cpu(dummy)
    cpu_time = time.time() - t0
    print("  CPU forward pass (B=%d, N=%d): %.2f s" % (BATCH, N, cpu_time))

    # Extrapolate training time
    steps_per_epoch = 200 // BATCH  # 200 samples / batch_size
    epoch_est = cpu_time * steps_per_epoch * 2  # x2 for backward
    print("  Estimated CPU epoch time     : %.1f - %.1f min" % (
        epoch_est / 60 * 0.8, epoch_est / 60 * 1.4))
    print("  Estimated 75 epochs (CPU)    : %.1f - %.1f hrs" % (
        epoch_est * 75 / 3600 * 0.8, epoch_est * 75 / 3600 * 1.4))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_gpu = PointNet2Regressor().to(device)
        model_gpu.eval()
        dummy_gpu = dummy.to(device)

        # Warmup
        with torch.no_grad():
            _ = model_gpu(dummy_gpu)
        torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = model_gpu(dummy_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - t0) / 5

        print("\n  GPU forward pass (B=%d, N=%d): %.4f s" % (BATCH, N, gpu_time))
        speedup = cpu_time / gpu_time
        print("  GPU speedup                  : %.0fx faster" % speedup)

        gpu_epoch = gpu_time * steps_per_epoch * 2
        print("  Estimated GPU epoch time     : %.1f s" % gpu_epoch)
        print("  Estimated 75 epochs (GPU)    : %.1f min" % (gpu_epoch * 75 / 60))
    else:
        print("\n  GPU not available -- install CUDA PyTorch for 50-100x speedup.")


def main():
    separator("GPU / CUDA Setup Check")
    check_environment()
    print()
    benchmark()
    separator()
    print()


if __name__ == "__main__":
    main()
