# Phase 5.1: Custom Image Restoration Models Training Pipeline
**Copyright © 2026 AMAN SAH (amansah1717@gmail.com). All rights reserved.**

## Overview
Phase 5.1 focuses on training custom, lightweight image restoration models designed specifically for enhancing degraded, blurry, or low-resolution license plates (the RLPR Dataset) within tight hardware constraints (NVIDIA GTX 1650 4GB VRAM).

The pipeline utilizes PyTorch Lightning to efficiently train various model families, employing advanced data augmentation strategies to prevent overfitting on the small dataset.

## Model Architectures and Registry
A custom `ModelRegistry` dynamically instantiates models of varying complexities. The sweet spot for our 4GB VRAM environment lies between the **300K** and **9M** parameter range.

### 1. SwinIR (Swin Transformer for Image Restoration)
Uses windowed attention to achieve high-quality restoration without the massive memory overhead of global attention.
* **SwinIR Small** (138.1K params): Trained for 50 epochs. Hit a performance plateau at ~14.9 PSNR. Lacks representational capacity.
* **SwinIR Base** (326.6K params): Trained successfully. Reached **15.026 PSNR** on the validation set. Excellent balance of speed (~25-30ms) and quality.
* **SwinIR Large** (611.3K params): Features 64 base channels and 6 blocks. Higher theoretical capacity.

### 2. U-Net Variants
Standard Convolutional Encoder-Decoder architectures with skip connections.
* **UNet Lite** (944.3K params): Simple 2-level encoder/decoder.
* **UNet Standard** (8.78M params): Moderate capacity with 3-level encoder/decoder. Heavy but fits in 4GB VRAM with `batch_size=1` or `4`.
* **UNet Dense** (17.19M params): Integrates DenseNet-inspired multi-path blocks.

*(Note: U-Net architectures were patched to handle dimensional shape mismatches during concatenation by dynamically padding tensors after max-pooling odd-sized images, e.g., 52x232).*

### 3. ResNet Variants
Residual networks focused on fast feature extraction.
* **ResNet Small** (224.2K params)
* **ResNet Medium** (2.35M params)

### 4. Hybrid Attention
* **Hybrid Attention Base** (1.79M params): Combines CNN local features with Transformer global context. Initial implementations suffered from CUDA Out-Of-Memory (OOM) due to flattened global attention on high-resolution plates.

## Data Augmentation Strategy
To counter the small dataset size (200 samples) and heavily degraded input quality, a robust CPU-bound augmentation pipeline (`datasets/augmentation.py`) is used:
* **Motion Blur:** Simulates fast-moving vehicle capture using a 2D convolution kernel applied channel-wise.
* **Elastic Deformation:** Simulates physical plate distortion and camera lens warping using Gaussian-smoothed random displacement fields (`torchvision.transforms.functional.gaussian_blur`) and Grid Sampling.
* **Photometric Distortions:** Color jitter, random hue/saturation, and Gaussian noise.

## Training Dynamics & Constraints
* **Loss Function:** `RestorationLoss` (L1/MSE + SSIM/Perceptual components).
* **Hardware Bottleneck:** The GTX 1650 4GB restricts batch sizes to `1` or `4`. 
* **CPU/RAM Load:** Setting `num_workers=0` forces the intense augmentation pipeline to run on the main CPU thread, resulting in high CPU (~95%) and RAM utilization to keep the GPU fed (at ~98% Compute utilization).
* **MLflow Tracking:** Live tracking of all hyperparameters, parameter counts, and `val_psnr`/loss metrics is executed silently via the `MLFlowLogger`.

## Execution Commands
Training can be initiated via the central command-line interface:
```bash
# Example: Train U-Net Lite with batch size 4 for 50 epochs
python train_custom_models.py --model-family unet --param-scale small --max-epochs 50 --batch-size 4
```
