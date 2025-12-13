# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSIRO Biomass prediction project for Kaggle competition. Uses a two-stream DINOv2 architecture to predict plant biomass from stereo image pairs. The model predicts 5 targets: Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g with competition-weighted R² metric (weights: 0.1, 0.1, 0.1, 0.2, 0.5).

## Common Commands

### Training

```bash
# Single GPU training (auto-detects CUDA/MPS/CPU)
python -m src.train

# Apple Silicon (MPS) training
python -m src.train --device-type mps

# NVIDIA GPU (CUDA) training
python -m src.train --device-type cuda

# Multi-GPU training with torchrun (CUDA only)
torchrun --nproc_per_node=2 -m src.train

# Custom training with arguments
python -m src.train --backbone vit_base_patch14_reg4_dinov2.lvd142m \
    --epochs 50 --batch-size 16 --lr 2e-4 --cv-strategy group_month
```

### Inference

```bash
# Single ensemble (5 fold models) - auto-detects device
python -m src.inference --model-dir outputs/exp1 --output submission.csv

# Apple Silicon (MPS) inference
python -m src.inference --device-type mps --model-dir outputs/exp1 --output submission.csv

# Dual ensemble with weights
python -m src.inference --model-dir outputs/exp1 outputs/exp2 \
    --weights 0.93 0.07 --output submission.csv

# With TTA (Test Time Augmentation)
python -m src.inference --model-dir outputs/exp1 --tta --output submission.csv
```

### Hyperparameter Search

```bash
# Auto-detect device
python -m src.optuna_search --backbone vit_base_patch14_reg4_dinov2.lvd142m \
    --n-trials 50 --max-epochs 20

# Apple Silicon (MPS)
python -m src.optuna_search --device-type mps --backbone vit_base_patch14_reg4_dinov2.lvd142m \
    --n-trials 50 --max-epochs 20
```

## Architecture

### Model Architecture (`src/models.py`)

Two-stream architecture using shared DINOv2 backbone for stereo image pairs:

- **TwoStreamDINOBase**: Base class with shared backbone and separate regression heads (green, dead, clover)
- **TwoStreamDINOPlain**: Direct feature extraction without tiling
- **TwoStreamDINOTiled**: Extracts and averages features from grid tiles (default 2x2)
- **TwoStreamDINOTiledFiLM**: Tiled extraction with FiLM (Feature-wise Linear Modulation) conditioning

The model outputs positive predictions via Softplus and computes derived targets (gdm = green + clover, total = gdm + dead).

### Data Pipeline (`src/dataset.py`)

- **BiomassDataset**: Loads stereo image pairs (left/right halves of single image), applies synchronized augmentations using albumentations ReplayCompose
- **prepare_dataframe()**: Converts long-format CSV to wide format (one row per image)
- **create_folds()**: Supports multiple CV strategies: `group_month`, `group_date`, `stratified`, `random`

### Training (`src/trainer.py`, `src/train.py`)

- Multi-device support: CUDA (NVIDIA GPU), MPS (Apple Silicon), CPU
- Multi-GPU support via DistributedDataParallel (DDP) - CUDA only
- Weighted MSE loss matching competition metric
- Mixed precision training (bfloat16 for CUDA, float16 for MPS)
- Optional torch.compile for PyTorch 2.0+ (CUDA and MPS)
- Linear LR scaling for multi-GPU

### Device Utilities (`src/device.py`)

Central device management for cross-platform support:
- Auto-detection of best available device (CUDA > MPS > CPU)
- Device-specific AMP settings (autocast device type, dtype)
- Device-specific optimizer settings (fused AdamW for CUDA only)
- Platform-appropriate cache clearing and memory management

### Configuration (`src/config.py`)

TrainConfig dataclass with all hyperparameters. Key settings:
- Default backbone: `vit_base_patch14_reg4_dinov2.lvd142m`
- Default image size: 518 (DINOv2 native)
- Default grid: (2, 2) for tiled models
- Target weights for weighted R² metric
- Device type: `cuda`, `mps`, `cpu`, or `None` for auto-detect

## Data Structure

```
data/
├── train.csv      # Long format: sample_id, image_path, target_name, target
├── train/         # Training images (stereo pairs as single images)
├── test.csv
├── test/
└── sample_submission.csv
```

## Key Implementation Details

- Stereo images stored as single image files; left/right split at `width // 2`
- Synchronized augmentations for stereo pairs using ReplayCompose
- Predictions enforced non-negative via Softplus activation
- Competition metric: weighted R² with weights [0.1, 0.1, 0.1, 0.2, 0.5]
- Optimal random seed: 18 (found via seed search for fold balance)
