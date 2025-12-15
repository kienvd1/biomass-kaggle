# 5-Head DINOv2 Training & Evaluation Arguments

Comprehensive documentation for `train_5head.py` and `eval_5head_oof.py`.

---

## Table of Contents

1. [Training Script (`train_5head.py`)](#training-script-train_5headpy)
   - [Paths](#paths)
   - [Model Architecture](#model-architecture)
   - [Training Configuration](#training-configuration)
   - [Learning Rates](#learning-rates)
   - [2-Stage Training](#2-stage-training)
   - [Auxiliary Heads (Multi-Task Learning)](#auxiliary-heads-multi-task-learning)
   - [Loss Function](#loss-function)
   - [Post-Processing](#post-processing)
   - [Augmentation](#augmentation)
   - [Target Normalization](#target-normalization)
   - [MixUp/CutMix](#mixupcutmix)
   - [AMP (Automatic Mixed Precision)](#amp-automatic-mixed-precision)
   - [Cross-Validation](#cross-validation)
   - [Device & Misc](#device--misc)
2. [Evaluation Script (`eval_5head_oof.py`)](#evaluation-script-eval_5head_oofpy)
3. [Example Commands](#example-commands)

---

## Training Script (`train_5head.py`)

```bash
python -m src.train_5head [OPTIONS]
```

### Paths

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--base-path` | str | `./data` | Base directory containing `train.csv` and `train/` image folder |
| `--output-dir` | str | Auto-generated | Output directory for checkpoints and logs. If not specified, creates `./outputs/5head_YYYYMMDD_HHMMSS/` |

### Model Architecture

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--backbone` | str | `vit_base_patch14_reg4_dinov2.lvd142m` | DINOv2 backbone model name from timm. Options: `vit_small_patch14_reg4_dinov2.lvd142m`, `vit_base_patch14_reg4_dinov2.lvd142m`, `vit_large_patch14_reg4_dinov2.lvd142m` |
| `--grid` | int int | `2 2` | Tile grid for processing large images. `2 2` splits each image into 4 tiles. Higher grids capture more detail but use more memory |
| `--dropout` | float | `0.2` | Dropout rate in prediction heads. Range: 0.0-0.5. Higher values reduce overfitting |
| `--hidden-ratio` | float | `0.5` | Hidden layer size as ratio of combined feature dim. `0.5` means hidden_dim = feat_dim × 2 × 0.5 |
| `--no-film` | flag | False | Disable FiLM (Feature-wise Linear Modulation) conditioning between stereo views. FiLM helps the model learn cross-view relationships |
| `--no-attention-pool` | flag | False | Disable attention pooling. Uses mean pooling instead. Attention pooling learns to weight important spatial regions |
| `--grad-ckpt` | flag | False | Enable gradient checkpointing to reduce memory usage at cost of ~20% slower training |

### Training Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | `50` | Total training epochs |
| `--batch-size` | int | `8` | Batch size per GPU. Reduce if OOM. Effective batch = batch_size × grad_accum |
| `--num-workers` | int | `4` | DataLoader workers. Set to 0 if using `--cache-images` |
| `--cache-images` | flag | False | Cache all images in RAM. Faster training but requires ~16GB+ free RAM |
| `--prefetch-factor` | int | `4` | DataLoader prefetch factor. Higher = more memory, slightly faster |
| `--grad-accum` | int | `1` | Gradient accumulation steps. Use >1 to simulate larger batch sizes |
| `--grad-clip` | float | `1.0` | Gradient clipping norm. Prevents exploding gradients. Recommended: 0.5-1.0 |
| `--patience` | int | `10` | Early stopping patience. Stops training after N epochs without improvement |

### Learning Rates

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | float | `2e-4` | Head learning rate for stage 2 (full finetuning) or single-stage training |
| `--backbone-lr` | float | `1e-5` | Backbone learning rate for stage 2. Should be 10-100x smaller than head LR |
| `--weight-decay` | float | `0.01` | AdamW weight decay for regularization |
| `--warmup-epochs` | int | `2` | Linear warmup epochs before cosine annealing |

### 2-Stage Training

Two-stage training freezes the backbone first to train only the heads, then unfreezes for full finetuning.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--two-stage` | flag | False | Enable 2-stage training (recommended for best results) |
| `--freeze-epochs` | int | `5` | Number of epochs in stage 1 (frozen backbone) |
| `--head-lr-stage1` | float | `1e-3` | Head learning rate for stage 1. Can be higher since backbone is frozen |
| `--freeze-backbone` | flag | False | Keep backbone frozen for ALL epochs (head-only training). Useful for quick experiments or when backbone is already well-pretrained |

**Training Modes:**
- **Single-stage** (default): Train backbone + heads together from start
- **Two-stage** (`--two-stage`): Stage 1 trains heads only, Stage 2 finetunes everything
- **Head-only** (`--freeze-backbone`): Never unfreeze backbone, fastest training

### Auxiliary Heads (Multi-Task Learning)

Auxiliary classification heads that provide additional supervision signals.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-aux-heads` | flag | False | Enable State, Month, and Species classification heads |
| `--aux-state-weight` | float | `5.0` | Loss weight for State classification (4 classes: NSW, Tas, Vic, WA) |
| `--aux-month-weight` | float | `3.0` | Loss weight for Month classification (10 classes) |
| `--aux-species-weight` | float | `2.0` | Loss weight for Species classification (8 groups) |
| `--apply-context-adjustment` | flag | False | Adjust predictions based on predicted state/month/species at inference |

**Why use auxiliary heads?**
- Forces the model to learn meaningful features about the environment
- Acts as regularization to prevent overfitting
- State and month are strong predictors of biomass distribution

### Loss Function

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--target-weights` | float×5 | `0.2 0.2 0.2 0.2 0.2` | Loss weights for [Green, Dead, Clover, GDM, Total]. Competition uses [0.1, 0.1, 0.1, 0.2, 0.5] for scoring |
| `--constraint-weight` | float | `0.05` | Weight for consistency constraints (e.g., Total ≈ Green + Dead + Clover) |
| `--use-focal-loss` | flag | False | Use Focal MSE loss to focus on hard examples |
| `--use-dead-aware-loss` | flag | False | Special loss for Dead prediction with log-space and auxiliary loss |
| `--no-huber-for-dead` | flag | False | Disable Huber loss for Dead target (Huber is more robust to outliers) |

**Loss Components:**
- **Base MSE/Huber**: Per-target regression loss
- **Constraint Loss**: Penalizes inconsistent predictions (e.g., Total ≠ components sum)
- **Auxiliary Loss**: Cross-entropy for classification heads

### Post-Processing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--correction-threshold` | float | `0.15` | Threshold for Dead correction post-processing |
| `--always-correct-dead` | flag | False | Always apply Dead = Total - GDM correction during validation |

### Augmentation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--img-size` | int | `518` | Input image size. DINOv2 native resolution is 518 |
| `--aug-prob` | float | `0.5` | Base probability for augmentation transforms |
| `--stereo-correct-aug` | flag | False | **CRITICAL**: Apply photometric transforms independently per stereo view. Prevents model from "cheating" by matching noise patterns |
| `--stereo-swap-prob` | float | `0.0` | Probability of swapping Left/Right images. Recommended: `0.5`. Doubles effective training data |

**Stereo-Correct Augmentation:**
- **Geometric transforms** (flip, rotate, crop): Applied identically to both views via replay
- **Photometric transforms** (brightness, noise, blur): Applied independently per view
- **Why it matters**: Without this, identical noise on L/R views lets model "cheat" by matching pixels instead of learning 3D geometry

### Target Normalization

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-log-target` | flag | False | **CRITICAL**: Apply log1p to targets, expm1 on predictions. Essential for biomass data which has long-tail distribution |

**Why log1p?**
- Biomass values range from 0 to 1000+ with most samples < 100
- Standard MSE over-penalizes large values, under-learns small values
- Log transform compresses the range, allowing model to learn entire distribution

### MixUp/CutMix

Constrained mixing augmentation - only mixes samples with same species, month, AND state.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mixup-prob` | float | `0.0` | MixUp probability. Blends two images pixel-wise |
| `--mixup-alpha` | float | `0.4` | Beta distribution α for MixUp λ sampling. Lower = more extreme mixing |
| `--cutmix-prob` | float | `0.0` | CutMix probability. Pastes rectangular region from another image |
| `--cutmix-alpha` | float | `1.0` | Beta distribution α for CutMix λ sampling |

**Why constrained mixing?**
- Mixing different species/environments creates unrealistic samples
- Constraining to same context ensures realistic augmentations
- Both L/R views receive same mixing operation for consistency

### AMP (Automatic Mixed Precision)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--amp-dtype` | str | `float16` | AMP dtype. Options: `float16`, `bfloat16`. bfloat16 is more stable but requires Ampere+ GPU |

### Cross-Validation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-folds` | int | `5` | Number of CV folds |
| `--train-folds` | int+ | `0 1 2 3 4` | Which folds to train. Use `--train-folds 0` for quick testing |
| `--cv-strategy` | str | `group_month` | CV split strategy. Options: `group_month`, `group_date`, `group_date_state`, `group_date_state_bin`, `stratified`, `random` |
| `--fold-csv` | str | None | Path to CSV with pre-defined folds. Must have `sample_id_prefix` and `fold` columns. Overrides `--cv-strategy` |

**Using Pre-defined Folds:**
```bash
# Use folds from external CSV (e.g., trainfold.csv)
python -m src.train_5head --fold-csv data/trainfold.csv --train-folds 0 1 2 3 4
```

**CV Strategies:**
- `group_month`: Group by month, stratify by target bins (recommended)
- `group_date`: Group by exact date, stratify by target bins
- `group_date_state`: Group by month, stratify by State
- `group_date_state_bin`: Group by month, stratify by State × target bin
- `stratified`: Stratify by target bins only (may leak temporal info)
- `random`: Random split (not recommended)

### Device & Misc

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device-type` | str | Auto | Device type. Options: `cuda`, `mps`, `cpu`. Auto-detects if not specified |
| `--seed` | int | `18` | Random seed for reproducibility |

---

## Evaluation Script (`eval_5head_oof.py`)

Out-of-Fold evaluation: each sample is predicted by the model that was NOT trained on it.

```bash
python -m src.eval_5head_oof [OPTIONS]
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--model-dir` | str | - | **Yes** | Directory containing fold checkpoints (`5head_best_fold{0-4}.pth`) and `folds.csv` |
| `--data-dir` | str | `./data` | No | Data directory with `train/` images |
| `--backbone` | str | `vit_base_patch14_reg4_dinov2.lvd142m` | No | Must match the backbone used for training |
| `--batch-size` | int | `4` | No | Inference batch size. Can be larger than training since no gradients |
| `--num-workers` | int | `4` | No | DataLoader workers |
| `--num-folds` | int | `5` | No | Number of folds to evaluate |
| `--derive-dead` | flag | False | No | Apply Dead = max(0, Total - GDM) fix. Often improves Dead R² significantly |
| `--grid` | int int | `2 2` | No | Must match training grid |
| `--dropout` | float | `0.2` | No | Must match training dropout |
| `--hidden-ratio` | float | `0.5` | No | Must match training hidden ratio |
| `--device` | str | Auto | No | Device: `cuda`, `mps`, `cpu` |
| `--tta` | flag | False | No | Enable TTA (Test Time Augmentation) with 3 views: original, hflip, vflip |
| `--img-size` | int | `518` | No | Image size for inference |

**Outputs:**
- `oof_predictions.csv`: Per-sample predictions with metadata
- `oof_metrics.json`: R², RMSE, MAE per target and weighted R²

---

## Example Commands

### Basic Training (Quick Test)

```bash
# Single fold, minimal epochs
python -m src.train_5head \
    --epochs 10 \
    --train-folds 0 \
    --device-type mps
```

### Recommended Training Setup

```bash
# Full pipeline with all improvements
python -m src.train_5head \
    --epochs 50 \
    --freeze-backbone \
    --head-lr-stage1 3e-4 \
    --grad-clip 0.5 \
    --warmup-epochs 5 \
    --use-aux-heads \
    --aux-month-weight 0.5 \
    --aux-state-weight 0.5 \
    --aux-species-weight 0.25 \
    --use-log-target \
    --stereo-correct-aug \
    --stereo-swap-prob 0.5 \
    --cv-strategy group_date \
    --device-type mps
```

### Two-Stage Training (Best for Full Finetuning)

```bash
python -m src.train_5head \
    --epochs 30 \
    --two-stage \
    --freeze-epochs 5 \
    --head-lr-stage1 1e-3 \
    --lr 2e-4 \
    --backbone-lr 1e-5 \
    --use-log-target \
    --stereo-correct-aug \
    --stereo-swap-prob 0.5 \
    --device-type cuda
```

### With MixUp/CutMix

```bash
python -m src.train_5head \
    --epochs 50 \
    --freeze-backbone \
    --use-log-target \
    --mixup-prob 0.2 \
    --cutmix-prob 0.2 \
    --device-type cuda
```

### With Pre-defined Folds

```bash
# Use external fold assignments (e.g., from trainfold.csv)
python -m src.train_5head \
    --fold-csv data/trainfold.csv \
    --epochs 50 \
    --freeze-backbone \
    --use-log-target \
    --device-type cuda

# Train only specific folds
python -m src.train_5head \
    --fold-csv data/trainfold.csv \
    --train-folds 0 2 4 \
    --epochs 30
```

### OOF Evaluation

```bash
# Basic evaluation
python -m src.eval_5head_oof \
    --model-dir ./outputs/5head_20251214_123456

# With TTA and Dead derivation (recommended)
python -m src.eval_5head_oof \
    --model-dir ./outputs/5head_20251214_123456 \
    --tta \
    --derive-dead \
    --device mps
```

---

## Priority Guide: What to Enable First

1. **`--use-log-target`** - Critical for long-tail biomass distribution
2. **`--stereo-correct-aug`** - Prevents model from cheating on stereo pairs
3. **`--stereo-swap-prob 0.5`** - Free 2x data augmentation
4. **`--use-aux-heads`** - Multi-task learning regularization
5. **`--freeze-backbone` or `--two-stage`** - Faster convergence, less overfitting
6. **`--derive-dead`** (eval) - Physics-based Dead prediction improvement
7. **`--tta`** (eval) - 1-2% R² improvement at 3x inference cost
8. **`--mixup-prob`/`--cutmix-prob`** - Additional regularization (experiment)
