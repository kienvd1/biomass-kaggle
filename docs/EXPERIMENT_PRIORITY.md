# 🧪 Experiment Priority Guide

## 🖥️ Hardware & Setup

| Resource | Specification |
|----------|--------------|
| GPUs | 2× NVIDIA H200 (80GB each) |
| Batch Size | 16 (auto-adjusted: 8/GPU × 2 = 16 effective) |
| DDP Behavior | Auto-halves batch per GPU to match single-GPU dynamics |
| CV Folds | `data/trainfold_group_location.csv` (5 folds, State×Season×Species stratified) |

### Quick Commands
```bash
# Multi-GPU training (recommended) - matches single-GPU performance
torchrun --nproc_per_node=2 -m src.dinov3_train [OPTIONS]

# Single GPU
python -m src.dinov3_train [OPTIONS]

# Train specific folds only
torchrun --nproc_per_node=2 -m src.dinov3_train --folds 0 1 2
```

### 🎯 All Defaults (enabled without flags)

| Category | Feature | Value |
|----------|---------|-------|
| **Training** | Two-stage | ON (Stage 1 → Stage 2 on patience) |
| **Training** | Stage 1 patience | 10 epochs (triggers Stage 2) |
| **Training** | Stage 1 LR | 2e-4 (heads only) |
| **Training** | Stage 2 LR | 1e-5 (backbone) |
| **Loss** | SmoothL1 | ON (G=0.1, D=derived, C=0.1, GDM=0.2, T=0.5) |
| **Heads** | Train Dead | OFF (derived from Total - GDM) |
| **Heads** | Train Clover | ON (direct head) |
| **Augmentation** | Strong aug | ON (Rotate90, ColorJitter, CLAHE, MotionBlur) |
| **Depth** | Depth features | ON (Depth Anything V2, r=0.63 with Green) |
| **Depth** | Depth attention | ON (weights tiles by vegetation height) |
| **CV** | Fold CSV | `data/trainfold_group_location.csv` |
| **CV** | Strategy | `group_location` (State×Season×Species) |
| **DDP** | Batch adjustment | ON (auto-halve per GPU) |

---

## 🔧 DDP Configuration

### How DDP Matching Works

| Mode | Batch/GPU | Effective Batch | LR | Updates/Epoch | Expected R² |
|------|-----------|-----------------|-----|---------------|-------------|
| **MPS** | 16 | 16 | 2e-4 | ~17 | ~0.64 |
| **Single CUDA** | 16 | 16 | 2e-4 | ~17 | ~0.64 |
| **DDP (2 GPU)** | 8 (auto) | 16 | 2e-4 | ~17 | ~0.64 |

The code automatically halves `--batch-size` per GPU so effective batch matches single-GPU.

### DDP Fixes Applied
- ✅ Auto batch-size adjustment (16 → 8/GPU)
- ✅ Validation on full dataset (no DistributedSampler for val)
- ✅ Only rank 0 saves/prints
- ✅ Barriers before file operations
- ✅ Race condition handling for checkpoint files

---

## 📊 Expected Results Summary

| Phase | CV R² | Time (2×H200) | Key Features |
|-------|-------|---------------|--------------|
| 1. Baseline | 0.60-0.65 | 30 min | **Defaults**: Two-stage + Depth + SmoothL1 |
| 2. PlantHydra | 0.65-0.70 | 1 hour | + Log + Cosine + Compositional |
| 3. + Aux Heads | 0.68-0.73 | 1.5 hours | + Height/NDVI distillation |
| 4. + Advanced | 0.72-0.78 | 2 hours | + Cross-view |
| 5. Ensemble | 0.78-0.85 | 8+ hours | 3-5 model average |

---

## 🚀 Phase 1: Quick Start (30 min)

### 1.1 Baseline (All Defaults)
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Purpose**: Full two-stage training with all defaults.

**Training Flow**:
1. **Stage 1** (frozen backbone): Train heads at lr=2e-4 until patience=10
2. **Stage 2** (finetuning): Unfreeze backbone, continue at backbone-lr=1e-5

**Defaults (no flags needed)**:
| Feature | Default | Disable with |
|---------|---------|--------------|
| Two-stage training | ✅ ON | `--no-two-stage` |
| Stage 1 patience | 10 | `--stage1-patience N` |
| Strong augmentation | ✅ ON | `--no-strong-aug` |
| SmoothL1 loss | ✅ ON | `--no-smoothl1` |
| Train Dead head | ❌ OFF | `--train-dead` |
| Train Clover head | ✅ ON | `--no-train-clover` |
| Depth features | ✅ ON | `--no-use-depth` |
| Depth attention | ✅ ON | `--no-depth-attention` |
| Group location folds | ✅ ON | `--fold-csv other.csv` |

**Default Loss**: SmoothL1 (Green=0.1, Clover=0.1, GDM=0.2, Total=0.5, Dead=derived)

### 1.2 Head-Only Training (Quick Test)
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --no-two-stage \
    --freeze-backbone \
    --epochs 30
```
**Purpose**: Quick test with frozen backbone only (no Stage 2).

### 1.3 Single Fold Quick Test
```bash
# Test single fold first
python -m src.dinov3_train \
    --folds 0 \
    --lr 0.0005 \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Purpose**: Verify setup on single GPU before DDP.

---

## ⭐ Phase 2: Core Features (HIGH PRIORITY)

These features have the highest impact-to-effort ratio.

### 2.1 PlantHydra Loss ⭐⭐⭐ START HERE
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Impact**: +5-8% R²  
**Why**: Log-transform (official metric) + cosine similarity + compositional consistency

### 2.2 Auxiliary Supervision (Height + NDVI) ⭐⭐
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --use-height-head --height-weight 0.2 \
    --use-ndvi-head --ndvi-weight 0.2 \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Impact**: +2-4% R²  
**Why**: Distills training-only signals (Height_Ave_cm, Pre_GSHH_NDVI)

### 2.3 Cross-View Consistency ⭐
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --use-height-head --use-ndvi-head \
    --use-cross-view-consistency --cross-view-weight 0.1 \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Impact**: +1-2% R²  
**Why**: Enforces L/R stereo agreement, stabilizes dead/clover

---

## 🔬 Phase 3: Advanced Features

### 3.1 Depth Attention (Already Default)
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --use-height-head --use-ndvi-head \
    --depth-model-size base \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Impact**: +1-3% R²  
**Note**: Depth is ON by default; use `base` for better accuracy

### 3.2 Species Prior Blending
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --use-species-head --use-species-prior \
    --aux-species-weight 0.1 \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Impact**: +0.5-1.5% R²  
**Why**: PlantHydra-style species→biomass lookup

### 3.3 DINOv3-Large
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --backbone-size large \
    --use-planthydra-loss \
    --use-height-head --use-ndvi-head \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Impact**: +1-2% R²  
**Note**: Needs gradient checkpointing if OOM: `--gradient-checkpointing`

---

## 🧠 Phase 4: Innovative Ideas (from Dataset Paper)

These exploit specific characteristics of the dataset creation process.

### 4.1 Physical Mass Balance
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --use-mass-balance --mass-balance-weight 0.2 \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Why**: Enforces `Green + Dead + Clover ≈ Total` (physical constraint)

### 4.2 AOS Sensor Hallucination
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --use-ndvi-head \
    --use-aos-hallucination --aos-weight 0.3 \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Why**: Predicts lighting-invariant AOS NDVI from RGB

### 4.3 State Density Scaling (Climate Correction)
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --use-state-scaling \
    --freeze-epochs 10 \
    --stage2-epochs 40
```
**Why**: Tasmania grass ≠ WA grass (different water content)

---

## 🛡️ Phase 5: Robustness

### 5.1 Strong Augmentation (Already Default)
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --use-height-head --use-ndvi-head \
    --use-perspective-jitter \
    --use-border-mask \
    --freeze-epochs 10 \
    --stage2-epochs 40
```

### 5.2 Uncertainty-Aware + QC Plausibility
```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 \
    --use-planthydra-loss \
    --use-uncertainty \
    --use-qc-plausibility --qc-weight 0.1 \
    --freeze-epochs 10 \
    --stage2-epochs 40
```

---

## 🏆 Phase 6: Best Combination

Based on results from previous phases, combine the winning features:

```bash
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --folds 0 1 2 3 4 \
    --use-planthydra-loss \
    --use-height-head --height-weight 0.2 \
    --use-ndvi-head --ndvi-weight 0.2 \
    --use-cross-view-consistency --cross-view-weight 0.1 \
    --use-mass-balance --mass-balance-weight 0.1 \
    --use-perspective-jitter \
    --freeze-epochs 15 \
    --stage2-epochs 45 \
    --lr-mult 0.8
```

---

## 🎯 Phase 7: Ensemble

Train multiple diverse models:

| Model | Seed | Backbone | Key Feature |
|-------|------|----------|-------------|
| M1 | 18 | base | PlantHydra + Aux |
| M2 | 42 | base | + Cross-view + Depth |
| M3 | 123 | large | PlantHydra + Aux |
| M4 | 18 | base | + Mass Balance + AOS |
| M5 | 42 | base | + Strong Aug + QC |

```bash
# Example: Train M1
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --seed 18 --output-dir ./outputs/M1 \
    --use-planthydra-loss --use-height-head --use-ndvi-head \
    --freeze-epochs 10 --stage2-epochs 40

# Example: Train M2
torchrun --nproc_per_node=2 -m src.dinov3_train \
    --seed 42 --output-dir ./outputs/M2 \
    --use-planthydra-loss --use-height-head --use-ndvi-head \
    --use-cross-view-consistency --depth-model-size base \
    --freeze-epochs 10 --stage2-epochs 40
```

Final prediction: Average of M1-M5 predictions.

---

## 📋 Quick Reference: All Flags

### Loss Functions
| Flag | Description | Default |
|------|-------------|---------|
| `--smoothl1` / `--no-smoothl1` | SmoothL1 on 3 targets | **ON** |
| `--use-huber` | Huber loss for Dead (outlier robust) | off |
| `--huber-delta` | Huber delta | 5.0 |
| `--use-planthydra-loss` | Log + Cosine + Compositional | off |
| `--use-log-transform` | Official log(1+y) metric | off |
| `--use-cosine-sim` | Cosine similarity loss | off |
| `--cosine-weight` | Weight for cosine loss | 0.4 |
| `--use-compositional` | GDM=G+C, Total=G+D+C | off |
| `--compositional-weight` | Weight for compositional | 0.1 |
| `--use-mass-balance` | Physical mass balance | off |
| `--mass-balance-weight` | Weight for mass balance | 0.2 |

### Output Heads
| Flag | Description | Default |
|------|-------------|---------|
| `--train-dead` / `--no-train-dead` | Train Dead directly | **OFF** |
| `--train-clover` / `--no-train-clover` | Train Clover directly | **ON** |

### Auxiliary Heads
| Flag | Description | Default |
|------|-------------|---------|
| `--use-height-head` | Predict Height_Ave_cm | off |
| `--height-weight` | Weight for height loss | 0.3 |
| `--use-ndvi-head` | Predict Pre_GSHH_NDVI | off |
| `--ndvi-weight` | Weight for NDVI loss | 0.3 |
| `--use-species-head` | Species classification | off |
| `--use-species-prior` | Species→biomass blending | off |
| `--use-aos-hallucination` | AOS sensor prediction | off |
| `--aos-weight` | Weight for AOS loss | 0.3 |

### Advanced Features
| Flag | Description | Default |
|------|-------------|---------|
| `--use-cross-view-consistency` | L/R stereo agreement | off |
| `--cross-view-weight` | Weight for cross-view | 0.1 |
| `--use-depth` / `--no-use-depth` | Depth Anything V2 | **ON** |
| `--depth-attention` / `--no-depth-attention` | Depth-guided attention | **ON** |
| `--depth-model-size` | small or base | small |
| `--use-state-scaling` | Climate/location bias | off |
| `--use-uncertainty` | Gaussian NLL | off |
| `--use-qc-plausibility` | Domain constraints | off |

### Augmentation
| Flag | Description | Default |
|------|-------------|---------|
| `--strong-aug` / `--no-strong-aug` | Heavy augmentation | **ON** |
| `--use-perspective-jitter` | Annotation noise sim | off |
| `--use-border-mask` | Frame artifact prevention | off |
| `--use-pad-to-square` | Preserve aspect ratio | off |

### Training
| Flag | Description | Default |
|------|-------------|---------|
| `--two-stage` / `--no-two-stage` | Freeze then finetune | **ON** |
| `--stage1-patience` | Patience for Stage 1 | **10** |
| `--freeze-epochs` | Max Stage 1 epochs | 30 |
| `--stage2-epochs` | Stage 2 epochs | 20 |
| `--epochs` | Total epochs (alternative) | 50 |
| `--lr` | Stage 1 LR (heads) | **2e-4** |
| `--backbone-lr` | Stage 2 LR (backbone) | **1e-5** |
| `--batch-size` | Per-GPU batch size | 16 |
| `--grad-accum` | Gradient accumulation | 1 |
| `--backbone-size` | small/base/large | base |
| `--lr-mult` | Layer-wise LR decay | 1.0 |
| `--gradient-checkpointing` | Memory savings | off |

### Cross-Validation
| Flag | Description | Default |
|------|-------------|---------|
| `--use-predefined-folds` / `--no-predefined-folds` | Use fold CSV | **ON** |
| `--fold-csv` | Path to fold CSV | `data/trainfold_group_location.csv` |
| `--n-folds` | Number of folds | 5 |
| `--folds` | Specific folds to train | all |
| `--cv-strategy` | Fold creation strategy | `group_location` |

### DDP (Automatic)
| Behavior | Description |
|----------|-------------|
| Batch adjustment | `batch_size // world_size` per GPU |
| Validation | Full dataset on all GPUs, report from rank 0 |
| Checkpoints | Only rank 0 saves |
| Print statements | Only rank 0 prints |

---

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Option 1: Reduce batch size
--batch-size 8

# Option 2: Gradient checkpointing
--gradient-checkpointing

# Option 3: Smaller depth model
--depth-model-size small

# Option 4: Smaller backbone
--backbone-size small
```

### Poor Dead/Clover Predictions
```bash
# Option 1: Train Dead directly
--train-dead

# Option 2: Presence heads
--use-presence-heads

# Option 3: Cross-view consistency
--use-cross-view-consistency
```

### Training Instability
```bash
# Option 1: Lower learning rate
--lr 1e-4 --backbone-lr 5e-6

# Option 2: More warmup
--warmup-epochs 5

# Option 3: Gradient clipping
--grad-clip 0.5
```

### DDP Performance Mismatch
```bash
# Verify batch is auto-adjusted (should print):
# "DDP batch adjustment: 16 → 8/GPU (effective: 16)"

# If still issues, manually set:
--batch-size 8  # Will become 4/GPU with 2 GPUs
```

---

## 📈 Monitoring Checklist

During training, watch:

- [ ] `weighted_r2` → Official metric (should increase)
- [ ] `r2_dead`, `r2_clover` → Hardest targets (>0.3 is good)
- [ ] `r2_total` → Easiest, highest weight (>0.7 expected)
- [ ] Train/Val loss gap → Overfitting if >2× difference
- [ ] Loss components → If using PlantHydra, check breakdown

Early stopping signals:
- `r2_dead` < 0.2 after 20 epochs → Try `--train-dead`
- `r2_clover` stuck → Try `--use-cross-view-consistency`
- Val loss increasing → Reduce LR or add regularization
