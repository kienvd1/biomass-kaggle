# DINOv3 Direct Model - Ablation Study

Systematic testing of options from basic to advanced to identify what works best.

**Defaults**: epochs=50, batch-size=16, lr=5e-4, patience=10

## Quick Start

```bash
# Run baseline first
python -m src.dinov3_train --freeze-backbone --device-type mps
```

---

## 0. BASELINE

Minimal configuration - this is the reference point.

```bash
python -m src.dinov3_train \
    --freeze-backbone \
    --device-type mps
```

**Expected**: ~0.70-0.75 CV aR²

---

## 1. OPTIONAL HEADS

Test whether training Dead/Clover directly improves over deriving them.

### 1a. Train Clover Head

```bash
python -m src.dinov3_train \
    --train-clover \
    --freeze-backbone \
    --device-type mps
```

### 1b. Train Dead Head

```bash
python -m src.dinov3_train \
    --train-dead \
    --freeze-backbone \
    --device-type mps
```

### 1c. Train Both Dead + Clover

```bash
python -m src.dinov3_train \
    --train-dead --train-clover \
    --freeze-backbone \
    --device-type mps
```

**Hypothesis**: Training Clover directly should help since it's hardest to predict (small values, hard to see).

---

## 2. PHOTOMETRIC MODES

Test how photometric augmentation affects stereo learning.

| Mode | Description |
|------|-------------|
| `same` | Same photometric to both L/R (default) |
| `independent` | Different photometric to L/R |
| `left` | Only apply to left |
| `right` | Only apply to right |
| `none` | No photometric (geometric only) |

### 2a. Independent Photometric

```bash
python -m src.dinov3_train \
    --photometric independent \
    --freeze-backbone \
    --device-type mps
```

### 2b. Left Only Photometric

```bash
python -m src.dinov3_train \
    --photometric left \
    --freeze-backbone \
    --device-type mps
```

### 2c. Right Only Photometric

```bash
python -m src.dinov3_train \
    --photometric right \
    --freeze-backbone \
    --device-type mps
```

### 2d. No Photometric (Geometric Only)

```bash
python -m src.dinov3_train \
    --photometric none \
    --freeze-backbone \
    --device-type mps
```

**Hypothesis**: `independent` should help prevent model from matching pixel noise between views.

---

## 3. STEREO SWAP

Test if swapping L/R views helps regularization.

### 3a. Light Stereo Swap (10%)

```bash
python -m src.dinov3_train \
    --stereo-swap-prob 0.1 \
    --freeze-backbone \
    --device-type mps
```

### 3b. Medium Stereo Swap (20%)

```bash
python -m src.dinov3_train \
    --stereo-swap-prob 0.2 \
    --freeze-backbone \
    --device-type mps
```

### 3c. High Stereo Swap (30%)

```bash
python -m src.dinov3_train \
    --stereo-swap-prob 0.3 \
    --freeze-backbone \
    --device-type mps
```

**Hypothesis**: Works because FiLM/attention pooling don't assume L/R order.

---

## 4. MIXUP

Test MixUp augmentation (mixes samples with same species/month/state).

### 4a. Light MixUp

```bash
python -m src.dinov3_train \
    --mixup-prob 0.2 --mixup-alpha 0.4 \
    --freeze-backbone \
    --device-type mps
```

### 4b. Medium MixUp

```bash
python -m src.dinov3_train \
    --mixup-prob 0.3 --mixup-alpha 0.5 \
    --freeze-backbone \
    --device-type mps
```

### 4c. Strong MixUp

```bash
python -m src.dinov3_train \
    --mixup-prob 0.4 --mixup-alpha 0.6 \
    --freeze-backbone \
    --device-type mps
```

**Note**: MixUp only mixes samples with same species/month/state to preserve context.

---

## 5. CUTMIX

Test CutMix augmentation (pastes rectangular region from partner).

### 5a. Light CutMix

```bash
python -m src.dinov3_train \
    --cutmix-prob 0.2 --cutmix-alpha 1.0 \
    --freeze-backbone \
    --device-type mps
```

### 5b. Medium CutMix

```bash
python -m src.dinov3_train \
    --cutmix-prob 0.3 --cutmix-alpha 1.0 \
    --freeze-backbone \
    --device-type mps
```

**Note**: Same rectangular region is applied to both L/R to preserve stereo geometry.

---

## 6. ARCHITECTURE

Test architectural changes.

### 6a. Grid 3x3 (More Tiles)

```bash
python -m src.dinov3_train \
    --grid 3 \
    --freeze-backbone \
    --device-type mps
```

### 6b. No FiLM

```bash
python -m src.dinov3_train \
    --no-film \
    --freeze-backbone \
    --device-type mps
```

### 6c. No Attention Pooling

```bash
python -m src.dinov3_train \
    --no-attention-pool \
    --freeze-backbone \
    --device-type mps
```

### 6d. Higher Dropout (0.4)

```bash
python -m src.dinov3_train \
    --dropout 0.4 \
    --freeze-backbone \
    --device-type mps
```

### 6e. Larger Hidden Ratio (0.35)

```bash
python -m src.dinov3_train \
    --hidden-ratio 0.35 \
    --freeze-backbone \
    --device-type mps
```

---

## 7. TRAINING DYNAMICS

Test learning rate and regularization.

### 7a. Lower LR (1e-4)

```bash
python -m src.dinov3_train \
    --lr 1e-4 \
    --freeze-backbone \
    --device-type mps
```

### 7b. Higher LR (1e-3)

```bash
python -m src.dinov3_train \
    --lr 1e-3 \
    --freeze-backbone \
    --device-type mps
```

### 7c. More Weight Decay (0.05)

```bash
python -m src.dinov3_train \
    --weight-decay 0.05 \
    --freeze-backbone \
    --device-type mps
```

### 7d. Lower Augmentation (0.3)

```bash
python -m src.dinov3_train \
    --aug-prob 0.3 \
    --freeze-backbone \
    --device-type mps
```

### 7e. Higher Augmentation (0.7)

```bash
python -m src.dinov3_train \
    --aug-prob 0.7 \
    --freeze-backbone \
    --device-type mps
```

---

## 8. LOSS FUNCTIONS

Test different loss configurations.

| Loss | Targets | Weights |
|------|---------|---------|
| MSE (default) | All 5 | G=0.1, D=0.1, C=0.1, GDM=0.2, T=0.5 |
| SmoothL1 | 3 only | G=0.125, GDM=0.25, T=0.625 |

### 8a. SmoothL1 Loss (3 targets)

```bash
python -m src.dinov3_train \
    --smoothl1 \
    --freeze-backbone \
    --device-type mps
```

### 8b. No Huber for Dead

```bash
python -m src.dinov3_train \
    --no-huber \
    --freeze-backbone \
    --device-type mps
```

### 8c. Higher Huber Delta (10.0)

```bash
python -m src.dinov3_train \
    --huber-delta 10.0 \
    --freeze-backbone \
    --device-type mps
```

**Hypothesis**: SmoothL1 on 3 main targets may be more robust to outliers. Huber helps with Dead which has extreme values.

---

## 9. INNOVATIVE FEATURES 🔥

Test domain-knowledge and 3D features for potential big gains.

### 9a. Vegetation Indices (ExG, ExR, GRVI)

```bash
python -m src.dinov3_train \
    --use-vegetation-indices \
    --freeze-backbone \
    --device-type mps
```

**What it adds**: ExG (green biomass), ExR (dead biomass), GRVI, VARI indices - 48 extra features from RGB statistics.

### 9b. Stereo Disparity Features

```bash
python -m src.dinov3_train \
    --use-disparity \
    --freeze-backbone \
    --device-type mps
```

**What it adds**: 3D volume features from stereo correspondence. Taller vegetation → larger disparity → more biomass.

### 9c. Both VI + Disparity

```bash
python -m src.dinov3_train \
    --use-vegetation-indices \
    --use-disparity \
    --freeze-backbone \
    --device-type mps
```

### 9d. VI + Disparity + Train Clover

```bash
python -m src.dinov3_train \
    --use-vegetation-indices \
    --use-disparity \
    --train-clover \
    --freeze-backbone \
    --device-type mps
```

**Hypothesis**: These domain-specific features provide explicit signals that complement learned DINOv3 features. Expected: +2-4% aR².

---

## 10. STRONG AUGMENTATION

Strong augmentations from dinov3-5tar.ipynb notebook that achieved good results:
- `RandomRotate90` - 90-degree rotations
- `ColorJitter` - comprehensive color augmentation
- `HueSaturationValue` - HSV augmentation (stronger)
- `CLAHE` - adaptive histogram equalization  
- `MotionBlur` - motion blur

### 10a. Strong Augmentation (Default Mode)

```bash
python -m src.dinov3_train \
    --strong-aug \
    --freeze-backbone \
    --device-type mps
```

### 10b. Strong Augmentation + Independent Photometric

```bash
python -m src.dinov3_train \
    --strong-aug \
    --photometric independent \
    --freeze-backbone \
    --device-type mps
```

### 10c. Strong Augmentation + VI + Disparity

```bash
python -m src.dinov3_train \
    --strong-aug \
    --use-vegetation-indices \
    --use-disparity \
    --freeze-backbone \
    --device-type mps
```

**Hypothesis**: These augmentations are battle-tested from the notebook that achieved 0.6497 CV score.

---

## 11. LEARNABLE AUGMENTATION

Learnable augmentation learns optimal augmentation parameters during training instead of using fixed random augmentations. Incorporates all strong augmentations from `dinov3-5tar.ipynb` as differentiable, learnable modules.

### Augmentation Components

| Component | Parameters | What It Learns | Equivalent To |
|-----------|------------|----------------|---------------|
| **Color** | 6 params | brightness, contrast, saturation, hue, val_shift, sat_shift | ColorJitter + HueSaturationValue |
| **Spatial** | 5 params | scale_x, scale_y, rotation, tx, ty | Affine transforms |
| **Blur/Noise** | 2 params + 3×3 kernel | blur_strength, noise_strength, kernel shape | GaussianBlur/MotionBlur/GaussNoise |
| **Local Contrast** | 2 params | clip_limit, strength | CLAHE |

### Key Features

- **Input-dependent**: Small predictor network adapts augmentation to each image
- **Diversity loss**: Prevents collapse to identity transform
- **End-to-end differentiable**: Learns what augmentations help biomass prediction
- **Training only**: Disabled during inference (identity transform)

### 11a. Color Augmentation Only

Learns ColorJitter + HSV style transforms (brightness, contrast, saturation, hue).

```bash
python -m src.dinov3_train \
    --use-learnable-aug \
    --learnable-aug-color \
    --no-learnable-aug-spatial \
    --freeze-backbone \
    --device-type mps
```

### 11b. Spatial Augmentation Only

Learns affine transforms (scale, rotation, translation).

```bash
python -m src.dinov3_train \
    --use-learnable-aug \
    --no-learnable-aug-color \
    --learnable-aug-spatial \
    --freeze-backbone \
    --device-type mps
```

### 11c. Color + Spatial (Full)

All learnable augmentations: color, spatial, blur/noise, local contrast.

```bash
python -m src.dinov3_train \
    --use-learnable-aug \
    --learnable-aug-color \
    --learnable-aug-spatial \
    --freeze-backbone \
    --device-type mps
```

### 11d. Learnable Aug + VI + Disparity

Combine learnable augmentation with domain features.

```bash
python -m src.dinov3_train \
    --use-learnable-aug \
    --learnable-aug-color \
    --use-vegetation-indices \
    --use-disparity \
    --freeze-backbone \
    --device-type mps
```

### 11e. Learnable Aug + Strong Aug Combo

Use learnable on top of standard strong augmentations.

```bash
python -m src.dinov3_train \
    --use-learnable-aug \
    --learnable-aug-color \
    --strong-aug \
    --freeze-backbone \
    --device-type mps
```

### Technical Details

**Color Transform** (`_apply_color_transform`):
- Denormalizes image → applies brightness/contrast/saturation/hue → renormalizes
- Uses `color_strength=0.25` scaling factor
- Separate L/R augmenters allow stereo-aware color variation

**Spatial Transform** (`_apply_spatial_transform`):
- Uses `F.affine_grid` + `F.grid_sample` for differentiable warping
- Uses `spatial_strength=0.15` (~10° rotation, ~15% scale)
- Reflection padding to avoid black borders

**Blur/Noise** (`_apply_blur_noise`):
- Learnable 3×3 kernel (starts as Gaussian, can learn motion blur direction)
- Gaussian noise added after blur
- Max 40% blur strength, 8% noise strength

**Local Contrast** (`_apply_local_contrast`):
- CLAHE-like differentiable approximation
- 16×16 local mean pooling + soft clipping via tanh
- Enhances vegetation texture details

**Diversity Loss**:
- `exp(-params.abs().mean())` for each component
- Penalizes near-zero (identity) augmentations
- Weights: color=0.01, spatial=0.01, blur=0.005, contrast=0.005

**Hypothesis**: The model learns to apply augmentations that are most beneficial for biomass prediction, adapting to each image's characteristics. Color augmentation is likely most impactful (vegetation color variation), while spatial may help with viewpoint invariance.

---

## 12. COMBINED CONFIGURATIONS

Test combinations of best options.

### 12a. Clover + Independent Photometric

```bash
python -m src.dinov3_train \
    --train-clover \
    --photometric independent \
    --freeze-backbone \
    --device-type mps
```

### 12b. Clover + Stereo Swap + MixUp

```bash
python -m src.dinov3_train \
    --train-clover \
    --stereo-swap-prob 0.15 \
    --mixup-prob 0.2 --mixup-alpha 0.4 \
    --freeze-backbone \
    --device-type mps
```

### 12c. All Heads + Grid 3 + Independent

```bash
python -m src.dinov3_train \
    --train-dead --train-clover \
    --grid 3 \
    --photometric independent \
    --freeze-backbone \
    --device-type mps
```

### 12d. Kitchen Sink (All Options)

```bash
python -m src.dinov3_train \
    --train-dead --train-clover \
    --grid 3 \
    --dropout 0.35 \
    --hidden-ratio 0.3 \
    --photometric independent \
    --stereo-swap-prob 0.15 \
    --mixup-prob 0.2 --mixup-alpha 0.4 \
    --freeze-backbone \
    --device-type mps
```

---

## 13. TWO-STAGE TRAINING

After finding best head-only config, test finetuning backbone.

### 13a. Two-Stage with Best Config

```bash
python -m src.dinov3_train \
    --train-clover \
    --photometric independent \
    --stereo-swap-prob 0.15 \
    --two-stage \
    --freeze-epochs 15 \
    --backbone-lr 1e-5 \
    --device-type mps
```

---

## Results Tracking

| Experiment | Config | CV aR² | G | D | C | GDM | T | Notes |
|------------|--------|--------|---|---|---|-----|---|-------|
| 0. Baseline | default | | | | | | | |
| 1a. +Clover | train-clover | | | | | | | |
| 1b. +Dead | train-dead | | | | | | | |
| 1c. +Both | train-dead,clover | | | | | | | |
| 2a. Photo indep | photometric=independent | | | | | | | |
| 2b. Photo left | photometric=left | | | | | | | |
| 2c. Photo right | photometric=right | | | | | | | |
| 2d. Photo none | photometric=none | | | | | | | |
| 3a. Swap 10% | stereo-swap=0.1 | | | | | | | |
| 3b. Swap 20% | stereo-swap=0.2 | | | | | | | |
| 3c. Swap 30% | stereo-swap=0.3 | | | | | | | |
| 4a. MixUp light | mixup=0.2 | | | | | | | |
| 4b. MixUp med | mixup=0.3 | | | | | | | |
| 4c. MixUp strong | mixup=0.4 | | | | | | | |
| 5a. CutMix light | cutmix=0.2 | | | | | | | |
| 5b. CutMix med | cutmix=0.3 | | | | | | | |
| 6a. Grid 3 | grid=3 | | | | | | | |
| 6b. No FiLM | no-film | | | | | | | |
| 6c. No AttnPool | no-attention-pool | | | | | | | |
| 6d. Dropout 0.4 | dropout=0.4 | | | | | | | |
| 7a. LR 1e-4 | lr=1e-4 | | | | | | | |
| 7b. LR 1e-3 | lr=1e-3 | | | | | | | |
| 8a. SmoothL1 | smoothl1 | | | | | | | |
| 8b. No Huber | no-huber | | | | | | | |
| 8c. Huber δ=10 | huber-delta=10 | | | | | | | |
| 9a. VI | use-vegetation-indices | | | | | | | 🔥 |
| 9b. Disparity | use-disparity | | | | | | | 🔥 |
| 9c. VI+Disparity | vi+disparity | | | | | | | 🔥 |
| 9d. VI+Disp+Clover | vi+disp+clover | | | | | | | 🔥 |
| 10a. Strong Aug | strong-aug | | | | | | | 🆕 |
| 10b. Strong+Indep | strong+indep | | | | | | | 🆕 |
| 10c. Strong+VI+Disp | strong+vi+disp | | | | | | | 🆕 |
| 11a. Learn Aug Color | learnable-aug-color | | | | | | | 🆕 |
| 11b. Learn Aug Spatial | learnable-aug-spatial | | | | | | | 🆕 |
| 11c. Learn Aug Full | color+spatial+blur+CLAHE | | | | | | | 🆕 |
| 11d. Learn+VI+Disp | learnable+vi+disp | | | | | | | 🆕 |
| 11e. Learn+Strong | learnable+strong-aug | | | | | | | 🆕 |
| 12a. Combo 1 | clover+indep | | | | | | | |
| 12b. Combo 2 | clover+swap+mixup | | | | | | | |
| 12c. Combo 3 | all+grid3+indep | | | | | | | |
| 12d. Kitchen sink | all options | | | | | | | |
| 13a. Two-stage | best+finetune | | | | | | | |

---

## Recommended Order

1. Run **Baseline** first
2. Test **Optional Heads** (1a-1c) - usually quick wins
3. 🔥 Test **Innovative Features** (9a-9d) - HIGH IMPACT: VI and Disparity
4. 🆕 Test **Strong Augmentation** (10a-10c) - battle-tested from notebook
5. 🆕 Test **Learnable Augmentation** (11a-11e) - learns optimal augmentations
6. Test **Photometric Modes** (2a-2d) - affects overfitting
7. Test **Stereo Swap** (3a-3c) - easy regularization
8. Test **MixUp** (4a-4c) - context-aware augmentation
9. Test **Architecture** (6a-6e) - if still underfitting
10. Test **Loss Functions** (8a-8c) - robustness to outliers
11. **Combine** best options (12a-12d)
12. Try **Two-Stage** training (13a) for final push

---

## Tips

- Each run takes ~60-90 min on MPS (5 folds × 50 epochs)
- Watch for **overfitting**: high train R², low val R²
- Watch for **Dead R²**: often the hardest target
- **Clover R²** can go negative if too few clover samples in fold
- Results saved to `./outputs/dinov3_YYYYMMDD_HHMMSS/results.json`
- **aR²** = weighted avg of per-target R² (0.1×G + 0.1×D + 0.1×C + 0.2×GDM + 0.5×T)
- **Total R²** has 50% impact on aR² - focus optimization there
