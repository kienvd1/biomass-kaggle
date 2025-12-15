# 5-Head Model Improvement Proposals

## 🏆 Top 3 High-Impact Improvements

Based on analysis of the codebase and test set constraints (no metadata), these three changes are prioritized for maximum performance gain.

### 1. Stereo Swap Augmentation (Training & Inference)
- **Impact**: 🔴 High
- **Why**: The dataset consists of stereo pairs (left/right). The model must be invariant to which image is "left" or "right" as they capture the same scene. Swapping them during training effectively doubles the dataset size and enforces symmetry.
- **Action**:
  - **Train**: Add `if random() < 0.5: left, right = right, left` in `BiomassDataset`.
  - **Inference**: Add `(right, left)` as a 4th TTA view.

### 2. MixUp Regularization
- **Impact**: 🔴 High
- **Why**: Regression models on small datasets (like this one) are prone to overfitting. MixUp (blending images and targets) smooths the decision boundary and improves generalization, especially for skewed targets like `Clover` and `Dead`.
- **Action**: Implement MixUp in the training loop with Beta distribution ($\alpha=0.4$).

### 3. Grid Ensemble (2x2 + 3x3) & Physics Post-Processing
- **Impact**: 🔴 High
- **Why**: 
  - **Ensemble**: A 3x3 grid captures finer spatial details than 2x2. Averaging them reduces variance.
  - **Post-Processing**: Since metadata is missing at inference, we cannot use it to adjust predictions. Instead, we must rely on physical constraints: `Total ≈ Green + Dead + Clover` and `Total = Dead + GDM`.
- **Action**:
  - Train a separate model with `--grid 3 3`.
  - Ensemble predictions from 2x2 and 3x3 models.
  - Apply `Dead = Total - GDM` correction at inference.

---

## Current Architecture Summary

- **Model**: 5-Head DINOv2 with FiLM conditioning + Attention pooling
- **Backbone**: `vit_base_patch14_reg4_dinov2.lvd142m` (frozen or finetuned)
- **Targets**: Green, Dead, Clover, GDM, Total (5 regression heads)
- **Auxiliary heads**: State (4 classes), Month (10 classes), Species (8 groups)
- **Input**: Stereo image pairs (left/right) with grid-based tile processing

---

## 🚨 Critical Constraint: No Metadata at Inference

The test set does **not** provide State, Month, or Species information. This means:

1. ✅ Auxiliary heads can help during **training** (multi-task regularization)
2. ❌ Cannot rely on ground-truth metadata for inference adjustments
3. ⚠️ Using **predicted** metadata for `apply_context_adjustment` is risky (error propagation)

**Recommendation**: Use auxiliary heads for training only. At inference, use physics-based post-processing.

---

## Proposed Improvements

### 1. Training Strategy (High Impact)

#### A. Keep Auxiliary Heads for Regularization Only

The auxiliary heads help the backbone learn discriminative features that correlate with biomass (e.g., seasonal patterns, regional characteristics). However, don't use them for inference adjustment.

```bash
# Training: Use aux heads
--use-aux-heads --aux-month-weight 5.0 --aux-state-weight 3.0 --aux-species-weight 2.0

# Inference: Do NOT use context adjustment
# Omit --apply-context-adjustment flag
```

#### B. Log-Scale Targets for Skewed Distributions

Dead and Clover have highly skewed distributions (many zeros, few large values).

```python
# Transform targets before loss computation
target_log = torch.log1p(target)  # log(1 + x) is safe for x >= 0
pred_log = torch.log1p(pred)
loss = F.mse_loss(pred_log, target_log)
```

#### C. MixUp Augmentation

MixUp regularization is effective for regression tasks with limited data.

```python
# In train_one_epoch
if use_mixup and random.random() < mixup_prob:
    lam = np.random.beta(0.4, 0.4)
    idx = torch.randperm(x_left.size(0))
    x_left = lam * x_left + (1 - lam) * x_left[idx]
    x_right = lam * x_right + (1 - lam) * x_right[idx]
    targets = lam * targets + (1 - lam) * targets[idx]
```

#### D. Knowledge Distillation from Auxiliary Heads

Train auxiliary heads first, then use their predictions as soft targets to guide biomass heads.

---

### 2. Data Augmentations (Medium Impact)

#### Current Augmentations
- HorizontalFlip, VerticalFlip
- Affine (translate, scale, rotate)
- GaussNoise, GaussianBlur
- RandomBrightnessContrast, HueSaturationValue
- CoarseDropout

#### Proposed Additional Augmentations

| Augmentation | Rationale |
|--------------|-----------|
| **Stereo Swap** | Swap left↔right images (stereo symmetry) |
| **MixUp** | Blend images and targets for regularization |
| **CutMix** | Cut and paste patches between images |
| **RandomResizedCrop** | Force model to handle partial views |
| **ColorJitter (green-focused)** | Vegetation-specific color variations |

```python
# Stereo swap augmentation (add to dataset.py)
if self.is_train and random.random() < 0.5:
    left, right = right, left  # Swap stereo pair
```

---

### 3. Model Architecture (Medium Impact)

#### A. Multi-Scale Features

Currently only using CLS token. Consider concatenating features from different layers.

```python
# Use intermediate layers
features = []
for i, block in enumerate(self.backbone.blocks):
    x = block(x)
    if i in [6, 9, 11]:  # Multi-scale extraction
        features.append(x[:, 0])  # CLS token at each scale
final_feat = torch.cat(features, dim=-1)
```

#### B. Cross-Attention Instead of FiLM

FiLM is additive modulation. Cross-attention allows richer interaction between stereo views.

```python
class CrossAttention(nn.Module):
    def forward(self, x_left, x_right):
        # Left attends to right, right attends to left
        attn_left = self.cross_attn(x_left, x_right, x_right)
        attn_right = self.cross_attn(x_right, x_left, x_left)
        return attn_left, attn_right
```

#### C. Uncertainty Estimation

Add variance heads for weighted loss and prediction confidence.

```python
# Each head outputs (mean, log_var)
mean = self.head_mean(f)
log_var = self.head_var(f)
# Gaussian NLL loss
loss = 0.5 * (torch.exp(-log_var) * (pred - target)**2 + log_var)
```

---

### 4. Inference Strategy (High Impact)

#### A. Test-Time Augmentation (TTA)

```python
def predict_with_tta(model, x_left, x_right):
    preds = []
    # Original
    preds.append(model(x_left, x_right))
    # Horizontal flip
    preds.append(model(x_left.flip(-1), x_right.flip(-1)))
    # Vertical flip
    preds.append(model(x_left.flip(-2), x_right.flip(-2)))
    # Stereo swap
    preds.append(model(x_right, x_left))
    return torch.stack(preds).mean(dim=0)
```

#### B. Ensemble Strategy

```bash
# Ensemble different configurations
- 5-fold CV ensemble (average predictions)
- Grid size ensemble (2x2, 3x3, 2x3)
- Different dropout values
```

#### C. Physics-Based Post-Processing (No Metadata Needed)

```python
class ImprovedPostProcessor:
    def __call__(self, green, dead, clover, gdm, total):
        # Physics constraint: total = gdm + dead = green + clover + dead
        
        # 1. Derive dead from physics when inconsistent
        derived_dead = F.relu(total - gdm)
        
        # 2. Check consistency
        expected_total = green + dead + clover
        error = torch.abs(total - expected_total) / (total + 1e-8)
        
        # 3. Correct if error is high
        needs_correction = error > 0.15
        dead_corrected = torch.where(needs_correction, derived_dead, dead)
        
        # 4. Clamp to training distribution bounds
        dead_corrected = torch.clamp(dead_corrected, min=0, max=50)
        clover = torch.clamp(clover, min=0, max=40)
        
        return green, dead_corrected, clover, gdm, total
```

---

### 5. Cross-Validation Strategy

#### Current Strategy
- `group_month`: StratifiedGroupKFold grouped by month, stratified by target bins

#### Alternative Strategies to Test

| Strategy | Use Case |
|----------|----------|
| `group_date_state` | Group by month, stratify by State |
| Leave-one-state-out | Test generalization to unseen states |
| Leave-one-month-out | Test generalization to unseen seasons |
| Adversarial validation | Find train/test distribution shift |

---

## Recommended Training Commands

### Baseline (Current Best)

```bash
python -m src.train_5head \
    --epochs 50 \
    --freeze-backbone \
    --use-aux-heads \
    --aux-month-weight 5.0 \
    --aux-state-weight 3.0 \
    --aux-species-weight 2.0 \
    --constraint-weight 0 \
    --always-correct-dead \
    --device-type mps
```

### With Grid 3x3

```bash
python -m src.train_5head \
    --epochs 50 \
    --freeze-backbone \
    --grid 3 3 \
    --use-aux-heads \
    --aux-month-weight 5.0 \
    --aux-state-weight 3.0 \
    --aux-species-weight 2.0 \
    --train-folds 0 \
    --device-type mps
```

### Full Finetuning (Stage 2)

```bash
python -m src.train_5head \
    --epochs 30 \
    --two-stage \
    --freeze-epochs 10 \
    --lr 2e-4 \
    --backbone-lr 1e-5 \
    --use-aux-heads \
    --device-type mps
```

---

## Hyperparameter Search Space

| Parameter | Search Range | Notes |
|-----------|--------------|-------|
| `--grid` | `[2,2]`, `[3,3]`, `[2,3]`, `[4,4]` | Start with 2x2 and 3x3 |
| `--head-lr-stage1` | `[5e-4, 1e-3, 2e-3]` | For frozen backbone |
| `--dropout` | `[0.1, 0.2, 0.3]` | Higher if overfitting |
| `--hidden-ratio` | `[0.25, 0.5, 1.0]` | Head capacity |
| `--aux-state-weight` | `[1.0, 3.0, 5.0]` | State classification |
| `--aux-month-weight` | `[1.0, 3.0, 5.0]` | Month classification |
| `--aux-species-weight` | `[1.0, 2.0, 3.0]` | Species classification |
| `--target-weights` | Equal `[0.2]*5`, Total-heavy `[0.1,0.1,0.1,0.2,0.5]` | Loss weighting |
| `--grad-clip` | `[0.5, 1.0, 2.0]` | Gradient clipping |

---

## Priority Implementation Order

1. **High Priority** (Quick wins)
   - [ ] Disable `apply_context_adjustment` at inference
   - [ ] Add `--always-correct-dead` flag
   - [ ] Test grid 3x3 vs 2x2

2. **Medium Priority** (Moderate effort)
   - [ ] Add stereo swap augmentation
   - [ ] Implement MixUp
   - [ ] Add TTA at inference

3. **Lower Priority** (Significant effort)
   - [ ] Multi-scale feature extraction
   - [ ] Cross-attention for stereo
   - [ ] Uncertainty estimation heads

---

## Summary

**Key Insight**: Since metadata isn't available at inference, the auxiliary heads should be used **only for training regularization**. At inference, rely on:

1. **Physics-based post-processing**: `dead = total - gdm`
2. **TTA**: Flip + stereo swap averaging
3. **Ensemble**: Multiple folds and grid sizes

This approach extracts maximum value from the auxiliary tasks during training while ensuring robust inference without metadata dependency.
