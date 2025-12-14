# CSIRO Biomass Model Improvement Strategies

This document outlines strategies to improve the weighted R² score for the CSIRO Biomass prediction competition.

---

## 1. Model Architecture Improvements

### 1.1 Larger DINOv2 Backbones

The current default is `vit_base_patch14_reg4_dinov2.lvd142m`. Larger models typically yield better feature representations.

```bash
# Try the large variant (requires gradient checkpointing for memory)
python -m src.train --backbone vit_large_patch14_dinov2.lvd142m --grad-ckpt

# Or the non-reg4 base variant
python -m src.train --backbone vit_base_patch14_dinov2.lvd142m
```

### 1.2 Grid Size Experiments

The tiled encoding splits images into a grid. More tiles capture finer local details.

```bash
# Default is 2x2, try 3x3 for more local detail
python -m src.train --grid 3 3

# Or 4x4 (may need smaller batch size)
python -m src.train --grid 4 4 --batch-size 8
```

### 1.3 Attention Pooling (Code Change)

Replace mean pooling over tiles with learnable attention weights in `src/models.py`:

```python
# In TwoStreamDINOTiled or TwoStreamDINOTiledFiLM
class TileAttentionPool(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.Tanh(),
            nn.Linear(feat_dim // 4, 1),
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        # tiles: (B, num_tiles, D)
        weights = F.softmax(self.attn(tiles), dim=1)  # (B, num_tiles, 1)
        return (tiles * weights).sum(dim=1)  # (B, D)
```

### 1.4 Deeper Regression Heads

Current heads are shallow (2 linear layers). Try deeper heads:

```python
def _make_head() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(self.combined, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Dropout(dropout / 2),
        nn.Linear(hidden // 2, 1),
    )
```

---

## 2. Training Strategy Refinements

### 2.1 Two-Stage Training Tuning

Current defaults: Stage 1 = 5 epochs frozen backbone, Stage 2 = full finetune.

```bash
# Longer Stage 1 for better head initialization
python -m src.train --freeze-epochs 8 --head-lr-stage1 5e-4

# Shorter Stage 1, more aggressive Stage 2
python -m src.train --freeze-epochs 3 --head-lr-stage1 1e-3 --backbone-lr 2e-5
```

### 2.2 Learning Rate Experiments

```bash
# Higher backbone LR in Stage 2
python -m src.train --backbone-lr 2e-5 --lr 1e-4

# Lower head LR in Stage 1 for stability
python -m src.train --head-lr-stage1 5e-4
```

### 2.3 Loss Function Experiments

```bash
# Huber loss (robust to outliers)
python -m src.train --loss huber

# RMSE loss (directly optimizes RMSE)
python -m src.train --loss rmse

# MAE loss (L1, robust to outliers)
python -m src.train --loss mae
```

### 2.4 Longer Training

```bash
# Increase epochs (early stopping will prevent overfitting)
python -m src.train --epochs 80 --patience 15
```

### 2.5 Best Metric Selection

```bash
# Select best model by R² instead of loss
python -m src.train --best-metric r2
```

---

## 3. Data & Augmentation Improvements

### 3.1 Stronger Augmentations

Modify `src/dataset.py` `get_train_transforms()`:

```python
def get_train_transforms(img_size: int = 518, aug_prob: float = 0.5) -> A.ReplayCompose:
    return A.ReplayCompose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=aug_prob),
        A.VerticalFlip(p=aug_prob),
        A.RandomRotate90(p=aug_prob),  # ADD: 90° rotations
        A.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},  # Increase from 0.1
            scale=(0.8, 1.2),  # Wider scale range
            rotate=(-30, 30),  # Wider rotation
            border_mode=cv2.BORDER_REFLECT_101,
            p=aug_prob,
        ),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),  # ADD
        ], p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),  # ADD
        ], p=aug_prob),
        A.CoarseDropout(
            num_holes_range=(1, 12),  # More holes
            hole_height_range=(16, 64),  # Larger holes
            hole_width_range=(16, 64),
            fill=0,
            p=0.4,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

### 3.2 MixUp / CutMix (Code Change)

Add to training loop in `src/trainer.py`:

```python
def mixup_data(x_left, x_right, targets, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x_left.size(0)
    index = torch.randperm(batch_size, device=x_left.device)

    mixed_left = lam * x_left + (1 - lam) * x_left[index]
    mixed_right = lam * x_right + (1 - lam) * x_right[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]

    return mixed_left, mixed_right, mixed_targets
```

### 3.3 Target Normalization (Code Change)

Log-transform targets to handle wide value ranges:

```python
# In BiomassDataset.__init__
self.targets = np.log1p(self.targets)  # log(1 + x) transform

# In inference, reverse transform
preds = np.expm1(preds)  # exp(x) - 1
```

---

## 4. Test Time Augmentation (TTA)

### 4.1 Enhanced TTA Transforms

Modify `src/inference.py` `get_tta_transforms()`:

```python
def get_tta_transforms(img_size: int = 518) -> List[A.Compose]:
    base = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    return [
        # Original
        A.Compose([A.Resize(img_size, img_size), *base]),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(img_size, img_size), *base]),
        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0), A.Resize(img_size, img_size), *base]),
        # 90° rotation
        A.Compose([A.Rotate(limit=(90, 90), p=1.0), A.Resize(img_size, img_size), *base]),
        # 180° rotation
        A.Compose([A.Rotate(limit=(180, 180), p=1.0), A.Resize(img_size, img_size), *base]),
        # 270° rotation
        A.Compose([A.Rotate(limit=(270, 270), p=1.0), A.Resize(img_size, img_size), *base]),
        # H+V flip
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Resize(img_size, img_size), *base]),
    ]
```

### 4.2 Multi-Scale TTA

```python
def get_multiscale_tta_transforms() -> List[A.Compose]:
    base = [A.Normalize(...), ToTensorV2()]
    scales = [480, 518, 560]
    transforms = []
    for size in scales:
        transforms.append(A.Compose([A.Resize(size, size), *base]))
        transforms.append(A.Compose([A.HorizontalFlip(p=1.0), A.Resize(size, size), *base]))
    return transforms
```

### 4.3 Run Inference with TTA

```bash
python -m src.inference --model-dir outputs/exp1 --tta --output submission_tta.csv
```

---

## 5. Ensemble Strategies

### 5.1 Multiple Backbones

Train models with different backbones and ensemble:

```bash
# Train different backbones
python -m src.train --backbone vit_base_patch14_reg4_dinov2.lvd142m --output-dir outputs/base_reg4
python -m src.train --backbone vit_base_patch14_dinov2.lvd142m --output-dir outputs/base
python -m src.train --backbone vit_small_patch14_reg4_dinov2.lvd142m --output-dir outputs/small_reg4

# Ensemble
python -m src.inference \
    --model-dir outputs/base_reg4 outputs/base outputs/small_reg4 \
    --weights 0.5 0.3 0.2 \
    --output submission_ensemble.csv
```

### 5.2 Different CV Strategies

Different fold splits create model diversity:

```bash
python -m src.train --cv-strategy group_month --output-dir outputs/cv_month
python -m src.train --cv-strategy group_date --output-dir outputs/cv_date
python -m src.train --cv-strategy stratified --output-dir outputs/cv_strat

# Ensemble all
python -m src.inference \
    --model-dir outputs/cv_month outputs/cv_date outputs/cv_strat \
    --output submission_cv_ensemble.csv
```

### 5.3 Different Random Seeds

```bash
python -m src.train --seed 18 --output-dir outputs/seed18
python -m src.train --seed 42 --output-dir outputs/seed42
python -m src.train --seed 123 --output-dir outputs/seed123
```

### 5.4 Optimizing Ensemble Weights

Use validation R² to find optimal weights:

```python
# Pseudo-code for weight optimization
from scipy.optimize import minimize

def objective(weights, oof_preds_list, oof_targets):
    weights = weights / weights.sum()  # Normalize
    blended = sum(w * p for w, p in zip(weights, oof_preds_list))
    r2 = compute_weighted_r2(blended, oof_targets, [0.1, 0.1, 0.1, 0.2, 0.5])
    return -r2  # Minimize negative R²

result = minimize(objective, x0=[1/N]*N, args=(oof_preds, targets), method='Nelder-Mead')
optimal_weights = result.x / result.x.sum()
```

---

## 6. Hyperparameter Search with Optuna

### 6.1 Basic Search

```bash
python -m src.optuna_search \
    --backbone vit_base_patch14_reg4_dinov2.lvd142m \
    --n-trials 100 \
    --max-epochs 25 \
    --cv-strategy group_month
```

### 6.2 Extended Search Space

Modify `src/optuna_search.py` to include more parameters:

```python
params = {
    "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
    "backbone_lr": trial.suggest_float("backbone_lr", 5e-7, 5e-5, log=True),
    "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    "hidden_ratio": trial.suggest_float("hidden_ratio", 0.1, 0.5),
    "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
    "batch_size": trial.suggest_categorical("batch_size", [8, 12, 16, 24]),
    "freeze_epochs": trial.suggest_int("freeze_epochs", 2, 10),
    "head_lr_stage1": trial.suggest_float("head_lr_stage1", 1e-4, 5e-3, log=True),
    "aug_prob": trial.suggest_float("aug_prob", 0.3, 0.7),
    "grid_size": trial.suggest_categorical("grid_size", [2, 3]),
}
```

---

## 7. Advanced Techniques

### 7.1 Stochastic Weight Averaging (SWA)

Add SWA to capture flatter minima:

```python
from torch.optim.swa_utils import AveragedModel, SWALR

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

# After warmup epochs, switch to SWA
for epoch in range(swa_start, total_epochs):
    train_one_epoch(...)
    swa_model.update_parameters(model)
    swa_scheduler.step()

# Update batch norm stats
torch.optim.swa_utils.update_bn(train_loader, swa_model)
```

### 7.2 Pseudo-Labeling

Use confident predictions on test data as additional training samples:

```python
# 1. Train initial model
# 2. Predict on test set
# 3. Select high-confidence samples (low variance across folds)
# 4. Add pseudo-labels to training data
# 5. Retrain with augmented dataset
```

### 7.3 Knowledge Distillation

Train a smaller/faster model using ensemble as teacher:

```python
# Teacher: ensemble of 5 fold models
# Student: single model

def distillation_loss(student_preds, teacher_preds, targets, alpha=0.5, temp=3.0):
    soft_loss = F.mse_loss(student_preds / temp, teacher_preds / temp)
    hard_loss = F.mse_loss(student_preds, targets)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### 7.4 Auxiliary Tasks

Add auxiliary prediction heads for metadata:

```python
# Predict NDVI and Height as auxiliary tasks
self.head_ndvi = nn.Linear(self.combined, 1)
self.head_height = nn.Linear(self.combined, 1)

# Multi-task loss
loss = main_loss + 0.1 * ndvi_loss + 0.1 * height_loss
```

---

## 10. 5-Head model + auxiliary heads (no metadata at inference)

This repo also includes a dedicated 5-output model (`src/train_5head.py`, `src/models_5head.py`) with **auxiliary classification heads** (State/Month/Species).

**Important:** this setup is compatible with **no metadata at inference**, because:
- The model input is still only `(left_image, right_image)`.
- Metadata is used only as **training labels** for auxiliary losses (`--use-aux-heads`).
- At inference, auxiliary heads are optional. If you enable context adjustment (`--apply-context-adjustment`), it uses **predicted** aux logits (still no metadata required).

### 10.1 Recommended, inference-safe improvements (highest ROI)

#### A) Stereo augmentation: same geometry, independent photometric
Current `BiomassDataset` uses `A.ReplayCompose` so the **same** transform is replayed on both views. That’s correct for **geometry**, but it also unintentionally ties **photometric** noise/jitter between left/right.

Recommendation:
- Apply **the same** spatial transform (crop/resize/flip/affine) to both views.
- Apply **independent** color/noise/blur transforms to each view.

Why:
- Improves robustness to real stereo differences (exposure, blur, sensor noise).
- Reduces “shortcut learning” where the model expects perfectly matched photometrics.

#### B) Replace hard-coded context adjustment with a learned adjustment head
`FiveHeadDINO._apply_context_adjustment()` currently uses fixed multipliers (hand-crafted priors for WA/month/species). This can help, but can also hurt under distribution shift or misclassified aux labels.

Recommendation:
- Use predicted aux probabilities and learn a small MLP that outputs **soft adjustment factors** (or residuals) for Dead/Clover (and optionally others).
- Keep it differentiable and trained end-to-end.

This keeps inference metadata-free while letting the model learn calibration instead of relying on constants.

#### C) Species imbalance: simplify or stabilize
Species groups are highly imbalanced (e.g. `Mixed` is extremely rare) and current class weights are very large for rare groups.

Recommendations (pick 1):
- Reduce to fewer species groups (e.g. 4 buckets) to avoid ultra-rare classes, or
- Keep 8 groups but lower `--aux-species-weight` and add label smoothing to the species CE, or
- Compute species weights **per fold** from `train_df` instead of hard-coded global weights.

### 10.2 Suggested hyperparameter search space (for `src/train_5head.py`)

Start with single-fold sweeps (e.g. `--train-folds 0`) to iterate quickly, then confirm with full CV.

- **grid**: `(2,2)`, `(3,3)` (optionally `(2,3)` as a middle point)
- **dropout**: `0.1 – 0.35`
- **hidden_ratio**: `0.25 – 1.0`
- **loss**:
  - `ConstrainedMSELoss` with `--constraint-weight 0` (disable constraints), OR
  - `--use-dead-aware-loss` (often improves Dead stability)
- **frozen backbone (head-only)**
  - `--freeze-backbone`
  - `--head-lr-stage1`: `5e-4 – 3e-3` (log-scale)
- **two-stage finetune (optional)**
  - `--two-stage`
  - `--freeze-epochs`: `2 – 10`
  - `--head-lr-stage1`: `5e-4 – 3e-3`
  - `--lr`: `5e-5 – 3e-4`
  - `--backbone-lr`: `5e-6 – 5e-5`
- **aux head weights**
  - `--aux-month-weight`: `3 – 8`
  - `--aux-state-weight`: `1 – 5`
  - `--aux-species-weight`: `0.5 – 3` (keep lower if species is noisy/rare)
- **aug_prob**: `0.3 – 0.7`
- **grad_clip**: `0.5 – 2.0`

Example “fast sweep” command:

```bash
python -m src.train_5head \
  --train-folds 0 \
  --epochs 25 \
  --freeze-backbone \
  --grid 2 2 \
  --constraint-weight 0 \
  --use-aux-heads \
  --aux-month-weight 5.0 \
  --aux-state-weight 3.0 \
  --aux-species-weight 2.0
```

### 10.3 A/B tests worth running (in order)
- **2x2 vs 3x3 grid** (keep everything else fixed)
- **constraint on vs off** (`--constraint-weight 0` vs default)
- **aux heads on vs off** (`--use-aux-heads`)
- **DeadAwareLoss vs baseline** (`--use-dead-aware-loss`)
- **context adjustment off vs on** (`--apply-context-adjustment`) — only if you use a learned adjustment; hard-coded priors can overfit

---

## 8. Priority Ranking

| Improvement | Expected Gain | Effort | Priority |
|------------|---------------|--------|----------|
| TTA (rotations) | +0.005-0.01 | Low | **High** |
| Ensemble 2+ backbones | +0.02-0.03 | Medium | **High** |
| Optuna search | +0.02-0.05 | Medium | **High** |
| Longer training (80 epochs) | +0.01-0.02 | Low | **High** |
| Different CV strategies ensemble | +0.01-0.02 | Medium | Medium |
| Grid 3x3 | +0.005-0.015 | Low | Medium |
| Stronger augmentations | +0.005-0.01 | Low | Medium |
| vit_large backbone | +0.01-0.02 | Medium | Medium |
| MixUp/CutMix | +0.005-0.01 | Medium | Medium |
| Attention pooling | +0.005-0.01 | Medium | Low |
| SWA | +0.005-0.01 | Medium | Low |
| Pseudo-labeling | +0.01-0.02 | High | Low |

---

## 9. Quick Start Commands

```bash
# 1. Train baseline with longer epochs
python -m src.train --epochs 80 --patience 15 --output-dir outputs/baseline_long

# 2. Train second backbone
python -m src.train --backbone vit_base_patch14_dinov2.lvd142m --epochs 80 --output-dir outputs/base_noreg

# 3. Ensemble inference with TTA
python -m src.inference \
    --model-dir outputs/baseline_long outputs/base_noreg \
    --weights 0.6 0.4 \
    --tta \
    --output submission_final.csv

# 4. Run Optuna to find better hyperparameters
python -m src.optuna_search \
    --backbone vit_base_patch14_reg4_dinov2.lvd142m \
    --n-trials 50 \
    --max-epochs 20
```
