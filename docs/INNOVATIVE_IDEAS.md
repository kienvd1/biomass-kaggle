# Innovative Ideas for Biomass Prediction Improvement

## Overview

This document outlines innovative approaches to improve performance on the CSIRO Biomass prediction task. Ideas are ranked by potential impact and implementation complexity.

---

## 🔥 High Impact, Moderate Effort

### 1. Stereo Disparity Features (3D Volume Exploitation)

The stereo pair contains **3D depth information** that's currently underutilized. Disparity/depth directly correlates with biomass volume.

#### How Stereo → 3D Works

```
Left Camera                Right Camera
    |                          |
    |    baseline (B)          |
    |<------------------------>|
    |                          |
    ↓          ↓               ↓
   [==========VEGETATION==========]
        ↑ closer = larger disparity
        ↓ farther = smaller disparity
```

**Key insight**: Taller/denser vegetation is **closer** to the camera → **larger disparity** between left/right views → correlates with **biomass volume**.

#### Current vs Potential

| Current Approach | What's Missing |
|-----------------|----------------|
| FiLM conditioning between views | Only learns implicit relationships |
| Concatenate left+right features | Doesn't explicitly compute depth |

#### Why 3D Volume Helps Biomass

| Feature | Information | Helps With |
|---------|-------------|------------|
| **Disparity mean** | Average vegetation height | Total biomass |
| **Disparity variance** | Height variation | Clover patches (shorter) |
| **Disparity max** | Tallest vegetation | Grass vs clover |
| **Cost volume shape** | Height distribution | Density estimation |

#### Option 1: Feature Correlation (Simplest) ⭐ Recommended

Compute correlation between left/right DINOv2 features at different shifts:

```python
class DisparityFeatures(nn.Module):
    """Extract disparity-based features from stereo pair."""
    
    def __init__(self, feat_dim: int, max_disparity: int = 8):
        super().__init__()
        self.max_disparity = max_disparity
        # Project correlation volume to features
        self.proj = nn.Sequential(
            nn.Linear(max_disparity, feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, feat_dim // 4),
        )
    
    def forward(
        self, 
        feat_left: torch.Tensor,   # (B, N, D) - N tiles, D features
        feat_right: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        B, N, D = feat_left.shape
        
        # Normalize for correlation
        feat_l = F.normalize(feat_left, dim=-1)
        feat_r = F.normalize(feat_right, dim=-1)
        
        # Compute correlation at different "shifts" (simulating disparity)
        # In practice, tiles approximate spatial shifts
        correlations = []
        for d in range(self.max_disparity):
            # Shift right features (circular for simplicity)
            shifted_r = torch.roll(feat_r, shifts=d, dims=1)
            corr = (feat_l * shifted_r).sum(dim=-1)  # (B, N)
            correlations.append(corr)
        
        # Stack: (B, N, max_disparity)
        corr_volume = torch.stack(correlations, dim=-1)
        
        # Soft argmax for disparity estimate per tile
        disparity_weights = F.softmax(corr_volume * 10, dim=-1)  # sharpen
        disparity = (disparity_weights * torch.arange(
            self.max_disparity, device=corr_volume.device
        )).sum(dim=-1)  # (B, N)
        
        # Project correlation volume to features
        disp_features = self.proj(corr_volume)  # (B, N, D//4)
        
        return disp_features.mean(dim=1), disparity.mean(dim=1)  # (B, D//4), (B,)
```

#### Option 2: Cost Volume (More Sophisticated)

Build a proper stereo matching cost volume with 3D convolutions:

```python
class StereoCostVolume(nn.Module):
    """Build cost volume for stereo matching."""
    
    def __init__(self, feat_dim: int, num_disparities: int = 16):
        super().__init__()
        self.num_disparities = num_disparities
        
        # Project features for matching
        self.proj = nn.Linear(feat_dim, feat_dim // 4)
        
        # 3D conv to aggregate cost volume
        self.cost_agg = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )
        
        # Output projection
        self.out_proj = nn.Linear(num_disparities, feat_dim // 4)
    
    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_left: (B, H, W, D) spatial features from left image
            feat_right: (B, H, W, D) spatial features from right image
        Returns:
            disparity_features: (B, D//4) volume-based features
        """
        B, H, W, D = feat_left.shape
        
        # Project
        left = self.proj(feat_left)   # (B, H, W, D//4)
        right = self.proj(feat_right)
        
        # Build cost volume: correlation at each disparity
        cost_volume = []
        for d in range(self.num_disparities):
            if d == 0:
                shifted = right
            else:
                # Shift right image left by d pixels
                shifted = F.pad(right[:, :, d:, :], (0, 0, 0, d), value=0)
            
            # Correlation
            cost = (left * shifted).sum(dim=-1)  # (B, H, W)
            cost_volume.append(cost)
        
        # (B, 1, num_disp, H, W) for 3D conv
        cost_volume = torch.stack(cost_volume, dim=2).unsqueeze(1)
        
        # Aggregate
        cost_agg = self.cost_agg(cost_volume).squeeze(1)  # (B, num_disp, H, W)
        
        # Soft argmax disparity
        cost_agg = cost_agg.permute(0, 2, 3, 1)  # (B, H, W, num_disp)
        disparity_dist = F.softmax(cost_agg, dim=-1)
        
        # Pool to single feature vector
        pooled = disparity_dist.mean(dim=(1, 2))  # (B, num_disparities)
        
        return self.out_proj(pooled)  # (B, D//4)
```

#### Option 3: Simple Cross-Correlation Statistics

Lightweight approach using just correlation statistics:

```python
class SimpleDisparityStats(nn.Module):
    """Extract simple disparity statistics from stereo features."""
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(6, feat_dim // 8)  # 6 stats → compact features
    
    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_left, feat_right: (B, D) pooled features per view
        Returns:
            stats: (B, D//8) disparity-related statistics
        """
        # Normalize
        fl = F.normalize(feat_left, dim=-1)
        fr = F.normalize(feat_right, dim=-1)
        
        # Correlation = similarity between views
        correlation = (fl * fr).sum(dim=-1, keepdim=True)  # (B, 1)
        
        # Difference features (encode disparity indirectly)
        diff = feat_left - feat_right
        diff_norm = diff.norm(dim=-1, keepdim=True)  # (B, 1)
        diff_mean = diff.mean(dim=-1, keepdim=True)  # (B, 1)
        diff_std = diff.std(dim=-1, keepdim=True)    # (B, 1)
        
        # Ratio features
        ratio = feat_left / (feat_right + 1e-6)
        ratio_mean = ratio.mean(dim=-1, keepdim=True)
        ratio_std = ratio.std(dim=-1, keepdim=True)
        
        stats = torch.cat([
            correlation, diff_norm, diff_mean, diff_std, ratio_mean, ratio_std
        ], dim=-1)  # (B, 6)
        
        return self.proj(stats)  # (B, D//8)
```

#### Integration into Current Model

```python
class HierarchicalRatioDINO(nn.Module):
    def __init__(self, ..., use_disparity: bool = False):
        # ... existing init ...
        
        self.use_disparity = use_disparity
        if use_disparity:
            self.disparity_module = DisparityFeatures(feat_dim)
            self.combined_dim += feat_dim // 4  # add disparity features
    
    def forward(self, x_left, x_right, ...):
        # ... existing feature extraction (before pooling) ...
        # feats_left: (B, num_tiles, D), feats_right: (B, num_tiles, D)
        
        if self.use_disparity:
            disp_feat, disp_values = self.disparity_module(feats_left, feats_right)
            # disp_feat: (B, D//4), disp_values: (B,) - can log for debugging
        
        # ... FiLM, attention pooling ...
        f = torch.cat([f_l, f_r], dim=1)
        
        if self.use_disparity:
            f = torch.cat([f, disp_feat], dim=-1)
        
        # ... rest of forward ...
```

**Why it helps:**
- Taller/denser vegetation → larger disparity → more biomass
- Volume estimation is more direct than 2D appearance alone
- Exploits the stereo setup that's currently underutilized by FiLM
- Disparity variance can distinguish grass (uniform) from clover (patchy)

**Implementation effort:** Medium (3-4 hours)

**Recommendation:** Start with **Option 1 (Feature Correlation)** - works directly with tile features. If helpful, upgrade to Cost Volume.

---

### 2. Multi-Scale Feature Fusion

DINOv2 has rich hierarchical features. Currently we only use the final layer.

**Concept:**
```python
class MultiScaleBackbone(nn.Module):
    """Extract and fuse features from multiple transformer blocks."""
    
    def __init__(self, backbone, extract_layers: List[int] = [5, 8, 11]):
        super().__init__()
        self.backbone = backbone
        self.extract_layers = extract_layers
        
        # Projection for each scale
        feat_dim = backbone.embed_dim
        self.scale_projs = nn.ModuleList([
            nn.Linear(feat_dim, feat_dim // len(extract_layers))
            for _ in extract_layers
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        
        features = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in self.extract_layers:
                # CLS token or mean pool
                feat = x[:, 0] if self.backbone.num_prefix_tokens else x.mean(dim=1)
                features.append(feat)
        
        # Project and concatenate
        projected = [proj(f) for proj, f in zip(self.scale_projs, features)]
        combined = torch.cat(projected, dim=-1)
        
        return combined  # Richer multi-scale representation
```

**Why it helps:**
- Early layers capture texture/color (green vs dead)
- Middle layers capture patterns (clover patches)
- Late layers capture semantic content (overall biomass)
- Ensemble of scales is more robust

**Implementation effort:** Low-Medium (2-3 hours)

---

### 3. Cross-Attention Between Views

Currently FiLM provides simple modulation. Full cross-attention is more powerful.

**Concept:**
```python
class CrossViewAttention(nn.Module):
    """Bidirectional cross-attention between stereo views."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn_l2r = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_r2l = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_l = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)
        self.ffn_l = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ffn_r = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        
    def forward(
        self, 
        feat_left: torch.Tensor, 
        feat_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Left attends to right
        attn_l, _ = self.cross_attn_l2r(
            query=feat_left, 
            key=feat_right, 
            value=feat_right
        )
        feat_left = self.norm_l(feat_left + attn_l)
        feat_left = feat_left + self.ffn_l(feat_left)
        
        # Right attends to left
        attn_r, _ = self.cross_attn_r2l(
            query=feat_right, 
            key=feat_left, 
            value=feat_left
        )
        feat_right = self.norm_r(feat_right + attn_r)
        feat_right = feat_right + self.ffn_r(feat_right)
        
        return feat_left, feat_right
```

**Why it helps:**
- Captures correspondences between views
- More expressive than FiLM conditioning
- Can learn to focus on informative regions

**Implementation effort:** Medium (3-4 hours)

---

## 🎯 Medium Impact, Lower Effort

### 4. Uncertainty-Aware Predictions (Heteroscedastic Regression)

Predict both mean and variance. Model learns which samples are hard.

**Concept:**
```python
class UncertaintyHead(nn.Module):
    """Predict mean and log-variance for uncertainty-aware regression."""
    
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.mean_head = nn.Linear(in_dim, out_dim)
        self.logvar_head = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        log_var = self.logvar_head(x)
        # Clamp log_var for stability
        log_var = torch.clamp(log_var, min=-10, max=10)
        return mean, log_var


def gaussian_nll_loss(
    pred_mean: torch.Tensor, 
    pred_logvar: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood loss for heteroscedastic regression."""
    precision = torch.exp(-pred_logvar)
    loss = 0.5 * (precision * (target - pred_mean)**2 + pred_logvar)
    return loss.mean()
```

**Why it helps:**
- Model learns to be uncertain on hard samples
- Reduces overfitting to noisy labels
- Provides calibrated confidence scores
- Can weight ensemble predictions by uncertainty

**Implementation effort:** Low (1-2 hours)

---

### 5. Ordinal Regression for Ratios

Ratios are naturally ordered (0→1). Ordinal regression is more robust than direct regression.

**Concept:**
```python
class OrdinalRatioHead(nn.Module):
    """Predict ratios using ordinal regression."""
    
    def __init__(self, in_dim: int, num_bins: int = 20):
        super().__init__()
        self.num_bins = num_bins
        # Predict cumulative probabilities: P(ratio > threshold_k)
        self.head = nn.Linear(in_dim, num_bins - 1)
        
        # Fixed thresholds: [0.05, 0.10, 0.15, ..., 0.95]
        self.register_buffer(
            'thresholds', 
            torch.linspace(1/num_bins, 1 - 1/num_bins, num_bins - 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cumulative probabilities (monotonic via sigmoid)
        logits = self.head(x)
        cum_probs = torch.sigmoid(logits)  # P(ratio > threshold_k)
        
        # Convert to expected value
        # P(bin_k) = P(ratio > t_{k-1}) - P(ratio > t_k)
        probs = torch.cat([
            cum_probs[:, :1],  # P(ratio > t_0)
            cum_probs[:, :-1] - cum_probs[:, 1:],  # P(t_{k-1} < ratio <= t_k)
            1 - cum_probs[:, -1:]  # P(ratio <= t_{n-1})
        ], dim=1)
        
        # Bin centers
        bin_centers = torch.linspace(0, 1, self.num_bins + 1, device=x.device)
        bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
        
        # Expected value
        ratio = (probs * bin_centers).sum(dim=1, keepdim=True)
        return ratio
```

**Why it helps:**
- More robust to outliers
- Naturally handles bounded outputs [0, 1]
- Can capture multimodal distributions

**Implementation effort:** Medium (2-3 hours)

---

### 6. Test-Time Augmentation (TTA) ✅ IMPLEMENTED

Average predictions from augmented inputs at inference time.

**Current Implementation:** `inference_ratio.ipynb`
- 3 views: original, horizontal flip, brightness/contrast adjustment

**Concept - Extended TTA:**
```python
def get_extended_tta_transforms(img_size: int) -> List[A.Compose]:
    """
    Extended TTA with more geometric and color augmentations.
    
    Current: 3 views (orig, hflip, brightness)
    Extended: 8 views for more robust predictions
    """
    base_norm = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    transforms = [
        # 1. Original
        A.Compose(base_norm),
        
        # 2. Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)] + base_norm),
        
        # 3. Vertical flip (field images can be viewed upside down)
        A.Compose([A.VerticalFlip(p=1.0)] + base_norm),
        
        # 4. Both flips
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base_norm),
        
        # 5. Slight brightness up (simulate sunny conditions)
        A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), p=1.0)] + base_norm),
        
        # 6. Slight brightness down (simulate cloudy conditions)  
        A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), p=1.0)] + base_norm),
        
        # 7. Slight scale up (1.05x zoom, center crop)
        A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.95, 0.95), p=1.0),
        ] + base_norm[1:]),  # Skip resize since RandomResizedCrop handles it
        
        # 8. Slight scale down (0.95x, pad)
        A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(1.0, 1.0), ratio=(0.95, 1.05), p=1.0),
        ] + base_norm[1:]),
    ]
    return transforms


# Stereo-aware TTA: swap left/right views
def predict_with_stereo_tta(model, x_left, x_right):
    """TTA that also swaps stereo views."""
    pred_normal = model(x_left, x_right)
    pred_swapped = model(x_right, x_left)  # Swap views
    return (pred_normal + pred_swapped) / 2
```

**Why it helps:**
- Free performance boost at inference
- Reduces variance in predictions
- Robust to orientation variations
- **Stereo swap TTA:** Model should be symmetric to view order

**Implementation effort:** ✅ Basic done, extended: Low (1 hour)

---

### 7. Pseudo-Labeling with Test Data

Use model predictions on test data as additional training signal.

**Concept:**
```python
def pseudo_label_training(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    confidence_threshold: float = 0.8,
    num_iterations: int = 3
):
    """Iterative pseudo-labeling."""
    
    for iteration in range(num_iterations):
        # 1. Predict on test data with uncertainty
        test_preds, test_uncertainties = predict_with_uncertainty(model, test_loader)
        
        # 2. Select high-confidence predictions
        confident_mask = test_uncertainties < confidence_threshold
        pseudo_labels = test_preds[confident_mask]
        pseudo_samples = test_loader.dataset[confident_mask]
        
        print(f"Iteration {iteration}: {confident_mask.sum()} pseudo-labels")
        
        # 3. Create combined dataset
        combined_loader = create_combined_loader(train_loader, pseudo_samples, pseudo_labels)
        
        # 4. Retrain model
        model = train_model(model, combined_loader)
        
    return model
```

**Why it helps:**
- Leverages unlabeled test data
- Can discover patterns in test distribution
- Particularly useful when test distribution differs from train

**Implementation effort:** Medium (3-4 hours)

---

## 💡 Innovative/Experimental

### 8. Contrastive Pre-training on Stereo Pairs

Self-supervised learning: left and right views of same scene should be similar.

**Concept:**
```python
class StereoContrastiveLoss(nn.Module):
    """NT-Xent loss for stereo pairs."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, 
        feat_left: torch.Tensor,   # (B, D)
        feat_right: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        # Normalize features
        feat_left = F.normalize(feat_left, dim=1)
        feat_right = F.normalize(feat_right, dim=1)
        
        batch_size = feat_left.size(0)
        
        # Similarity matrix
        features = torch.cat([feat_left, feat_right], dim=0)  # (2B, D)
        sim_matrix = torch.mm(features, features.t()) / self.temperature  # (2B, 2B)
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(sim_matrix.device)
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
```

**Why it helps:**
- Pre-train backbone on stereo correspondence
- Learn view-invariant representations
- Useful if you have more unlabeled stereo pairs

**Implementation effort:** Medium-High (4-6 hours)

---

### 9. Biomass-Specific Color Features (Vegetation Indices)

Extract vegetation indices as explicit input features. These are well-established in **agricultural remote sensing** and designed specifically to distinguish vegetation types.

#### Scientific Background

**Why this works - Chlorophyll Spectral Response:**

| Vegetation Type | Color | Chlorophyll | R | G | B | ExG | GRVI |
|-----------------|-------|-------------|---|---|---|-----|------|
| **Green grass** | Green | High | Low | High | Low | High (+) | High (+) |
| **Dead grass** | Brown/Yellow | None | High | Med | Low | Low/Neg | Low/Neg |
| **Clover** | Dark green | High | Low | High | Med | High (+) | High (+) |
| **Soil** | Brown | None | High | Med | Med | Negative | Negative |

**Chlorophyll physics:**
- **Living green plants** contain chlorophyll which **strongly absorbs red (660nm) and blue (450nm)** light for photosynthesis, but **reflects green (550nm)**
- **Dead/senescent vegetation** has degraded chlorophyll → reflects more evenly across R/G/B, appears brown/yellow
- This is why these indices have been used in precision agriculture for 30+ years

#### Index Definitions

| Index | Formula | What it measures |
|-------|---------|------------------|
| **ExG** (Excess Green) | `2G - R - B` | Green vegetation amount (high chlorophyll) |
| **ExR** (Excess Red) | `1.4R - G` | Dead matter/soil (no chlorophyll) |
| **ExGR** | `ExG - ExR` | Green vs Dead ratio |
| **GRVI** | `(G-R)/(G+R)` | Normalized greenness (lighting-robust) |
| **VARI** | `(G-R)/(G+R-B)` | Atmospherically resistant greenness |
| **NormG** | `G/(R+G+B)` | Normalized green channel |

**Concept:**
```python
class VegetationIndices(nn.Module):
    """
    Compute vegetation indices from RGB image.
    
    Based on agricultural remote sensing research:
    - Woebbecke et al. (1995) - ExG, ExR for crop/weed discrimination
    - Gitelson et al. (2002) - VARI for vegetation fraction estimation
    
    These indices exploit chlorophyll's spectral signature:
    - Chlorophyll absorbs R and B, reflects G → high ExG for green plants
    - Dead vegetation lacks chlorophyll → low ExG, higher ExR
    """
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, H, W) RGB image normalized to [0, 1]
        Returns:
            indices: (B, num_indices, H, W) vegetation indices
        """
        r, g, b = img.unbind(dim=1)
        
        # === GREEN BIOMASS INDICATORS ===
        # Excess Green Index (ExG) - HIGH for green, LOW/NEG for dead
        exg = 2 * g - r - b
        
        # Green-Red Vegetation Index (GRVI) - normalized, lighting-robust
        # Range: [-1, 1], positive = green, negative = dead/soil
        grvi = (g - r) / (g + r + 1e-6)
        
        # Visible Atmospherically Resistant Index (VARI)
        # More robust to shadows and atmospheric effects
        vari = (g - r) / (g + r - b + 1e-6)
        
        # === DEAD BIOMASS INDICATORS ===
        # Excess Red Index (ExR) - HIGH for dead/brown, LOW for green
        exr = 1.4 * r - g
        
        # ExGR: Green minus Red excess - separates green from dead
        exgr = exg - exr
        
        # === GENERAL ===
        # Normalized Green - proportion of green in total light
        norm_g = g / (r + g + b + 1e-6)
        
        return torch.stack([exg, exr, exgr, grvi, norm_g, vari], dim=1)


class VegetationFeaturesPooled(nn.Module):
    """
    Extract pooled vegetation index statistics as features.
    
    Returns mean, std, and percentiles of each index across the image,
    providing a compact representation of vegetation distribution.
    """
    
    def __init__(self):
        super().__init__()
        self.vi = VegetationIndices()
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, H, W) RGB image (ImageNet normalized)
        Returns:
            features: (B, num_features) pooled vegetation features
        """
        # Denormalize from ImageNet stats to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img_denorm = (img * std + mean).clamp(0, 1)
        
        # Compute vegetation indices: (B, 6, H, W)
        vi = self.vi(img_denorm)
        
        # Pool statistics per index
        feats = []
        for i in range(vi.size(1)):
            idx = vi[:, i]  # (B, H, W)
            feats.extend([
                idx.mean(dim=(-2, -1)),       # Mean
                idx.std(dim=(-2, -1)),        # Std (texture/variation)
                idx.quantile(0.1, dim=-1).quantile(0.1, dim=-1),  # Low percentile
                idx.quantile(0.9, dim=-1).quantile(0.9, dim=-1),  # High percentile
            ])
        
        return torch.stack(feats, dim=1)  # (B, 24) = 6 indices × 4 stats
```

#### Integration Example

```python
class HierarchicalRatioDINOWithVI(HierarchicalRatioDINO):
    """Add vegetation indices as auxiliary features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vi_features = VegetationFeaturesPooled()
        
        # Expand shared projection to include VI features
        # Original: combined_dim (1536) → hidden_dim
        # New: combined_dim + 48 (24 per view) → hidden_dim
        vi_dim = 24 * 2  # 24 features × 2 views
        old_in = self.shared_proj[1].in_features
        new_in = old_in + vi_dim
        self.shared_proj[1] = nn.Linear(new_in, self.hidden_dim)
    
    def forward(self, x_left, x_right, return_ratios=False):
        # Get DINOv2 features as before
        tiles_left, tiles_right = self._extract_tiles_fused(x_left, x_right)
        # ... FiLM, attention pooling ...
        f = torch.cat([f_l, f_r], dim=1)
        
        # Add vegetation index features
        vi_left = self.vi_features(x_left)    # (B, 24)
        vi_right = self.vi_features(x_right)  # (B, 24)
        f = torch.cat([f, vi_left, vi_right], dim=1)
        
        f = self.shared_proj(f)
        # ... rest of forward pass ...
```

**Why it helps:**
- **Domain knowledge** - designed by agronomists for exactly this task
- **ExG strongly correlates with green biomass** (chlorophyll content)
- **ExR correlates with dead biomass** (brown/yellow color)
- **GRVI is lighting-invariant** - handles varying field conditions
- **Complements learned features** - explicit signal the model might miss
- **Cheap to compute** - no learnable parameters, just RGB math

**Expected impact:** +1-2% R², especially for Green and Dead targets

**Implementation effort:** Low (1-2 hours)

**References:**
- Woebbecke et al. (1995) - "Color Indices for Weed Identification Under Various Soil, Residue, and Lighting Conditions"
- Gitelson et al. (2002) - "Novel algorithms for remote estimation of vegetation fraction"

---

### 10. Learnable Augmentation Policy (AutoAugment-style)

Learn which augmentations help most for this specific task.

**Concept:**
```python
class LearnableAugmentation(nn.Module):
    """Differentiable augmentation with learnable parameters."""
    
    def __init__(self):
        super().__init__()
        # Learnable augmentation magnitudes
        self.brightness_mag = nn.Parameter(torch.tensor(0.2))
        self.contrast_mag = nn.Parameter(torch.tensor(0.2))
        self.saturation_mag = nn.Parameter(torch.tensor(0.2))
        self.hue_mag = nn.Parameter(torch.tensor(0.05))
        
        # Learnable probabilities
        self.aug_probs = nn.Parameter(torch.ones(4) * 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(self.aug_probs)
        
        if torch.rand(1) < probs[0]:
            x = adjust_brightness(x, 1 + torch.randn(1) * self.brightness_mag)
        if torch.rand(1) < probs[1]:
            x = adjust_contrast(x, 1 + torch.randn(1) * self.contrast_mag)
        # ... etc
        
        return x
```

**Why it helps:**
- Task-specific augmentation policy
- May discover useful augmentations for field images
- Reduces manual tuning

**Implementation effort:** High (6-8 hours)

---

### 11. Graph Neural Network on Tiles

Treat tiles as graph nodes, learn spatial relationships.

**Concept:**
```python
class TileGNN(nn.Module):
    """GNN over image tiles."""
    
    def __init__(self, feat_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConvLayer(feat_dim, feat_dim)
            for _ in range(num_layers)
        ])
        
        # Adjacency based on spatial proximity
        # 4 tiles in 2x2 grid: all connected
        self.register_buffer('adj', torch.ones(4, 4) - torch.eye(4))
        
    def forward(self, tile_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tile_features: (B, num_tiles, D)
        Returns:
            updated_features: (B, num_tiles, D)
        """
        x = tile_features
        for layer in self.layers:
            x = layer(x, self.adj)
        return x
```

**Why it helps:**
- Captures spatial relationships between tiles
- "Green patch next to dead patch" patterns
- More flexible than fixed pooling

**Implementation effort:** Medium-High (4-5 hours)

---

## ⚠️ Known Challenges

### The Dead Biomass Prediction Dilemma

**Problem:** Dead consistently has the worst R² among all targets, even when Green/Total/GDM perform well.

#### Why This Happens

**Option A: Derive Dead (current approach)**
```
Dead = Total - GDM  (hierarchical)
Dead = Total × dead_ratio  (softmax)
```
- ✅ Guarantees mathematical consistency
- ❌ Dead becomes the "error sink" - all prediction errors accumulate in Dead

**Option B: Directly Predict Dead**
- ❌ Weak visual signal (brown ≈ soil ≈ shadows)
- ❌ Noisy gradients corrupt the shared backbone
- ❌ Risk of constraint violations (Dead > Total, negative values)
- ❌ Overall model performance degrades

#### Visual Signal Comparison

| Component | Visual Signature | Distinguishability |
|-----------|-----------------|-------------------|
| **Green** | Strong chlorophyll → high G channel | ✅ Easy |
| **Total** | Overall vegetation density/volume | ✅ Easy |
| **GDM** | Green + Clover (both alive) | ✅ Moderate |
| **Dead** | Brown/tan, similar to soil/shadows | ❌ Hard |

Dead lacks chlorophyll → no distinctive spectral signature. It visually resembles:
- Dry soil
- Shadows  
- Old thatch/residue

#### Potential Solutions

**1. Residual Correction Head** ⭐ Recommended
```python
class HierarchicalWithDeadCorrection(nn.Module):
    def forward(self, ...):
        # Standard hierarchical prediction
        dead_base = total - gdm  # derived
        
        # Small learned correction (bounded to prevent large deviations)
        dead_correction = torch.tanh(self.head_dead_residual(f)) * 0.1 * total
        dead = dead_base + dead_correction
        
        return green, dead, clover, gdm, total
```
- Maintains mathematical consistency (mostly)
- Allows model to learn Dead-specific corrections
- Correction is bounded to prevent wild deviations

**2. Auxiliary Dead Head with Soft Constraint**
```python
dead_derived = total - gdm
dead_direct = self.head_dead(f)
dead = 0.7 * dead_derived + 0.3 * dead_direct  # blend
```
- Soft constraint keeps consistency while allowing direct learning
- Tunable blend ratio

**3. Uncertainty-Weighted Loss**
```python
# Model predicts uncertainty for each target
loss_dead = (pred_dead - true_dead)**2 / (2 * sigma_dead**2) + log(sigma_dead)
```
- Model learns to downweight uncertain Dead predictions
- Prevents Dead errors from dominating the loss

**4. Dead-Specific Features**
- **ExR (Excess Red)** specifically correlates with dead/brown material
- **ExGR = ExG - ExR** separates green from dead
- Use `--use-vegetation-indices` flag

**5. Two-Stage Model**
1. Stage 1: Train on Total, GDM, Green (current model)
2. Stage 2: Small separate model predicts Dead using:
   - Stage 1 outputs as features
   - VI features (especially ExR)
   - Original image features

**6. Ensemble Dead Only**
- Train multiple models with different seeds
- Average only Dead predictions (reduces variance)
- Keep single model for other targets

#### Priority for Dead Improvement

| Solution | Impact | Effort | Priority |
|----------|--------|--------|----------|
| Vegetation Indices (ExR) | +0.5-1% Dead R² | 1 hr | ⭐⭐⭐⭐⭐ |
| Residual Correction | +1-2% Dead R² | 2 hrs | ⭐⭐⭐⭐ |
| Uncertainty Loss | +0.5-1% Dead R² | 2-3 hrs | ⭐⭐⭐ |
| Soft Constraint Blend | +0.5-1% Dead R² | 1 hr | ⭐⭐⭐ |
| Two-Stage Model | +1-2% Dead R² | 4 hrs | ⭐⭐ |
| Ensemble Dead | +0.5% Dead R² | 0 hrs (inference) | ⭐⭐ |

---

## 📊 Summary: Priority Matrix

### Current Implementation Status

| Feature | Status | Location |
|---------|--------|----------|
| TTA (3 views) | ✅ Implemented | `inference_ratio.ipynb` |
| FiLM Conditioning | ✅ Implemented | `models_ratio.py` |
| Attention Pooling | ✅ Implemented | `models_ratio.py` |
| Hierarchical Ratios | ✅ Implemented | `models_ratio.py` |
| Multi-fold Ensemble | ✅ Implemented | Training + Inference |
| Vegetation Indices | ✅ Implemented | `models_ratio.py` (`--use-vegetation-indices`) |
| Multi-Scale Features | ✅ Implemented | `models_ratio.py` (`--use-multiscale`) |

### Priority Matrix (Remaining Ideas)

| Idea | Impact | Effort | Priority | Notes |
|------|--------|--------|----------|-------|
| TTA | +1-2% R² | 1-2 hrs | ✅ **Done** | 3 views: orig, hflip, brightness |
| Vegetation Indices | +1-2% R² | 1-2 hrs | ✅ **Done** | `--use-vegetation-indices` |
| Multi-Scale Features | +2-3% R² | 2-3 hrs | ✅ **Done** | `--use-multiscale --multiscale-layers 5 8 11` |
| Uncertainty Loss | +1-2% R² | 2-3 hrs | ⭐⭐⭐⭐ | Heteroscedastic regression |
| Cross-Attention | +1-3% R² | 3-4 hrs | ⭐⭐⭐ | Replace FiLM for stereo fusion |
| **Stereo Disparity** | +2-3% R² | 3-4 hrs | ⭐⭐⭐⭐ | 3D volume from stereo pairs |
| Pseudo-Labeling | +2-4% R² | 3-4 hrs | ⭐⭐⭐ | Use test predictions |
| Ordinal Regression | +1-2% R² | 2-3 hrs | ⭐⭐ | For bounded [0,1] ratios |
| Contrastive Pre-train | +1-3% R² | 4-6 hrs | ⭐⭐ | Self-supervised on stereo |
| Tile GNN | +1-2% R² | 4-5 hrs | ⭐ | Spatial tile relationships |
| AutoAugment | +1-2% R² | 6-8 hrs | ⭐ | Learn augmentation policy |

---

## 🚀 Recommended Implementation Order

1. ~~**TTA**~~ ✅ Done - 3 views (original, hflip, brightness)
2. ~~**Vegetation Indices**~~ ✅ Done - `--use-vegetation-indices`
3. ~~**Multi-Scale Features**~~ ✅ Done - `--use-multiscale`
4. **Stereo Disparity** - Exploit 3D volume from stereo pairs (high potential)
5. **Dead Residual Correction** - Address the Dead prediction dilemma
6. **Uncertainty Loss** - Helps with noisy labels, calibrated confidence
7. **Cross-Attention** - If above improvements plateau

---

## Notes

- Expected gains are rough estimates based on similar tasks
- Actual gains depend on current model performance and data characteristics
- Some ideas may be complementary (e.g., TTA + multi-scale)
- Others may overlap (e.g., cross-attention vs FiLM)
- Always validate on holdout set before submitting

