# Improvement Strategies for CSIRO Biomass Prediction

## Problem Analysis

**Competition Metric**: Weighted R² with fixed weights:
| Target | Weight | Difficulty |
|--------|--------|------------|
| Dry_Green_g | 0.1 | Medium |
| Dry_Dead_g | 0.1 | **Hard** |
| Dry_Clover_g | 0.1 | **Hard** |
| GDM_g | 0.2 | Medium |
| Dry_Total_g | 0.5 | Easier |

**Key Insight**: Clover and dead are hardest to predict but only contribute 20% of the score. Focus on improving them without hurting total/GDM.

---

## Strategy 1: Hierarchical Prediction (Decomposition)

Instead of predicting all 5 targets independently, use relationships:
- `GDM = Green + Clover`
- `Total = GDM + Dead`

### Approach A: Predict components, derive sums
```
Predict: Green, Dead, Clover
Derive:  GDM = Green + Clover
         Total = Green + Dead + Clover
```
**Pros**: Ensures mathematical consistency
**Cons**: Errors in components propagate to sums

### Approach B: Predict sums, derive components (Top-Down)
```
Predict: Total, GDM, Green
Derive:  Dead = Total - GDM
         Clover = GDM - Green
```
**Pros**: Focuses model capacity on high-weight targets
**Cons**: Derived values can be negative (needs clipping)

### Approach C: Residual prediction
```
Predict: Total (primary)
         Dead_ratio = Dead / Total
         Green_ratio = Green / GDM
Derive:  Dead = Total × Dead_ratio
         GDM = Total - Dead
         Green = GDM × Green_ratio
         Clover = GDM - Green
```

---

## Strategy 2: Multi-Scale Feature Extraction

Clover appears as small patches, dead grass has texture patterns.

### Implementation
```python
# Use intermediate DINOv2 features
feat_layer6 = backbone.get_intermediate_layers(x, n=[6])[0]   # Fine-grained
feat_layer12 = backbone.get_intermediate_layers(x, n=[12])[0] # Semantic

# Combine for different targets
feat_clover = concat(feat_layer6, feat_layer12)  # Needs fine details
feat_total = feat_layer12  # Semantic is enough
```

### Tiling Strategy
Current 2×2 grid may miss small clover patches. Consider:
- 3×3 grid for clover head only
- Multi-scale tiling: 2×2 + 4×4 averaged

---

## Strategy 3: Target-Specific Heads

Different targets may benefit from different head architectures.

### Larger heads for hard targets
```python
self.head_green = MLP(embed_dim, hidden_dim=256, layers=2)
self.head_dead = MLP(embed_dim, hidden_dim=512, layers=3)    # Larger
self.head_clover = MLP(embed_dim, hidden_dim=512, layers=3)  # Larger
```

### Attention-based heads for clover
```python
class CloverHead(nn.Module):
    def __init__(self, embed_dim):
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.mlp = MLP(embed_dim, hidden_dim=256)
    
    def forward(self, x):
        # x: (B, num_patches, embed_dim)
        attn_out, weights = self.attention(x, x, x)
        pooled = attn_out.mean(dim=1)
        return self.mlp(pooled)
```

---

## Strategy 4: Auxiliary Losses

Add extra training signals without changing evaluation metric.

### Consistency loss
```python
# Ensure predictions satisfy constraints
gdm_pred = green_pred + clover_pred
total_pred = gdm_pred + dead_pred

consistency_loss = (
    F.mse_loss(gdm_pred, gdm_target) +
    F.mse_loss(total_pred, total_target)
)

total_loss = main_loss + 0.1 * consistency_loss
```

### Ordinal loss
```python
# Total should always be >= GDM >= max(Green, Clover)
ordering_loss = F.relu(gdm_pred - total_pred) + F.relu(green_pred - gdm_pred)
```

---

## Strategy 5: Data Augmentation for Hard Targets

### Color-aware augmentation
Dead grass has brown/yellow hue, clover has distinct green.

```python
# Augment color channels differently
class ColorAwareAugment:
    def __call__(self, img):
        # Random hue shift in green channel (affects clover visibility)
        # Random saturation (affects dead/live distinction)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] += np.random.randint(-10, 10)  # Hue
        hsv[:,:,1] *= np.random.uniform(0.8, 1.2)  # Saturation
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
```

### Cutout/GridMask focused on challenging regions
```python
# Apply stronger augmentation to force model to learn from partial views
A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.5)
```

---

## Strategy 6: Ensemble Strategies

### Model diversity
1. **Backbone diversity**: Base vs Small DINOv2
2. **Grid diversity**: 2×2 vs 3×3 vs 4×4
3. **Training diversity**: Different folds, seeds, augmentations

### Target-specialized ensembles
```python
# Train models optimized for different targets
model_total = train(loss_weights=[0.05, 0.05, 0.05, 0.1, 0.75])  # Focus on total
model_components = train(loss_weights=[0.2, 0.2, 0.2, 0.2, 0.2])  # Balanced

# Ensemble predictions
total_final = 0.6 * model_total.total + 0.4 * model_components.total
clover_final = 0.3 * model_total.clover + 0.7 * model_components.clover
```

### Stacking
```python
# Train meta-model on OOF predictions
meta_features = [model1_oof, model2_oof, model3_oof]  # (N, 5, 3)
meta_model = Ridge().fit(meta_features, targets)
```

---

## Strategy 7: External Features & Signal Processing

### HSV Color Space Analysis
Dead grass is brown/yellow (hue 10-40°), clover is distinct green (hue 60-90°):

```python
import cv2
import numpy as np

def extract_hsv_features(img):
    """Extract HSV-based features for vegetation analysis."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    # Hue ranges (OpenCV: 0-180)
    green_mask = (h >= 30) & (h <= 45)      # Green vegetation
    brown_mask = (h >= 5) & (h <= 15)        # Dead/brown grass
    clover_mask = (h >= 35) & (h <= 50) & (s > 80)  # Clover (saturated green)
    
    features = {
        # Hue statistics
        'hue_mean': h.mean() / 180,
        'hue_std': h.std() / 180,
        
        # Saturation (live vegetation is more saturated)
        'sat_mean': s.mean() / 255,
        'sat_std': s.std() / 255,
        
        # Value/brightness
        'val_mean': v.mean() / 255,
        'val_std': v.std() / 255,
        
        # Vegetation ratios
        'green_ratio': green_mask.mean(),
        'brown_ratio': brown_mask.mean(),
        'clover_ratio': clover_mask.mean(),
        
        # Green vs brown ratio (indicator of dead proportion)
        'green_brown_ratio': (green_mask.sum() + 1) / (brown_mask.sum() + 1),
    }
    return features
```

### Hue Histogram Features
```python
def hue_histogram_features(img, bins=18):
    """Extract hue histogram as feature vector."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    
    # Histogram of hue values
    hist, _ = np.histogram(h, bins=bins, range=(0, 180))
    hist = hist / hist.sum()  # Normalize
    
    # Key bins for vegetation
    features = {
        'hue_hist': hist,  # Full histogram (18 features)
        'hue_peak': hist.argmax() * (180 / bins),  # Dominant hue
        'hue_entropy': -np.sum(hist * np.log(hist + 1e-8)),  # Color diversity
    }
    return features
```

### Fourier Transform for Texture Analysis
Clover has distinct leaf patterns, dead grass has uniform texture:

```python
def fourier_features(img, n_features=16):
    """Extract frequency domain features for texture analysis."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Log magnitude spectrum
    log_mag = np.log1p(magnitude)
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    
    # Radial frequency bands
    features = {}
    radii = np.linspace(0, min(cy, cx), n_features + 1)
    
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    for i in range(n_features):
        mask = (dist >= radii[i]) & (dist < radii[i+1])
        features[f'freq_band_{i}'] = log_mag[mask].mean() if mask.sum() > 0 else 0
    
    # Summary statistics
    features['freq_low'] = log_mag[dist < radii[2]].mean()   # Low freq (structure)
    features['freq_high'] = log_mag[dist > radii[-3]].mean() # High freq (texture)
    features['freq_ratio'] = features['freq_high'] / (features['freq_low'] + 1e-8)
    
    return features
```

### Gabor Filters for Texture
```python
def gabor_features(img, frequencies=[0.1, 0.2, 0.4], orientations=4):
    """Extract Gabor filter responses for texture analysis."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    features = {}
    for freq in frequencies:
        for theta_idx in range(orientations):
            theta = theta_idx * np.pi / orientations
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=3, theta=theta, 
                lambd=1/freq, gamma=0.5, psi=0
            )
            filtered = cv2.filter2D(gray, -1, kernel)
            
            key = f'gabor_f{freq:.1f}_t{theta_idx}'
            features[f'{key}_mean'] = filtered.mean()
            features[f'{key}_std'] = filtered.std()
            features[f'{key}_energy'] = (filtered ** 2).mean()
    
    return features
```

### Local Binary Patterns (LBP) for Texture
```python
from skimage.feature import local_binary_pattern

def lbp_features(img, radius=3, n_points=24):
    """Extract LBP texture features."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Histogram of LBP values
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist / hist.sum()
    
    return {
        'lbp_hist': hist,
        'lbp_uniformity': hist.max(),  # Peak indicates dominant pattern
        'lbp_entropy': -np.sum(hist * np.log(hist + 1e-8)),
    }
```

### Combined Feature Extractor
```python
class VegetationFeatureExtractor:
    """Extract all handcrafted features for vegetation analysis."""
    
    def __init__(self):
        self.feature_dim = 64  # Approximate total features
    
    def extract(self, img):
        """Extract all features from image."""
        features = {}
        
        # Color features
        features.update(extract_hsv_features(img))
        features.update(hue_histogram_features(img))
        
        # Texture features
        features.update(fourier_features(img))
        features.update(gabor_features(img))
        features.update(lbp_features(img))
        
        # Convert to vector
        vec = []
        for k, v in sorted(features.items()):
            if isinstance(v, np.ndarray):
                vec.extend(v.tolist())
            else:
                vec.append(v)
        
        return np.array(vec)
```

### Integration with DINOv2
```python
class TwoStreamDINOWithHandcrafted(TwoStreamDINOBase):
    """DINOv2 + handcrafted features hybrid model."""
    
    def __init__(self, backbone_name, handcrafted_dim=64, **kwargs):
        super().__init__(backbone_name, **kwargs)
        
        # Projection for handcrafted features
        self.handcrafted_proj = nn.Sequential(
            nn.Linear(handcrafted_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.feat_dim)
        )
        
        # Update combined dim
        self.combined = self.feat_dim * 2 + self.feat_dim  # left + right + handcrafted
        
        # Rebuild heads with new dimension
        hidden = max(8, int(self.combined * 0.25))
        self.head_green = self._make_head(hidden)
        self.head_clover = self._make_head(hidden)
        self.head_dead = self._make_head(hidden)
    
    def forward(self, x_left, x_right, handcrafted_features):
        f_l = self._encode_tiles(x_left)
        f_r = self._encode_tiles(x_right)
        f_hc = self.handcrafted_proj(handcrafted_features)
        
        f = torch.cat([f_l, f_r, f_hc], dim=1)
        
        green = self.softplus(self.head_green(f))
        clover = self.softplus(self.head_clover(f))
        dead = self.softplus(self.head_dead(f))
        
        gdm = green + clover
        total = gdm + dead
        
        return green, dead, clover, gdm, total
```

### Vegetation Indices (from RGB approximation)
```python
def rgb_vegetation_indices(img):
    """Approximate vegetation indices from RGB."""
    r, g, b = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
    
    # Excess Green Index (ExG)
    exg = 2*g - r - b
    
    # Excess Red Index (ExR) - indicates dead/stressed vegetation
    exr = 1.4*r - g
    
    # Excess Green minus Excess Red (ExGR)
    exgr = exg - exr
    
    # Green-Red Vegetation Index (GRVI)
    grvi = (g - r) / (g + r + 1e-8)
    
    # Visible Atmospherically Resistant Index (VARI)
    vari = (g - r) / (g + r - b + 1e-8)
    
    return {
        'exg_mean': exg.mean(),
        'exg_std': exg.std(),
        'exr_mean': exr.mean(),
        'exgr_mean': exgr.mean(),
        'grvi_mean': grvi.mean(),
        'vari_mean': np.clip(vari, -1, 1).mean(),
        'live_ratio': (exgr > 0).mean(),  # Proportion of live vegetation
    }
```

---

## Strategy 8: Post-Processing

### Constraint enforcement
```python
def enforce_constraints(preds):
    green, dead, clover, gdm, total = preds
    
    # Ensure non-negative
    green = max(0, green)
    dead = max(0, dead)
    clover = max(0, clover)
    
    # Ensure GDM >= Green and GDM >= Clover
    gdm = max(gdm, green, clover)
    
    # Ensure Total >= GDM + Dead
    total = max(total, gdm + dead)
    
    return [green, dead, clover, gdm, total]
```

### Distribution matching
```python
# Match prediction distribution to training distribution
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
preds_calibrated = qt.fit_transform(train_preds).transform(test_preds)
```

---

## Strategy 9: Loss Function Improvements

### Focal loss for hard samples
```python
class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0):
        self.gamma = gamma
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # Upweight hard samples (high error)
        focal_weight = mse.detach() ** self.gamma
        return (focal_weight * mse).mean()
```

### Quantile loss for robustness
```python
def quantile_loss(pred, target, quantile=0.5):
    error = target - pred
    return torch.max(quantile * error, (quantile - 1) * error).mean()
```

---

## Strategy 10: Target Domain Transformation

Transform targets to a space that's easier to learn, then inverse transform predictions.

### Log Transform (for skewed distributions)
```python
# Training: transform targets
y_log = np.log1p(y)  # log(1 + y) to handle zeros

# Inference: inverse transform
y_pred = np.expm1(y_log_pred)  # exp(y) - 1
```

### Box-Cox Transform (optimal power transform)
```python
from scipy.stats import boxcox, inv_boxcox

# Fit on training data
y_bc, lambda_ = boxcox(y + 1)  # +1 to handle zeros

# Inverse transform predictions
y_pred = inv_boxcox(y_bc_pred, lambda_) - 1
```

### Ratio/Proportion Transform
Since `Total = Green + Dead + Clover`, predict proportions:
```python
# Transform to proportions (simplex)
total = green + dead + clover
p_green = green / total
p_dead = dead / total  
p_clover = clover / total
# Note: p_green + p_dead + p_clover = 1

# Predict: total (absolute) + proportions
# Derive: green = total × p_green, etc.
```

### Isometric Log-Ratio (ILR) for Compositional Data
Better than raw proportions for compositional data:
```python
import numpy as np

def ilr_transform(proportions):
    """Transform proportions to unbounded space."""
    # proportions: [p_green, p_dead, p_clover] summing to 1
    p = np.array(proportions)
    
    # ILR coordinates (2D for 3 components)
    ilr1 = np.sqrt(1/2) * np.log(p[0] / p[1])
    ilr2 = np.sqrt(2/3) * np.log(np.sqrt(p[0] * p[1]) / p[2])
    return [ilr1, ilr2]

def ilr_inverse(ilr_coords, total):
    """Transform back to original space."""
    ilr1, ilr2 = ilr_coords
    
    # Inverse transform
    x1 = np.exp(ilr1 / np.sqrt(1/2))
    x2 = np.exp(ilr2 / np.sqrt(2/3))
    
    p_clover = 1 / (1 + x2 * np.sqrt(x1) + x2 / np.sqrt(x1))
    p_green = p_clover * x2 * np.sqrt(x1)
    p_dead = p_clover * x2 / np.sqrt(x1)
    
    return [total * p_green, total * p_dead, total * p_clover]
```

### Quantile Transform (Gaussian)
```python
from sklearn.preprocessing import QuantileTransformer

# Fit on training targets
qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
y_gaussian = qt.fit_transform(y_train)

# Inverse transform predictions
y_pred = qt.inverse_transform(y_gaussian_pred)
```

### Residual Transform (Predict deviations)
```python
# Instead of predicting raw values, predict deviations from mean
y_mean = y_train.mean(axis=0)
y_residual = y_train - y_mean

# Model predicts residuals
# Final prediction: y_pred = y_mean + residual_pred
```

### Recommended Approach: Hybrid Transform
```python
class TargetTransformer:
    def __init__(self):
        self.total_scaler = QuantileTransformer(output_distribution='normal')
    
    def fit_transform(self, green, dead, clover):
        total = green + dead + clover
        
        # Transform total to Gaussian
        total_t = self.total_scaler.fit_transform(total.reshape(-1, 1))
        
        # Transform components to proportions
        p_green = green / (total + 1e-8)
        p_dead = dead / (total + 1e-8)
        p_clover = clover / (total + 1e-8)
        
        # Logit transform proportions (unbounded)
        logit_green = np.log(p_green / (1 - p_green + 1e-8))
        logit_dead = np.log(p_dead / (1 - p_dead + 1e-8))
        
        return total_t, logit_green, logit_dead  # clover derived
    
    def inverse_transform(self, total_t, logit_green, logit_dead):
        total = self.total_scaler.inverse_transform(total_t)
        
        p_green = 1 / (1 + np.exp(-logit_green))
        p_dead = 1 / (1 + np.exp(-logit_dead))
        p_clover = 1 - p_green - p_dead
        p_clover = np.clip(p_clover, 0, 1)
        
        green = total * p_green
        dead = total * p_dead
        clover = total * p_clover
        
        return green, dead, clover
```

### Implementation in Dataset
```python
class BiomassDatasetTransformed(BiomassDataset):
    def __init__(self, df, image_dir, transform, transformer):
        super().__init__(df, image_dir, transform)
        self.transformer = transformer
        
        # Pre-transform all targets
        self.targets_transformed = self.transformer.fit_transform(
            df['Dry_Green_g'].values,
            df['Dry_Dead_g'].values,
            df['Dry_Clover_g'].values
        )
    
    def __getitem__(self, idx):
        img_left, img_right = self.load_images(idx)
        target = self.targets_transformed[idx]
        return img_left, img_right, target
```

---

## Strategy 11: Training Curriculum

### Stage 1: Easy targets first
```bash
# Train focusing on total/GDM (easier, high weight)
python -m src.train --epochs 20 ...
```

### Stage 2: Fine-tune on hard targets
```bash
# Continue training with balanced focus
python -m src.train --resume checkpoint.pth --epochs 30 ...
```

---

## Recommended Experiment Priority

### Tier 1: Quick Wins (hours)
- [ ] Try 3×3 grid instead of 2×2
- [ ] Ensemble base + small models
- [ ] Multi-scale TTA (0.9x, 1.0x, 1.1x)
- [ ] Mixup augmentation (alpha=0.4)
- [ ] Log transform on targets

### Tier 2: Low-Hanging Fruit (1-2 days)
- [ ] HSV color features (green/brown ratio)
- [ ] RGB vegetation indices (ExG, GRVI)
- [ ] Temporal features (month sin/cos encoding)
- [ ] Larger heads for clover/dead
- [ ] Proportion + total prediction

### Tier 3: Medium Effort (2-5 days)
- [ ] Cross-stream attention (exploit stereo)
- [ ] Metadata conditioning (state embeddings)
- [ ] MC Dropout uncertainty
- [ ] Hierarchical prediction
- [ ] CutMix augmentation

### Tier 4: Higher Effort (1+ week)
- [ ] Fourier/Gabor texture features
- [ ] DINOv2 + handcrafted fusion
- [ ] Mixture of Experts
- [ ] Self-supervised domain adaptation
- [ ] Transformer decoder for targets

### Most Promising (ranked)
1. **Stereo cross-attention** - unexploited signal
2. **Temporal/state conditioning** - metadata is free
3. **Multi-scale TTA** - easy inference boost
4. **Proportion transform** - better for compositional data
5. **Texture features** - key for clover detection

---

## Strategy 12: Exploit Stereo Vision

We have left/right image pairs but treat them independently. Exploit stereo geometry:

### Depth-Aware Features
```python
def stereo_features(left, right):
    """Extract stereo matching features."""
    # Simple block matching for disparity
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    gray_l = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    
    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    
    # Disparity correlates with height/depth
    # Taller vegetation (clover) has different disparity pattern
    return {
        'disp_mean': disparity.mean(),
        'disp_std': disparity.std(),
        'disp_range': disparity.max() - disparity.min(),
    }
```

### Cross-Stream Attention
```python
class CrossStreamAttention(nn.Module):
    """Attention between left and right streams."""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8)
    
    def forward(self, f_left, f_right):
        # f_left, f_right: (B, num_patches, D)
        # Query from left, key/value from right
        f_left_enhanced, _ = self.cross_attn(f_left, f_right, f_right)
        f_right_enhanced, _ = self.cross_attn(f_right, f_left, f_left)
        return f_left_enhanced, f_right_enhanced
```

### Stereo Consistency Loss
```python
# Features from left and right should be similar (same scene)
consistency_loss = F.mse_loss(f_left.mean(1), f_right.mean(1))
```

---

## Strategy 13: Temporal & Metadata Features

### Sampling Date Features
```python
def temporal_features(sampling_date):
    """Extract temporal features from sampling date."""
    date = pd.to_datetime(sampling_date)
    
    # Cyclical encoding for seasonality
    day_of_year = date.dayofyear
    features = {
        'day_sin': np.sin(2 * np.pi * day_of_year / 365),
        'day_cos': np.cos(2 * np.pi * day_of_year / 365),
        'month': date.month / 12,
        'is_growing_season': int(date.month in [9, 10, 11, 12, 1, 2, 3]),  # Southern hemisphere
    }
    return features
```

### State/Location Embeddings
```python
class StateEmbedding(nn.Module):
    """Learnable state embeddings."""
    
    def __init__(self, num_states, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_states, embed_dim)
    
    def forward(self, state_idx):
        return self.embedding(state_idx)
```

### Metadata-Conditioned Model
```python
class MetadataConditionedModel(TwoStreamDINOBase):
    def __init__(self, backbone_name, num_states=8, **kwargs):
        super().__init__(backbone_name, **kwargs)
        
        # Metadata embeddings
        self.state_embed = nn.Embedding(num_states, 64)
        self.temporal_proj = nn.Linear(4, 64)  # 4 temporal features
        
        # FiLM conditioning from metadata
        self.meta_film = nn.Linear(128, self.combined * 2)
    
    def forward(self, x_left, x_right, state_idx, temporal_feats):
        # Get visual features
        f = self.get_visual_features(x_left, x_right)
        
        # Get metadata features
        meta = torch.cat([
            self.state_embed(state_idx),
            self.temporal_proj(temporal_feats)
        ], dim=1)
        
        # FiLM modulation
        gamma, beta = self.meta_film(meta).chunk(2, dim=1)
        f = f * (1 + gamma) + beta
        
        return self.predict_heads(f)
```

---

## Strategy 14: Uncertainty Quantification

### MC Dropout for Uncertainty
```python
def predict_with_uncertainty(model, x_left, x_right, n_samples=20):
    """Monte Carlo dropout for uncertainty estimation."""
    model.train()  # Keep dropout active
    
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(x_left, x_right)
        predictions.append(pred)
    
    predictions = torch.stack(predictions)
    
    mean_pred = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)  # Epistemic uncertainty
    
    return mean_pred, uncertainty
```

### Deep Ensembles
```python
# Train N models with different seeds
# Ensemble prediction = mean of all models
# Uncertainty = std across models
```

### Evidential Regression
```python
class EvidentialHead(nn.Module):
    """Predict distribution parameters instead of point estimates."""
    
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 4)  # mu, v, alpha, beta
    
    def forward(self, x):
        out = self.fc(x)
        mu = out[:, 0]
        v = F.softplus(out[:, 1])  # > 0
        alpha = F.softplus(out[:, 2]) + 1  # > 1
        beta = F.softplus(out[:, 3])  # > 0
        return mu, v, alpha, beta
    
    def nll_loss(self, mu, v, alpha, beta, y):
        """Negative log-likelihood for evidential regression."""
        omega = 2 * beta * (1 + v)
        nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(omega) \
            + (alpha + 0.5) * torch.log(v * (y - mu)**2 + omega) \
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        return nll.mean()
```

---

## Strategy 15: Advanced Data Augmentation

### Mixup for Regression
```python
def mixup_data(x_left, x_right, y, alpha=0.4):
    """Mixup augmentation for regression."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x_left.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x_left = lam * x_left + (1 - lam) * x_left[index]
    mixed_x_right = lam * x_right + (1 - lam) * x_right[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x_left, mixed_x_right, mixed_y
```

### CutMix for Vegetation
```python
def cutmix_vegetation(x1, x2, y1, y2, alpha=1.0):
    """CutMix with area-proportional target mixing."""
    lam = np.random.beta(alpha, alpha)
    
    # Random box
    H, W = x1.shape[2:]
    cut_h = int(H * np.sqrt(1 - lam))
    cut_w = int(W * np.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1_box = max(0, cy - cut_h // 2)
    y2_box = min(H, cy + cut_h // 2)
    x1_box = max(0, cx - cut_w // 2)
    x2_box = min(W, cx + cut_w // 2)
    
    # Apply cutmix
    x1[:, :, y1_box:y2_box, x1_box:x2_box] = x2[:, :, y1_box:y2_box, x1_box:x2_box]
    
    # Adjust lambda based on actual area
    lam = 1 - (y2_box - y1_box) * (x2_box - x1_box) / (H * W)
    y_mixed = lam * y1 + (1 - lam) * y2
    
    return x1, y_mixed
```

### Grid Shuffle Augmentation
```python
def grid_shuffle(img, grid_size=4):
    """Shuffle grid tiles to break spatial bias."""
    H, W = img.shape[:2]
    tile_h, tile_w = H // grid_size, W // grid_size
    
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            tile = img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tiles.append(tile)
    
    np.random.shuffle(tiles)
    
    shuffled = np.zeros_like(img)
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            shuffled[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = tiles[idx]
            idx += 1
    
    return shuffled
```

---

## Strategy 16: Test-Time Adaptation (TTA)

### Multi-Scale TTA
```python
def tta_multiscale(model, x_left, x_right, scales=[0.9, 1.0, 1.1]):
    """TTA with multiple scales."""
    predictions = []
    
    for scale in scales:
        # Resize
        H, W = x_left.shape[2:]
        new_H, new_W = int(H * scale), int(W * scale)
        x_l = F.interpolate(x_left, (new_H, new_W), mode='bilinear')
        x_r = F.interpolate(x_right, (new_H, new_W), mode='bilinear')
        
        pred = model(x_l, x_r)
        predictions.append(pred)
    
    return torch.stack(predictions).mean(dim=0)
```

### TTA with Color Augmentation
```python
def tta_color(model, x_left, x_right):
    """TTA with color augmentations."""
    predictions = [model(x_left, x_right)]  # Original
    
    # Brightness variations
    for factor in [0.9, 1.1]:
        x_l = torch.clamp(x_left * factor, 0, 1)
        x_r = torch.clamp(x_right * factor, 0, 1)
        predictions.append(model(x_l, x_r))
    
    return torch.stack(predictions).mean(dim=0)
```

---

## Strategy 17: Self-Supervised Domain Adaptation

### DINOv2 Feature Distribution Alignment
```python
class DomainAlignmentLoss(nn.Module):
    """Align feature distributions between train and test."""
    
    def forward(self, train_features, test_features):
        # MMD (Maximum Mean Discrepancy)
        train_mean = train_features.mean(dim=0)
        test_mean = test_features.mean(dim=0)
        return F.mse_loss(train_mean, test_mean)
```

### Pseudo-Labeling
```python
def pseudo_label_training(model, labeled_loader, unlabeled_loader, threshold=0.9):
    """Self-training with pseudo-labels."""
    model.eval()
    
    # Generate pseudo-labels for high-confidence predictions
    pseudo_data = []
    for x_l, x_r in unlabeled_loader:
        pred, uncertainty = predict_with_uncertainty(model, x_l, x_r)
        
        # Only use low-uncertainty samples
        confident_mask = uncertainty.mean(dim=1) < threshold
        if confident_mask.any():
            pseudo_data.append((x_l[confident_mask], x_r[confident_mask], pred[confident_mask]))
    
    # Train on labeled + pseudo-labeled data
    ...
```

---

## Strategy 18: Physical Constraints

### Vegetation Growth Priors
```python
# Biomass relationships
# Total = Green + Dead + Clover  (exact)
# GDM = Green + Clover           (exact)
# Typically: Total > GDM > Green > Clover
# Dead is often correlated with season (more in dry periods)

class PhysicsConstrainedLoss(nn.Module):
    def forward(self, pred, target):
        green, dead, clover, gdm, total = pred.unbind(dim=1)
        
        # Main regression loss
        main_loss = F.mse_loss(pred, target)
        
        # Physics constraints as soft penalties
        ordering_loss = (
            F.relu(clover - green) +           # Usually green > clover
            F.relu(green - total) +            # Total > green
            F.relu(-dead) + F.relu(-clover)    # Non-negative
        ).mean()
        
        return main_loss + 0.1 * ordering_loss
```

### Seasonal Priors
```python
# In Southern Hemisphere (Australia):
# - Spring (Sep-Nov): High green, low dead
# - Summer (Dec-Feb): Moderate green, increasing dead
# - Autumn (Mar-May): Decreasing green, high dead
# - Winter (Jun-Aug): Low growth overall

def seasonal_prior(month):
    """Expected relative proportions by season."""
    if month in [9, 10, 11]:  # Spring
        return {'green_ratio': 0.6, 'dead_ratio': 0.2}
    elif month in [12, 1, 2]:  # Summer
        return {'green_ratio': 0.4, 'dead_ratio': 0.4}
    elif month in [3, 4, 5]:  # Autumn
        return {'green_ratio': 0.3, 'dead_ratio': 0.5}
    else:  # Winter
        return {'green_ratio': 0.3, 'dead_ratio': 0.3}
```

---

## Strategy 19: Model Architectures

### Transformer Decoder for Targets
```python
class TargetTransformerDecoder(nn.Module):
    """Decode targets autoregressively."""
    
    def __init__(self, embed_dim, num_targets=5):
        super().__init__()
        self.target_embeddings = nn.Embedding(num_targets, embed_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, nheads=8),
            num_layers=2
        )
        self.head = nn.Linear(embed_dim, 1)
    
    def forward(self, visual_features):
        # visual_features: (B, D)
        # Predict targets one by one, conditioning on previous
        B = visual_features.size(0)
        memory = visual_features.unsqueeze(0)  # (1, B, D)
        
        outputs = []
        for i in range(5):
            tgt = self.target_embeddings.weight[i:i+1].expand(B, -1).unsqueeze(0)
            decoded = self.decoder(tgt, memory)
            out = self.head(decoded.squeeze(0))
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)
```

### Mixture of Experts
```python
class MixtureOfExperts(nn.Module):
    """Different experts for different vegetation conditions."""
    
    def __init__(self, embed_dim, num_experts=4):
        super().__init__()
        self.gate = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Linear(embed_dim, 5) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        gate_weights = F.softmax(self.gate(x), dim=1)  # (B, num_experts)
        
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, 5)
        
        # Weighted sum of expert predictions
        output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return output
```

---

## Strategy 20: Debugging & Analysis

### Feature Visualization
```python
def visualize_attention(model, x_left, x_right):
    """Visualize what the model attends to."""
    # Get attention weights from DINOv2
    features = model.backbone.get_intermediate_layers(x_left, n=[11])[0]
    
    # Reshape to spatial
    B, N, D = features.shape
    H = W = int(np.sqrt(N - 1))  # -1 for CLS token
    
    spatial_features = features[:, 1:].reshape(B, H, W, D)
    
    # Compute importance per location
    importance = spatial_features.norm(dim=-1)
    
    return importance  # Visualize as heatmap
```

### Per-Sample Error Analysis
```python
def error_analysis(model, loader, df):
    """Analyze which samples have high error."""
    errors = []
    
    for (x_l, x_r, y), idx in zip(loader, range(len(df))):
        pred = model(x_l, x_r)
        error = (pred - y).abs().mean().item()
        
        errors.append({
            'idx': idx,
            'error': error,
            'state': df.iloc[idx]['State'],
            'month': df.iloc[idx]['Sampling_Date'].month,
            'target_total': y[:, -1].item(),
        })
    
    # Find patterns in high-error samples
    error_df = pd.DataFrame(errors)
    print("High error by state:", error_df.groupby('state')['error'].mean())
    print("High error by month:", error_df.groupby('month')['error'].mean())
```

---

## Strategy 21: Constrained 5-Target Model with Selective Post-Processing

### Concept
1. **Predict all 5 targets independently** (more model capacity)
2. **Apply soft constraints during training** (consistency regularization)
3. **Post-process at inference** - fix bad predictions using "good" ones

### Model: 5 Independent Heads
```python
class FiveHeadModel(TwoStreamDINOBase):
    """Predict all 5 targets with independent heads."""
    
    def __init__(self, backbone_name, **kwargs):
        super().__init__(backbone_name, **kwargs)
        
        # 5 independent heads (not 3 + derived)
        self.head_green = self._make_head()
        self.head_dead = self._make_head()
        self.head_clover = self._make_head()
        self.head_gdm = self._make_head()
        self.head_total = self._make_head()
    
    def forward(self, x_left, x_right):
        f = self.get_combined_features(x_left, x_right)
        
        green = self.softplus(self.head_green(f))
        dead = self.softplus(self.head_dead(f))
        clover = self.softplus(self.head_clover(f))
        gdm = self.softplus(self.head_gdm(f))
        total = self.softplus(self.head_total(f))
        
        return green, dead, clover, gdm, total
```

### Training with Soft Constraints
```python
class ConstrainedMSELoss(nn.Module):
    """MSE loss with soft consistency constraints."""
    
    def __init__(self, target_weights, constraint_weight=0.1):
        super().__init__()
        self.weights = torch.tensor(target_weights)
        self.constraint_weight = constraint_weight
    
    def forward(self, pred, target):
        green, dead, clover, gdm, total = pred.unbind(dim=1)
        
        # Main weighted MSE loss
        main_loss = ((pred - target) ** 2 * self.weights).sum() / self.weights.sum()
        
        # Soft constraint losses (don't enforce exactly, just encourage)
        gdm_constraint = F.mse_loss(gdm, green + clover)
        total_constraint = F.mse_loss(total, green + dead + clover)
        
        # Ordering constraints (soft)
        ordering_loss = (
            F.relu(clover - gdm) +      # gdm >= clover
            F.relu(green - gdm) +       # gdm >= green
            F.relu(gdm - total) +       # total >= gdm
            F.relu(dead - total)        # total >= dead
        ).mean()
        
        constraint_loss = gdm_constraint + total_constraint + ordering_loss
        
        return main_loss + self.constraint_weight * constraint_loss
```

### Post-Processing: Selective Correction

```python
class SelectivePostProcessor:
    """
    Post-process predictions based on confidence and consistency.
    
    Strategy:
    - Trust total and GDM predictions (higher weight, more stable)
    - Correct green/dead/clover if they're inconsistent
    """
    
    def __init__(self, correction_threshold=0.15):
        self.threshold = correction_threshold
    
    def __call__(self, green, dead, clover, gdm, total):
        """
        Args:
            All predictions as numpy arrays (N,) or scalars
        
        Returns:
            Corrected predictions
        """
        # Check consistency
        gdm_error = np.abs(gdm - (green + clover)) / (gdm + 1e-8)
        total_error = np.abs(total - (green + dead + clover)) / (total + 1e-8)
        
        # If predictions are consistent, return as-is
        if gdm_error.mean() < self.threshold and total_error.mean() < self.threshold:
            return green, dead, clover, gdm, total
        
        # Otherwise, correct using trusted predictions (total, gdm)
        corrected = self.correct_components(green, dead, clover, gdm, total)
        return corrected
    
    def correct_components(self, green, dead, clover, gdm, total):
        """Correct components using total and GDM as anchors."""
        
        # Method 1: Scale to match constraints
        # Derive dead from total - gdm (trust total and gdm most)
        dead_corrected = np.maximum(0, total - gdm)
        
        # Scale green and clover to match gdm
        green_clover_sum = green + clover + 1e-8
        green_ratio = green / green_clover_sum
        clover_ratio = clover / green_clover_sum
        
        green_corrected = gdm * green_ratio
        clover_corrected = gdm * clover_ratio
        
        return green_corrected, dead_corrected, clover_corrected, gdm, total
    
    def correct_outliers(self, green, dead, clover, gdm, total, 
                         train_stats, n_std=3):
        """Clip outliers based on training distribution."""
        
        def clip_to_range(x, mean, std):
            lower = max(0, mean - n_std * std)
            upper = mean + n_std * std
            return np.clip(x, lower, upper)
        
        green = clip_to_range(green, train_stats['green_mean'], train_stats['green_std'])
        dead = clip_to_range(dead, train_stats['dead_mean'], train_stats['dead_std'])
        clover = clip_to_range(clover, train_stats['clover_mean'], train_stats['clover_std'])
        
        return green, dead, clover, gdm, total
```

### Advanced: Learn Post-Processing Corrections
```python
class LearnedPostProcessor(nn.Module):
    """Neural network to correct predictions."""
    
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)
        )
    
    def forward(self, predictions):
        """
        predictions: (B, 5) - green, dead, clover, gdm, total
        returns: (B, 5) - corrected predictions
        """
        # Residual learning: predict corrections
        corrections = self.net(predictions)
        corrected = predictions + corrections
        
        # Ensure non-negative
        corrected = F.relu(corrected)
        
        return corrected

# Train on OOF predictions vs ground truth
def train_postprocessor(oof_preds, oof_targets):
    model = LearnedPostProcessor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        corrected = model(oof_preds)
        loss = F.mse_loss(corrected, oof_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model
```

### Hybrid: Model Selection per Target
```python
class HybridEnsemble:
    """
    Use different model predictions for different targets.
    
    - Use 5-head model for total/GDM (independent prediction)
    - Use 3-head+derive model for green/clover/dead (consistent)
    """
    
    def __init__(self, model_5head, model_3head):
        self.model_5head = model_5head
        self.model_3head = model_3head
    
    def predict(self, x_left, x_right):
        # Get predictions from both models
        pred_5head = self.model_5head(x_left, x_right)
        pred_3head = self.model_3head(x_left, x_right)
        
        # Use 5-head for total/gdm (higher weights, trained independently)
        # Use 3-head for components (mathematically consistent)
        green = pred_3head[0]
        dead = pred_3head[1]
        clover = pred_3head[2]
        gdm = pred_5head[3]    # From 5-head
        total = pred_5head[4]  # From 5-head
        
        return green, dead, clover, gdm, total
```

### Confidence-Based Correction
```python
class ConfidenceBasedPostProcessor:
    """Use model uncertainty to decide correction strategy."""
    
    def __init__(self, models):
        self.models = models  # Ensemble of models
    
    def predict_with_confidence(self, x_left, x_right):
        predictions = []
        for model in self.models:
            pred = model(x_left, x_right)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)  # Uncertainty per target
        
        return mean_pred, std_pred
    
    def correct_uncertain(self, mean_pred, std_pred, threshold=0.2):
        """Correct targets with high uncertainty using constraints."""
        green, dead, clover, gdm, total = mean_pred.unbind(dim=1)
        green_std, dead_std, clover_std, gdm_std, total_std = std_pred.unbind(dim=1)
        
        # Normalize uncertainties
        mean_val = mean_pred.mean(dim=1, keepdim=True)
        rel_std = std_pred / (mean_val + 1e-8)
        
        # If clover is uncertain but total is confident, derive clover
        clover_uncertain = rel_std[:, 2] > threshold
        total_confident = rel_std[:, 4] < threshold
        gdm_confident = rel_std[:, 3] < threshold
        
        # Correct uncertain clover: clover = gdm - green
        mask = clover_uncertain & gdm_confident
        if mask.any():
            clover[mask] = gdm[mask] - green[mask]
            clover = F.relu(clover)  # Non-negative
        
        # Similarly for dead: dead = total - gdm
        dead_uncertain = rel_std[:, 1] > threshold
        mask = dead_uncertain & total_confident & gdm_confident
        if mask.any():
            dead[mask] = total[mask] - gdm[mask]
            dead = F.relu(dead)
        
        return torch.stack([green, dead, clover, gdm, total], dim=1)
```

### Implementation in Inference Pipeline
```python
def inference_with_postprocessing(model, loader, postprocessor):
    """Full inference pipeline with post-processing."""
    
    all_preds = []
    
    for x_left, x_right in loader:
        # Get raw predictions
        with torch.no_grad():
            raw_pred = model(x_left, x_right)
        
        # Stack predictions
        raw_pred = torch.stack(raw_pred, dim=1)  # (B, 5)
        
        # Apply post-processing
        corrected = postprocessor(raw_pred)
        
        all_preds.append(corrected.cpu().numpy())
    
    return np.concatenate(all_preds, axis=0)
```

### Training Recipe
```bash
# Step 1: Train 5-head model with constraints
python -m src.train --model-type five_head \
    --constraint-weight 0.1 \
    --output-dir outputs/five_head

# Step 2: Evaluate OOF predictions
python -m src.evaluate_oof --model-dir outputs/five_head

# Step 3: Train post-processor on OOF errors
python -m src.train_postprocessor \
    --oof-preds outputs/five_head/oof_preds.npy \
    --oof-targets outputs/five_head/oof_targets.npy

# Step 4: Inference with post-processing
python -m src.inference \
    --model-dir outputs/five_head \
    --postprocessor outputs/postprocessor.pth \
    --output submission.csv
```

---

## Tracking Experiments

Use consistent naming:
```
outputs/
├── baseline_2x2_base/
├── grid_3x3_base/
├── hierarchical_base/
├── large_heads_base/
└── ensemble_final/
```

Track per-target R² to measure improvements on clover/dead specifically.

