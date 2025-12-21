# Depth Integration for Biomass Prediction

## Overview

This document outlines how depth estimation (Depth Anything V2) can be integrated into the biomass prediction model to improve accuracy.

**Key Finding:** Correlation analysis (`scripts/check_depth_usefulness.py`) shows strong correlations between depth features and biomass targets:

| Target | Best Depth Stat | Correlation | Significance |
|--------|-----------------|-------------|--------------|
| **Green** | depth_gradient | **r=+0.631** | *** |
| **GDM** | depth_gradient | **r=+0.562** | *** |
| **Total** | depth_gradient | **r=+0.514** | *** |
| Clover | depth_mean | r=+0.402 | *** |
| Dead | depth_p90 | r=-0.143 | * |

---

## Why Depth Helps Biomass Prediction

### Physical Intuition

```
Side View of Vegetation:
                    ↑ HEIGHT (depth)
                    │
        ████████    │  Tall grass = MORE biomass
       ██████████   │
      ████████████  │
     ██████████████ │
    ████████████████│  Short grass = LESS biomass
════════════════════╧════ Ground
```

- **Taller vegetation** → closer to camera → larger depth values → more biomass
- **Depth gradient** captures vegetation boundaries and density variation
- **Stereo L-R difference** provides 3D volume information

### State-Specific Correlations

| State | Total Corr | Green Corr | Best Feature | Notes |
|-------|------------|------------|--------------|-------|
| **WA** | r=+0.657 | r=+0.871 | depth_gradient | Best performance |
| **Vic** | r=+0.484 | r=+0.446 | depth_gradient/mean | Good |
| **Tas** | r=+0.342 | r=+0.427 | depth_gradient | Moderate |
| **NSW** | r=-0.379 | r=-0.419 | depth_high_ratio | Inverted (?) |

---

## Current Implementation (v1)

### Method: Depth Statistics as Features

**File:** `src/dinov3_models.py` → `DepthFeatures` class

```python
# Usage
python -m src.dinov3_train --use-depth --epochs 30
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Stereo Images                      │
│                  x_left, x_right (B, 3, H, W)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐
│   DINOv3 Backbone   │   │  Depth Anything V2  │
│   (768-dim feat)    │   │     (frozen)        │
└─────────┬───────────┘   └─────────┬───────────┘
          │                         │
          │               ┌─────────▼─────────┐
          │               │  Depth Statistics │
          │               │  - gradient ⭐     │
          │               │  - mean, range    │
          │               │  - volume         │
          │               │  - lr_diff        │
          │               │  → 32-dim         │
          │               └─────────┬─────────┘
          │                         │
          └────────────┬────────────┘
                       ▼
             ┌─────────────────┐
             │  Concat + MLP   │
             │  768*2 + 32     │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ Prediction Heads│
             │ G, D, C, GDM, T │
             └─────────────────┘
```

### Depth Statistics Extracted

| Statistic | Description | Biomass Relevance |
|-----------|-------------|-------------------|
| `depth_mean` | Average height | Total biomass proxy |
| `depth_std` | Height variation | Uniformity (grass vs clover) |
| `depth_min` | Lowest point | Ground level |
| `depth_max` | Highest point | Tallest vegetation |
| `depth_range` | max - min | Height spread |
| `depth_p10` | 10th percentile | Low vegetation |
| `depth_p90` | 90th percentile | High vegetation |
| `depth_gradient` | Edge magnitude | **Vegetation boundaries** ⭐ |
| `depth_volume` | Sum above min | **Volume proxy** |
| `depth_high_ratio` | % high pixels | Dense tall areas |
| `depth_lr_diff` | L-R difference | **Stereo disparity** |
| `depth_lr_corr` | L-R correlation | Stereo consistency |

### Code

```python
class DepthFeatures(nn.Module):
    """Extract depth-based features using Depth Anything V2."""
    
    def __init__(self, out_dim: int = 32, model_size: str = "small"):
        # Lazy-load frozen DA2 model
        # Project 22 statistics → out_dim features
        
    def forward(self, x_left, x_right):
        depth_left = self._get_depth_map(x_left)
        depth_right = self._get_depth_map(x_right)
        
        stats_left = self._compute_stats(depth_left)    # 10 stats
        stats_right = self._compute_stats(depth_right)  # 10 stats
        lr_diff, lr_corr = ...  # 2 stereo stats
        
        return self.proj(concat([stats_left, stats_right, lr_diff, lr_corr]))
```

---

## Alternative Integration Methods

### Method 2: Depth as 4th Input Channel

Concatenate depth map with RGB before backbone.

```python
class DepthAsInput(nn.Module):
    def __init__(self):
        self.depth_proj = nn.Conv2d(1, 3, kernel_size=1)
        
    def forward(self, rgb, depth):
        depth_rgb = self.depth_proj(depth)
        x = rgb + depth_rgb * 0.3  # Blend depth into RGB
        return self.backbone(x)
```

**Pros:** Simple, end-to-end learnable  
**Cons:** Loses explicit depth information, may not help much

---

### Method 3: Depth-Guided Attention ⭐ Recommended

Use depth to weight which spatial regions matter more.

```python
class DepthGuidedAttention(nn.Module):
    """Weight features by depth - taller vegetation gets more attention."""
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.depth_to_weight = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, depth):
        # Normalize depth
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        
        # Learn attention from depth
        attn_weights = self.depth_to_weight(depth_norm)
        
        # Apply spatial attention
        return features * attn_weights
```

**Intuition:** Tall vegetation (high depth) should contribute more to biomass prediction.

**Integration:**
```python
# In DINOv3Direct.forward():
if self.use_depth_attention:
    depth_map = self.depth_module._get_depth_map(x_left)
    tiles_left = self.depth_attention(tiles_left, depth_map)
```

---

### Method 4: Multi-Task Learning (Auxiliary Depth)

Predict depth as auxiliary task to regularize feature learning.

```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        self.backbone = DINOv3()
        self.biomass_head = ...
        self.depth_head = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        biomass = self.biomass_head(features.mean((-2, -1)))
        depth_pred = self.depth_head(features)
        return biomass, depth_pred

# Training:
loss = biomass_loss + 0.1 * depth_reconstruction_loss
```

**Pros:** Forces backbone to learn 3D structure  
**Cons:** Needs ground truth depth (use DA2 pseudo-labels)

---

### Method 5: Depth-Weighted Loss

Weight loss by depth - focus on tall vegetation.

```python
def depth_weighted_loss(pred, target, depth_stats):
    """Errors on tall vegetation matter more."""
    # Use depth_mean as weight proxy
    weights = 0.5 + 0.5 * torch.sigmoid(depth_stats[:, 0])  # [0.5, 1.0]
    
    loss = ((pred - target) ** 2 * weights.unsqueeze(1)).mean()
    return loss
```

**Intuition:** Tall, dense vegetation is harder to predict → weight more.

---

### Method 6: Dual-Encoder Fusion

Separate encoders for RGB and depth, then fuse.

```python
class DualEncoder(nn.Module):
    def __init__(self):
        self.rgb_encoder = DINOv3()  # Heavy
        self.depth_encoder = nn.Sequential(  # Light
            nn.Conv2d(1, 64, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
        )
        self.fusion = nn.Linear(768 + 128, 768)
    
    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)
        return self.fusion(torch.cat([rgb_feat, depth_feat], dim=1))
```

---

### Method 7: Height-Based Segmentation

Predict biomass for different height layers, then sum.

```python
def height_layered_prediction(features, depth, heads, n_layers=3):
    """Predict per height layer, sum for total."""
    B = depth.size(0)
    
    # Create height bins using quantiles
    thresholds = [0.0, 0.33, 0.67, 1.0]  # Low, mid, high
    
    predictions = []
    for i in range(n_layers):
        lo = torch.quantile(depth.flatten(1), thresholds[i], dim=1)
        hi = torch.quantile(depth.flatten(1), thresholds[i+1], dim=1)
        
        mask = (depth >= lo.view(B,1,1)) & (depth < hi.view(B,1,1))
        layer_feat = (features * mask.unsqueeze(1)).sum((2,3)) / mask.sum((1,2), keepdim=True)
        predictions.append(heads[i](layer_feat))
    
    return sum(predictions)
```

**Intuition:** Different height layers have different green/dead compositions.

---

## Comparison of Methods

| Method | Complexity | Expected Gain | Training Cost | Notes |
|--------|------------|---------------|---------------|-------|
| **Stats (current)** | Low | +2-5% | +30% slower | Baseline |
| Depth as Input | Low | +0-2% | Same | May not help |
| **Depth Attention** | Medium | +3-6% | +20% slower | Recommended |
| Multi-Task | Medium | +2-4% | +50% slower | Regularization |
| Depth-Weighted Loss | Low | +1-3% | Same | Easy add-on |
| Dual-Encoder | High | +3-5% | +40% slower | More params |
| Height Layers | High | +2-4% | +30% slower | Complex |

---

## Usage

### Current Implementation

```bash
# Train with depth STATS features (adds 22-dim vector)
python -m src.dinov3_train --use-depth --epochs 30

# Train with depth-guided ATTENTION (weights tiles by depth)
python -m src.dinov3_train --depth-attention --epochs 30

# Combine both approaches
python -m src.dinov3_train --use-depth --depth-attention --epochs 30

# With larger depth model (better but slower)
python -m src.dinov3_train --depth-attention --depth-model-size base --epochs 30

# Ablation: No attention pooling at all
python -m src.dinov3_train --no-attention-pool --epochs 30
```

### Inference

```python
# In inference notebook, model automatically uses depth if trained with it
model = DINOv3Direct(use_depth=True)
model.load_state_dict(torch.load("model.pth"))

# Forward pass computes depth internally
green, dead, clover, gdm, total, _ = model(x_left, x_right)
```

---

## Analysis Script

Run correlation analysis:

```bash
# Analyze depth-biomass correlations
python scripts/check_depth_usefulness.py --model da2 --all

# Output:
# - outputs/depth_analysis/depth_stats.csv
# - outputs/depth_analysis/correlations.csv
# - outputs/depth_analysis/correlations_by_state.csv
# - outputs/depth_analysis/correlation_heatmap.png
```

---

## Future Work

1. **Depth Attention** - Implement Method 3 for spatial weighting
2. **Multi-Task** - Add depth reconstruction as auxiliary loss
3. **State-Specific** - Different depth weights per state (NSW behaves differently)
4. **Temporal** - Depth patterns may vary by month/season
5. **DA3 Integration** - When HuggingFace support improves

---

## References

- [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- [Depth Anything 3](https://huggingface.co/collections/depth-anything/depth-anything-3)
- Correlation analysis: `scripts/check_depth_usefulness.py`
- Model implementation: `src/dinov3_models.py` → `DepthFeatures`

