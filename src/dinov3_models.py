"""
DINOv3 Direct Model for Biomass Prediction.

Simple, focused architecture:
- DINOv3 backbone (vit_base_patch16_dinov3)
- Predict Total, Green, GDM directly
- Derive Dead = Total - GDM, Clover = GDM - Green
- FiLM for stereo fusion, attention pooling for tiles
"""
import os
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Backbone
# =============================================================================

def build_dinov3_backbone(
    pretrained: bool = True,
    gradient_checkpointing: bool = False,
    backbone_size: str = "base",
    backbone_type: str = "dinov3",
    ckpt_path: Optional[str] = None,
) -> Tuple[nn.Module, int, int]:
    """
    Build DINOv2/DINOv3 backbone.
    
    Args:
        pretrained: Use pretrained weights
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        backbone_size: Model size - "small", "base", or "large"
        backbone_type: "dinov3" (default) or "dinov2" (for PlantCLEF weights)
        ckpt_path: Path to custom weights (.safetensors for PlantCLEF)
    
    Returns:
        backbone: DINOv2/DINOv3 model
        feat_dim: Feature dimension (384 for S, 768 for B, 1024 for L)
        input_res: Native input resolution
    """
    if backbone_type == "dinov2":
        # DINOv2 with reg4 tokens (for PlantCLEF pretrained weights)
        model_names = {
            "small": "vit_small_patch14_reg4_dinov2.lvd142m",
            "base": "vit_base_patch14_reg4_dinov2.lvd142m",
            "large": "vit_large_patch14_reg4_dinov2.lvd142m",
        }
        name = model_names.get(backbone_size, model_names["base"])
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        input_res = 518  # DINOv2 native resolution (patch14 * 37)
        
        # Load PlantCLEF weights if provided
        if ckpt_path and os.path.exists(ckpt_path):
            if ckpt_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file as load_safetensors
                    print(f"Loading PlantCLEF weights from: {ckpt_path}")
                    state_dict = load_safetensors(ckpt_path)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    print(f"  Loaded {len(state_dict) - len(unexpected)} weights, "
                          f"missing: {len(missing)}, unexpected: {len(unexpected)}")
                except ImportError:
                    print("Warning: safetensors not installed, using ImageNet weights")
            else:
                # Regular PyTorch checkpoint
                print(f"Loading checkpoint from: {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location="cpu")
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=False)
    else:
        # DINOv3 (default)
        model_names = {
            "small": "vit_small_patch16_dinov3",
            "base": "vit_base_patch16_dinov3",
            "large": "vit_large_patch16_dinov3",
        }
        name = model_names.get(backbone_size, model_names["base"])
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        input_res = 256  # DINOv3 default
    
    feat_dim = model.num_features  # 384 for S, 768 for B, 1024 for L
    
    if gradient_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)
    
    return model, feat_dim, input_res


# =============================================================================
# Vegetation Indices (Domain Knowledge Features)
# =============================================================================

def compute_rgb_ndvi(img: torch.Tensor) -> torch.Tensor:
    """
    Compute pseudo-NDVI from RGB image.
    
    Since we don't have NIR, we use GRVI (Green-Red Vegetation Index)
    as a proxy: GRVI = (G - R) / (G + R), which correlates with green biomass.
    
    Args:
        img: (B, 3, H, W) RGB image (ImageNet normalized)
    Returns:
        ndvi: (B, 1) mean GRVI value per image, range approximately [-1, 1]
    """
    # Denormalize from ImageNet stats to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    img_denorm = (img * std + mean).clamp(0, 1)
    
    r, g, b = img_denorm.unbind(dim=1)
    
    # GRVI as pseudo-NDVI
    grvi = (g - r) / (g + r + 1e-6)
    
    # Return mean per image
    return grvi.mean(dim=(-2, -1), keepdim=True)  # (B, 1)


class VegetationIndices(nn.Module):
    """
    Compute vegetation indices from RGB image.
    
    Based on agricultural remote sensing research:
    - ExG (Excess Green): High for green vegetation, low/negative for dead
    - ExR (Excess Red): High for dead/brown material, low for green
    - GRVI (Green-Red VI): Normalized, lighting-robust
    
    These exploit chlorophyll's spectral signature:
    - Chlorophyll absorbs R and B, reflects G → high ExG for green plants
    - Dead vegetation lacks chlorophyll → low ExG, higher ExR
    """
    
    def __init__(self, out_dim: int = 24) -> None:
        super().__init__()
        # 6 indices × 4 stats = 24 features, project to out_dim
        self.proj = nn.Sequential(
            nn.Linear(24, out_dim),
            nn.GELU(),
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, H, W) RGB image (ImageNet normalized)
        Returns:
            features: (B, out_dim) pooled vegetation features
        """
        # Denormalize from ImageNet stats to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img_denorm = (img * std + mean).clamp(0, 1)
        
        r, g, b = img_denorm.unbind(dim=1)
        
        # === GREEN BIOMASS INDICATORS ===
        exg = 2 * g - r - b  # Excess Green
        grvi = (g - r) / (g + r + 1e-6)  # Green-Red VI
        vari = (g - r) / (g + r - b + 1e-6)  # VARI
        
        # === DEAD BIOMASS INDICATORS ===
        exr = 1.4 * r - g  # Excess Red
        exgr = exg - exr  # ExG - ExR
        
        # === GENERAL ===
        norm_g = g / (r + g + b + 1e-6)  # Normalized Green
        
        indices = torch.stack([exg, exr, exgr, grvi, norm_g, vari], dim=1)  # (B, 6, H, W)
        
        # Pool statistics per index
        feats = []
        for i in range(indices.size(1)):
            idx = indices[:, i]  # (B, H, W)
            feats.extend([
                idx.mean(dim=(-2, -1)),  # Mean
                idx.std(dim=(-2, -1)),   # Std
                idx.flatten(1).quantile(0.1, dim=1),  # 10th percentile
                idx.flatten(1).quantile(0.9, dim=1),  # 90th percentile
            ])
        
        stats = torch.stack(feats, dim=1)  # (B, 24)
        return self.proj(stats)


# =============================================================================
# Learnable Augmentation Module
# =============================================================================

class LearnableAugmentation(nn.Module):
    """
    Learnable augmentation module that learns optimal augmentation parameters.
    
    Incorporates all strong augmentations from dinov3-5tar.ipynb as learnable:
    - ColorJitter (brightness, contrast, saturation, hue) → learnable
    - HueSaturationValue → learnable  
    - CLAHE (local contrast) → learnable approximation
    - MotionBlur/GaussianBlur/GaussNoise → learnable blur kernel + noise
    - Affine (scale, rotation, translation) → learnable
    
    Features:
    - Differentiable for end-to-end training
    - Input-dependent augmentation prediction
    - Diversity loss to prevent collapse to identity
    
    During training: applies learned augmentations with noise for diversity
    During inference: disabled (identity transform)
    """
    
    def __init__(
        self,
        enable_color: bool = True,
        enable_spatial: bool = False,
        enable_blur: bool = True,
        enable_local_contrast: bool = True,
        color_strength: float = 0.25,
        spatial_strength: float = 0.15,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.enable_color = enable_color
        self.enable_spatial = enable_spatial
        self.enable_blur = enable_blur
        self.enable_local_contrast = enable_local_contrast
        self.color_strength = color_strength
        self.spatial_strength = spatial_strength
        self.noise_std = noise_std
        
        if enable_color:
            # Learnable color transform parameters (ColorJitter + HSV style)
            # [brightness, contrast, saturation, hue, val_shift, sat_shift_extra]
            self.color_params = nn.Parameter(torch.zeros(6))
            
            # Network to predict input-dependent augmentation
            self.color_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(3, 32),
                nn.GELU(),
                nn.Linear(32, 6),
                nn.Tanh(),
            )
        
        if enable_spatial:
            # Learnable affine parameters: [scale_x, scale_y, rotation, tx, ty]
            self.spatial_params = nn.Parameter(torch.zeros(5))
            
            # Network to predict input-dependent spatial transform
            self.spatial_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(3, 32),
                nn.GELU(),
                nn.Linear(32, 5),
                nn.Tanh(),
            )
        
        if enable_blur:
            # Learnable blur/noise parameters: [blur_strength, noise_strength]
            self.blur_params = nn.Parameter(torch.zeros(2))
            
            self.blur_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(3, 16),
                nn.GELU(),
                nn.Linear(16, 2),
                nn.Sigmoid(),
            )
            
            # Learnable blur kernel (3x3 Gaussian-like, can learn motion blur)
            self.blur_kernel = nn.Parameter(torch.tensor([
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.],
            ]) / 16.0)
        
        if enable_local_contrast:
            # Learnable local contrast (CLAHE-like)
            self.contrast_params = nn.Parameter(torch.zeros(2))  # [clip_limit, strength]
            
            self.contrast_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(3, 16),
                nn.GELU(),
                nn.Linear(16, 2),
                nn.Sigmoid(),
            )
    
    def _apply_color_transform(
        self, img: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply differentiable color transformations (ColorJitter + HSV style).
        
        Args:
            img: (B, 3, H, W) normalized image
            params: (B, 6) [brightness, contrast, saturation, hue, val_shift, sat_shift]
        """
        brightness = params[:, 0]
        contrast = params[:, 1]
        saturation = params[:, 2]
        hue = params[:, 3]
        val_shift = params[:, 4]
        sat_shift = params[:, 5]
        
        # Denormalize for color ops
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img_denorm = img * std + mean  # [0, 1]
        
        # Brightness (like ColorJitter brightness=0.2)
        brightness = brightness.view(-1, 1, 1, 1) * self.color_strength
        img_denorm = img_denorm + brightness
        
        # Contrast (like ColorJitter contrast=0.2)
        contrast = 1.0 + contrast.view(-1, 1, 1, 1) * self.color_strength
        img_mean = img_denorm.mean(dim=(1, 2, 3), keepdim=True)
        img_denorm = (img_denorm - img_mean) * contrast + img_mean
        
        # Saturation (like ColorJitter saturation=0.2 + HSV sat_shift=30)
        saturation = 1.0 + saturation.view(-1, 1, 1, 1) * self.color_strength * 1.5
        gray = 0.2989 * img_denorm[:, 0:1] + 0.587 * img_denorm[:, 1:2] + 0.114 * img_denorm[:, 2:3]
        gray = gray.expand_as(img_denorm)
        img_denorm = gray + saturation * (img_denorm - gray)
        
        # Hue shift (like ColorJitter hue=0.1 + HSV hue_shift=20)
        hue = hue.view(-1, 1, 1, 1) * self.color_strength * 0.8
        r, g, b = img_denorm.unbind(dim=1)
        r_new = r - hue.squeeze(1) * g + hue.squeeze(1) * 0.3 * b
        g_new = g + hue.squeeze(1) * r - hue.squeeze(1) * 0.3 * b
        b_new = b + hue.squeeze(1) * 0.2 * (r - g)
        img_denorm = torch.stack([r_new, g_new, b_new], dim=1)
        
        # Value shift (like HSV val_shift=20)
        val_shift = val_shift.view(-1, 1, 1, 1) * self.color_strength
        img_denorm = img_denorm + val_shift
        
        # Extra saturation shift
        sat_shift = 1.0 + sat_shift.view(-1, 1, 1, 1) * self.color_strength
        gray = 0.2989 * img_denorm[:, 0:1] + 0.587 * img_denorm[:, 1:2] + 0.114 * img_denorm[:, 2:3]
        gray = gray.expand_as(img_denorm)
        img_denorm = gray + sat_shift * (img_denorm - gray)
        
        # Clamp and renormalize
        img_denorm = img_denorm.clamp(0, 1)
        img_aug = (img_denorm - mean) / std
        
        return img_aug
    
    def _apply_spatial_transform(
        self, img: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply differentiable spatial transformations (Affine-like).
        
        Args:
            img: (B, 3, H, W) image
            params: (B, 5) [scale_x, scale_y, rotation, tx, ty]
        """
        B, _, H, W = img.shape
        
        # Scale (like Affine scale=(0.9, 1.1))
        scale_x = 1.0 + params[:, 0] * self.spatial_strength
        scale_y = 1.0 + params[:, 1] * self.spatial_strength
        # Rotation (like Affine rotate=(-10, 10))
        rotation = params[:, 2] * self.spatial_strength * 0.3  # ~17 degrees max
        # Translation (like Affine translate=(0.05, 0.05))
        tx = params[:, 3] * self.spatial_strength * 0.15
        ty = params[:, 4] * self.spatial_strength * 0.15
        
        # Build affine matrices
        cos_r = torch.cos(rotation)
        sin_r = torch.sin(rotation)
        
        theta = torch.zeros(B, 2, 3, device=img.device)
        theta[:, 0, 0] = scale_x * cos_r
        theta[:, 0, 1] = -scale_x * sin_r
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = scale_y * sin_r
        theta[:, 1, 1] = scale_y * cos_r
        theta[:, 1, 2] = ty
        
        grid = F.affine_grid(theta, img.shape, align_corners=False)
        img_aug = F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        
        return img_aug
    
    def _apply_blur_noise(
        self, img: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply learnable blur and noise (GaussianBlur/MotionBlur/GaussNoise style).
        
        Args:
            img: (B, 3, H, W) image
            params: (B, 2) [blur_strength, noise_strength]
        """
        blur_strength = params[:, 0:1].view(-1, 1, 1, 1) * 0.4  # Max 40% blur
        noise_strength = params[:, 1:2].view(-1, 1, 1, 1) * 0.08  # Max 8% noise
        
        # Apply learnable blur kernel (can learn motion blur direction)
        kernel = self.blur_kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3)
        # Normalize kernel
        kernel = kernel / (kernel.sum(dim=(-2, -1), keepdim=True) + 1e-6)
        blurred = F.conv2d(img, kernel, padding=1, groups=3)
        
        # Interpolate between original and blurred
        img_blur = img * (1 - blur_strength) + blurred * blur_strength
        
        # Add learnable noise (like GaussNoise std=0.05-0.1)
        noise = torch.randn_like(img) * noise_strength
        img_aug = img_blur + noise
        
        return img_aug
    
    def _apply_local_contrast(
        self, img: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply learnable local contrast enhancement (CLAHE-like).
        
        Args:
            img: (B, 3, H, W) image
            params: (B, 2) [clip_limit, strength]
        """
        clip_limit = params[:, 0:1].view(-1, 1, 1, 1) * 2.0 + 1.0  # [1, 3]
        strength = params[:, 1:2].view(-1, 1, 1, 1) * 0.5  # [0, 0.5] - subtle effect
        
        # Compute local mean using average pooling
        kernel_size = 16
        pad = kernel_size // 2
        local_mean = F.avg_pool2d(
            F.pad(img, [pad, pad, pad, pad], mode='reflect'),
            kernel_size, stride=1
        )
        
        # Compute local contrast
        local_diff = img - local_mean
        
        # Soft clipping (differentiable CLAHE approximation)
        clipped_diff = torch.tanh(local_diff / clip_limit) * clip_limit
        
        # Enhanced image
        enhanced = local_mean + clipped_diff * 1.2  # Slight boost
        
        # Interpolate based on strength
        img_aug = img * (1 - strength) + enhanced * strength
        
        return img_aug
    
    def forward(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learnable augmentation (all strong augmentations from notebook).
        
        Args:
            img: (B, 3, H, W) input image
            
        Returns:
            aug_img: Augmented image
            diversity_loss: Loss to encourage diverse augmentations
        """
        B = img.size(0)
        diversity_loss = torch.tensor(0.0, device=img.device)
        
        if not self.training:
            return img, diversity_loss
        
        aug_img = img
        
        # 1. Local contrast (CLAHE-like) - first to enhance details
        if self.enable_local_contrast:
            pred_contrast = self.contrast_predictor(img)
            base_contrast = torch.sigmoid(self.contrast_params).unsqueeze(0).expand(B, -1)
            noise_contrast = torch.rand_like(pred_contrast) * self.noise_std
            contrast_params = (base_contrast + pred_contrast * 0.3 + noise_contrast).clamp(0, 1)
            
            aug_img = self._apply_local_contrast(aug_img, contrast_params)
            diversity_loss = diversity_loss + torch.exp(-contrast_params.abs().mean()) * 0.005
        
        # 2. Color transforms (ColorJitter + HSV)
        if self.enable_color:
            pred_color = self.color_predictor(aug_img)
            base_color = self.color_params.unsqueeze(0).expand(B, -1)
            noise_color = torch.randn_like(pred_color) * self.noise_std
            color_params = base_color + pred_color + noise_color
            
            aug_img = self._apply_color_transform(aug_img, color_params)
            diversity_loss = diversity_loss + torch.exp(-color_params.abs().mean()) * 0.01
        
        # 3. Spatial transforms (Affine)
        if self.enable_spatial:
            pred_spatial = self.spatial_predictor(img)  # Use original for spatial
            base_spatial = self.spatial_params.unsqueeze(0).expand(B, -1)
            noise_spatial = torch.randn_like(pred_spatial) * self.noise_std
            spatial_params = base_spatial + pred_spatial + noise_spatial
            
            aug_img = self._apply_spatial_transform(aug_img, spatial_params)
            diversity_loss = diversity_loss + torch.exp(-spatial_params.abs().mean()) * 0.01
        
        # 4. Blur/noise (GaussianBlur/MotionBlur/GaussNoise) - last
        if self.enable_blur:
            pred_blur = self.blur_predictor(aug_img)
            base_blur = torch.sigmoid(self.blur_params).unsqueeze(0).expand(B, -1)
            noise_blur = torch.rand_like(pred_blur) * self.noise_std
            blur_params = (base_blur + pred_blur * 0.2 + noise_blur).clamp(0, 1)
            
            aug_img = self._apply_blur_noise(aug_img, blur_params)
            diversity_loss = diversity_loss + torch.exp(-blur_params.abs().mean()) * 0.005
        
        return aug_img, diversity_loss
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """Get current learned parameters for logging."""
        params = {}
        if self.enable_color:
            params["color"] = self.color_params.detach().cpu()
        if self.enable_spatial:
            params["spatial"] = self.spatial_params.detach().cpu()
        if self.enable_blur:
            params["blur"] = self.blur_params.detach().cpu()
            params["blur_kernel"] = self.blur_kernel.detach().cpu()
        if self.enable_local_contrast:
            params["contrast"] = self.contrast_params.detach().cpu()
        return params


# =============================================================================
# Stereo Disparity Features (3D Volume Exploitation)
# =============================================================================

class DisparityFeatures(nn.Module):
    """
    Extract disparity-based features from stereo tile features.
    
    Key insight: Taller/denser vegetation is closer to camera → larger disparity
    between left/right views → correlates with biomass volume.
    
    This exploits 3D information that FiLM only learns implicitly.
    """
    
    def __init__(self, feat_dim: int, max_disparity: int = 8, out_dim: int = None) -> None:
        super().__init__()
        self.max_disparity = max_disparity
        if out_dim is None:
            out_dim = feat_dim // 4
        
        # Project correlation volume to features
        self.proj = nn.Sequential(
            nn.Linear(max_disparity, out_dim),
            nn.GELU(),
        )
        
        # Also learn from difference statistics
        self.diff_proj = nn.Sequential(
            nn.Linear(6, out_dim // 2),
            nn.GELU(),
        )
        
        self.out_dim = out_dim + out_dim // 2
    
    def forward(
        self,
        feat_left: torch.Tensor,   # (B, N, D) - N tiles, D features
        feat_right: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        """
        Args:
            feat_left: (B, num_tiles, feat_dim) tile features from left view
            feat_right: (B, num_tiles, feat_dim) tile features from right view
        Returns:
            disp_features: (B, out_dim) disparity-based features
        """
        B, N, D = feat_left.shape
        
        # Normalize for correlation
        feat_l = F.normalize(feat_left, dim=-1)
        feat_r = F.normalize(feat_right, dim=-1)
        
        # Compute correlation at different "shifts" (simulating disparity)
        correlations = []
        for d in range(self.max_disparity):
            # Shift right features (circular)
            shifted_r = torch.roll(feat_r, shifts=d, dims=1)
            corr = (feat_l * shifted_r).sum(dim=-1)  # (B, N)
            correlations.append(corr.mean(dim=1))  # Average over tiles
        
        # Stack: (B, max_disparity)
        corr_volume = torch.stack(correlations, dim=-1)
        
        # Project correlation volume
        corr_feat = self.proj(corr_volume)  # (B, out_dim)
        
        # === Additional: Difference statistics ===
        # These encode disparity indirectly
        fl_pooled = feat_left.mean(dim=1)   # (B, D)
        fr_pooled = feat_right.mean(dim=1)  # (B, D)
        
        fl_norm = F.normalize(fl_pooled, dim=-1)
        fr_norm = F.normalize(fr_pooled, dim=-1)
        
        # Correlation = similarity between views
        correlation = (fl_norm * fr_norm).sum(dim=-1, keepdim=True)  # (B, 1)
        
        # Difference features
        diff = fl_pooled - fr_pooled
        diff_norm = diff.norm(dim=-1, keepdim=True)  # (B, 1)
        diff_mean = diff.mean(dim=-1, keepdim=True)  # (B, 1)
        diff_std = diff.std(dim=-1, keepdim=True)    # (B, 1)
        
        # Ratio features
        ratio = fl_pooled / (fr_pooled + 1e-6)
        ratio_mean = ratio.mean(dim=-1, keepdim=True)
        ratio_std = ratio.std(dim=-1, keepdim=True)
        
        stats = torch.cat([
            correlation, diff_norm, diff_mean, diff_std, ratio_mean, ratio_std
        ], dim=-1)  # (B, 6)
        
        diff_feat = self.diff_proj(stats)  # (B, out_dim//2)
        
        return torch.cat([corr_feat, diff_feat], dim=-1)  # (B, out_dim + out_dim//2)


# =============================================================================
# Depth Features (Depth Anything V2)
# =============================================================================

class DepthFeatures(nn.Module):
    """
    Extract depth-based features using Depth Anything V2.
    
    Based on correlation analysis (scripts/check_depth_usefulness.py):
    - depth_gradient → r=0.63 for green, r=0.51 for total (BEST)
    - depth_mean → r=0.40 for clover
    - depth_range → r=0.40 for green  
    - depth_volume → r=0.35 for gdm
    - depth_lr_diff → r=0.32 for green (stereo signal)
    
    Uses frozen DA2 model to generate depth maps, then extracts statistics.
    """
    
    def __init__(self, out_dim: int = 32, model_size: str = "small") -> None:
        super().__init__()
        self.out_dim = out_dim
        self.model_size = model_size
        self._depth_model = None  # Lazy load
        self._processor = None
        
        # Project depth statistics to feature space
        # 10 stats per view × 2 views + 2 stereo stats = 22 features
        self.proj = nn.Sequential(
            nn.Linear(22, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def _load_depth_model(self, device: torch.device) -> None:
        """Lazy load depth model on first use."""
        if self._depth_model is not None:
            return
        
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        
        model_names = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
        }
        model_name = model_names.get(self.model_size, model_names["small"])
        
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self._depth_model.eval()
        self._depth_model = self._depth_model.to(device)
        
        # Freeze depth model
        for p in self._depth_model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def _get_depth_map(self, img: torch.Tensor) -> torch.Tensor:
        """
        Get depth map from image tensor.
        
        Args:
            img: (B, 3, H, W) ImageNet-normalized tensor
        Returns:
            depth: (B, H, W) depth map
        """
        device = img.device
        self._load_depth_model(device)
        
        # Denormalize from ImageNet to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img_denorm = (img * std + mean).clamp(0, 1)
        
        # DA2 expects specific preprocessing - resize to 518
        B, _, H, W = img.shape
        img_resized = F.interpolate(img_denorm, size=(518, 518), mode="bilinear", align_corners=False)
        
        # Convert to processor format (already normalized by processor internally)
        # DA2 processor expects PIL or numpy, but we can use tensor directly
        # by applying the same normalization
        da2_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        da2_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img_normalized = (img_resized - da2_mean) / da2_std
        
        # Get depth
        outputs = self._depth_model(pixel_values=img_normalized)
        depth = outputs.predicted_depth  # (B, h, w)
        
        # Resize back to original
        depth = F.interpolate(
            depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)
        
        return depth
    
    def _compute_stats(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute statistics from depth map.
        
        Args:
            depth: (B, H, W) depth map
        Returns:
            stats: (B, 10) statistics
        """
        B, H, W = depth.shape
        flat = depth.view(B, -1)
        
        # Basic statistics
        depth_mean = flat.mean(dim=1)
        depth_std = flat.std(dim=1)
        depth_min = flat.min(dim=1).values
        depth_max = flat.max(dim=1).values
        depth_range = depth_max - depth_min
        
        # Percentiles (using quantile)
        depth_p10 = flat.quantile(0.1, dim=1)
        depth_p90 = flat.quantile(0.9, dim=1)
        
        # Gradient (vegetation boundaries) - KEY FEATURE (r=0.63)
        grad_y = torch.abs(depth[:, 1:, :] - depth[:, :-1, :]).mean(dim=(1, 2))
        grad_x = torch.abs(depth[:, :, 1:] - depth[:, :, :-1]).mean(dim=(1, 2))
        depth_gradient = grad_y + grad_x
        
        # Volume proxy (sum above minimum)
        depth_volume = (flat - depth_min.unsqueeze(1)).mean(dim=1)
        
        # High depth ratio
        threshold = flat.quantile(0.75, dim=1, keepdim=True)
        depth_high_ratio = (flat > threshold).float().mean(dim=1)
        
        return torch.stack([
            depth_mean, depth_std, depth_min, depth_max, depth_range,
            depth_p10, depth_p90, depth_gradient, depth_volume, depth_high_ratio
        ], dim=1)  # (B, 10)
    
    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        """
        Extract depth features from stereo images.
        
        Args:
            x_left: (B, 3, H, W) left image (ImageNet normalized)
            x_right: (B, 3, H, W) right image (ImageNet normalized)
        Returns:
            features: (B, out_dim) depth features
        """
        # Get depth maps
        depth_left = self._get_depth_map(x_left)
        depth_right = self._get_depth_map(x_right)
        
        # Compute per-view statistics
        stats_left = self._compute_stats(depth_left)    # (B, 10)
        stats_right = self._compute_stats(depth_right)  # (B, 10)
        
        # Stereo statistics (L-R difference as disparity proxy)
        depth_lr_diff = torch.abs(depth_left - depth_right).mean(dim=(1, 2)).unsqueeze(1)  # (B, 1)
        depth_lr_corr = F.cosine_similarity(
            depth_left.flatten(1), depth_right.flatten(1), dim=1
        ).unsqueeze(1)  # (B, 1)
        
        # Combine all features
        all_stats = torch.cat([
            stats_left, stats_right, depth_lr_diff, depth_lr_corr
        ], dim=1)  # (B, 22)
        
        return self.proj(all_stats)  # (B, out_dim)


# =============================================================================
# Utility Modules
# =============================================================================

class FiLM(nn.Module):
    """Feature-wise Linear Modulation for stereo fusion."""
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(64, dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim * 2),
        )
    
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns gamma (scale) and beta (shift) for FiLM."""
        gb = self.mlp(context)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma, beta


class AttentionPooling(nn.Module):
    """Learnable attention pooling over tiles."""
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_tiles, D)
        Returns:
            (B, D) - attention-weighted mean
        """
        q = self.query(x.mean(dim=1, keepdim=True))  # (B, 1, D)
        k = self.key(x)  # (B, T, D)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, 1, T)
        attn = F.softmax(attn, dim=-1)
        
        return (attn @ x).squeeze(1)  # (B, D)


class DepthGuidedAttention(nn.Module):
    """
    Depth-guided attention for tile pooling.
    Uses depth maps to weight which spatial regions contribute more to predictions.
    
    Intuition: Taller vegetation (higher depth values) = more biomass → higher attention weight.
    """
    
    def __init__(
        self, 
        feat_dim: int, 
        grid: Tuple[int, int] = (2, 2),
        model_size: str = "small"
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.grid = grid
        self.num_tiles = grid[0] * grid[1]
        
        # Lazy load depth model
        self._depth_model = None
        self._model_size = model_size
        
        # Learn how to combine depth with attention
        # Input: depth stats per tile (5 stats: mean, max, gradient, volume, high_ratio)
        self.depth_stats_dim = 5
        
        # Depth → attention weight (per tile)
        self.depth_to_attn = nn.Sequential(
            nn.Linear(self.depth_stats_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        # Feature-based attention (like original)
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.scale = feat_dim ** -0.5
        
        # Combine depth attention and feature attention
        self.gate = nn.Parameter(torch.tensor(0.5))  # Learnable balance
    
    def _ensure_model_loaded(self, device: torch.device) -> None:
        """Lazy load the depth model."""
        if self._depth_model is None:
            from transformers import AutoModelForDepthEstimation
            model_name = f"depth-anything/Depth-Anything-V2-{self._model_size.capitalize()}-hf"
            self._depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self._depth_model.to(device)
            self._depth_model.eval()
            for p in self._depth_model.parameters():
                p.requires_grad = False
    
    @torch.no_grad()
    def _get_depth_map(self, img: torch.Tensor) -> torch.Tensor:
        """Get depth map from normalized image - fully on GPU."""
        device = img.device
        self._ensure_model_loaded(device)
        
        B, C, H, W = img.shape
        
        # De-normalize from ImageNet to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img_denorm = (img * std + mean).clamp(0, 1)
        
        # DA2 expects 518x518 input
        img_resized = F.interpolate(img_denorm, size=(518, 518), mode="bilinear", align_corners=False)
        
        # Apply DA2 normalization directly on GPU (same as ImageNet)
        da2_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        da2_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img_normalized = (img_resized - da2_mean) / da2_std
        
        # Get depth directly on GPU
        outputs = self._depth_model(pixel_values=img_normalized)
        depth = outputs.predicted_depth  # (B, h, w)
        
        # Resize back to original
        depth = F.interpolate(
            depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)
        
        return depth  # (B, H, W)
    
    def _compute_tile_depth_stats(
        self, 
        depth: torch.Tensor, 
        img_h: int, 
        img_w: int
    ) -> torch.Tensor:
        """
        Compute depth statistics per tile.
        
        Args:
            depth: (B, H, W) depth map
            img_h, img_w: original image dimensions
        Returns:
            tile_stats: (B, num_tiles, depth_stats_dim)
        """
        B = depth.shape[0]
        gh, gw = self.grid
        
        # Create tile regions
        h_edges = _make_edges(img_h, gh)
        w_edges = _make_edges(img_w, gw)
        
        tile_stats = []
        for i in range(gh):
            for j in range(gw):
                h_start, h_end = h_edges[i]
                w_start, w_end = w_edges[j]
                
                tile_depth = depth[:, h_start:h_end, w_start:w_end]  # (B, th, tw)
                flat = tile_depth.reshape(B, -1)
                
                # Stats for this tile
                d_mean = flat.mean(dim=1)
                d_max = flat.max(dim=1).values
                
                # Gradient
                grad_y = torch.abs(tile_depth[:, 1:, :] - tile_depth[:, :-1, :]).mean(dim=(1, 2))
                grad_x = torch.abs(tile_depth[:, :, 1:] - tile_depth[:, :, :-1]).mean(dim=(1, 2))
                d_gradient = grad_y + grad_x
                
                # Volume
                d_volume = (flat - flat.min(dim=1, keepdim=True).values).mean(dim=1)
                
                # High ratio
                thresh = flat.quantile(0.75, dim=1, keepdim=True)
                d_high = (flat > thresh).float().mean(dim=1)
                
                tile_stats.append(torch.stack([d_mean, d_max, d_gradient, d_volume, d_high], dim=1))
        
        return torch.stack(tile_stats, dim=1)  # (B, num_tiles, 5)
    
    def forward(
        self, 
        tiles: torch.Tensor,
        x_img: torch.Tensor
    ) -> torch.Tensor:
        """
        Depth-guided attention pooling.
        
        Args:
            tiles: (B, num_tiles, D) tile features
            x_img: (B, 3, H, W) original image for depth computation
        Returns:
            pooled: (B, D) attention-weighted features
        """
        B, T, D = tiles.shape
        _, _, H, W = x_img.shape
        
        # 1. Feature-based attention (like original)
        q = self.query(tiles.mean(dim=1, keepdim=True))  # (B, 1, D)
        k = self.key(tiles)  # (B, T, D)
        feat_attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, 1, T)
        feat_attn = F.softmax(feat_attn, dim=-1).squeeze(1)  # (B, T)
        
        # 2. Depth-based attention
        depth = self._get_depth_map(x_img)  # (B, H, W)
        tile_stats = self._compute_tile_depth_stats(depth, H, W)  # (B, T, 5)
        
        # Normalize depth stats
        tile_stats = (tile_stats - tile_stats.mean(dim=1, keepdim=True)) / (
            tile_stats.std(dim=1, keepdim=True) + 1e-6
        )
        
        depth_scores = self.depth_to_attn(tile_stats).squeeze(-1)  # (B, T)
        depth_attn = F.softmax(depth_scores, dim=-1)  # (B, T)
        
        # 3. Combine with learnable gate
        gate = torch.sigmoid(self.gate)
        combined_attn = gate * feat_attn + (1 - gate) * depth_attn  # (B, T)
        combined_attn = combined_attn / combined_attn.sum(dim=1, keepdim=True)  # Renormalize
        
        # 4. Apply attention
        return (combined_attn.unsqueeze(1) @ tiles).squeeze(1)  # (B, D)


def _make_edges(length: int, parts: int) -> List[Tuple[int, int]]:
    """Create edge indices for tiling."""
    step = length // parts
    edges = []
    start = 0
    for _ in range(parts - 1):
        edges.append((start, start + step))
        start += step
    edges.append((start, length))
    return edges


# =============================================================================
# DINOv3 Direct Model
# =============================================================================

class DINOv3Direct(nn.Module):
    """
    DINOv3 Direct Model for Biomass Prediction.
    
    Always predicts: Total, Green, GDM
    Optionally predicts: Dead, Clover (via separate heads)
    
    Derivation rules:
    - If not train_dead: Dead = Total - GDM
    - If not train_clover: Clover = GDM - Green
    
    This guarantees: Green + Dead + Clover = Total
    
    Optional features:
    - Vegetation Indices (VI): ExG, ExR, GRVI etc. for green/dead signals
    - Stereo Disparity: 3D volume features from stereo correspondence
    - Depth Features: Depth Anything V2 depth maps (r=0.63 correlation with green!)
    - Auxiliary Heads: State, Month, Species classification for multi-task learning
    """
    
    # Auxiliary head class counts (must match dataset.py)
    NUM_STATES = 4
    NUM_MONTHS = 10
    NUM_SPECIES = 8
    
    def __init__(
        self,
        grid: Tuple[int, int] = (2, 2),
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_ratio: float = 0.25,
        use_film: bool = True,
        use_attention_pool: bool = True,
        gradient_checkpointing: bool = False,
        train_dead: bool = False,
        train_clover: bool = False,
        use_vegetation_indices: bool = False,
        use_disparity: bool = False,
        use_depth: bool = False,
        depth_model_size: str = "small",
        use_depth_attention: bool = False,
        use_learnable_aug: bool = False,
        learnable_aug_color: bool = True,
        learnable_aug_spatial: bool = False,
        use_aux_heads: bool = False,
        backbone_size: str = "base",
        backbone_type: str = "dinov3",
        ckpt_path: Optional[str] = None,
        use_whole_image: bool = False,
        use_presence_heads: bool = False,
        use_ndvi_head: bool = False,
        use_height_head: bool = False,
        use_species_head: bool = False,
    ) -> None:
        super().__init__()
        
        # Build backbone
        self.backbone, self.feat_dim, self.input_res = build_dinov3_backbone(
            pretrained=pretrained,
            gradient_checkpointing=gradient_checkpointing,
            backbone_size=backbone_size,
            backbone_type=backbone_type,
            ckpt_path=ckpt_path,
        )
        self.backbone_type = backbone_type
        self.backbone_size = backbone_size
        self.use_whole_image = use_whole_image
        self.use_presence_heads = use_presence_heads
        self.use_ndvi_head = use_ndvi_head
        self.use_height_head = use_height_head
        self.use_species_head = use_species_head
        
        self.grid = tuple(grid)
        self.use_film = use_film
        self.use_attention_pool = use_attention_pool
        self.train_dead = train_dead
        self.train_clover = train_clover
        self.use_vegetation_indices = use_vegetation_indices
        self.use_disparity = use_disparity
        self.use_depth = use_depth
        self.use_depth_attention = use_depth_attention
        self.use_learnable_aug = use_learnable_aug
        self.depth_model_size = depth_model_size
        
        # Learnable augmentation (applied during training only)
        # Includes all strong augmentations: color, spatial, blur, CLAHE
        if use_learnable_aug:
            self.learnable_aug_left = LearnableAugmentation(
                enable_color=learnable_aug_color,
                enable_spatial=learnable_aug_spatial,
                enable_blur=learnable_aug_color,  # Enable blur with color
                enable_local_contrast=learnable_aug_color,  # Enable CLAHE with color
            )
            self.learnable_aug_right = LearnableAugmentation(
                enable_color=learnable_aug_color,
                enable_spatial=learnable_aug_spatial,
                enable_blur=learnable_aug_color,
                enable_local_contrast=learnable_aug_color,
            )
        
        # FiLM for cross-view conditioning
        if use_film:
            self.film_left = FiLM(self.feat_dim)
            self.film_right = FiLM(self.feat_dim)
        
        # Attention pooling for tiles
        if use_depth_attention:
            # Depth-guided attention (uses depth maps to weight tiles)
            self.attn_pool_left = DepthGuidedAttention(
                self.feat_dim, grid=grid, model_size=depth_model_size
            )
            self.attn_pool_right = DepthGuidedAttention(
                self.feat_dim, grid=grid, model_size=depth_model_size
            )
        elif use_attention_pool:
            self.attn_pool_left = AttentionPooling(self.feat_dim)
            self.attn_pool_right = AttentionPooling(self.feat_dim)
        
        # === Optional feature modules ===
        extra_dim = 0
        
        # Vegetation Indices (24 features per view → 48 total, projected to 48)
        if use_vegetation_indices:
            vi_out = 24
            self.vi_left = VegetationIndices(out_dim=vi_out)
            self.vi_right = VegetationIndices(out_dim=vi_out)
            extra_dim += vi_out * 2  # 48
        
        # Stereo Disparity Features
        if use_disparity:
            self.disparity_module = DisparityFeatures(
                self.feat_dim, 
                max_disparity=8, 
                out_dim=self.feat_dim // 4
            )
            extra_dim += self.disparity_module.out_dim  # ~288
        
        # Depth Features (Depth Anything V2) - r=0.63 correlation with green!
        if use_depth:
            depth_out_dim = 32
            self.depth_module = DepthFeatures(
                out_dim=depth_out_dim,
                model_size=depth_model_size,
            )
            extra_dim += depth_out_dim
        
        # Whole Image Features (global context from non-tiled images)
        # Provides field-level patterns that tiles might miss
        if use_whole_image:
            # Cross-attention to fuse whole-image features with tiled features
            self.whole_img_proj = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.GELU(),
            )
            # Gate to control contribution of whole-image features
            self.whole_img_gate = nn.Sequential(
                nn.Linear(self.feat_dim * 2, self.feat_dim),  # tiled + whole → gate
                nn.Sigmoid(),
            )
            extra_dim += self.feat_dim  # Add whole-image features to combined dim
        
        # Head dimensions
        combined_dim = self.feat_dim * 2 + extra_dim
        hidden_dim = max(64, int((self.feat_dim * 2) * hidden_ratio))  # Base on original feat_dim
        self.hidden_dim = hidden_dim
        self.combined_dim = combined_dim
        
        # Shared projection
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Prediction heads
        def _make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim, 1),
            )
        
        # Core heads (always used)
        self.head_total = _make_head()
        self.head_green = _make_head()
        self.head_gdm = _make_head()
        
        # Optional heads
        self.head_dead = _make_head() if train_dead else None
        self.head_clover = _make_head() if train_clover else None
        
        # Auxiliary heads for multi-task learning (State, Month, Species)
        self.use_aux_heads = use_aux_heads
        if use_aux_heads:
            self.head_state = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.NUM_STATES),
            )
            self.head_month = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.NUM_MONTHS),
            )
            self.head_species = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.NUM_SPECIES),
            )
        
        # Presence heads: Binary classification for "has Dead?" / "has Clover?"
        # Helps with zero-inflated targets: predict IF present, then HOW MUCH
        if use_presence_heads:
            self.head_dead_presence = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.head_clover_presence = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        
        # NDVI auxiliary head: Predict ground-truth NDVI (Pre_GSHH_NDVI)
        # Range: 0.16-0.91, correlates strongly with green biomass
        if use_ndvi_head:
            self.head_ndvi = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # NDVI is in [0, 1] for ground-truth
            )
        
        # Height auxiliary head: Predict average height (Height_Ave_cm)
        # Range: 1-70 cm, correlates with total biomass volume
        if use_height_head:
            self.head_height = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # Height must be positive
            )
        
        # Species-only classification head (when not using full aux_heads)
        if use_species_head and not use_aux_heads:
            self.head_species_only = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.NUM_SPECIES),
            )
        
        self.softplus = nn.Softplus(beta=1.0)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize head weights."""
        heads = [self.head_total, self.head_green, self.head_gdm]
        if self.head_dead is not None:
            heads.append(self.head_dead)
        if self.head_clover is not None:
            heads.append(self.head_clover)
        if self.use_aux_heads:
            heads.extend([self.head_state, self.head_month, self.head_species])
        if self.use_presence_heads:
            heads.extend([self.head_dead_presence, self.head_clover_presence])
        if self.use_ndvi_head:
            heads.append(self.head_ndvi)
        if self.use_height_head:
            heads.append(self.head_height)
        if self.use_species_head and not self.use_aux_heads:
            heads.append(self.head_species_only)
        
        for head in heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _collect_tiles(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split image into grid of tiles."""
        _, _, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)
        
        tiles = []
        for rs, re in rows:
            for cs, ce in cols:
                tile = x[:, :, rs:re, cs:ce]
                # Resize to backbone's native resolution
                if tile.shape[-2:] != (self.input_res, self.input_res):
                    tile = F.interpolate(
                        tile,
                        size=(self.input_res, self.input_res),
                        mode="bilinear",
                        align_corners=False,
                    )
                tiles.append(tile)
        return tiles
    
    def _extract_features(
        self, x_left: torch.Tensor, x_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract tile features from both views in one backbone call."""
        B = x_left.size(0)
        
        tiles_left = self._collect_tiles(x_left)
        tiles_right = self._collect_tiles(x_right)
        num_tiles = len(tiles_left)
        
        # Process all tiles in one forward pass
        all_tiles = torch.cat(tiles_left + tiles_right, dim=0)
        all_feats = self.backbone(all_tiles)  # (2*T*B, D)
        
        # Reshape: (2*T, B, D) -> (B, 2*T, D) -> split
        all_feats = all_feats.view(2 * num_tiles, B, -1).permute(1, 0, 2)
        feats_left = all_feats[:, :num_tiles, :]   # (B, T, D)
        feats_right = all_feats[:, num_tiles:, :]  # (B, T, D)
        
        return feats_left, feats_right
    
    def _extract_whole_image_features(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        f_left: torch.Tensor,
        f_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract global features from whole images (non-tiled).
        
        Provides field-level context that tiles might miss.
        Uses gated fusion to combine with tiled features.
        
        Args:
            x_left: (B, 3, H, W) left image
            x_right: (B, 3, H, W) right image
            f_left: (B, D) pooled tiled features from left
            f_right: (B, D) pooled tiled features from right
        Returns:
            whole_feat: (B, D) gated whole-image features
        """
        B = x_left.size(0)
        
        # Resize whole images to backbone input resolution
        x_left_resized = F.interpolate(
            x_left, size=(self.input_res, self.input_res),
            mode="bilinear", align_corners=False
        )
        x_right_resized = F.interpolate(
            x_right, size=(self.input_res, self.input_res),
            mode="bilinear", align_corners=False
        )
        
        # Extract global features (batch both images together)
        whole_imgs = torch.cat([x_left_resized, x_right_resized], dim=0)  # (2B, 3, H, W)
        whole_feats = self.backbone(whole_imgs)  # (2B, D)
        
        # Split and average left/right
        whole_left = whole_feats[:B]   # (B, D)
        whole_right = whole_feats[B:]  # (B, D)
        whole_combined = (whole_left + whole_right) / 2  # (B, D)
        
        # Project whole-image features
        whole_proj = self.whole_img_proj(whole_combined)  # (B, D)
        
        # Gated fusion: let model learn how much to trust whole-image vs tiled
        tiled_combined = (f_left + f_right) / 2
        gate_input = torch.cat([tiled_combined, whole_proj], dim=1)  # (B, 2*D)
        gate = self.whole_img_gate(gate_input)  # (B, D), values in [0, 1]
        
        # Gated output: blend based on learned gate
        whole_feat = gate * whole_proj  # (B, D)
        
        return whole_feat
    
    def forward(
        self, x_left: torch.Tensor, x_right: torch.Tensor, return_aux: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x_left: Left stereo image (B, 3, H, W)
            x_right: Right stereo image (B, 3, H, W)
            return_aux: If True and use_aux_heads, return auxiliary logits
        
        Returns:
            If return_aux=False: (green, dead, clover, gdm, total, aux_loss)
            If return_aux=True: (green, dead, clover, gdm, total, aux_loss, state_logits, month_logits, species_logits)
            
            aux_loss includes learnable augmentation diversity loss
        """
        # Apply learnable augmentation (only during training)
        aux_loss = torch.tensor(0.0, device=x_left.device)
        if self.use_learnable_aug and self.training:
            x_left, div_loss_l = self.learnable_aug_left(x_left)
            x_right, div_loss_r = self.learnable_aug_right(x_right)
            aux_loss = aux_loss + div_loss_l + div_loss_r
        
        # Extract tile features
        tiles_left, tiles_right = self._extract_features(x_left, x_right)
        
        # === Stereo Disparity Features (before FiLM, uses raw tile features) ===
        disp_feat = None
        if self.use_disparity:
            disp_feat = self.disparity_module(tiles_left, tiles_right)  # (B, disp_dim)
        
        # Context for FiLM (mean of tiles)
        ctx_left = tiles_left.mean(dim=1)   # (B, D)
        ctx_right = tiles_right.mean(dim=1)  # (B, D)
        
        # Apply FiLM cross-conditioning
        if self.use_film:
            gamma_l, beta_l = self.film_left(ctx_right)
            gamma_r, beta_r = self.film_right(ctx_left)
            tiles_left = tiles_left * (1 + gamma_l.unsqueeze(1)) + beta_l.unsqueeze(1)
            tiles_right = tiles_right * (1 + gamma_r.unsqueeze(1)) + beta_r.unsqueeze(1)
        
        # Pool tiles
        if self.use_depth_attention:
            # Depth-guided attention needs the original images
            f_left = self.attn_pool_left(tiles_left, x_left)
            f_right = self.attn_pool_right(tiles_right, x_right)
        elif self.use_attention_pool:
            f_left = self.attn_pool_left(tiles_left)
            f_right = self.attn_pool_right(tiles_right)
        else:
            f_left = tiles_left.mean(dim=1)
            f_right = tiles_right.mean(dim=1)
        
        # Combine features
        features_list = [f_left, f_right]
        
        # === Vegetation Indices Features ===
        if self.use_vegetation_indices:
            vi_left = self.vi_left(x_left)    # (B, vi_dim)
            vi_right = self.vi_right(x_right)  # (B, vi_dim)
            features_list.extend([vi_left, vi_right])
        
        # === Add Disparity Features ===
        if disp_feat is not None:
            features_list.append(disp_feat)
        
        # === Add Depth Features (Depth Anything V2) ===
        if self.use_depth:
            depth_feat = self.depth_module(x_left, x_right)  # (B, depth_dim)
            features_list.append(depth_feat)
        
        # === Whole Image Features (global context) ===
        if self.use_whole_image:
            whole_feat = self._extract_whole_image_features(x_left, x_right, f_left, f_right)
            features_list.append(whole_feat)
        
        f = torch.cat(features_list, dim=1)  # (B, combined_dim)
        f = self.shared_proj(f)
        
        # Core predictions (always computed)
        total_raw = self.softplus(self.head_total(f))
        green_raw = self.softplus(self.head_green(f))
        gdm_raw = self.softplus(self.head_gdm(f))
        
        # Enforce constraints: Total >= GDM >= Green
        total = total_raw
        gdm = torch.minimum(gdm_raw, total)
        green = torch.minimum(green_raw, gdm)
        
        # Presence probabilities for gating (if enabled)
        dead_presence_logit = None
        clover_presence_logit = None
        if self.use_presence_heads:
            dead_presence_logit = self.head_dead_presence(f)      # (B, 1)
            clover_presence_logit = self.head_clover_presence(f)  # (B, 1)
            dead_presence_prob = torch.sigmoid(dead_presence_logit)
            clover_presence_prob = torch.sigmoid(clover_presence_logit)
        
        # Dead: predicted or derived
        if self.head_dead is not None:
            dead_raw = self.softplus(self.head_dead(f))
            # Constraint: Dead <= Total - GDM (shouldn't exceed available space)
            dead = torch.minimum(dead_raw, total - gdm + 1e-6)
            dead = F.relu(dead)
            # Gate by presence probability if enabled
            if self.use_presence_heads:
                dead = dead * dead_presence_prob
        else:
            # Derive: Dead = Total - GDM
            dead = F.relu(total - gdm)
        
        # Clover: predicted or derived
        if self.head_clover is not None:
            clover_raw = self.softplus(self.head_clover(f))
            # Constraint: Clover <= GDM - Green
            clover = torch.minimum(clover_raw, gdm - green + 1e-6)
            clover = F.relu(clover)
            # Gate by presence probability if enabled
            if self.use_presence_heads:
                clover = clover * clover_presence_prob
        else:
            # Derive: Clover = GDM - Green
            clover = F.relu(gdm - green)
        
        # NDVI prediction (auxiliary) - ground-truth range [0, 1]
        ndvi_pred = None
        if self.use_ndvi_head:
            ndvi_pred = self.head_ndvi(f)  # (B, 1), range [0, 1]
        
        # Height prediction (auxiliary) - range [1, 70] cm
        height_pred = None
        if self.use_height_head:
            height_pred = self.head_height(f)  # (B, 1), positive values
        
        # Species-only prediction (when not using full aux heads)
        species_logits = None
        if self.use_species_head and not self.use_aux_heads:
            species_logits = self.head_species_only(f)  # (B, NUM_SPECIES)
        
        # Return auxiliary logits if requested
        if return_aux and self.use_aux_heads:
            state_logits = self.head_state(f)
            month_logits = self.head_month(f)
            species_logits = self.head_species(f)
            return green, dead, clover, gdm, total, aux_loss, state_logits, month_logits, species_logits
        
        # Return with presence/NDVI/Height/Species if enabled
        if self.use_presence_heads or self.use_ndvi_head or self.use_height_head or self.use_species_head:
            return green, dead, clover, gdm, total, aux_loss, dead_presence_logit, clover_presence_logit, ndvi_pred, height_pred, species_logits
        
        return green, dead, clover, gdm, total, aux_loss


# =============================================================================
# Loss Function
# =============================================================================

# Competition metric weights (official)
COMPETITION_WEIGHTS = {
    "green": 0.1,
    "dead": 0.1,
    "clover": 0.1,
    "gdm": 0.2,
    "total": 0.5,
}

# Target indices
TARGET_NAMES = ["green", "dead", "clover", "gdm", "total"]
TARGET_WEIGHTS = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])


# =============================================================================
# Log Transform Encoder (Official Competition Metric)
# =============================================================================

class LogTransformEncoder(nn.Module):
    """
    Official log transformation for competition metric: y_trans = log(1 + y)
    
    From paper: "To stabilize variance and improve model evaluation robustness,
    particularly for variables with high dynamic range or right-skewed distributions,
    a log-stabilizing transformation is applied to all target variables."
    
    Usage:
        encoder = LogTransformEncoder()
        y_log = encoder.transform(targets)  # For training
        pred = encoder.inverse_transform(pred_log)  # For inference
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Transform targets: y_trans = log(1 + y)"""
        return torch.log1p(y)
    
    def inverse_transform(self, y_log: torch.Tensor) -> torch.Tensor:
        """Inverse transform: y = exp(y_trans) - 1"""
        return torch.expm1(y_log)


class OfficialWeightedR2Loss(nn.Module):
    """
    Official competition metric as a differentiable loss function.
    
    From paper:
        Final Score = Σ wi × R²i
        where R²i is calculated using log-transformed values: y_trans = log(1 + y)
        weights: Green=0.1, Dead=0.1, Clover=0.1, GDM=0.2, Total=0.5
    
    This loss:
    1. Applies log(1+y) transform to both predictions and targets
    2. Computes per-target R²
    3. Returns weighted (1 - R²) as loss (since we minimize loss)
    """
    
    def __init__(
        self, 
        use_log_transform: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.use_log_transform = use_log_transform
        self.eps = eps
        self.register_buffer("weights", TARGET_WEIGHTS)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) predictions [Green, Dead, Clover, GDM, Total]
            target: (B, 5) targets
        
        Returns:
            loss: Weighted (1 - R²) loss
        """
        # Apply log transform if targets are in original space
        if self.use_log_transform:
            pred_log = torch.log1p(pred.clamp(min=0))
            target_log = torch.log1p(target)
        else:
            # Assume already log-transformed
            pred_log = pred
            target_log = target
        
        # Compute per-target R²
        # R² = 1 - SS_res / SS_tot
        ss_res = ((target_log - pred_log) ** 2).sum(dim=0)  # (5,)
        ss_tot = ((target_log - target_log.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)  # (5,)
        
        r2_per_target = 1 - ss_res / (ss_tot + self.eps)  # (5,)
        
        # Weighted average R²
        weights = self.weights.to(pred.device)
        weighted_r2 = (weights * r2_per_target).sum()
        
        # Return 1 - R² as loss (minimize loss = maximize R²)
        return 1 - weighted_r2


class CompositionalConsistencyLoss(nn.Module):
    """
    Enforce mathematical relationships between biomass components.
    
    Constraints:
        GDM = Green + Clover
        Total = Green + Dead + Clover
    
    This regularization helps ensure predictions are physically consistent.
    """
    
    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) predictions [Green, Dead, Clover, GDM, Total]
            target: (B, 5) targets (optional, for additional consistency with targets)
        
        Returns:
            loss: Consistency loss
        """
        green, dead, clover, gdm, total = pred.unbind(dim=1)
        
        # Internal consistency
        gdm_from_parts = green + clover
        total_from_parts = green + dead + clover
        
        loss = (
            F.mse_loss(gdm, gdm_from_parts) +
            F.mse_loss(total, total_from_parts)
        )
        
        # Non-negativity penalty (predictions should be >= 0)
        non_neg_penalty = (
            F.relu(-green).mean() +
            F.relu(-dead).mean() +
            F.relu(-clover).mean()
        )
        
        return self.weight * (loss + 0.1 * non_neg_penalty)


class PlantHydraLoss(nn.Module):
    """
    Combined loss inspired by PlantTraits2024 1st place solution + CSIRO paper insights.
    
    Components:
    1. Official R² loss (log-transformed, weighted)
    2. Cosine similarity loss (from PlantHydra: maintains correlation between traits)
    3. Compositional consistency loss (GDM=G+C, Total=G+D+C)
    4. Auxiliary losses: Height, NDVI, Species classification
    
    Usage:
        loss_fn = PlantHydraLoss(
            use_log_transform=True,
            use_cosine_sim=True,
            use_compositional=True,
        )
        loss, loss_dict = loss_fn(pred, target, height_pred, height_target, ...)
    """
    
    def __init__(
        self,
        use_log_transform: bool = True,
        use_smoothl1: bool = False,
        smoothl1_beta: float = 1.0,
        use_cosine_sim: bool = True,
        cosine_weight: float = 0.4,
        use_compositional: bool = True,
        compositional_weight: float = 0.1,
        use_height_aux: bool = False,
        height_weight: float = 0.2,
        use_ndvi_aux: bool = False,
        ndvi_weight: float = 0.2,
        use_species_aux: bool = False,
        species_weight: float = 0.1,
        use_state_aux: bool = False,
        state_weight: float = 0.1,
        train_dead: bool = False,
        train_clover: bool = False,
    ) -> None:
        super().__init__()
        
        self.use_log_transform = use_log_transform
        self.use_smoothl1 = use_smoothl1
        self.smoothl1_beta = smoothl1_beta
        self.use_cosine_sim = use_cosine_sim
        self.cosine_weight = cosine_weight
        self.use_compositional = use_compositional
        self.compositional_weight = compositional_weight
        self.use_height_aux = use_height_aux
        self.height_weight = height_weight
        self.use_ndvi_aux = use_ndvi_aux
        self.ndvi_weight = ndvi_weight
        self.use_species_aux = use_species_aux
        self.species_weight = species_weight
        self.use_state_aux = use_state_aux
        self.state_weight = state_weight
        self.train_dead = train_dead
        self.train_clover = train_clover
        
        self.register_buffer("weights", TARGET_WEIGHTS)
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.log_encoder = LogTransformEncoder()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        height_pred: torch.Tensor = None,
        height_target: torch.Tensor = None,
        ndvi_pred: torch.Tensor = None,
        ndvi_target: torch.Tensor = None,
        species_logits: torch.Tensor = None,
        species_labels: torch.Tensor = None,
        state_logits: torch.Tensor = None,
        state_labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: (B, 5) biomass predictions [Green, Dead, Clover, GDM, Total]
            target: (B, 5) biomass targets
            height_pred: (B, 1) predicted height
            height_target: (B,) ground-truth Height_Ave_cm
            ndvi_pred: (B, 1) predicted NDVI
            ndvi_target: (B,) ground-truth Pre_GSHH_NDVI
            species_logits: (B, num_species) species classification logits
            species_labels: (B,) species labels
            state_logits: (B, num_states) state classification logits
            state_labels: (B,) state labels
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components for logging
        """
        loss_dict = {}
        weights = self.weights.to(pred.device)
        
        # Clamp predictions to be non-negative for log transform
        pred_clamped = pred.clamp(min=0)
        
        # 1. Log-transform both pred and target
        if self.use_log_transform:
            pred_log = self.log_encoder.transform(pred_clamped)
            target_log = self.log_encoder.transform(target)
        else:
            pred_log = pred_clamped
            target_log = target
        
        # 2. Weighted base loss (MSE or SmoothL1)
        if self.use_smoothl1:
            # SmoothL1 per target
            loss_per_target = F.smooth_l1_loss(
                pred_log, target_log, reduction='none', beta=self.smoothl1_beta
            ).mean(dim=0)  # (5,)
            loss_dict["smoothl1"] = (weights * loss_per_target).sum().item()
        else:
            # MSE per target (approximates R² optimization)
            loss_per_target = ((pred_log - target_log) ** 2).mean(dim=0)  # (5,)
            loss_dict["mse"] = (weights * loss_per_target).sum().item()
        
        weighted_loss = (weights * loss_per_target).sum()
        total_loss = weighted_loss
        
        # 3. Cosine similarity loss (from PlantHydra)
        # Maintains correlation structure between all 5 targets
        if self.use_cosine_sim:
            cos_loss = (1 - self.cos_sim(pred_log, target_log)).mean()
            total_loss = total_loss + self.cosine_weight * cos_loss
            loss_dict["cosine"] = cos_loss.item()
        
        # 4. Compositional consistency loss
        if self.use_compositional:
            green, dead, clover, gdm, total = pred_clamped.unbind(dim=1)
            gdm_check = green + clover
            total_check = green + dead + clover
            
            comp_loss = (
                F.mse_loss(gdm, gdm_check) +
                F.mse_loss(total, total_check)
            )
            total_loss = total_loss + self.compositional_weight * comp_loss
            loss_dict["compositional"] = comp_loss.item()
        
        # 5. Height auxiliary loss
        if self.use_height_aux and height_pred is not None and height_target is not None:
            # Normalize height to [0, 1] range (max ~70cm)
            height_pred_norm = height_pred.squeeze(-1) / 70.0
            height_target_norm = height_target / 70.0
            height_loss = F.mse_loss(height_pred_norm, height_target_norm)
            total_loss = total_loss + self.height_weight * height_loss
            loss_dict["height"] = height_loss.item()
        
        # 6. NDVI auxiliary loss
        if self.use_ndvi_aux and ndvi_pred is not None and ndvi_target is not None:
            ndvi_loss = F.mse_loss(ndvi_pred.squeeze(-1), ndvi_target)
            total_loss = total_loss + self.ndvi_weight * ndvi_loss
            loss_dict["ndvi"] = ndvi_loss.item()
        
        # 7. Species classification loss
        if self.use_species_aux and species_logits is not None and species_labels is not None:
            species_loss = F.cross_entropy(species_logits, species_labels, label_smoothing=0.1)
            total_loss = total_loss + self.species_weight * species_loss
            loss_dict["species"] = species_loss.item()
        
        # 8. State classification loss
        if self.use_state_aux and state_logits is not None and state_labels is not None:
            state_loss = F.cross_entropy(state_logits, state_labels, label_smoothing=0.1)
            total_loss = total_loss + self.state_weight * state_loss
            loss_dict["state"] = state_loss.item()
        
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# Cross-View Consistency Loss (Exploits Stereo)
# =============================================================================

class CrossViewConsistencyLoss(nn.Module):
    """
    Enforce consistency between left and right view predictions.
    
    The dataset has stereo pairs (2000×1000 split into two 1000×1000 halves).
    Both views should predict similar biomass values since they're the same quadrat.
    
    This regularizer:
    1. Encourages agreement in log(1+y) space
    2. Helps stabilize dead/clover predictions (most variable)
    3. Acts as self-distillation between views
    
    Usage:
        loss = CrossViewConsistencyLoss(weight=0.1)
        total_loss += loss(pred_left, pred_right)
    """
    
    def __init__(
        self, 
        weight: float = 0.1,
        use_log_space: bool = True,
        per_target_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.use_log_space = use_log_space
        if per_target_weights is not None:
            self.register_buffer("per_target_weights", per_target_weights)
        else:
            # Higher weight on unstable targets (dead, clover)
            self.register_buffer("per_target_weights", torch.tensor([0.1, 0.3, 0.3, 0.15, 0.15]))
    
    def forward(
        self, 
        pred_left: torch.Tensor, 
        pred_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_left: (B, 5) predictions from left view
            pred_right: (B, 5) predictions from right view
        
        Returns:
            consistency_loss: Weighted MSE between views in log space
        """
        if self.use_log_space:
            left_log = torch.log1p(pred_left.clamp(min=0))
            right_log = torch.log1p(pred_right.clamp(min=0))
            diff = (left_log - right_log) ** 2
        else:
            diff = (pred_left - pred_right) ** 2
        
        # Weighted per-target consistency
        weights = self.per_target_weights.to(pred_left.device)
        weighted_diff = (weights * diff).sum(dim=1)  # (B,)
        
        return self.weight * weighted_diff.mean()


# =============================================================================
# Species Prior Blending (PlantHydra-style for Biomass)
# =============================================================================

class SpeciesPriorHead(nn.Module):
    """
    PlantHydra-style species prior blending for biomass.
    
    From training data, we know mean biomass per species group (8 species).
    This head:
    1. Classifies image into species (soft probabilities)
    2. Computes expected biomass: E[y|x] = p^T @ species_means
    3. Blends with direct regression: final = α * regression + (1-α) * prior
    
    The α weights are learnable per-target (some targets benefit more from prior).
    
    Works at test time because it only needs the image for classification.
    """
    
    def __init__(
        self,
        feat_dim: int,
        num_species: int = 8,
        num_targets: int = 5,
        init_blend_weight: float = 0.7,  # Start favoring regression
    ) -> None:
        super().__init__()
        
        self.num_species = num_species
        self.num_targets = num_targets
        
        # Species classifier
        self.species_classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_species),
        )
        
        # Learnable blend weights (per target) - initialized to favor regression
        # α=0.7 means 70% regression + 30% prior
        self.blend_weights = nn.Parameter(
            torch.full((num_targets,), init_blend_weight)
        )
        
        # Species mean biomass lookup table (filled from training data)
        # Shape: (num_species, num_targets)
        self.register_buffer(
            "species_means", 
            torch.zeros(num_species, num_targets)
        )
        self.species_means_initialized = False
    
    def set_species_means(self, species_means: torch.Tensor) -> None:
        """Set the species mean biomass lookup table from training data."""
        self.species_means.copy_(species_means)
        self.species_means_initialized = True
    
    def forward(
        self, 
        features: torch.Tensor,
        regression_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, feat_dim) backbone features
            regression_pred: (B, 5) direct regression predictions
        
        Returns:
            blended_pred: (B, 5) blended predictions
            species_logits: (B, num_species) for auxiliary loss
        """
        # Species classification
        species_logits = self.species_classifier(features)  # (B, num_species)
        species_probs = F.softmax(species_logits, dim=1)  # (B, num_species)
        
        # Expected biomass from species prior: E[y|x] = p^T @ means
        # (B, num_species) @ (num_species, num_targets) -> (B, num_targets)
        prior_pred = species_probs @ self.species_means  # (B, 5)
        
        # Blend regression and prior with learnable weights
        # α controls regression weight, (1-α) controls prior weight
        alpha = torch.sigmoid(self.blend_weights)  # (5,) in [0,1]
        blended_pred = alpha * regression_pred + (1 - alpha) * prior_pred
        
        return blended_pred, species_logits


# =============================================================================
# Uncertainty-Aware Loss (Label Noise Modeling)
# =============================================================================

class UncertaintyHead(nn.Module):
    """
    Predict per-target log variance for uncertainty-aware training.
    
    Instead of just predicting y, predict (μ, log σ²) and train with Gaussian NLL.
    This helps with label noise (manual sorting, subsampling errors in dataset).
    
    The predicted uncertainty can also be used for:
    - Per-sample loss weighting (down-weight high-uncertainty samples)
    - Ensemble weighting at inference
    - Identifying difficult samples
    """
    
    def __init__(self, feat_dim: int, num_targets: int = 5) -> None:
        super().__init__()
        
        # Predict log variance for each target
        self.log_var_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_targets),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feat_dim) backbone features
        
        Returns:
            log_var: (B, 5) predicted log variance per target
        """
        return self.log_var_head(features)


class UncertaintyAwareLoss(nn.Module):
    """
    Gaussian NLL loss with heteroscedastic uncertainty.
    
    Loss = 0.5 * (log σ² + (y - μ)² / σ²)
    
    The model learns to:
    - Predict high uncertainty for noisy/difficult samples
    - Focus on samples where it's confident
    
    Applied in log(1+y) space for stability.
    """
    
    def __init__(
        self,
        use_log_space: bool = True,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0,
    ) -> None:
        super().__init__()
        self.use_log_space = use_log_space
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.register_buffer("weights", TARGET_WEIGHTS)
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_mean: (B, 5) predicted means
            pred_log_var: (B, 5) predicted log variances
            target: (B, 5) ground truth
        
        Returns:
            loss: Weighted Gaussian NLL
            loss_dict: Loss breakdown
        """
        # Transform to log space
        if self.use_log_space:
            pred_log = torch.log1p(pred_mean.clamp(min=0))
            target_log = torch.log1p(target)
        else:
            pred_log = pred_mean
            target_log = target
        
        # Clamp log variance for stability
        log_var = pred_log_var.clamp(self.min_log_var, self.max_log_var)
        
        # Gaussian NLL: 0.5 * (log σ² + (y - μ)² / σ²)
        precision = torch.exp(-log_var)  # 1/σ²
        nll = 0.5 * (log_var + precision * (target_log - pred_log) ** 2)
        
        # Weighted average
        weights = self.weights.to(pred_mean.device)
        weighted_nll = (weights * nll).sum(dim=1).mean()
        
        loss_dict = {
            "nll": weighted_nll.item(),
            "mean_uncertainty": log_var.mean().item(),
        }
        
        return weighted_nll, loss_dict


# =============================================================================
# QC-Inspired Plausibility Penalties (Domain Knowledge)
# =============================================================================

class QCPlausibilityLoss(nn.Module):
    """
    Soft constraints based on dataset QC rules and domain knowledge.
    
    From the paper:
    - QC flags weird biomass-to-height ratios
    - Very low predicted height shouldn't allow extreme total
    - Very high dead fraction shouldn't co-occur with very high NDVI proxy
    
    These are soft regularizers that penalize implausible predictions.
    """
    
    def __init__(
        self,
        weight: float = 0.1,
        max_biomass_per_cm: float = 15.0,  # Max grams per cm height
        max_dead_with_high_ndvi: float = 0.7,  # Max dead fraction when NDVI > 0.6
    ) -> None:
        super().__init__()
        self.weight = weight
        self.max_biomass_per_cm = max_biomass_per_cm
        self.max_dead_with_high_ndvi = max_dead_with_high_ndvi
    
    def forward(
        self,
        pred: torch.Tensor,
        height_pred: Optional[torch.Tensor] = None,
        ndvi_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) biomass predictions [green, dead, clover, gdm, total]
            height_pred: (B, 1) predicted height in cm (optional)
            ndvi_pred: (B, 1) predicted NDVI 0-1 (optional)
        
        Returns:
            penalty: Soft constraint violation penalty
        """
        green, dead, clover, gdm, total = pred.unbind(dim=1)
        penalty = torch.tensor(0.0, device=pred.device)
        
        # 1. Non-negativity (already in compositional, but reinforce)
        non_neg = (
            F.relu(-green).mean() +
            F.relu(-dead).mean() +
            F.relu(-clover).mean()
        )
        penalty = penalty + non_neg
        
        # 2. Fraction bounds: each component should be <= total
        fraction_penalty = (
            F.relu(green - total).mean() +
            F.relu(dead - total).mean() +
            F.relu(clover - total).mean() +
            F.relu(gdm - total).mean()
        )
        penalty = penalty + fraction_penalty
        
        # 3. Height-biomass plausibility (if height available)
        if height_pred is not None:
            height = height_pred.squeeze(-1).clamp(min=1.0)  # At least 1cm
            # Biomass per cm height should be reasonable
            biomass_per_cm = total / height
            height_penalty = F.relu(biomass_per_cm - self.max_biomass_per_cm).mean()
            penalty = penalty + height_penalty
        
        # 4. NDVI-dead consistency (if NDVI available)
        # High NDVI (>0.6) suggests lots of green, shouldn't have very high dead fraction
        if ndvi_pred is not None:
            ndvi = ndvi_pred.squeeze(-1)
            dead_fraction = dead / (total + 1e-6)
            # When NDVI > 0.6, penalize dead_fraction > 0.7
            high_ndvi_mask = (ndvi > 0.6).float()
            dead_excess = F.relu(dead_fraction - self.max_dead_with_high_ndvi)
            ndvi_dead_penalty = (high_ndvi_mask * dead_excess).mean()
            penalty = penalty + ndvi_dead_penalty
        
        return self.weight * penalty


# =============================================================================
# Physical Mass Balance Loss (Idea #2 from Dataset Paper)
# =============================================================================

class MassBalanceLoss(nn.Module):
    """
    Physical constraint: The sum of parts must equal the whole.
    
    From paper: Biomass was sorted into Green, Dead, and Legume fractions.
    This implies: Green + Dead + Clover ≈ Total (for grass+legume pastures)
    
    Neural networks often violate this: predict Total=50 but Green=40, Dead=20.
    This loss constrains predictions to be physically consistent.
    """
    
    def __init__(
        self, 
        weight: float = 0.2,
        use_gdm_balance: bool = True,  # GDM = Green + Clover
    ) -> None:
        super().__init__()
        self.weight = weight
        self.use_gdm_balance = use_gdm_balance
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) predictions [green, dead, clover, gdm, total]
            target: (B, 5) targets (optional, for adaptive weighting)
        
        Returns:
            loss: Mass balance violation penalty
        """
        green, dead, clover, gdm, total = pred.unbind(dim=1)
        
        # Primary constraint: Green + Dead + Clover ≈ Total
        pred_sum = green + dead + clover
        balance_loss = F.mse_loss(pred_sum, total)
        
        # Secondary constraint: GDM = Green + Clover (Green Dry Matter)
        if self.use_gdm_balance:
            gdm_from_parts = green + clover
            gdm_loss = F.mse_loss(gdm, gdm_from_parts)
            balance_loss = balance_loss + gdm_loss
        
        return self.weight * balance_loss


# =============================================================================
# Feature Orthogonality Loss (Idea #5: Digital Sorting / Feature Unmixing)
# =============================================================================

class FeatureOrthogonalityLoss(nn.Module):
    """
    Force "digital sorting" by making features for different targets orthogonal.
    
    Ground truth was generated by physically sorting grass vs. clover vs. dead.
    The features used to predict "Green Mass" should be distinct from those
    predicting "Dead Mass" - they look at different visual patterns.
    
    This prevents feature leakage (e.g., Green head using brown texture features).
    """
    
    def __init__(
        self, 
        weight: float = 0.1,
        pairs: List[Tuple[str, str]] = None,  # Which target pairs to orthogonalize
    ) -> None:
        super().__init__()
        self.weight = weight
        # Default: Green ⟂ Dead, Green ⟂ Clover, Dead ⟂ Clover
        self.pairs = pairs or [("green", "dead"), ("green", "clover"), ("dead", "clover")]
    
    def forward(
        self, 
        features_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            features_dict: Dictionary mapping target names to their feature vectors
                           e.g., {"green": (B, D), "dead": (B, D), "clover": (B, D)}
        
        Returns:
            loss: Sum of cosine similarities (should be minimized → orthogonal)
        """
        loss = torch.tensor(0.0, device=list(features_dict.values())[0].device)
        
        for name1, name2 in self.pairs:
            if name1 in features_dict and name2 in features_dict:
                feat1 = features_dict[name1]  # (B, D)
                feat2 = features_dict[name2]  # (B, D)
                
                # Normalize features
                feat1_norm = F.normalize(feat1, dim=1)
                feat2_norm = F.normalize(feat2, dim=1)
                
                # Cosine similarity (average over batch)
                cos_sim = (feat1_norm * feat2_norm).sum(dim=1).abs().mean()
                loss = loss + cos_sim
        
        return self.weight * loss


# =============================================================================
# Calibrated Depth Volume (Idea #4: Height-Calibrated 3D Volume)
# =============================================================================

class CalibratedDepthHead(nn.Module):
    """
    Calibrate relative depth map using ground-truth Height.
    
    Biomass ≈ Height × Coverage × Density
    
    Depth Anything gives relative depth (unknown scale). This module:
    1. Predicts a global scale factor α from image features
    2. Calibrated_Height_Map = Depth_Map × α
    3. Volume_Proxy = Sum(Calibrated_Height_Map)
    4. Uses Volume_Proxy as direct feature for biomass regression
    
    At inference, α is predicted from the image, so no GT height needed.
    """
    
    def __init__(self, feat_dim: int, depth_feat_dim: int = 256) -> None:
        super().__init__()
        
        # Predict scale factor from image features
        self.scale_predictor = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Softplus(),  # Ensure positive scale
        )
        
        # Project calibrated depth features
        self.depth_projector = nn.Sequential(
            nn.Linear(depth_feat_dim + 1, 128),  # depth features + volume proxy
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
        )
    
    def forward(
        self, 
        image_features: torch.Tensor,
        depth_features: torch.Tensor,
        depth_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: (B, feat_dim) from backbone
            depth_features: (B, depth_feat_dim) from depth encoder
            depth_map: (B, H, W) relative depth map
        
        Returns:
            calibrated_features: (B, feat_dim) depth-calibrated features
            predicted_scale: (B, 1) predicted height scale factor
        """
        # Predict scale factor
        scale = self.scale_predictor(image_features)  # (B, 1)
        
        # Apply scale to depth map
        calibrated_depth = depth_map * scale.unsqueeze(-1)  # (B, H, W)
        
        # Compute volume proxy (sum of calibrated heights)
        volume_proxy = calibrated_depth.sum(dim=(1, 2), keepdim=True)  # (B, 1)
        volume_proxy = volume_proxy / (depth_map.shape[1] * depth_map.shape[2])  # Normalize
        
        # Combine with depth features
        combined = torch.cat([depth_features, volume_proxy], dim=1)
        calibrated_features = self.depth_projector(combined)
        
        return calibrated_features, scale


class CalibratedDepthLoss(nn.Module):
    """
    Auxiliary loss to train the depth scale predictor using GT height.
    
    This is used during training only (GT height available).
    At inference, the learned scale predictor generalizes.
    """
    
    def __init__(self, weight: float = 0.2) -> None:
        super().__init__()
        self.weight = weight
    
    def forward(
        self, 
        predicted_scale: torch.Tensor,
        depth_map: torch.Tensor,
        gt_height: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_scale: (B, 1) predicted scale factor
            depth_map: (B, H, W) relative depth map
            gt_height: (B,) ground-truth Height_Ave_cm
        
        Returns:
            loss: Scale calibration loss
        """
        # Mean depth as proxy for average vegetation height
        mean_depth = depth_map.mean(dim=(1, 2))  # (B,)
        
        # Calibrated mean height
        calibrated_height = mean_depth * predicted_scale.squeeze(-1)  # (B,)
        
        # MSE with GT height
        loss = F.mse_loss(calibrated_height, gt_height)
        
        return self.weight * loss


# =============================================================================
# State/Climate Density Scaler (Idea #6: Location Bias Correction)
# =============================================================================

class StateDensityScaler(nn.Module):
    """
    Climate/Location bias correction using State embeddings.
    
    Grass in Tasmania (wet/cold) has different density-to-mass ratio than
    grass in WA (dry). Visual features alone cannot see "water weight".
    
    This module learns per-state scaling factors for the final biomass prediction.
    It specifically targets the density estimation component.
    """
    
    NUM_STATES = 4  # NSW, VIC, TAS, WA
    NUM_MONTHS = 12
    
    def __init__(
        self, 
        feat_dim: int,
        use_month: bool = True,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_month = use_month
        
        # State embeddings
        self.state_embedding = nn.Embedding(self.NUM_STATES, 32)
        
        # Optional month embeddings (seasonality affects water content)
        if use_month:
            self.month_embedding = nn.Embedding(self.NUM_MONTHS, 32)
            embed_dim = 64
        else:
            embed_dim = 32
        
        # Predict per-target density scaling factors
        self.scale_predictor = nn.Sequential(
            nn.Linear(embed_dim + feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5),  # 5 targets
        )
        
        # Initialize to output ~1.0 (no scaling initially)
        nn.init.zeros_(self.scale_predictor[-1].weight)
        nn.init.constant_(self.scale_predictor[-1].bias, init_scale)
    
    def forward(
        self, 
        features: torch.Tensor,
        state_labels: torch.Tensor,
        month_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, feat_dim) image features
            state_labels: (B,) state indices (0=NSW, 1=VIC, 2=TAS, 3=WA)
            month_labels: (B,) month indices (0-11)
        
        Returns:
            scales: (B, 5) per-target density scaling factors
        """
        # Get embeddings
        state_emb = self.state_embedding(state_labels)  # (B, 32)
        
        if self.use_month and month_labels is not None:
            month_emb = self.month_embedding(month_labels % 12)  # (B, 32)
            context_emb = torch.cat([state_emb, month_emb], dim=1)  # (B, 64)
        else:
            context_emb = state_emb
        
        # Combine with image features
        combined = torch.cat([context_emb, features], dim=1)
        
        # Predict scaling factors (use softplus to ensure positive)
        scales = F.softplus(self.scale_predictor(combined))  # (B, 5)
        
        return scales


# =============================================================================
# AOS Sensor Hallucination Loss (Idea #1: Enhanced NDVI Prediction)
# =============================================================================

class AOSHallucinationLoss(nn.Module):
    """
    Train model to "hallucinate" Active Optical Sensor (AOS) NDVI reading.
    
    The dataset includes Pre_GSHH_NDVI from a GreenSeeker sensor.
    This is NOT the same as NDVI computed from RGB pixels:
    - AOS uses active illumination (lighting-invariant)
    - AOS measures specific wavelength reflectance
    
    If the model can predict AOS NDVI from RGB, it has learned a 
    lighting-invariant "greenness" feature that correlates with green biomass.
    
    This is essentially knowledge distillation from a better sensor.
    """
    
    def __init__(
        self, 
        weight: float = 0.3,
        use_gradient_reversal: bool = False,  # Adversarial training option
    ) -> None:
        super().__init__()
        self.weight = weight
        self.use_gradient_reversal = use_gradient_reversal
    
    def forward(
        self, 
        pred_ndvi: torch.Tensor,
        gt_ndvi: torch.Tensor,
        green_pred: torch.Tensor = None,
        green_target: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_ndvi: (B, 1) or (B,) predicted NDVI
            gt_ndvi: (B,) ground-truth AOS NDVI (Pre_GSHH_NDVI)
            green_pred: (B,) optional green biomass prediction (for correlation bonus)
            green_target: (B,) optional green biomass target
        
        Returns:
            loss: NDVI hallucination loss
            loss_dict: Loss breakdown
        """
        pred_ndvi = pred_ndvi.squeeze(-1)
        
        # Main MSE loss
        mse_loss = F.mse_loss(pred_ndvi, gt_ndvi)
        
        loss_dict = {"ndvi_mse": mse_loss.item()}
        total_loss = mse_loss
        
        # Bonus: correlation between predicted NDVI and green biomass
        # This reinforces that NDVI should correlate with green
        if green_pred is not None and green_target is not None:
            # Correlation bonus: pred_ndvi should positively correlate with green
            # Using cosine similarity of centered values
            ndvi_centered = pred_ndvi - pred_ndvi.mean()
            green_centered = green_target - green_target.mean()
            
            correlation = F.cosine_similarity(
                ndvi_centered.unsqueeze(0), 
                green_centered.unsqueeze(0)
            )
            # Penalize negative correlation (NDVI should increase with green)
            corr_penalty = F.relu(-correlation)
            total_loss = total_loss + 0.1 * corr_penalty
            loss_dict["ndvi_green_corr"] = correlation.item()
        
        return self.weight * total_loss, loss_dict


# =============================================================================
# Combined Innovative Loss (All 6 Ideas)
# =============================================================================

class InnovativeBiomassLoss(nn.Module):
    """
    Combined loss incorporating all 6 innovative ideas from the dataset paper:
    
    1. AOS Sensor Hallucination (NDVI prediction)
    2. Physical Mass Balance (Green + Dead ≈ Total)
    3. Calibrated Depth Volume (height-scaled 3D)
    4. Feature Orthogonality (digital sorting)
    5. QC Plausibility Penalties
    6. State Density Scaling (implicit in forward)
    
    Plus base PlantHydra loss components.
    """
    
    def __init__(
        self,
        use_log_transform: bool = True,
        use_mass_balance: bool = True,
        mass_balance_weight: float = 0.2,
        use_aos_hallucination: bool = True,
        aos_weight: float = 0.3,
        use_qc_plausibility: bool = True,
        qc_weight: float = 0.1,
        use_calibrated_depth: bool = False,
        depth_weight: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.use_log_transform = use_log_transform
        self.register_buffer("weights", TARGET_WEIGHTS)
        
        # Component losses
        self.mass_balance = MassBalanceLoss(weight=mass_balance_weight) if use_mass_balance else None
        self.aos_loss = AOSHallucinationLoss(weight=aos_weight) if use_aos_hallucination else None
        self.qc_loss = QCPlausibilityLoss(weight=qc_weight) if use_qc_plausibility else None
        self.depth_loss = CalibratedDepthLoss(weight=depth_weight) if use_calibrated_depth else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ndvi_pred: torch.Tensor = None,
        ndvi_target: torch.Tensor = None,
        height_pred: torch.Tensor = None,
        height_target: torch.Tensor = None,
        depth_scale: torch.Tensor = None,
        depth_map: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: (B, 5) biomass predictions
            target: (B, 5) biomass targets
            ndvi_pred: (B, 1) predicted NDVI
            ndvi_target: (B,) ground-truth AOS NDVI
            height_pred: (B, 1) predicted height
            height_target: (B,) ground-truth height
            depth_scale: (B, 1) predicted depth scale
            depth_map: (B, H, W) relative depth map
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual components
        """
        loss_dict = {}
        weights = self.weights.to(pred.device)
        
        # Clamp predictions
        pred_clamped = pred.clamp(min=0)
        
        # Base MSE loss (log-transformed)
        if self.use_log_transform:
            pred_log = torch.log1p(pred_clamped)
            target_log = torch.log1p(target)
        else:
            pred_log = pred_clamped
            target_log = target
        
        mse_per_target = ((pred_log - target_log) ** 2).mean(dim=0)
        base_loss = (weights * mse_per_target).sum()
        loss_dict["base_mse"] = base_loss.item()
        total_loss = base_loss
        
        # 1. AOS Hallucination Loss
        if self.aos_loss is not None and ndvi_pred is not None and ndvi_target is not None:
            green_t = target[:, 0]
            aos_loss, aos_dict = self.aos_loss(ndvi_pred, ndvi_target, pred_clamped[:, 0], green_t)
            total_loss = total_loss + aos_loss
            loss_dict.update(aos_dict)
        
        # 2. Mass Balance Loss
        if self.mass_balance is not None:
            balance_loss = self.mass_balance(pred_clamped)
            total_loss = total_loss + balance_loss
            loss_dict["mass_balance"] = balance_loss.item()
        
        # 3. Calibrated Depth Loss
        if self.depth_loss is not None and depth_scale is not None and depth_map is not None and height_target is not None:
            depth_loss = self.depth_loss(depth_scale, depth_map, height_target)
            total_loss = total_loss + depth_loss
            loss_dict["depth_cal"] = depth_loss.item()
        
        # 4. QC Plausibility Loss
        if self.qc_loss is not None:
            qc_loss = self.qc_loss(pred_clamped, height_pred, ndvi_pred)
            total_loss = total_loss + qc_loss
            loss_dict["qc_plausibility"] = qc_loss.item()
        
        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


# =============================================================================
# Homography Jitter Augmentation (for warp/annotation noise)
# =============================================================================

def random_perspective_transform(
    img: torch.Tensor,
    max_warp: float = 0.05,
    border_jitter: int = 5,
) -> torch.Tensor:
    """
    Apply random perspective warp + border jitter to simulate annotation noise.
    
    The dataset has manual corner annotations + affine/perspective normalization.
    This augmentation simulates imperfect corner placement and warp errors.
    
    Args:
        img: (C, H, W) image tensor
        max_warp: Maximum perspective distortion as fraction of image size
        border_jitter: Random crop/pad at borders in pixels
    
    Returns:
        Warped image tensor
    """
    C, H, W = img.shape
    device = img.device
    
    # Generate random perspective transformation
    # 4 corners with small random offsets
    src_pts = torch.tensor([
        [0, 0], [W-1, 0], [W-1, H-1], [0, H-1]
    ], dtype=torch.float32, device=device)
    
    # Random offsets for each corner
    offsets = (torch.rand(4, 2, device=device) - 0.5) * 2 * max_warp
    offsets[:, 0] *= W
    offsets[:, 1] *= H
    
    dst_pts = src_pts + offsets
    
    # Compute perspective transform matrix
    # This is a simplified version - for full implementation use kornia
    # Here we just do affine approximation
    
    # Border jitter: random crop
    if border_jitter > 0:
        crop_top = torch.randint(0, border_jitter + 1, (1,)).item()
        crop_left = torch.randint(0, border_jitter + 1, (1,)).item()
        crop_bottom = torch.randint(0, border_jitter + 1, (1,)).item()
        crop_right = torch.randint(0, border_jitter + 1, (1,)).item()
        
        img = img[:, crop_top:H-crop_bottom, crop_left:W-crop_right]
        
        # Resize back to original size
        img = F.interpolate(
            img.unsqueeze(0), 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    return img


class BiomassLoss(nn.Module):
    """
    Weighted loss for biomass prediction.
    
    Two modes:
    1. Default (MSE): MSE on 4 targets (Green, Clover, GDM, Total) - Dead excluded when derived
    2. SmoothL1 mode: SmoothL1 on 3-4 targets (Total, GDM, Green, +Clover if train_clover)
    
    Competition weights: Green=0.1, Dead=0.1, Clover=0.1, GDM=0.2, Total=0.5
    
    Key changes:
    - MSE mode: Dead is EXCLUDED from loss when derived (train_dead=False)
    - SmoothL1 mode: Clover is INCLUDED when train_clover=True
    """
    
    def __init__(
        self,
        use_huber_for_dead: bool = True,
        huber_delta: float = 5.0,
        train_dead: bool = False,
        train_clover: bool = False,
        derived_loss_scale: float = 0.5,
        smoothl1_mode: bool = False,
    ) -> None:
        super().__init__()
        self.use_huber_for_dead = use_huber_for_dead
        self.huber_delta = huber_delta
        self.train_dead = train_dead
        self.train_clover = train_clover
        self.derived_loss_scale = derived_loss_scale
        self.smoothl1_mode = smoothl1_mode
        
        if smoothl1_mode:
            # SmoothL1 on 3-4 targets: total=0.5, gdm=0.2, green=0.1, +clover=0.1 if trained
            w_green = 0.1
            w_clover = 0.1 if train_clover else 0.0  # Include clover if trained
            w_gdm = 0.2
            w_total = 0.5
            total_w = w_green + w_clover + w_gdm + w_total
            
            weights = torch.tensor([
                w_green / total_w,   # green
                0.0,                 # dead (never used in smoothl1)
                w_clover / total_w,  # clover (if trained)
                w_gdm / total_w,     # gdm
                w_total / total_w,   # total
            ], dtype=torch.float32)
        else:
            # MSE mode: exclude Dead when derived, include others
            w_green = COMPETITION_WEIGHTS["green"]
            w_dead = COMPETITION_WEIGHTS["dead"] if train_dead else 0.0  # EXCLUDE Dead when derived
            w_clover = COMPETITION_WEIGHTS["clover"]
            w_gdm = COMPETITION_WEIGHTS["gdm"]
            w_total = COMPETITION_WEIGHTS["total"]
            
            # Scale down Clover if derived (but keep it in loss unlike Dead)
            if not train_clover:
                w_clover *= derived_loss_scale
            
            # Normalize weights to sum to 1
            total_w = w_green + w_dead + w_clover + w_gdm + w_total
            weights = torch.tensor([
                w_green / total_w,
                w_dead / total_w,
                w_clover / total_w,
                w_gdm / total_w,
                w_total / total_w,
            ], dtype=torch.float32)
        
        self.register_buffer("weights", weights)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) predictions [green, dead, clover, gdm, total]
            target: (B, 5) ground truth [green, dead, clover, gdm, total]
        """
        green_p, dead_p, clover_p, gdm_p, total_p = pred.unbind(dim=1)
        green_t, dead_t, clover_t, gdm_t, total_t = target.unbind(dim=1)
        
        w = self.weights
        
        if self.smoothl1_mode:
            # SmoothL1Loss on 3-4 targets (Total, GDM, Green, +Clover if trained)
            loss_green = F.smooth_l1_loss(green_p, green_t)
            loss_gdm = F.smooth_l1_loss(gdm_p, gdm_t)
            loss_total = F.smooth_l1_loss(total_p, total_t)
            
            loss = (
                w[0] * loss_green +
                w[3] * loss_gdm +
                w[4] * loss_total
            )
            
            # Add clover loss if trained
            if self.train_clover:
                loss_clover = F.smooth_l1_loss(clover_p, clover_t)
                loss = loss + w[2] * loss_clover
        else:
            # MSE mode: exclude Dead when derived
            loss_green = F.mse_loss(green_p, green_t)
            loss_clover = F.mse_loss(clover_p, clover_t)
            loss_gdm = F.mse_loss(gdm_p, gdm_t)
            loss_total = F.mse_loss(total_p, total_t)
            
            loss = (
                w[0] * loss_green +
                w[2] * loss_clover +
                w[3] * loss_gdm +
                w[4] * loss_total
            )
            
            # Only add Dead loss if trained (has its own head)
            if self.train_dead:
                if self.use_huber_for_dead:
                    loss_dead = F.huber_loss(dead_p, dead_t, delta=self.huber_delta)
                else:
                    loss_dead = F.mse_loss(dead_p, dead_t)
                loss = loss + w[1] * loss_dead
        
        return loss


class AuxiliaryBiomassLoss(nn.Module):
    """
    Combined loss for biomass regression + auxiliary classification tasks.
    
    The auxiliary heads (State, Month, Species) help the backbone learn discriminative features.
    Based on analysis: State+Month explains 34% of Dead variance, Species helps with Clover.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        state_weight: float = 1.0,
        month_weight: float = 1.0,
        species_weight: float = 1.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.state_weight = state_weight
        self.month_weight = month_weight
        self.species_weight = species_weight
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        pred_biomass: torch.Tensor,
        target_biomass: torch.Tensor,
        state_logits: torch.Tensor,
        state_labels: torch.Tensor,
        month_logits: torch.Tensor,
        month_labels: torch.Tensor,
        species_logits: torch.Tensor,
        species_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_biomass: (B, 5) biomass predictions
            target_biomass: (B, 5) biomass targets
            state_logits: (B, NUM_STATES) state classification logits
            state_labels: (B,) state labels (0-3)
            month_logits: (B, NUM_MONTHS) month classification logits
            month_labels: (B,) month labels (0-9)
            species_logits: (B, NUM_SPECIES) species classification logits
            species_labels: (B,) species labels (0-7)
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components for logging
        """
        # Base biomass loss
        biomass_loss = self.base_loss(pred_biomass, target_biomass)
        
        # Auxiliary classification losses with label smoothing
        state_loss = F.cross_entropy(
            state_logits, state_labels, label_smoothing=self.label_smoothing
        )
        month_loss = F.cross_entropy(
            month_logits, month_labels, label_smoothing=self.label_smoothing
        )
        species_loss = F.cross_entropy(
            species_logits, species_labels, label_smoothing=self.label_smoothing
        )
        
        # Weighted sum
        aux_loss = (
            self.state_weight * state_loss +
            self.month_weight * month_loss +
            self.species_weight * species_loss
        )
        
        total_loss = biomass_loss + aux_loss
        
        # Build loss dict for logging
        loss_dict = {
            "biomass": biomass_loss.item(),
            "state": state_loss.item(),
            "month": month_loss.item(),
            "species": species_loss.item(),
            "aux_total": aux_loss.item(),
        }
        
        return total_loss, loss_dict


def tweedie_loss(pred: torch.Tensor, target: torch.Tensor, p: float = 1.5) -> torch.Tensor:
    """
    Tweedie loss for zero-inflated continuous targets.
    
    Good for targets with exact zeros (like Dead, Clover).
    p=1.5 is a good default (between Poisson p=1 and Gamma p=2).
    
    Args:
        pred: Predicted values (must be > 0, use softplus output)
        target: Ground truth values (>= 0)
        p: Tweedie power parameter (1 < p < 2 for compound Poisson-Gamma)
    
    Returns:
        loss: Mean Tweedie deviance
    """
    # Ensure pred is positive
    pred = pred.clamp(min=1e-6)
    
    # Tweedie deviance: 2 * (y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p))
    # Simplified form for loss (constant terms removed):
    loss = -target * torch.pow(pred, 1 - p) / (1 - p) + torch.pow(pred, 2 - p) / (2 - p)
    
    return loss.mean()


class PresenceNDVILoss(nn.Module):
    """
    Extended loss with presence classification, NDVI prediction, Height prediction,
    Species classification, and Tweedie for Dead/Clover.
    
    Components:
    1. Base biomass loss (MSE/SmoothL1)
    2. Presence classification loss (BCE) for Dead/Clover
    3. NDVI regression loss (MSE) - uses ground-truth Pre_GSHH_NDVI
    4. Height regression loss (MSE) - uses ground-truth Height_Ave_cm
    5. Species classification loss (CrossEntropy)
    6. Optional Tweedie loss for Dead/Clover instead of MSE
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        use_presence: bool = False,
        use_ndvi: bool = False,
        use_height: bool = False,
        use_species: bool = False,
        use_tweedie: bool = False,
        tweedie_p: float = 1.5,
        presence_weight: float = 0.5,
        ndvi_weight: float = 0.3,
        height_weight: float = 0.3,
        species_weight: float = 0.5,
        presence_threshold: float = 0.5,  # Threshold for creating binary labels
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.use_presence = use_presence
        self.use_ndvi = use_ndvi
        self.use_height = use_height
        self.use_species = use_species
        self.use_tweedie = use_tweedie
        self.tweedie_p = tweedie_p
        self.presence_weight = presence_weight
        self.ndvi_weight = ndvi_weight
        self.height_weight = height_weight
        self.species_weight = species_weight
        self.presence_threshold = presence_threshold
    
    def forward(
        self,
        pred_biomass: torch.Tensor,
        target_biomass: torch.Tensor,
        dead_presence_logit: torch.Tensor = None,
        clover_presence_logit: torch.Tensor = None,
        ndvi_pred: torch.Tensor = None,
        ndvi_target: torch.Tensor = None,
        height_pred: torch.Tensor = None,
        height_target: torch.Tensor = None,
        species_logits: torch.Tensor = None,
        species_labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_biomass: (B, 5) biomass predictions
            target_biomass: (B, 5) biomass targets
            dead_presence_logit: (B, 1) dead presence logits
            clover_presence_logit: (B, 1) clover presence logits
            ndvi_pred: (B, 1) NDVI predictions
            ndvi_target: (B, 1) NDVI targets (computed from image)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components for logging
        """
        loss_dict = {}
        
        # Base biomass loss
        if self.use_tweedie:
            # Use Tweedie for Dead and Clover
            green_p, dead_p, clover_p, gdm_p, total_p = pred_biomass.unbind(dim=1)
            green_t, dead_t, clover_t, gdm_t, total_t = target_biomass.unbind(dim=1)
            
            # Standard losses for Green, GDM, Total
            loss_green = F.mse_loss(green_p, green_t)
            loss_gdm = F.mse_loss(gdm_p, gdm_t)
            loss_total = F.mse_loss(total_p, total_t)
            
            # Tweedie for Dead and Clover (zero-inflated)
            loss_dead = tweedie_loss(dead_p.clamp(min=0) + 1e-6, dead_t, p=self.tweedie_p)
            loss_clover = tweedie_loss(clover_p.clamp(min=0) + 1e-6, clover_t, p=self.tweedie_p)
            
            # Weighted sum (competition weights)
            biomass_loss = (
                0.1 * loss_green +
                0.1 * loss_dead +
                0.1 * loss_clover +
                0.2 * loss_gdm +
                0.5 * loss_total
            )
            loss_dict["green"] = loss_green.item()
            loss_dict["dead"] = loss_dead.item()
            loss_dict["clover"] = loss_clover.item()
            loss_dict["gdm"] = loss_gdm.item()
            loss_dict["total"] = loss_total.item()
        else:
            biomass_loss = self.base_loss(pred_biomass, target_biomass)
        
        loss_dict["biomass"] = biomass_loss.item()
        total_loss = biomass_loss
        
        # Presence classification loss
        if self.use_presence and dead_presence_logit is not None:
            # Create binary labels from targets
            _, dead_t, clover_t, _, _ = target_biomass.unbind(dim=1)
            dead_label = (dead_t > self.presence_threshold).float().unsqueeze(1)
            clover_label = (clover_t > self.presence_threshold).float().unsqueeze(1)
            
            # BCE loss
            dead_presence_loss = F.binary_cross_entropy_with_logits(
                dead_presence_logit, dead_label
            )
            clover_presence_loss = F.binary_cross_entropy_with_logits(
                clover_presence_logit, clover_label
            )
            presence_loss = (dead_presence_loss + clover_presence_loss) / 2
            
            total_loss = total_loss + self.presence_weight * presence_loss
            loss_dict["dead_presence"] = dead_presence_loss.item()
            loss_dict["clover_presence"] = clover_presence_loss.item()
        
        # NDVI regression loss (ground-truth Pre_GSHH_NDVI, range 0-1)
        if self.use_ndvi and ndvi_pred is not None and ndvi_target is not None:
            ndvi_loss = F.mse_loss(ndvi_pred.squeeze(-1), ndvi_target)
            total_loss = total_loss + self.ndvi_weight * ndvi_loss
            loss_dict["ndvi"] = ndvi_loss.item()
        
        # Height regression loss (ground-truth Height_Ave_cm, range 1-70)
        # Normalize height to [0, 1] range for stable training
        if self.use_height and height_pred is not None and height_target is not None:
            # Normalize to similar scale as other losses
            height_pred_norm = height_pred.squeeze(-1) / 70.0  # Max ~70cm
            height_target_norm = height_target / 70.0
            height_loss = F.mse_loss(height_pred_norm, height_target_norm)
            total_loss = total_loss + self.height_weight * height_loss
            loss_dict["height"] = height_loss.item()
        
        # Species classification loss
        if self.use_species and species_logits is not None and species_labels is not None:
            species_loss = F.cross_entropy(species_logits, species_labels, label_smoothing=0.1)
            total_loss = total_loss + self.species_weight * species_loss
            loss_dict["species"] = species_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# Utility Functions
# =============================================================================

def freeze_backbone(model: DINOv3Direct) -> None:
    """Freeze backbone parameters (head-only training)."""
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False


def unfreeze_backbone(model: DINOv3Direct) -> None:
    """Unfreeze backbone parameters."""
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = True


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

