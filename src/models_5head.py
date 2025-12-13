"""5-Head Model Architecture for CSIRO Biomass prediction.

Optimized version with FiLM conditioning, attention pooling, and deeper heads.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import _build_dino_by_name, _make_edges


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        hidden = max(64, in_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim * 2),
        )
    
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            (B, D) - attention-weighted pooling
        """
        # Global query from mean
        q = self.query(x.mean(dim=1, keepdim=True))  # (B, 1, D)
        k = self.key(x)  # (B, T, D)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, 1, T)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ x).squeeze(1)  # (B, D)
        return out


class FiveHeadDINO(nn.Module):
    """
    Optimized 5-Head DINOv2 model with FiLM and attention pooling.
    
    Key improvements over basic version:
    - FiLM conditioning for left/right stream interaction
    - Attention-based tile pooling instead of mean
    - Deeper heads with LayerNorm and residual connections
    - Shared feature projection before target-specific heads
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        grid: Tuple[int, int] = (2, 2),
        pretrained: bool = True,
        dropout: float = 0.2,
        hidden_ratio: float = 0.5,
        use_film: bool = True,
        use_attention_pool: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        
        self.backbone, feat_dim, input_res = _build_dino_by_name(
            backbone_name, pretrained, gradient_checkpointing
        )
        self.used_backbone_name = backbone_name
        self.input_res = int(input_res)
        self.feat_dim = feat_dim
        self.grid = tuple(grid)
        self.use_film = use_film
        self.use_attention_pool = use_attention_pool
        
        # FiLM for left-right stream conditioning
        if use_film:
            self.film_left = FiLM(feat_dim)
            self.film_right = FiLM(feat_dim)
        
        # Attention pooling for tiles
        if use_attention_pool:
            self.attn_pool_left = AttentionPooling(feat_dim)
            self.attn_pool_right = AttentionPooling(feat_dim)
        
        # Combined features from left + right streams
        self.combined_dim = feat_dim * 2
        hidden_dim = max(64, int(self.combined_dim * hidden_ratio))
        
        # Shared feature projection with LayerNorm
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(self.combined_dim),
            nn.Linear(self.combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        def _make_head(in_dim: int) -> nn.Sequential:
            """Create a deeper head with residual-like structure."""
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(in_dim, 1),
            )
        
        # 5 independent heads
        self.head_green = _make_head(hidden_dim)
        self.head_dead = _make_head(hidden_dim)
        self.head_clover = _make_head(hidden_dim)
        self.head_gdm = _make_head(hidden_dim)
        self.head_total = _make_head(hidden_dim)
        
        self.softplus = nn.Softplus(beta=1.0)
        
        # Initialize heads with small weights for stable training
        self._init_heads()
    
    def _init_heads(self) -> None:
        """Initialize head weights for stable training."""
        for head in [self.head_green, self.head_dead, self.head_clover, 
                     self.head_gdm, self.head_total]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _extract_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Extract tile features without pooling."""
        B, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)
        
        feats = []
        for rs, re in rows:
            for cs, ce in cols:
                xt = x[:, :, rs:re, cs:ce]
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(
                        xt,
                        size=(self.input_res, self.input_res),
                        mode="bilinear",
                        align_corners=False,
                    )
                ft = self.backbone(xt)
                feats.append(ft)
        
        feats = torch.stack(feats, dim=0).permute(1, 0, 2)  # (B, num_tiles, D)
        return feats
    
    def _encode_stream(
        self, 
        x: torch.Tensor, 
        film: FiLM | None, 
        attn_pool: AttentionPooling | None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a single stream with optional FiLM and attention pooling."""
        tiles = self._extract_tiles(x)  # (B, num_tiles, D)
        
        # Apply FiLM conditioning if available
        if film is not None and context is not None:
            gamma, beta = film(context)
            tiles = tiles * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        
        # Pool tiles
        if attn_pool is not None:
            feat = attn_pool(tiles)
        else:
            feat = tiles.mean(dim=1)
        
        return feat
    
    def forward(
        self, x_left: torch.Tensor, x_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with FiLM cross-conditioning."""
        # Extract tile features first for context
        tiles_left = self._extract_tiles(x_left)
        tiles_right = self._extract_tiles(x_right)
        
        # Get initial context (mean of tiles)
        ctx_left = tiles_left.mean(dim=1)
        ctx_right = tiles_right.mean(dim=1)
        
        # Apply FiLM cross-conditioning
        if self.use_film:
            gamma_l, beta_l = self.film_left(ctx_right)  # Right conditions left
            gamma_r, beta_r = self.film_right(ctx_left)  # Left conditions right
            tiles_left = tiles_left * (1 + gamma_l.unsqueeze(1)) + beta_l.unsqueeze(1)
            tiles_right = tiles_right * (1 + gamma_r.unsqueeze(1)) + beta_r.unsqueeze(1)
        
        # Pool tiles
        if self.use_attention_pool:
            f_l = self.attn_pool_left(tiles_left)
            f_r = self.attn_pool_right(tiles_right)
        else:
            f_l = tiles_left.mean(dim=1)
            f_r = tiles_right.mean(dim=1)
        
        # Combine and project
        f = torch.cat([f_l, f_r], dim=1)
        f = self.shared_proj(f)
        
        # All 5 predictions are independent
        green = self.softplus(self.head_green(f))
        dead = self.softplus(self.head_dead(f))
        clover = self.softplus(self.head_clover(f))
        gdm = self.softplus(self.head_gdm(f))
        total = self.softplus(self.head_total(f))
        
        return green, dead, clover, gdm, total


class ConstrainedMSELoss(nn.Module):
    """
    Optimized loss with consistency constraints and optional target-specific handling.
    
    Key improvements:
    - Huber loss option for dead target (handles outliers better)
    - Simplified constraint (no ordering penalties - they add noise)
    - Option to weight base targets more heavily during training
    """
    
    def __init__(
        self,
        target_weights: List[float] = [0.2, 0.2, 0.2, 0.2, 0.2],  # Equal by default
        constraint_weight: float = 0.05,
        use_huber_for_dead: bool = True,
        huber_delta: float = 5.0,
    ) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(target_weights, dtype=torch.float32))
        self.constraint_weight = constraint_weight
        self.use_huber_for_dead = use_huber_for_dead
        self.huber_delta = huber_delta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) - green, dead, clover, gdm, total
            target: (B, 5) - ground truth
        """
        weights = self.weights.to(pred.device)
        B = pred.size(0)
        
        # Per-target losses
        green_pred, dead_pred, clover_pred, gdm_pred, total_pred = pred.unbind(dim=1)
        green_gt, dead_gt, clover_gt, gdm_gt, total_gt = target.unbind(dim=1)
        
        # MSE for most targets
        loss_green = F.mse_loss(green_pred, green_gt, reduction='mean')
        loss_clover = F.mse_loss(clover_pred, clover_gt, reduction='mean')
        loss_gdm = F.mse_loss(gdm_pred, gdm_gt, reduction='mean')
        loss_total = F.mse_loss(total_pred, total_gt, reduction='mean')
        
        # Huber loss for dead (more robust to outliers/sparse values)
        if self.use_huber_for_dead:
            loss_dead = F.huber_loss(dead_pred, dead_gt, delta=self.huber_delta, reduction='mean')
        else:
            loss_dead = F.mse_loss(dead_pred, dead_gt, reduction='mean')
        
        # Weighted combination
        main_loss = (
            weights[0] * loss_green +
            weights[1] * loss_dead +
            weights[2] * loss_clover +
            weights[3] * loss_gdm +
            weights[4] * loss_total
        ) / weights.sum()
        
        # Soft consistency constraints (simplified - no ordering penalties)
        if self.constraint_weight > 0:
            # GDM should be close to green + clover
            gdm_consistency = F.mse_loss(gdm_pred, green_pred + clover_pred)
            # Total should be close to gdm + dead
            total_consistency = F.mse_loss(total_pred, gdm_pred + dead_pred)
            
            constraint_loss = gdm_consistency + total_consistency
            return main_loss + self.constraint_weight * constraint_loss
        
        return main_loss


class FocalMSELoss(nn.Module):
    """
    Focal-style MSE that focuses on harder samples.
    
    Higher errors get more weight, helping with difficult predictions.
    """
    
    def __init__(
        self,
        target_weights: List[float] = [0.2, 0.2, 0.2, 0.2, 0.2],
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(target_weights, dtype=torch.float32))
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(pred.device)
        
        # Per-sample, per-target squared errors
        sq_error = (pred - target) ** 2  # (B, 5)
        
        # Focal weighting: higher errors get more weight
        # Normalize errors to [0, 1] range for stable focal weighting
        with torch.no_grad():
            max_err = sq_error.max() + 1e-8
            focal_weight = (sq_error / max_err) ** self.gamma
            focal_weight = focal_weight + 0.1  # Minimum weight
        
        weighted_error = sq_error * focal_weight * weights.unsqueeze(0)
        return weighted_error.sum() / (pred.size(0) * weights.sum())


class DeadAwareLoss(nn.Module):
    """
    Loss function with special handling for Dead target.
    
    Strategies for improving Dead prediction:
    1. Log-space loss for Dead (handles sparse/skewed distribution)
    2. Auxiliary loss: Dead should equal Total - GDM
    3. Higher weight for Dead
    """
    
    def __init__(
        self,
        target_weights: List[float] = [0.15, 0.35, 0.15, 0.15, 0.2],
        use_log_for_dead: bool = True,
        aux_dead_weight: float = 0.2,
        huber_delta: float = 3.0,
    ) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(target_weights, dtype=torch.float32))
        self.use_log_for_dead = use_log_for_dead
        self.aux_dead_weight = aux_dead_weight
        self.huber_delta = huber_delta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) - green, dead, clover, gdm, total
            target: (B, 5) - ground truth
        """
        weights = self.weights.to(pred.device)
        
        green_pred, dead_pred, clover_pred, gdm_pred, total_pred = pred.unbind(dim=1)
        green_gt, dead_gt, clover_gt, gdm_gt, total_gt = target.unbind(dim=1)
        
        # Standard MSE for Green, Clover, GDM, Total
        loss_green = F.mse_loss(green_pred, green_gt, reduction='mean')
        loss_clover = F.mse_loss(clover_pred, clover_gt, reduction='mean')
        loss_gdm = F.mse_loss(gdm_pred, gdm_gt, reduction='mean')
        loss_total = F.mse_loss(total_pred, total_gt, reduction='mean')
        
        # Special handling for Dead
        if self.use_log_for_dead:
            # Log-space loss: better for sparse/low values
            # log1p is safe for values >= 0
            dead_pred_log = torch.log1p(dead_pred)
            dead_gt_log = torch.log1p(dead_gt)
            loss_dead = F.huber_loss(dead_pred_log, dead_gt_log, delta=1.0, reduction='mean')
        else:
            loss_dead = F.huber_loss(dead_pred, dead_gt, delta=self.huber_delta, reduction='mean')
        
        # Main weighted loss
        main_loss = (
            weights[0] * loss_green +
            weights[1] * loss_dead +
            weights[2] * loss_clover +
            weights[3] * loss_gdm +
            weights[4] * loss_total
        ) / weights.sum()
        
        # Auxiliary loss: Dead should equal Total - GDM
        # This teaches the model the structural relationship
        if self.aux_dead_weight > 0:
            derived_dead = F.relu(total_pred - gdm_pred)  # Dead = Total - GDM
            aux_loss = F.mse_loss(dead_pred, derived_dead.detach())  # Detach to not backprop through derived
            # Also: encourage Total = GDM + Dead
            consistency_loss = F.mse_loss(total_pred, gdm_pred + dead_pred)
            
            return main_loss + self.aux_dead_weight * (aux_loss + consistency_loss)
        
        return main_loss


class DeadPostProcessor:
    """
    Post-processor that corrects dead predictions.
    
    Strategy: Trust total and gdm (higher weights, more stable),
    derive dead = total - gdm when predictions are inconsistent.
    """
    
    def __init__(
        self,
        correction_threshold: float = 0.15,
        always_correct: bool = False,
    ) -> None:
        """
        Args:
            correction_threshold: Relative error threshold to trigger correction
            always_correct: If True, always derive dead from total - gdm
        """
        self.threshold = correction_threshold
        self.always_correct = always_correct
    
    def __call__(
        self,
        green: torch.Tensor,
        dead: torch.Tensor,
        clover: torch.Tensor,
        gdm: torch.Tensor,
        total: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Post-process predictions.
        
        Args:
            All tensors of shape (B,) or (B, 1)
        
        Returns:
            Corrected predictions
        """
        # Flatten if needed
        green = green.squeeze(-1) if green.dim() > 1 else green
        dead = dead.squeeze(-1) if dead.dim() > 1 else dead
        clover = clover.squeeze(-1) if clover.dim() > 1 else clover
        gdm = gdm.squeeze(-1) if gdm.dim() > 1 else gdm
        total = total.squeeze(-1) if total.dim() > 1 else total
        
        if self.always_correct:
            # Always derive dead from total - gdm
            dead_corrected = F.relu(total - gdm)
            return green, dead_corrected, clover, gdm, total
        
        # Check consistency
        expected_total = green + dead + clover
        total_error = torch.abs(total - expected_total) / (total + 1e-8)
        
        # Correct dead where error is above threshold
        needs_correction = total_error > self.threshold
        
        if needs_correction.any():
            dead_corrected = dead.clone()
            dead_corrected[needs_correction] = F.relu(
                total[needs_correction] - gdm[needs_correction]
            )
            return green, dead_corrected, clover, gdm, total
        
        return green, dead, clover, gdm, total


def build_5head_model(
    backbone_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
    grid: Tuple[int, int] = (2, 2),
    pretrained: bool = True,
    dropout: float = 0.2,
    hidden_ratio: float = 0.5,
    use_film: bool = True,
    use_attention_pool: bool = True,
    gradient_checkpointing: bool = False,
) -> FiveHeadDINO:
    """
    Build optimized 5-head model.
    
    Args:
        backbone_name: DINOv2 backbone (base recommended for accuracy)
        grid: Tile grid for spatial processing
        pretrained: Use pretrained weights
        dropout: Dropout rate (lower for more capacity)
        hidden_ratio: Hidden layer size relative to combined dim
        use_film: Enable FiLM cross-stream conditioning
        use_attention_pool: Use attention pooling instead of mean
        gradient_checkpointing: Enable for memory efficiency
    """
    return FiveHeadDINO(
        backbone_name=backbone_name,
        grid=grid,
        pretrained=pretrained,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
        use_film=use_film,
        use_attention_pool=use_attention_pool,
        gradient_checkpointing=gradient_checkpointing,
    )

