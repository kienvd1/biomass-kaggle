"""Softmax Ratio Model Architecture for CSIRO Biomass prediction.

Key insight: Since Green + Dead + Clover = Total, we can predict:
1. Total (absolute biomass) - strongest visual signal
2. Softmax ratios for (Green, Dead, Clover) that sum to 1

Then: component = Total × ratio

This guarantees mathematical consistency: Green + Dead + Clover = Total (always!)
"""
from typing import Dict, List, Optional, Tuple

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
        q = self.query(x.mean(dim=1, keepdim=True))
        k = self.key(x)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ x).squeeze(1)
        return out


class SoftmaxRatioDINO(nn.Module):
    """
    Softmax Ratio Model: Predict Total + component ratios.
    
    Architecture:
    - Predicts Total biomass (absolute value) - strong visual signal
    - Predicts 3 logits -> softmax -> ratios (green_r, dead_r, clover_r)
    - Components = Total × ratio (guaranteed to sum to Total)
    - GDM = Green + Clover (exact)
    
    Advantages:
    - Green + Dead + Clover = Total (always, by construction)
    - No negative predictions possible
    - Model learns proportions (bounded [0,1]) which may be easier than absolute values
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
        ratio_temperature: float = 1.0,
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
        self.ratio_temperature = ratio_temperature
        
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
        self.hidden_dim = hidden_dim
        
        # Shared feature projection
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(self.combined_dim),
            nn.Linear(self.combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        def _make_head(in_dim: int, out_dim: int = 1) -> nn.Sequential:
            """Create a head with residual-like structure."""
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(in_dim, out_dim),
            )
        
        # Head for Total biomass (absolute value)
        self.head_total = _make_head(hidden_dim, 1)
        
        # Head for component ratios (3 logits -> softmax)
        # Output: (green_ratio, dead_ratio, clover_ratio)
        self.head_ratios = _make_head(hidden_dim, 3)
        
        self.softplus = nn.Softplus(beta=1.0)
        
        self._init_heads()
    
    def _init_heads(self) -> None:
        """Initialize head weights for stable training."""
        for head in [self.head_total, self.head_ratios]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Initialize ratio head bias based on data statistics
        # Mean ratios: green=0.54, dead=0.28, clover=0.18
        # Softmax inv: logit = log(ratio) - log(1/3) ≈ log(ratio) + 1.1
        with torch.no_grad():
            self.head_ratios[-1].bias.copy_(torch.tensor([0.5, 0.0, -0.5]))
    
    def _collect_tiles(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Collect tiles from image."""
        _, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)

        tiles = []
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
                tiles.append(xt)
        return tiles

    def _extract_tiles_fused(
        self, x_left: torch.Tensor, x_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract tile features from BOTH streams in ONE backbone call."""
        B = x_left.size(0)

        tiles_left = self._collect_tiles(x_left)
        tiles_right = self._collect_tiles(x_right)
        num_tiles = len(tiles_left)

        all_tiles = torch.cat(tiles_left + tiles_right, dim=0)
        all_feats = self.backbone(all_tiles)

        total_tiles = 2 * num_tiles
        all_feats = all_feats.view(total_tiles, B, -1).permute(1, 0, 2)
        feats_left = all_feats[:, :num_tiles, :]
        feats_right = all_feats[:, num_tiles:, :]

        return feats_left, feats_right

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        return_ratios: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            x_left: Left image tensor (B, C, H, W)
            x_right: Right image tensor (B, C, H, W)
            return_ratios: If True, also return the raw ratios

        Returns:
            (green, dead, clover, gdm, total) - each (B, 1)
            If return_ratios: also returns ratios (B, 3)
        """
        # Extract features
        tiles_left, tiles_right = self._extract_tiles_fused(x_left, x_right)

        ctx_left = tiles_left.mean(dim=1)
        ctx_right = tiles_right.mean(dim=1)
        
        # Apply FiLM cross-conditioning
        if self.use_film:
            gamma_l, beta_l = self.film_left(ctx_right)
            gamma_r, beta_r = self.film_right(ctx_left)
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
        
        # Predict Total (absolute biomass)
        total = self.softplus(self.head_total(f))  # (B, 1), always positive
        
        # Predict ratios via softmax (guaranteed to sum to 1)
        logits = self.head_ratios(f)  # (B, 3)
        ratios = F.softmax(logits / self.ratio_temperature, dim=1)  # (B, 3)
        
        # Derive components: component = Total × ratio
        green = total * ratios[:, 0:1]   # (B, 1)
        dead = total * ratios[:, 1:2]    # (B, 1)
        clover = total * ratios[:, 2:3]  # (B, 1)
        
        # GDM = Green + Clover (exact by construction)
        gdm = green + clover  # (B, 1)
        
        # Note: Green + Dead + Clover = Total (guaranteed by softmax!)
        
        if return_ratios:
            return green, dead, clover, gdm, total, ratios
        
        return green, dead, clover, gdm, total


class HierarchicalRatioDINO(nn.Module):
    """
    Hierarchical Ratio Model: Decompose prediction hierarchically.
    
    Stage 1: Predict Total (strongest visual signal)
    Stage 2: Predict alive_ratio = GDM / Total ∈ [0,1]
    Stage 3: Predict green_ratio = Green / GDM ∈ [0,1]
    
    Then derive:
    - GDM = Total × alive_ratio
    - Dead = Total - GDM = Total × (1 - alive_ratio)
    - Green = GDM × green_ratio
    - Clover = GDM - Green = GDM × (1 - green_ratio)
    
    All intermediate values are bounded [0, 1], making optimization easier.
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
        
        if use_film:
            self.film_left = FiLM(feat_dim)
            self.film_right = FiLM(feat_dim)
        
        if use_attention_pool:
            self.attn_pool_left = AttentionPooling(feat_dim)
            self.attn_pool_right = AttentionPooling(feat_dim)
        
        self.combined_dim = feat_dim * 2
        hidden_dim = max(64, int(self.combined_dim * hidden_ratio))
        self.hidden_dim = hidden_dim
        
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(self.combined_dim),
            nn.Linear(self.combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        def _make_head(in_dim: int, out_dim: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(in_dim, out_dim),
            )
        
        # Head for Total biomass
        self.head_total = _make_head(hidden_dim, 1)
        
        # Head for alive ratio (GDM / Total) - sigmoid output
        self.head_alive_ratio = _make_head(hidden_dim, 1)
        
        # Head for green ratio (Green / GDM) - sigmoid output
        self.head_green_ratio = _make_head(hidden_dim, 1)
        
        self.softplus = nn.Softplus(beta=1.0)
        
        self._init_heads()
    
    def _init_heads(self) -> None:
        for head in [self.head_total, self.head_alive_ratio, self.head_green_ratio]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Initialize ratio biases based on data statistics
        # Mean alive_ratio (GDM/Total) ≈ 0.72 -> logit ≈ 0.94
        # Mean green_ratio (Green/GDM) ≈ 0.77 -> logit ≈ 1.2
        with torch.no_grad():
            self.head_alive_ratio[-1].bias.fill_(0.9)
            self.head_green_ratio[-1].bias.fill_(1.2)
    
    def _collect_tiles(self, x: torch.Tensor) -> List[torch.Tensor]:
        _, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)

        tiles = []
        for rs, re in rows:
            for cs, ce in cols:
                xt = x[:, :, rs:re, cs:ce]
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res),
                                       mode="bilinear", align_corners=False)
                tiles.append(xt)
        return tiles

    def _extract_tiles_fused(
        self, x_left: torch.Tensor, x_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_left.size(0)
        tiles_left = self._collect_tiles(x_left)
        tiles_right = self._collect_tiles(x_right)
        num_tiles = len(tiles_left)

        all_tiles = torch.cat(tiles_left + tiles_right, dim=0)
        all_feats = self.backbone(all_tiles)

        total_tiles = 2 * num_tiles
        all_feats = all_feats.view(total_tiles, B, -1).permute(1, 0, 2)
        feats_left = all_feats[:, :num_tiles, :]
        feats_right = all_feats[:, num_tiles:, :]

        return feats_left, feats_right

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        return_ratios: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with hierarchical ratio prediction.
        """
        tiles_left, tiles_right = self._extract_tiles_fused(x_left, x_right)

        ctx_left = tiles_left.mean(dim=1)
        ctx_right = tiles_right.mean(dim=1)
        
        if self.use_film:
            gamma_l, beta_l = self.film_left(ctx_right)
            gamma_r, beta_r = self.film_right(ctx_left)
            tiles_left = tiles_left * (1 + gamma_l.unsqueeze(1)) + beta_l.unsqueeze(1)
            tiles_right = tiles_right * (1 + gamma_r.unsqueeze(1)) + beta_r.unsqueeze(1)
        
        if self.use_attention_pool:
            f_l = self.attn_pool_left(tiles_left)
            f_r = self.attn_pool_right(tiles_right)
        else:
            f_l = tiles_left.mean(dim=1)
            f_r = tiles_right.mean(dim=1)
        
        f = torch.cat([f_l, f_r], dim=1)
        f = self.shared_proj(f)
        
        # Stage 1: Total biomass
        total = self.softplus(self.head_total(f))  # (B, 1)
        
        # Stage 2: Alive ratio (GDM / Total) ∈ [0, 1]
        alive_ratio = torch.sigmoid(self.head_alive_ratio(f))  # (B, 1)
        
        # Stage 3: Green ratio (Green / GDM) ∈ [0, 1]
        green_ratio = torch.sigmoid(self.head_green_ratio(f))  # (B, 1)
        
        # Derive all targets hierarchically
        gdm = total * alive_ratio                    # GDM = Total × alive_ratio
        dead = total * (1 - alive_ratio)             # Dead = Total × (1 - alive_ratio)
        green = gdm * green_ratio                    # Green = GDM × green_ratio
        clover = gdm * (1 - green_ratio)             # Clover = GDM × (1 - green_ratio)
        
        if return_ratios:
            return green, dead, clover, gdm, total, alive_ratio, green_ratio
        
        return green, dead, clover, gdm, total


class RatioMSELoss(nn.Module):
    """
    Loss function for Softmax Ratio model.
    
    Computes:
    1. MSE on all 5 targets (green, dead, clover, gdm, total)
    2. Optional: auxiliary loss on ratios directly (MSE or KL)
    """
    
    def __init__(
        self,
        target_weights: List[float] = [0.1, 0.1, 0.1, 0.2, 0.5],
        use_huber_for_dead: bool = True,
        huber_delta: float = 5.0,
        ratio_loss_weight: float = 0.0,
        ratio_loss_type: str = "mse",  # "mse" or "kl"
    ) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(target_weights, dtype=torch.float32))
        self.use_huber_for_dead = use_huber_for_dead
        self.huber_delta = huber_delta
        self.ratio_loss_weight = ratio_loss_weight
        self.ratio_loss_type = ratio_loss_type
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_ratios: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) - green, dead, clover, gdm, total
            target: (B, 5) - ground truth
            pred_ratios: (B, 3) - predicted ratios (optional)
        """
        weights = self.weights.to(pred.device)
        
        green_pred, dead_pred, clover_pred, gdm_pred, total_pred = pred.unbind(dim=1)
        green_gt, dead_gt, clover_gt, gdm_gt, total_gt = target.unbind(dim=1)
        
        # MSE for most targets
        loss_green = F.mse_loss(green_pred, green_gt, reduction='mean')
        loss_clover = F.mse_loss(clover_pred, clover_gt, reduction='mean')
        loss_gdm = F.mse_loss(gdm_pred, gdm_gt, reduction='mean')
        loss_total = F.mse_loss(total_pred, total_gt, reduction='mean')
        
        # Huber for Dead (handles outliers/sparse values better)
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
        
        # Optional ratio loss (helps learn proportions)
        if self.ratio_loss_weight > 0 and pred_ratios is not None:
            # Compute target ratios
            target_total = total_gt + 1e-8
            target_ratios = torch.stack([
                green_gt / target_total,
                dead_gt / target_total,
                clover_gt / target_total,
            ], dim=1)  # (B, 3)
            
            # Normalize to sum to 1 (handle edge cases)
            target_ratios = target_ratios.clamp(min=1e-6)
            target_ratios = target_ratios / target_ratios.sum(dim=1, keepdim=True)
            
            if self.ratio_loss_type == "kl":
                # KL divergence - good for probability distributions
                # Scale factor to match main loss magnitude
                ratio_loss = F.kl_div(
                    torch.log(pred_ratios + 1e-8),
                    target_ratios,
                    reduction='batchmean'
                )
                # KL is typically 0.01-2.0, scale up to match main loss (~100-1000)
                ratio_loss_scaled = ratio_loss * 500.0
            else:
                # MSE on ratios - simpler and more stable
                ratio_loss = F.mse_loss(pred_ratios, target_ratios, reduction='mean')
                # Ratio MSE is ~0.01-0.1, scale up to match main loss
                ratio_loss_scaled = ratio_loss * 1000.0
            
            main_loss = main_loss + self.ratio_loss_weight * ratio_loss_scaled
        
        return main_loss


def build_ratio_model(
    backbone_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
    grid: Tuple[int, int] = (2, 2),
    pretrained: bool = True,
    dropout: float = 0.2,
    hidden_ratio: float = 0.5,
    use_film: bool = True,
    use_attention_pool: bool = True,
    gradient_checkpointing: bool = False,
    model_type: str = "softmax",
    ratio_temperature: float = 1.0,
) -> nn.Module:
    """
    Build ratio-based model.
    
    Args:
        model_type: "softmax" for SoftmaxRatioDINO, "hierarchical" for HierarchicalRatioDINO
    """
    if model_type == "hierarchical":
        return HierarchicalRatioDINO(
            backbone_name=backbone_name,
            grid=grid,
            pretrained=pretrained,
            dropout=dropout,
            hidden_ratio=hidden_ratio,
            use_film=use_film,
            use_attention_pool=use_attention_pool,
            gradient_checkpointing=gradient_checkpointing,
        )
    else:
        return SoftmaxRatioDINO(
            backbone_name=backbone_name,
            grid=grid,
            pretrained=pretrained,
            dropout=dropout,
            hidden_ratio=hidden_ratio,
            use_film=use_film,
            use_attention_pool=use_attention_pool,
            gradient_checkpointing=gradient_checkpointing,
            ratio_temperature=ratio_temperature,
        )
