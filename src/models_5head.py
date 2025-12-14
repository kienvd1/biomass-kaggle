"""5-Head Model Architecture for CSIRO Biomass prediction.

Optimized version with FiLM conditioning, attention pooling, and deeper heads.
Now includes auxiliary heads for State and Month classification.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import _build_dino_by_name, _make_edges

# Label mappings for auxiliary heads
STATE_LABELS: Dict[str, int] = {"NSW": 0, "Tas": 1, "Vic": 2, "WA": 3}
NUM_STATES = 4

# Month labels - individual months are more predictive than seasons (R²=0.26 vs 0.05)
# Months present in data: Jan=1, Feb=2, Apr=4, May=5, Jun=6, Jul=7, Aug=8, Sep=9, Oct=10, Nov=11
# Dead by month: Apr=0.4g, Jul/Aug=5g, Jan/Feb=10g, May/Jun=15g, Oct=26g
MONTH_LABELS: Dict[int, int] = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}
NUM_MONTHS = 10

# Species labels - grouped by similarity for better learning
# Group 1: Pure clover types (high clover, often zero dead)
# Group 2: Ryegrass types (moderate dead/clover)
# Group 3: Phalaris types (varied)
# Group 4: Other (Lucerne, Fescue, etc.)
SPECIES_LABELS: Dict[str, int] = {
    # Clover-dominant (usually high clover, low/zero dead in WA)
    "Clover": 0,
    "WhiteClover": 0,
    "SubcloverLosa": 0,
    "SubcloverDalkeith": 0,
    # Ryegrass types
    "Ryegrass": 1,
    "Ryegrass_Clover": 2,
    # Phalaris types
    "Phalaris": 3,
    "Phalaris_Clover": 4,
    "Phalaris_Ryegrass_Clover": 4,
    "Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass": 4,
    "Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed": 4,
    # Other grasses (often zero clover)
    "Fescue": 5,
    "Fescue_CrumbWeed": 5,
    "Lucerne": 6,
    "Mixed": 7,
}
NUM_SPECIES_GROUPS = 8



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
    - Auxiliary heads for State and Month classification (multi-task learning)
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
        use_aux_heads: bool = False,
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
        self.use_aux_heads = use_aux_heads
        
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
        
        # 5 independent heads for biomass targets
        self.head_green = _make_head(hidden_dim)
        self.head_dead = _make_head(hidden_dim)
        self.head_clover = _make_head(hidden_dim)
        self.head_gdm = _make_head(hidden_dim)
        self.head_total = _make_head(hidden_dim)
        
        # Auxiliary heads for State, Month, and Species classification
        if use_aux_heads:
            self.head_state = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, NUM_STATES),
            )
            self.head_month = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, NUM_MONTHS),
            )
            self.head_species = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, NUM_SPECIES_GROUPS),
            )
        
        self.softplus = nn.Softplus(beta=1.0)
        
        # Initialize heads with small weights for stable training
        self._init_heads()
    
    def _init_heads(self) -> None:
        """Initialize head weights for stable training."""
        heads = [self.head_green, self.head_dead, self.head_clover,
                 self.head_gdm, self.head_total]
        if self.use_aux_heads:
            heads.extend([self.head_state, self.head_month, self.head_species])
        
        for head in heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _collect_tiles(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Collect tiles from image without backbone forward (for fused processing)."""
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
        """Extract tile features from BOTH streams in ONE backbone call.

        This is more efficient than calling backbone separately for each stream
        because it reduces kernel launch overhead and improves GPU utilization.
        """
        B = x_left.size(0)

        # Collect tiles from both images
        tiles_left = self._collect_tiles(x_left)
        tiles_right = self._collect_tiles(x_right)
        num_tiles = len(tiles_left)  # Same for both (grid is fixed)

        # Fuse all tiles into ONE tensor and process with ONE backbone call
        all_tiles = torch.cat(tiles_left + tiles_right, dim=0)  # (2 * num_tiles * B, C, H, W)
        all_feats = self.backbone(all_tiles)  # (2 * num_tiles * B, D)

        # Split back into left and right
        total_tiles = 2 * num_tiles
        all_feats = all_feats.view(total_tiles, B, -1).permute(1, 0, 2)  # (B, 2*num_tiles, D)
        feats_left = all_feats[:, :num_tiles, :]   # (B, num_tiles, D)
        feats_right = all_feats[:, num_tiles:, :]  # (B, num_tiles, D)

        return feats_left, feats_right

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        return_aux: bool = False,
        apply_context_adjustment: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with FiLM cross-conditioning.

        Args:
            x_left: Left image tensor (B, C, H, W)
            x_right: Right image tensor (B, C, H, W)
            return_aux: If True and use_aux_heads is True, return auxiliary logits
            apply_context_adjustment: If True, use predicted state/month/species to adjust outputs

        Returns:
            If return_aux=False: (green, dead, clover, gdm, total) - each (B, 1)
            If return_aux=True: (green, dead, clover, gdm, total, state_logits, month_logits, species_logits)
                - state_logits: (B, NUM_STATES)
                - month_logits: (B, NUM_MONTHS)
                - species_logits: (B, NUM_SPECIES_GROUPS)
        """
        # Extract tile features from BOTH streams in ONE backbone call (faster)
        tiles_left, tiles_right = self._extract_tiles_fused(x_left, x_right)

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
        
        # Apply context-based adjustment using predicted state/month/species
        if apply_context_adjustment and self.use_aux_heads:
            state_logits = self.head_state(f)
            month_logits = self.head_month(f)
            species_logits = self.head_species(f)
            dead, clover = self._apply_context_adjustment(
                dead, clover, state_logits, month_logits, species_logits
            )

            if return_aux:
                return green, dead, clover, gdm, total, state_logits, month_logits, species_logits
            return green, dead, clover, gdm, total

        if return_aux and self.use_aux_heads:
            state_logits = self.head_state(f)
            month_logits = self.head_month(f)
            species_logits = self.head_species(f)
            return green, dead, clover, gdm, total, state_logits, month_logits, species_logits

        return green, dead, clover, gdm, total
    
    def _apply_context_adjustment(
        self,
        dead: torch.Tensor,
        clover: torch.Tensor,
        state_logits: torch.Tensor,
        month_logits: torch.Tensor,
        species_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust predictions based on predicted state, month, and species.

        Based on our analysis:
        - WA state: dead weight is always 0, clover is high (mean 22g)
        - Species determines clover potential (Lucerne/Fescue = 0 clover)
        - Month strongly affects dead: Apr=0.4g, Jul/Aug=5g, Oct=26g
        """
        state_probs = F.softmax(state_logits, dim=1)  # (B, 4)
        month_probs = F.softmax(month_logits, dim=1)  # (B, 10)
        species_probs = F.softmax(species_logits, dim=1)  # (B, 8)

        # State adjustment factors for dead weight
        # States: NSW=0, Tas=1, Vic=2, WA=3
        # WA always has 0 dead
        state_dead_factors = torch.tensor(
            [1.0, 1.0, 0.9, 0.05],  # WA -> near zero dead
            device=dead.device, dtype=dead.dtype
        )
        state_dead_adjust = (state_probs * state_dead_factors.unsqueeze(0)).sum(dim=1, keepdim=True)

        # Month adjustment factors for dead weight (R²=0.26 - very predictive!)
        # Months: Jan=0, Feb=1, Apr=2, May=3, Jun=4, Jul=5, Aug=6, Sep=7, Oct=8, Nov=9
        # Mean dead: Jan=10.3, Feb=10.2, Apr=0.4, May=15.6, Jun=14.7, Jul=5.5, Aug=5.3, Sep=8.7, Oct=26.5, Nov=18.2
        # Normalized to overall mean (11.7): Apr=0.03, Jul=0.47, Aug=0.45, Sep=0.74, Jan=0.88, Feb=0.87, Jun=1.26, May=1.33, Nov=1.56, Oct=2.27
        month_dead_factors = torch.tensor(
            [0.88, 0.87, 0.03, 1.33, 1.26, 0.47, 0.45, 0.74, 2.27, 1.56],
            device=dead.device, dtype=dead.dtype
        )
        month_dead_adjust = (month_probs * month_dead_factors.unsqueeze(0)).sum(dim=1, keepdim=True)

        # Species adjustment factors for clover
        # Groups: 0=Clover types (high), 1=Ryegrass (low), 2=Ryegrass_Clover (med),
        #         3=Phalaris (zero), 4=Phalaris_Clover (med), 5=Fescue (zero),
        #         6=Lucerne (zero), 7=Mixed (low)
        species_clover_factors = torch.tensor(
            [2.5, 0.3, 1.0, 0.05, 1.0, 0.05, 0.05, 0.3],
            device=clover.device, dtype=clover.dtype
        )
        species_clover_adjust = (species_probs * species_clover_factors.unsqueeze(0)).sum(dim=1, keepdim=True)

        # State adjustment for clover (WA has high clover, NSW has low)
        state_clover_factors = torch.tensor(
            [0.5, 1.0, 1.0, 2.0],  # NSW, Tas, Vic, WA
            device=clover.device, dtype=clover.dtype
        )
        state_clover_adjust = (state_probs * state_clover_factors.unsqueeze(0)).sum(dim=1, keepdim=True)

        # Combine adjustments (geometric mean to avoid extreme values)
        dead_adjust = torch.sqrt(state_dead_adjust * month_dead_adjust)
        clover_adjust = torch.sqrt(state_clover_adjust * species_clover_adjust)

        # Apply soft adjustment (blend with original)
        # Use 0.5 blend to not over-rely on potentially incorrect predictions
        blend = 0.5
        dead_adjusted = dead * (1 - blend + blend * dead_adjust)
        clover_adjusted = clover * (1 - blend + blend * clover_adjust)

        return dead_adjusted, clover_adjusted


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


class AuxiliaryLoss(nn.Module):
    """
    Combined loss for biomass regression + auxiliary classification tasks.

    The auxiliary heads (State, Month, Species) help the backbone learn discriminative features.
    Based on analysis: State+Month explains 34% of Dead variance, Species helps with Clover.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        state_weight: float = 5.0,
        month_weight: float = 3.0,
        species_weight: float = 2.0,
        state_class_weights: Optional[List[float]] = None,
        month_class_weights: Optional[List[float]] = None,
        species_class_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            base_loss: The main biomass loss function (e.g., ConstrainedMSELoss)
            state_weight: Weight for state classification loss
            month_weight: Weight for month classification loss
            species_weight: Weight for species classification loss
            state_class_weights: Optional class weights for imbalanced states [NSW, Tas, Vic, WA]
            month_class_weights: Optional class weights for imbalanced months
            species_class_weights: Optional class weights for imbalanced species groups
        """
        super().__init__()
        self.base_loss = base_loss
        self.state_weight = state_weight
        self.month_weight = month_weight
        self.species_weight = species_weight

        # Default class weights based on inverse frequency from training data
        # State: NSW=75, Tas=138, Vic=112, WA=32 -> weights inversely proportional
        if state_class_weights is None:
            state_class_weights = [1.19, 0.65, 0.80, 2.79]  # NSW, Tas, Vic, WA
        self.register_buffer(
            "state_class_weights",
            torch.tensor(state_class_weights, dtype=torch.float32)
        )

        # Month: Jan=17, Feb=24, Apr=10, May=42, Jun=53, Jul=41, Aug=37, Sep=67, Oct=29, Nov=37
        # Normalized inverse frequency
        if month_class_weights is None:
            month_class_weights = [2.1, 1.5, 3.6, 0.85, 0.67, 0.87, 0.96, 0.53, 1.23, 0.96]
        self.register_buffer(
            "month_class_weights",
            torch.tensor(month_class_weights, dtype=torch.float32)
        )

        # Species groups: 0=Clover(61), 1=Ryegrass(62), 2=Ryegrass_Clover(98), 3=Phalaris(8),
        #                 4=Phalaris_Clover(26), 5=Fescue(38), 6=Lucerne(22), 7=Mixed(2)
        # Capped extreme weights to prevent gradient explosion
        if species_class_weights is None:
            species_class_weights = [0.58, 0.57, 0.36, 3.0, 1.36, 0.93, 1.61, 5.0]
        self.register_buffer(
            "species_class_weights",
            torch.tensor(species_class_weights, dtype=torch.float32)
        )
    
    def forward(
        self,
        pred_biomass: torch.Tensor,
        target_biomass: torch.Tensor,
        state_logits: Optional[torch.Tensor] = None,
        state_labels: Optional[torch.Tensor] = None,
        month_logits: Optional[torch.Tensor] = None,
        month_labels: Optional[torch.Tensor] = None,
        species_logits: Optional[torch.Tensor] = None,
        species_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            pred_biomass: (B, 5) biomass predictions
            target_biomass: (B, 5) biomass targets
            state_logits: (B, NUM_STATES) state classification logits
            state_labels: (B,) state labels (0-3)
            month_logits: (B, NUM_MONTHS) month classification logits
            month_labels: (B,) month labels (0-9)
            species_logits: (B, NUM_SPECIES_GROUPS) species classification logits
            species_labels: (B,) species group labels (0-7)

        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dict with individual loss components for logging
        """
        # Main biomass loss
        loss_biomass = self.base_loss(pred_biomass, target_biomass)

        loss_dict = {"loss_biomass": loss_biomass.item()}
        total_loss = loss_biomass

        # Auxiliary state loss
        if state_logits is not None and state_labels is not None:
            state_logits_clamped = state_logits.clamp(-10, 10)
            loss_state = F.cross_entropy(
                state_logits_clamped,
                state_labels,
                weight=self.state_class_weights.to(state_logits.device),
                label_smoothing=0.1,
            )
            total_loss = total_loss + self.state_weight * loss_state
            loss_dict["loss_state"] = loss_state.item()
        
        # Auxiliary month loss
        if month_logits is not None and month_labels is not None:
            month_logits_clamped = month_logits.clamp(-10, 10)
            loss_month = F.cross_entropy(
                month_logits_clamped,
                month_labels,
                weight=self.month_class_weights.to(month_logits.device),
                label_smoothing=0.1,
            )
            total_loss = total_loss + self.month_weight * loss_month
            loss_dict["loss_month"] = loss_month.item()

        # Auxiliary species loss
        if species_logits is not None and species_labels is not None:
            # Clamp logits to prevent extreme gradients
            species_logits_clamped = species_logits.clamp(-10, 10)
            loss_species = F.cross_entropy(
                species_logits_clamped,
                species_labels,
                weight=self.species_class_weights.to(species_logits.device),
                label_smoothing=0.1,  # Prevent overconfident predictions
            )
            total_loss = total_loss + self.species_weight * loss_species
            loss_dict["loss_species"] = loss_species.item()

        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict


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
    use_aux_heads: bool = False,
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
        use_aux_heads: Add auxiliary heads for State and Month classification
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
        use_aux_heads=use_aux_heads,
    )

