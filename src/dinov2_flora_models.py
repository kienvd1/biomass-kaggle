"""
DINOv2 Flora Model for Biomass Prediction.

Based on PlantHydra (1st place PlantTraits2024 solution):
https://www.kaggle.com/competitions/planttraits2024/discussion/510393

Key components adapted:
1. DINOv2 backbone with flora pretraining (vit_base_patch14_reg4_dinov2)
2. StructuredSelfAttention for tabular metadata fusion
3. Multi-task learning: Regression + Species Classification
4. LabelEncoder for log-scale target normalization
5. R2Loss + Cosine Similarity loss
6. Soft classification with species-based trait prior
7. Learnable blending of regression and classification outputs
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import AttentionPoolLatent

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# =============================================================================
# Label Encoder (Log-scale + Standardization)
# =============================================================================

class LabelEncoder(nn.Module):
    """
    Optional target normalization for stable training.
    
    Two modes:
    1. Log-scale (use_log=True): Log10(x + 1) then standardize (PlantHydra style)
    2. Raw-scale (use_log=False): Just standardize raw values (default)
    
    Pre-computed from CSIRO biomass training data:
    - Green: mean ~15, range 0-215
    - Dead: mean ~8, range 0-200 (high variance, many zeros)
    - Clover: mean ~4, range 0-85 (sparse)
    - GDM: mean ~25, range 0-235
    - Total: mean ~40, range 0-315
    """
    
    def __init__(self, use_log: bool = False) -> None:
        super().__init__()
        self.use_log = use_log
        
        if use_log:
            # Log10(target + 1) statistics
            self.mean = nn.Parameter(
                torch.tensor([1.05, 0.55, 0.35, 1.30, 1.55], dtype=torch.float32),
                requires_grad=False,
            )
            self.std = nn.Parameter(
                torch.tensor([0.45, 0.60, 0.50, 0.42, 0.40], dtype=torch.float32),
                requires_grad=False,
            )
        else:
            # Raw-scale statistics (from CSIRO training data)
            self.mean = nn.Parameter(
                torch.tensor([15.0, 8.0, 4.0, 25.0, 40.0], dtype=torch.float32),
                requires_grad=False,
            )
            self.std = nn.Parameter(
                torch.tensor([22.0, 18.0, 10.0, 32.0, 48.0], dtype=torch.float32),
                requires_grad=False,
            )
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform targets to normalized scale.
        
        Args:
            x: Raw targets (B, 5) [green, dead, clover, gdm, total]
        Returns:
            Normalized targets (B, 5)
        """
        with torch.no_grad():
            if self.use_log:
                x = torch.log10(x + 1.0)
            standardized = (x - self.mean) / (self.std + 1e-6)
        return standardized
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized predictions back to original scale.
        
        Args:
            x: Normalized predictions (B, 5)
        Returns:
            Raw-scale predictions (B, 5)
        """
        with torch.no_grad():
            denorm = x * self.std + self.mean
            if self.use_log:
                denorm = torch.pow(10.0, denorm) - 1.0
            original = F.relu(denorm)  # Ensure non-negative
        return original


# =============================================================================
# Structured Self-Attention for Tabular Features
# =============================================================================

class StructuredSelfAttention(nn.Module):
    """
    Multi-head self-attention for tabular metadata features.
    
    Learns feature interactions across metadata dimensions:
    - State (4), Month (10), Species (8), NDVI (1), Height (1)
    - One-hot encoded: 4 + 10 + 8 + 2 = 24 dimensions
    
    Based on PlantHydra's implementation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        
        # Multi-head attention blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                "query": nn.Linear(input_dim, output_dim),
                "key": nn.Linear(input_dim, output_dim),
                "value": nn.Linear(input_dim, output_dim),
            }))
        
        # Project concatenated blocks to output
        self.output = nn.Linear(output_dim * num_blocks, output_dim)
        nn.init.kaiming_normal_(self.output.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tabular features (B, input_dim) or (B, N, input_dim)
        Returns:
            Processed features (B, output_dim) or (B, N, output_dim)
        """
        # Handle 2D input by adding sequence dimension
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, input_dim)
            squeeze_output = True
        
        block_outputs = []
        for block in self.blocks:
            Q = block["query"](x)
            K = block["key"](x)
            V = block["value"](x)
            
            # Scaled dot-product attention
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.output_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)
            block_outputs.append(attn_output)
        
        # Concatenate and project
        combined = torch.cat(block_outputs, dim=-1)
        output = self.output(combined)
        
        if squeeze_output:
            output = output.squeeze(1)
        
        return output


# =============================================================================
# FiLM for Stereo Fusion (from DINOv3Direct)
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


# =============================================================================
# DINOv2 Flora Model
# =============================================================================

class DINOv2Flora(nn.Module):
    """
    DINOv2 Flora model for biomass prediction.
    
    Architecture (based on PlantHydra):
    1. DINOv2 backbone (vit_base_patch14_reg4_dinov2) - optionally Flora pretrained
    2. AttentionPoolLatent for global image features
    3. StructuredSelfAttention for tabular metadata
    4. Regression head for 5 biomass targets
    5. Classification head for species (multi-task)
    6. Optional: Blending with soft classification
    
    Stereo support:
    - Processes left/right tiles separately
    - FiLM for cross-view conditioning
    - Attention pooling over tiles
    """
    
    # Class labels (must match dataset.py)
    NUM_STATES = 4
    NUM_MONTHS = 10
    NUM_SPECIES = 8
    
    # Tabular feature dimension
    # One-hot: State(4) + Month(10) + Species(8) + NDVI(1) + Height(1) = 24
    TABULAR_DIM = 24
    
    def __init__(
        self,
        num_targets: int = 5,
        train_blocks: int = 4,
        train_tokens: bool = False,
        backbone: str = "vitb",
        ckpt_path: Optional[str] = None,
        use_reg_head: bool = True,
        use_clf_head: bool = True,
        use_soft_clf: bool = False,
        use_blending: bool = False,
        grid: Tuple[int, int] = (2, 2),
        use_film: bool = True,
        use_attention_pool: bool = True,
        dropout: float = 0.4,
        tabular_hidden_dim: int = 128,
        tabular_num_blocks: int = 4,
        gradient_checkpointing: bool = False,
        use_log_scale: bool = False,
    ) -> None:
        """
        Args:
            num_targets: Number of regression targets (5 for biomass)
            train_blocks: Number of transformer blocks to fine-tune (from end)
            train_tokens: Whether to train cls/pos/reg tokens
            backbone: Backbone size - "vitb", "vitl", or "vitg"
            ckpt_path: Optional path to Flora pretrained weights
            use_reg_head: Enable regression head
            use_clf_head: Enable species classification head
            use_soft_clf: Enable soft classification (species probs × species means)
            use_blending: Enable learnable blending of reg/clf outputs
            grid: Tiling grid for stereo images
            use_film: Use FiLM for stereo fusion
            use_attention_pool: Use attention pooling for tiles
            dropout: Dropout rate in heads
            tabular_hidden_dim: Hidden dimension for tabular features
            tabular_num_blocks: Number of attention blocks for tabular features
            gradient_checkpointing: Enable gradient checkpointing
            use_log_scale: Use log-scale target normalization (default: False, raw-scale)
        """
        super().__init__()
        
        self.num_targets = num_targets
        self.train_blocks = train_blocks
        self.use_reg_head = use_reg_head
        self.use_clf_head = use_clf_head
        self.use_soft_clf = use_soft_clf
        self.use_blending = use_blending
        self.grid = grid
        self.use_film = use_film
        self.use_attention_pool = use_attention_pool
        self.use_log_scale = use_log_scale
        
        # Label encoder for target normalization (raw-scale by default)
        self.label_encoder = LabelEncoder(use_log=use_log_scale)
        
        # Build backbone
        self._build_backbone(backbone, ckpt_path, gradient_checkpointing)
        
        # Freeze layers
        self._freeze_layers(train_blocks, train_tokens)
        
        # Tabular feature processing
        self.tabular = StructuredSelfAttention(
            input_dim=self.TABULAR_DIM,
            output_dim=tabular_hidden_dim,
            num_blocks=tabular_num_blocks,
        )
        self.tabular_hidden_dim = tabular_hidden_dim
        
        # Stereo fusion modules
        if use_film:
            self.film_left = FiLM(self.feat_dim)
            self.film_right = FiLM(self.feat_dim)
        
        if use_attention_pool:
            self.attn_pool_left = AttentionPooling(self.feat_dim)
            self.attn_pool_right = AttentionPooling(self.feat_dim)
        
        # Combined feature dimension
        # Stereo: feat_dim * 2, Tabular: tabular_hidden_dim
        self.combined_dim = self.feat_dim * 2 + tabular_hidden_dim
        
        # Build heads
        self._build_heads(dropout)
        
        # Species biomass prior (mean biomass per species, pre-computed)
        # Shape: (NUM_SPECIES, num_targets)
        # Initialized with zeros, should be set from data
        self.register_buffer(
            "species_biomass_prior",
            torch.zeros(self.NUM_SPECIES, num_targets),
        )
    
    def _build_backbone(
        self,
        backbone: str,
        ckpt_path: Optional[str],
        gradient_checkpointing: bool,
    ) -> None:
        """Build DINOv2 backbone with optional PlantCLEF pretrained weights."""
        backbone_configs = {
            "vitb": "vit_base_patch14_reg4_dinov2.lvd142m",
            "vitl": "vit_large_patch14_reg4_dinov2.lvd142m",
            "vitg": "vit_giant_patch14_reg4_dinov2.lvd142m",
        }
        model_name = backbone_configs.get(backbone, backbone_configs["vitb"])
        
        # Check if ckpt_path is safetensors (load separately) or .pth/.tar (let timm handle)
        use_safetensors = ckpt_path and ckpt_path.endswith('.safetensors')
        
        self.body = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            checkpoint_path=None if use_safetensors else ckpt_path,
        )
        
        # Load safetensors weights if provided
        if use_safetensors:
            if not HAS_SAFETENSORS:
                raise ImportError("safetensors not installed. Run: pip install safetensors")
            if os.path.exists(ckpt_path):
                print(f"Loading PlantCLEF weights from: {ckpt_path}")
                state_dict = load_safetensors(ckpt_path)
                # Load only backbone weights (ignore classifier head)
                missing, unexpected = self.body.load_state_dict(state_dict, strict=False)
                print(f"  Loaded {len(state_dict) - len(unexpected)} weights, "
                      f"missing: {len(missing)}, unexpected: {len(unexpected)}")
            else:
                print(f"Warning: ckpt_path not found: {ckpt_path}, using ImageNet weights")
        
        self.feat_dim = self.body.num_features  # 768 for base
        self.input_res = 518  # Native DINOv2 resolution
        
        # Get num_heads from the first attention block (timm doesn't expose it at top level)
        num_heads = self.body.blocks[0].attn.num_heads if hasattr(self.body.blocks[0].attn, 'num_heads') else 12
        mlp_ratio = getattr(self.body, 'mlp_ratio', 4.0)
        norm_layer = getattr(self.body, 'norm_layer', nn.LayerNorm)
        
        # Replace classifier with attention pooling
        self.body.reset_classifier(0, "avg")
        self.body.global_pool = "map"
        self.body.attn_pool = AttentionPoolLatent(
            self.body.embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        
        if gradient_checkpointing and hasattr(self.body, "set_grad_checkpointing"):
            self.body.set_grad_checkpointing(enable=True)
    
    def _freeze_layers(self, train_blocks: int, train_tokens: bool) -> None:
        """Freeze backbone layers except last train_blocks."""
        # Always freeze patch_embed and norm
        for layer in [self.body.patch_embed, self.body.norm]:
            for p in layer.parameters():
                p.requires_grad = False
        
        # Optionally freeze tokens
        if not train_tokens:
            self.body.cls_token.requires_grad = False
            self.body.pos_embed.requires_grad = False
            if hasattr(self.body, "reg_token"):
                self.body.reg_token.requires_grad = False
        
        # Freeze early blocks
        if train_blocks is not None:
            num_blocks = len(self.body.blocks)
            freeze_until = num_blocks - train_blocks
            for i in range(freeze_until):
                for p in self.body.blocks[i].parameters():
                    p.requires_grad = False
    
    def _build_heads(self, dropout: float) -> None:
        """Build regression and classification heads."""
        # Regression head
        if self.use_reg_head:
            self.reg_head = nn.Sequential(
                nn.Linear(self.combined_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, self.num_targets),
            )
        
        # Classification head (species)
        if self.use_clf_head or self.use_soft_clf:
            self.clf_head = nn.Sequential(
                nn.Linear(self.combined_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, self.NUM_SPECIES),
            )
        
        # Blending weights (learnable per target)
        if self.use_blending:
            if self.use_reg_head:
                self.reg_weight = nn.Parameter(
                    torch.ones(self.num_targets, dtype=torch.float32)
                )
            if self.use_clf_head:
                self.clf_weight = nn.Parameter(
                    torch.ones(self.num_targets, dtype=torch.float32)
                )
            if self.use_soft_clf:
                self.soft_clf_weight = nn.Parameter(
                    torch.ones(self.num_targets, dtype=torch.float32)
                )
    
    def set_species_prior(self, species_means: torch.Tensor) -> None:
        """
        Set species biomass prior from training data.
        
        Args:
            species_means: (NUM_SPECIES, num_targets) mean biomass per species
        """
        self.species_biomass_prior.copy_(species_means)
    
    def _make_edges(self, length: int, parts: int) -> List[Tuple[int, int]]:
        """Create edge indices for tiling."""
        step = length // parts
        edges = []
        start = 0
        for _ in range(parts - 1):
            edges.append((start, start + step))
            start += step
        edges.append((start, length))
        return edges
    
    def _collect_tiles(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split image into grid of tiles."""
        _, _, H, W = x.shape
        r, c = self.grid
        rows = self._make_edges(H, r)
        cols = self._make_edges(W, c)
        
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
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract tile features from both stereo views."""
        B = x_left.size(0)
        
        tiles_left = self._collect_tiles(x_left)
        tiles_right = self._collect_tiles(x_right)
        num_tiles = len(tiles_left)
        
        # Process all tiles in one forward pass
        all_tiles = torch.cat(tiles_left + tiles_right, dim=0)
        all_feats = self.body(all_tiles)  # (2*T*B, D)
        
        # Reshape and split
        all_feats = all_feats.view(2 * num_tiles, B, -1).permute(1, 0, 2)
        feats_left = all_feats[:, :num_tiles, :]   # (B, T, D)
        feats_right = all_feats[:, num_tiles:, :]  # (B, T, D)
        
        return feats_left, feats_right
    
    def encode_tabular(
        self,
        state: torch.Tensor,
        month: torch.Tensor,
        species: torch.Tensor,
        ndvi: torch.Tensor,
        height: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode tabular metadata as one-hot + continuous features.
        
        Args:
            state: (B,) state labels [0-3]
            month: (B,) month labels [0-9]
            species: (B,) species labels [0-7]
            ndvi: (B,) NDVI values [0-1]
            height: (B,) height in cm [1-70]
        
        Returns:
            tabular_features: (B, TABULAR_DIM)
        """
        B = state.size(0)
        device = state.device
        
        # One-hot encodings
        state_oh = F.one_hot(state.long(), self.NUM_STATES).float()  # (B, 4)
        month_oh = F.one_hot(month.long(), self.NUM_MONTHS).float()  # (B, 10)
        species_oh = F.one_hot(species.long(), self.NUM_SPECIES).float()  # (B, 8)
        
        # Normalize continuous features
        ndvi_norm = ndvi.unsqueeze(1)  # Already [0, 1]
        height_norm = (height.unsqueeze(1) / 70.0).clamp(0, 1)  # Normalize to [0, 1]
        
        # Concatenate all features
        tabular = torch.cat([
            state_oh, month_oh, species_oh, ndvi_norm, height_norm
        ], dim=1)  # (B, 24)
        
        return tabular
    
    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        species: Optional[torch.Tensor] = None,
        ndvi: Optional[torch.Tensor] = None,
        height: Optional[torch.Tensor] = None,
        return_encoded: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_left: Left stereo image (B, 3, H, W)
            x_right: Right stereo image (B, 3, H, W)
            state: State labels (B,) [optional, for multi-task]
            month: Month labels (B,) [optional]
            species: Species labels (B,) [optional, for classification head]
            ndvi: NDVI values (B,) [optional]
            height: Height values (B,) [optional]
            return_encoded: If True, return encoded (normalized) predictions too
        
        Returns:
            Dictionary with:
            - "pred": Raw-scale predictions (B, 5)
            - "pred_enc": Encoded predictions (B, 5) [if return_encoded]
            - "species_logits": Species classification logits (B, NUM_SPECIES) [if clf_head]
            - "blended": Blended predictions (B, 5) [if blending]
        """
        B = x_left.size(0)
        device = x_left.device
        outputs = {}
        
        # Extract image features
        tiles_left, tiles_right = self._extract_features(x_left, x_right)
        
        # Context for FiLM
        ctx_left = tiles_left.mean(dim=1)   # (B, D)
        ctx_right = tiles_right.mean(dim=1)  # (B, D)
        
        # Apply FiLM cross-conditioning
        if self.use_film:
            gamma_l, beta_l = self.film_left(ctx_right)
            gamma_r, beta_r = self.film_right(ctx_left)
            tiles_left = tiles_left * (1 + gamma_l.unsqueeze(1)) + beta_l.unsqueeze(1)
            tiles_right = tiles_right * (1 + gamma_r.unsqueeze(1)) + beta_r.unsqueeze(1)
        
        # Pool tiles
        if self.use_attention_pool:
            f_left = self.attn_pool_left(tiles_left)
            f_right = self.attn_pool_right(tiles_right)
        else:
            f_left = tiles_left.mean(dim=1)
            f_right = tiles_right.mean(dim=1)
        
        # Tabular features (default to zeros if not provided)
        if state is None:
            state = torch.zeros(B, dtype=torch.long, device=device)
        if month is None:
            month = torch.zeros(B, dtype=torch.long, device=device)
        if species is None:
            species = torch.zeros(B, dtype=torch.long, device=device)
        if ndvi is None:
            ndvi = torch.full((B,), 0.5, dtype=torch.float32, device=device)
        if height is None:
            height = torch.full((B,), 10.0, dtype=torch.float32, device=device)
        
        tabular_raw = self.encode_tabular(state, month, species, ndvi, height)
        tabular_feat = self.tabular(tabular_raw)  # (B, tabular_hidden_dim)
        
        # Combine image and tabular features
        combined = torch.cat([f_left, f_right, tabular_feat], dim=1)  # (B, combined_dim)
        
        # Regression head
        pred_enc = None
        pred = None
        if self.use_reg_head:
            pred_enc = self.reg_head(combined)  # Encoded predictions
            pred = self.label_encoder.inverse_transform(pred_enc)
            outputs["pred"] = pred
            if return_encoded:
                outputs["pred_enc"] = pred_enc
        
        # Classification head
        species_logits = None
        if self.use_clf_head or self.use_soft_clf:
            species_logits = self.clf_head(combined)  # (B, NUM_SPECIES)
            outputs["species_logits"] = species_logits
            
            # Classification-based prediction (argmax species → prior)
            if self.use_clf_head:
                pred_species = torch.argmax(species_logits, dim=1)  # (B,)
                pred_from_clf = self.species_biomass_prior[pred_species]  # (B, 5)
                outputs["pred_clf"] = pred_from_clf
        
        # Soft classification (weighted average by species probs)
        if self.use_soft_clf and species_logits is not None:
            species_probs = F.softmax(species_logits, dim=1)  # (B, NUM_SPECIES)
            pred_soft = torch.matmul(species_probs, self.species_biomass_prior)  # (B, 5)
            outputs["pred_soft"] = pred_soft
        
        # Blending
        if self.use_blending:
            blended = torch.zeros_like(pred) if pred is not None else None
            denominator = torch.zeros(self.num_targets, device=device)
            
            if self.use_reg_head and pred is not None:
                blended = blended + self.reg_weight * pred
                denominator = denominator + self.reg_weight
            
            if self.use_clf_head and "pred_clf" in outputs:
                blended = blended + self.clf_weight * outputs["pred_clf"]
                denominator = denominator + self.clf_weight
            
            if self.use_soft_clf and "pred_soft" in outputs:
                blended = blended + self.soft_clf_weight * outputs["pred_soft"]
                denominator = denominator + self.soft_clf_weight
            
            if blended is not None:
                blended = blended / (denominator + 1e-6)
                outputs["blended"] = blended
        
        return outputs


# =============================================================================
# Loss Functions
# =============================================================================

class R2Loss(nn.Module):
    """
    R² loss for multi-target regression with CSIRO competition weights.
    
    Minimizes weighted (1 - R²) = weighted SS_res / SS_tot
    Competition weights: Green=0.1, Dead=0.1, Clover=0.1, GDM=0.2, Total=0.5
    """
    
    def __init__(self, num_targets: int = 5, use_competition_weights: bool = True) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.use_competition_weights = use_competition_weights
        
        # CSIRO competition weights: [green, dead, clover, gdm, total]
        if use_competition_weights:
            self.register_buffer(
                "weights",
                torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "weights",
                torch.ones(num_targets, dtype=torch.float32) / num_targets,
            )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, num_targets)
            target: Ground truth (B, num_targets)
        Returns:
            Weighted R² loss across targets (CSIRO competition metric)
        """
        # Residual sum of squares per target
        ss_res = torch.sum((target - pred) ** 2, dim=0)  # (num_targets,)
        
        # Total sum of squares per target
        ss_tot = torch.sum((target - target.mean(dim=0)) ** 2, dim=0)  # (num_targets,)
        
        # R² loss per target (avoid division by zero)
        r2_loss = ss_res / (ss_tot + 1e-6)  # (num_targets,)
        
        # Apply competition weights
        weighted_loss = (r2_loss * self.weights).sum()
        
        return weighted_loss


class FloraLoss(nn.Module):
    """
    Combined loss for DINOv2 Flora model.
    
    Components:
    1. R² loss on encoded predictions (PlantHydra style)
    2. Cosine similarity loss (directional accuracy)
    3. Species classification loss (focal loss)
    4. Optional: Blending loss
    
    Competition weights: Green=0.1, Dead=0.1, Clover=0.1, GDM=0.2, Total=0.5
    """
    
    def __init__(
        self,
        use_r2_loss: bool = True,
        use_cosine_loss: bool = True,
        use_clf_loss: bool = True,
        use_mse_loss: bool = True,
        r2_weight: float = 1.0,
        cosine_weight: float = 0.4,
        clf_weight: float = 0.01,
        mse_weight: float = 0.5,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_r2_loss = use_r2_loss
        self.use_cosine_loss = use_cosine_loss
        self.use_clf_loss = use_clf_loss
        self.use_mse_loss = use_mse_loss
        
        self.r2_weight = r2_weight
        self.cosine_weight = cosine_weight
        self.clf_weight = clf_weight
        self.mse_weight = mse_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        if use_r2_loss:
            self.r2_loss = R2Loss()
        
        # Competition target weights
        self.register_buffer(
            "target_weights",
            torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32),
        )
    
    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss for classification."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal.mean()
    
    def forward(
        self,
        pred_enc: torch.Tensor,
        target_enc: torch.Tensor,
        pred_raw: Optional[torch.Tensor] = None,
        target_raw: Optional[torch.Tensor] = None,
        species_logits: Optional[torch.Tensor] = None,
        species_labels: Optional[torch.Tensor] = None,
        blended: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_enc: Encoded predictions (B, 5)
            target_enc: Encoded targets (B, 5)
            pred_raw: Raw-scale predictions (B, 5)
            target_raw: Raw-scale targets (B, 5)
            species_logits: Species classification logits (B, NUM_SPECIES)
            species_labels: Species ground truth (B,)
            blended: Blended predictions (B, 5)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred_enc.device)
        
        # R² loss on encoded predictions
        if self.use_r2_loss:
            r2_loss = self.r2_loss(pred_enc, target_enc)
            total_loss = total_loss + self.r2_weight * r2_loss
            loss_dict["r2"] = r2_loss.item()
        
        # Cosine similarity loss (per-target weighted)
        if self.use_cosine_loss:
            # Weighted cosine: each target contributes based on competition weight
            diff = pred_enc - target_enc
            weighted_diff = diff * self.target_weights.unsqueeze(0)  # (B, 5)
            cos_loss = (weighted_diff ** 2).sum(dim=1).mean()  # Weighted L2 as proxy
            total_loss = total_loss + self.cosine_weight * cos_loss
            loss_dict["cosine"] = cos_loss.item()
        
        # MSE loss on raw predictions (competition metric proxy)
        if self.use_mse_loss and pred_raw is not None and target_raw is not None:
            mse_per_target = F.mse_loss(pred_raw, target_raw, reduction="none")
            weighted_mse = (mse_per_target * self.target_weights).sum(dim=1).mean()
            total_loss = total_loss + self.mse_weight * weighted_mse
            loss_dict["mse"] = weighted_mse.item()
        
        # Species classification loss
        if self.use_clf_loss and species_logits is not None and species_labels is not None:
            clf_loss = self.focal_loss(species_logits, species_labels)
            total_loss = total_loss + self.clf_weight * clf_loss
            loss_dict["clf"] = clf_loss.item()
        
        # Blending loss (if using blended predictions)
        if blended is not None and target_raw is not None:
            blend_loss = self.r2_loss(
                torch.log10(blended + 1),
                torch.log10(target_raw + 1),
            )
            total_loss = total_loss + 0.1 * blend_loss
            loss_dict["blend"] = blend_loss.item()
        
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# Utility Functions
# =============================================================================

def compute_species_prior(
    df,
    species_col: str = "Species",
    target_cols: List[str] = None,
    species_labels: Dict[str, int] = None,
) -> torch.Tensor:
    """
    Compute mean biomass per species from training data.
    
    Args:
        df: Training DataFrame
        species_col: Column name for species
        target_cols: List of target column names
        species_labels: Species to label mapping
    
    Returns:
        species_means: (NUM_SPECIES, num_targets) tensor
    """
    if target_cols is None:
        target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    
    if species_labels is None:
        species_labels = {
            "Clover": 0, "WhiteClover": 0, "SubcloverLosa": 0, "SubcloverDalkeith": 0,
            "Ryegrass": 1, "Ryegrass_Clover": 2,
            "Phalaris": 3, "Phalaris_Clover": 4, "Phalaris_Ryegrass_Clover": 4,
            "Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass": 4,
            "Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed": 4,
            "Fescue": 5, "Fescue_CrumbWeed": 5,
            "Lucerne": 6, "Mixed": 7,
        }
    
    num_species = max(species_labels.values()) + 1
    num_targets = len(target_cols)
    
    species_means = torch.zeros(num_species, num_targets)
    
    for species_name, label in species_labels.items():
        mask = df[species_col] == species_name
        if mask.sum() > 0:
            means = df.loc[mask, target_cols].mean().values
            species_means[label] = torch.tensor(means, dtype=torch.float32)
    
    return species_means


def freeze_backbone(model: DINOv2Flora) -> None:
    """Freeze backbone parameters."""
    for name, param in model.named_parameters():
        if "body" in name:
            param.requires_grad = False


def unfreeze_backbone(model: DINOv2Flora, train_blocks: int = 4) -> None:
    """Unfreeze last train_blocks of backbone."""
    num_blocks = len(model.body.blocks)
    freeze_until = num_blocks - train_blocks
    
    for i, block in enumerate(model.body.blocks):
        for p in block.parameters():
            p.requires_grad = i >= freeze_until


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

