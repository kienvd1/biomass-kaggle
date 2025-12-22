
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from torchmetrics.regression import R2Score
from torchmetrics.classification import MulticlassAccuracy

# Import DINOv3 backbone builder and depth components
from src.dinov3_models import build_dinov3_backbone, FiLM, AttentionPooling, DepthFeatures, DepthGuidedAttention

class LabelEncoder(nn.Module):
    """
    Encodes targets using Log10 transform and Standardization.
    Based on PlantTraits2024 1st place solution.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Original -> Log10 -> Standardized"""
        # Add epsilon to avoid log(0)
        log_x = torch.log10(x + 1e-6)
        return (log_x - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Standardized -> Log10 -> Original"""
        # Revert standardization
        log_x = x * (self.std + 1e-6) + self.mean
        # Revert log10
        return 10 ** log_x

class R2Loss(nn.Module):
    """
    R2 Loss: 1 - R2 (maximized) = Loss (minimized).
    Based on PlantTraits2024 1st place solution.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # SS_res = Sum of Squared Residuals
        ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
        # SS_tot = Total Sum of Squares
        ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
        
        # R2 per target (add epsilon to avoid div by zero)
        r2_loss = ss_res / (ss_tot + 1e-6)
        
        # Mean over targets
        return torch.mean(r2_loss)

class PlantTraitsBiomassModel(nn.Module):
    """
    Biomass Prediction Model inspired by PlantTraits2024 1st Place Solution.
    
    Key features:
    - DINOv3 Backbone
    - Species Classification Head (Auxiliary)
    - Soft-Classification-Guided Regression (Weighted average of species priors)
    - Direct Regression Head
    - Blending of Direct and Soft-Classified predictions
    - Target Label Encoding (Log+Standardization)
    - Optional: Depth Features (Depth Anything V2)
    """
    
    def __init__(
        self,
        num_targets: int = 5,
        num_species: int = 8,
        backbone_size: str = "base",
        pretrained: bool = True,
        ckpt_path: Optional[str] = None,
        specie_traits_mean: Optional[torch.Tensor] = None, # (num_species, num_targets)
        target_mean: Optional[torch.Tensor] = None,        # (num_targets,)
        target_std: Optional[torch.Tensor] = None,         # (num_targets,)
        grid: Tuple[int, int] = (2, 2),
        use_film: bool = True,
        use_depth: bool = False,
        use_depth_attention: bool = False,
        depth_model_size: str = "small",
    ):
        super().__init__()
        
        # 1. Backbone
        self.backbone, self.feat_dim, self.input_res = build_dinov3_backbone(
            pretrained=pretrained,
            backbone_size=backbone_size,
            ckpt_path=ckpt_path
        )
        
        # Stereo / Tiling Setup (copied from DINOv3Direct)
        self.grid = grid
        self.use_film = use_film
        self.use_depth = use_depth
        self.use_depth_attention = use_depth_attention
        
        if use_film:
            self.film_left = FiLM(self.feat_dim)
            self.film_right = FiLM(self.feat_dim)
            
        # Attention Pooling (Standard or Depth-Guided)
        if use_depth_attention:
            self.attn_pool_left = DepthGuidedAttention(self.feat_dim, grid=grid, model_size=depth_model_size)
            self.attn_pool_right = DepthGuidedAttention(self.feat_dim, grid=grid, model_size=depth_model_size)
        else:
            self.attn_pool_left = AttentionPooling(self.feat_dim)
            self.attn_pool_right = AttentionPooling(self.feat_dim)
            
        # Feature processing
        self.combined_dim = self.feat_dim * 2
        
        # Add Depth Features dimension if enabled
        if use_depth:
            depth_out_dim = 32
            self.depth_module = DepthFeatures(out_dim=depth_out_dim, model_size=depth_model_size)
            self.combined_dim += depth_out_dim
            
        self.hidden_dim = 256 # Consistent with PlantTraits solution
        
        # Projection after concatenation
        self.proj = nn.Sequential(
            nn.Linear(self.combined_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 2. Label Encoder
        if target_mean is None or target_std is None:
            # Default values (should be overwritten by computed stats)
            target_mean = torch.zeros(num_targets)
            target_std = torch.ones(num_targets)
            
        self.le = LabelEncoder(target_mean, target_std)
        
        # 3. Species Traits Prior (Biomass averages per species)
        if specie_traits_mean is None:
            specie_traits_mean = torch.zeros(num_species, num_targets)
        
        # Store species traits in ENCODED space (Log + Standardized)
        self.register_buffer("raw_specie_traits", specie_traits_mean)
        
        # 4. Heads
        
        # A) Direct Regression Head (predicts Encoded targets)
        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_targets)
        )
        
        # B) Species Classification Head (predicts Species logits)
        self.clf_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_species)
        )
        
        # C) Blending Weights (Learnable)
        self.blend_weights = nn.Parameter(torch.ones(3, num_targets)) 
        
        # Losses
        self.r2_loss = R2Loss()
        self.clf_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.num_targets = num_targets

    def _collect_tiles(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Helper to tile images (same as DINOv3Direct)
        _, _, H, W = x.shape
        r, c = self.grid
        h_step, w_step = H // r, W // c
        tiles = []
        for i in range(r):
            for j in range(c):
                tile = x[:, :, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                if tile.shape[-2:] != (self.input_res, self.input_res):
                    tile = F.interpolate(tile, size=(self.input_res, self.input_res), mode='bilinear', align_corners=False)
                tiles.append(tile)
        return tiles

    def extract_features(self, x_left, x_right):
        # 1. Tile and Extract
        tiles_l = self._collect_tiles(x_left)
        tiles_r = self._collect_tiles(x_right)
        
        # Batch processing
        B = x_left.shape[0]
        all_tiles = torch.cat(tiles_l + tiles_r, dim=0)
        all_feats = self.backbone(all_tiles) # (2*T*B, D)
        
        # Reshape back
        num_tiles = len(tiles_l)
        all_feats = all_feats.view(2 * num_tiles, B, -1).permute(1, 0, 2) # (B, 2T, D)
        feats_l = all_feats[:, :num_tiles, :]
        feats_r = all_feats[:, num_tiles:, :]
        
        # 2. FiLM (Stereo Fusion)
        if self.use_film:
            ctx_l = feats_l.mean(dim=1)
            ctx_r = feats_r.mean(dim=1)
            gamma_l, beta_l = self.film_left(ctx_r)
            gamma_r, beta_r = self.film_right(ctx_l)
            feats_l = feats_l * (1 + gamma_l.unsqueeze(1)) + beta_l.unsqueeze(1)
            feats_r = feats_r * (1 + gamma_r.unsqueeze(1)) + beta_r.unsqueeze(1)
            
        # 3. Pooling (Standard or Depth-Guided)
        if self.use_depth_attention:
            # DepthGuidedAttention requires original images
            pooled_l = self.attn_pool_left(feats_l, x_left)
            pooled_r = self.attn_pool_right(feats_r, x_right)
        else:
            pooled_l = self.attn_pool_left(feats_l)
            pooled_r = self.attn_pool_right(feats_r)
        
        # 4. Concatenate
        features_list = [pooled_l, pooled_r]
        
        # 5. Depth Features (if enabled)
        if self.use_depth:
            depth_feat = self.depth_module(x_left, x_right)
            features_list.append(depth_feat)
        
        # 6. Project
        f = torch.cat(features_list, dim=1)
        f = self.proj(f)
        return f

    def forward(self, x_left, x_right, species_label=None, targets=None):
        """
        Args:
            x_left: (B, 3, H, W)
            x_right: (B, 3, H, W)
            species_label: (B,) optional, for loss calculation
            targets: (B, num_targets) optional, for loss calculation (raw values)
        """
        f = self.extract_features(x_left, x_right)
        
        # 1. Direct Regression (Predicted Encoded Values)
        pred_enc_direct = self.reg_head(f) # (B, num_targets)
        
        # 2. Species Classification
        species_logits = self.clf_head(f) # (B, num_species)
        species_probs = F.softmax(species_logits, dim=1) # (B, num_species)
        
        # 3. Species-Guided Priors
        # Transform cached raw traits to encoded space on the fly (or cache it)
        # (num_species, num_targets)
        encoded_traits_prior = self.le.transform(self.raw_specie_traits)
        
        # A) Soft Classification Regression
        # Weighted average of priors based on predicted probabilities
        # (B, num_species) @ (num_species, num_targets) -> (B, num_targets)
        pred_enc_soft = torch.matmul(species_probs, encoded_traits_prior)
        
        # B) Hard Classification Regression (Argmax)
        # Use the trait of the most likely species
        pred_species_idx = torch.argmax(species_probs, dim=1)
        pred_enc_hard = encoded_traits_prior[pred_species_idx]
        
        # 4. Blending
        # We blend the ENCODED predictions
        # Weights: (3, num_targets) -> softmax over dim 0 to sum to 1
        blend_w = F.softmax(self.blend_weights, dim=0)
        
        pred_enc_blend = (
            blend_w[0] * pred_enc_direct +
            blend_w[1] * pred_enc_hard +
            blend_w[2] * pred_enc_soft
        )
        
        # Decode to raw space for final output
        pred_raw_direct = self.le.inverse_transform(pred_enc_direct)
        pred_raw_blend = self.le.inverse_transform(pred_enc_blend)
        
        # 5. Loss Calculation (if targets provided)
        loss_dict = {}
        total_loss = 0.0
        
        if targets is not None:
            targets_enc = self.le.transform(targets)
            
            # Regression Loss (on blended output)
            loss_r2 = self.r2_loss(pred_enc_blend, targets_enc)
            loss_dict['r2_loss'] = loss_r2
            
            # Direct Reg Head Loss (auxiliary)
            loss_reg_aux = self.r2_loss(pred_enc_direct, targets_enc)
            loss_dict['reg_aux_loss'] = loss_reg_aux
            
            # Similarity Loss (Cosine Similarity between pred and target in encoded space)
            sim_loss = 1.0 - F.cosine_similarity(pred_enc_blend, targets_enc, dim=1).mean()
            loss_dict['sim_loss'] = sim_loss
            
            total_loss += 1.0 * loss_r2 + 0.5 * loss_reg_aux + 0.2 * sim_loss
            
            if species_label is not None:
                # Classification Loss
                loss_clf = self.clf_loss(species_logits, species_label)
                loss_dict['clf_loss'] = loss_clf
                total_loss += 0.5 * loss_clf # Weight from PlantTraits solution
        
        return {
            'pred': pred_raw_blend,       # Main prediction
            'pred_direct': pred_raw_direct, # Direct head prediction
            'species_logits': species_logits,
            'loss': total_loss,
            'losses': loss_dict
        }

def compute_dataset_stats(df: pd.DataFrame, species_col='Species', target_cols=None):
    """
    Computes Mean/Std for targets and Mean targets per Species.
    """
    if target_cols is None:
        target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
        
    vals = df[target_cols].values
    log_vals = np.log10(vals + 1e-6)
    
    target_mean = torch.tensor(np.mean(log_vals, axis=0), dtype=torch.float32)
    target_std = torch.tensor(np.std(log_vals, axis=0), dtype=torch.float32)
    
    # Species mapping from dataset
    from src.dataset import BiomassDataset
    species_map = BiomassDataset.SPECIES_LABELS
    
    num_species = max(species_map.values()) + 1
    num_targets = len(target_cols)
    specie_traits_mean = torch.zeros(num_species, num_targets)
    
    for species_name, idx in species_map.items():
        mask = df[species_col].map(species_map) == idx
        if mask.sum() > 0:
            avg_vals = df.loc[mask, target_cols].mean().values
            specie_traits_mean[idx] = torch.tensor(avg_vals, dtype=torch.float32)
        else:
            specie_traits_mean[idx] = torch.tensor(df[target_cols].mean().values, dtype=torch.float32)
            
    return target_mean, target_std, specie_traits_mean
