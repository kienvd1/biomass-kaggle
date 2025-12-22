#!/usr/bin/env python3
"""
DINOv3 Direct Model Training Script.

Clean, focused training for DINOv3 with stereo images.

Usage (Single GPU):
    python -m src.dinov3_train --epochs 30

Usage (Multi-GPU with torchrun):
    torchrun --nproc_per_node=2 -m src.dinov3_train --epochs 30

Usage (Multi-GPU with accelerate):
    accelerate launch --num_processes=2 -m src.dinov3_train --epochs 30

Examples:
    # Head-only training (frozen backbone) - DEFAULT
    python -m src.dinov3_train --epochs 30

    # Full training (unfreeze backbone)
    python -m src.dinov3_train --train-backbone --epochs 50

    # Two-stage training (freeze then finetune)
    python -m src.dinov3_train --two-stage --freeze-epochs 10 --epochs 50
    
    # Multi-GPU with PlantHydra loss
    torchrun --nproc_per_node=2 -m src.dinov3_train --use-planthydra-loss --batch-size 32
"""
import argparse
import gc
import json
import os
import random
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .dataset import (
    BiomassDataset,
    create_folds,
    get_train_transforms,
    get_stereo_geometric_transforms,
    get_stereo_photometric_transforms,
    get_valid_transforms,
    prepare_dataframe,
)
from .device import (
    DeviceType,
    empty_cache,
    get_device,
    get_device_type,
    set_device_seed,
    supports_fused_optimizer,
)
from .dinov3_models import (
    DINOv3Direct,
    BiomassLoss,
    PresenceNDVILoss,
    PlantHydraLoss,
    LogTransformEncoder,
    OfficialWeightedR2Loss,
    CrossViewConsistencyLoss,
    SpeciesPriorHead,
    UncertaintyHead,
    UncertaintyAwareLoss,
    QCPlausibilityLoss,
    MassBalanceLoss,
    FeatureOrthogonalityLoss,
    AOSHallucinationLoss,
    InnovativeBiomassLoss,
    StateDensityScaler,
    CalibratedDepthHead,
    TARGET_WEIGHTS,
    compute_rgb_ndvi,
    freeze_backbone,
    unfreeze_backbone,
    count_parameters,
)


# Image sizes by backbone type
IMG_SIZE_DINOV3 = 576  # For grid 3x3 (672/3 = 224px per tile)
IMG_SIZE_DINOV2 = 518  # Native DINOv2 resolution (518/14 = 37 patches)


# =============================================================================
# Distributed Training Helpers
# =============================================================================

def is_dist_initialized() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if is_dist_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_dist_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed() -> Tuple[int, int, torch.device]:
    """
    Setup distributed training environment.
    
    Returns:
        rank: Current process rank
        world_size: Total number of processes
        device: CUDA device for this process
    """
    # Check if launched with torchrun/accelerate
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank == -1:
        # Not distributed
        return 0, 1, None
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return rank, world_size, device


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if is_dist_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes (average)."""
    if not is_dist_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


class AverageMeter:
    """Track running average."""
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int, device_type: DeviceType) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_device_seed(seed, device_type)


def get_layer_wise_params(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    lr_mult: float = 0.8,
) -> list:
    """
    Get parameter groups with layer-wise learning rate decay.
    
    Earlier layers get lower learning rates: lr * (lr_mult ^ depth)
    
    Args:
        model: The model with a 'backbone' attribute containing 'blocks'
        backbone_lr: Base learning rate for the deepest backbone layer
        head_lr: Learning rate for head parameters
        lr_mult: Decay multiplier (0.8 = earlier layers get 0.8x LR)
    
    Returns:
        List of param groups for optimizer
    """
    param_groups = []
    
    # Head parameters (highest LR)
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    
    # If no layer-wise decay, just add all backbone params with backbone_lr
    if lr_mult >= 1.0:
        backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
        return param_groups
    
    # Layer-wise decay for backbone
    # Get the backbone blocks
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
        blocks = model.backbone.blocks
        num_layers = len(blocks)
        
        # Non-block backbone params (patch_embed, norm, etc.) - lowest LR
        non_block_params = []
        for n, p in model.backbone.named_parameters():
            if p.requires_grad and not any(f"blocks.{i}" in n for i in range(num_layers)):
                non_block_params.append(p)
        
        if non_block_params:
            lowest_lr = backbone_lr * (lr_mult ** num_layers)
            param_groups.append({"params": non_block_params, "lr": lowest_lr, "name": "backbone_base"})
        
        # Each block gets progressively higher LR
        for i, block in enumerate(blocks):
            block_params = [p for p in block.parameters() if p.requires_grad]
            if block_params:
                # Layer i from end = num_layers - 1 - i
                depth_from_end = num_layers - 1 - i
                layer_lr = backbone_lr * (lr_mult ** depth_from_end)
                param_groups.append({"params": block_params, "lr": layer_lr, "name": f"block_{i}"})
    else:
        # Fallback: all backbone params with same LR
        backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    
    return param_groups


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    grad_clip: float = 1.0,
    epoch: int = 0,
    show_progress: bool = True,
    use_aux_heads: bool = False,
    use_presence_heads: bool = False,
    use_ndvi_head: bool = False,
    use_height_head: bool = False,
    use_species_head: bool = False,
    use_planthydra: bool = False,
    use_state_head: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch. Returns (total_loss, loss_breakdown_dict)."""
    model.train()
    losses = AverageMeter()
    # Track individual loss components
    loss_components: Dict[str, AverageMeter] = {}
    
    # MPS optimizations
    use_channels_last = device_type == DeviceType.MPS
    
    pbar = tqdm(loader, desc=f"Ep {epoch:02d} Train", leave=False, disable=not show_progress)
    
    for batch in pbar:
        x_left, x_right, targets = batch[:3]
        
        # Get auxiliary labels from batch
        state_labels = None
        month_labels = None
        species_labels = None
        ndvi_target = None
        height_target = None
        
        if len(batch) >= 6:
            state_labels = batch[3].to(device, non_blocking=True)
            month_labels = batch[4].to(device, non_blocking=True)
            species_labels = batch[5].to(device, non_blocking=True)
        if len(batch) >= 8:
            ndvi_target = batch[6].to(device, non_blocking=True)
            height_target = batch[7].to(device, non_blocking=True)
        
        # Use channels_last for MPS performance
        if use_channels_last:
            x_left = x_left.to(device, non_blocking=True, memory_format=torch.channels_last)
            x_right = x_right.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_left = x_left.to(device, non_blocking=True)
            x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        if use_planthydra:
            # PlantHydra mode: use combined loss with all features
            outputs = model(x_left, x_right)
            green, dead, clover, gdm, total, aux_loss = outputs[:6]
            
            # Get optional outputs
            dead_presence_logit = outputs[6] if len(outputs) > 6 else None
            clover_presence_logit = outputs[7] if len(outputs) > 7 else None
            ndvi_pred = outputs[8] if len(outputs) > 8 else None
            height_pred = outputs[9] if len(outputs) > 9 else None
            species_logits = outputs[10] if len(outputs) > 10 else None
            
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            
            # PlantHydraLoss with all auxiliary targets
            loss, loss_dict = loss_fn(
                preds, targets,
                height_pred=height_pred,
                height_target=height_target if use_height_head else None,
                ndvi_pred=ndvi_pred,
                ndvi_target=ndvi_target if use_ndvi_head else None,
                species_logits=species_logits,
                species_labels=species_labels if use_species_head else None,
                state_logits=None,  # TODO: Add state head to model if needed
                state_labels=state_labels if use_state_head else None,
            )
            loss = loss + aux_loss
            
            # Track individual loss components
            for key, val in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = AverageMeter()
                loss_components[key].update(val, x_left.size(0))
        elif use_aux_heads:
            green, dead, clover, gdm, total, aux_loss, state_logits, month_logits, species_logits = model(x_left, x_right, return_aux=True)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            loss, _ = loss_fn(preds, targets, state_logits, state_labels, month_logits, month_labels, species_logits, species_labels)
            loss = loss + aux_loss
        elif use_presence_heads or use_ndvi_head or use_height_head or use_species_head:
            # Model returns: green, dead, clover, gdm, total, aux_loss, dead_presence_logit, clover_presence_logit, ndvi_pred, height_pred, species_logits
            outputs = model(x_left, x_right)
            green, dead, clover, gdm, total, aux_loss = outputs[:6]
            dead_presence_logit = outputs[6] if len(outputs) > 6 else None
            clover_presence_logit = outputs[7] if len(outputs) > 7 else None
            ndvi_pred = outputs[8] if len(outputs) > 8 else None
            height_pred = outputs[9] if len(outputs) > 9 else None
            species_logits_out = outputs[10] if len(outputs) > 10 else None
            
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            
            loss, loss_dict = loss_fn(
                preds, targets,
                dead_presence_logit=dead_presence_logit,
                clover_presence_logit=clover_presence_logit,
                ndvi_pred=ndvi_pred,
                ndvi_target=ndvi_target if use_ndvi_head else None,
                height_pred=height_pred,
                height_target=height_target if use_height_head else None,
                species_logits=species_logits_out,
                species_labels=species_labels if use_species_head else None,
            )
            loss = loss + aux_loss
            
            # Track individual loss components
            for key, val in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = AverageMeter()
                loss_components[key].update(val, x_left.size(0))
        else:
            green, dead, clover, gdm, total, aux_loss = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            # Base loss might also return tuple
            loss_result = loss_fn(preds, targets)
            if isinstance(loss_result, tuple):
                loss, loss_dict = loss_result
                for key, val in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = AverageMeter()
                    loss_components[key].update(val, x_left.size(0))
            else:
                loss = loss_result
            loss = loss + aux_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        losses.update(loss.item(), x_left.size(0))
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})
    
    # Sync MPS before returning to prevent delay in next phase
    if device_type == DeviceType.MPS:
        torch.mps.synchronize()
    
    # Build loss breakdown dict
    loss_breakdown = {key: meter.avg for key, meter in loss_components.items()}
    
    return losses.avg, loss_breakdown


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    epoch: int = 0,
    show_progress: bool = True,
    use_aux_heads: bool = False,
    use_log_transform: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """Validate model with official competition metric (log-space R²)."""
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []
    
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    target_names = ["green", "dead", "clover", "gdm", "total"]
    
    # For validation, use simple MSE if PlantHydraLoss
    # PlantHydraLoss returns (loss, loss_dict), but for validation we just want a scalar
    is_planthydra = isinstance(loss_fn, PlantHydraLoss)
    
    # MPS optimizations
    use_channels_last = device_type == DeviceType.MPS
    
    pbar = tqdm(loader, desc=f"Ep {epoch:02d} Valid", leave=False, disable=not show_progress)
    
    for batch in pbar:
        x_left, x_right, targets = batch[:3]
        
        # Use channels_last for MPS performance
        if use_channels_last:
            x_left = x_left.to(device, non_blocking=True, memory_format=torch.channels_last)
            x_right = x_right.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_left = x_left.to(device, non_blocking=True)
            x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass (no aux needed for validation metrics)
        # Model may return extra values for presence/ndvi/height/species heads
        outputs = model(x_left, x_right, return_aux=False)
        green, dead, clover, gdm, total = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
        preds = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        # Compute validation loss
        if is_planthydra:
            loss, _ = loss_fn(preds, targets)
        elif hasattr(loss_fn, 'base_loss'):
            loss = loss_fn.base_loss(preds, targets)
        else:
            loss = loss_fn(preds, targets)
        
        losses.update(loss.item(), x_left.size(0))
        # Detach and move to CPU immediately to free GPU memory
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})
    
    # Concatenate once
    all_preds_t = torch.cat(all_preds, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    
    # NOTE: No gather needed - all GPUs validate on full set (no DistributedSampler for validation)
    # This avoids the duplicate sample bug from DistributedSampler padding
    
    # Clamp predictions to non-negative for R² computation
    all_preds_t = all_preds_t.clamp(min=0)
    
    # Compute per-target R² for logging
    metrics: Dict[str, float] = {}
    
    # ============================================================================
    # OFFICIAL COMPETITION METRIC: R² in ORIGINAL space (not log-transformed)
    # Weights: Green=0.1, Dead=0.1, Clover=0.1, GDM=0.2, Total=0.5
    # ============================================================================
    for i, name in enumerate(target_names):
        pred_i = all_preds_t[:, i]
        target_i = all_targets_t[:, i]
        
        # R² = 1 - SS_res / SS_tot
        ss_res = ((pred_i - target_i) ** 2).sum()
        ss_tot = ((target_i - target_i.mean()) ** 2).sum()
        r2_i = 1 - (ss_res / (ss_tot + 1e-8))
        metrics[f"r2_{name}"] = float(r2_i.item())
    
    # Compute weighted average of per-target R² 
    avg_r2 = sum(target_weights[i] * metrics[f"r2_{name}"] for i, name in enumerate(target_names))
    metrics["avg_r2"] = avg_r2  # sum of weights = 1.0
    
    # Compute GLOBAL weighted R² (official competition metric)
    # R²_w = 1 - SS_res / SS_tot where weights are applied per (sample, target) pair
    n_samples = all_preds_t.size(0)
    weights_t = torch.tensor(target_weights, device=all_preds_t.device)
    weights_expanded = weights_t.unsqueeze(0).expand(n_samples, -1)  # (N, 5)
    
    # Flatten for global computation
    w = weights_expanded.flatten()  # (N * 5,)
    y = all_targets_t.flatten()  # (N * 5,)
    y_hat = all_preds_t.flatten()  # (N * 5,)
    
    # Weighted mean of targets
    y_bar_w = (w * y).sum() / w.sum()
    
    # Global weighted R²
    ss_res = (w * (y - y_hat) ** 2).sum()
    ss_tot = (w * (y - y_bar_w) ** 2).sum()
    weighted_r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    metrics["weighted_r2"] = float(weighted_r2.item())
    
    # Also compute R² in log-space for reference (if training uses log transform)
    if use_log_transform:
        all_preds_log = torch.log1p(all_preds_t)
        all_targets_log = torch.log1p(all_targets_t)
        for i, name in enumerate(target_names):
            pred_i = all_preds_log[:, i]
            target_i = all_targets_log[:, i]
            ss_res = ((pred_i - target_i) ** 2).sum()
            ss_tot = ((target_i - target_i.mean()) ** 2).sum()
            r2_i = 1 - (ss_res / (ss_tot + 1e-8))
            metrics[f"r2_{name}_log"] = float(r2_i.item())
    
    # Sync MPS before returning
    if device_type == DeviceType.MPS:
        torch.mps.synchronize()
    
    return losses.avg, metrics["avg_r2"], metrics


# Per-fold seeds for reproducibility (each fold starts from known state)
# Now uses cfg.seed for all folds (consistent training)
FOLD_SEEDS = [18, 18, 18, 18, 18]  # Kept for backward compatibility


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    cfg: argparse.Namespace,
    device: torch.device,
    device_type: DeviceType,
    img_size: int = 672,
) -> Dict:
    """Train a single fold."""
    # Reset seed for this fold - use config seed for consistency
    fold_seed = cfg.seed
    set_seed(fold_seed, device_type)
    
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Fold {fold} (seed={fold_seed})")
        print(f"{'='*60}")
        print(f"Train: {len(train_df)} | Valid: {len(valid_df)}")
    
    heads_info = "Heads: Total, Green, GDM"
    if cfg.train_dead:
        heads_info += ", Dead"
    if cfg.train_clover:
        heads_info += ", Clover"
    if is_main_process():
        print(heads_info)
    
    # Transforms - choose photometric mode
    photometric_mode = cfg.photometric
    strong_aug = getattr(cfg, 'strong_aug', False)
    no_lighting = getattr(cfg, 'no_lighting', False)
    use_perspective_jitter = getattr(cfg, 'use_perspective_jitter', False)
    use_border_mask = getattr(cfg, 'use_border_mask', False)
    
    if is_main_process():
        if strong_aug:
            print("Augmentation: STRONG (from dinov3-5tar.ipynb)")
        if no_lighting:
            print("Augmentation: NO LIGHTING (ColorJitter, HSV, Brightness, CLAHE removed)")
        if use_perspective_jitter:
            print("Augmentation: +Perspective jitter (annotation noise sim)")
        if use_border_mask:
            print("Augmentation: +Border masking (frame shortcut prevention)")
    
    if photometric_mode == "same":
        # Same geometric + photometric to both L/R (default)
        train_transform = get_train_transforms(
            img_size, cfg.aug_prob, strong=strong_aug, no_lighting=no_lighting,
            use_perspective_jitter=use_perspective_jitter, 
            use_border_mask=use_border_mask
        )
        photometric_transform = None
        photometric_left_only = False
        photometric_right_only = False
        if is_main_process():
            print("Photometric: SAME to both L/R")
    elif photometric_mode == "independent":
        # Geometric same, photometric independent
        train_transform = get_stereo_geometric_transforms(img_size, cfg.aug_prob, strong=strong_aug)
        photometric_transform = get_stereo_photometric_transforms(cfg.aug_prob, strong=strong_aug)
        photometric_left_only = False
        photometric_right_only = False
        if is_main_process():
            print("Photometric: INDEPENDENT to L/R")
    elif photometric_mode == "left":
        # Only apply photometric to left
        train_transform = get_stereo_geometric_transforms(img_size, cfg.aug_prob, strong=strong_aug)
        photometric_transform = get_stereo_photometric_transforms(cfg.aug_prob, strong=strong_aug)
        photometric_left_only = True
        photometric_right_only = False
        if is_main_process():
            print("Photometric: LEFT only")
    elif photometric_mode == "right":
        # Only apply photometric to right
        train_transform = get_stereo_geometric_transforms(img_size, cfg.aug_prob, strong=strong_aug)
        photometric_transform = get_stereo_photometric_transforms(cfg.aug_prob, strong=strong_aug)
        photometric_left_only = False
        photometric_right_only = True
        if is_main_process():
            print("Photometric: RIGHT only")
    else:  # none
        # Geometric only, no photometric
        train_transform = get_stereo_geometric_transforms(img_size, cfg.aug_prob, strong=strong_aug)
        photometric_transform = None
        photometric_left_only = False
        photometric_right_only = False
        if is_main_process():
            print("Photometric: NONE (geometric only)")
    
    valid_transform = get_valid_transforms(img_size)
    
    # Datasets
    use_aux_heads = getattr(cfg, 'use_aux_heads', False)
    # Need aux labels if using aux heads OR NDVI/Height/Species heads (for ground-truth targets)
    need_aux_labels = (use_aux_heads or getattr(cfg, 'use_ndvi_head', False) or 
                       getattr(cfg, 'use_height_head', False) or getattr(cfg, 'use_species_head', False))
    mix_same_context = not getattr(cfg, 'no_mix_same_context', False)
    train_ds = BiomassDataset(
        train_df, cfg.image_dir, train_transform,
        is_train=True,
        return_aux_labels=need_aux_labels,
        stereo_swap_prob=cfg.stereo_swap_prob,
        photometric_transform=photometric_transform,
        photometric_left_only=photometric_left_only,
        photometric_right_only=photometric_right_only,
        mixup_prob=cfg.mixup_prob,
        mixup_alpha=cfg.mixup_alpha,
        cutmix_prob=cfg.cutmix_prob,
        cutmix_alpha=cfg.cutmix_alpha,
        mix_same_context=mix_same_context,
    )
    valid_ds = BiomassDataset(
        valid_df, cfg.image_dir, valid_transform,
        is_train=False,
        return_aux_labels=need_aux_labels,
    )
    
    # Print augmentation info (main process only)
    if is_main_process():
        if cfg.stereo_swap_prob > 0:
            print(f"Stereo swap: {cfg.stereo_swap_prob:.0%}")
        if cfg.mixup_prob > 0:
            context_str = "any sample" if getattr(cfg, 'no_mix_same_context', False) else "same context only"
            print(f"MixUp: prob={cfg.mixup_prob:.0%}, alpha={cfg.mixup_alpha} ({context_str})")
        if cfg.cutmix_prob > 0:
            context_str = "any sample" if getattr(cfg, 'no_mix_same_context', False) else "same context only"
            print(f"CutMix: prob={cfg.cutmix_prob:.0%}, alpha={cfg.cutmix_alpha} ({context_str})")
        if cfg.use_learnable_aug:
            aug_types = []
            if cfg.learnable_aug_color:
                aug_types.append("color")
            if cfg.learnable_aug_spatial:
                aug_types.append("spatial")
            print(f"Learnable Aug: {'+'.join(aug_types)}")
    
    # Create samplers for distributed training
    # NOTE: Only use DistributedSampler for training, not validation
    # DistributedSampler pads with duplicates which corrupts validation metrics
    is_distributed = getattr(cfg, 'distributed', False)
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
    else:
        train_sampler = None
    valid_sampler = None  # All GPUs validate on full set, report from rank 0
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers, drop_last=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size * 2, 
        shuffle=False,
        sampler=valid_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    # Model
    use_aux_heads = getattr(cfg, 'use_aux_heads', False)
    model = DINOv3Direct(
        grid=(cfg.grid, cfg.grid),
        pretrained=True,
        dropout=cfg.dropout,
        hidden_ratio=cfg.hidden_ratio,
        use_film=cfg.use_film,
        use_attention_pool=cfg.use_attention_pool,
        train_dead=cfg.train_dead,
        train_clover=cfg.train_clover,
        use_vegetation_indices=cfg.use_vegetation_indices,
        use_disparity=cfg.use_disparity,
        use_depth=cfg.use_depth,
        depth_model_size=cfg.depth_model_size,
        use_depth_attention=cfg.depth_attention,
        use_learnable_aug=cfg.use_learnable_aug,
        learnable_aug_color=cfg.learnable_aug_color,
        learnable_aug_spatial=cfg.learnable_aug_spatial,
        use_aux_heads=use_aux_heads,
        backbone_size=cfg.backbone_size,
        backbone_type=getattr(cfg, 'backbone_type', 'dinov3'),
        ckpt_path=getattr(cfg, 'ckpt_path', None),
        use_whole_image=cfg.use_whole_image,
        use_presence_heads=cfg.use_presence_heads,
        use_ndvi_head=cfg.use_ndvi_head,
        use_height_head=cfg.use_height_head,
        use_species_head=cfg.use_species_head,
    ).to(device)
    
    # Use channels_last memory format for MPS (faster convolutions)
    if device_type == DeviceType.MPS:
        model = model.to(memory_format=torch.channels_last)
    
    # Wrap model in DDP for distributed training
    is_distributed = getattr(cfg, 'distributed', False)
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # find_unused_parameters=True needed for depth model and optional aux heads
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    
    # Loss
    use_smoothl1 = getattr(cfg, 'smoothl1', False)
    use_presence = getattr(cfg, 'use_presence_heads', False)
    use_ndvi = getattr(cfg, 'use_ndvi_head', False)
    use_height = getattr(cfg, 'use_height_head', False)
    use_species = getattr(cfg, 'use_species_head', False)
    use_tweedie = getattr(cfg, 'use_tweedie', False)
    use_planthydra = getattr(cfg, 'use_planthydra_loss', False)
    use_log_transform = getattr(cfg, 'use_log_transform', False)
    use_cosine_sim = getattr(cfg, 'use_cosine_sim', False)
    use_compositional = getattr(cfg, 'use_compositional', False)
    use_state_head = getattr(cfg, 'use_state_head', False)
    
    # PlantHydra-style loss (best option for official metric)
    if use_planthydra or use_log_transform or use_cosine_sim or use_compositional:
        # Determine if using log transform (default True for planthydra, can override with --no-log-transform)
        planthydra_use_log = (use_log_transform or use_planthydra) and not getattr(cfg, 'no_log_transform', False)
        loss_fn = PlantHydraLoss(
            use_log_transform=planthydra_use_log,
            use_smoothl1=getattr(cfg, 'smoothl1', False),
            smoothl1_beta=getattr(cfg, 'smoothl1_beta', 1.0),
            use_cosine_sim=use_cosine_sim or use_planthydra,
            cosine_weight=getattr(cfg, 'cosine_weight', 0.4),
            use_compositional=use_compositional or use_planthydra,
            compositional_weight=getattr(cfg, 'compositional_weight', 0.1),
            use_height_aux=use_height,
            height_weight=getattr(cfg, 'height_weight', 0.2),
            use_ndvi_aux=use_ndvi,
            ndvi_weight=getattr(cfg, 'ndvi_weight', 0.2),
            use_species_aux=use_species,
            species_weight=getattr(cfg, 'aux_species_weight', 0.1),
            use_state_aux=use_state_head,
            state_weight=getattr(cfg, 'state_weight', 0.1),
            train_dead=cfg.train_dead,
            train_clover=cfg.train_clover,
        )
        extras_loss = ["PlantHydra"]
        if planthydra_use_log:  # Fixed: use actual log setting
            extras_loss.append("LogTransform")
        else:
            extras_loss.append("RawValues")
        if use_cosine_sim or use_planthydra:
            extras_loss.append(f"CosineSim(w={getattr(cfg, 'cosine_weight', 0.4):.2f})")
        if use_compositional or use_planthydra:
            extras_loss.append(f"Compositional(w={getattr(cfg, 'compositional_weight', 0.1):.2f})")
        if use_height:
            extras_loss.append(f"Height(w={cfg.height_weight:.1f})")
        if use_ndvi:
            extras_loss.append(f"NDVI(w={cfg.ndvi_weight:.1f})")
        if use_species:
            extras_loss.append(f"Species(w={cfg.aux_species_weight:.1f})")
        if use_state_head:
            extras_loss.append(f"State(w={cfg.state_weight:.1f})")
        if is_main_process():
            print(f"Loss: {' + '.join(extras_loss)}")
    else:
        base_loss_fn = BiomassLoss(
            use_huber_for_dead=cfg.use_huber,
            huber_delta=cfg.huber_delta,
            train_dead=cfg.train_dead,
            train_clover=cfg.train_clover,
            smoothl1_mode=use_smoothl1,
        )
        
        # Wrap with PresenceNDVILoss if any of those features are enabled
        if use_presence or use_ndvi or use_height or use_species or use_tweedie:
            loss_fn = PresenceNDVILoss(
                base_loss=base_loss_fn,
                use_presence=use_presence,
                use_ndvi=use_ndvi,
                use_height=use_height,
                use_species=use_species,
                use_tweedie=use_tweedie,
                tweedie_p=getattr(cfg, 'tweedie_p', 1.5),
                presence_weight=getattr(cfg, 'presence_weight', 0.5),
                ndvi_weight=getattr(cfg, 'ndvi_weight', 0.3),
                height_weight=getattr(cfg, 'height_weight', 0.3),
                species_weight=getattr(cfg, 'aux_species_weight', 0.5),
            )
            extras_loss = []
            if use_presence:
                extras_loss.append("Presence Heads")
            if use_ndvi:
                extras_loss.append("NDVI Head (GT)")
            if use_height:
                extras_loss.append("Height Head")
            if use_species:
                extras_loss.append("Species Head")
            if use_tweedie:
                extras_loss.append(f"Tweedie(p={cfg.tweedie_p})")
            if is_main_process():
                print(f"Enhanced loss: {', '.join(extras_loss)}")
        elif use_aux_heads:
            from src.dinov3_models import AuxiliaryBiomassLoss
            loss_fn = AuxiliaryBiomassLoss(
                base_loss=base_loss_fn,
                state_weight=getattr(cfg, 'aux_state_weight', 1.0),
                month_weight=getattr(cfg, 'aux_month_weight', 1.0),
                species_weight=getattr(cfg, 'aux_species_weight', 1.0),
            )
            if is_main_process():
                print(f"Using auxiliary heads: State={cfg.aux_state_weight}, Month={cfg.aux_month_weight}, Species={cfg.aux_species_weight}")
        else:
            loss_fn = base_loss_fn
    
    # Training mode setup
    use_fused = supports_fused_optimizer(device_type)
    
    if cfg.freeze_backbone:
        # Head-only training
        freeze_backbone(model)
        trainable = count_parameters(model, trainable_only=True)
        total_params = count_parameters(model, trainable_only=False)
        if is_main_process():
            print(f"Mode: HEAD-ONLY (frozen backbone)")
            print(f"Params: {trainable:,} trainable / {total_params:,} total")
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=use_fused)
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])
        stage = 1
        
    elif cfg.two_stage:
        # Two-stage: freeze first, then finetune
        freeze_backbone(model)
        stage1_patience = cfg.stage1_patience if cfg.stage1_patience else cfg.freeze_epochs
        stage2_patience = cfg.stage2_patience if cfg.stage2_patience else cfg.patience
        stage2_epochs = cfg.stage2_epochs if cfg.stage2_epochs else (cfg.epochs - cfg.freeze_epochs)
        
        if is_main_process():
            if cfg.stage1_patience:
                print(f"Mode: TWO-STAGE (Stage 1: patience={stage1_patience}, Stage 2: {stage2_epochs} ep, patience={stage2_patience})")
            else:
                print(f"Mode: TWO-STAGE (Stage 1: {cfg.freeze_epochs} ep, Stage 2: {stage2_epochs} ep)")
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=use_fused)
        # For Stage 1: use freeze_epochs as expected duration (not total epochs)
        # This ensures LR decays properly even with patience-based early stopping
        stage1_expected = cfg.freeze_epochs  # Expected ~30 epochs for Stage 1
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=min(3, stage1_expected))
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, stage1_expected - 3), eta_min=1e-7)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[min(3, stage1_expected)])
        stage = 1
        
    else:
        # Full training from start
        lr_mult = getattr(cfg, 'lr_mult', 1.0)
        if is_main_process():
            if lr_mult < 1.0:
                print(f"Mode: FULL (layer-wise LR, mult={lr_mult})")
            else:
                print(f"Mode: FULL (train everything)")
            print(f"Params: {count_parameters(model):,}")
        
        # Use layer-wise LR if lr_mult < 1.0
        if lr_mult < 1.0:
            param_groups = get_layer_wise_params(
                model,
                backbone_lr=cfg.backbone_lr,
                head_lr=cfg.lr,
                lr_mult=lr_mult,
            )
            optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay, fused=use_fused)
        else:
            # Differential learning rates (2 groups)
            backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
            head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
            
            optimizer = AdamW([
                {"params": head_params, "lr": cfg.lr},
                {"params": backbone_params, "lr": cfg.backbone_lr},
            ], weight_decay=cfg.weight_decay, fused=use_fused)
        
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])
        stage = 2
    
    # Training loop
    best_r2 = -float("inf")
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    best_epoch = 0
    
    # Stage-specific tracking for two-stage training
    stage1_best_r2 = -float("inf")
    stage1_best_epoch = 0
    stage2_epoch_offset = 0  # Track total epochs when Stage 2 starts
    
    # Temp paths for checkpoints
    temp_ckpt_path = os.path.join(cfg.output_dir, f"_temp_fold{fold}.pth") if cfg.output_dir else None
    stage1_ckpt_path = os.path.join(cfg.output_dir, f"_stage1_fold{fold}.pth") if cfg.output_dir else None
    
    # Determine patience for current stage
    if cfg.two_stage:
        current_patience = cfg.stage1_patience if cfg.stage1_patience else cfg.freeze_epochs
        stage1_max_epochs = cfg.epochs if cfg.stage1_patience else cfg.freeze_epochs
        stage2_patience = cfg.stage2_patience if cfg.stage2_patience else cfg.patience
        stage2_epochs = cfg.stage2_epochs if cfg.stage2_epochs else (cfg.epochs - cfg.freeze_epochs)
    else:
        current_patience = cfg.patience
    
    epoch = 0
    while True:
        epoch += 1
        
        # Set epoch on sampler for proper shuffling in distributed mode
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Check epoch limits
        if cfg.two_stage:
            if stage == 1:
                # Stage 1: check max epochs (if not using patience-based)
                if not cfg.stage1_patience and epoch > cfg.freeze_epochs:
                    # Transition to Stage 2
                    pass  # Will be handled below
                elif cfg.stage1_patience and epoch > cfg.epochs:
                    # Safety limit
                    break
            else:  # stage == 2
                stage2_current = epoch - stage2_epoch_offset
                if stage2_current > stage2_epochs:
                    break
        else:
            if epoch > cfg.epochs:
                break
        
        # Stage transition for two-stage (epoch-based or early-stop based)
        should_transition = False
        if cfg.two_stage and stage == 1:
            if cfg.stage1_patience:
                # Patience-based: transition when Stage 1 early stops
                if patience_counter >= current_patience:
                    should_transition = True
                    if is_main_process():
                        print(f"\n>>> Stage 1 converged (best epoch: {stage1_best_epoch}, aR²={stage1_best_r2:.4f})")
            else:
                # Epoch-based: transition at fixed epoch
                if epoch == cfg.freeze_epochs + 1:
                    should_transition = True
        
        if should_transition:
            lr_mult = getattr(cfg, 'lr_mult', 1.0)
            if is_main_process():
                if lr_mult < 1.0:
                    print(f">>> Stage 2: Unfreezing backbone (layer-wise LR, mult={lr_mult})")
                else:
                    print(f">>> Stage 2: Unfreezing backbone (lr={cfg.backbone_lr:.2e})")
            stage = 2
            stage2_epoch_offset = epoch - 1  # So stage2 epoch 1 = current epoch
            
            # Load best Stage 1 checkpoint (barrier ensures rank 0 finished saving)
            if is_dist_initialized():
                dist.barrier()
            if stage1_ckpt_path and os.path.exists(stage1_ckpt_path):
                if is_main_process():
                    print(f">>> Loading best Stage 1 checkpoint (epoch {stage1_best_epoch})")
                model.load_state_dict(torch.load(stage1_ckpt_path, map_location=device, weights_only=True))
            
            unfreeze_backbone(model)
            
            # Use layer-wise LR if lr_mult < 1.0
            if lr_mult < 1.0:
                param_groups = get_layer_wise_params(
                    model, 
                    backbone_lr=cfg.backbone_lr,
                    head_lr=cfg.lr * 0.5,  # Reduce head LR
                    lr_mult=lr_mult,
                )
                optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay, fused=use_fused)
            else:
                backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
                head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
                
                optimizer = AdamW([
                    {"params": head_params, "lr": cfg.lr * 0.5},  # Reduce head LR
                    {"params": backbone_params, "lr": cfg.backbone_lr},
                ], weight_decay=cfg.weight_decay, fused=use_fused)
            
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=max(1, stage2_epochs - cfg.warmup_epochs), eta_min=1e-7)
            scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])
            
            # Reset patience for Stage 2
            patience_counter = 0
            current_patience = stage2_patience
            
            # Keep best from Stage 1 as baseline
            # (Stage 2 must beat Stage 1's best to save new checkpoint)
        
        train_loss, loss_breakdown = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, device_type, cfg.grad_clip,
            epoch=epoch, use_aux_heads=use_aux_heads,
            use_presence_heads=use_presence, use_ndvi_head=use_ndvi, 
            use_height_head=use_height, use_species_head=use_species,
            use_planthydra=use_planthydra, use_state_head=use_state_head
        )
        val_loss, r2, metrics = validate(
            model, valid_loader, loss_fn, device, device_type, epoch=epoch, 
            use_aux_heads=use_aux_heads, use_log_transform=use_log_transform or use_planthydra
        )
        scheduler.step()
        
        # Track memory usage
        if device_type == DeviceType.MPS:
            mem_alloc = torch.mps.current_allocated_memory() / 1e9
            mem_driver = torch.mps.driver_allocated_memory() / 1e9
            mem_info = f" | Mem: {mem_alloc:.1f}/{mem_driver:.1f}GB"
        elif device_type == DeviceType.CUDA:
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            mem_info = f" | Mem: {mem_alloc:.1f}/{mem_reserved:.1f}GB"
        else:
            mem_info = ""
        
        improved = ""
        if r2 > best_r2 and not np.isnan(r2):
            best_r2 = r2
            best_metrics = metrics.copy()
            best_epoch = epoch
            patience_counter = 0
            improved = " *"
            # Save best checkpoint (only on main process to avoid race)
            if temp_ckpt_path and is_main_process():
                torch.save(model.state_dict(), temp_ckpt_path)
            # Also save as Stage 1 best if in Stage 1
            if cfg.two_stage and stage == 1:
                stage1_best_r2 = r2
                stage1_best_epoch = epoch
                if stage1_ckpt_path and is_main_process():
                    torch.save(model.state_dict(), stage1_ckpt_path)
        else:
            patience_counter += 1
        
        # Per-target R²
        per_target = " | ".join([
            f"G={metrics['r2_green']:.3f}",
            f"D={metrics['r2_dead']:.3f}",
            f"C={metrics['r2_clover']:.3f}",
            f"GDM={metrics['r2_gdm']:.3f}",
            f"T={metrics['r2_total']:.3f}",
        ])
        
        stage_info = f"[S{stage}] " if cfg.two_stage else ""
        
        # Format loss breakdown if available
        loss_breakdown_str = ""
        if loss_breakdown:
            # Separate biomass loss from auxiliary losses
            biomass_loss = loss_breakdown.get('biomass', 0)
            aux_parts = []
            for key, val in loss_breakdown.items():
                if key != 'biomass':
                    aux_parts.append(f"{key}={val:.3f}")
            if aux_parts:
                loss_breakdown_str = f" (bio={biomass_loss:.3f}, {', '.join(aux_parts)})"
        
        # Only print on main process to avoid interleaved output
        if is_main_process():
            print(f"  {stage_info}Ep {epoch:02d}: loss={train_loss:.4f}/{val_loss:.4f}{loss_breakdown_str} R²={r2:.4f} [{per_target}]{mem_info}{improved}", flush=True)
        
        # Early stopping
        if np.isnan(train_loss) or np.isnan(val_loss):
            if is_main_process():
                print("  NaN detected, stopping", flush=True)
            break
        
        # Check patience (stage-aware)
        if cfg.two_stage and stage == 1 and cfg.stage1_patience:
            # Stage 1 with patience: will transition, not break
            pass
        elif patience_counter >= current_patience:
            stage_name = f"Stage {stage}" if cfg.two_stage else "Training"
            if is_main_process():
                print(f"  {stage_name} early stop (no improvement for {current_patience} epochs)", flush=True)
            break
    
    # Sync all ranks before cleanup
    if is_dist_initialized():
        dist.barrier()
    
    # Cleanup Stage 1 checkpoint (only on main process)
    if stage1_ckpt_path and is_main_process():
        try:
            os.remove(stage1_ckpt_path)
        except FileNotFoundError:
            pass
    
    # Rename temp checkpoint to final name (only on main process)
    if temp_ckpt_path and is_main_process():
        save_path = os.path.join(cfg.output_dir, f"dinov3_best_fold{fold}.pth")
        try:
            os.rename(temp_ckpt_path, save_path)
            print(f"  Saved: {save_path} (best epoch: {best_epoch}, R²={best_r2:.4f})")
        except FileNotFoundError:
            pass
    
    # Cleanup
    del model, optimizer, scheduler
    empty_cache(device_type)
    gc.collect()
    
    return {
        "fold": fold,
        "best_r2": best_r2,
        "metrics": best_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 Direct Model Training")
    
    # Data
    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--fold-csv", type=str, default="data/trainfold_group_month_species.csv",
                        help="Path to CSV with predefined folds (sample_id_prefix, fold)")
    parser.add_argument("--use-predefined-folds", action="store_true", default=True,
                        help="Use predefined folds from --fold-csv (default: True)")
    parser.add_argument("--no-predefined-folds", action="store_false", dest="use_predefined_folds",
                        help="Create new folds using --cv-strategy instead of predefined")
    parser.add_argument("--cv-strategy", type=str, default="group_location",
                        help="Strategy for creating folds (only used with --no-predefined-folds). "
                             "group_location=RECOMMENDED (State×Season×Species stratification)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Specific folds to train (default: all)")
    
    # Architecture
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=None,
                        help="Image size (default: 672 for DINOv3, 518 for DINOv2). Must be divisible by 16.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-ratio", type=float, default=0.25)
    parser.add_argument("--backbone-size", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="DINOv3 backbone size (small=384d, base=768d, large=1024d)")
    parser.add_argument("--backbone-type", type=str, default="dinov3",
                        choices=["dinov3", "dinov2"],
                        help="Backbone type: dinov3 (default) or dinov2 (for PlantCLEF weights)")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Path to pretrained weights (.safetensors for PlantCLEF)")
    parser.add_argument("--use-whole-image", action="store_true",
                        help="Add whole-image branch for global context (3-view: left, right, whole)")
    parser.add_argument("--use-film", action="store_true", default=True)
    parser.add_argument("--no-film", action="store_false", dest="use_film")
    parser.add_argument("--use-attention-pool", action="store_true", default=True)
    parser.add_argument("--no-attention-pool", action="store_false", dest="use_attention_pool")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save ~30%% memory (slower training)")
    
    # Optional heads (default: derive Dead from Total - Green - Clover)
    parser.add_argument("--train-dead", action="store_true", default=False,
                        help="Add head to train Dead directly (instead of deriving)")
    parser.add_argument("--no-train-dead", action="store_false", dest="train_dead",
                        help="Disable Dead head, derive from Total - Green - Clover (default)")
    parser.add_argument("--train-clover", action="store_true", default=True,
                        help="Add head to train Clover directly (instead of deriving)")
    parser.add_argument("--no-train-clover", action="store_false", dest="train_clover",
                        help="Disable Clover head, derive from GDM - Green - Dead")
    
    # Innovative features
    parser.add_argument("--use-vegetation-indices", action="store_true",
                        help="Add vegetation indices (ExG, ExR, GRVI) as features")
    parser.add_argument("--use-disparity", action="store_true",
                        help="Add stereo disparity features (3D volume exploitation)")
    parser.add_argument("--use-depth", action="store_true", default=True,
                        help="Add Depth Anything V2 depth features (r=0.63 for green!)")
    parser.add_argument("--no-use-depth", action="store_false", dest="use_depth",
                        help="Disable depth features")
    parser.add_argument("--depth-model-size", type=str, default="small",
                        choices=["small", "base"],
                        help="Depth Anything model size (small=faster, base=better)")
    parser.add_argument("--depth-attention", action="store_true", default=True,
                        help="Use depth-guided attention for tile pooling (weights tiles by depth)")
    parser.add_argument("--no-depth-attention", action="store_false", dest="depth_attention",
                        help="Disable depth-guided attention")
    parser.add_argument("--use-learnable-aug", action="store_true",
                        help="Enable learnable augmentation (learns optimal aug params)")
    parser.add_argument("--learnable-aug-color", action="store_true", default=True,
                        help="Learnable color augmentation (brightness, contrast, saturation)")
    parser.add_argument("--no-learnable-aug-color", action="store_false", dest="learnable_aug_color")
    parser.add_argument("--learnable-aug-spatial", action="store_true", default=False,
                        help="Learnable spatial augmentation (scale, rotation, translation)")
    
    # Auxiliary heads for multi-task learning
    parser.add_argument("--use-aux-heads", action="store_true",
                        help="Add auxiliary heads for State/Month/Species classification (all three)")
    parser.add_argument("--use-species-head", action="store_true",
                        help="Add Species classification head only (without State/Month)")
    parser.add_argument("--aux-state-weight", type=float, default=1.0,
                        help="Weight for State classification loss")
    parser.add_argument("--aux-month-weight", type=float, default=1.0,
                        help="Weight for Month classification loss")
    parser.add_argument("--aux-species-weight", type=float, default=1.0,
                        help="Weight for Species classification loss")
    
    # Presence heads and NDVI
    parser.add_argument("--use-presence-heads", action="store_true",
                        help="Add binary presence heads for Dead/Clover (predict IF present, then HOW MUCH)")
    parser.add_argument("--use-ndvi-head", action="store_true",
                        help="Add NDVI auxiliary head (predict pseudo-NDVI from features)")
    parser.add_argument("--use-tweedie", action="store_true",
                        help="Use Tweedie loss (p=1.5) for Dead/Clover (good for zero-inflated targets)")
    parser.add_argument("--tweedie-p", type=float, default=1.5,
                        help="Tweedie power parameter (1 < p < 2)")
    parser.add_argument("--presence-weight", type=float, default=0.5,
                        help="Weight for presence classification loss")
    parser.add_argument("--ndvi-weight", type=float, default=0.3,
                        help="Weight for NDVI regression loss (uses ground-truth Pre_GSHH_NDVI)")
    parser.add_argument("--use-height-head", action="store_true",
                        help="Add Height auxiliary head (predict Height_Ave_cm from features)")
    parser.add_argument("--height-weight", type=float, default=0.3,
                        help="Weight for Height regression loss")
    
    # Advanced features (from dataset paper insights)
    parser.add_argument("--use-cross-view-consistency", action="store_true",
                        help="Add cross-view consistency loss (enforces L/R agreement)")
    parser.add_argument("--cross-view-weight", type=float, default=0.1,
                        help="Weight for cross-view consistency loss")
    parser.add_argument("--use-uncertainty", action="store_true",
                        help="Add uncertainty-aware regression (predict log variance, use Gaussian NLL)")
    parser.add_argument("--use-qc-plausibility", action="store_true",
                        help="Add QC-inspired plausibility penalties (domain knowledge constraints)")
    parser.add_argument("--qc-weight", type=float, default=0.1,
                        help="Weight for QC plausibility penalties")
    parser.add_argument("--use-species-prior", action="store_true",
                        help="Add species prior blending (PlantHydra-style for biomass)")
    
    # Augmentation options
    parser.add_argument("--use-perspective-jitter", action="store_true",
                        help="Add perspective jitter to simulate annotation/warp noise")
    parser.add_argument("--use-border-mask", action="store_true",
                        help="Add border masking to prevent frame-based shortcuts")
    parser.add_argument("--use-pad-to-square", action="store_true",
                        help="Use padding instead of resize (preserves aspect ratio & density patterns)")
    
    # Innovative Loss Features (from dataset paper)
    parser.add_argument("--use-innovative-loss", action="store_true",
                        help="Use InnovativeBiomassLoss with all 6 paper-inspired ideas")
    parser.add_argument("--use-mass-balance", action="store_true",
                        help="Add physical mass balance loss (Green+Dead≈Total)")
    parser.add_argument("--mass-balance-weight", type=float, default=0.2,
                        help="Weight for mass balance loss")
    parser.add_argument("--use-aos-hallucination", action="store_true",
                        help="Train NDVI head to hallucinate AOS sensor (lighting-invariant greenness)")
    parser.add_argument("--aos-weight", type=float, default=0.3,
                        help="Weight for AOS hallucination loss")
    parser.add_argument("--use-state-scaling", action="store_true",
                        help="Add state-based density scaling (climate/location bias correction)")
    parser.add_argument("--use-calibrated-depth", action="store_true",
                        help="Use height-calibrated depth volume features")
    parser.add_argument("--calibrated-depth-weight", type=float, default=0.2,
                        help="Weight for depth calibration loss")
    
    # Training mode
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze backbone (head-only training, default)")
    parser.add_argument("--train-backbone", action="store_true",
                        help="Train full model including backbone")
    parser.add_argument("--two-stage", action="store_true", default=True,
                        help="Two-stage: freeze first, then finetune (default: ON)")
    parser.add_argument("--no-two-stage", action="store_false", dest="two_stage",
                        help="Disable two-stage, train full model from start")
    parser.add_argument("--freeze-epochs", type=int, default=30,
                        help="Max epochs for Stage 1 (safety limit when using --stage1-patience)")
    parser.add_argument("--stage1-patience", type=int, default=10,
                        help="Early stopping patience for Stage 1 (default: 10, triggers Stage 2 transition)")
    parser.add_argument("--stage2-epochs", type=int, default=None,
                        help="Max epochs for Stage 2 (default: remaining from --epochs)")
    parser.add_argument("--stage2-patience", type=int, default=None,
                        help="Early stopping patience for Stage 2 (default: --patience)")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps. Use with DDP to restore single-GPU update frequency")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: 0 for MPS, 8 for CUDA)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for heads (Stage 1)")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="Learning rate for backbone (Stage 2)")
    parser.add_argument("--lr-mult", type=float, default=1.0,
                        help="Layer-wise LR decay multiplier (0.8 = earlier layers get 0.8x LR). "
                             "1.0 = no decay (default)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    
    # Augmentation
    parser.add_argument("--aug-prob", type=float, default=0.5)
    parser.add_argument("--strong-aug", action="store_true", default=True,
                        help="Use strong augmentations (RandomRotate90, ColorJitter, CLAHE, MotionBlur)")
    parser.add_argument("--no-strong-aug", action="store_false", dest="strong_aug",
                        help="Disable strong augmentations, use basic transforms only")
    parser.add_argument("--no-lighting", action="store_true", default=False,
                        help="Remove lighting augmentations (ColorJitter, HSV, Brightness, CLAHE)")
    parser.add_argument("--photometric", type=str, default="same",
                        choices=["same", "independent", "left", "right", "none"],
                        help="Photometric transform mode: "
                             "'same' = identical to both L/R (default), "
                             "'independent' = different to L/R, "
                             "'left' = only to left, "
                             "'right' = only to right, "
                             "'none' = no photometric transforms")
    parser.add_argument("--stereo-swap-prob", type=float, default=0.0,
                        help="Probability of swapping L/R views")
    parser.add_argument("--mixup-prob", type=float, default=0.0,
                        help="MixUp probability (0 to disable)")
    parser.add_argument("--mixup-alpha", type=float, default=0.4,
                        help="MixUp beta distribution alpha")
    parser.add_argument("--cutmix-prob", type=float, default=0.0,
                        help="CutMix probability (0 to disable)")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                        help="CutMix beta distribution alpha")
    parser.add_argument("--no-mix-same-context", action="store_true",
                        help="Allow MixUp/CutMix with ANY sample (not just same species/month/state)")
    
    # Loss
    parser.add_argument("--use-huber", action="store_true", default=False,
                        help="Use Huber loss for Dead (reduces outlier impact)")
    parser.add_argument("--no-huber", action="store_false", dest="use_huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument("--smoothl1", action="store_true", default=True,
                        help="Use SmoothL1 loss on 3 targets (Total, GDM, Green) - default")
    parser.add_argument("--no-smoothl1", action="store_false", dest="smoothl1",
                        help="Use MSE loss on all 5 targets instead of SmoothL1")
    parser.add_argument("--smoothl1-beta", type=float, default=1.0,
                        help="Beta for SmoothL1 loss (transition point between L1 and L2)")
    
    # PlantHydra-style loss (from 1st place solution)
    parser.add_argument("--use-planthydra-loss", action="store_true",
                        help="Use PlantHydra-style loss: log-transform + cosine similarity + compositional consistency")
    parser.add_argument("--use-log-transform", action="store_true",
                        help="Apply log(1+y) transform (official competition metric)")
    parser.add_argument("--no-log-transform", action="store_true", default=False,
                        help="Disable log transform for PlantHydra loss (train on raw values)")
    parser.add_argument("--use-cosine-sim", action="store_true",
                        help="Add cosine similarity loss to maintain correlation structure")
    parser.add_argument("--cosine-weight", type=float, default=0.4,
                        help="Weight for cosine similarity loss")
    parser.add_argument("--use-compositional", action="store_true",
                        help="Add compositional consistency loss (GDM=G+C, Total=G+D+C)")
    parser.add_argument("--compositional-weight", type=float, default=0.1,
                        help="Weight for compositional consistency loss")
    parser.add_argument("--use-state-head", action="store_true",
                        help="Add State classification head (NSW, VIC, TAS, WA)")
    parser.add_argument("--state-weight", type=float, default=0.1,
                        help="Weight for state classification loss")
    
    # System
    parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup distributed training (if launched with torchrun)
    rank, world_size, dist_device = setup_distributed()
    args.rank = rank
    args.world_size = world_size
    args.distributed = world_size > 1
    
    # Setup device
    if dist_device is not None:
        # Distributed training - use assigned GPU
        device = dist_device
        device_type = DeviceType.CUDA
    else:
        device_type = DeviceType(args.device_type) if args.device_type else get_device_type()
        device = get_device(device_type)
    
    # Seed with rank offset for distributed
    set_seed(args.seed + rank, device_type)
    
    # Auto-adjust for DDP to match single-GPU training dynamics
    # Instead of scaling LR (which changes optimization), we halve batch size per GPU
    # This keeps: effective_batch = batch_size, updates/epoch = same as single GPU
    if args.distributed and world_size > 1:
        original_batch = args.batch_size
        # Halve batch size per GPU so effective batch = original
        args.batch_size = max(1, args.batch_size // world_size)
        if is_main_process():
            effective_batch = args.batch_size * world_size
            print(f"DDP batch adjustment: {original_batch} → {args.batch_size}/GPU (effective: {effective_batch})")
            if effective_batch != original_batch:
                print(f"  Note: effective batch {effective_batch} != original {original_batch} due to rounding")
    
    # Auto-set num_workers based on device
    if args.num_workers is None:
        args.num_workers = 0 if device_type == DeviceType.MPS else 8
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"./outputs/dinov3_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command for reproducibility (only log relevant args)
    command_path = os.path.join(args.output_dir, "command.txt")
    
    # Map weight args to their enabling flags (skip weight if feature disabled)
    weight_to_feature = {
        "ndvi_weight": "use_ndvi_head",
        "height_weight": "use_height_head",
        "cross_view_weight": "use_cross_view_consistency",
        "qc_weight": "use_qc_plausibility",
        "mass_balance_weight": "use_mass_balance",
        "aos_weight": "use_aos_hallucination",
        "calibrated_depth_weight": "use_calibrated_depth",
        "presence_weight": "use_presence_heads",
        "cosine_weight": "use_cosine_sim",
        "compositional_weight": "use_compositional",
        "state_weight": "use_state_head",
        "aux_state_weight": "use_aux_heads",
        "aux_month_weight": "use_aux_heads",
        "aux_species_weight": "use_aux_heads",
        "tweedie_p": "use_tweedie",
        "huber_delta": "use_huber",
        "depth_model_size": "use_depth",
        "depth_attention": "use_depth",
    }
    # Skip these if they have default values
    skip_defaults = {"rank", "world_size", "device_type"}
    
    with open(command_path, "w") as f:
        f.write("# Command to reproduce this training run\n")
        f.write("python -m src.dinov3_train \\\n")
        for key, value in vars(args).items():
            if key in ("output_dir", "image_dir"):
                continue
            if key in skip_defaults:
                continue
            if value is None or value is False:
                continue
            # Skip weight args if their feature is disabled
            if key in weight_to_feature:
                feature_flag = weight_to_feature[key]
                if not getattr(args, feature_flag, False):
                    continue
            if value is True:
                f.write(f"    --{key.replace('_', '-')} \\\n")
            elif isinstance(value, list):
                f.write(f"    --{key.replace('_', '-')} {' '.join(map(str, value))} \\\n")
            else:
                f.write(f"    --{key.replace('_', '-')} {value} \\\n")
        f.write("\n")
    
    # Print config (only on main process)
    if is_main_process():
        print("=" * 60)
        print("DINOv3 Direct Model Training")
        if args.distributed:
            print(f"Distributed: {world_size} GPUs")
        print("=" * 60)
        print(f"Device: {device}")
    
    # Backbone info and image size
    backbone_type = getattr(args, 'backbone_type', 'dinov3')
    if args.img_size is not None:
        img_size = args.img_size  # User-specified
    elif backbone_type == "dinov2":
        img_size = IMG_SIZE_DINOV2  # 518px native for DINOv2
    else:
        img_size = IMG_SIZE_DINOV3  # 576px for DINOv3
    
    # Store computed img_size back to args for config saving
    args.img_size = img_size
    
    if backbone_type == "dinov2":
        print(f"Backbone: DINOv2-{args.backbone_size.upper()} (PlantCLEF pretrained)")
        if args.ckpt_path:
            print(f"  Weights: {args.ckpt_path}")
    else:
        print(f"Backbone: DINOv3-{args.backbone_size.upper()}")
    
    print(f"Image size: {img_size}")
    print(f"Grid: {args.grid}×{args.grid}")
    print(f"Dropout: {args.dropout} | Hidden ratio: {args.hidden_ratio}")
    print(f"FiLM: {args.use_film} | Attention pool: {args.use_attention_pool}")
    if args.use_vegetation_indices or args.use_disparity or args.use_depth or args.depth_attention or args.use_learnable_aug:
        extras = []
        if args.backbone_size != "base":
            extras.append(f"DINOv3-{args.backbone_size.upper()}")
        if args.use_whole_image:
            extras.append("3-View (L+R+Whole)")
        if args.use_vegetation_indices:
            extras.append("Vegetation Indices")
        if args.use_disparity:
            extras.append("Stereo Disparity")
        if args.use_depth:
            extras.append(f"Depth Stats (DA2-{args.depth_model_size})")
        if args.depth_attention:
            extras.append(f"Depth Attention (DA2-{args.depth_model_size})")
        if args.use_presence_heads:
            extras.append("Presence Heads (Dead/Clover)")
        if args.use_ndvi_head:
            extras.append("NDVI Head (GT)")
        if args.use_height_head:
            extras.append("Height Head")
        if args.use_species_head:
            extras.append("Species Head")
        if args.use_tweedie:
            extras.append(f"Tweedie Loss (p={args.tweedie_p})")
        if args.use_learnable_aug:
            aug_types = []
            if args.learnable_aug_color:
                aug_types.append("color")
            if args.learnable_aug_spatial:
                aug_types.append("spatial")
            extras.append(f"Learnable Aug ({'+'.join(aug_types)})")
        print(f"Innovative features: {', '.join(extras)}")
    
    # Heads info
    heads = ["Total", "Green", "GDM"]
    derived = []
    if args.train_dead:
        heads.append("Dead")
    else:
        derived.append("Dead")
    if args.train_clover:
        heads.append("Clover")
    else:
        derived.append("Clover")
    print(f"Predict: {', '.join(heads)}")
    if derived:
        print(f"Derive:  {', '.join(derived)}")
    
    # --train-backbone overrides --freeze-backbone
    if args.train_backbone:
        args.freeze_backbone = False
    
    if args.freeze_backbone:
        print(f"Mode: HEAD-ONLY (frozen backbone, default)")
    elif args.two_stage:
        if args.stage1_patience:
            stage2_eps = args.stage2_epochs if args.stage2_epochs else (args.epochs - args.freeze_epochs)
            stage2_pat = args.stage2_patience if args.stage2_patience else args.patience
            print(f"Mode: TWO-STAGE (Stage 1: patience={args.stage1_patience}, Stage 2: {stage2_eps} ep, patience={stage2_pat})")
        else:
            print(f"Mode: TWO-STAGE (freeze {args.freeze_epochs} ep, then finetune)")
    else:
        print(f"Mode: FULL TRAINING")
    print(f"LR: {args.lr:.2e} | Backbone LR: {args.backbone_lr:.2e}")
    if args.smoothl1:
        print(f"Loss: SmoothL1 on 3 targets (G=0.125, GDM=0.25, T=0.625)")
    else:
        print(f"Loss: MSE on 5 targets (G=0.1, D=0.1, C=0.1, GDM=0.2, T=0.5)")
    print(f"Epochs: {args.epochs} | Patience: {args.patience}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Load data
    train_csv = os.path.join(args.base_path, "train.csv")
    args.image_dir = os.path.join(args.base_path, "train")
    
    df = prepare_dataframe(train_csv)
    
    # Load or create folds
    if args.use_predefined_folds:
        if not os.path.exists(args.fold_csv):
            raise FileNotFoundError(
                f"Predefined folds file not found: {args.fold_csv}\n"
                f"Use --no-predefined-folds to create new folds, or provide --fold-csv path"
            )
        print(f"Loading predefined folds from: {args.fold_csv}")
        fold_df = pd.read_csv(args.fold_csv)
        fold_mapping = fold_df.set_index("sample_id_prefix")["fold"].to_dict()
        df["fold"] = df["sample_id_prefix"].map(fold_mapping).fillna(0).astype(int)
    else:
        print(f"Creating folds with {args.cv_strategy} strategy")
        df = create_folds(df, n_folds=args.n_folds, seed=args.seed, cv_strategy=args.cv_strategy)
    
    # Save fold info
    fold_df = df[["sample_id_prefix", "fold"]].drop_duplicates()
    fold_df.to_csv(os.path.join(args.output_dir, "folds.csv"), index=False)
    
    print(f"Total samples: {len(df)}")
    print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")
    
    # Train folds
    folds_to_train = args.folds if args.folds else list(range(args.n_folds))
    results = []
    results_path = os.path.join(args.output_dir, "results.json")
    
    # Save initial config before training starts
    with open(results_path, "w") as f:
        json.dump({
            "status": "training",
            "cv_aR2_mean": None,
            "cv_aR2_std": None,
            "cv_gR2_mean": None,
            "cv_gR2_std": None,
            "folds": [],
            "config": vars(args),
        }, f, indent=2, default=str)
    print(f"Config saved to: {results_path}")
    
    for fold in folds_to_train:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        
        fold_result = train_fold(fold, train_df, valid_df, args, device, device_type, img_size)
        results.append(fold_result)
        
        # Save results after each fold
        avg_r2_scores = [r["best_r2"] for r in results]
        
        with open(results_path, "w") as f:
            json.dump({
                "status": "training" if len(results) < len(folds_to_train) else "complete",
                "cv_R2_mean": float(np.mean(avg_r2_scores)),
                "cv_R2_std": float(np.std(avg_r2_scores)) if len(results) > 1 else 0.0,
                "folds": results,
                "config": vars(args),
            }, f, indent=2, default=str)
        print(f"  Results updated: {results_path} ({len(results)}/{len(folds_to_train)} folds)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    avg_r2_scores = [r["best_r2"] for r in results]
    print(f"Mean CV R²: {np.mean(avg_r2_scores):.4f} ± {np.std(avg_r2_scores):.4f}")
    
    for r in results:
        m = r["metrics"]
        print(f"  Fold {r['fold']}: R²={r['best_r2']:.4f} "
              f"[G={m['r2_green']:.3f} D={m['r2_dead']:.3f} C={m['r2_clover']:.3f} "
              f"GDM={m['r2_gdm']:.3f} T={m['r2_total']:.3f}]")
    
    # Save Depth Anything model separately for Kaggle (if depth features used)
    if is_main_process() and (args.use_depth or args.depth_attention):
        depth_model_dir = os.path.join(args.output_dir, "depth_model")
        try:
            from transformers import AutoModelForDepthEstimation, AutoImageProcessor
            model_name = f"depth-anything/Depth-Anything-V2-{args.depth_model_size.capitalize()}-hf"
            print(f"\nSaving Depth Anything model for Kaggle...")
            
            # Load and save model
            depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
            depth_model.save_pretrained(depth_model_dir)
            
            # Save processor too
            processor = AutoImageProcessor.from_pretrained(model_name)
            processor.save_pretrained(depth_model_dir)
            
            print(f"  Saved to: {depth_model_dir}")
            print(f"  Upload this folder to Kaggle as a dataset for inference")
        except Exception as e:
            print(f"  Warning: Could not save depth model: {e}")
    
    if is_main_process():
        print(f"\nFinal results saved to: {results_path}")
    
    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()

