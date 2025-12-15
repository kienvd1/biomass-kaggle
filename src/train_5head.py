#!/usr/bin/env python3
"""
Optimized training script for 5-Head DINOv2 model.

Key improvements:
- 2-stage training (freeze backbone first, then finetune)
- Differential learning rates for backbone vs heads
- GradScaler for CUDA float16 stability
- Warmup epochs
- Gradient clipping
- Better loss function with Huber for dead target

Usage:
    python -m src.train_5head --device-type mps
    python -m src.train_5head --device-type cuda --batch-size 16
    
    # With 2-stage training (recommended):
    python -m src.train_5head --two-stage --freeze-epochs 5
"""
import argparse
import gc
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
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
    get_amp_settings,
    get_device,
    get_device_type,
    set_device_seed,
    supports_fused_optimizer,
)
from .models_5head import (
    AuxiliaryLoss,
    BalancedMSELoss,
    ConstrainedMSELoss,
    DeadAwareLoss,
    DeadPostProcessor,
    FocalMSELoss,
    UncertaintyWeightedLoss,
    build_5head_model,
)
from .trainer import compute_weighted_r2, compute_per_target_metrics_np


class AverageMeter:
    """Computes and stores average and current value."""
    
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
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_device_seed(seed, device_type)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    use_aux: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch with gradient scaling and clipping.
    
    Returns:
        avg_loss: Average total loss
        loss_components: Dict with individual loss components (for auxiliary losses)
    """
    model.train()
    losses = AverageMeter()
    loss_components: Dict[str, AverageMeter] = {
        "biomass": AverageMeter(),
        "state": AverageMeter(),
        "month": AverageMeter(),
        "species": AverageMeter(),
    }

    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)

    pbar = tqdm(loader, desc="Training", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        # Unpack batch - may or may not have aux labels (now includes species)
        if use_aux and len(batch) == 6:
            x_left, x_right, targets, state_labels, month_labels, species_labels = batch
            state_labels = state_labels.to(device, non_blocking=True)
            month_labels = month_labels.to(device, non_blocking=True)
            species_labels = species_labels.to(device, non_blocking=True)
        else:
            x_left, x_right, targets = batch[:3]
            state_labels = None
            month_labels = None
            species_labels = None

        # Use channels-last for MPS performance
        if device_type == DeviceType.MPS:
            x_left = x_left.to(device, non_blocking=True, memory_format=torch.channels_last)
            x_right = x_right.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_left = x_left.to(device, non_blocking=True)
            x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            if use_aux:
                green, dead, clover, gdm, total, state_logits, month_logits, species_logits = model(
                    x_left, x_right, return_aux=True
                )
            else:
                green, dead, clover, gdm, total = model(x_left, x_right)
                state_logits = None
                month_logits = None
                species_logits = None

            preds = torch.cat([green, dead, clover, gdm, total], dim=1)

            # Compute loss
            if use_aux and isinstance(loss_fn, AuxiliaryLoss):
                loss, loss_dict = loss_fn(
                    preds, targets,
                    state_logits=state_logits, state_labels=state_labels,
                    month_logits=month_logits, month_labels=month_labels,
                    species_logits=species_logits, species_labels=species_labels,
                )
                # Track individual components
                loss_components["biomass"].update(loss_dict.get("loss_biomass", 0), x_left.size(0))
                loss_components["state"].update(loss_dict.get("loss_state", 0), x_left.size(0))
                loss_components["month"].update(loss_dict.get("loss_month", 0), x_left.size(0))
                loss_components["species"].update(loss_dict.get("loss_species", 0), x_left.size(0))
            else:
                loss = loss_fn(preds, targets)
            
            loss = loss / grad_accum_steps
        
        # Backward with optional gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step with gradient accumulation
        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        losses.update(loss.item() * grad_accum_steps, x_left.size(0))
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})
    
    return losses.avg, {k: v.avg for k, v in loss_components.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    postprocessor: Optional[DeadPostProcessor] = None,
    use_aux: bool = False,
    apply_context_adjustment: bool = False,
    use_log_target: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """Validate with optional post-processing and auxiliary metrics.
    
    Args:
        use_log_target: If True, apply expm1 to predictions before computing metrics
                       (inverse of log1p applied during training)
    """
    model.eval()
    losses = AverageMeter()
    
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    
    all_preds = []
    all_preds_post = []
    all_preds_ctx = []  # Predictions with context adjustment
    all_targets = []
    
    # For auxiliary accuracy
    state_correct = 0
    state_total = 0
    month_correct = 0
    month_total = 0
    species_correct = 0
    species_total = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        # Unpack batch (now includes species)
        if use_aux and len(batch) == 6:
            x_left, x_right, targets, state_labels, month_labels, species_labels = batch
            state_labels = state_labels.to(device, non_blocking=True)
            month_labels = month_labels.to(device, non_blocking=True)
            species_labels = species_labels.to(device, non_blocking=True)
        else:
            x_left, x_right, targets = batch[:3]
            state_labels = None
            month_labels = None
            species_labels = None

        # Use channels-last for MPS performance
        if device_type == DeviceType.MPS:
            x_left = x_left.to(device, non_blocking=True, memory_format=torch.channels_last)
            x_right = x_right.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_left = x_left.to(device, non_blocking=True)
            x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            if use_aux:
                # Get predictions with aux logits
                green, dead, clover, gdm, total, state_logits, month_logits, species_logits = model(
                    x_left, x_right, return_aux=True
                )

                # Compute auxiliary accuracy
                if state_labels is not None:
                    state_pred = state_logits.argmax(dim=1)
                    state_correct += (state_pred == state_labels).sum().item()
                    state_total += state_labels.size(0)

                if month_labels is not None:
                    month_pred = month_logits.argmax(dim=1)
                    month_correct += (month_pred == month_labels).sum().item()
                    month_total += month_labels.size(0)

                if species_labels is not None:
                    species_pred = species_logits.argmax(dim=1)
                    species_correct += (species_pred == species_labels).sum().item()
                    species_total += species_labels.size(0)

                # Also get context-adjusted predictions for comparison
                if apply_context_adjustment:
                    green_ctx, dead_ctx, clover_ctx, gdm_ctx, total_ctx = model(
                        x_left, x_right, return_aux=False, apply_context_adjustment=True
                    )
                    preds_ctx = torch.cat([green_ctx, dead_ctx, clover_ctx, gdm_ctx, total_ctx], dim=1)
                    all_preds_ctx.append(preds_ctx.float().cpu())
            else:
                green, dead, clover, gdm, total = model(x_left, x_right)
            
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            
            # Compute loss (base loss only for validation metric)
            if isinstance(loss_fn, AuxiliaryLoss):
                loss = loss_fn.base_loss(preds, targets)
            else:
                loss = loss_fn(preds, targets)
        
        losses.update(loss.item(), x_left.size(0))
        
        # Store raw predictions
        all_preds.append(preds.float().cpu())
        all_targets.append(targets.cpu())
        
        # Apply post-processing
        if postprocessor is not None:
            green_p, dead_p, clover_p, gdm_p, total_p = postprocessor(
                green.squeeze(-1), dead.squeeze(-1), clover.squeeze(-1),
                gdm.squeeze(-1), total.squeeze(-1)
            )
            preds_post = torch.stack([green_p, dead_p, clover_p, gdm_p, total_p], dim=1)
            all_preds_post.append(preds_post.float().cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Apply inverse transform if using log targets
    if use_log_target:
        all_preds = np.expm1(all_preds)
        all_targets = np.expm1(all_targets)
    
    # Compute metrics for raw predictions
    r2_raw = compute_weighted_r2(all_preds, all_targets, target_weights)
    
    # Compute metrics for post-processed predictions
    if postprocessor is not None and all_preds_post:
        all_preds_post = torch.cat(all_preds_post, dim=0).numpy()
        if use_log_target:
            all_preds_post = np.expm1(all_preds_post)
        r2_post = compute_weighted_r2(all_preds_post, all_targets, target_weights)
    else:
        r2_post = r2_raw
        all_preds_post = all_preds
    
    # Compute metrics for context-adjusted predictions
    if all_preds_ctx:
        all_preds_ctx = torch.cat(all_preds_ctx, dim=0).numpy()
        if use_log_target:
            all_preds_ctx = np.expm1(all_preds_ctx)
        r2_ctx = compute_weighted_r2(all_preds_ctx, all_targets, target_weights)
    else:
        r2_ctx = r2_raw
    
    # Per-target metrics (for post-processed)
    target_names = ["green", "dead", "clover", "gdm", "total"]
    per_target = compute_per_target_metrics_np(all_preds_post, all_targets, target_names, target_weights)
    
    metrics = {}
    for row in per_target:
        t = row["target"]
        metrics[f"r2_{t}"] = float(row["r2"])
        metrics[f"rmse_{t}"] = float(row["rmse"])
    
    metrics["r2_raw"] = r2_raw
    metrics["r2_post"] = r2_post
    metrics["r2_ctx"] = r2_ctx  # Context-adjusted R²
    metrics["weighted_r2"] = r2_post  # Use post-processed for best metric
    
    # Add auxiliary accuracy metrics
    if state_total > 0:
        metrics["state_acc"] = state_correct / state_total
    if month_total > 0:
        metrics["month_acc"] = month_correct / month_total
    if species_total > 0:
        metrics["species_acc"] = species_correct / species_total

    return losses.avg, r2_post, metrics


def freeze_backbone_fn(model: nn.Module) -> None:
    """Freeze backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone_fn(model: nn.Module) -> None:
    """Unfreeze backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = True


def get_optimizer(
    model: nn.Module,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    device_type: DeviceType,
    stage: int = 2,
    loss_fn: Optional[nn.Module] = None,
) -> AdamW:
    """Create optimizer with differential LR for backbone vs heads.
    
    Also includes learnable loss parameters (e.g., UncertaintyWeightedLoss.log_vars).
    """
    use_fused = supports_fused_optimizer(device_type)
    
    # Collect loss function parameters if it has any
    loss_params = []
    if loss_fn is not None:
        loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    
    if stage == 1:
        # Stage 1: Only head parameters (backbone frozen) + loss params
        params = [p for p in model.parameters() if p.requires_grad]
        params.extend(loss_params)
        return AdamW(params, lr=lr, weight_decay=weight_decay, fused=use_fused)
    else:
        # Stage 2: Differential LR
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": lr},
        ]
        
        # Add loss params to head group (same LR as heads)
        if loss_params:
            param_groups.append({"params": loss_params, "lr": lr})
        
        return AdamW(param_groups, weight_decay=weight_decay, fused=use_fused)


def get_scheduler_with_warmup(
    optimizer: AdamW,
    num_epochs: int,
    warmup_epochs: int = 2,
    min_lr: float = 1e-7,
) -> Any:
    """Create scheduler with linear warmup."""
    if warmup_epochs > 0 and num_epochs > warmup_epochs:
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    cfg: argparse.Namespace,
    device: torch.device,
    device_type: DeviceType,
) -> Dict:
    """Train a single fold with optional 2-stage training and auxiliary heads."""
    two_stage = getattr(cfg, "two_stage", False)
    freeze_epochs = getattr(cfg, "freeze_epochs", 5)
    use_aux_heads = getattr(cfg, "use_aux_heads", False)
    freeze_backbone = getattr(cfg, "freeze_backbone", False)
    
    # If freeze_backbone is set, it overrides two_stage
    if freeze_backbone:
        two_stage = False  # Disable 2-stage, just freeze throughout
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}")
    if freeze_backbone:
        print(f"HEAD-ONLY MODE: Backbone frozen, training heads only")
    elif two_stage:
        print(f"2-Stage: Stage 1 ({freeze_epochs} epochs frozen), Stage 2 (finetune)")
    if use_aux_heads:
        print(f"Auxiliary Heads: State + Month classification enabled")
    print(f"{'='*60}")
    
    # Datasets
    use_log_target = getattr(cfg, "use_log_target", False)
    stereo_correct_aug = getattr(cfg, "stereo_correct_aug", False)
    stereo_swap_prob = getattr(cfg, "stereo_swap_prob", 0.0)
    mixup_prob = getattr(cfg, "mixup_prob", 0.0)
    mixup_alpha = getattr(cfg, "mixup_alpha", 0.4)
    cutmix_prob = getattr(cfg, "cutmix_prob", 0.0)
    cutmix_alpha = getattr(cfg, "cutmix_alpha", 1.0)
    
    if stereo_correct_aug:
        # Stereo-correct: geometric transforms via replay, photometric independently
        train_transform = get_stereo_geometric_transforms(cfg.img_size, cfg.aug_prob)
        photometric_transform = get_stereo_photometric_transforms(cfg.aug_prob)
        print(f"STEREO-CORRECT AUG: photometric transforms applied independently per view")
    else:
        train_transform = get_train_transforms(cfg.img_size, cfg.aug_prob)
        photometric_transform = None
    
    if stereo_swap_prob > 0:
        print(f"STEREO SWAP: L/R swap probability = {stereo_swap_prob:.1%}")
    
    if use_log_target:
        print(f"LOG TARGET: Using log1p transformation on targets")
    
    if mixup_prob > 0:
        print(f"MIXUP: prob={mixup_prob:.1%}, alpha={mixup_alpha} (same species/month/state only)")
    
    if cutmix_prob > 0:
        print(f"CUTMIX: prob={cutmix_prob:.1%}, alpha={cutmix_alpha} (same species/month/state only)")
    
    valid_transform = get_valid_transforms(cfg.img_size)
    
    # When caching images, must use num_workers=0 to avoid memory explosion
    # (each worker would copy the entire cache)
    num_workers = 0 if cfg.cache_images else cfg.num_workers
    if cfg.cache_images and cfg.num_workers > 0:
        print(f"WARNING: cache_images=True, forcing num_workers=0 (was {cfg.num_workers})")
    
    train_ds = BiomassDataset(
        train_df, cfg.train_image_dir, train_transform, 
        is_train=True, cache_images=cfg.cache_images, return_aux_labels=use_aux_heads,
        use_log_target=use_log_target, stereo_swap_prob=stereo_swap_prob,
        photometric_transform=photometric_transform,
        mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
        cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha,
    )
    valid_ds = BiomassDataset(
        valid_df, cfg.train_image_dir, valid_transform, 
        is_train=False, cache_images=cfg.cache_images, return_aux_labels=use_aux_heads,
        use_log_target=use_log_target,  # Also apply log for validation targets
    )
    
    pin_memory = device_type == DeviceType.CUDA
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=num_workers > 0, prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
    )
    # Validation can use larger batch size (no gradients needed)
    valid_batch_multiplier = 4 if device_type == DeviceType.MPS else 2
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size * valid_batch_multiplier, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
    )

    # Model
    model = build_5head_model(
        backbone_name=cfg.backbone,
        grid=tuple(cfg.grid),
        pretrained=True,
        dropout=cfg.dropout,
        hidden_ratio=cfg.hidden_ratio,
        use_film=getattr(cfg, "use_film", True),
        use_attention_pool=getattr(cfg, "use_attention_pool", True),
        gradient_checkpointing=getattr(cfg, "grad_ckpt", False),
        use_aux_heads=use_aux_heads,
    ).to(device)

    # Use channels-last memory format for better performance on MPS
    if device_type == DeviceType.MPS:
        model = model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: 5-Head DINOv2 ({cfg.backbone})")
    print(f"Grid: {cfg.grid}, FiLM: {getattr(cfg, 'use_film', True)}, AttnPool: {getattr(cfg, 'use_attention_pool', True)}")
    if use_aux_heads:
        print(f"Auxiliary heads: State (4 classes), Month (10 classes)")
    print(f"Total params: {total_params:,}")
    
    # Setup for 2-stage, freeze-backbone, or single-stage training
    current_stage = 1 if two_stage else 2
    
    if freeze_backbone:
        # HEAD-ONLY MODE: Freeze backbone for entire training
        freeze_backbone_fn(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"HEAD-ONLY: Backbone FROZEN ({trainable_params:,} / {total_params:,} params trainable)")
        # Use higher LR for heads since backbone is frozen
        head_lr = getattr(cfg, "head_lr_stage1", 1e-3)
        optimizer = get_optimizer(model, head_lr, cfg.backbone_lr, cfg.weight_decay, device_type, stage=1)
        scheduler = get_scheduler_with_warmup(optimizer, cfg.epochs, warmup_epochs=cfg.warmup_epochs)
        current_stage = 1  # Stay in stage 1 (frozen) forever
    elif two_stage:
        freeze_backbone_fn(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Stage 1: Backbone FROZEN ({trainable_params:,} / {total_params:,} params trainable)")
        optimizer = get_optimizer(model, cfg.head_lr_stage1, cfg.backbone_lr, cfg.weight_decay, device_type, stage=1)
        scheduler = get_scheduler_with_warmup(optimizer, freeze_epochs, warmup_epochs=1)
    else:
        optimizer = get_optimizer(model, cfg.lr, cfg.backbone_lr, cfg.weight_decay, device_type, stage=2)
        scheduler = get_scheduler_with_warmup(optimizer, cfg.epochs, warmup_epochs=cfg.warmup_epochs)
    
    # GradScaler for CUDA float16
    scaler = None
    if device_type == DeviceType.CUDA and cfg.amp_dtype == "float16":
        scaler = GradScaler()
    
    # Loss function
    target_weights = getattr(cfg, "target_weights", [0.2, 0.2, 0.2, 0.2, 0.2])
    
    if getattr(cfg, "use_dead_aware_loss", False):
        # Special loss for improving Dead prediction
        base_loss_fn = DeadAwareLoss(
            target_weights=target_weights,
            use_log_for_dead=True,
            aux_dead_weight=0.2,
        )
        print("Using DeadAwareLoss (log-space + auxiliary Dead loss)")
    elif getattr(cfg, "use_uncertainty_loss", False):
        # Learnable uncertainty weights (Kendall et al. 2018)
        base_loss_fn = UncertaintyWeightedLoss(
            n_targets=5,
            use_huber_for_dead=getattr(cfg, "use_huber_for_dead", True),
        )
        print("Using UncertaintyWeightedLoss (learnable per-target weights)")
    elif getattr(cfg, "use_balanced_loss", False):
        # Balance penalty for per-target losses
        base_loss_fn = BalancedMSELoss(
            target_weights=target_weights,
            balance_weight=getattr(cfg, "balance_weight", 0.1),
            use_huber_for_dead=getattr(cfg, "use_huber_for_dead", True),
        )
        print(f"Using BalancedMSELoss (balance_weight={cfg.balance_weight})")
    elif getattr(cfg, "use_focal_loss", False):
        base_loss_fn = FocalMSELoss(target_weights=target_weights, gamma=1.0)
    else:
        base_loss_fn = ConstrainedMSELoss(
            target_weights=target_weights,
            constraint_weight=cfg.constraint_weight,
            use_huber_for_dead=getattr(cfg, "use_huber_for_dead", True),
        )
    
    # Wrap with auxiliary loss if using aux heads
    if use_aux_heads:
        loss_fn = AuxiliaryLoss(
            base_loss=base_loss_fn,
            state_weight=getattr(cfg, "aux_state_weight", 5.0),
            month_weight=getattr(cfg, "aux_month_weight", 3.0),
            species_weight=getattr(cfg, "aux_species_weight", 2.0),
        )
        print(f"Using AuxiliaryLoss: state={cfg.aux_state_weight}, month={cfg.aux_month_weight}, species={getattr(cfg, 'aux_species_weight', 2.0)}")
    else:
        loss_fn = base_loss_fn
    
    # Add loss function's learnable parameters to optimizer (e.g., UncertaintyWeightedLoss)
    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    if loss_params:
        # Get current LR from optimizer
        current_lr = optimizer.param_groups[-1]["lr"]
        optimizer.add_param_group({"params": loss_params, "lr": current_lr})
        print(f"Added {len(loss_params)} learnable loss parameters to optimizer")
    
    # Post-processor for validation
    postprocessor = DeadPostProcessor(
        correction_threshold=cfg.correction_threshold,
        always_correct=cfg.always_correct_dead,
    )
    
    # Training loop
    best_r2 = 0.0
    best_loss = float("inf")
    patience_counter = 0
    best_metrics_snapshot: Optional[Dict[str, float]] = None
    history: Dict[str, List] = {
        "train_loss": [], "valid_loss": [], "r2_raw": [], "r2_post": [],
        "per_target_r2": [], "stage": [], "lr": [],
    }
    
    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()
        
        # Stage transition
        if two_stage and current_stage == 1 and epoch > freeze_epochs:
            current_stage = 2
            print(f"\n{'='*60}")
            print(f"Stage 2: Unfreezing backbone for full finetuning")
            print(f"  Head LR: {cfg.lr:.2e}, Backbone LR: {cfg.backbone_lr:.2e}")
            print(f"{'='*60}")
            
            unfreeze_backbone_fn(model)
            optimizer = get_optimizer(model, cfg.lr, cfg.backbone_lr, cfg.weight_decay, device_type, stage=2)
            remaining_epochs = cfg.epochs - freeze_epochs
            scheduler = get_scheduler_with_warmup(optimizer, remaining_epochs, warmup_epochs=cfg.warmup_epochs)
            patience_counter = 0  # Reset patience for stage 2
        
        # Train
        train_loss, train_loss_components = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, device_type,
            scaler=scaler,
            grad_clip=cfg.grad_clip,
            grad_accum_steps=getattr(cfg, "grad_accum", 1),
            use_aux=use_aux_heads,
        )
        
        # Validate
        apply_ctx = getattr(cfg, "apply_context_adjustment", False)
        valid_loss, r2, metrics = validate(
            model, valid_loader, loss_fn, device, device_type, postprocessor,
            use_aux=use_aux_heads,
            apply_context_adjustment=apply_ctx and use_aux_heads,
            use_log_target=use_log_target,
        )
        
        scheduler.step()
        
        # Logging
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[-1]["lr"]
        stage_str = f"S{current_stage}" if two_stage else ""
        
        r2_vec = [metrics.get(f"r2_{n}", 0.0) for n in ["green", "dead", "clover", "gdm", "total"]]
        
        r2_ctx = metrics.get('r2_ctx', metrics['r2_raw'])
        # Note: Loss is in log-space when use_log_target=True, R² is always on original scale (competition metric)
        loss_suffix = " (log)" if use_log_target else ""
        print(
            f"Epoch {epoch:02d} {stage_str:>3} | "
            f"Train{loss_suffix}: {train_loss:.4f} | "
            f"Valid{loss_suffix}: {valid_loss:.4f} | "
            f"Comp.R²: {metrics['r2_post']:.4f} | "
            f"R²_ctx: {r2_ctx:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )
        print(f"   per-target R²[g,d,c,gdm,t]=[{', '.join(f'{x:.3f}' for x in r2_vec)}]")
        
        # Log auxiliary metrics if available
        if use_aux_heads:
            state_acc = metrics.get("state_acc", 0)
            month_acc = metrics.get("month_acc", 0)
            species_acc = metrics.get("species_acc", 0)
            print(f"   aux acc: State={state_acc:.3f}, Month={month_acc:.3f}, Species={species_acc:.3f}")
            print(f"   aux loss: biomass={train_loss_components.get('biomass', 0):.3f}, "
                  f"state={train_loss_components.get('state', 0):.3f}, "
                  f"month={train_loss_components.get('month', 0):.3f}, "
                  f"species={train_loss_components.get('species', 0):.3f}")
        
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["r2_raw"].append(metrics["r2_raw"])
        history["r2_post"].append(metrics["r2_post"])
        history["per_target_r2"].append(r2_vec)
        history["stage"].append(current_stage)
        history["lr"].append(lr)
        
        # Save best model
        # - Always save if not using two-stage
        # - In two-stage: save in stage 2, OR save in stage 1 if freeze_epochs >= epochs (head-only mode)
        # - In freeze_backbone mode: always save (it's head-only training)
        head_only_mode = two_stage and freeze_epochs >= cfg.epochs
        save_ok = (current_stage == 2) or (not two_stage) or head_only_mode or freeze_backbone
        
        # Handle NaN - don't save if metrics are NaN
        if save_ok and not np.isnan(r2) and r2 > best_r2:
            best_r2 = r2
            best_loss = valid_loss
            best_metrics_snapshot = {k: float(v) for k, v in metrics.items()}
            patience_counter = 0
            
            save_path = os.path.join(cfg.output_dir, f"5head_best_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (Competition R²: {best_r2:.4f})")
        elif save_ok:
            patience_counter += 1
        
        # Early stopping on NaN - training has diverged
        if np.isnan(train_loss) or np.isnan(valid_loss):
            print(f"Training diverged (NaN detected) at epoch {epoch}")
            break
        
        # Early stopping (only in stage 2 or head-only mode)
        if save_ok and patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Cleanup
    del model, optimizer, scheduler
    if scaler is not None:
        del scaler
    empty_cache(device_type)
    gc.collect()
    
    return {
        "fold": fold,
        "best_r2": best_r2,
        "best_loss": best_loss,
        "best_metrics": best_metrics_snapshot or {},
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized 5-Head DINOv2 Training")
    
    # Paths
    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default=None)
    
    # Model architecture
    parser.add_argument(
        "--backbone", type=str, default="vit_base_patch14_reg4_dinov2.lvd142m",
        help="DINOv2 backbone (base recommended for accuracy)"
    )
    parser.add_argument("--grid", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-ratio", type=float, default=0.5)
    parser.add_argument("--no-film", action="store_true", help="Disable FiLM conditioning")
    parser.add_argument("--no-attention-pool", action="store_true", help="Disable attention pooling")
    parser.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-images", action="store_true", help="Cache images in RAM for faster training")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad-clip", type=float, default=0.5, help="Gradient clipping norm")
    parser.add_argument("--patience", type=int, default=10)
    
    # Learning rates
    parser.add_argument("--lr", type=float, default=2e-4, help="Head learning rate for stage 2")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Backbone LR for stage 2")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    
    # 2-Stage training
    parser.add_argument("--two-stage", action="store_true", help="Enable 2-stage training")
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Stage 1 epochs with frozen backbone")
    parser.add_argument("--head-lr-stage1", type=float, default=1e-3, help="Head LR for stage 1")
    parser.add_argument("--freeze-backbone", action="store_true", 
                        help="Keep backbone frozen throughout training (head-only mode)")
    
    # Auxiliary heads for multi-task learning
    parser.add_argument("--use-aux-heads", action="store_true",
                        help="Add auxiliary heads for State, Month, and Species classification")
    parser.add_argument("--aux-state-weight", type=float, default=0.5,
                        help="Weight for state classification loss")
    parser.add_argument("--aux-month-weight", type=float, default=0.5,
                        help="Weight for month classification loss")
    parser.add_argument("--aux-species-weight", type=float, default=0.25,
                        help="Weight for species classification loss")
    parser.add_argument("--apply-context-adjustment", action="store_true",
                        help="Use predicted state/month/species to adjust predictions at inference")
    
    # Loss function
    parser.add_argument("--constraint-weight", type=float, default=0.05,
                        help="Weight for consistency constraint loss")
    parser.add_argument("--use-focal-loss", action="store_true", help="Use focal MSE loss")
    parser.add_argument("--use-dead-aware-loss", action="store_true", 
                        help="Use DeadAwareLoss with log-space and auxiliary loss for Dead")
    parser.add_argument("--use-uncertainty-loss", action="store_true",
                        help="Use learnable uncertainty weights to balance per-target losses (Kendall et al.)")
    parser.add_argument("--use-balanced-loss", action="store_true",
                        help="Add variance penalty to balance per-target losses")
    parser.add_argument("--balance-weight", type=float, default=0.1,
                        help="Weight for balance penalty in BalancedMSELoss")
    parser.add_argument("--no-huber-for-dead", action="store_true", help="Disable Huber loss for dead target")
    parser.add_argument("--target-weights", type=float, nargs=5, 
                        default=[0.2, 0.2, 0.2, 0.2, 0.2],
                        help="Loss weights for [green, dead, clover, gdm, total]")
    
    # Post-processing
    parser.add_argument("--correction-threshold", type=float, default=0.15)
    parser.add_argument("--always-correct-dead", action="store_true")
    
    # Augmentation
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--aug-prob", type=float, default=0.5)
    parser.add_argument("--stereo-correct-aug", action="store_true",
                        help="Apply photometric transforms independently per stereo view")
    parser.add_argument("--stereo-swap-prob", type=float, default=0.0,
                        help="Probability of swapping L/R stereo images (0.0-1.0, recommended: 0.5)")
    
    # Target normalization
    parser.add_argument("--use-log-target", action="store_true",
                        help="Apply log1p to targets (for long-tail distributions)")
    
    # MixUp/CutMix (constrained to same species/month/state)
    parser.add_argument("--mixup-prob", type=float, default=0.0,
                        help="MixUp probability (0.0-1.0, only mixes same species/month/state)")
    parser.add_argument("--mixup-alpha", type=float, default=0.4,
                        help="MixUp beta distribution alpha parameter")
    parser.add_argument("--cutmix-prob", type=float, default=0.0,
                        help="CutMix probability (0.0-1.0, only mixes same species/month/state)")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                        help="CutMix beta distribution alpha parameter")
    
    # AMP
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    
    # CV
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--train-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cv-strategy", type=str, default="group_date_state",
                        help="CV strategy: group_date_state (default), group_month, group_date, stratified, random")
    
    # Device
    parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"])
    
    # Misc
    parser.add_argument("--seed", type=int, default=18)
    
    args = parser.parse_args()
    
    # Derived args
    args.use_film = not args.no_film
    args.use_attention_pool = not args.no_attention_pool
    args.use_huber_for_dead = not args.no_huber_for_dead
    
    # Setup paths
    args.train_csv = os.path.join(args.base_path, "train.csv")
    args.train_image_dir = os.path.join(args.base_path, "train")
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("./outputs", f"5head_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device_type = DeviceType(args.device_type) if args.device_type else get_device_type()
    device = get_device(device_type)
    
    print("=" * 60)
    print("Optimized 5-Head DINOv2 Training")
    print("=" * 60)
    print(f"Device: {device} ({device_type.value})")
    print(f"Backbone: {args.backbone}")
    print(f"Grid: {args.grid}")
    print(f"FiLM: {args.use_film}, AttnPool: {args.use_attention_pool}")
    print(f"Hidden ratio: {args.hidden_ratio}, Dropout: {args.dropout}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    if args.freeze_backbone:
        print(f"HEAD-ONLY Training: Backbone frozen, head LR: {args.head_lr_stage1:.2e}")
    elif args.two_stage:
        print(f"2-Stage Training:")
        print(f"  Stage 1: {args.freeze_epochs} epochs, head LR: {args.head_lr_stage1:.2e}")
        print(f"  Stage 2: remaining epochs, head LR: {args.lr:.2e}, backbone LR: {args.backbone_lr:.2e}")
    else:
        print(f"Single-stage: LR: {args.lr:.2e}, backbone LR: {args.backbone_lr:.2e}")
    print(f"Target weights: {args.target_weights}")
    print(f"Constraint weight: {args.constraint_weight}")
    print(f"Dead-aware loss: {args.use_dead_aware_loss}")
    print(f"Huber for dead: {args.use_huber_for_dead}")
    if args.use_aux_heads:
        print(f"Auxiliary heads: State (weight={args.aux_state_weight}), Month (weight={args.aux_month_weight})")
        if args.apply_context_adjustment:
            print(f"Context adjustment: ENABLED (predictions adjusted based on predicted state/month)")
    if args.use_log_target:
        print(f"Log target: ENABLED (log1p on targets, expm1 on predictions)")
    if args.stereo_correct_aug:
        print(f"Stereo-correct aug: ENABLED (photometric transforms independent per view)")
    if args.stereo_swap_prob > 0:
        print(f"Stereo swap prob: {args.stereo_swap_prob:.1%}")
    if args.mixup_prob > 0:
        print(f"MixUp: prob={args.mixup_prob:.1%}, alpha={args.mixup_alpha} (same species/month/state)")
    if args.cutmix_prob > 0:
        print(f"CutMix: prob={args.cutmix_prob:.1%}, alpha={args.cutmix_alpha} (same species/month/state)")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed, device_type)
    
    # Load data
    print("\nPreparing data...")
    df = prepare_dataframe(args.train_csv)
    df = create_folds(df, n_folds=args.num_folds, seed=args.seed, cv_strategy=args.cv_strategy)
    print(f"Total samples: {len(df)}")
    print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")
    
    # Save folds
    df.to_csv(os.path.join(args.output_dir, "folds.csv"), index=False)
    
    # Train
    results = []
    for fold in args.train_folds:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        
        result = train_fold(fold, train_df, valid_df, args, device, device_type)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary (Competition Weighted R²)")
    print("=" * 60)
    print("Weights: Green=0.1, Dead=0.1, Clover=0.1, GDM=0.2, Total=0.5")
    print("-" * 60)
    
    r2_scores = [r["best_r2"] for r in results]
    loss_scores = [r["best_loss"] for r in results]
    
    for r in results:
        metrics = r.get("best_metrics", {})
        # Show per-target R² breakdown
        r2_g = metrics.get("r2_green", 0)
        r2_d = metrics.get("r2_dead", 0)
        r2_c = metrics.get("r2_clover", 0)
        r2_gdm = metrics.get("r2_gdm", 0)
        r2_t = metrics.get("r2_total", 0)
        print(f"Fold {r['fold']}: Comp.R²={r['best_r2']:.4f} | "
              f"G={r2_g:.3f}, D={r2_d:.3f}, C={r2_c:.3f}, GDM={r2_gdm:.3f}, T={r2_t:.3f}")
    
    print("-" * 60)
    print(f"CV Mean Comp.R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"CV Mean Loss: {np.mean(loss_scores):.4f} ± {np.std(loss_scores):.4f}")
    print("\nNOTE: Run eval_5head_oof.py for true OOF R² (may be ~0.05-0.10 lower)")
    
    # Save results
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "backbone": args.backbone,
            "grid": args.grid,
            "dropout": args.dropout,
            "hidden_ratio": args.hidden_ratio,
            "use_film": args.use_film,
            "use_attention_pool": args.use_attention_pool,
            "two_stage": args.two_stage,
            "freeze_epochs": args.freeze_epochs if args.two_stage else 0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "head_lr_stage1": args.head_lr_stage1 if args.two_stage else None,
            "target_weights": args.target_weights,
            "constraint_weight": args.constraint_weight,
            "use_huber_for_dead": args.use_huber_for_dead,
            "correction_threshold": args.correction_threshold,
            "always_correct_dead": args.always_correct_dead,
            "cv_strategy": args.cv_strategy,
            "seed": args.seed,
            # New augmentation / target normalization settings
            "use_log_target": args.use_log_target,
            "stereo_correct_aug": args.stereo_correct_aug,
            "stereo_swap_prob": args.stereo_swap_prob,
            "mixup_prob": args.mixup_prob,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_prob": args.cutmix_prob,
            "cutmix_alpha": args.cutmix_alpha,
        },
        "fold_results": [
            {
                "fold": r["fold"],
                "best_r2": float(r["best_r2"]),
                "best_loss": float(r["best_loss"]),
                "best_metrics": r.get("best_metrics", {}),
            }
            for r in results
        ],
        "summary": {
            "cv_mean_r2": float(np.mean(r2_scores)),
            "cv_std_r2": float(np.std(r2_scores)),
            "cv_min_r2": float(np.min(r2_scores)),
            "cv_max_r2": float(np.max(r2_scores)),
            "cv_mean_loss": float(np.mean(loss_scores)),
            "cv_std_loss": float(np.std(loss_scores)),
        },
    }
    
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()

