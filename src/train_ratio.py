#!/usr/bin/env python3
"""
Training script for Softmax Ratio DINOv2 model.

Key innovation: Instead of predicting 5 targets independently, we predict:
1. Total biomass (strongest visual signal)
2. Softmax ratios for (Green, Dead, Clover) that sum to 1

This guarantees: Green + Dead + Clover = Total (always!)

Usage:
    python -m src.train_ratio --device-type mps
    python -m src.train_ratio --device-type cuda --batch-size 16
    
    # With hierarchical model:
    python -m src.train_ratio --model-type hierarchical
    
    # With 2-stage training:
    python -m src.train_ratio --two-stage --freeze-epochs 5
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
from .models_ratio import build_ratio_model, RatioMSELoss
from .trainer import compute_weighted_r2, compute_per_target_metrics_np
from .eval_ratio_oof import (
    OOFDataset,
    get_val_transform as get_oof_transform,
    load_model as load_oof_model,
    predict_fold as predict_oof_fold,
    compute_metrics,
)


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
    use_ratio_loss: bool = False,
    disable_amp: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch with gradient scaling and clipping."""
    model.train()
    losses = AverageMeter()
    loss_components: Dict[str, AverageMeter] = {
        "main": AverageMeter(),
        "ratio": AverageMeter(),
    }

    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    if disable_amp:
        use_amp = False

    pbar = tqdm(loader, desc="Training", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        x_left, x_right, targets = batch[:3]

        # Use channels-last for MPS performance
        if device_type == DeviceType.MPS:
            x_left = x_left.to(device, non_blocking=True, memory_format=torch.channels_last)
            x_right = x_right.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_left = x_left.to(device, non_blocking=True)
            x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            if use_ratio_loss:
                outputs = model(x_left, x_right, return_ratios=True)
                # SoftmaxRatioDINO returns 6 values, HierarchicalRatioDINO returns 7
                if len(outputs) == 6:
                    green, dead, clover, gdm, total, ratios = outputs
                else:
                    green, dead, clover, gdm, total, alive_ratio, green_ratio = outputs
                    # For ratio loss, construct ratios tensor
                    total_safe = total + 1e-8
                    ratios = torch.cat([
                        green / total_safe,
                        dead / total_safe,
                        clover / total_safe,
                    ], dim=1)
            else:
                green, dead, clover, gdm, total = model(x_left, x_right)
                ratios = None

            preds = torch.cat([green, dead, clover, gdm, total], dim=1)

            # Compute loss
            if use_ratio_loss and ratios is not None:
                loss = loss_fn(preds, targets, pred_ratios=ratios)
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
    use_log_target: bool = False,
    disable_amp: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """Validate model."""
    model.eval()
    losses = AverageMeter()
    
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    if disable_amp:
        use_amp = False
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]  # Competition weights
    
    all_preds = []
    all_targets = []
    all_ratios = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        x_left, x_right, targets = batch[:3]

        # Use channels-last for MPS performance
        if device_type == DeviceType.MPS:
            x_left = x_left.to(device, non_blocking=True, memory_format=torch.channels_last)
            x_right = x_right.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_left = x_left.to(device, non_blocking=True)
            x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            outputs = model(x_left, x_right, return_ratios=True)
            # SoftmaxRatioDINO returns 6 values, HierarchicalRatioDINO returns 7
            if len(outputs) == 6:
                green, dead, clover, gdm, total, ratios = outputs
            else:
                # HierarchicalRatioDINO: returns alive_ratio, green_ratio separately
                green, dead, clover, gdm, total, alive_ratio, green_ratio = outputs
                # Construct ratios tensor: [green/total, dead/total, clover/total]
                total_safe = total + 1e-8
                ratios = torch.cat([
                    green / total_safe,
                    dead / total_safe,
                    clover / total_safe,
                ], dim=1)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            loss = loss_fn(preds, targets)
        
        losses.update(loss.item(), x_left.size(0))
        all_preds.append(preds.float().cpu())
        all_targets.append(targets.cpu())
        all_ratios.append(ratios.float().cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_ratios = torch.cat(all_ratios, dim=0).numpy()
    
    # Apply inverse transform if using log targets
    if use_log_target:
        all_preds = np.expm1(all_preds)
        all_targets = np.expm1(all_targets)
    
    # Compute metrics
    r2 = compute_weighted_r2(all_preds, all_targets, target_weights)
    
    # Per-target metrics
    target_names = ["green", "dead", "clover", "gdm", "total"]
    per_target = compute_per_target_metrics_np(all_preds, all_targets, target_names, target_weights)
    
    metrics = {}
    for row in per_target:
        t = row["target"]
        metrics[f"r2_{t}"] = float(row["r2"])
        metrics[f"rmse_{t}"] = float(row["rmse"])
    
    metrics["weighted_r2"] = r2
    
    # Ratio statistics
    metrics["ratio_green_mean"] = float(all_ratios[:, 0].mean())
    metrics["ratio_dead_mean"] = float(all_ratios[:, 1].mean())
    metrics["ratio_clover_mean"] = float(all_ratios[:, 2].mean())

    return losses.avg, r2, metrics


def freeze_backbone_fn(model: nn.Module) -> None:
    """Freeze backbone parameters."""
    base_model = model.module if hasattr(model, "module") else model
    for param in base_model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone_fn(model: nn.Module) -> None:
    """Unfreeze backbone parameters."""
    base_model = model.module if hasattr(model, "module") else model
    for param in base_model.backbone.parameters():
        param.requires_grad = True


def get_optimizer(
    model: nn.Module,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    device_type: DeviceType,
    stage: int = 2,
) -> AdamW:
    """Create optimizer with differential LR for backbone vs heads."""
    use_fused = supports_fused_optimizer(device_type)
    
    if stage == 1:
        # Stage 1: Only head parameters (backbone frozen)
        params = [p for p in model.parameters() if p.requires_grad]
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
    """Train a single fold."""
    two_stage = getattr(cfg, "two_stage", False)
    freeze_epochs = getattr(cfg, "freeze_epochs", 5)
    freeze_backbone = getattr(cfg, "freeze_backbone", False)
    
    if freeze_backbone:
        two_stage = False
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}")
    print(f"Model type: {cfg.model_type}")
    if freeze_backbone:
        print(f"HEAD-ONLY MODE: Backbone frozen, training heads only")
    elif two_stage:
        print(f"2-Stage: Stage 1 ({freeze_epochs} epochs frozen), Stage 2 (finetune)")
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
        print(f"MIXUP: prob={mixup_prob:.1%}, alpha={mixup_alpha}")
    
    if cutmix_prob > 0:
        print(f"CUTMIX: prob={cutmix_prob:.1%}, alpha={cutmix_alpha}")
    
    valid_transform = get_valid_transforms(cfg.img_size)
    
    num_workers = 0 if cfg.cache_images else cfg.num_workers
    if cfg.cache_images and cfg.num_workers > 0:
        print(f"WARNING: cache_images=True, forcing num_workers=0 (was {cfg.num_workers})")
    
    train_ds = BiomassDataset(
        train_df, cfg.train_image_dir, train_transform, 
        is_train=True, cache_images=cfg.cache_images, return_aux_labels=False,
        use_log_target=use_log_target, stereo_swap_prob=stereo_swap_prob,
        photometric_transform=photometric_transform,
        mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
        cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha,
    )
    valid_ds = BiomassDataset(
        valid_df, cfg.train_image_dir, valid_transform, 
        is_train=False, cache_images=cfg.cache_images, return_aux_labels=False,
        use_log_target=use_log_target,
    )
    
    pin_memory = device_type == DeviceType.CUDA
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=num_workers > 0, prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
    )
    valid_batch_multiplier = 4 if device_type == DeviceType.MPS else 2
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size * valid_batch_multiplier, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
    )

    # Model
    model = build_ratio_model(
        backbone_name=cfg.backbone,
        grid=tuple(cfg.grid),
        pretrained=True,
        dropout=cfg.dropout,
        hidden_ratio=cfg.hidden_ratio,
        use_film=getattr(cfg, "use_film", True),
        use_attention_pool=getattr(cfg, "use_attention_pool", True),
        gradient_checkpointing=getattr(cfg, "grad_ckpt", False),
        model_type=cfg.model_type,
        ratio_temperature=getattr(cfg, "ratio_temperature", 1.0),
        use_vegetation_indices=getattr(cfg, "use_vegetation_indices", False),
        use_multiscale=getattr(cfg, "use_multiscale", False),
        multiscale_layers=getattr(cfg, "multiscale_layers", None),
    ).to(device)

    if device_type == DeviceType.MPS:
        model = model.to(memory_format=torch.channels_last)
    
    use_multi_gpu = getattr(cfg, "multi_gpu", False) and device_type == DeviceType.CUDA
    num_gpus = torch.cuda.device_count() if use_multi_gpu else 1
    if use_multi_gpu and num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {num_gpus} GPUs")

    total_params = sum(p.numel() for p in model.parameters())
    model_names = {"hierarchical": "HierarchicalRatioDINO", "direct": "DirectDINO", "softmax": "SoftmaxRatioDINO"}
    model_name = model_names.get(cfg.model_type, cfg.model_type)
    print(f"Model: {model_name} ({cfg.backbone})")
    print(f"Grid: {cfg.grid}, FiLM: {getattr(cfg, 'use_film', True)}, AttnPool: {getattr(cfg, 'use_attention_pool', True)}")
    print(f"Total params: {total_params:,}")
    
    # Setup training stages
    current_stage = 1 if two_stage else 2
    
    if freeze_backbone:
        freeze_backbone_fn(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"HEAD-ONLY: Backbone FROZEN ({trainable_params:,} / {total_params:,} params trainable)")
        head_lr = getattr(cfg, "head_lr_stage1", 1e-3)
        optimizer = get_optimizer(model, head_lr, cfg.backbone_lr, cfg.weight_decay, device_type, stage=1)
        scheduler = get_scheduler_with_warmup(optimizer, cfg.epochs, warmup_epochs=cfg.warmup_epochs)
        current_stage = 1
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
    target_weights = getattr(cfg, "target_weights", [0.1, 0.1, 0.1, 0.2, 0.5])
    loss_fn = RatioMSELoss(
        target_weights=target_weights,
        use_huber_for_dead=getattr(cfg, "use_huber_for_dead", True),
        huber_delta=getattr(cfg, "huber_delta", 5.0),
        ratio_loss_weight=getattr(cfg, "ratio_loss_weight", 0.0),
        ratio_loss_type=getattr(cfg, "ratio_loss_type", "mse"),
    )
    
    # Training loop
    best_r2 = 0.0
    best_loss = float("inf")
    patience_counter = 0
    best_metrics_snapshot: Optional[Dict[str, float]] = None
    history: Dict[str, List] = {
        "train_loss": [], "valid_loss": [], "weighted_r2": [],
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
            patience_counter = 0
        
        # Train
        train_loss, _ = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, device_type,
            scaler=scaler,
            grad_clip=cfg.grad_clip,
            grad_accum_steps=getattr(cfg, "grad_accum", 1),
            use_ratio_loss=cfg.ratio_loss_weight > 0,
            disable_amp=getattr(cfg, "no_amp", False),
        )
        
        # Validate
        valid_loss, r2, metrics = validate(
            model, valid_loader, loss_fn, device, device_type,
            use_log_target=use_log_target,
            disable_amp=getattr(cfg, "no_amp", False),
        )
        
        scheduler.step()
        
        # Logging
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[-1]["lr"]
        stage_str = f"S{current_stage}" if two_stage else ""
        
        r2_vec = [metrics.get(f"r2_{n}", 0.0) for n in ["green", "dead", "clover", "gdm", "total"]]
        
        loss_suffix = " (log)" if use_log_target else ""
        print(
            f"Epoch {epoch:02d} {stage_str:>3} | "
            f"Train{loss_suffix}: {train_loss:.4f} | "
            f"Valid{loss_suffix}: {valid_loss:.4f} | "
            f"Comp.R²: {r2:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )
        print(f"   per-target R²[g,d,c,gdm,t]=[{', '.join(f'{x:.3f}' for x in r2_vec)}]")
        print(f"   ratios[g,d,c]=[{metrics['ratio_green_mean']:.3f}, {metrics['ratio_dead_mean']:.3f}, {metrics['ratio_clover_mean']:.3f}]")
        
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["weighted_r2"].append(r2)
        history["per_target_r2"].append(r2_vec)
        history["stage"].append(current_stage)
        history["lr"].append(lr)
        
        # Save best model
        head_only_mode = two_stage and freeze_epochs >= cfg.epochs
        save_ok = (current_stage == 2) or (not two_stage) or head_only_mode or freeze_backbone
        
        is_better = r2 > best_r2 and not np.isnan(r2)
        
        if save_ok and is_better:
            best_r2 = r2
            best_loss = valid_loss
            best_metrics_snapshot = {k: float(v) for k, v in metrics.items()}
            patience_counter = 0
            
            save_path = os.path.join(cfg.output_dir, f"ratio_best_fold{fold}.pth")
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), save_path)
            print(f"  -> Saved best model (R²: {best_r2:.4f})")
        elif save_ok:
            patience_counter += 1
        
        # Early stopping on NaN
        if np.isnan(train_loss) or np.isnan(valid_loss):
            print(f"Training diverged (NaN detected) at epoch {epoch}")
            break
        
        # Early stopping
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
    parser = argparse.ArgumentParser(description="Softmax Ratio DINOv2 Training")
    
    # Paths
    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default=None)
    
    # Model architecture
    parser.add_argument(
        "--backbone", type=str, default="vit_base_patch14_reg4_dinov2.lvd142m",
        help="Vision backbone. Options: "
             "DINOv2: vit_base_patch14_reg4_dinov2.lvd142m (518x518), "
             "vit_large_patch14_reg4_dinov2.lvd142m; "
             "DINOv3: vit_base_patch16_dinov3 (256x256), "
             "vit_large_patch16_dinov3, vit_huge_plus_patch16_dinov3"
    )
    parser.add_argument("--grid", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-ratio", type=float, default=0.5)
    parser.add_argument("--no-film", action="store_true", help="Disable FiLM conditioning")
    parser.add_argument("--no-attention-pool", action="store_true", help="Disable attention pooling")
    parser.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--use-vegetation-indices", action="store_true",
                        help="Add vegetation indices (ExG, ExR, GRVI, etc.) as auxiliary features")
    parser.add_argument("--use-multiscale", action="store_true",
                        help="Enable multi-scale feature extraction from multiple transformer layers")
    parser.add_argument("--multiscale-layers", type=int, nargs="+", default=[5, 8, 11],
                        help="Transformer layers to extract for multi-scale (default: 5 8 11)")
    
    # Model type
    parser.add_argument("--model-type", type=str, default="softmax", choices=["softmax", "hierarchical", "direct"],
                        help="Model type: 'softmax', 'hierarchical', or 'direct' (predict Total/Green/GDM, derive Dead/Clover)")
    parser.add_argument("--ratio-temperature", type=float, default=1.0,
                        help="Temperature for softmax ratio (only for softmax model)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-images", action="store_true", help="Cache images in RAM")
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
    
    # Loss function
    parser.add_argument("--target-weights", type=float, nargs=5, 
                        default=[0.1, 0.1, 0.1, 0.2, 0.5],
                        help="Loss weights for [green, dead, clover, gdm, total]")
    parser.add_argument("--no-huber-for-dead", action="store_true", help="Disable Huber loss for dead target")
    parser.add_argument("--huber-delta", type=float, default=5.0, help="Huber loss delta for dead")
    parser.add_argument("--ratio-loss-weight", type=float, default=0.0,
                        help="Weight for auxiliary ratio loss")
    parser.add_argument("--ratio-loss-type", type=str, default="mse", choices=["mse", "kl"],
                        help="Ratio loss type: 'mse' or 'kl' (KL divergence)")
    
    # Augmentation
    parser.add_argument("--img-size", type=int, default=518,
                        help="Input image size. Use 518 for DINOv2, 512 for DINOv3 with 2x2 grid, "
                             "256 for DINOv3 with 1x1 grid")
    parser.add_argument("--aug-prob", type=float, default=0.5)
    parser.add_argument("--stereo-correct-aug", action="store_true",
                        help="Apply photometric transforms independently per stereo view")
    parser.add_argument("--stereo-swap-prob", type=float, default=0.0,
                        help="Probability of swapping L/R stereo images")
    
    # Target normalization
    parser.add_argument("--use-log-target", action="store_true",
                        help="Apply log1p to targets")
    
    # MixUp/CutMix
    parser.add_argument("--mixup-prob", type=float, default=0.0,
                        help="MixUp probability")
    parser.add_argument("--mixup-alpha", type=float, default=0.4,
                        help="MixUp beta distribution alpha parameter")
    parser.add_argument("--cutmix-prob", type=float, default=0.0,
                        help="CutMix probability")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                        help="CutMix beta distribution alpha parameter")
    
    # AMP
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--no-amp", action="store_true", 
                        help="Disable automatic mixed precision (fixes NaN with DINOv3 on MPS)")
    
    # CV
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--train-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cv-strategy", type=str, default="group_date",
                        help="CV strategy: group_date_state (default), group_month, group_date, stratified, random")
    parser.add_argument("--fold-csv", type=str, default=None,
                        help="Path to CSV with pre-defined folds")
    
    # Device
    parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--multi-gpu", action="store_true", 
                        help="Use DataParallel for multi-GPU training")
    
    # OOF computation
    parser.add_argument("--compute-oof", action="store_true",
                        help="Compute true OOF metrics after training completes")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    
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
        args.output_dir = os.path.join("./outputs", f"ratio_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device_type = DeviceType(args.device_type) if args.device_type else get_device_type()
    device = get_device(device_type)
    
    print("=" * 60)
    print("Softmax Ratio DINOv2 Training")
    print("=" * 60)
    print(f"Device: {device} ({device_type.value})")
    print(f"Model type: {args.model_type}")
    print(f"Backbone: {args.backbone}")
    print(f"Grid: {args.grid}")
    print(f"FiLM: {args.use_film}, AttnPool: {args.use_attention_pool}, VegIndices: {args.use_vegetation_indices}")
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
    print(f"Huber for dead: {args.use_huber_for_dead}")
    if args.ratio_loss_weight > 0:
        print(f"Ratio loss weight: {args.ratio_loss_weight}")
    if args.use_log_target:
        print(f"Log target: ENABLED")
    if args.stereo_correct_aug:
        print(f"Stereo-correct aug: ENABLED")
    if args.stereo_swap_prob > 0:
        print(f"Stereo swap prob: {args.stereo_swap_prob:.1%}")
    if args.mixup_prob > 0:
        print(f"MixUp: prob={args.mixup_prob:.1%}, alpha={args.mixup_alpha}")
    if args.cutmix_prob > 0:
        print(f"CutMix: prob={args.cutmix_prob:.1%}, alpha={args.cutmix_alpha}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed, device_type)
    
    # Load data
    print("\nPreparing data...")
    df = prepare_dataframe(args.train_csv)
    
    # Use pre-defined folds from CSV or create new folds
    if args.fold_csv:
        print(f"Loading pre-defined folds from: {args.fold_csv}")
        fold_df = pd.read_csv(args.fold_csv)
        if "sample_id_prefix" in fold_df.columns and "fold" in fold_df.columns:
            fold_mapping = fold_df.set_index("sample_id_prefix")["fold"].to_dict()
            df["fold"] = df["sample_id_prefix"].map(fold_mapping)
            unmapped = df["fold"].isna().sum()
            if unmapped > 0:
                print(f"WARNING: {unmapped} samples not found in fold CSV, assigning to fold 0")
                df["fold"] = df["fold"].fillna(0).astype(int)
            else:
                df["fold"] = df["fold"].astype(int)
        else:
            raise ValueError("fold_csv must have 'sample_id_prefix' and 'fold' columns")
    else:
        df = create_folds(df, n_folds=args.num_folds, seed=args.seed, cv_strategy=args.cv_strategy)
    
    print(f"Total samples: {len(df)}")
    print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")
    
    # Save folds
    df.to_csv(os.path.join(args.output_dir, "folds.csv"), index=False)
    
    # Initialize OOF tracking if enabled
    n_samples = len(df)
    oof_preds = np.full((n_samples, 5), np.nan, dtype=np.float32) if args.compute_oof else None
    oof_targets = np.zeros((n_samples, 5), dtype=np.float32) if args.compute_oof else None
    oof_ratios = np.zeros((n_samples, 3), dtype=np.float32) if args.compute_oof else None
    oof_filled = np.zeros(n_samples, dtype=bool) if args.compute_oof else None
    
    if args.compute_oof:
        image_dir = os.path.join(args.base_path, "train")
        oof_transform = get_oof_transform(args.img_size)
    
    # Train
    results = []
    for fold in args.train_folds:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        
        result = train_fold(fold, train_df, valid_df, args, device, device_type)
        results.append(result)
        
        # Compute OOF for this fold immediately after training
        if args.compute_oof:
            ckpt_path = os.path.join(args.output_dir, f"ratio_best_fold{fold}.pth")
            if os.path.exists(ckpt_path):
                print(f"\n  Computing OOF predictions for fold {fold}...")
                
                # Load best model for this fold
                model = load_oof_model(
                    ckpt_path, args.backbone, device,
                    model_type=args.model_type,
                    grid=tuple(args.grid), dropout=args.dropout, hidden_ratio=args.hidden_ratio,
                    use_film=args.use_film, use_attention_pool=args.use_attention_pool,
                    use_vegetation_indices=args.use_vegetation_indices,
                    use_multiscale=args.use_multiscale,
                    multiscale_layers=args.multiscale_layers,
                )
                
                # Get validation indices and predictions
                val_mask = df["fold"] == fold
                val_indices = df.index[val_mask].tolist()
                val_df_oof = df[val_mask].reset_index(drop=True)
                
                val_ds = OOFDataset(val_df_oof, image_dir, oof_transform)
                val_loader = DataLoader(
                    val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=device.type == "cuda"
                )
                
                fold_preds, fold_targets, fold_ratios = predict_oof_fold(
                    model, val_loader, device, desc=f"  OOF Fold {fold}"
                )
                
                # Store in OOF arrays
                for i, idx in enumerate(val_indices):
                    oof_preds[idx] = fold_preds[i]
                    oof_targets[idx] = fold_targets[i]
                    oof_ratios[idx] = fold_ratios[i]
                    oof_filled[idx] = True
                
                # Cleanup
                del model
                gc.collect()
                empty_cache(device_type)
                
                # Compute cumulative OOF metrics
                filled_mask = oof_filled
                if filled_mask.sum() > 0:
                    oof_metrics = compute_metrics(oof_preds[filled_mask], oof_targets[filled_mask])
                    n_filled = filled_mask.sum()
                    pct_filled = 100.0 * n_filled / n_samples
                    print(f"\n  OOF after fold {fold} ({n_filled}/{n_samples} = {pct_filled:.0f}% samples):")
                    print(f"    Weighted R²: {oof_metrics['weighted_r2']:.4f} | "
                          f"G={oof_metrics['r2_green']:.3f}, D={oof_metrics['r2_dead']:.3f}, "
                          f"C={oof_metrics['r2_clover']:.3f}, GDM={oof_metrics['r2_gdm']:.3f}, T={oof_metrics['r2_total']:.3f}")
    
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
    
    # Save results
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "backbone": args.backbone,
            "model_type": args.model_type,
            "img_size": args.img_size,
            "grid": args.grid,
            "dropout": args.dropout,
            "hidden_ratio": args.hidden_ratio,
            "use_film": args.use_film,
            "use_attention_pool": args.use_attention_pool,
            "use_vegetation_indices": args.use_vegetation_indices,
            "use_multiscale": args.use_multiscale,
            "multiscale_layers": args.multiscale_layers,
            "two_stage": args.two_stage,
            "freeze_epochs": args.freeze_epochs if args.two_stage else 0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "head_lr_stage1": args.head_lr_stage1 if args.two_stage else None,
            "target_weights": args.target_weights,
            "use_huber_for_dead": args.use_huber_for_dead,
            "ratio_loss_weight": args.ratio_loss_weight,
            "cv_strategy": args.cv_strategy,
            "seed": args.seed,
            "use_log_target": args.use_log_target,
            "stereo_correct_aug": args.stereo_correct_aug,
            "stereo_swap_prob": args.stereo_swap_prob,
            "mixup_prob": args.mixup_prob,
            "cutmix_prob": args.cutmix_prob,
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
    
    # Final OOF summary
    if args.compute_oof and oof_filled is not None and oof_filled.all():
        oof_metrics_raw = compute_metrics(oof_preds, oof_targets)
        
        print("\n" + "=" * 60)
        print("Final OOF Results (Competition Weighted R²)")
        print("=" * 60)
        for name in ["Green", "Dead", "Clover", "GDM", "Total"]:
            r2 = oof_metrics_raw[f"r2_{name.lower()}"]
            rmse = oof_metrics_raw[f"rmse_{name.lower()}"]
            mae = oof_metrics_raw[f"mae_{name.lower()}"]
            print(f"  {name:8s}: R² = {r2:7.4f}, RMSE = {rmse:7.2f}, MAE = {mae:7.2f}")
        print(f"\n  Weighted R²: {oof_metrics_raw['weighted_r2']:.4f}")
        
        # Check constraint: G+D+C = Total
        component_sum = oof_preds[:, 0] + oof_preds[:, 1] + oof_preds[:, 2]
        total_pred = oof_preds[:, 4]
        constraint_diff = np.abs(component_sum - total_pred)
        print(f"\n  Constraint check (G+D+C=T): max_diff={constraint_diff.max():.6f}, mean_diff={constraint_diff.mean():.6f}")
        
        # Ratio statistics
        print(f"  Ratios: G={oof_ratios[:, 0].mean():.3f}, D={oof_ratios[:, 1].mean():.3f}, C={oof_ratios[:, 2].mean():.3f}")
        
        # Save OOF predictions
        oof_df_out = df[["sample_id_prefix", "fold"]].copy()
        oof_df_out["pred_green"] = oof_preds[:, 0]
        oof_df_out["pred_dead"] = oof_preds[:, 1]
        oof_df_out["pred_clover"] = oof_preds[:, 2]
        oof_df_out["pred_gdm"] = oof_preds[:, 3]
        oof_df_out["pred_total"] = oof_preds[:, 4]
        oof_df_out["true_green"] = oof_targets[:, 0]
        oof_df_out["true_dead"] = oof_targets[:, 1]
        oof_df_out["true_clover"] = oof_targets[:, 2]
        oof_df_out["true_gdm"] = oof_targets[:, 3]
        oof_df_out["true_total"] = oof_targets[:, 4]
        oof_df_out["ratio_green"] = oof_ratios[:, 0]
        oof_df_out["ratio_dead"] = oof_ratios[:, 1]
        oof_df_out["ratio_clover"] = oof_ratios[:, 2]
        
        oof_path = os.path.join(args.output_dir, "oof_predictions.csv")
        oof_df_out.to_csv(oof_path, index=False)
        print(f"\nOOF predictions saved to: {oof_path}")
        
        # Save OOF metrics
        oof_metrics_path = os.path.join(args.output_dir, "oof_metrics.json")
        with open(oof_metrics_path, "w") as f:
            json.dump({"raw": oof_metrics_raw}, f, indent=2)
        print(f"OOF metrics saved to: {oof_metrics_path}")
        
        # Update results summary with OOF
        results_summary["oof_metrics"] = {"raw": oof_metrics_raw}
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
