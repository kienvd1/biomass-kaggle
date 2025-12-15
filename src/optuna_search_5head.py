#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for 5-Head DINOv2 Biomass Prediction.

Comprehensive search across all training options:
- Learning rates, dropout, hidden ratio
- Training strategies (two-stage, freeze-backbone, single-stage)
- Auxiliary heads configuration
- Loss functions
- Data augmentation strategies
- Target normalization

Usage:
    # Quick search (fewer trials, shorter epochs)
    python -m src.optuna_search_5head --n-trials 20 --max-epochs 15 --device-type cuda

    # Full search
    python -m src.optuna_search_5head --n-trials 100 --max-epochs 30 --device-type cuda

    # Resume search
    python -m src.optuna_search_5head --study-name my_study --storage sqlite:///optuna.db --resume
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
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
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
    ConstrainedMSELoss,
    DeadAwareLoss,
    DeadPostProcessor,
    FocalMSELoss,
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
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_device_seed(seed, device_type)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
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
    """Create optimizer with differential LR for backbone vs heads."""
    use_fused = supports_fused_optimizer(device_type)
    
    loss_params = []
    if loss_fn is not None:
        loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    
    if stage == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        params.extend(loss_params)
        return AdamW(params, lr=lr, weight_decay=weight_decay, fused=use_fused)
    else:
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
        
        if loss_params:
            param_groups.append({"params": loss_params, "lr": lr})
        
        return AdamW(param_groups, weight_decay=weight_decay, fused=use_fused)


def get_scheduler(
    optimizer: AdamW,
    num_epochs: int,
    warmup_epochs: int = 2,
    min_lr: float = 1e-7,
) -> Any:
    """Create scheduler with linear warmup."""
    if warmup_epochs > 0 and num_epochs > warmup_epochs:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0,
    use_aux: bool = False,
) -> float:
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    
    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        if use_aux and len(batch) == 6:
            x_left, x_right, targets, state_labels, month_labels, species_labels = batch
            state_labels = state_labels.to(device, non_blocking=True)
            month_labels = month_labels.to(device, non_blocking=True)
            species_labels = species_labels.to(device, non_blocking=True)
        else:
            x_left, x_right, targets = batch[:3]
            state_labels = month_labels = species_labels = None
        
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            if use_aux:
                green, dead, clover, gdm, total, state_logits, month_logits, species_logits = model(
                    x_left, x_right, return_aux=True
                )
            else:
                green, dead, clover, gdm, total = model(x_left, x_right)
                state_logits = month_logits = species_logits = None
            
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            
            if use_aux and isinstance(loss_fn, AuxiliaryLoss):
                loss, _ = loss_fn(
                    preds, targets,
                    state_logits=state_logits, state_labels=state_labels,
                    month_logits=month_logits, month_labels=month_labels,
                    species_logits=species_logits, species_labels=species_labels,
                )
            else:
                loss = loss_fn(preds, targets)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        losses.update(loss.item(), x_left.size(0))
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})
    
    return losses.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    use_aux: bool = False,
    use_log_target: bool = False,
    return_predictions: bool = False,
) -> Tuple[float, float, Dict[str, float], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Validate and return loss, R², per-target metrics, and optionally predictions."""
    model.eval()
    losses = AverageMeter()
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    
    all_preds, all_targets = [], []
    
    pbar = tqdm(loader, desc="  Valid", leave=False)
    for batch in pbar:
        if use_aux and len(batch) == 6:
            x_left, x_right, targets = batch[:3]
        else:
            x_left, x_right, targets = batch[:3]
        
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            if use_aux:
                green, dead, clover, gdm, total, _, _, _ = model(x_left, x_right, return_aux=True)
            else:
                green, dead, clover, gdm, total = model(x_left, x_right)
            
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            
            if isinstance(loss_fn, AuxiliaryLoss):
                loss = loss_fn.base_loss(preds, targets)
            else:
                loss = loss_fn(preds, targets)
        
        losses.update(loss.item(), x_left.size(0))
        all_preds.append(preds.float().cpu())
        all_targets.append(targets.cpu())
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Apply inverse transform if using log targets
    if use_log_target:
        all_preds = np.expm1(all_preds)
        all_targets = np.expm1(all_targets)
    
    r2 = compute_weighted_r2(all_preds, all_targets, target_weights)
    
    target_names = ["green", "dead", "clover", "gdm", "total"]
    per_target = compute_per_target_metrics_np(all_preds, all_targets, target_names, target_weights)
    
    metrics = {"weighted_r2": r2}
    for row in per_target:
        metrics[f"r2_{row['target']}"] = float(row["r2"])
    
    if return_predictions:
        return losses.avg, r2, metrics, (all_preds, all_targets)
    return losses.avg, r2, metrics, None


def train_fold_with_params(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: Dict[str, Any],
    backbone: str,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    max_epochs: int = 30,
    patience: int = 7,
    trial: Optional[optuna.Trial] = None,
    report_epoch_offset: int = 0,
) -> Tuple[float, float, Dict[str, float], np.ndarray, np.ndarray]:
    """Train a single fold with given hyperparameters."""
    
    # Extract parameters
    lr = params["lr"]
    backbone_lr = params["backbone_lr"]
    dropout = params["dropout"]
    hidden_ratio = params["hidden_ratio"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    grad_clip = params["grad_clip"]
    warmup_epochs = params["warmup_epochs"]
    
    training_mode = params["training_mode"]  # "freeze", "two_stage", "single"
    freeze_epochs = params.get("freeze_epochs", 5)
    head_lr_stage1 = params.get("head_lr_stage1", 1e-3)
    
    use_aux_heads = params["use_aux_heads"]
    aux_state_weight = params.get("aux_state_weight", 0.5)
    aux_month_weight = params.get("aux_month_weight", 0.5)
    aux_species_weight = params.get("aux_species_weight", 0.25)
    
    use_log_target = params["use_log_target"]
    stereo_correct_aug = params["stereo_correct_aug"]
    stereo_swap_prob = params["stereo_swap_prob"]
    mixup_prob = params.get("mixup_prob", 0.0)
    cutmix_prob = params.get("cutmix_prob", 0.0)
    
    constraint_weight = params.get("constraint_weight", 0.05)
    use_focal_loss = params.get("use_focal_loss", False)
    use_dead_aware_loss = params.get("use_dead_aware_loss", False)
    target_weights = params.get("target_weights", [0.2, 0.2, 0.2, 0.2, 0.2])
    
    img_size = params.get("img_size", 518)
    grid = params.get("grid", (2, 2))
    aug_prob = params.get("aug_prob", 0.5)
    
    # Create transforms
    if stereo_correct_aug:
        train_transform = get_stereo_geometric_transforms(img_size, aug_prob)
        photometric_transform = get_stereo_photometric_transforms(aug_prob)
    else:
        train_transform = get_train_transforms(img_size, aug_prob)
        photometric_transform = None
    
    valid_transform = get_valid_transforms(img_size)
    
    # Datasets
    train_ds = BiomassDataset(
        train_df, image_dir, train_transform,
        is_train=True, return_aux_labels=use_aux_heads,
        use_log_target=use_log_target, stereo_swap_prob=stereo_swap_prob,
        photometric_transform=photometric_transform,
        mixup_prob=mixup_prob, cutmix_prob=cutmix_prob,
    )
    valid_ds = BiomassDataset(
        valid_df, image_dir, valid_transform,
        is_train=False, return_aux_labels=use_aux_heads,
        use_log_target=use_log_target,
    )
    
    pin_memory = device_type == DeviceType.CUDA
    num_workers = 4
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    
    # Build model
    model = build_5head_model(
        backbone_name=backbone,
        grid=grid,
        pretrained=True,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
        use_film=True,
        use_attention_pool=True,
        use_aux_heads=use_aux_heads,
    ).to(device)
    
    # Loss function
    if use_dead_aware_loss:
        base_loss_fn = DeadAwareLoss(target_weights=target_weights, use_log_for_dead=True, aux_dead_weight=0.2)
    elif use_focal_loss:
        base_loss_fn = FocalMSELoss(target_weights=target_weights, gamma=1.0)
    else:
        base_loss_fn = ConstrainedMSELoss(target_weights=target_weights, constraint_weight=constraint_weight)
    
    if use_aux_heads:
        loss_fn = AuxiliaryLoss(
            base_loss=base_loss_fn,
            state_weight=aux_state_weight,
            month_weight=aux_month_weight,
            species_weight=aux_species_weight,
        )
    else:
        loss_fn = base_loss_fn
    
    # GradScaler for CUDA float16
    scaler = GradScaler("cuda") if device_type == DeviceType.CUDA else None
    
    # Training state
    current_stage = 1 if training_mode == "two_stage" else 2
    
    if training_mode == "freeze":
        freeze_backbone(model)
        optimizer = get_optimizer(model, head_lr_stage1, backbone_lr, weight_decay, device_type, stage=1, loss_fn=loss_fn)
        scheduler = get_scheduler(optimizer, max_epochs, warmup_epochs)
        current_stage = 1
    elif training_mode == "two_stage":
        freeze_backbone(model)
        optimizer = get_optimizer(model, head_lr_stage1, backbone_lr, weight_decay, device_type, stage=1, loss_fn=loss_fn)
        scheduler = get_scheduler(optimizer, freeze_epochs, warmup_epochs=1)
    else:  # single stage
        optimizer = get_optimizer(model, lr, backbone_lr, weight_decay, device_type, stage=2, loss_fn=loss_fn)
        scheduler = get_scheduler(optimizer, max_epochs, warmup_epochs)
    
    # Training loop
    best_loss = float("inf")
    best_r2 = 0.0
    best_metrics: Dict[str, float] = {}
    best_preds: Optional[np.ndarray] = None
    best_targets: Optional[np.ndarray] = None
    patience_counter = 0
    
    for epoch in range(1, max_epochs + 1):
        # Stage transition for two-stage training
        if training_mode == "two_stage" and current_stage == 1 and epoch > freeze_epochs:
            current_stage = 2
            print(f"    -> Stage 2: Unfreezing backbone")
            unfreeze_backbone(model)
            optimizer = get_optimizer(model, lr, backbone_lr, weight_decay, device_type, stage=2, loss_fn=loss_fn)
            remaining = max_epochs - freeze_epochs
            scheduler = get_scheduler(optimizer, remaining, warmup_epochs)
            patience_counter = 0
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, device_type,
            scaler=scaler, grad_clip=grad_clip, use_aux=use_aux_heads,
        )
        
        valid_loss, r2, metrics, preds_targets = validate(
            model, valid_loader, loss_fn, device, device_type,
            use_aux=use_aux_heads, use_log_target=use_log_target,
            return_predictions=True,
        )
        
        scheduler.step()
        
        # Check for NaN
        if np.isnan(train_loss) or np.isnan(valid_loss):
            print(f"    Epoch {epoch:02d} | NaN detected - pruning")
            raise optuna.TrialPruned()
        
        # Save best based on loss
        save_ok = (training_mode == "freeze") or (current_stage == 2)
        
        improved = ""
        if save_ok and valid_loss < best_loss:
            best_loss = valid_loss
            best_r2 = r2
            best_metrics = metrics
            # Store predictions for true OOF
            if preds_targets is not None:
                best_preds, best_targets = preds_targets
            patience_counter = 0
            improved = " *"
        elif save_ok:
            patience_counter += 1
        
        # Log epoch progress
        stage_str = f"S{current_stage}" if training_mode == "two_stage" else ""
        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"    Epoch {epoch:02d} {stage_str:>3} | Train: {train_loss:.4f} | Val: {valid_loss:.4f} | R²: {r2:.4f} | LR: {current_lr:.2e}{improved}")
        
        # Optuna pruning - report R² (we're maximizing)
        if trial is not None:
            trial.report(r2, report_epoch_offset + epoch)
            if trial.should_prune():
                print(f"    -> Pruned at epoch {epoch}")
                raise optuna.TrialPruned()
        
        # Early stopping
        if save_ok and patience_counter >= patience:
            print(f"    -> Early stop at epoch {epoch} (patience={patience})")
            break
    
    # Cleanup
    del model, optimizer, scheduler
    if scaler is not None:
        del scaler
    empty_cache(device_type)
    gc.collect()
    
    # Return predictions for true OOF computation
    if best_preds is None:
        best_preds = np.zeros((len(valid_df), 5))
        best_targets = np.zeros((len(valid_df), 5))
    
    return best_loss, best_r2, best_metrics, best_preds, best_targets


def define_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space - focused on learning rates."""
    
    # Training mode - focus on two_stage (has both head_lr_stage1 and lr)
    training_mode = trial.suggest_categorical("training_mode", ["freeze", "two_stage"])
    
    # Learning rates - MAIN TUNING TARGET
    head_lr_stage1 = trial.suggest_float("head_lr_stage1", 1e-4, 5e-3, log=True)
    
    if training_mode == "two_stage":
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)  # Head LR for stage 2
        backbone_lr = trial.suggest_float("backbone_lr", 1e-6, 5e-5, log=True)
        freeze_epochs = trial.suggest_int("freeze_epochs", 3, 8)
    else:  # freeze mode - only head_lr_stage1 matters
        lr = head_lr_stage1
        backbone_lr = 1e-6  # Not used
        freeze_epochs = 0
    
    # Fixed values
    batch_size = 32
    dropout = 0.2
    hidden_ratio = 0.5
    weight_decay = 0.01
    grad_clip = 0.5
    warmup_epochs = 2
    
    # Auxiliary heads - fixed off for simpler search
    use_aux_heads = False
    aux_state_weight = aux_month_weight = aux_species_weight = 0.0
    
    # Target normalization - disabled
    use_log_target = False
    
    # Data augmentation - fixed good defaults
    stereo_correct_aug = True
    stereo_swap_prob = 0.5
    mixup_prob = 0.0
    cutmix_prob = 0.0
    
    # Loss function - fixed
    loss_type = "constrained"
    constraint_weight = 0.05
    target_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    return {
        "training_mode": training_mode,
        "lr": lr,
        "backbone_lr": backbone_lr,
        "head_lr_stage1": head_lr_stage1,
        "dropout": dropout,
        "hidden_ratio": hidden_ratio,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "warmup_epochs": warmup_epochs,
        "freeze_epochs": freeze_epochs,
        "use_aux_heads": use_aux_heads,
        "aux_state_weight": aux_state_weight,
        "aux_month_weight": aux_month_weight,
        "aux_species_weight": aux_species_weight,
        "use_log_target": use_log_target,
        "stereo_correct_aug": stereo_correct_aug,
        "stereo_swap_prob": stereo_swap_prob,
        "mixup_prob": mixup_prob,
        "cutmix_prob": cutmix_prob,
        "loss_type": loss_type,
        "constraint_weight": constraint_weight,
        "use_focal_loss": False,
        "use_dead_aware_loss": False,
        "target_weight_strategy": "equal",
        "target_weights": target_weights,
        "img_size": 518,
        "grid": (2, 2),
        "aug_prob": 0.5,
    }


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    backbone: str,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    max_epochs: int,
    num_folds: int,
    patience: int,
) -> float:
    """Optuna objective - minimize CV loss."""
    
    params = define_search_space(trial)
    
    print(f"\n{'='*70}")
    print(f"Trial {trial.number}")
    print(f"{'='*70}")
    for k, v in params.items():
        if k not in ["target_weights", "img_size", "grid", "aug_prob"]:
            print(f"  {k}: {v}")
    print(f"{'='*70}")
    
    fold_losses = []
    fold_r2s = []
    all_metrics: List[Dict[str, float]] = []
    
    # For true OOF computation
    oof_preds_list = []
    oof_targets_list = []
    
    for fold in range(num_folds):
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        
        print(f"  Fold {fold}/{num_folds}...", end=" ", flush=True)
        
        try:
            loss, r2, metrics, fold_preds, fold_targets = train_fold_with_params(
                fold=fold,
                train_df=train_df,
                valid_df=valid_df,
                params=params,
                backbone=backbone,
                image_dir=image_dir,
                device=device,
                device_type=device_type,
                max_epochs=max_epochs,
                patience=patience,
                trial=trial if fold == 0 else None,  # Prune based on fold 0
                report_epoch_offset=fold * max_epochs,
            )
            fold_losses.append(loss)
            fold_r2s.append(r2)
            all_metrics.append(metrics)
            oof_preds_list.append(fold_preds)
            oof_targets_list.append(fold_targets)
            print(f"Loss: {loss:.4f}, R²: {r2:.4f}")
            
        except optuna.TrialPruned:
            print("PRUNED")
            raise
    
    cv_loss = np.mean(fold_losses)
    
    # Compute TRUE OOF R² (single R² on all predictions)
    oof_preds = np.concatenate(oof_preds_list, axis=0)
    oof_targets = np.concatenate(oof_targets_list, axis=0)
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    cv_r2 = compute_weighted_r2(oof_preds, oof_targets, target_weights)
    
    print(f"\n  CV Loss: {cv_loss:.4f} ± {np.std(fold_losses):.4f}")
    print(f"  TRUE OOF R² (competition): {cv_r2:.4f}")
    print(f"  Per-fold R² (mean): {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")
    
    # Per-target R² summary
    target_names = ["green", "dead", "clover", "gdm", "total"]
    for t in target_names:
        r2_vals = [m.get(f"r2_{t}", 0) for m in all_metrics]
        print(f"    {t}: {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
    
    # Store metrics
    trial.set_user_attr("cv_loss", float(cv_loss))
    trial.set_user_attr("cv_loss_std", float(np.std(fold_losses)))
    trial.set_user_attr("cv_r2_std", float(np.std(fold_r2s)))
    trial.set_user_attr("fold_losses", [float(l) for l in fold_losses])
    trial.set_user_attr("fold_r2s", [float(r) for r in fold_r2s])
    
    return cv_r2  # Maximize R² (competition metric)


def run_optuna_search(
    backbone: str,
    base_path: str = "./data",
    output_dir: Optional[str] = None,
    n_trials: int = 50,
    max_epochs: int = 30,
    num_folds: int = 5,
    patience: int = 7,
    seed: int = 18,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    cv_strategy: str = "group_date_state",
    device_type_str: Optional[str] = None,
    resume: bool = False,
    fold_csv: Optional[str] = None,
    gpu_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Run comprehensive Optuna hyperparameter search."""
    
    # Device setup
    device_type = DeviceType(device_type_str) if device_type_str else get_device_type()
    
    if device_type == DeviceType.CUDA and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = get_device(device_type)
    
    # Paths
    train_csv = os.path.join(base_path, "train.csv")
    image_dir = os.path.join(base_path, "train")
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("./outputs", f"optuna_5head_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    if study_name is None:
        backbone_short = backbone.split(".")[-1][:10]
        study_name = f"5head_{backbone_short}"
    
    print("=" * 70)
    print("Optuna 5-Head Hyperparameter Search")
    print("=" * 70)
    print(f"Backbone: {backbone}")
    print(f"Device: {device} ({device_type.value})")
    if device_type == DeviceType.CUDA:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Trials: {n_trials}")
    print(f"Max epochs/fold: {max_epochs}")
    print(f"Folds: {num_folds}")
    print(f"Patience: {patience}")
    print(f"CV strategy: {cv_strategy}")
    print(f"Study name: {study_name}")
    print(f"Storage: {storage or 'in-memory'}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Set seed
    set_seed(seed, device_type)
    
    # Load data
    print("\nPreparing data...")
    df = prepare_dataframe(train_csv)
    
    if fold_csv:
        print(f"Loading folds from: {fold_csv}")
        fold_df = pd.read_csv(fold_csv)
        fold_mapping = fold_df.set_index("sample_id_prefix")["fold"].to_dict()
        df["fold"] = df["sample_id_prefix"].map(fold_mapping).fillna(0).astype(int)
    else:
        df = create_folds(df, n_folds=num_folds, seed=seed, cv_strategy=cv_strategy)
    
    print(f"Total samples: {len(df)}")
    print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")
    
    # Save folds
    df.to_csv(os.path.join(output_dir, "folds.csv"), index=False)
    
    # Create study
    sampler = TPESampler(seed=seed, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)
    
    # Always load_if_exists when using storage (for parallel workers)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Maximize R² (competition metric)
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True if storage else False,
    )
    
    if storage and len(study.trials) > 0:
        print(f"Loaded existing study with {len(study.trials)} trials")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, df, backbone, image_dir, device, device_type, max_epochs, num_folds, patience
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    # Results
    print(f"\n{'='*70}")
    print("Optimization Complete!")
    print(f"{'='*70}")
    
    best_trial = study.best_trial
    print(f"\nBest Trial: {best_trial.number}")
    print(f"Best CV R² (competition): {best_trial.value:.4f}")
    print(f"Best CV Loss: {best_trial.user_attrs.get('cv_loss', 'N/A'):.4f}")
    
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "backbone": backbone,
        "study_name": study_name,
        "n_trials": n_trials,
        "max_epochs": max_epochs,
        "num_folds": num_folds,
        "patience": patience,
        "cv_strategy": cv_strategy,
        "best_trial": {
            "number": best_trial.number,
            "cv_r2": float(best_trial.value),  # Competition metric (maximized)
            "cv_r2_std": best_trial.user_attrs.get("cv_r2_std", 0),
            "cv_loss": best_trial.user_attrs.get("cv_loss", 0),
            "cv_loss_std": best_trial.user_attrs.get("cv_loss_std", 0),
            "fold_losses": best_trial.user_attrs.get("fold_losses", []),
            "fold_r2s": best_trial.user_attrs.get("fold_r2s", []),
            "params": best_trial.params,
        },
        "all_trials": [
            {
                "number": t.number,
                "cv_r2": float(t.value) if t.value is not None else None,  # Competition metric
                "cv_loss": t.user_attrs.get("cv_loss"),
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    
    results_path = os.path.join(output_dir, "optuna_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save best params
    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"Best params saved to: {best_params_path}")
    
    # Generate training command
    bp = best_trial.params
    cmd_parts = [
        "python -m src.train_5head",
        f"--backbone {backbone}",
        f"--dropout {bp.get('dropout', 0.2):.3f}",
        f"--hidden-ratio {bp.get('hidden_ratio', 0.5):.3f}",
        f"--batch-size {bp.get('batch_size', 8)}",
        f"--weight-decay {bp.get('weight_decay', 0.01):.4f}",
        f"--grad-clip {bp.get('grad_clip', 0.5):.2f}",
        f"--warmup-epochs {bp.get('warmup_epochs', 2)}",
    ]
    
    training_mode = bp.get("training_mode", "freeze")
    if training_mode == "freeze":
        cmd_parts.append("--freeze-backbone")
        cmd_parts.append(f"--head-lr-stage1 {bp.get('head_lr_stage1', 1e-3):.2e}")
    elif training_mode == "two_stage":
        cmd_parts.append("--two-stage")
        cmd_parts.append(f"--freeze-epochs {bp.get('freeze_epochs', 5)}")
        cmd_parts.append(f"--head-lr-stage1 {bp.get('head_lr_stage1', 1e-3):.2e}")
        cmd_parts.append(f"--lr {bp.get('lr', 2e-4):.2e}")
        cmd_parts.append(f"--backbone-lr {bp.get('backbone_lr', 1e-5):.2e}")
    else:
        cmd_parts.append(f"--lr {bp.get('lr', 2e-4):.2e}")
        cmd_parts.append(f"--backbone-lr {bp.get('backbone_lr', 1e-5):.2e}")
    
    if bp.get("use_aux_heads", False):
        cmd_parts.append("--use-aux-heads")
        cmd_parts.append(f"--aux-state-weight {bp.get('aux_state_weight', 0.5):.2f}")
        cmd_parts.append(f"--aux-month-weight {bp.get('aux_month_weight', 0.5):.2f}")
        cmd_parts.append(f"--aux-species-weight {bp.get('aux_species_weight', 0.25):.2f}")
    
    if bp.get("stereo_correct_aug", False):
        cmd_parts.append("--stereo-correct-aug")
    
    if bp.get("stereo_swap_prob", 0) > 0:
        cmd_parts.append(f"--stereo-swap-prob {bp.get('stereo_swap_prob', 0):.2f}")
    
    if bp.get("mixup_prob", 0) > 0:
        cmd_parts.append(f"--mixup-prob {bp.get('mixup_prob', 0):.2f}")
    
    if bp.get("cutmix_prob", 0) > 0:
        cmd_parts.append(f"--cutmix-prob {bp.get('cutmix_prob', 0):.2f}")
    
    loss_type = bp.get("loss_type", "constrained")
    if loss_type == "focal":
        cmd_parts.append("--use-focal-loss")
    elif loss_type == "dead_aware":
        cmd_parts.append("--use-dead-aware-loss")
    else:
        cmd_parts.append(f"--constraint-weight {bp.get('constraint_weight', 0.05):.3f}")
    
    cmd = " \\\n    ".join(cmd_parts)
    print(f"\n{'='*70}")
    print("Recommended Training Command:")
    print("="*70)
    print(cmd)
    
    # Save command
    cmd_path = os.path.join(output_dir, "best_training_cmd.sh")
    with open(cmd_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Generated from Optuna search - Trial {best_trial.number}\n")
        f.write(f"# CV Loss: {best_trial.value:.4f}, CV R²: {best_trial.user_attrs.get('cv_r2', 0):.4f}\n\n")
        f.write(cmd + "\n")
    print(f"\nTraining command saved to: {cmd_path}")
    
    return best_trial.params


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna 5-Head Hyperparameter Search")
    
    # Required
    parser.add_argument(
        "--backbone", type=str, default="vit_base_patch14_reg4_dinov2.lvd142m",
        help="DINOv2 backbone model"
    )
    
    # Paths
    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--fold-csv", type=str, default=None,
                        help="CSV with pre-defined folds (sample_id_prefix, fold columns)")
    
    # Search config
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--max-epochs", type=int, default=30, help="Max epochs per fold")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=18)
    
    # Optuna
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    
    # CV
    parser.add_argument("--cv-strategy", type=str, default="group_date_state",
                        choices=["group_month", "group_date", "group_date_state", "stratified", "random"])
    
    # Device
    parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="Specific GPU ID to use (for parallel workers). If not set, uses GPU 0.")
    
    args = parser.parse_args()
    
    run_optuna_search(
        backbone=args.backbone,
        base_path=args.base_path,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        num_folds=args.num_folds,
        patience=args.patience,
        seed=args.seed,
        study_name=args.study_name,
        storage=args.storage,
        cv_strategy=args.cv_strategy,
        device_type_str=args.device_type,
        resume=args.resume,
        fold_csv=args.fold_csv,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()

