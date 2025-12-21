#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Ratio DINOv2 Biomass Prediction.

Searches across:
- Model type (softmax vs hierarchical)
- Architecture (dropout, hidden_ratio, FiLM, attention pooling)
- Learning rates
- Loss configuration
- Data augmentation

Search Phases (progressive complexity):
- Phase 0: BASIC - Essential params only (model_type, training_mode, lr, batch_size)
- Phase 1: CORE  - Add architecture params (dropout, hidden_ratio, grad_clip)
- Phase 2: ARCH  - Add advanced architecture (FiLM, attention pool, grid, weight_decay)
- Phase 3: REG   - Add regularization & augmentation (huber, ratio_loss, mixup, cutmix)

Recommended workflow:
1. Start with Phase 0 to find optimal basic setup (~20-30 trials)
2. Move to Phase 1 with best Phase 0 params as starting point (~30-50 trials)
3. Continue to Phase 2/3 for fine-tuning

Usage:
    # RECOMMENDED: Run all phases with one command (auto-persistence)
    python -m src.optuna_search_ratio --search-all --device-type mps

    # Custom trials per phase (default: 30 40 40 50)
    python -m src.optuna_search_ratio --search-all --trials-per-phase 20 30 30 40

    # Single phase mode
    python -m src.optuna_search_ratio --search-phase 0 --n-trials 30 --max-epochs 15

    # Resume interrupted search (auto-resumes from where it left off)
    python -m src.optuna_search_ratio --search-all --output-dir ./outputs/optuna_ratio_YYYYMMDD
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
from optuna.pruners import MedianPruner
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
from .models_ratio import build_ratio_model, RatioMSELoss
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
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze backbone parameters."""
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = True


def get_optimizer(
    model: nn.Module,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    device_type: DeviceType,
    stage: int = 2,
) -> AdamW:
    """Create optimizer with differential LR."""
    use_fused = supports_fused_optimizer(device_type)
    
    if stage == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return AdamW(params, lr=lr, weight_decay=weight_decay, fused=use_fused)
    
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
        {"params": head_params, "lr": lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]
    
    return AdamW(param_groups, weight_decay=weight_decay, fused=use_fused)


def get_scheduler(optimizer: AdamW, epochs: int, warmup_epochs: int = 2) -> SequentialLR:
    """Create scheduler with warmup."""
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-7)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    
    for batch in loader:
        x_left, x_right, targets = batch[:3]
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            green, dead, clover, gdm, total = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
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
    
    return losses.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    use_log_target: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """Validate model."""
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []
    
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    
    for batch in loader:
        x_left, x_right, targets = batch[:3]
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            green, dead, clover, gdm, total = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            loss = loss_fn(preds, targets)
        
        losses.update(loss.item(), x_left.size(0))
        all_preds.append(preds.float().cpu())
        all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    if use_log_target:
        all_preds = np.expm1(all_preds)
        all_targets = np.expm1(all_targets)
    
    r2 = compute_weighted_r2(all_preds, all_targets, target_weights)
    
    target_names = ["green", "dead", "clover", "gdm", "total"]
    per_target = compute_per_target_metrics_np(all_preds, all_targets, target_names, target_weights)
    
    metrics = {"weighted_r2": r2}
    for row in per_target:
        metrics[f"r2_{row['target']}"] = float(row["r2"])
    
    return losses.avg, r2, metrics


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
) -> Tuple[float, float, Dict[str, float]]:
    """Train a single fold with given hyperparameters."""
    
    # Extract parameters
    model_type = params["model_type"]
    dropout = params["dropout"]
    hidden_ratio = params["hidden_ratio"]
    use_film = params["use_film"]
    use_attention_pool = params["use_attention_pool"]
    grid_size = params["grid_size"]
    
    lr = params["lr"]
    backbone_lr = params["backbone_lr"]
    head_lr_stage1 = params["head_lr_stage1"]
    weight_decay = params["weight_decay"]
    warmup_epochs = params["warmup_epochs"]
    grad_clip = params["grad_clip"]
    batch_size = params["batch_size"]
    
    training_mode = params["training_mode"]
    freeze_epochs = params.get("freeze_epochs", 5)
    
    use_huber_for_dead = params["use_huber_for_dead"]
    huber_delta = params["huber_delta"]
    ratio_loss_weight = params["ratio_loss_weight"]
    ratio_loss_type = params["ratio_loss_type"]
    ratio_temperature = params.get("ratio_temperature", 1.0)
    
    use_log_target = params["use_log_target"]
    stereo_correct_aug = params["stereo_correct_aug"]
    stereo_swap_prob = params["stereo_swap_prob"]
    mixup_prob = params["mixup_prob"]
    mixup_alpha = params.get("mixup_alpha", 0.4)
    cutmix_prob = params.get("cutmix_prob", 0.0)
    cutmix_alpha = params.get("cutmix_alpha", 1.0)

    img_size = params.get("img_size", 518)
    aug_prob = params.get("aug_prob", 0.5)
    grid = (grid_size, grid_size)
    
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
        is_train=True, use_log_target=use_log_target,
        stereo_swap_prob=stereo_swap_prob,
        photometric_transform=photometric_transform,
        mixup_prob=mixup_prob,
        mixup_alpha=mixup_alpha,
        cutmix_prob=cutmix_prob,
        cutmix_alpha=cutmix_alpha,
    )
    valid_ds = BiomassDataset(
        valid_df, image_dir, valid_transform,
        is_train=False, use_log_target=use_log_target,
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
    model = build_ratio_model(
        backbone_name=backbone,
        grid=grid,
        pretrained=True,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
        use_film=use_film,
        use_attention_pool=use_attention_pool,
        model_type=model_type,
        ratio_temperature=ratio_temperature,
    ).to(device)
    
    # Loss function
    loss_fn = RatioMSELoss(
        target_weights=[0.1, 0.1, 0.1, 0.2, 0.5],
        use_huber_for_dead=use_huber_for_dead,
        huber_delta=huber_delta,
        ratio_loss_weight=ratio_loss_weight,
        ratio_loss_type=ratio_loss_type,
    )
    
    # GradScaler
    scaler = GradScaler("cuda") if device_type == DeviceType.CUDA else None
    
    # Training mode setup
    current_stage = 1 if training_mode in ["freeze", "two_stage"] else 2
    
    if training_mode == "freeze":
        freeze_backbone(model)
        optimizer = get_optimizer(model, head_lr_stage1, backbone_lr, weight_decay, device_type, stage=1)
        scheduler = get_scheduler(optimizer, max_epochs, warmup_epochs)
    elif training_mode == "two_stage":
        freeze_backbone(model)
        optimizer = get_optimizer(model, head_lr_stage1, backbone_lr, weight_decay, device_type, stage=1)
        scheduler = get_scheduler(optimizer, freeze_epochs, warmup_epochs=1)
    else:
        optimizer = get_optimizer(model, lr, backbone_lr, weight_decay, device_type, stage=2)
        scheduler = get_scheduler(optimizer, max_epochs, warmup_epochs)
    
    # Training loop
    best_r2 = -float("inf")
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    
    epoch_pbar = tqdm(range(1, max_epochs + 1), desc=f"F{fold}", leave=False, ncols=120)
    for epoch in epoch_pbar:
        # Stage transition for two-stage
        if training_mode == "two_stage" and epoch == freeze_epochs + 1:
            unfreeze_backbone(model)
            optimizer = get_optimizer(model, lr, backbone_lr, weight_decay, device_type, stage=2)
            remaining = max_epochs - freeze_epochs
            scheduler = get_scheduler(optimizer, remaining, warmup_epochs)
            current_stage = 2
            patience_counter = 0
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, device_type,
            scaler=scaler, grad_clip=grad_clip,
        )
        
        valid_loss, r2, metrics = validate(
            model, valid_loader, loss_fn, device, device_type,
            use_log_target=use_log_target,
        )
        
        scheduler.step()
        
        # Update progress bar with metrics
        stage_str = "S1" if current_stage == 1 else "S2"
        epoch_pbar.set_postfix_str(
            f"{stage_str} loss={valid_loss:.4f} R²={r2:.4f} best={best_r2:.4f}"
        )
        
        # Report to Optuna for pruning
        if trial is not None:
            trial.report(r2, report_epoch_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Track best
        save_ok = (current_stage == 2) or (training_mode == "freeze")
        
        if save_ok and r2 > best_r2 and not np.isnan(r2):
            best_r2 = r2
            best_metrics = metrics.copy()
            patience_counter = 0
        elif save_ok:
            patience_counter += 1
        
        # Early stopping
        if np.isnan(train_loss) or np.isnan(valid_loss):
            break
        
        if patience_counter >= patience:
            break
    
    # Cleanup
    del model, optimizer, scheduler
    if scaler is not None:
        del scaler
    empty_cache(device_type)
    gc.collect()
    
    return best_r2, valid_loss, best_metrics


def create_objective(
    df: pd.DataFrame,
    backbone: str,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    max_epochs: int,
    patience: int,
    n_folds_eval: int,
    seed: int,
    search_phase: int = 0,
):
    """Create Optuna objective function.

    Search Phases:
    - Phase 0 (BASIC): model_type, training_mode, head_lr_stage1, batch_size
    - Phase 1 (CORE):  + dropout, hidden_ratio, grad_clip, lr/backbone_lr
    - Phase 2 (ARCH):  + use_film, use_attention_pool, grid_size, weight_decay, warmup_epochs
    - Phase 3 (REG):   + huber, ratio_loss, mixup, cutmix, stereo augmentation
    """

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed + trial.number, device_type)

        # =====================================================================
        # PHASE 0: BASIC - Essential parameters (always searched)
        # These have the highest impact on performance
        # =====================================================================
        model_type = trial.suggest_categorical("model_type", ["softmax", "hierarchical", "direct"])
        training_mode = "freeze"  # Fixed: train heads only, backbone frozen
        head_lr_stage1 = trial.suggest_float("head_lr_stage1", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

        # Conditional: ratio_temperature only for softmax model
        if model_type == "softmax":
            ratio_temperature = trial.suggest_float("ratio_temperature", 0.5, 2.0)
        else:
            ratio_temperature = 1.0

        # =====================================================================
        # PHASE 1: CORE - Architecture and training parameters
        # =====================================================================
        if search_phase >= 1:
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            hidden_ratio = trial.suggest_float("hidden_ratio", 0.25, 1.0)
            grad_clip = trial.suggest_float("grad_clip", 0.3, 2.0)
        else:
            dropout = 0.2
            hidden_ratio = 0.5
            grad_clip = 0.5

        # Two-stage specific parameters
        if training_mode == "two_stage":
            if search_phase >= 1:
                freeze_epochs = trial.suggest_int("freeze_epochs", 3, 8)
                lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
                backbone_lr = trial.suggest_float("backbone_lr", 1e-6, 5e-5, log=True)
            else:
                freeze_epochs = 5
                lr = 2e-4
                backbone_lr = 1e-5
        else:
            freeze_epochs = max_epochs
            lr = head_lr_stage1
            backbone_lr = 1e-5

        # =====================================================================
        # PHASE 2: ARCH - Advanced architecture parameters
        # =====================================================================
        if search_phase >= 2:
            use_film = trial.suggest_categorical("use_film", [True, False])
            use_attention_pool = trial.suggest_categorical("use_attention_pool", [True, False])
            grid_size = trial.suggest_categorical("grid_size", [1, 2, 3])
            weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
            warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)
        else:
            use_film = True
            use_attention_pool = True
            grid_size = 2
            weight_decay = 0.01
            warmup_epochs = 2

        # =====================================================================
        # PHASE 3: REG - Regularization and augmentation
        # =====================================================================
        if search_phase >= 3:
            # Loss configuration
            use_huber_for_dead = trial.suggest_categorical("use_huber_for_dead", [True, False])
            huber_delta = trial.suggest_float("huber_delta", 1.0, 10.0) if use_huber_for_dead else 5.0
            ratio_loss_weight = trial.suggest_float("ratio_loss_weight", 0.0, 0.3)
            ratio_loss_type = trial.suggest_categorical("ratio_loss_type", ["mse", "kl"]) if ratio_loss_weight > 0 else "mse"

            # Data augmentation
            mixup_prob = trial.suggest_float("mixup_prob", 0.0, 0.4)
            mixup_alpha = trial.suggest_float("mixup_alpha", 0.2, 1.0) if mixup_prob > 0 else 0.4
            cutmix_prob = trial.suggest_float("cutmix_prob", 0.0, 0.4)
            cutmix_alpha = trial.suggest_float("cutmix_alpha", 0.5, 1.5) if cutmix_prob > 0 else 1.0

            # Stereo augmentation
            stereo_swap_prob = trial.suggest_float("stereo_swap_prob", 0.0, 0.5)
            stereo_correct_aug = trial.suggest_categorical("stereo_correct_aug", [True, False])

            # Target transformation
            use_log_target = trial.suggest_categorical("use_log_target", [True, False])
        else:
            use_huber_for_dead = True
            huber_delta = 5.0
            ratio_loss_weight = 0.1  # Enable ratio loss
            ratio_loss_type = "kl"   # Use KL divergence
            mixup_prob = 0.0
            mixup_alpha = 0.4
            cutmix_prob = 0.0
            cutmix_alpha = 1.0
            stereo_swap_prob = 0.0
            stereo_correct_aug = False
            use_log_target = False

        params = {
            "model_type": model_type,
            "dropout": dropout,
            "hidden_ratio": hidden_ratio,
            "use_film": use_film,
            "use_attention_pool": use_attention_pool,
            "grid_size": grid_size,
            "lr": lr,
            "backbone_lr": backbone_lr,
            "head_lr_stage1": head_lr_stage1,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "grad_clip": grad_clip,
            "batch_size": batch_size,
            "training_mode": training_mode,
            "freeze_epochs": freeze_epochs,
            "use_huber_for_dead": use_huber_for_dead,
            "huber_delta": huber_delta,
            "ratio_loss_weight": ratio_loss_weight,
            "ratio_loss_type": ratio_loss_type,
            "ratio_temperature": ratio_temperature,
            "use_log_target": use_log_target,
            "stereo_correct_aug": stereo_correct_aug,
            "stereo_swap_prob": stereo_swap_prob,
            "mixup_prob": mixup_prob,
            "mixup_alpha": mixup_alpha,
            "cutmix_prob": cutmix_prob,
            "cutmix_alpha": cutmix_alpha,
        }
        
        # Print trial info with all configs
        print(f"\n{'='*70}")
        print(f"[Trial {trial.number}] Phase {search_phase}")
        print(f"{'='*70}")
        print(f"  Model:    {model_type} | film={use_film} | attn_pool={use_attention_pool} | grid={grid_size}")
        print(f"  Training: {training_mode} | freeze_ep={freeze_epochs} | batch={batch_size}")
        print(f"  LR:       head={head_lr_stage1:.2e} | backbone={backbone_lr:.2e} | stage2={lr:.2e}")
        print(f"  Arch:     dropout={dropout:.3f} | hidden_ratio={hidden_ratio:.3f} | grad_clip={grad_clip:.2f}")
        print(f"  Sched:    warmup={warmup_epochs} | weight_decay={weight_decay:.4f}")
        if model_type == "softmax":
            print(f"  Softmax:  temperature={ratio_temperature:.2f}")
        if search_phase >= 3:
            print(f"  Loss:     huber_dead={use_huber_for_dead} | huber_δ={huber_delta:.1f} | ratio_loss={ratio_loss_weight:.3f} ({ratio_loss_type})")
            print(f"  Aug:      mixup={mixup_prob:.2f} | cutmix={cutmix_prob:.2f} | stereo_swap={stereo_swap_prob:.2f} | stereo_correct={stereo_correct_aug}")
            print(f"  Target:   log={use_log_target}")
        print(f"{'-'*70}")
        
        # Cross-validation
        fold_scores = []
        folds_to_eval = list(range(n_folds_eval))
        
        for i, fold in enumerate(folds_to_eval):
            train_df = df[df["fold"] != fold].reset_index(drop=True)
            valid_df = df[df["fold"] == fold].reset_index(drop=True)
            
            try:
                r2, _, metrics = train_fold_with_params(
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
                    trial=trial,
                    report_epoch_offset=i * max_epochs,
                )
                fold_scores.append(r2)
                
                # Log fold result with per-target breakdown
                per_target = " | ".join([
                    f"G={metrics.get('r2_green', 0):.3f}",
                    f"D={metrics.get('r2_dead', 0):.3f}",
                    f"C={metrics.get('r2_clover', 0):.3f}",
                    f"GDM={metrics.get('r2_gdm', 0):.3f}",
                    f"T={metrics.get('r2_total', 0):.3f}",
                ])
                print(f"  Fold {fold}: R²={r2:.4f} [{per_target}]")
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"  Fold {fold} failed: {e}")
                return -1.0
        
        cv_score = np.mean(fold_scores)
        print(f"  CV Score: {cv_score:.4f} ± {np.std(fold_scores):.4f}")
        
        return cv_score
    
    return objective


def run_single_phase(
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    device_type: DeviceType,
    image_dir: str,
    search_phase: int,
    n_trials: int,
    study_name: str,
    phase_descriptions: Dict[int, str],
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """Run optimization for a single phase."""
    print("\n" + "=" * 60)
    print(f"PHASE {search_phase}: {phase_descriptions[search_phase]}")
    print("=" * 60)

    sampler = TPESampler(seed=args.seed, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # Use phase-specific study name for persistence
    phase_study_name = f"{study_name}_phase{search_phase}"

    if args.storage:
        study = optuna.create_study(
            study_name=phase_study_name,
            storage=args.storage,
            load_if_exists=True,  # Always allow resuming
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
    else:
        study = optuna.create_study(
            study_name=phase_study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

    # Check how many trials already completed
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = max(0, n_trials - completed_trials)

    if remaining_trials == 0:
        print(f"Phase {search_phase} already completed ({completed_trials} trials). Skipping...")
    else:
        if completed_trials > 0:
            print(f"Resuming phase {search_phase}: {completed_trials} trials done, {remaining_trials} remaining")

        objective = create_objective(
            df=df,
            backbone=args.backbone,
            image_dir=image_dir,
            device=device,
            device_type=device_type,
            max_epochs=args.max_epochs,
            patience=args.patience,
            n_folds_eval=args.n_folds_eval,
            seed=args.seed,
            search_phase=search_phase,
        )

        print(f"Running {remaining_trials} trials...")
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True)

    # Results for this phase
    print(f"\nPhase {search_phase} Results:")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best CV R²: {study.best_value:.4f}")
    print("  Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return study, study.best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna Search for Ratio Models")

    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_reg4_dinov2.lvd142m",
                        help="Vision backbone: DINOv2 (vit_base_patch14_reg4_dinov2.lvd142m) or "
                             "DINOv3 (vit_base_patch16_dinov3, vit_large_patch16_dinov3)")
    parser.add_argument("--fold-csv", type=str, default="./data/trainfold.csv",
                        help="Path to CSV with predefined folds (default: ./data/trainfold.csv)")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n-folds-eval", type=int, default=3,
                        help="Number of folds to evaluate per trial (1-5)")
    parser.add_argument("--search-phase", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Search phase: 0=basic (START HERE), 1=+core, 2=+architecture, 3=+regularization")
    parser.add_argument("--search-all", action="store_true",
                        help="Run all phases sequentially (0->1->2->3) with persistence")
    parser.add_argument("--trials-per-phase", type=int, nargs=4, default=[30, 40, 40, 50],
                        help="Number of trials for each phase when using --search-all (default: 30 40 40 50)")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    # Setup
    device_type = DeviceType(args.device_type) if args.device_type else get_device_type()
    device = get_device(device_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"./outputs/optuna_ratio_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    phase_descriptions = {
        0: "BASIC (model_type, training_mode, lr, batch_size)",
        1: "CORE (+dropout, hidden_ratio, grad_clip, freeze_epochs)",
        2: "ARCH (+FiLM, attention_pool, grid, weight_decay)",
        3: "REG (+huber, ratio_loss, mixup, cutmix, stereo_aug)",
    }

    # Auto-enable storage for --search-all
    if args.search_all and not args.storage:
        args.storage = f"sqlite:///{args.output_dir}/optuna.db"
        print(f"Auto-enabled persistence: {args.storage}")

    print("=" * 60)
    print("Optuna Hyperparameter Search - Ratio Models")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    if args.search_all:
        print(f"Mode: SEARCH ALL PHASES (trials per phase: {args.trials_per_phase})")
        print(f"Total trials: {sum(args.trials_per_phase)}")
    else:
        print(f"Trials: {args.n_trials}")
        print(f"Search phase: {args.search_phase} - {phase_descriptions[args.search_phase]}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Folds per trial: {args.n_folds_eval}")
    print(f"Output: {args.output_dir}")
    if args.storage:
        print(f"Storage: {args.storage}")
    print("-" * 60)
    print("Phase progression:")
    for p, desc in phase_descriptions.items():
        if args.search_all:
            marker = f"[{args.trials_per_phase[p]:3d}]"
        else:
            marker = ">>>" if p == args.search_phase else "   "
        print(f"  {marker} Phase {p}: {desc}")
    print("=" * 60)

    # Load data
    train_csv = os.path.join(args.base_path, "train.csv")
    image_dir = os.path.join(args.base_path, "train")

    df = prepare_dataframe(train_csv)

    # Load predefined folds
    if args.fold_csv and os.path.exists(args.fold_csv):
        print(f"Loading predefined folds from: {args.fold_csv}")
        fold_df = pd.read_csv(args.fold_csv)
        fold_mapping = fold_df.set_index("sample_id_prefix")["fold"].to_dict()
        df["fold"] = df["sample_id_prefix"].map(fold_mapping).fillna(0).astype(int)
    else:
        print("Creating folds using group_date strategy (no predefined folds found)")
        df = create_folds(df, n_folds=5, seed=args.seed, cv_strategy="group_date")

    print(f"Samples: {len(df)}")
    print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")

    # Study name
    study_name = args.study_name or f"ratio_search_{timestamp}"

    # Run search
    if args.search_all:
        # Run all phases sequentially
        all_results = {}
        best_params = {}

        for phase in range(4):
            n_trials = args.trials_per_phase[phase]
            study, phase_best_params = run_single_phase(
                df=df,
                args=args,
                device=device,
                device_type=device_type,
                image_dir=image_dir,
                search_phase=phase,
                n_trials=n_trials,
                study_name=study_name,
                phase_descriptions=phase_descriptions,
            )

            all_results[f"phase_{phase}"] = {
                "best_value": study.best_value,
                "best_params": phase_best_params,
                "n_trials": len(study.trials),
            }
            best_params.update(phase_best_params)

            # Memory cleanup between phases
            gc.collect()
            empty_cache(device_type)

        # Final summary
        print("\n" + "=" * 60)
        print("ALL PHASES COMPLETE!")
        print("=" * 60)

        for phase in range(4):
            r = all_results[f"phase_{phase}"]
            print(f"Phase {phase}: Best R² = {r['best_value']:.4f} ({r['n_trials']} trials)")

        # Use best params from phase 3 (most comprehensive)
        final_study = study
        final_best_params = best_params

    else:
        # Single phase mode
        sampler = TPESampler(seed=args.seed, multivariate=True)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

        if args.storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=args.storage,
                load_if_exists=args.resume,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
            )
        else:
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
            )

        objective = create_objective(
            df=df,
            backbone=args.backbone,
            image_dir=image_dir,
            device=device,
            device_type=device_type,
            max_epochs=args.max_epochs,
            patience=args.patience,
            n_folds_eval=args.n_folds_eval,
            seed=args.seed,
            search_phase=args.search_phase,
        )

        print(f"\nStarting optimization with {args.n_trials} trials...")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        final_study = study
        final_best_params = study.best_params

        # Results
        print("\n" + "=" * 60)
        print("Search Complete!")
        print("=" * 60)

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best CV R²: {study.best_value:.4f}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    
    # Save results
    if args.search_all:
        results = {
            "study_name": study_name,
            "search_mode": "all_phases",
            "best_params": final_best_params,
            "phase_results": all_results,
            "config": {
                "backbone": args.backbone,
                "max_epochs": args.max_epochs,
                "n_folds_eval": args.n_folds_eval,
                "trials_per_phase": args.trials_per_phase,
            },
        }
    else:
        results = {
            "study_name": study_name,
            "search_mode": f"phase_{args.search_phase}",
            "best_value": final_study.best_value,
            "best_params": final_best_params,
            "best_trial_number": final_study.best_trial.number,
            "n_trials": len(final_study.trials),
            "search_phase": args.search_phase,
            "config": {
                "backbone": args.backbone,
                "max_epochs": args.max_epochs,
                "n_folds_eval": args.n_folds_eval,
            },
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in final_study.trials
            ],
        }

    results_path = os.path.join(args.output_dir, "search_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate training command
    print("\n" + "=" * 60)
    print("Recommended training command:")
    print("=" * 60)

    bp = final_best_params
    cmd_parts = [
        f"python -m src.train_ratio",
        f"    --base-path {args.base_path}",
        f"    --model-type {bp.get('model_type', 'hierarchical')}",
        f"    --grid {bp.get('grid_size', 2)} {bp.get('grid_size', 2)}",
        f"    --dropout {bp.get('dropout', 0.2):.3f}",
        f"    --hidden-ratio {bp.get('hidden_ratio', 0.5):.3f}",
    ]

    # Architecture flags
    if not bp.get('use_film', True):
        cmd_parts.append("    --no-film")
    if not bp.get('use_attention_pool', True):
        cmd_parts.append("    --no-attention-pool")

    # Training mode
    if bp.get('training_mode') == 'freeze':
        cmd_parts.append("    --freeze-backbone")
    else:
        cmd_parts.append("    --two-stage")
        cmd_parts.append(f"    --freeze-epochs {bp.get('freeze_epochs', 5)}")

    # Learning rates
    cmd_parts.extend([
        f"    --head-lr-stage1 {bp.get('head_lr_stage1', 1e-4):.2e}",
        f"    --lr {bp.get('lr', 2e-4):.2e}",
        f"    --backbone-lr {bp.get('backbone_lr', 1e-5):.2e}",
        f"    --weight-decay {bp.get('weight_decay', 0.01):.4f}",
        f"    --warmup-epochs {bp.get('warmup_epochs', 2)}",
        f"    --grad-clip {bp.get('grad_clip', 0.5):.2f}",
        f"    --batch-size {bp.get('batch_size', 8)}",
    ])

    # Softmax temperature
    if bp.get('model_type') == 'softmax' and 'ratio_temperature' in bp:
        cmd_parts.append(f"    --ratio-temperature {bp['ratio_temperature']:.2f}")

    # Loss configuration (Phase 3)
    if not bp.get('use_huber_for_dead', True):
        cmd_parts.append("    --no-huber-for-dead")
    if bp.get('huber_delta', 5.0) != 5.0:
        cmd_parts.append(f"    --huber-delta {bp['huber_delta']:.1f}")
    if bp.get('ratio_loss_weight', 0.0) > 0:
        cmd_parts.append(f"    --ratio-loss-weight {bp['ratio_loss_weight']:.3f}")
        cmd_parts.append(f"    --ratio-loss-type {bp.get('ratio_loss_type', 'mse')}")

    # Augmentation (Phase 3)
    if bp.get('mixup_prob', 0.0) > 0:
        cmd_parts.append(f"    --mixup-prob {bp['mixup_prob']:.2f}")
        cmd_parts.append(f"    --mixup-alpha {bp.get('mixup_alpha', 0.4):.2f}")
    if bp.get('cutmix_prob', 0.0) > 0:
        cmd_parts.append(f"    --cutmix-prob {bp['cutmix_prob']:.2f}")
        cmd_parts.append(f"    --cutmix-alpha {bp.get('cutmix_alpha', 1.0):.2f}")
    if bp.get('stereo_swap_prob', 0.0) > 0:
        cmd_parts.append(f"    --stereo-swap-prob {bp['stereo_swap_prob']:.2f}")
    if bp.get('stereo_correct_aug', False):
        cmd_parts.append("    --stereo-correct-aug")
    if bp.get('use_log_target', False):
        cmd_parts.append("    --use-log-target")

    cmd_parts.extend([
        "    --epochs 50",
        "    --compute-oof",
        f"    --device-type {args.device_type or 'auto'}",
    ])

    cmd = " \\\n".join(cmd_parts)
    print(cmd)


if __name__ == "__main__":
    main()
