#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for CSIRO Biomass Prediction.

Searches best hyperparameters across 5-fold CV for a given backbone.

Usage:
    # Single GPU or MPS
    python -m src.optuna_search --backbone vit_base_patch14_reg4_dinov2.lvd142m --n-trials 50

    # Multi-GPU (each trial uses all GPUs) - CUDA only
    python -m src.optuna_search --backbone vit_base_patch14_reg4_dinov2.lvd142m --n-trials 50 --num-gpus 2

    # Apple Silicon (MPS)
    python -m src.optuna_search --backbone vit_base_patch14_reg4_dinov2.lvd142m --device-type mps --n-trials 50
"""
import argparse
import gc
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .dataset import BiomassDataset, create_folds, get_train_transforms, get_valid_transforms, prepare_dataframe
from .device import DeviceType, empty_cache, get_amp_settings, get_device, get_device_type
from .models import build_model
from .trainer import (
    AverageMeter,
    WeightedMSELoss,
    compute_per_target_metrics_np,
    compute_weighted_r2,
    get_optimizer,
)


# Global config for distributed setup
_DISTRIBUTED = False
_WORLD_SIZE = 1
_LOCAL_RANK = 0
_DEVICE = None
_DEVICE_TYPE = DeviceType.CPU


def setup_device(num_gpus: int = 1, device_type: Optional[str] = None) -> Tuple[torch.device, DeviceType]:
    """Setup device for training."""
    global _DISTRIBUTED, _WORLD_SIZE, _LOCAL_RANK, _DEVICE, _DEVICE_TYPE
    
    # Determine device type
    if device_type is not None:
        _DEVICE_TYPE = DeviceType(device_type)
    else:
        _DEVICE_TYPE = get_device_type()
    
    # Setup based on device type
    if _DEVICE_TYPE == DeviceType.CUDA:
        _DEVICE = torch.device("cuda:0")
        _WORLD_SIZE = min(num_gpus, torch.cuda.device_count())
        _DISTRIBUTED = _WORLD_SIZE > 1
    elif _DEVICE_TYPE == DeviceType.MPS:
        _DEVICE = torch.device("mps")
        _WORLD_SIZE = 1
        _DISTRIBUTED = False
    else:
        _DEVICE = torch.device("cpu")
        _WORLD_SIZE = 1
        _DISTRIBUTED = False
    
    return _DEVICE, _DEVICE_TYPE


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
    device_type: DeviceType = DeviceType.CUDA,
) -> float:
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    
    # Get AMP settings for device
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    
    for x_left, x_right, targets in loader:
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            green, dead, clover, gdm, total = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            loss = loss_fn(preds, targets)
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), x_left.size(0))
    
    return losses.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    target_weights: List[float],
    device_type: DeviceType = DeviceType.CUDA,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """Validate and return loss, weighted R², and per-target metrics."""
    model.eval()
    losses = AverageMeter()
    all_preds, all_targets = [], []
    
    # Get AMP settings for device
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    
    for x_left, x_right, targets in loader:
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            green, dead, clover, gdm, total = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            loss = loss_fn(preds, targets)
        
        losses.update(loss.item(), x_left.size(0))
        all_preds.append(preds.float().cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    r2 = compute_weighted_r2(all_preds, all_targets, target_weights)
    per_target = compute_per_target_metrics_np(
        all_preds,
        all_targets,
        target_names=["green", "dead", "clover", "gdm", "total"],
        target_weights=target_weights,
    )
    return losses.avg, r2, per_target


def train_fold_with_params(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: Dict[str, Any],
    backbone: str,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType = DeviceType.CUDA,
    max_epochs: int = 20,
    early_stop_patience: int = 5,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[float, float]:
    """
    Train a single fold with given hyperparameters.
    
    Returns:
        best_loss, best_r2
    """
    # Extract params
    lr = params["lr"]
    backbone_lr = params["backbone_lr"]
    dropout = params["dropout"]
    hidden_ratio = params["hidden_ratio"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    img_size = params.get("img_size", 518)
    grid = params.get("grid", (2, 2))
    
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    
    # Create datasets
    train_transform = get_train_transforms(img_size, aug_prob=0.5)
    valid_transform = get_valid_transforms(img_size)
    
    train_ds = BiomassDataset(train_df, image_dir, train_transform, is_train=True)
    valid_ds = BiomassDataset(valid_df, image_dir, valid_transform, is_train=False)
    
    # pin_memory only beneficial on CUDA
    pin_memory = device_type == DeviceType.CUDA
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=pin_memory, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=pin_memory,
    )
    
    # Build model
    model = build_model(
        backbone_name=backbone,
        model_type="tiled_film",
        grid=grid,
        pretrained=True,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model, lr, backbone_lr, weight_decay, device_type=device_type)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-7)
    loss_fn = WeightedMSELoss(target_weights)
    
    # Training loop
    best_loss = float("inf")
    best_r2 = 0.0
    patience_counter = 0
    best_per_target: Optional[List[Dict[str, float]]] = None
    
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, device_type)
        valid_loss, valid_r2, per_target = validate(model, valid_loader, loss_fn, device, target_weights, device_type)
        scheduler.step()
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_r2 = valid_r2
            best_per_target = per_target
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Optuna pruning
        if trial is not None:
            trial.report(valid_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            # Store fold0 best per-target snapshot for debugging
            if best_per_target is not None:
                trial.set_user_attr("fold0_best_per_target", best_per_target)
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            break
    
    # Cleanup
    del model, optimizer, scheduler
    empty_cache(device_type)
    gc.collect()
    
    return best_loss, best_r2


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    backbone: str,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType = DeviceType.CUDA,
    max_epochs: int = 20,
    num_folds: int = 5,
) -> float:
    """Optuna objective function - minimize CV loss."""
    
    # Define hyperparameter search space
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "backbone_lr": trial.suggest_float("backbone_lr", 1e-6, 5e-5, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "hidden_ratio": trial.suggest_float("hidden_ratio", 0.1, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 12, 16]),
        "img_size": 518,
        "grid": (2, 2),
    }
    
    print(f"\n{'='*50}")
    print(f"Trial {trial.number}: {params}")
    print(f"{'='*50}")
    
    # Cross-validation
    fold_losses = []
    fold_r2s = []
    
    for fold in range(num_folds):
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        
        print(f"  Fold {fold}/{num_folds}...", end=" ", flush=True)
        
        try:
            loss, r2 = train_fold_with_params(
                fold=fold,
                train_df=train_df,
                valid_df=valid_df,
                params=params,
                backbone=backbone,
                image_dir=image_dir,
                device=device,
                device_type=device_type,
                max_epochs=max_epochs,
                early_stop_patience=5,
                trial=trial if fold == 0 else None,  # Prune based on fold 0
            )
            fold_losses.append(loss)
            fold_r2s.append(r2)
            print(f"Loss: {loss:.4f}, R²: {r2:.4f}")
        except optuna.TrialPruned:
            print("PRUNED")
            raise
    
    cv_loss = np.mean(fold_losses)
    cv_r2 = np.mean(fold_r2s)
    
    print(f"\n  CV Loss: {cv_loss:.4f} ± {np.std(fold_losses):.4f}")
    print(f"  CV R²: {cv_r2:.4f} ± {np.std(fold_r2s):.4f}")
    
    # Store R² for reference
    trial.set_user_attr("cv_r2", cv_r2)
    trial.set_user_attr("cv_r2_std", np.std(fold_r2s))
    trial.set_user_attr("cv_loss_std", np.std(fold_losses))
    # Per-target breakdown (available for fold0 via fold0_best_per_target attr)
    
    return cv_loss  # Minimize loss


def run_optuna_search(
    backbone: str,
    base_path: str = "/workspace",
    output_dir: Optional[str] = None,
    n_trials: int = 50,
    max_epochs: int = 20,
    num_folds: int = 5,
    num_gpus: int = 1,
    seed: int = 18,
    study_name: Optional[str] = None,
    cv_strategy: str = "stratified",
    device_type_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search for a backbone.
    
    Args:
        backbone: Backbone model name
        base_path: Base data path
        output_dir: Output directory for results
        n_trials: Number of Optuna trials
        max_epochs: Max epochs per fold (reduced for faster search)
        num_folds: Number of CV folds
        num_gpus: Number of GPUs (CUDA only)
        seed: Random seed
        study_name: Optuna study name
        cv_strategy: CV strategy ("stratified", "group_date", "random")
        device_type_str: Device type ("cuda", "mps", "cpu") or None for auto
    
    Returns:
        Best hyperparameters dict
    """
    # Setup
    device, device_type = setup_device(num_gpus, device_type_str)
    train_csv = os.path.join(base_path, "train.csv")
    image_dir = os.path.join(base_path, "train")
    
    if output_dir is None:
        timestamp = int(time.time())
        output_dir = os.path.join(base_path, "outputs", f"optuna_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    if study_name is None:
        backbone_short = backbone.replace(".", "_").replace("-", "_")
        study_name = f"csiro_{backbone_short}"
    
    print(f"{'='*60}")
    print(f"Optuna Hyperparameter Search")
    print(f"{'='*60}")
    print(f"Backbone: {backbone}")
    print(f"Device type: {device_type.value}")
    print(f"Device: {device}")
    print(f"Trials: {n_trials}")
    print(f"Max epochs/fold: {max_epochs}")
    print(f"Folds: {num_folds}")
    print(f"CV strategy: {cv_strategy}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Load and prepare data
    print("\nPreparing data...")
    df = prepare_dataframe(train_csv)
    df = create_folds(df, n_folds=num_folds, seed=seed, cv_strategy=cv_strategy)
    print(f"Total samples: {len(df)}")
    
    # Create Optuna study
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # Minimize loss
        sampler=sampler,
        pruner=pruner,
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, df, backbone, image_dir, device, device_type, max_epochs, num_folds
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Results
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")
    
    best_trial = study.best_trial
    print(f"\nBest Trial: {best_trial.number}")
    print(f"Best CV Loss: {best_trial.value:.4f}")
    print(f"Best CV R²: {best_trial.user_attrs.get('cv_r2', 'N/A'):.4f}")
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
        "cv_strategy": cv_strategy,
        "best_trial": {
            "number": best_trial.number,
            "cv_loss": best_trial.value,
            "cv_loss_std": best_trial.user_attrs.get("cv_loss_std", 0),
            "cv_r2": best_trial.user_attrs.get("cv_r2", 0),
            "cv_r2_std": best_trial.user_attrs.get("cv_r2_std", 0),
            "params": best_trial.params,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value if t.value is not None else None,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    
    results_path = os.path.join(output_dir, f"optuna_results_{backbone.replace('.', '_')}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save best params for easy loading
    best_params_path = os.path.join(output_dir, f"best_params_{backbone.replace('.', '_')}.json")
    with open(best_params_path, "w") as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"Best params saved to: {best_params_path}")
    
    return best_trial.params


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search")
    parser.add_argument(
        "--backbone", type=str, required=True,
        help="Backbone model name (e.g., vit_base_patch14_reg4_dinov2.lvd142m)"
    )
    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--max-epochs", type=int, default=20, help="Max epochs per fold")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (CUDA only)")
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument(
        "--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help="Device type: cuda (NVIDIA GPU), mps (Apple Silicon), cpu. Auto-detected if not specified."
    )
    parser.add_argument(
        "--cv-strategy", type=str, default="group_month",
        choices=[
            "group_month",
            "group_date",
            "group_date_state",
            "group_date_state_bin",
            "stratified",
            "random",
        ],
        help="CV strategy: group_month, group_date, group_date_state (month-grouped, state-stratified), stratified, random"
    )
    
    args = parser.parse_args()
    
    run_optuna_search(
        backbone=args.backbone,
        base_path=args.base_path,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        num_folds=args.num_folds,
        num_gpus=args.num_gpus,
        seed=args.seed,
        study_name=args.study_name,
        cv_strategy=args.cv_strategy,
        device_type_str=args.device_type,
    )


if __name__ == "__main__":
    main()
