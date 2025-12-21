#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for DINOv3 Direct Model.

Staged search approach:
- Stage 1: Architecture (grid, film, attn_pool, dropout, hidden_ratio)
- Stage 2: Training dynamics (lr, weight_decay, warmup, grad_clip)
- Stage 3: Augmentation (aug_prob, mixup)

Usage:
    # Architecture search (default)
    python -m src.optuna_dinov3_direct --n-trials 10 --device-type mps

    # With clover head
    python -m src.optuna_dinov3_direct --train-clover --n-trials 10

    # With both dead and clover heads
    python -m src.optuna_dinov3_direct --train-dead --train-clover --n-trials 10

    # Training dynamics search (after finding best arch)
    python -m src.optuna_dinov3_direct --stage training \\
        --grid 3 --dropout 0.35 --hidden-ratio 0.3 --n-trials 10
"""
import argparse
import gc
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

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
from .dinov3_models import DINOv3Direct, BiomassLoss, freeze_backbone


IMG_SIZE = 576
TARGET_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]  # Competition weights
TARGET_NAMES = ["green", "dead", "clover", "gdm", "total"]


class AverageMeter:
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_device_seed(seed, device_type)


def compute_weighted_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute competition-weighted R²."""
    r2_scores = []
    for i in range(5):
        r2 = r2_score(targets[:, i], preds[:, i])
        r2_scores.append(r2)
    
    weighted = sum(w * r for w, r in zip(TARGET_WEIGHTS, r2_scores))
    return weighted / sum(TARGET_WEIGHTS)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    losses = AverageMeter()
    
    for batch in loader:
        x_left, x_right, targets = batch[:3]
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        green, dead, clover, gdm, total = model(x_left, x_right)
        preds = torch.cat([green, dead, clover, gdm, total], dim=1)
        loss = loss_fn(preds, targets)
        
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
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []
    
    for batch in loader:
        x_left, x_right, targets = batch[:3]
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        green, dead, clover, gdm, total = model(x_left, x_right)
        preds = torch.cat([green, dead, clover, gdm, total], dim=1)
        loss = loss_fn(preds, targets)
        
        losses.update(loss.item(), x_left.size(0))
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    r2 = compute_weighted_r2(all_preds, all_targets)
    
    metrics = {"weighted_r2": r2}
    for i, name in enumerate(TARGET_NAMES):
        metrics[f"r2_{name}"] = float(r2_score(all_targets[:, i], all_preds[:, i]))
    
    return losses.avg, r2, metrics


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: Dict[str, Any],
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    train_dead: bool,
    train_clover: bool,
    max_epochs: int = 25,
    patience: int = 6,
) -> Tuple[float, Dict[str, float]]:
    """Train single fold with given params."""
    
    # Architecture params
    dropout = params["dropout"]
    hidden_ratio = params["hidden_ratio"]
    use_film = params["use_film"]
    use_attention_pool = params["use_attention_pool"]
    grid_size = params["grid_size"]
    
    # Training params
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    warmup_epochs = params["warmup_epochs"]
    grad_clip = params["grad_clip"]
    batch_size = params["batch_size"]
    aug_prob = params["aug_prob"]
    
    # Augmentation params
    stereo_independent = params.get("stereo_independent", False)
    stereo_swap_prob = params.get("stereo_swap_prob", 0.0)
    mixup_prob = params.get("mixup_prob", 0.0)
    mixup_alpha = params.get("mixup_alpha", 0.4)
    cutmix_prob = params.get("cutmix_prob", 0.0)
    cutmix_alpha = params.get("cutmix_alpha", 1.0)
    
    # Loss params
    use_huber = params.get("use_huber", True)
    huber_delta = params.get("huber_delta", 5.0)
    
    grid = (grid_size, grid_size)
    
    # Photometric mode
    photometric_mode = params.get("photometric_mode", "same")
    
    if photometric_mode == "same":
        train_transform = get_train_transforms(IMG_SIZE, aug_prob)
        photometric_transform = None
        photometric_left_only = False
        photometric_right_only = False
    elif photometric_mode == "independent":
        train_transform = get_stereo_geometric_transforms(IMG_SIZE, aug_prob)
        photometric_transform = get_stereo_photometric_transforms(aug_prob)
        photometric_left_only = False
        photometric_right_only = False
    elif photometric_mode == "left":
        train_transform = get_stereo_geometric_transforms(IMG_SIZE, aug_prob)
        photometric_transform = get_stereo_photometric_transforms(aug_prob)
        photometric_left_only = True
        photometric_right_only = False
    elif photometric_mode == "right":
        train_transform = get_stereo_geometric_transforms(IMG_SIZE, aug_prob)
        photometric_transform = get_stereo_photometric_transforms(aug_prob)
        photometric_left_only = False
        photometric_right_only = True
    else:  # none
        train_transform = get_stereo_geometric_transforms(IMG_SIZE, aug_prob)
        photometric_transform = None
        photometric_left_only = False
        photometric_right_only = False
    
    valid_transform = get_valid_transforms(IMG_SIZE)
    
    # Datasets
    train_ds = BiomassDataset(
        train_df, image_dir, train_transform,
        is_train=True,
        stereo_swap_prob=stereo_swap_prob,
        photometric_transform=photometric_transform,
        photometric_left_only=photometric_left_only,
        photometric_right_only=photometric_right_only,
        mixup_prob=mixup_prob,
        mixup_alpha=mixup_alpha,
        cutmix_prob=cutmix_prob,
        cutmix_alpha=cutmix_alpha,
    )
    valid_ds = BiomassDataset(valid_df, image_dir, valid_transform, is_train=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=4,
    )
    
    # Model
    model = DINOv3Direct(
        grid=grid,
        pretrained=True,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
        use_film=use_film,
        use_attention_pool=use_attention_pool,
        train_dead=train_dead,
        train_clover=train_clover,
    ).to(device)
    
    # Freeze backbone (head-only training for search)
    freeze_backbone(model)
    
    # Loss
    loss_fn = BiomassLoss(
        use_huber_for_dead=use_huber,
        huber_delta=huber_delta,
        train_dead=train_dead,
        train_clover=train_clover,
    )
    
    # Optimizer & scheduler
    use_fused = supports_fused_optimizer(device_type)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, fused=use_fused)
    
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    
    # Training loop
    best_r2 = -float("inf")
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip)
        val_loss, r2, metrics = validate(model, valid_loader, loss_fn, device)
        scheduler.step()
        
        improved = ""
        if r2 > best_r2 and not np.isnan(r2):
            best_r2 = r2
            best_metrics = metrics.copy()
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1
        
        # Print progress
        print(f"    Ep {epoch:02d}: loss={train_loss:.4f}/{val_loss:.4f} R²={r2:.4f} (best={best_r2:.4f}){improved}", flush=True)
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print("    NaN detected, stopping", flush=True)
            break
        
        if patience_counter >= patience:
            print(f"    Early stop (no improvement for {patience} epochs)", flush=True)
            break
    
    # Cleanup
    del model, optimizer, scheduler
    empty_cache(device_type)
    gc.collect()
    
    return best_r2, best_metrics


def create_arch_objective(
    df: pd.DataFrame,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    train_dead: bool,
    train_clover: bool,
    max_epochs: int,
    patience: int,
    n_folds_eval: int,
    seed: int,
):
    """Objective for ARCHITECTURE search (Stage 1)."""
    
    # Fixed training params for arch search
    fixed = {
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_epochs": 2,
        "grad_clip": 1.0,
        "batch_size": 8,
        "aug_prob": 0.5,
        "use_huber": True,
        "huber_delta": 5.0,
    }

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed + trial.number, device_type)

        # Architecture params to tune
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        hidden_ratio = trial.suggest_float("hidden_ratio", 0.15, 0.5)
        use_film = trial.suggest_categorical("use_film", [True, False])
        use_attention_pool = trial.suggest_categorical("use_attention_pool", [True, False])
        grid_size = trial.suggest_categorical("grid_size", [2, 3])

        params = {
            **fixed,
            "dropout": dropout,
            "hidden_ratio": hidden_ratio,
            "use_film": use_film,
            "use_attention_pool": use_attention_pool,
            "grid_size": grid_size,
        }
        
        heads = "Total, Green, GDM"
        if train_dead:
            heads += ", Dead"
        if train_clover:
            heads += ", Clover"
        
        print(f"\n{'='*60}")
        print(f"[Trial {trial.number}] Architecture Search")
        print(f"{'='*60}")
        print(f"  Heads: {heads}")
        print(f"  dropout={dropout:.3f} | hidden={hidden_ratio:.3f} | grid={grid_size}")
        print(f"  film={use_film} | attn_pool={use_attention_pool}")
        print(f"  (fixed: lr=1e-4, wd=0.01, batch=8, aug=0.5)")
        print(f"{'-'*60}")
        
        # Cross-validation
        fold_scores = []
        
        for fold in range(n_folds_eval):
            train_df = df[df["fold"] != fold].reset_index(drop=True)
            valid_df = df[df["fold"] == fold].reset_index(drop=True)
            
            print(f"  Fold {fold}:")
            try:
                r2, metrics = train_fold(
                    fold=fold,
                    train_df=train_df,
                    valid_df=valid_df,
                    params=params,
                    image_dir=image_dir,
                    device=device,
                    device_type=device_type,
                    train_dead=train_dead,
                    train_clover=train_clover,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                fold_scores.append(r2)
                
                per_target = " | ".join([
                    f"G={metrics.get('r2_green', 0):.3f}",
                    f"D={metrics.get('r2_dead', 0):.3f}",
                    f"C={metrics.get('r2_clover', 0):.3f}",
                    f"GDM={metrics.get('r2_gdm', 0):.3f}",
                    f"T={metrics.get('r2_total', 0):.3f}",
                ])
                print(f"    >> Fold {fold} Best: R²={r2:.4f} [{per_target}]")
                
            except Exception as e:
                print(f"    Fold {fold} failed: {e}")
                return -1.0
        
        cv_score = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        print(f"  CV: {cv_score:.4f} ± {cv_std:.4f}")
        
        return cv_score
    
    return objective


def create_training_objective(
    df: pd.DataFrame,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    train_dead: bool,
    train_clover: bool,
    fixed_arch: Dict[str, Any],
    max_epochs: int,
    patience: int,
    n_folds_eval: int,
    seed: int,
):
    """Objective for TRAINING DYNAMICS search (Stage 2)."""

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed + trial.number, device_type)

        # Training params to tune
        lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.005, 0.1, log=True)
        warmup_epochs = trial.suggest_int("warmup_epochs", 1, 4)
        grad_clip = trial.suggest_float("grad_clip", 0.3, 1.5)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        aug_prob = trial.suggest_float("aug_prob", 0.3, 0.7)

        params = {
            **fixed_arch,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "grad_clip": grad_clip,
            "batch_size": batch_size,
            "aug_prob": aug_prob,
            "use_huber": True,
            "huber_delta": 5.0,
        }
        
        print(f"\n{'='*60}")
        print(f"[Trial {trial.number}] Training Dynamics Search")
        print(f"{'='*60}")
        print(f"  lr={lr:.2e} | wd={weight_decay:.4f} | warmup={warmup_epochs}")
        print(f"  grad_clip={grad_clip:.2f} | batch={batch_size} | aug={aug_prob:.2f}")
        print(f"  (fixed arch: grid={fixed_arch['grid_size']}, dropout={fixed_arch['dropout']:.3f})")
        print(f"{'-'*60}")
        
        fold_scores = []
        
        for fold in range(n_folds_eval):
            train_df = df[df["fold"] != fold].reset_index(drop=True)
            valid_df = df[df["fold"] == fold].reset_index(drop=True)
            
            print(f"  Fold {fold}:")
            try:
                r2, metrics = train_fold(
                    fold=fold,
                    train_df=train_df,
                    valid_df=valid_df,
                    params=params,
                    image_dir=image_dir,
                    device=device,
                    device_type=device_type,
                    train_dead=train_dead,
                    train_clover=train_clover,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                fold_scores.append(r2)
                print(f"    >> Fold {fold} Best: R²={r2:.4f}")
                
            except Exception as e:
                print(f"    Fold {fold} failed: {e}")
                return -1.0
        
        cv_score = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        print(f"  CV: {cv_score:.4f} ± {cv_std:.4f}")
        
        return cv_score
    
    return objective


def create_augmentation_objective(
    df: pd.DataFrame,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    train_dead: bool,
    train_clover: bool,
    fixed_arch: Dict[str, Any],
    fixed_training: Dict[str, Any],
    max_epochs: int,
    patience: int,
    n_folds_eval: int,
    seed: int,
):
    """Objective for AUGMENTATION search (Stage 3)."""

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed + trial.number, device_type)

        # Augmentation params to tune
        aug_prob = trial.suggest_float("aug_prob", 0.3, 0.7)
        photometric_mode = trial.suggest_categorical("photometric_mode", 
                                                      ["same", "independent", "left", "right", "none"])
        stereo_swap_prob = trial.suggest_float("stereo_swap_prob", 0.0, 0.3)
        
        # MixUp/CutMix (mutually exclusive or both disabled)
        mix_type = trial.suggest_categorical("mix_type", ["none", "mixup", "cutmix"])
        
        if mix_type == "mixup":
            mixup_prob = trial.suggest_float("mixup_prob", 0.1, 0.5)
            mixup_alpha = trial.suggest_float("mixup_alpha", 0.2, 0.8)
            cutmix_prob = 0.0
            cutmix_alpha = 1.0
        elif mix_type == "cutmix":
            mixup_prob = 0.0
            mixup_alpha = 0.4
            cutmix_prob = trial.suggest_float("cutmix_prob", 0.1, 0.5)
            cutmix_alpha = trial.suggest_float("cutmix_alpha", 0.5, 1.5)
        else:
            mixup_prob = 0.0
            mixup_alpha = 0.4
            cutmix_prob = 0.0
            cutmix_alpha = 1.0

        params = {
            **fixed_arch,
            **fixed_training,
            "aug_prob": aug_prob,
            "photometric_mode": photometric_mode,
            "stereo_swap_prob": stereo_swap_prob,
            "mixup_prob": mixup_prob,
            "mixup_alpha": mixup_alpha,
            "cutmix_prob": cutmix_prob,
            "cutmix_alpha": cutmix_alpha,
            "use_huber": True,
            "huber_delta": 5.0,
        }
        
        print(f"\n{'='*60}")
        print(f"[Trial {trial.number}] Augmentation Search")
        print(f"{'='*60}")
        print(f"  aug_prob={aug_prob:.2f} | photometric={photometric_mode}")
        print(f"  stereo_swap={stereo_swap_prob:.2f}")
        if mix_type == "mixup":
            print(f"  MixUp: prob={mixup_prob:.2f}, alpha={mixup_alpha:.2f}")
        elif mix_type == "cutmix":
            print(f"  CutMix: prob={cutmix_prob:.2f}, alpha={cutmix_alpha:.2f}")
        else:
            print(f"  No MixUp/CutMix")
        print(f"{'-'*60}")
        
        fold_scores = []
        
        for fold in range(n_folds_eval):
            train_df = df[df["fold"] != fold].reset_index(drop=True)
            valid_df = df[df["fold"] == fold].reset_index(drop=True)
            
            print(f"  Fold {fold}:")
            try:
                r2, metrics = train_fold(
                    fold=fold,
                    train_df=train_df,
                    valid_df=valid_df,
                    params=params,
                    image_dir=image_dir,
                    device=device,
                    device_type=device_type,
                    train_dead=train_dead,
                    train_clover=train_clover,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                fold_scores.append(r2)
                print(f"    >> Fold {fold} Best: R²={r2:.4f}")
                
            except Exception as e:
                print(f"    Fold {fold} failed: {e}")
                return -1.0
        
        cv_score = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        print(f"  CV: {cv_score:.4f} ± {cv_std:.4f}")
        
        return cv_score
    
    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 Direct Optuna Search")
    
    # Data
    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--cv-strategy", type=str, default="group_date_state")
    
    # Search config
    parser.add_argument("--stage", type=str, default="arch", choices=["arch", "training", "augmentation"],
                        help="Search stage: 'arch', 'training', or 'augmentation'")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--n-folds-eval", type=int, default=3)
    
    # Head options
    parser.add_argument("--train-dead", action="store_true",
                        help="Train Dead head directly")
    parser.add_argument("--train-clover", action="store_true",
                        help="Train Clover head directly")
    
    # Fixed architecture (for training/augmentation stage)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-ratio", type=float, default=0.25)
    parser.add_argument("--use-film", action="store_true", default=True)
    parser.add_argument("--no-film", action="store_false", dest="use_film")
    parser.add_argument("--use-attention-pool", action="store_true", default=True)
    parser.add_argument("--no-attention-pool", action="store_false", dest="use_attention_pool")
    
    # Fixed training params (for augmentation stage)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    
    # System
    parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    device_type = DeviceType(args.device_type) if args.device_type else get_device_type()
    device = get_device(device_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"./outputs/optuna_dinov3_{args.stage}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    storage = f"sqlite:///{args.output_dir}/optuna.db"

    # Heads info
    heads = ["Total", "Green", "GDM"]
    if args.train_dead:
        heads.append("Dead")
    if args.train_clover:
        heads.append("Clover")
    
    print("=" * 60)
    print(f"DINOv3 Direct - Optuna Search ({args.stage.upper()})")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Heads: {', '.join(heads)}")
    print(f"Stage: {args.stage}")
    print(f"Trials: {args.n_trials}")
    print(f"Folds per trial: {args.n_folds_eval}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load data
    train_csv = os.path.join(args.base_path, "train.csv")
    image_dir = os.path.join(args.base_path, "train")
    df = prepare_dataframe(train_csv)
    df = create_folds(df, n_folds=5, seed=args.seed, cv_strategy=args.cv_strategy)

    print(f"Samples: {len(df)}")
    print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")

    # Create study
    sampler = TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=f"dinov3_{args.stage}_{timestamp}",
        storage=storage,
        direction="maximize",
        sampler=sampler,
    )

    # Fixed params for later stages
    fixed_arch = {
        "grid_size": args.grid,
        "dropout": args.dropout,
        "hidden_ratio": args.hidden_ratio,
        "use_film": args.use_film,
        "use_attention_pool": args.use_attention_pool,
    }
    fixed_training = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
    }
    
    # Create objective based on stage
    if args.stage == "arch":
        objective = create_arch_objective(
            df=df,
            image_dir=image_dir,
            device=device,
            device_type=device_type,
            train_dead=args.train_dead,
            train_clover=args.train_clover,
            max_epochs=args.max_epochs,
            patience=args.patience,
            n_folds_eval=args.n_folds_eval,
            seed=args.seed,
        )
    elif args.stage == "training":
        objective = create_training_objective(
            df=df,
            image_dir=image_dir,
            device=device,
            device_type=device_type,
            train_dead=args.train_dead,
            train_clover=args.train_clover,
            fixed_arch=fixed_arch,
            max_epochs=args.max_epochs,
            patience=args.patience,
            n_folds_eval=args.n_folds_eval,
            seed=args.seed,
        )
    else:  # augmentation
        objective = create_augmentation_objective(
            df=df,
            image_dir=image_dir,
            device=device,
            device_type=device_type,
            train_dead=args.train_dead,
            train_clover=args.train_clover,
            fixed_arch=fixed_arch,
            fixed_training=fixed_training,
            max_epochs=args.max_epochs,
            patience=args.patience,
            n_folds_eval=args.n_folds_eval,
            seed=args.seed,
        )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Results
    print("\n" + "=" * 60)
    print(f"Search Complete! ({args.stage.upper()})")
    print("=" * 60)
    print(f"Best CV R²: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    results = {
        "stage": args.stage,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "train_dead": args.train_dead,
        "train_clover": args.train_clover,
        "img_size": IMG_SIZE,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print next steps
    bp = study.best_params
    print("\n" + "=" * 60)
    
    if args.stage == "arch":
        print("Next: Run training dynamics search with best arch:")
        print("=" * 60)
        cmd = f"""python -m src.optuna_dinov3_direct --stage training \\
    --grid {bp['grid_size']} \\
    --dropout {bp['dropout']:.3f} \\
    --hidden-ratio {bp['hidden_ratio']:.3f} \\
    {'--use-film' if bp['use_film'] else '--no-film'} \\
    {'--use-attention-pool' if bp['use_attention_pool'] else '--no-attention-pool'} \\
    {'--train-dead' if args.train_dead else ''} \\
    {'--train-clover' if args.train_clover else ''} \\
    --n-trials 10 \\
    --device-type {args.device_type or 'mps'}"""
        print(cmd)
    elif args.stage == "training":
        print("Next: Run augmentation search with best training params:")
        print("=" * 60)
        cmd = f"""python -m src.optuna_dinov3_direct --stage augmentation \\
    --grid {args.grid} \\
    --dropout {args.dropout:.3f} \\
    --hidden-ratio {args.hidden_ratio:.3f} \\
    {'--use-film' if args.use_film else '--no-film'} \\
    {'--use-attention-pool' if args.use_attention_pool else '--no-attention-pool'} \\
    --lr {bp['lr']:.2e} \\
    --weight-decay {bp['weight_decay']:.4f} \\
    --warmup-epochs {bp['warmup_epochs']} \\
    --grad-clip {bp['grad_clip']:.2f} \\
    --batch-size {bp['batch_size']} \\
    {'--train-dead' if args.train_dead else ''} \\
    {'--train-clover' if args.train_clover else ''} \\
    --n-trials 10 \\
    --device-type {args.device_type or 'mps'}"""
        print(cmd)
    else:  # augmentation
        print("Done! Run full training with best params:")
        print("=" * 60)
        
        # Build augmentation args
        aug_args = []
        photometric_mode = bp.get('photometric_mode', 'same')
        if photometric_mode != 'same':
            aug_args.append(f"--photometric {photometric_mode}")
        if bp.get('stereo_swap_prob', 0) > 0:
            aug_args.append(f"--stereo-swap-prob {bp['stereo_swap_prob']:.2f}")
        if bp.get('mixup_prob', 0) > 0:
            aug_args.append(f"--mixup-prob {bp['mixup_prob']:.2f}")
            aug_args.append(f"--mixup-alpha {bp.get('mixup_alpha', 0.4):.2f}")
        if bp.get('cutmix_prob', 0) > 0:
            aug_args.append(f"--cutmix-prob {bp['cutmix_prob']:.2f}")
            aug_args.append(f"--cutmix-alpha {bp.get('cutmix_alpha', 1.0):.2f}")
        
        aug_str = " \\\n    ".join(aug_args) if aug_args else ""
        
        cmd = f"""python -m src.dinov3_train \\
    --grid {args.grid} \\
    --dropout {args.dropout:.3f} \\
    --hidden-ratio {args.hidden_ratio:.3f} \\
    {'--no-film' if not args.use_film else ''} \\
    {'--no-attention-pool' if not args.use_attention_pool else ''} \\
    {'--train-dead' if args.train_dead else ''} \\
    {'--train-clover' if args.train_clover else ''} \\
    --lr {args.lr:.2e} \\
    --weight-decay {args.weight_decay:.4f} \\
    --warmup-epochs {args.warmup_epochs} \\
    --grad-clip {args.grad_clip:.2f} \\
    --batch-size {args.batch_size} \\
    --aug-prob {bp['aug_prob']:.2f} \\
    {aug_str}
    --freeze-backbone \\
    --epochs 40 \\
    --device-type {args.device_type or 'mps'}"""
        print(cmd)


if __name__ == "__main__":
    main()

