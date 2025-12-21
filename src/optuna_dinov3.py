#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for DINOv3 Direct Model (Frozen Backbone).

Focused search for LB generalization (target ~0.77+):
- DINOv3 backbone only (vit_base_patch16_dinov3)
- Direct model (predict Total, Green, GDM; derive Dead, Clover)
- Frozen backbone (head-only training)
- Image size 992
- Emphasis on regularization to prevent overfitting

Key insight: The gap between CV and LB suggests overfitting.
This search focuses on regularization params (dropout, hidden_ratio, augmentation).

Usage:
    # Quick search (recommended first)
    python -m src.optuna_dinov3 --n-trials 30 --max-epochs 25 --device-type mps

    # Full search with persistence
    python -m src.optuna_dinov3 --n-trials 50 --max-epochs 35 --device-type mps

    # Resume interrupted search
    python -m src.optuna_dinov3 --resume --output-dir ./outputs/optuna_dinov3_YYYYMMDD
"""
import argparse
import gc
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import torch
import torch.nn as nn
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
    get_device,
    get_device_type,
    set_device_seed,
    supports_fused_optimizer,
)
from .models_ratio import build_ratio_model, RatioMSELoss
from .trainer import compute_weighted_r2, compute_per_target_metrics_np


# Fixed DINOv3 configuration
BACKBONE = "vit_base_patch16_dinov3"
IMG_SIZE = 992


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


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = True


def get_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    device_type: DeviceType,
    backbone_lr: float = None,
    stage: int = 1,
) -> AdamW:
    """Create optimizer. Stage 1 = head only, Stage 2 = differential LR."""
    use_fused = supports_fused_optimizer(device_type)
    
    if stage == 1 or backbone_lr is None:
        # Head-only training
        params = [p for p in model.parameters() if p.requires_grad]
        return AdamW(params, lr=lr, weight_decay=weight_decay, fused=use_fused)
    else:
        # Differential LR for backbone vs heads
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
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-7)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


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
        
        # No AMP for MPS stability with DINOv3
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
    use_log_target: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []
    
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    
    for batch in loader:
        x_left, x_right, targets = batch[:3]
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
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
    dropout = params["dropout"]
    hidden_ratio = params["hidden_ratio"]
    use_film = params["use_film"]
    use_attention_pool = params["use_attention_pool"]
    grid_size = params["grid_size"]
    
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    warmup_epochs = params["warmup_epochs"]
    grad_clip = params["grad_clip"]
    batch_size = params["batch_size"]
    
    use_huber_for_dead = params["use_huber_for_dead"]
    huber_delta = params["huber_delta"]
    
    use_log_target = params["use_log_target"]
    stereo_correct_aug = params["stereo_correct_aug"]
    stereo_swap_prob = params["stereo_swap_prob"]
    mixup_prob = params["mixup_prob"]
    mixup_alpha = params["mixup_alpha"]
    aug_prob = params["aug_prob"]
    use_vegetation_indices = params["use_vegetation_indices"]
    
    # Training mode params
    training_mode = params.get("training_mode", "freeze")
    backbone_lr = params.get("backbone_lr", 1e-5)
    freeze_epochs = params.get("freeze_epochs", 5)
    
    grid = (grid_size, grid_size)
    
    # Create transforms
    if stereo_correct_aug:
        train_transform = get_stereo_geometric_transforms(IMG_SIZE, aug_prob)
        photometric_transform = get_stereo_photometric_transforms(aug_prob)
    else:
        train_transform = get_train_transforms(IMG_SIZE, aug_prob)
        photometric_transform = None
    
    valid_transform = get_valid_transforms(IMG_SIZE)
    
    # Datasets
    train_ds = BiomassDataset(
        train_df, image_dir, train_transform,
        is_train=True, use_log_target=use_log_target,
        stereo_swap_prob=stereo_swap_prob,
        photometric_transform=photometric_transform,
        mixup_prob=mixup_prob,
        mixup_alpha=mixup_alpha,
    )
    valid_ds = BiomassDataset(
        valid_df, image_dir, valid_transform,
        is_train=False, use_log_target=use_log_target,
    )
    
    num_workers = 4
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers,
    )
    
    # Build model - Direct model only
    model = build_ratio_model(
        backbone_name=BACKBONE,
        grid=grid,
        pretrained=True,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
        use_film=use_film,
        use_attention_pool=use_attention_pool,
        model_type="direct",
        use_vegetation_indices=use_vegetation_indices,
    ).to(device)
    
    # Loss function
    loss_fn = RatioMSELoss(
        target_weights=[0.1, 0.1, 0.1, 0.2, 0.5],
        use_huber_for_dead=use_huber_for_dead,
        huber_delta=huber_delta,
    )
    
    # Setup training mode
    current_stage = 1
    if training_mode == "freeze":
        # Head-only training (backbone frozen throughout)
        freeze_backbone(model)
        optimizer = get_optimizer(model, lr, weight_decay, device_type, stage=1)
        scheduler = get_scheduler(optimizer, max_epochs, warmup_epochs)
    elif training_mode == "two_stage":
        # Stage 1: freeze backbone, Stage 2: finetune all
        freeze_backbone(model)
        optimizer = get_optimizer(model, lr, weight_decay, device_type, stage=1)
        scheduler = get_scheduler(optimizer, freeze_epochs, warmup_epochs=1)
    else:  # "full"
        # Train everything from start
        current_stage = 2
        optimizer = get_optimizer(model, lr, weight_decay, device_type, backbone_lr=backbone_lr, stage=2)
        scheduler = get_scheduler(optimizer, max_epochs, warmup_epochs)
    
    # Training loop
    best_r2 = -float("inf")
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    
    for epoch in range(1, max_epochs + 1):
        # Stage transition for two-stage training
        if training_mode == "two_stage" and epoch == freeze_epochs + 1:
            current_stage = 2
            unfreeze_backbone(model)
            optimizer = get_optimizer(model, lr, weight_decay, device_type, backbone_lr=backbone_lr, stage=2)
            remaining = max_epochs - freeze_epochs
            scheduler = get_scheduler(optimizer, remaining, warmup_epochs)
            patience_counter = 0  # Reset patience for stage 2
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip=grad_clip,
        )
        
        valid_loss, r2, metrics = validate(
            model, valid_loader, loss_fn, device, use_log_target=use_log_target,
        )
        
        scheduler.step()
        
        # Report to Optuna for pruning
        if trial is not None:
            trial.report(r2, report_epoch_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Track best (only save in stage 2 for two-stage, or always for freeze/full)
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
    empty_cache(device_type)
    gc.collect()
    
    return best_r2, valid_loss, best_metrics


def create_objective(
    df: pd.DataFrame,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    max_epochs: int,
    patience: int,
    n_folds_eval: int,
    seed: int,
):
    """Create Optuna objective function for DINOv3 direct model."""

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed + trial.number, device_type)

        # =====================================================================
        # HEAD ARCHITECTURE - Key for generalization
        # =====================================================================
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        hidden_ratio = trial.suggest_float("hidden_ratio", 0.15, 0.5)
        
        # Attention/FiLM - can help or hurt generalization
        use_film = trial.suggest_categorical("use_film", [True, False])
        use_attention_pool = trial.suggest_categorical("use_attention_pool", [True, False])
        grid_size = trial.suggest_categorical("grid_size", [2, 3])

        # =====================================================================
        # TRAINING MODE - freeze, two_stage, or full
        # =====================================================================
        training_mode = trial.suggest_categorical("training_mode", ["freeze", "two_stage"])
        
        # Learning rate for head
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.005, 0.1, log=True)
        warmup_epochs = trial.suggest_int("warmup_epochs", 1, 4)
        grad_clip = trial.suggest_float("grad_clip", 0.3, 1.5)
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        
        # Backbone training params (only used if training_mode != "freeze")
        if training_mode == "two_stage":
            freeze_epochs = trial.suggest_int("freeze_epochs", 3, 10)
            backbone_lr = trial.suggest_float("backbone_lr", 1e-6, 5e-5, log=True)
        else:
            freeze_epochs = 0
            backbone_lr = 1e-5

        # =====================================================================
        # LOSS CONFIGURATION
        # =====================================================================
        use_huber_for_dead = trial.suggest_categorical("use_huber_for_dead", [True, False])
        huber_delta = trial.suggest_float("huber_delta", 2.0, 10.0) if use_huber_for_dead else 5.0

        # =====================================================================
        # AUGMENTATION - Critical for LB generalization
        # =====================================================================
        aug_prob = trial.suggest_float("aug_prob", 0.4, 0.7)
        stereo_correct_aug = trial.suggest_categorical("stereo_correct_aug", [True, False])
        stereo_swap_prob = trial.suggest_float("stereo_swap_prob", 0.0, 0.3)
        mixup_prob = trial.suggest_float("mixup_prob", 0.0, 0.4)
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.2, 0.8) if mixup_prob > 0 else 0.4

        # =====================================================================
        # TARGET TRANSFORMATION
        # =====================================================================
        use_log_target = trial.suggest_categorical("use_log_target", [True, False])

        # =====================================================================
        # VEGETATION INDICES - Domain-specific features (ExG, ExR, GRVI, etc.)
        # =====================================================================
        use_vegetation_indices = trial.suggest_categorical("use_vegetation_indices", [True, False])

        params = {
            "dropout": dropout,
            "hidden_ratio": hidden_ratio,
            "use_film": use_film,
            "use_attention_pool": use_attention_pool,
            "grid_size": grid_size,
            "training_mode": training_mode,
            "lr": lr,
            "backbone_lr": backbone_lr,
            "freeze_epochs": freeze_epochs,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "grad_clip": grad_clip,
            "batch_size": batch_size,
            "use_huber_for_dead": use_huber_for_dead,
            "huber_delta": huber_delta,
            "aug_prob": aug_prob,
            "stereo_correct_aug": stereo_correct_aug,
            "stereo_swap_prob": stereo_swap_prob,
            "mixup_prob": mixup_prob,
            "mixup_alpha": mixup_alpha,
            "use_log_target": use_log_target,
            "use_vegetation_indices": use_vegetation_indices,
        }
        
        # Print trial info
        print(f"\n{'='*70}")
        print(f"[Trial {trial.number}] DINOv3 Direct Model")
        print(f"{'='*70}")
        print(f"  Arch:    dropout={dropout:.3f} | hidden={hidden_ratio:.3f} | grid={grid_size}")
        print(f"           film={use_film} | attn_pool={use_attention_pool} | veg_idx={use_vegetation_indices}")
        print(f"  Mode:    {training_mode}", end="")
        if training_mode == "two_stage":
            print(f" | freeze_ep={freeze_epochs} | backbone_lr={backbone_lr:.2e}")
        else:
            print(" (head-only)")
        print(f"  Train:   lr={lr:.2e} | wd={weight_decay:.4f} | grad_clip={grad_clip:.2f}")
        print(f"           warmup={warmup_epochs} | batch={batch_size}")
        print(f"  Loss:    huber_dead={use_huber_for_dead} | huber_δ={huber_delta:.1f}")
        print(f"  Aug:     prob={aug_prob:.2f} | stereo_correct={stereo_correct_aug}")
        print(f"           stereo_swap={stereo_swap_prob:.2f} | mixup={mixup_prob:.2f}")
        print(f"  Target:  log={use_log_target}")
        print(f"{'-'*70}")
        
        # Cross-validation
        fold_scores = []
        fold_details = []
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
                    image_dir=image_dir,
                    device=device,
                    device_type=device_type,
                    max_epochs=max_epochs,
                    patience=patience,
                    trial=trial,
                    report_epoch_offset=i * max_epochs,
                )
                fold_scores.append(r2)
                fold_details.append(metrics)
                
                # Log fold result
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
        cv_std = np.std(fold_scores)
        
        # Penalize high variance (suggests overfitting potential)
        # A consistent model across folds is more likely to generalize
        adjusted_score = cv_score - 0.5 * cv_std
        
        print(f"  CV Score: {cv_score:.4f} ± {cv_std:.4f} (adjusted: {adjusted_score:.4f})")
        
        return cv_score
    
    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna Search for DINOv3 Direct Model")

    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--fold-csv", type=str, default=None,
                        help="Path to CSV with predefined folds (default: None, create folds)")
    parser.add_argument("--cv-strategy", type=str, default="group_date_state",
                        help="CV strategy: group_date_state (group by month, stratify by State), "
                             "group_month (group by month, stratify by target_bin), "
                             "group_date, stratified, random")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--n-folds-eval", type=int, default=3,
                        help="Number of folds to evaluate per trial (1-5)")
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
        args.output_dir = f"./outputs/optuna_dinov3_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-enable storage for persistence
    if not args.storage:
        args.storage = f"sqlite:///{args.output_dir}/optuna.db"

    print("=" * 70)
    print("Optuna Search - DINOv3 Direct Model (Frozen Backbone)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Backbone: {BACKBONE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Model: Direct (predict Total, Green, GDM)")
    print(f"Training: HEAD-ONLY (frozen backbone)")
    print(f"Trials: {args.n_trials}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Folds per trial: {args.n_folds_eval}")
    print(f"Output: {args.output_dir}")
    print(f"Storage: {args.storage}")
    print("=" * 70)

    # Load data
    train_csv = os.path.join(args.base_path, "train.csv")
    image_dir = os.path.join(args.base_path, "train")

    df = prepare_dataframe(train_csv)

    # Load predefined folds or create
    if args.fold_csv and os.path.exists(args.fold_csv):
        print(f"Loading predefined folds from: {args.fold_csv}")
        fold_df = pd.read_csv(args.fold_csv)
        fold_mapping = fold_df.set_index("sample_id_prefix")["fold"].to_dict()
        df["fold"] = df["sample_id_prefix"].map(fold_mapping).fillna(0).astype(int)
    else:
        print(f"Creating folds using {args.cv_strategy} strategy")
        df = create_folds(df, n_folds=5, seed=args.seed, cv_strategy=args.cv_strategy)

    print(f"Samples: {len(df)}")
    print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")

    # Study name
    study_name = args.study_name or f"dinov3_direct_{timestamp}"

    # Create study
    sampler = TPESampler(seed=args.seed, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=8)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=args.resume,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Check if resuming
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed > 0:
        print(f"Resuming: {completed} trials already completed")
        remaining = max(0, args.n_trials - completed)
    else:
        remaining = args.n_trials

    if remaining > 0:
        objective = create_objective(
            df=df,
            image_dir=image_dir,
            device=device,
            device_type=device_type,
            max_epochs=args.max_epochs,
            patience=args.patience,
            n_folds_eval=args.n_folds_eval,
            seed=args.seed,
        )

        print(f"\nRunning {remaining} trials...")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    # Results
    print("\n" + "=" * 70)
    print("Search Complete!")
    print("=" * 70)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best CV R²: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    results = {
        "study_name": study_name,
        "backbone": BACKBONE,
        "img_size": IMG_SIZE,
        "model_type": "direct",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "n_trials": len(study.trials),
        "config": {
            "max_epochs": args.max_epochs,
            "n_folds_eval": args.n_folds_eval,
            "cv_strategy": args.cv_strategy,
        },
    }

    results_path = os.path.join(args.output_dir, "search_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate training command
    bp = study.best_params
    print("\n" + "=" * 70)
    print("Recommended training command:")
    print("=" * 70)

    cmd_parts = [
        "python -m src.train_ratio",
        f"    --backbone {BACKBONE}",
        f"    --img-size {IMG_SIZE}",
        "    --model-type direct",
        "    --freeze-backbone",
        f"    --grid {bp.get('grid_size', 2)} {bp.get('grid_size', 2)}",
        f"    --dropout {bp.get('dropout', 0.3):.3f}",
        f"    --hidden-ratio {bp.get('hidden_ratio', 0.25):.3f}",
    ]

    if not bp.get('use_film', True):
        cmd_parts.append("    --no-film")
    if not bp.get('use_attention_pool', True):
        cmd_parts.append("    --no-attention-pool")

    # Training mode
    training_mode = bp.get('training_mode', 'freeze')
    if training_mode == "freeze":
        cmd_parts.append("    --freeze-backbone")
        cmd_parts.append(f"    --head-lr-stage1 {bp.get('lr', 5e-4):.2e}")
    elif training_mode == "two_stage":
        cmd_parts.append("    --two-stage")
        cmd_parts.append(f"    --freeze-epochs {bp.get('freeze_epochs', 5)}")
        cmd_parts.append(f"    --head-lr-stage1 {bp.get('lr', 5e-4):.2e}")
        cmd_parts.append(f"    --lr {bp.get('lr', 2e-4):.2e}")
        cmd_parts.append(f"    --backbone-lr {bp.get('backbone_lr', 1e-5):.2e}")

    cmd_parts.extend([
        f"    --weight-decay {bp.get('weight_decay', 0.01):.4f}",
        f"    --warmup-epochs {bp.get('warmup_epochs', 2)}",
        f"    --grad-clip {bp.get('grad_clip', 0.5):.2f}",
        f"    --batch-size {bp.get('batch_size', 4)}",
        f"    --aug-prob {bp.get('aug_prob', 0.5):.2f}",
    ])

    if not bp.get('use_huber_for_dead', True):
        cmd_parts.append("    --no-huber-for-dead")
    if bp.get('huber_delta', 5.0) != 5.0:
        cmd_parts.append(f"    --huber-delta {bp['huber_delta']:.1f}")

    if bp.get('stereo_correct_aug', False):
        cmd_parts.append("    --stereo-correct-aug")
    if bp.get('stereo_swap_prob', 0.0) > 0:
        cmd_parts.append(f"    --stereo-swap-prob {bp['stereo_swap_prob']:.2f}")
    if bp.get('mixup_prob', 0.0) > 0:
        cmd_parts.append(f"    --mixup-prob {bp['mixup_prob']:.2f}")
        cmd_parts.append(f"    --mixup-alpha {bp.get('mixup_alpha', 0.4):.2f}")
    if bp.get('use_log_target', False):
        cmd_parts.append("    --use-log-target")
    if bp.get('use_vegetation_indices', False):
        cmd_parts.append("    --use-vegetation-indices")

    cmd_parts.extend([
        "    --cv-strategy group_date_state",
        "    --epochs 50",
        "    --patience 10",
        "    --no-amp",
        "    --device-type mps",
        "    --compute-oof",
    ])

    cmd = " \\\n".join(cmd_parts)
    print(cmd)

    # Save command to file
    cmd_path = os.path.join(args.output_dir, "best_train_cmd.sh")
    with open(cmd_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(cmd + "\n")
    print(f"\nCommand saved to: {cmd_path}")


if __name__ == "__main__":
    main()

