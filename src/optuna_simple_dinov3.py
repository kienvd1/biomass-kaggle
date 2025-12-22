#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Simple DINOv3 Model.

Simple setup inspired by top LB insights:
- DINOv3 backbone only (no depth, no attention pool)
- 4 heads: Green, Clover, GDM, Total
- Dead derived from Total - GDM
- Focus on finding the right hyperparameters for generalization

Usage:
    # Quick search (3 folds, 30 trials)
    python -m src.optuna_simple_dinov3 --n-trials 30 --n-folds-eval 3

    # Full search with all folds
    python -m src.optuna_simple_dinov3 --n-trials 50 --n-folds-eval 5

    # Test different fold strategies
    python -m src.optuna_simple_dinov3 --fold-csv data/trainfold_group_location.csv
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    BiomassDataset,
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
from .dinov3_models import DINOv3Direct, TARGET_WEIGHTS


# Fixed configuration for simple model
IMG_SIZES = [518, 576, 672]  # Search over 3 image sizes (518=native DINOv3)
BACKBONE_SIZE = "base"


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


class SimpleLoss(nn.Module):
    """Simple weighted loss for biomass prediction."""
    
    def __init__(
        self,
        use_smoothl1: bool = True,
        smoothl1_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_smoothl1 = use_smoothl1
        self.smoothl1_beta = smoothl1_beta
        self.register_buffer("weights", TARGET_WEIGHTS)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(pred.device)
        
        if self.use_smoothl1:
            # Per-target SmoothL1
            loss_per_target = F.smooth_l1_loss(
                pred, target, reduction='none', beta=self.smoothl1_beta
            ).mean(dim=0)
        else:
            # MSE
            loss_per_target = ((pred - target) ** 2).mean(dim=0)
        
        return (weights * loss_per_target).sum()


def compute_r2(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Compute weighted R² on original values (competition metric)."""
    target_names = ["green", "dead", "clover", "gdm", "total"]
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    
    metrics = {}
    
    # Per-target R²
    for i, name in enumerate(target_names):
        y = targets[:, i]
        y_hat = preds[:, i]
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        metrics[f"r2_{name}"] = float(r2)
    
    # Weighted average
    avg_r2 = sum(target_weights[i] * metrics[f"r2_{name}"] for i, name in enumerate(target_names))
    metrics["avg_r2"] = avg_r2
    
    # Global weighted R² (competition metric)
    weights = np.array(target_weights)
    n_samples = preds.shape[0]
    w = np.tile(weights, n_samples)
    y = targets.flatten()
    y_hat = preds.flatten()
    y_bar_w = np.sum(w * y) / np.sum(w)
    ss_res = np.sum(w * (y - y_hat) ** 2)
    ss_tot = np.sum(w * (y - y_bar_w) ** 2)
    weighted_r2 = 1 - ss_res / (ss_tot + 1e-8)
    metrics["weighted_r2"] = float(weighted_r2)
    
    return avg_r2, metrics


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
        
        optimizer.zero_grad()
        
        outputs = model(x_left, x_right, return_aux=False)
        green, dead, clover, gdm, total = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
        preds = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        loss = loss_fn(preds, targets)
        loss.backward()
        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
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
        
        outputs = model(x_left, x_right, return_aux=False)
        green, dead, clover, gdm, total = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
        preds = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        loss = loss_fn(preds, targets)
        losses.update(loss.item(), x_left.size(0))
        
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Clamp predictions
    all_preds = np.clip(all_preds, 0, None)
    
    r2, metrics = compute_r2(all_preds, all_targets)
    
    return losses.avg, r2, metrics


def train_fold_with_params(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: Dict[str, Any],
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    max_epochs: int = 40,
    patience: int = 10,
    trial: Optional[optuna.Trial] = None,
    report_epoch_offset: int = 0,
) -> Tuple[float, float, Dict[str, float]]:
    """Train a single fold with given hyperparameters."""
    
    # Extract parameters
    img_size = params["img_size"]
    dropout = params["dropout"]
    hidden_ratio = params["hidden_ratio"]
    use_film = params["use_film"]
    grid_size = params["grid_size"]
    
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    warmup_epochs = params["warmup_epochs"]
    grad_clip = params["grad_clip"]
    batch_size = params["batch_size"]
    
    use_smoothl1 = params["use_smoothl1"]
    smoothl1_beta = params["smoothl1_beta"]
    
    aug_prob = params["aug_prob"]
    strong_aug = params["strong_aug"]
    
    # Two-stage training (patience-based transition)
    stage1_patience = params["stage1_patience"]
    backbone_lr = params["backbone_lr"]
    
    grid = (grid_size, grid_size)
    
    # Transforms
    train_transform = get_train_transforms(img_size, aug_prob, strong=strong_aug)
    valid_transform = get_valid_transforms(img_size)
    
    # Datasets
    train_ds = BiomassDataset(train_df, image_dir, train_transform, is_train=True)
    valid_ds = BiomassDataset(valid_df, image_dir, valid_transform, is_train=False)
    
    num_workers = 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    # Build simple model (no depth)
    model = DINOv3Direct(
        grid=grid,
        pretrained=True,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
        use_film=use_film,
        use_attention_pool=True,   # Use attention pooling
        use_depth=False,           # Simple: no depth
        use_depth_attention=False,
        train_dead=False,          # Dead derived from Total - GDM
        train_clover=True,         # Train Clover directly
        backbone_size=BACKBONE_SIZE,
    ).to(device)
    
    # Loss
    loss_fn = SimpleLoss(use_smoothl1=use_smoothl1, smoothl1_beta=smoothl1_beta)
    
    # Stage 1: Frozen backbone
    freeze_backbone(model)
    head_params = [p for p in model.parameters() if p.requires_grad]
    
    use_fused = supports_fused_optimizer(device_type)
    optimizer = AdamW(head_params, lr=lr, weight_decay=weight_decay, fused=use_fused)
    
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    main_sched = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup_epochs])
    
    best_r2 = -float("inf")
    best_r2_stage1 = -float("inf")
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    current_stage = 1
    
    for epoch in range(1, max_epochs + 1):
        # Stage 2 transition: when stage 1 plateaus (patience reached)
        if current_stage == 1 and patience_counter >= stage1_patience:
            print(f"    >>> Stage 1 saturated at R²={best_r2_stage1:.4f}, starting Stage 2 (unfreezing backbone)")
            current_stage = 2
            unfreeze_backbone(model)
            
            # Separate param groups for backbone and heads
            backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
            head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
            
            optimizer = AdamW([
                {"params": head_params, "lr": lr * 0.5},
                {"params": backbone_params, "lr": backbone_lr},
            ], weight_decay=weight_decay, fused=use_fused)
            
            remaining = max_epochs - epoch + 1
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
            patience_counter = 0  # Reset for stage 2
            best_r2 = best_r2_stage1  # Carry over best from stage 1
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip)
        valid_loss, r2, metrics = validate(model, valid_loader, loss_fn, device)
        
        scheduler.step()
        
        # Print epoch progress
        stage_str = f"S{current_stage}"
        best_str = " *" if r2 > best_r2 else ""
        print(f"    Ep {epoch:02d} [{stage_str}] | loss: {train_loss:.4f}/{valid_loss:.4f} | "
              f"R²: {r2:.4f} | G={metrics['r2_green']:.3f} D={metrics['r2_dead']:.3f} "
              f"C={metrics['r2_clover']:.3f}{best_str}")
        
        # Report to Optuna
        if trial is not None:
            trial.report(r2, report_epoch_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Track best
        if r2 > best_r2 and not np.isnan(r2):
            best_r2 = r2
            best_metrics = metrics.copy()
            if current_stage == 1:
                best_r2_stage1 = r2
            patience_counter = 0
        else:
            patience_counter += 1
        
        if np.isnan(train_loss) or np.isnan(valid_loss):
            break
        
        # Early stopping in stage 2
        if current_stage == 2 and patience_counter >= patience:
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
    start_fold: int,
    seed: int,
):
    """Create Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed + trial.number, device_type)

        # =====================================================================
        # IMAGE SIZE
        # =====================================================================
        img_size = trial.suggest_categorical("img_size", IMG_SIZES)
        
        # =====================================================================
        # HEAD ARCHITECTURE
        # =====================================================================
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        hidden_ratio = trial.suggest_float("hidden_ratio", 0.15, 0.5)
        use_film = True  # Fixed: always use FiLM for stereo cross-conditioning
        grid_size = trial.suggest_categorical("grid_size", [2, 3])

        # =====================================================================
        # TRAINING
        # =====================================================================
        lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        weight_decay = 0.01  # Fixed
        warmup_epochs = 3  # Fixed
        grad_clip = trial.suggest_float("grad_clip", 0.3, 2.0)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        
        # Two-stage training (stage transition via patience, not fixed epochs)
        stage1_patience = 10  # Fixed: move to stage 2 after 10 epochs without improvement
        backbone_lr = trial.suggest_float("backbone_lr", 1e-6, 5e-5, log=True)

        # =====================================================================
        # LOSS
        # =====================================================================
        use_smoothl1 = trial.suggest_categorical("use_smoothl1", [True, False])
        smoothl1_beta = trial.suggest_float("smoothl1_beta", 0.05, 0.5) if use_smoothl1 else 0.1

        # =====================================================================
        # AUGMENTATION
        # =====================================================================
        aug_prob = trial.suggest_float("aug_prob", 0.3, 0.7)
        strong_aug = True  # Fixed: always use strong augmentation

        params = {
            "img_size": img_size,
            "dropout": dropout,
            "hidden_ratio": hidden_ratio,
            "use_film": use_film,
            "grid_size": grid_size,
            "lr": lr,
            "backbone_lr": backbone_lr,
            "stage1_patience": stage1_patience,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "grad_clip": grad_clip,
            "batch_size": batch_size,
            "use_smoothl1": use_smoothl1,
            "smoothl1_beta": smoothl1_beta,
            "aug_prob": aug_prob,
            "strong_aug": strong_aug,
        }
        
        # Print trial info
        print(f"\n{'='*70}")
        print(f"[Trial {trial.number}] Simple DINOv3 (no depth, with attn pool)")
        print(f"{'='*70}")
        print(f"  Image:   {img_size}×{img_size}")
        print(f"  Arch:    dropout={dropout:.3f} | hidden={hidden_ratio:.3f} | grid={grid_size} | film=True (fixed)")
        print(f"  Train:   lr={lr:.2e} | backbone_lr={backbone_lr:.2e} | stage1_patience={stage1_patience}")
        print(f"           wd={weight_decay} (fixed) | grad_clip={grad_clip:.2f} | batch={batch_size}")
        print(f"  Loss:    smoothl1={use_smoothl1} | beta={smoothl1_beta:.3f}")
        print(f"  Aug:     prob={aug_prob:.2f} | strong=True (fixed)")
        print(f"{'-'*70}")
        
        # Cross-validation
        fold_scores = []
        fold_details = []
        folds_to_eval = list(range(start_fold, start_fold + n_folds_eval))
        
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
                
                per_target = " | ".join([
                    f"G={metrics.get('r2_green', 0):.3f}",
                    f"D={metrics.get('r2_dead', 0):.3f}",
                    f"C={metrics.get('r2_clover', 0):.3f}",
                ])
                print(f"  Fold {fold}: R²={r2:.4f} [{per_target}]")
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"  Fold {fold} failed: {e}")
                return -1.0
        
        cv_score = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        
        print(f"  CV Score: {cv_score:.4f} ± {cv_std:.4f}")
        
        return cv_score
    
    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna Search for Simple DINOv3")

    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--fold-csv", type=str, default="data/trainfold_group_month_species.csv")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n-folds-eval", type=int, default=5,
                        help="Number of folds to evaluate (default: 5 for robust params)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Starting fold index (default: 0)")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    # Setup
    device_type = get_device_type()
    device = get_device(device_type)
    print(f"Device: {device} ({device_type.value})")
    
    # Load data
    df = pd.read_csv(args.fold_csv)
    image_dir = os.path.join(args.base_path, "train")
    
    print(f"Loaded {len(df)} samples with {df['fold'].nunique()} folds")
    print(f"Evaluating fold(s): {list(range(args.fold, args.fold + args.n_folds_eval))}")
    
    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./outputs/optuna_simple_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Study
    study_name = args.study_name or f"simple_dinov3_{datetime.now().strftime('%Y%m%d')}"
    
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    
    objective = create_objective(
        df=df,
        image_dir=image_dir,
        device=device,
        device_type=device_type,
        max_epochs=args.max_epochs,
        patience=args.patience,
        n_folds_eval=args.n_folds_eval,
        start_fold=args.fold,
        seed=args.seed,
    )
    
    print(f"\nStarting optimization: {args.n_trials} trials")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    study.optimize(objective, n_trials=args.n_trials)
    
    # Save results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best CV R²: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best params
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "fold_csv": args.fold_csv,
    }
    
    results_path = os.path.join(args.output_dir, "best_params.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate training command
    bp = study.best_params
    cmd = f"""
# Train with best params:
torchrun --nproc_per_node=2 -m src.dinov3_train \\
    --no-use-depth \\
    --img-size {bp['img_size']} \\
    --dropout {bp['dropout']:.3f} \\
    --hidden-ratio {bp['hidden_ratio']:.3f} \\
    --grid {bp['grid_size']} \\
    --use-film \\
    --lr {bp['lr']:.6f} \\
    --backbone-lr {bp['backbone_lr']:.6f} \\
    --stage1-patience 10 \\
    --weight-decay 0.01 \\
    --grad-clip {bp['grad_clip']:.2f} \\
    --batch-size {bp['batch_size']} \\
    --aug-prob {bp['aug_prob']:.2f} \\
    --strong-aug \\
    {'--smoothl1' if bp['use_smoothl1'] else '--no-smoothl1'} \\
    --output-dir ./outputs/dinov3_simple_best
"""
    print(cmd)
    
    # Save command
    cmd_path = os.path.join(args.output_dir, "train_command.sh")
    with open(cmd_path, "w") as f:
        f.write(cmd)


if __name__ == "__main__":
    main()

