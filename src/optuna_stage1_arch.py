#!/usr/bin/env python3
"""
Stage 1: Architecture Tuning Only (5-10 trials)

Only tunes:
- grid_size, use_film, use_attention_pool, dropout, hidden_ratio, use_vegetation_indices

Fixed:
- lr=1e-4, wd=0.01, batch=8, training_mode=freeze
- aug_prob=0.5, no mixup, no stereo_correct
- huber_dead=True, huber_delta=5.0, log_target=False
"""
import argparse
import gc
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import optuna
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
from .trainer import compute_weighted_r2, compute_per_target_metrics_np


BACKBONE = "vit_base_patch16_dinov3"
IMG_SIZE = 576

# ========================================
# FIXED PARAMS FOR STAGE 1
# ========================================
FIXED_PARAMS = {
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 2,
    "grad_clip": 1.0,
    "batch_size": 8,
    "aug_prob": 0.5,
    "use_huber_for_dead": True,
    "huber_delta": 5.0,
}

# Will be set via args
TRAIN_DEAD = False
TRAIN_CLOVER = False


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
    
    r2 = compute_weighted_r2(all_preds, all_targets, target_weights)
    target_names = ["green", "dead", "clover", "gdm", "total"]
    per_target = compute_per_target_metrics_np(all_preds, all_targets, target_names, target_weights)
    
    metrics = {"weighted_r2": r2}
    for row in per_target:
        metrics[f"r2_{row['target']}"] = float(row["r2"])
    
    return losses.avg, r2, metrics


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    arch_params: Dict[str, Any],
    image_dir: str,
    device: torch.device,
    device_type: DeviceType,
    max_epochs: int = 25,
    patience: int = 6,
) -> Tuple[float, Dict[str, float]]:
    """Train single fold with architecture params (fixed training params)."""
    
    # Architecture params (tuned)
    dropout = arch_params["dropout"]
    hidden_ratio = arch_params["hidden_ratio"]
    use_film = arch_params["use_film"]
    use_attention_pool = arch_params["use_attention_pool"]
    grid_size = arch_params["grid_size"]
    use_vegetation_indices = arch_params["use_vegetation_indices"]
    
    # Fixed params
    lr = FIXED_PARAMS["lr"]
    weight_decay = FIXED_PARAMS["weight_decay"]
    warmup_epochs = FIXED_PARAMS["warmup_epochs"]
    grad_clip = FIXED_PARAMS["grad_clip"]
    batch_size = FIXED_PARAMS["batch_size"]
    aug_prob = FIXED_PARAMS["aug_prob"]
    
    grid = (grid_size, grid_size)
    
    # Transforms (simple, fixed augmentation)
    train_transform = get_train_transforms(IMG_SIZE, aug_prob)
    valid_transform = get_valid_transforms(IMG_SIZE)
    
    # Datasets (no mixup in stage 1)
    train_ds = BiomassDataset(
        train_df, image_dir, train_transform,
        is_train=True, use_log_target=False,
    )
    valid_ds = BiomassDataset(
        valid_df, image_dir, valid_transform,
        is_train=False, use_log_target=False,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=4,
    )
    
    # Build model
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
    
    # Freeze backbone (head-only training)
    freeze_backbone(model)
    
    # Loss
    loss_fn = RatioMSELoss(
        target_weights=[0.1, 0.1, 0.1, 0.2, 0.5],
        use_huber_for_dead=True,
        huber_delta=5.0,
    )
    
    # Optimizer & scheduler
    use_fused = supports_fused_optimizer(device_type)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, fused=use_fused)
    
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    
    # Training
    best_r2 = -float("inf")
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip)
        valid_loss, r2, metrics = validate(model, valid_loader, loss_fn, device)
        scheduler.step()
        
        improved = ""
        if r2 > best_r2 and not np.isnan(r2):
            best_r2 = r2
            best_metrics = metrics.copy()
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1
        
        # Print epoch progress
        print(f"    Ep {epoch:02d}: train={train_loss:.4f} val={valid_loss:.4f} R²={r2:.4f} (best={best_r2:.4f}){improved}", flush=True)
        
        if np.isnan(train_loss) or np.isnan(valid_loss):
            print("    NaN detected, stopping early", flush=True)
            break
        if patience_counter >= patience:
            print(f"    Early stop (no improvement for {patience} epochs)", flush=True)
            break
    
    del model, optimizer, scheduler
    empty_cache(device_type)
    gc.collect()
    
    return best_r2, best_metrics


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
    """Objective: tune ONLY architecture params."""

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed + trial.number, device_type)

        # =====================================================================
        # ARCHITECTURE PARAMS ONLY
        # =====================================================================
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        hidden_ratio = trial.suggest_float("hidden_ratio", 0.15, 0.5)
        use_film = trial.suggest_categorical("use_film", [True, False])
        use_attention_pool = trial.suggest_categorical("use_attention_pool", [True, False])
        grid_size = trial.suggest_categorical("grid_size", [2, 3])
        use_vegetation_indices = trial.suggest_categorical("use_vegetation_indices", [True, False])

        arch_params = {
            "dropout": dropout,
            "hidden_ratio": hidden_ratio,
            "use_film": use_film,
            "use_attention_pool": use_attention_pool,
            "grid_size": grid_size,
            "use_vegetation_indices": use_vegetation_indices,
        }
        
        print(f"\n{'='*60}")
        print(f"[Trial {trial.number}] Architecture Search")
        print(f"{'='*60}")
        print(f"  dropout={dropout:.3f} | hidden={hidden_ratio:.3f} | grid={grid_size}")
        print(f"  film={use_film} | attn_pool={use_attention_pool} | veg_idx={use_vegetation_indices}")
        print(f"  (fixed: lr=1e-4, wd=0.01, batch=8, aug=0.5, no mixup)")
        print(f"{'-'*60}")
        
        # Cross-validation
        fold_scores = []
        
        for fold in range(n_folds_eval):
            train_df = df[df["fold"] != fold].reset_index(drop=True)
            valid_df = df[df["fold"] == fold].reset_index(drop=True)
            
            try:
                r2, metrics = train_fold(
                    fold=fold,
                    train_df=train_df,
                    valid_df=valid_df,
                    arch_params=arch_params,
                    image_dir=image_dir,
                    device=device,
                    device_type=device_type,
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
                print(f"  Fold {fold}: R²={r2:.4f} [{per_target}]")
                
            except Exception as e:
                print(f"  Fold {fold} failed: {e}")
                return -1.0
        
        cv_score = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        print(f"  CV: {cv_score:.4f} ± {cv_std:.4f}")
        
        return cv_score
    
    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Architecture Tuning")
    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--n-folds-eval", type=int, default=3)
    parser.add_argument("--cv-strategy", type=str, default="group_date_state")
    parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    device_type = DeviceType(args.device_type) if args.device_type else get_device_type()
    device = get_device(device_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"./outputs/stage1_arch_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    storage = f"sqlite:///{args.output_dir}/optuna.db"

    print("=" * 60)
    print("STAGE 1: Architecture Tuning")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Tuning: dropout, hidden_ratio, grid, film, attn_pool, veg_idx")
    print(f"Fixed:  lr=1e-4, wd=0.01, batch=8, aug=0.5, no mixup")
    print(f"Trials: {args.n_trials}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load data
    train_csv = os.path.join(args.base_path, "train.csv")
    image_dir = os.path.join(args.base_path, "train")
    df = prepare_dataframe(train_csv)
    df = create_folds(df, n_folds=5, seed=args.seed, cv_strategy=args.cv_strategy)

    print(f"Samples: {len(df)}")

    # Create study
    sampler = TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=f"stage1_arch_{timestamp}",
        storage=storage,
        direction="maximize",
        sampler=sampler,
    )

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

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Results
    print("\n" + "=" * 60)
    print("Stage 1 Complete!")
    print("=" * 60)
    print(f"Best CV R²: {study.best_value:.4f}")
    print("\nBest architecture:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "stage": 1,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "fixed_params": FIXED_PARAMS,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print next stage command
    bp = study.best_params
    print("\n" + "=" * 60)
    print("Next: Run Stage 2 (training dynamics) with these arch params:")
    print("=" * 60)
    print(f"""
python -m src.optuna_stage2_training \\
    --dropout {bp['dropout']:.3f} \\
    --hidden-ratio {bp['hidden_ratio']:.3f} \\
    --grid-size {bp['grid_size']} \\
    {'--use-film' if bp['use_film'] else '--no-film'} \\
    {'--use-attention-pool' if bp['use_attention_pool'] else '--no-attention-pool'} \\
    {'--use-vegetation-indices' if bp['use_vegetation_indices'] else ''} \\
    --n-trials 10 \\
    --device-type {args.device_type or 'mps'}
""")


if __name__ == "__main__":
    main()

