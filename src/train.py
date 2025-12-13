#!/usr/bin/env python3
"""
Main training script for CSIRO Biomass prediction with multi-GPU/MPS support.

Usage (single GPU or MPS):
    python -m src.train

Usage (multi-GPU with torchrun - CUDA only):
    torchrun --nproc_per_node=2 -m src.train

Usage (multi-GPU with accelerate - CUDA only):
    accelerate launch --num_processes=2 -m src.train

Device selection:
    python -m src.train --device-type cuda    # NVIDIA GPU
    python -m src.train --device-type mps     # Apple Silicon
    python -m src.train --device-type cpu     # CPU fallback
"""
import argparse
import gc
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from .config import TrainConfig
from .dataset import create_folds, prepare_dataframe
from .device import (
    DeviceType,
    empty_cache,
    get_cudnn_settings,
    get_device,
    get_device_type,
    set_device_seed,
)
from .trainer import Trainer, cleanup_distributed, is_main_process, setup_distributed


def set_seed(seed: int, rank: int = 0, device_type: DeviceType = DeviceType.CUDA) -> None:
    """Set random seeds for reproducibility."""
    seed = seed + rank  # Different seed per process
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Device-specific seeding
    set_device_seed(seed, device_type)
    
    # CUDNN settings (only for CUDA)
    deterministic, benchmark = get_cudnn_settings(device_type)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CSIRO Biomass Model (Multi-GPU)")
    
    # Paths
    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default=None)
    
    # Model
    parser.add_argument(
        "--backbone",
        type=str,
        default="vit_base_patch14_reg4_dinov2.lvd142m",
        help="Backbone model name from timm",
    )
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--hidden-ratio", type=float, default=0.25)
    parser.add_argument("--grid", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    
    # Optimizer
    parser.add_argument("--lr", type=float, default=2e-4, help="Head learning rate for stage 2")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Backbone learning rate for stage 2")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    
    # 2-Stage Training
    parser.add_argument("--two-stage", action="store_true", default=True, help="Enable 2-stage training")
    parser.add_argument("--no-two-stage", action="store_false", dest="two_stage", help="Disable 2-stage training")
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Stage 1: epochs with frozen backbone")
    parser.add_argument("--head-lr-stage1", type=float, default=1e-3, help="Stage 1: head learning rate")
    
    # Scheduler
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "step", "plateau"]
    )
    parser.add_argument("--warmup-epochs", type=int, default=2)
    
    # Loss
    parser.add_argument(
        "--loss", type=str, default="mse", choices=["mse", "mae", "huber", "rmse"]
    )
    
    # Augmentation
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--aug-prob", type=float, default=0.5)
    
    # K-fold
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--train-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
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
    
    # Device configuration
    parser.add_argument(
        "--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help="Device type: cuda (NVIDIA GPU), mps (Apple Silicon), cpu. Auto-detected if not specified."
    )
    
    # Multi-GPU (CUDA only)
    parser.add_argument("--no-distributed", action="store_true", help="Disable DDP (CUDA multi-GPU)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (disabled by default)")
    
    # AMP
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"],
        help="AMP dtype (bfloat16 recommended for H200)"
    )
    
    # Best model selection
    parser.add_argument(
        "--best-metric", type=str, default="loss", choices=["loss", "r2"],
        help="Metric for best model selection: 'loss' (lower better) or 'r2' (higher better)"
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--patience", type=int, default=10)
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Create config
    cfg = TrainConfig()
    cfg.base_path = args.base_path
    cfg.train_csv = os.path.join(args.base_path, "train.csv")
    cfg.train_image_dir = os.path.join(args.base_path, "train")
    
    # Set device type (auto-detect if not specified)
    cfg.device_type = args.device_type
    device_type_enum = DeviceType(args.device_type) if args.device_type else get_device_type()
    cfg._device_type_enum = device_type_enum
    
    # Check if distributed (CUDA only, requires multiple GPUs)
    cfg.distributed = (
        not args.no_distributed
        and device_type_enum == DeviceType.CUDA
        and torch.cuda.device_count() > 1
    )
    
    # Setup distributed training FIRST (CUDA only)
    if cfg.distributed:
        setup_distributed(cfg)
    else:
        cfg.local_rank = 0
        cfg.world_size = 1
        cfg.device = get_device(device_type_enum, cfg.local_rank)
    
    # Output directory
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = os.path.join(args.base_path, "outputs", timestamp)
    
    if is_main_process(cfg):
        os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Model config
    cfg.backbone_name = args.backbone
    cfg.dropout = args.dropout
    cfg.hidden_ratio = args.hidden_ratio
    cfg.grid = tuple(args.grid)
    cfg.gradient_checkpointing = args.grad_ckpt
    
    # Training config
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.gradient_accumulation_steps = args.grad_accum
    
    # Optimizer config (scale LR by world_size for linear scaling rule)
    base_lr = args.lr
    base_backbone_lr = args.backbone_lr
    if cfg.distributed and cfg.world_size > 1:
        # Linear scaling rule
        cfg.lr = base_lr * cfg.world_size
        cfg.backbone_lr = base_backbone_lr * cfg.world_size
    else:
        cfg.lr = base_lr
        cfg.backbone_lr = base_backbone_lr
    
    cfg.weight_decay = args.weight_decay
    cfg.warmup_epochs = args.warmup_epochs
    
    # 2-Stage Training config
    cfg.two_stage = args.two_stage
    cfg.freeze_epochs = args.freeze_epochs
    cfg.head_lr_stage1 = args.head_lr_stage1
    
    # Scheduler config
    cfg.scheduler = args.scheduler
    
    # Loss config
    cfg.loss_type = args.loss
    
    # Augmentation config
    cfg.img_size = args.img_size
    cfg.aug_prob = args.aug_prob
    
    # K-fold config
    cfg.num_folds = args.num_folds
    cfg.train_folds = args.train_folds
    cfg.cv_strategy = args.cv_strategy
    
    # Multi-GPU config
    cfg.compile_model = args.compile
    
    # AMP config
    cfg.use_amp = not args.no_amp
    cfg.amp_dtype = args.amp_dtype
    
    # Best model selection
    cfg.best_metric = args.best_metric
    
    # Misc config
    cfg.seed = args.seed
    cfg.patience = args.patience
    
    # Set seed with rank offset
    set_seed(cfg.seed, cfg.local_rank, device_type_enum)
    
    # Print config (main process only)
    if is_main_process(cfg):
        print("=" * 60)
        print("CSIRO Biomass Training (Multi-GPU/MPS Optimized)")
        print("=" * 60)
        print(f"Device type: {device_type_enum.value}")
        print(f"Device: {cfg.device}")
        if device_type_enum == DeviceType.CUDA:
            print(f"World size: {cfg.world_size} GPUs")
            print(f"Distributed: {cfg.distributed}")
        elif device_type_enum == DeviceType.MPS:
            print("MPS (Apple Silicon) backend active")
        print(f"Output dir: {cfg.output_dir}")
        print(f"Backbone: {cfg.backbone_name}")
        print(f"Grid: {cfg.grid}")
        print(f"Gradient checkpointing: {cfg.gradient_checkpointing}")
        print(f"Epochs: {cfg.epochs}")
        print(f"Per-GPU batch size: {cfg.batch_size}")
        print(f"Effective batch size: {cfg.batch_size * cfg.world_size * cfg.gradient_accumulation_steps}")
        print(f"2-Stage Training: {cfg.two_stage}")
        if cfg.two_stage:
            print(f"  Stage 1: {cfg.freeze_epochs} epochs, head LR: {cfg.head_lr_stage1:.2e} (backbone frozen)")
            print(f"  Stage 2: remaining epochs, head LR: {cfg.lr:.2e}, backbone LR: {cfg.backbone_lr:.2e}")
        else:
            print(f"Learning rate: {cfg.lr:.2e} (backbone: {cfg.backbone_lr:.2e})")
        print(f"Loss: {cfg.loss_type}")
        print(f"AMP: {cfg.use_amp} ({cfg.amp_dtype})")
        print(f"Compile: {cfg.compile_model}")
        print(f"Best metric: {cfg.best_metric}")
        print(f"Patience: {cfg.patience}")
        print(f"CV strategy: {cfg.cv_strategy}")
        print(f"Folds to train: {cfg.train_folds}")
        print("=" * 60)
    
    # Prepare data
    if is_main_process(cfg):
        print("\nPreparing data...")
    
    df = prepare_dataframe(cfg.train_csv)
    df = create_folds(df, n_folds=cfg.num_folds, seed=cfg.seed, cv_strategy=cfg.cv_strategy)
    
    if is_main_process(cfg):
        print(f"Total samples: {len(df)}")
        print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")
        # Save fold info
        df.to_csv(os.path.join(cfg.output_dir, "folds.csv"), index=False)
    
    # Synchronize before training
    if cfg.distributed:
        torch.distributed.barrier()
    
    # Train
    trainer = Trainer(cfg)
    results = trainer.train_all_folds(df)
    
    # Save results (main process only)
    if is_main_process(cfg):
        import json
        import time
        
        r2_scores = [r["best_r2"] for r in results]
        
        results_summary = {
            "experiment": {
                "timestamp": int(time.time()),
                "timestamp_human": datetime.now().isoformat(),
                "output_dir": cfg.output_dir,
            },
            "config": {
                "backbone": cfg.backbone_name,
                "model_type": "tiled_film",
                "grid": list(cfg.grid),
                "img_size": cfg.img_size,
                "epochs": cfg.epochs,
                "patience": cfg.patience,
                "best_metric": cfg.best_metric,
                "cv_strategy": cfg.cv_strategy,
                "batch_size": cfg.batch_size,
                "effective_batch_size": cfg.batch_size * cfg.world_size * cfg.gradient_accumulation_steps,
                "world_size": cfg.world_size,
                "lr": float(cfg.lr),
                "backbone_lr": float(cfg.backbone_lr),
                "loss": "weighted_mse",
                "target_weights": list(cfg.target_weights),
                "dropout": float(cfg.dropout),
                "hidden_ratio": float(cfg.hidden_ratio),
                "amp_dtype": cfg.amp_dtype,
                "compile": cfg.compile_model,
                "gradient_checkpointing": cfg.gradient_checkpointing,
            },
            "fold_results": [
                {
                    "fold": int(r["fold"]),
                    "best_r2": float(r["best_r2"]),
                    "best_loss": float(r["best_loss"]),
                    "best_metrics": r.get("best_metrics", {}),
                    "best_epoch": int(np.argmin(r["history"]["valid_loss"]) + 1) if cfg.best_metric == "loss" else int(np.argmax(r["history"]["weighted_r2"]) + 1) if r["history"]["weighted_r2"] else 0,
                    "final_train_loss": float(r["history"]["train_loss"][-1]) if r["history"]["train_loss"] else 0,
                    "final_valid_loss": float(r["history"]["valid_loss"][-1]) if r["history"]["valid_loss"] else 0,
                }
                for r in results
            ],
            "summary": {
                "best_metric_used": cfg.best_metric,
                "cv_mean_r2": float(np.mean(r2_scores)),
                "cv_std_r2": float(np.std(r2_scores)),
                "cv_min_r2": float(np.min(r2_scores)),
                "cv_max_r2": float(np.max(r2_scores)),
                "cv_mean_loss": float(np.mean([r["best_loss"] for r in results])),
                "cv_std_loss": float(np.std([r["best_loss"] for r in results])),
                "num_folds": len(results),
            },
            "checkpoints": [
                f"tiled_film_best_model_fold{r['fold']}.pth" for r in results
            ],
        }
        
        with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nResults saved to {cfg.output_dir}")
    
    # Cleanup
    cleanup_distributed()
    gc.collect()
    empty_cache(device_type_enum)


if __name__ == "__main__":
    main()
