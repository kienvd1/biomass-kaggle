"""
Training script for DINOv2 Flora model.

Based on PlantHydra (1st place PlantTraits2024):
- 2-stage training: head-only → full fine-tuning
- Layer-wise learning rate decay
- R² + Cosine similarity loss
- Multi-task learning with species classification

Usage:
    python src/dinov2_flora_train.py --folds 0 1 2 3 4 --epochs 30 --batch-size 8
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from dataset import (
    BiomassDataset,
    create_folds,
    get_train_transforms,
    get_stereo_geometric_transforms,
    get_stereo_photometric_transforms,
    get_valid_transforms,
    prepare_dataframe,
)
from device import get_device
from dinov2_flora_models import (
    DINOv2Flora,
    FloraLoss,
    compute_species_prior,
    count_parameters,
    freeze_backbone,
    unfreeze_backbone,
)


# Competition weights
COMPETITION_WEIGHTS = {
    "green": 0.1,
    "dead": 0.1,
    "clover": 0.1,
    "gdm": 0.2,
    "total": 0.5,
}
TARGET_NAMES = ["green", "dead", "clover", "gdm", "total"]


def create_layer_wise_optimizer_scheduler(
    model: nn.Module,
    head_lr: float,
    backbone_lr: float,
    lr_mult: float,
    weight_decay: float,
    total_steps: int,
    warmup_pct: float = 0.2,
) -> Tuple[List[torch.optim.Optimizer], List[OneCycleLR]]:
    """
    Create layer-wise optimizers and schedulers like PlantHydra.
    
    Each transformer block gets its own optimizer with decayed learning rate:
    - Head (highest LR): head_lr
    - Later blocks: backbone_lr
    - Earlier blocks: backbone_lr * (lr_mult ^ depth)
    - Tokens: backbone_lr * (lr_mult ^ num_layers)
    
    Args:
        model: DINOv2Flora model
        head_lr: Learning rate for head/tabular/attention pool
        backbone_lr: Base learning rate for backbone (last block)
        lr_mult: Decay multiplier per layer (0.8 means each earlier layer gets 0.8x LR)
        weight_decay: Weight decay
        total_steps: Total training steps
        warmup_pct: Warmup percentage for OneCycleLR
    
    Returns:
        optimizers: List of optimizers (one per parameter group)
        schedulers: List of schedulers (one per optimizer)
    """
    optimizers = []
    schedulers = []
    
    # 1. Head parameters (regression, classification, tabular, attention pool)
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and "body" not in name:
            head_params.append(param)
    
    # Add attention pool from body
    if hasattr(model.body, 'attn_pool'):
        for param in model.body.attn_pool.parameters():
            if param.requires_grad:
                head_params.append(param)
    
    if head_params:
        opt = AdamW(head_params, lr=head_lr, weight_decay=weight_decay)
        sched = OneCycleLR(
            opt, max_lr=head_lr, total_steps=total_steps,
            pct_start=warmup_pct, anneal_strategy='cos',
            div_factor=10, final_div_factor=10,
        )
        optimizers.append(opt)
        schedulers.append(sched)
        print(f"  Head optimizer: {len(head_params)} params, lr={head_lr:.2e}")
    
    # 2. Transformer blocks (layer-wise decay)
    num_blocks = len(model.body.blocks)
    trainable_blocks = []
    
    for i, block in enumerate(model.body.blocks):
        block_params = [p for p in block.parameters() if p.requires_grad]
        if block_params:
            trainable_blocks.append((i, block_params))
    
    for block_idx, block_params in trainable_blocks:
        # Later blocks get higher LR, earlier blocks get lower LR
        # Layer i gets: backbone_lr * (lr_mult ^ (num_blocks - 1 - block_idx))
        layer_depth = num_blocks - 1 - block_idx
        layer_lr = backbone_lr * (lr_mult ** layer_depth)
        
        opt = AdamW(block_params, lr=backbone_lr, weight_decay=weight_decay)
        sched = OneCycleLR(
            opt, max_lr=layer_lr, total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos',
            div_factor=1e7, final_div_factor=10,
        )
        optimizers.append(opt)
        schedulers.append(sched)
        print(f"  Block {block_idx} optimizer: {len(block_params)} params, max_lr={layer_lr:.2e}")
    
    # 3. Token parameters (cls_token, pos_embed, reg_token) - lowest LR
    token_params = []
    token_names = ['cls_token', 'pos_embed', 'reg_token']
    for name in token_names:
        if hasattr(model.body, name):
            param = getattr(model.body, name)
            if param.requires_grad:
                token_params.append(param)
    
    if token_params:
        # Tokens get even lower LR (decay from layer 0)
        token_lr = backbone_lr * (lr_mult ** num_blocks)
        opt = AdamW(token_params, lr=backbone_lr, weight_decay=weight_decay)
        sched = OneCycleLR(
            opt, max_lr=token_lr, total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos',
            div_factor=1e7, final_div_factor=10,
        )
        optimizers.append(opt)
        schedulers.append(sched)
        print(f"  Tokens optimizer: {len(token_params)} params, max_lr={token_lr:.2e}")
    
    return optimizers, schedulers


def create_simple_optimizer_scheduler(
    model: nn.Module,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_pct: float = 0.2,
    train_backbone: bool = True,
) -> Tuple[torch.optim.Optimizer, OneCycleLR]:
    """
    Create a simple optimizer with 2 param groups (head + backbone).
    
    Used for Stage 1 (head-only) or simpler training.
    """
    param_groups = []
    
    # Head parameters
    head_params = [p for n, p in model.named_parameters() 
                   if p.requires_grad and "body" not in n]
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    
    # Backbone parameters (if training)
    if train_backbone:
        backbone_params = [p for n, p in model.named_parameters() 
                          if p.requires_grad and "body" in n]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
    
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    
    max_lrs = [pg["lr"] for pg in param_groups]
    scheduler = OneCycleLR(
        optimizer, max_lr=max_lrs, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy='cos',
    )
    
    return optimizer, scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv2 Flora model")
    
    # Data
    parser.add_argument("--train-csv", type=str, default="data/train.csv")
    parser.add_argument("--image-dir", type=str, default="data/train")
    parser.add_argument("--output-dir", type=str, default="outputs/dinov2_flora")
    parser.add_argument("--fold-csv", type=str, default="data/default_folds.csv",
                        help="Path to predefined folds CSV (with sample_id_prefix, fold columns)")
    
    # Model
    parser.add_argument("--backbone", type=str, default="vitb",
                        choices=["vitb", "vitl", "vitg"])
    parser.add_argument("--train-blocks", type=int, default=4,
                        help="Number of transformer blocks to fine-tune")
    parser.add_argument("--train-tokens", action="store_true",
                        help="Fine-tune cls/pos/reg tokens")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Path to pretrained weights (.safetensors for PlantCLEF, .pth/.tar for others)")
    parser.add_argument("--use-clf-head", action="store_true", default=True,
                        help="Enable species classification head")
    parser.add_argument("--use-soft-clf", action="store_true",
                        help="Enable soft classification (species probs × prior)")
    parser.add_argument("--use-blending", action="store_true",
                        help="Enable learnable blending of outputs")
    parser.add_argument("--grid", type=int, nargs=2, default=[2, 2],
                        help="Tile grid for stereo images")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    
    # Training
    parser.add_argument("--folds", type=int, nargs="+", default=[0],
                        help="Folds to train")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cv-strategy", type=str, default="group_date_state",
                        choices=["group_month", "group_date", "group_date_state",
                                "group_date_state_bin", "stratified", "random"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate for heads")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="Peak learning rate for backbone (base, decayed per layer)")
    parser.add_argument("--lr-mult", type=float, default=0.8,
                        help="Layer-wise LR decay multiplier (earlier layers get lr * mult^depth)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-pct", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=18)
    
    # Two-stage training
    parser.add_argument("--two-stage", action="store_true",
                        help="Two-stage training: Stage 1 then Stage 2 with layer-wise LR")
    parser.add_argument("--stage1-full", action="store_true", default=True,
                        help="Stage 1: full fine-tuning (default). Use --stage1-head-only for head-only")
    parser.add_argument("--stage1-head-only", action="store_false", dest="stage1_full",
                        help="Stage 1: head-only training (freeze backbone)")
    parser.add_argument("--stage1-epochs", type=int, default=50,
                        help="Max epochs for Stage 1")
    parser.add_argument("--stage1-patience", type=int, default=10,
                        help="Early stopping patience for Stage 1")
    parser.add_argument("--stage2-epochs", type=int, default=30,
                        help="Max epochs for Stage 2 (with layer-wise LR)")
    parser.add_argument("--stage2-patience", type=int, default=10,
                        help="Early stopping patience for Stage 2")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (single-stage mode)")
    parser.add_argument("--layer-wise-lr", action="store_true", default=True,
                        help="Use layer-wise learning rate decay in Stage 2 (default: True)")
    parser.add_argument("--no-layer-wise-lr", action="store_false", dest="layer_wise_lr")
    
    # Augmentation
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--aug-prob", type=float, default=0.5)
    parser.add_argument("--strong-aug", action="store_true",
                        help="Use strong augmentations (ColorJitter, CLAHE, MotionBlur, etc.)")
    parser.add_argument("--photometric", type=str, default="same",
                        choices=["same", "independent", "left", "right", "none"],
                        help="Photometric transform mode: "
                             "'same' = identical to both L/R (default), "
                             "'independent' = different to L/R, "
                             "'left' = only to left, "
                             "'right' = only to right, "
                             "'none' = no photometric transforms")
    parser.add_argument("--stereo-swap-prob", type=float, default=0.3,
                        help="Probability of swapping L/R views")
    parser.add_argument("--mixup-prob", type=float, default=0.0,
                        help="MixUp probability (0 to disable)")
    parser.add_argument("--mixup-alpha", type=float, default=0.4,
                        help="MixUp beta distribution alpha")
    parser.add_argument("--cutmix-prob", type=float, default=0.0,
                        help="CutMix probability (0 to disable)")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                        help="CutMix beta distribution alpha")
    parser.add_argument("--no-mix-same-context", action="store_true",
                        help="Allow MixUp/CutMix with ANY sample (not just same species/month/state)")
    
    # Loss
    parser.add_argument("--r2-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.4)
    parser.add_argument("--clf-weight", type=float, default=0.01)
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--use-log-scale", action="store_true",
                        help="Use log-scale target normalization (default: raw-scale)")
    
    # Hardware
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_competition_r2(
    pred: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute competition R² metric.
    
    Returns:
        weighted_r2: Weighted average R² (competition metric)
        per_target: Dict of R² per target
    """
    per_target = {}
    for i, name in enumerate(TARGET_NAMES):
        r2 = r2_score(target[:, i], pred[:, i])
        per_target[name] = r2
    
    weighted_r2 = sum(
        per_target[name] * COMPETITION_WEIGHTS[name]
        for name in TARGET_NAMES
    )
    
    return weighted_r2, per_target


def train_epoch(
    model: DINOv2Flora,
    loader: DataLoader,
    criterion: FloraLoss,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List,
    device: torch.device,
    scaler: Optional[GradScaler],
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch with multiple optimizers (layer-wise LR).
    
    Args:
        optimizers: List of optimizers (one per param group for layer-wise LR)
        schedulers: List of schedulers (one per optimizer)
    """
    model.train()
    
    # MPS optimizations
    use_channels_last = device.type == "mps"
    
    total_loss = 0.0
    loss_components = {k: 0.0 for k in ["r2", "cosine", "mse", "clf"]}
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Train epoch {epoch}")
    for batch in pbar:
        # Unpack batch
        left, right, targets = batch[:3]
        if len(batch) > 3:
            state, month, species, ndvi, height = batch[3:8]
        else:
            state = month = species = ndvi = height = None
        
        # Use channels_last for MPS performance + non_blocking transfers
        if use_channels_last:
            left = left.to(device, non_blocking=True, memory_format=torch.channels_last)
            right = right.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if state is not None:
            state = state.to(device, non_blocking=True)
            month = month.to(device, non_blocking=True)
            species = species.to(device, non_blocking=True)
            ndvi = ndvi.to(device, non_blocking=True)
            height = height.to(device, non_blocking=True)
        
        # Zero gradients for all optimizers (set_to_none for memory efficiency)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        
        # Forward pass (autocast only for CUDA with scaler)
        with autocast(device_type="cuda", enabled=scaler is not None):
            outputs = model(
                left, right,
                state=state, month=month, species=species,
                ndvi=ndvi, height=height,
                return_encoded=True,
            )
            
            # Encode targets
            target_enc = model.label_encoder.transform(targets)
            
            # Compute loss
            loss, loss_dict = criterion(
                pred_enc=outputs["pred_enc"],
                target_enc=target_enc,
                pred_raw=outputs.get("pred"),
                target_raw=targets,
                species_logits=outputs.get("species_logits"),
                species_labels=species,
                blended=outputs.get("blended"),
            )
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()
        
        # Step all schedulers
        for sched in schedulers:
            if sched is not None:
                sched.step()
        
        # Track losses
        total_loss += loss.item()
        for k in loss_components:
            if k in loss_dict:
                loss_components[k] += loss_dict[k]
        num_batches += 1
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Sync MPS before returning to prevent delay in next phase
    if device.type == "mps":
        torch.mps.synchronize()
    
    # Average losses
    metrics = {"loss": total_loss / num_batches}
    for k, v in loss_components.items():
        metrics[k] = v / num_batches
    
    return metrics


@torch.no_grad()
def validate(
    model: DINOv2Flora,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Validate model and return predictions."""
    model.eval()
    
    # MPS optimizations
    use_channels_last = device.type == "mps"
    
    all_preds = []
    all_targets = []
    
    for batch in tqdm(loader, desc="Validation"):
        left, right, targets = batch[:3]
        if len(batch) > 3:
            state, month, species, ndvi, height = batch[3:8]
        else:
            state = month = species = ndvi = height = None
        
        # Use channels_last for MPS performance + non_blocking transfers
        if use_channels_last:
            left = left.to(device, non_blocking=True, memory_format=torch.channels_last)
            right = right.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
        
        if state is not None:
            state = state.to(device, non_blocking=True)
            month = month.to(device, non_blocking=True)
            species = species.to(device, non_blocking=True)
            ndvi = ndvi.to(device, non_blocking=True)
            height = height.to(device, non_blocking=True)
        
        outputs = model(
            left, right,
            state=state, month=month, species=species,
            ndvi=ndvi, height=height,
        )
        
        # Use blended if available, otherwise raw predictions
        pred = outputs.get("blended", outputs.get("pred"))
        
        # Detach and move to CPU immediately to free GPU memory
        all_preds.append(pred.detach().cpu().numpy())
        all_targets.append(targets.numpy())
    
    # Sync MPS before returning
    if device.type == "mps":
        torch.mps.synchronize()
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    weighted_r2, per_target = compute_competition_r2(preds, targets)
    
    metrics = {"weighted_r2": weighted_r2}
    for name, r2 in per_target.items():
        metrics[f"r2_{name}"] = r2
    
    return metrics, preds, targets


def train_fold(
    fold: int,
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")
    
    # Split data
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create transforms based on photometric mode
    photometric_mode = args.photometric
    strong_aug = args.strong_aug
    
    if strong_aug:
        print(f"Augmentation: STRONG (ColorJitter, CLAHE, MotionBlur, etc.)")
    
    if photometric_mode == "same":
        # Same geometric + photometric to both L/R (via replay)
        train_transform = get_train_transforms(args.img_size, args.aug_prob, strong=strong_aug)
        photo_transform = None
        photometric_left_only = False
        photometric_right_only = False
        print("Photometric: SAME to both L/R")
    elif photometric_mode == "independent":
        # Geometric same, photometric independent
        train_transform = get_stereo_geometric_transforms(args.img_size, args.aug_prob, strong=strong_aug)
        photo_transform = get_stereo_photometric_transforms(args.aug_prob, strong=strong_aug)
        photometric_left_only = False
        photometric_right_only = False
        print("Photometric: INDEPENDENT to L/R")
    elif photometric_mode == "left":
        # Only apply photometric to left
        train_transform = get_stereo_geometric_transforms(args.img_size, args.aug_prob, strong=strong_aug)
        photo_transform = get_stereo_photometric_transforms(args.aug_prob, strong=strong_aug)
        photometric_left_only = True
        photometric_right_only = False
        print("Photometric: LEFT only")
    elif photometric_mode == "right":
        # Only apply photometric to right
        train_transform = get_stereo_geometric_transforms(args.img_size, args.aug_prob, strong=strong_aug)
        photo_transform = get_stereo_photometric_transforms(args.aug_prob, strong=strong_aug)
        photometric_left_only = False
        photometric_right_only = True
        print("Photometric: RIGHT only")
    else:  # none
        # Geometric only, no photometric
        train_transform = get_stereo_geometric_transforms(args.img_size, args.aug_prob, strong=strong_aug)
        photo_transform = None
        photometric_left_only = False
        photometric_right_only = False
        print("Photometric: NONE (geometric only)")
    
    val_transform = get_valid_transforms(img_size=args.img_size)
    
    # Print augmentation info
    if args.stereo_swap_prob > 0:
        print(f"Stereo swap: {args.stereo_swap_prob:.0%}")
    if args.mixup_prob > 0:
        context_str = "any sample" if args.no_mix_same_context else "same context only"
        print(f"MixUp: prob={args.mixup_prob:.0%}, alpha={args.mixup_alpha} ({context_str})")
    if args.cutmix_prob > 0:
        context_str = "any sample" if args.no_mix_same_context else "same context only"
        print(f"CutMix: prob={args.cutmix_prob:.0%}, alpha={args.cutmix_alpha} ({context_str})")
    
    mix_same_context = not args.no_mix_same_context
    
    # Create datasets
    train_dataset = BiomassDataset(
        df=train_df,
        image_dir=args.image_dir,
        transform=train_transform,
        is_train=True,
        return_aux_labels=True,
        stereo_swap_prob=args.stereo_swap_prob,
        photometric_transform=photo_transform,
        photometric_left_only=photometric_left_only,
        photometric_right_only=photometric_right_only,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
        cutmix_prob=args.cutmix_prob,
        cutmix_alpha=args.cutmix_alpha,
        mix_same_context=mix_same_context,
    )
    val_dataset = BiomassDataset(
        df=val_df,
        image_dir=args.image_dir,
        transform=val_transform,
        is_train=False,
        return_aux_labels=True,
    )
    
    # Create dataloaders
    # pin_memory not supported on MPS
    use_pin_memory = device.type == "cuda"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    
    # Create model
    model = DINOv2Flora(
        num_targets=5,
        train_blocks=args.train_blocks,
        train_tokens=args.train_tokens,
        backbone=args.backbone,
        ckpt_path=args.ckpt_path,
        use_reg_head=True,
        use_clf_head=args.use_clf_head,
        use_soft_clf=args.use_soft_clf,
        use_blending=args.use_blending,
        grid=tuple(args.grid),
        use_film=True,
        use_attention_pool=True,
        dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        use_log_scale=args.use_log_scale,
    ).to(device)
    
    # Use channels_last memory format for MPS (faster convolutions)
    if device.type == "mps":
        model = model.to(memory_format=torch.channels_last)
    
    # Set species prior from training data
    species_prior = compute_species_prior(train_df)
    model.set_species_prior(species_prior.to(device))
    
    print(f"Model parameters: {count_parameters(model):,} trainable")
    print(f"Target normalization: {'log-scale' if args.use_log_scale else 'raw-scale'}")
    
    # Create loss
    criterion = FloraLoss(
        use_r2_loss=True,
        use_cosine_loss=True,
        use_clf_loss=args.use_clf_head,
        use_mse_loss=True,
        r2_weight=args.r2_weight,
        cosine_weight=args.cosine_weight,
        clf_weight=args.clf_weight,
        mse_weight=args.mse_weight,
    ).to(device)
    
    # Mixed precision
    scaler = GradScaler() if args.amp and device.type == "cuda" else None
    
    # Checkpoint paths
    stage1_ckpt = output_dir / f"_stage1_fold{fold}.pth"
    best_ckpt = output_dir / f"dinov2_flora_best_fold{fold}.pth"
    
    # Training state
    best_r2 = -float("inf")
    best_epoch = 0
    stage1_best_r2 = -float("inf")
    stage1_best_epoch = 0
    
    # ===== Determine training mode =====
    if args.two_stage:
        # Two-stage training
        stage1_epochs = args.stage1_epochs
        stage1_patience = args.stage1_patience
        stage2_epochs = args.stage2_epochs
        stage2_patience = args.stage2_patience
        
        print(f"\n{'='*40}")
        if args.stage1_full:
            print(f"Stage 1: FULL fine-tuning (backbone + head)")
        else:
            print(f"Stage 1: HEAD-ONLY training (frozen backbone)")
        print(f"  Max epochs: {stage1_epochs}, Patience: {stage1_patience}")
        print(f"{'='*40}")
        
        # Stage 1 setup
        if args.stage1_full:
            # Full fine-tuning in Stage 1 (simple optimizer, no layer-wise LR)
            unfreeze_backbone(model, train_blocks=args.train_blocks)
        else:
            # Head-only in Stage 1
            freeze_backbone(model)
        
        print(f"Trainable parameters: {count_parameters(model):,}")
        
        # Stage 1 optimizer (simple, 2 param groups)
        total_steps = stage1_epochs * len(train_loader)
        optimizer, scheduler = create_simple_optimizer_scheduler(
            model, args.lr, args.backbone_lr, args.weight_decay,
            total_steps, args.warmup_pct, train_backbone=args.stage1_full,
        )
        optimizers = [optimizer]
        schedulers = [scheduler]
        
        # Stage 1 training loop
        patience_counter = 0
        for epoch in range(1, stage1_epochs + 1):
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizers, schedulers,
                device, scaler, epoch,
            )
            val_metrics, _, _ = validate(model, val_loader, device)
            
            improved = ""
            if val_metrics["weighted_r2"] > stage1_best_r2:
                stage1_best_r2 = val_metrics["weighted_r2"]
                stage1_best_epoch = epoch
                patience_counter = 0
                improved = " *"
                torch.save(model.state_dict(), stage1_ckpt)
            else:
                patience_counter += 1
            
            per_target = " | ".join([f"{n[0].upper()}={val_metrics[f'r2_{n}']:.3f}" for n in TARGET_NAMES])
            print(f"  [S1] Ep {epoch:02d}: loss={train_metrics['loss']:.4f}, "
                  f"R²={val_metrics['weighted_r2']:.4f} [{per_target}]{improved}")
            
            if args.stage1_patience and patience_counter >= stage1_patience:
                print(f"  Stage 1 early stop (no improvement for {stage1_patience} epochs)")
                break
        
        print(f"\nStage 1 best: R² = {stage1_best_r2:.4f} at epoch {stage1_best_epoch}")
        
        # Load best Stage 1 checkpoint
        if stage1_ckpt.exists():
            print(f"Loading best Stage 1 checkpoint...")
            model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        
        # ===== Stage 2: Refinement with layer-wise LR =====
        print(f"\n{'='*40}")
        print(f"Stage 2: Refinement with layer-wise LR")
        print(f"  Max epochs: {stage2_epochs}, Patience: {stage2_patience}")
        print(f"  LR mult: {args.lr_mult} (earlier layers get lower LR)")
        print(f"{'='*40}")
        
        # Ensure backbone is unfrozen for Stage 2
        unfreeze_backbone(model, train_blocks=args.train_blocks)
        print(f"Trainable parameters: {count_parameters(model):,}")
        
        # Create layer-wise optimizers/schedulers
        total_steps = stage2_epochs * len(train_loader)
        if args.layer_wise_lr:
            print(f"Creating layer-wise optimizers (lr_mult={args.lr_mult})...")
            optimizers, schedulers = create_layer_wise_optimizer_scheduler(
                model, args.lr * 0.5, args.backbone_lr, args.lr_mult,
                args.weight_decay, total_steps, args.warmup_pct,
            )
        else:
            optimizer, scheduler = create_simple_optimizer_scheduler(
                model, args.lr * 0.5, args.backbone_lr, args.weight_decay,
                total_steps, args.warmup_pct, train_backbone=True,
            )
            optimizers = [optimizer]
            schedulers = [scheduler]
        
        # Stage 2 training loop
        best_r2 = stage1_best_r2  # Start from Stage 1 best
        best_epoch = stage1_best_epoch
        patience_counter = 0
        
        for epoch in range(1, stage2_epochs + 1):
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizers, schedulers,
                device, scaler, epoch,
            )
            val_metrics, val_preds, val_targets = validate(model, val_loader, device)
            
            improved = ""
            if val_metrics["weighted_r2"] > best_r2:
                best_r2 = val_metrics["weighted_r2"]
                best_epoch = stage1_epochs + epoch
                patience_counter = 0
                improved = " *"
                torch.save(model.state_dict(), best_ckpt)
                
                # Save OOF predictions
                val_df_copy = val_df.copy()
                for i, name in enumerate(["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]):
                    val_df_copy[f"pred_{name}"] = val_preds[:, i]
                val_df_copy.to_csv(output_dir / f"oof_fold{fold}.csv", index=False)
            else:
                patience_counter += 1
            
            per_target = " | ".join([f"{n[0].upper()}={val_metrics[f'r2_{n}']:.3f}" for n in TARGET_NAMES])
            print(f"  [S2] Ep {epoch:02d}: loss={train_metrics['loss']:.4f}, "
                  f"R²={val_metrics['weighted_r2']:.4f} [{per_target}]{improved}")
            
            if patience_counter >= stage2_patience:
                print(f"  Stage 2 early stop (no improvement for {stage2_patience} epochs)")
                break
        
        # Cleanup Stage 1 checkpoint
        if stage1_ckpt.exists():
            stage1_ckpt.unlink()
    
    else:
        # Single-stage training (full or head-only based on train_blocks)
        print(f"\n{'='*40}")
        print(f"Single-stage training ({args.epochs} epochs)")
        print(f"{'='*40}")
        
        total_steps = args.epochs * len(train_loader)
        
        if args.layer_wise_lr and args.train_blocks > 0:
            print(f"Creating layer-wise optimizers (lr_mult={args.lr_mult})...")
            optimizers, schedulers = create_layer_wise_optimizer_scheduler(
                model, args.lr, args.backbone_lr, args.lr_mult,
                args.weight_decay, total_steps, args.warmup_pct,
            )
        else:
            optimizer, scheduler = create_simple_optimizer_scheduler(
                model, args.lr, args.backbone_lr, args.weight_decay,
                total_steps, args.warmup_pct, train_backbone=(args.train_blocks > 0),
            )
            optimizers = [optimizer]
            schedulers = [scheduler]
        
        patience_counter = 0
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizers, schedulers,
                device, scaler, epoch,
            )
            val_metrics, val_preds, val_targets = validate(model, val_loader, device)
            
            improved = ""
            if val_metrics["weighted_r2"] > best_r2:
                best_r2 = val_metrics["weighted_r2"]
                best_epoch = epoch
                patience_counter = 0
                improved = " *"
                torch.save(model.state_dict(), best_ckpt)
                
                # Save OOF predictions
                val_df_copy = val_df.copy()
                for i, name in enumerate(["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]):
                    val_df_copy[f"pred_{name}"] = val_preds[:, i]
                val_df_copy.to_csv(output_dir / f"oof_fold{fold}.csv", index=False)
            else:
                patience_counter += 1
            
            per_target = " | ".join([f"{n[0].upper()}={val_metrics[f'r2_{n}']:.3f}" for n in TARGET_NAMES])
            print(f"  Ep {epoch:02d}: loss={train_metrics['loss']:.4f}, "
                  f"R²={val_metrics['weighted_r2']:.4f} [{per_target}]{improved}")
            
            if patience_counter >= args.patience:
                print(f"  Early stop (no improvement for {args.patience} epochs)")
                break
    
    print(f"\nFold {fold} best: R² = {best_r2:.4f} at epoch {best_epoch}")
    
    return {
        "fold": fold,
        "best_r2": best_r2,
        "best_epoch": best_epoch,
        **{f"r2_{name}": val_metrics[f"r2_{name}"] for name in TARGET_NAMES},
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir + f"_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Save command
    with open(output_dir / "command.txt", "w") as f:
        f.write(" ".join(sys.argv))
    
    print(f"Output directory: {output_dir}")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Auto-adjust num_workers for MPS (0 is more stable/faster on M-series)
    if device.type == "mps" and args.num_workers > 0:
        print(f"  Auto-adjusting num_workers: {args.num_workers} -> 0 (MPS optimization)")
        args.num_workers = 0
    
    # Load and prepare data
    df = prepare_dataframe(args.train_csv)
    
    # Load predefined folds or create new ones
    if args.fold_csv and os.path.exists(args.fold_csv):
        print(f"Loading predefined folds from: {args.fold_csv}")
        fold_df = pd.read_csv(args.fold_csv)
        fold_mapping = fold_df.set_index("sample_id_prefix")["fold"].to_dict()
        df["fold"] = df["sample_id_prefix"].map(fold_mapping).fillna(0).astype(int)
        print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")
    else:
        print(f"Creating folds with {args.cv_strategy} strategy")
        df = create_folds(
            df,
            n_folds=args.n_folds,
            seed=args.seed,
            cv_strategy=args.cv_strategy,
        )
    
    # Save folds for reproducibility
    fold_df_out = df[["sample_id_prefix", "fold"]].drop_duplicates()
    fold_df_out.to_csv(output_dir / "folds.csv", index=False)
    
    # Train folds
    results = []
    for fold in args.folds:
        fold_result = train_fold(fold, df, args, output_dir, device)
        results.append(fold_result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nSummary:")
    print(results_df.to_string())
    
    mean_r2 = results_df["best_r2"].mean()
    print(f"\nMean R²: {mean_r2:.4f}")
    
    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "mean_r2": float(mean_r2),
            "folds": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()

