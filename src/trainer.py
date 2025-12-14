"""Training utilities and trainer class with multi-GPU DDP support."""
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .config import TrainConfig
from .dataset import BiomassDataset, get_train_transforms, get_valid_transforms
from .device import (
    DeviceType,
    empty_cache,
    get_amp_settings,
    get_dataloader_kwargs,
    supports_fused_optimizer,
)
from .models import build_model


class AverageMeter:
    """Computes and stores the average and current value."""
    
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


class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss."""
    
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target) + self.eps)


class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss matching competition metric."""
    
    def __init__(self, weights: List[float]) -> None:
        super().__init__()
        # weights: [Green, Dead, Clover, GDM, Total] = [0.1, 0.1, 0.1, 0.2, 0.5]
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 5)
        weights = self.weights.to(pred.device)
        sq_error = (pred - target) ** 2  # (B, 5)
        weighted_sq_error = sq_error * weights.unsqueeze(0)  # (B, 5)
        return weighted_sq_error.sum() / (pred.size(0) * weights.sum())

def _safe_r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² for a single target with edge-case handling."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    y_bar = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_bar) ** 2))
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_weighted_mse_loss_np(
    preds: np.ndarray, targets: np.ndarray, weights: List[float]
) -> float:
    """
    Weighted MSE loss matching training loss (WeightedMSELoss).
    Returns scalar: sum_i w_i * mse_i / sum(w).
    """
    w = np.array(weights, dtype=np.float64)
    mse = np.mean((preds.astype(np.float64) - targets.astype(np.float64)) ** 2, axis=0)  # (5,)
    return float(np.sum(w * mse) / np.sum(w))


def compute_per_target_metrics_np(
    preds: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
    target_weights: List[float],
) -> List[Dict[str, float]]:
    """Compute per-target metrics (R²/RMSE/MAE/MSE + weighted MSE component)."""
    preds_f = preds.astype(np.float64)
    targets_f = targets.astype(np.float64)
    mse = np.mean((preds_f - targets_f) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_f - targets_f), axis=0)
    r2 = np.array([_safe_r2_np(targets_f[:, i], preds_f[:, i]) for i in range(preds_f.shape[1])], dtype=np.float64)
    out: List[Dict[str, float]] = []
    for i, name in enumerate(target_names):
        out.append(
            {
                "target": name,
                "weight": float(target_weights[i]),
                "mse": float(mse[i]),
                "rmse": float(rmse[i]),
                "mae": float(mae[i]),
                "r2": float(r2[i]),
                "weighted_mse_component": float(target_weights[i] * mse[i]),
            }
        )
    return out


def compute_weighted_r2(
    preds: np.ndarray, 
    targets: np.ndarray, 
    weights: List[float]
) -> float:
    """
    Compute globally weighted R² as per competition metric.
    
    R²_w = 1 - SS_res / SS_tot
    where SS_res = Σ w_j * (y_j - ŷ_j)²
          SS_tot = Σ w_j * (y_j - ȳ_w)²
          ȳ_w = Σ w_j * y_j / Σ w_j (weighted mean)
    """
    weights = np.array(weights)  # [0.1, 0.1, 0.1, 0.2, 0.5]
    
    # Flatten: each sample has 5 targets, apply corresponding weight
    # preds, targets: (N, 5) -> flatten with weights
    n_samples = preds.shape[0]
    
    # Create weight array matching flattened structure
    w = np.tile(weights, n_samples)  # (N * 5,)
    y = targets.flatten()  # (N * 5,)
    y_hat = preds.flatten()  # (N * 5,)
    
    # Weighted mean of targets
    y_bar_w = np.sum(w * y) / np.sum(w)
    
    # Residual sum of squares
    ss_res = np.sum(w * (y - y_hat) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum(w * (y - y_bar_w) ** 2)
    
    # R² (handle edge case)
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0
    
    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2)


def get_loss_fn(loss_type: str = "mse", weights: Optional[List[float]] = None) -> nn.Module:
    """Get loss function by name."""
    if loss_type == "weighted_mse" and weights is not None:
        return WeightedMSELoss(weights)
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=1.0)
    elif loss_type == "rmse":
        return RMSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters for stage 1 training."""
    # Handle DDP wrapped models
    base_model = model.module if hasattr(model, "module") else model
    # Handle compiled models
    if hasattr(base_model, "_orig_mod"):
        base_model = base_model._orig_mod
    
    for name, param in base_model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze backbone parameters for stage 2 training."""
    # Handle DDP wrapped models
    base_model = model.module if hasattr(model, "module") else model
    # Handle compiled models
    if hasattr(base_model, "_orig_mod"):
        base_model = base_model._orig_mod
    
    for name, param in base_model.named_parameters():
        if "backbone" in name:
            param.requires_grad = True


def get_optimizer(
    model: nn.Module,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    stage: int = 2,
    device_type: Optional[DeviceType] = None,
) -> AdamW:
    """
    Create optimizer with different learning rates for backbone and heads.
    
    Args:
        model: Model to optimize
        lr: Learning rate for head parameters
        backbone_lr: Learning rate for backbone parameters (stage 2 only)
        weight_decay: Weight decay
        stage: Training stage (1 = head only, 2 = full finetune)
        device_type: Device type (for fused optimizer support)
    """
    # Handle DDP wrapped models
    base_model = model.module if hasattr(model, "module") else model
    # Handle compiled models
    if hasattr(base_model, "_orig_mod"):
        base_model = base_model._orig_mod
    
    backbone_params = []
    head_params = []
    
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen params
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    if stage == 1:
        # Stage 1: Only head params (backbone is frozen)
        param_groups = [{"params": head_params, "lr": lr}]
    else:
        # Stage 2: Both with differential LR
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": lr},
        ]
    
    # Fused optimizer only available on CUDA
    use_fused = device_type is not None and supports_fused_optimizer(device_type)
    return AdamW(param_groups, weight_decay=weight_decay, fused=use_fused)


def get_scheduler(
    optimizer: AdamW,
    scheduler_type: str,
    epochs: int,
    min_lr: float,
) -> Optional[object]:
    """Get learning rate scheduler."""
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=min_lr
        )
    else:
        return None


def is_main_process(cfg: TrainConfig) -> bool:
    """Check if this is the main process (rank 0)."""
    return not cfg.distributed or cfg.local_rank == 0


def setup_distributed(cfg: TrainConfig) -> None:
    """Initialize distributed training."""
    if not cfg.distributed:
        return
    
    # Get local rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.local_rank = local_rank
    cfg.world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Set device
    torch.cuda.set_device(local_rank)
    cfg.device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process(cfg):
        print(f"Initialized DDP: world_size={cfg.world_size}, local_rank={local_rank}")


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reduce tensor across all processes."""
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


class Trainer:
    """Trainer class for biomass prediction with multi-GPU/MPS/CPU support."""
    
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.device_type = cfg._device_type_enum
        
        # AMP settings based on device type
        self.use_amp, self.autocast_device_type, self.amp_dtype = get_amp_settings(
            self.device_type, cfg.amp_dtype
        )
        # Override AMP if disabled in config
        if not cfg.use_amp:
            self.use_amp = False
        
        # GradScaler only needed for CUDA float16, not bfloat16 or MPS
        self.scaler = None
        if self.use_amp and self.device_type == DeviceType.CUDA and cfg.amp_dtype == "float16":
            self.scaler = GradScaler()
        
    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: AdamW,
        loss_fn: nn.Module,
        epoch: int,
        sampler: Optional[DistributedSampler] = None,
    ) -> float:
        """Train for one epoch."""
        model.train()
        losses = AverageMeter()
        
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
            disable=not is_main_process(self.cfg),
        )
        
        optimizer.zero_grad(set_to_none=True)  # More efficient
        
        for step, (x_left, x_right, targets) in enumerate(pbar):
            # Use channels-last for MPS performance
            if self.device_type == DeviceType.MPS:
                x_left = x_left.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                x_right = x_right.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            else:
                x_left = x_left.to(self.device, non_blocking=True)
                x_right = x_right.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with AMP
            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                green, dead, clover, gdm, total = model(x_left, x_right)
                preds = torch.cat([green, dead, clover, gdm, total], dim=1)
                loss = loss_fn(preds, targets)
                loss = loss / self.cfg.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            loss_val = loss.item() * self.cfg.gradient_accumulation_steps
            if self.cfg.distributed:
                loss_tensor = torch.tensor(loss_val, device=self.device)
                loss_val = reduce_tensor(loss_tensor, self.cfg.world_size).item()
            
            losses.update(loss_val, x_left.size(0) * self.cfg.world_size)
            
            if step % self.cfg.log_interval == 0:
                pbar.set_postfix({"loss": f"{losses.avg:.4f}"})
        
        return losses.avg
    
    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        model.eval()
        losses = AverageMeter()
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(
            loader, desc="Validating", leave=False, disable=not is_main_process(self.cfg)
        )
        
        for x_left, x_right, targets in pbar:
            # Use channels-last for MPS performance
            if self.device_type == DeviceType.MPS:
                x_left = x_left.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                x_right = x_right.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            else:
                x_left = x_left.to(self.device, non_blocking=True)
                x_right = x_right.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                green, dead, clover, gdm, total = model(x_left, x_right)
                preds = torch.cat([green, dead, clover, gdm, total], dim=1)
                loss = loss_fn(preds, targets)
            
            losses.update(loss.item(), x_left.size(0))
            all_preds.append(preds.float().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        target_names = ["green", "dead", "clover", "gdm", "total"]
        weights = getattr(self.cfg, "target_weights", [0.1, 0.1, 0.1, 0.2, 0.5])

        per_target = compute_per_target_metrics_np(all_preds, all_targets, target_names, weights)

        metrics: Dict[str, float] = {}
        for row in per_target:
            t = row["target"]
            metrics[f"r2_{t}"] = float(row["r2"])
            metrics[f"rmse_{t}"] = float(row["rmse"])
            metrics[f"mae_{t}"] = float(row["mae"])
            metrics[f"mse_{t}"] = float(row["mse"])
            metrics[f"w_mse_{t}"] = float(row["weighted_mse_component"])

        metrics["rmse_mean"] = float(np.mean([metrics[f"rmse_{n}"] for n in target_names]))
        metrics["weighted_r2"] = compute_weighted_r2(all_preds, all_targets, weights)
        metrics["weighted_mse"] = compute_weighted_mse_loss_np(all_preds, all_targets, weights)
        
        # Reduce metrics across processes
        if self.cfg.distributed:
            loss_tensor = torch.tensor(losses.avg, device=self.device)
            losses.avg = reduce_tensor(loss_tensor, self.cfg.world_size).item()
        
        return losses.avg, metrics
    
    def train_fold(
        self,
        fold: int,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Train a single fold with optional 2-stage training."""
        two_stage = getattr(self.cfg, "two_stage", False)
        freeze_epochs = getattr(self.cfg, "freeze_epochs", 5)
        head_lr_stage1 = getattr(self.cfg, "head_lr_stage1", 1e-3)
        
        if is_main_process(self.cfg):
            print(f"\n{'='*60}")
            print(f"Training Fold {fold}")
            print(f"{'='*60}")
            print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
            if two_stage:
                print(f"2-Stage Training: Stage 1 ({freeze_epochs} epochs, head only), Stage 2 (full finetune)")
        
        # Create datasets
        train_transform = get_train_transforms(self.cfg.img_size, self.cfg.aug_prob)
        valid_transform = get_valid_transforms(self.cfg.img_size)
        
        train_ds = BiomassDataset(
            train_df, self.cfg.train_image_dir, train_transform, is_train=True
        )
        valid_ds = BiomassDataset(
            valid_df, self.cfg.train_image_dir, valid_transform, is_train=False
        )
        
        # Distributed sampler
        train_sampler = None
        if self.cfg.distributed:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.cfg.world_size,
                rank=self.cfg.local_rank,
                shuffle=True,
            )
        
        # Get device-specific DataLoader kwargs
        loader_kwargs = get_dataloader_kwargs(
            self.device_type,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
        # Validation can use larger batch size (no gradients needed)
        valid_batch_multiplier = 4 if self.device_type == DeviceType.MPS else 2
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.cfg.batch_size * valid_batch_multiplier,
            shuffle=False,
            **loader_kwargs,
        )
        
        # Build model
        model = build_model(
            backbone_name=self.cfg.backbone_name,
            model_type="tiled_film",
            grid=self.cfg.grid,
            pretrained=self.cfg.pretrained,
            dropout=self.cfg.dropout,
            hidden_ratio=self.cfg.hidden_ratio,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
        )
        model = model.to(self.device)

        # Use channels-last memory format for better performance on MPS
        if self.device_type == DeviceType.MPS:
            model = model.to(memory_format=torch.channels_last)

        # Compile model for faster training (PyTorch 2.0+)
        if self.cfg.compile_model:
            compile_mode = getattr(self.cfg, "compile_mode", "max-autotune")
            if is_main_process(self.cfg):
                print(f"Compiling model with torch.compile(mode='{compile_mode}')...")
            model = torch.compile(
                model,
                mode=compile_mode,
                dynamic=False,  # Fixed batch size with drop_last=True
                fullgraph=False,  # Allow graph breaks for compatibility
            )
        
        # 2-Stage Training: Freeze backbone for stage 1
        current_stage = 1 if two_stage else 2
        if two_stage:
            freeze_backbone(model)
            if is_main_process(self.cfg):
                print(f"Stage 1: Backbone FROZEN, training head only (LR: {head_lr_stage1:.1e})")
        
        # Wrap with DDP (after freezing to ensure proper gradient sync)
        if self.cfg.distributed:
            model = DDP(
                model,
                device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
                find_unused_parameters=two_stage,  # True if 2-stage (frozen params)
            )
        
        # Initialize optimizer for current stage
        if two_stage:
            optimizer = get_optimizer(
                model, head_lr_stage1, self.cfg.backbone_lr, self.cfg.weight_decay,
                stage=1, device_type=self.device_type
            )
            scheduler = get_scheduler(
                optimizer, self.cfg.scheduler, freeze_epochs, self.cfg.min_lr
            )
        else:
            optimizer = get_optimizer(
                model, self.cfg.lr, self.cfg.backbone_lr, self.cfg.weight_decay,
                stage=2, device_type=self.device_type
            )
            scheduler = get_scheduler(
                optimizer, self.cfg.scheduler, self.cfg.epochs, self.cfg.min_lr
            )
        
        # Use weighted MSE loss to match competition metric
        weights = getattr(self.cfg, "target_weights", [0.1, 0.1, 0.1, 0.2, 0.5])
        loss_fn = get_loss_fn("weighted_mse", weights=weights)
        
        # Best model selection: "loss" (lower better) or "r2" (higher better)
        best_metric = getattr(self.cfg, "best_metric", "loss")
        if best_metric == "loss":
            best_score = float("inf")
            is_better = lambda new, old: new < old
        else:  # r2
            best_score = float("-inf")
            is_better = lambda new, old: new > old
        
        patience_counter = 0
        best_r2 = 0.0  # Track best R2 for reporting
        best_loss = float("inf")  # Track best loss for reporting
        history = {
            "train_loss": [],
            "valid_loss": [],
            "valid_rmse": [],
            "weighted_r2": [],
            "weighted_mse": [],
            "per_target_r2": [],
            "per_target_rmse": [],
            "stage": [],
        }
        best_metrics_snapshot: Optional[Dict[str, float]] = None
        
        for epoch in range(1, self.cfg.epochs + 1):
            start_time = time.time()
            
            # Check for stage transition (2-stage training)
            if two_stage and current_stage == 1 and epoch > freeze_epochs:
                current_stage = 2
                if is_main_process(self.cfg):
                    print(f"\n{'='*60}")
                    print(f"Stage 2: Unfreezing backbone for full finetuning")
                    print(f"  Head LR: {self.cfg.lr:.1e}, Backbone LR: {self.cfg.backbone_lr:.1e}")
                    print(f"{'='*60}")
                
                # Unfreeze backbone
                unfreeze_backbone(model)
                
                # Create new optimizer with differential LR
                optimizer = get_optimizer(
                    model, self.cfg.lr, self.cfg.backbone_lr, self.cfg.weight_decay,
                    stage=2, device_type=self.device_type
                )
                # New scheduler for remaining epochs
                remaining_epochs = self.cfg.epochs - freeze_epochs
                scheduler = get_scheduler(
                    optimizer, self.cfg.scheduler, remaining_epochs, self.cfg.min_lr
                )
                
                # Reset patience for stage 2
                patience_counter = 0
            
            # Train
            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, loss_fn, epoch, train_sampler
            )
            
            # Validate
            valid_loss, metrics = self.validate(model, valid_loader, loss_fn)
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metrics["rmse_mean"])
                else:
                    scheduler.step()
            
            # Logging (main process only)
            if is_main_process(self.cfg):
                elapsed = time.time() - start_time
                # Get head LR (last param group)
                lr = optimizer.param_groups[-1]["lr"]
                stage_str = f"S{current_stage}" if two_stage else ""
                
                print(
                    f"Epoch {epoch:02d} {stage_str:>3} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Valid Loss: {valid_loss:.4f} | "
                    f"R²: {metrics['weighted_r2']:.4f} | "
                    f"RMSE: {metrics['rmse_mean']:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
                # Compact per-target view for debugging label quality
                r2_vec = [metrics.get(f"r2_{n}", 0.0) for n in ["green", "dead", "clover", "gdm", "total"]]
                rmse_vec = [metrics.get(f"rmse_{n}", 0.0) for n in ["green", "dead", "clover", "gdm", "total"]]
                print(
                    "   per-target "
                    f"R2[g,d,c,gdm,t]=[{', '.join(f'{x:.3f}' for x in r2_vec)}]  "
                    f"RMSE=[{', '.join(f'{x:.3f}' for x in rmse_vec)}]"
                )
                
                history["train_loss"].append(train_loss)
                history["valid_loss"].append(valid_loss)
                history["valid_rmse"].append(metrics["rmse_mean"])
                history["weighted_r2"].append(metrics["weighted_r2"])
                history["weighted_mse"].append(metrics["weighted_mse"])
                history["per_target_r2"].append(r2_vec)
                history["per_target_rmse"].append(rmse_vec)
                history["stage"].append(current_stage)
                
                # Get current score based on metric type
                if best_metric == "loss":
                    score = valid_loss
                else:
                    score = metrics["weighted_r2"]
                
                # Save best model
                if is_better(score, best_score):
                    best_score = score
                    best_r2 = metrics["weighted_r2"]
                    best_loss = valid_loss
                    best_metrics_snapshot = {k: float(v) for k, v in metrics.items()}
                    patience_counter = 0
                    
                    save_path = os.path.join(
                        self.cfg.output_dir, f"tiled_film_best_model_fold{fold}.pth"
                    )
                    # Save unwrapped model state dict
                    model_to_save = model.module if hasattr(model, "module") else model
                    # Handle compiled model
                    if hasattr(model_to_save, "_orig_mod"):
                        model_to_save = model_to_save._orig_mod
                    torch.save(model_to_save.state_dict(), save_path)
                    if best_metric == "loss":
                        print(f"  -> Saved best model (Loss: {best_loss:.4f}, R²: {best_r2:.4f})")
                    else:
                        print(f"  -> Saved best model (R²: {best_r2:.4f}, Loss: {best_loss:.4f})")
                else:
                    patience_counter += 1
            
            # Sync patience counter for early stopping
            if self.cfg.distributed:
                patience_tensor = torch.tensor(patience_counter, device=self.device)
                dist.broadcast(patience_tensor, src=0)
                patience_counter = patience_tensor.item()
            
            # Early stopping
            # For 2-stage: triggers in stage 2, or in stage 1 if freeze_epochs >= epochs (stage 1 only mode)
            stage1_only = two_stage and freeze_epochs >= self.cfg.epochs
            if (current_stage == 2 or stage1_only or not two_stage) and patience_counter >= self.cfg.patience:
                if is_main_process(self.cfg):
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Cleanup
        del model, optimizer, scheduler
        empty_cache(self.device_type)
        
        return {
            "fold": fold,
            "best_r2": best_r2,
            "best_loss": best_loss,
            "best_metric": best_metric,
            "history": history,
            "best_metrics": best_metrics_snapshot or {},
        }
    
    def train_all_folds(self, df: pd.DataFrame) -> List[Dict]:
        """Train all folds."""
        results = []
        
        for fold in self.cfg.train_folds:
            train_df = df[df["fold"] != fold].reset_index(drop=True)
            valid_df = df[df["fold"] == fold].reset_index(drop=True)
            
            fold_result = self.train_fold(fold, train_df, valid_df)
            results.append(fold_result)
            
            # Synchronize between folds
            if self.cfg.distributed:
                dist.barrier()
        
        # Summary (main process only)
        if is_main_process(self.cfg):
            print("\n" + "=" * 60)
            print("Training Summary")
            print("=" * 60)
            
            r2_scores = [r["best_r2"] for r in results]
            for r in results:
                print(f"Fold {r['fold']}: Best R² = {r['best_r2']:.4f}")
            
            print(f"\nCV Mean R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        
        return results
