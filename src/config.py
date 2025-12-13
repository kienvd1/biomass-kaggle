"""Training configuration for CSIRO Biomass prediction."""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .device import DeviceType, get_device, get_device_type


@dataclass
class TrainConfig:
    """Training configuration."""
    
    # Paths (use relative paths as defaults for local development)
    base_path: str = "./data"
    train_csv: str = field(default_factory=lambda: os.path.join("./data", "train.csv"))
    train_image_dir: str = field(default_factory=lambda: os.path.join("./data", "train"))
    output_dir: str = field(default_factory=lambda: os.path.join("./outputs"))
    
    # Model
    backbone_name: str = "vit_base_patch14_reg4_dinov2.lvd142m"
    pretrained: bool = True
    dropout: float = 0.30
    hidden_ratio: float = 0.25
    grid: Tuple[int, int] = (2, 2)
    gradient_checkpointing: bool = False  # Enable for memory savings
    
    # Training
    num_folds: int = 5
    train_folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    epochs: int = 50
    batch_size: int = 8  # Per-GPU batch size (increase for large GPU/unified memory)
    num_workers: int = 8  # Increased for faster data loading
    gradient_accumulation_steps: int = 1
    prefetch_factor: int = 4  # Prefetch batches per worker
    cache_images: bool = False  # Cache images in RAM (enable if you have lots of free memory)
    
    # Best model selection: "loss" or "r2"
    best_metric: str = "loss"
    
    # CV strategy:
    # - "group_month": group by month, stratify by Dry_Total_g bins
    # - "group_date": group by Sampling_Date, stratify by Dry_Total_g bins
    # - "group_date_state": group by Sampling_Date_Month, stratify by State
    # - "group_date_state_bin": group by Sampling_Date_Month, stratify by State × Dry_Total_g bin
    # - "stratified": stratify by Dry_Total_g bins only
    # - "random": random KFold
    cv_strategy: str = "group_month"
    
    # Device configuration
    # - "cuda": NVIDIA GPU with DDP support
    # - "mps": Apple Silicon GPU (single device, no DDP)
    # - "cpu": CPU fallback
    # - None: auto-detect best available
    device_type: Optional[str] = None  # "cuda", "mps", "cpu", or None for auto
    
    # Multi-GPU (CUDA only, auto-enabled when multiple GPUs detected)
    distributed: bool = False
    world_size: int = 1  # Number of GPUs (auto-set by DDP for multi-GPU)
    local_rank: int = 0
    
    # Optimizer
    lr: float = 2e-4  # Head learning rate for stage 2
    backbone_lr: float = 1e-5  # Backbone learning rate for stage 2 (very low)
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    
    # 2-Stage Training
    two_stage: bool = True  # Enable 2-stage training
    freeze_epochs: int = 5  # Stage 1: epochs with frozen backbone
    head_lr_stage1: float = 1e-3  # Stage 1: higher LR for head only
    
    # Scheduler
    scheduler: str = "cosine"  # cosine, step, plateau
    min_lr: float = 1e-7
    
    # Loss
    loss_type: str = "mse"  # mse, mae, huber, rmse
    
    # Augmentations
    img_size: int = 518
    aug_prob: float = 0.5
    
    # Mixed precision (bf16 for H200)
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 or float16
    
    # Compile model (PyTorch 2.0+)
    compile_model: bool = False
    compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    
    # Logging
    log_interval: int = 20
    save_best_only: bool = True
    
    # Early stopping
    patience: int = 10
    
    # Seed (18 found optimal for fold balance via seed search)
    seed: int = 18
    
    # Device (set by DDP init)
    device: Optional[torch.device] = None
    
    # Target columns and competition weights
    all_target_cols: List[str] = field(
        default_factory=lambda: ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    )
    # Competition weights for weighted R² metric
    target_weights: List[float] = field(
        default_factory=lambda: [0.1, 0.1, 0.1, 0.2, 0.5]  # Green, Dead, Clover, GDM, Total
    )
    
    # DINOv2 backbone candidates (small and base only, large excluded for efficiency)
    dino_candidates: List[str] = field(
        default_factory=lambda: [
            "vit_base_patch14_reg4_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m", 
            "vit_small_patch14_reg4_dinov2.lvd142m",
            "vit_small_patch14_dinov2.lvd142m",
        ]
    )
    
    def __post_init__(self) -> None:
        # Note: Don't create output_dir here - it's done in train.py after CLI args are applied
        # Auto-detect device type if not specified
        if self.device_type is None:
            self._device_type_enum = get_device_type()
        else:
            self._device_type_enum = DeviceType(self.device_type)
        
        # Set device
        if self.device is None:
            self.device = get_device(self._device_type_enum, self.local_rank)
        
        # Disable distributed for non-CUDA devices
        if self._device_type_enum != DeviceType.CUDA:
            self.distributed = False
            self.world_size = 1

