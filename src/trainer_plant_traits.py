
"""Trainer class for PlantTraits-inspired Biomass Model."""
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
from .plant_traits_strategy import PlantTraitsBiomassModel, compute_dataset_stats
from .trainer import AverageMeter, is_main_process, reduce_tensor, get_scheduler, get_optimizer

class TrainerPlantTraits:
    """Trainer class for PlantTraitsBiomassModel."""
    
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.device_type = cfg._device_type_enum
        
        self.use_amp, self.autocast_device_type, self.amp_dtype = get_amp_settings(
            self.device_type, cfg.amp_dtype
        )
        if not cfg.use_amp:
            self.use_amp = False
        
        self.scaler = None
        if self.use_amp and self.device_type == DeviceType.CUDA and cfg.amp_dtype == "float16":
            self.scaler = GradScaler()
            
    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: AdamW,
        epoch: int,
        sampler: Optional[DistributedSampler] = None,
    ) -> float:
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
        
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(pbar):
            # Unpack batch (expecting return_aux_labels=True)
            # left_t, right_t, targets, state_label, month_label, species_label, ndvi, height
            x_left, x_right, targets = batch[0], batch[1], batch[2]
            species_label = batch[5] if len(batch) > 5 else None
            
            x_left = x_left.to(self.device, non_blocking=True)
            x_right = x_right.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if species_label is not None:
                species_label = species_label.to(self.device, non_blocking=True)
                
            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                # Model returns dict with 'loss'
                outputs = model(x_left, x_right, species_label=species_label, targets=targets)
                loss = outputs['loss']
                loss = loss / self.cfg.gradient_accumulation_steps
                
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
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
    ) -> Tuple[float, Dict[str, float]]:
        model.eval()
        losses = AverageMeter()
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(loader, desc="Validating", leave=False, disable=not is_main_process(self.cfg))
        
        for batch in pbar:
            x_left, x_right, targets = batch[0], batch[1], batch[2]
            species_label = batch[5] if len(batch) > 5 else None
            
            x_left = x_left.to(self.device, non_blocking=True)
            x_right = x_right.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if species_label is not None:
                species_label = species_label.to(self.device, non_blocking=True)
                
            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                outputs = model(x_left, x_right, species_label=species_label, targets=targets)
                loss = outputs['loss']
                preds = outputs['pred'] # This is the final blended raw prediction
            
            losses.update(loss.item(), x_left.size(0))
            all_preds.append(preds.float().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Simple weighted R2
        # Competition weights: [0.1, 0.1, 0.1, 0.2, 0.5]
        weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
        
        # Flatten
        y_true = all_targets.flatten()
        y_pred = all_preds.flatten()
        w = np.tile(weights, all_targets.shape[0])
        
        ss_res = np.sum(w * (y_true - y_pred)**2)
        ss_tot = np.sum(w * (y_true - np.average(y_true, weights=w))**2)
        weighted_r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        
        metrics = {"weighted_r2": float(weighted_r2)}
        
        if self.cfg.distributed:
            loss_tensor = torch.tensor(losses.avg, device=self.device)
            losses.avg = reduce_tensor(loss_tensor, self.cfg.world_size).item()
            
        return losses.avg, metrics

    def train_fold(self, fold: int, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> Dict:
        # Precompute dataset stats for this fold's TRAIN set
        target_mean, target_std, specie_traits_mean = compute_dataset_stats(train_df)
        
        # Datasets
        # IMPORTANT: Enable return_aux_labels=True to get species labels
        train_ds = BiomassDataset(
            train_df, self.cfg.train_image_dir, get_train_transforms(self.cfg.img_size, self.cfg.aug_prob),
            is_train=True, return_aux_labels=True
        )
        valid_ds = BiomassDataset(
            valid_df, self.cfg.train_image_dir, get_valid_transforms(self.cfg.img_size),
            is_train=False, return_aux_labels=True
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True, 
            num_workers=self.cfg.num_workers, drop_last=True
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=self.cfg.batch_size * 2, shuffle=False, 
            num_workers=self.cfg.num_workers
        )
        
        # Model
        # Pass depth configurations from config
        use_depth = getattr(self.cfg, "use_depth", False)
        depth_model_size = getattr(self.cfg, "depth_model_size", "small")
        use_depth_attention = getattr(self.cfg, "use_depth_attention", False)
        
        model = PlantTraitsBiomassModel(
            backbone_size=self.cfg.backbone_name, # "base"
            pretrained=self.cfg.pretrained,
            target_mean=target_mean,
            target_std=target_std,
            specie_traits_mean=specie_traits_mean,
            grid=self.cfg.grid,
            use_film=True,
            use_depth=use_depth,
            depth_model_size=depth_model_size,
            use_depth_attention=use_depth_attention,
        )
        model = model.to(self.device)
        
        # Optimizer
        optimizer = get_optimizer(
            model, self.cfg.lr, self.cfg.backbone_lr, self.cfg.weight_decay,
            stage=2, device_type=self.device_type
        )
        scheduler = get_scheduler(
            optimizer, self.cfg.scheduler, self.cfg.epochs, self.cfg.min_lr
        )
        
        best_r2 = float("-inf")
        history = {"train_loss": [], "valid_loss": [], "weighted_r2": []}
        
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(model, train_loader, optimizer, epoch)
            valid_loss, metrics = self.validate(model, valid_loader)
            
            if scheduler:
                scheduler.step()
                
            if is_main_process(self.cfg):
                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | R2: {metrics['weighted_r2']:.4f}")
                
            if metrics['weighted_r2'] > best_r2:
                best_r2 = metrics['weighted_r2']
                # Save model...
                save_path = os.path.join(self.cfg.output_dir, f"plant_traits_best_fold{fold}.pth")
                torch.save(model.state_dict(), save_path)
                
        return {"best_r2": best_r2}
