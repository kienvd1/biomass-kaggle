#!/usr/bin/env python3
"""
Ensemble Inference Script for CSIRO Biomass Prediction.

Single ensemble (5 models):
    python -m src.inference --model-dir outputs/exp1 --output submission.csv

Dual ensemble (10 models):
    python -m src.inference \
        --model-dir outputs/ensemble_A outputs/ensemble_B \
        --weights 0.93 0.07 \
        --output submission.csv
    
With TTA:
    python -m src.inference --model-dir outputs/exp1 --tta --output submission.csv

Device selection:
    python -m src.inference --device-type mps --model-dir outputs/exp1 --output submission.csv
"""
import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .device import DeviceType, empty_cache, get_amp_settings, get_device, get_device_type
from .models import build_model


class TestDataset(Dataset):
    """Test dataset for inference."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: A.Compose,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.paths = self.df["image_path"].values
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = os.path.basename(self.paths[idx])
        full_path = os.path.join(self.image_dir, filename)
        
        img = cv2.imread(full_path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        
        left_t = self.transform(image=left)["image"]
        right_t = self.transform(image=right)["image"]
        
        return left_t, right_t


def get_transforms(img_size: int = 518) -> A.Compose:
    """Base transforms for inference."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 518) -> List[A.Compose]:
    """TTA transforms (original + flips)."""
    base = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    return [
        # Original
        A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            *base
        ]),
        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            *base
        ]),
        # Vertical flip
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            *base
        ]),
    ]


def auto_detect_config(model_dir: str) -> dict:
    """Auto-detect model config from results.json if available."""
    results_path = os.path.join(model_dir, "results.json")
    if os.path.exists(results_path):
        import json
        with open(results_path, "r") as f:
            results = json.load(f)
        config = results.get("config", {})
        return {
            "backbone": config.get("backbone", "vit_base_patch14_reg4_dinov2.lvd142m"),
            "grid": tuple(config.get("grid", [2, 2])),
            "img_size": int(config.get("img_size", 518)),
            "dropout": config.get("dropout", 0.30),
            "hidden_ratio": config.get("hidden_ratio", 0.25),
        }
    return {}


def load_fold_models(
    model_dir: str,
    device: torch.device,
    num_folds: int = 5,
    backbone: Optional[str] = None,
) -> List[nn.Module]:
    """Load all fold models. Auto-detects config from results.json if backbone not specified."""
    
    # Auto-detect config from results.json
    auto_config = auto_detect_config(model_dir)
    if backbone is None:
        backbone = auto_config.get("backbone", "vit_base_patch14_reg4_dinov2.lvd142m")
    grid = auto_config.get("grid", (2, 2))
    dropout = auto_config.get("dropout", 0.30)
    hidden_ratio = auto_config.get("hidden_ratio", 0.25)
    
    print(f"Config: backbone={backbone}, grid={grid}")
    
    models = []
    
    for fold in range(num_folds):
        ckpt_path = os.path.join(model_dir, f"tiled_film_best_model_fold{fold}.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue
        
        model = build_model(
            backbone_name=backbone,
            model_type="tiled_film",
            grid=grid,
            pretrained=False,
            dropout=dropout,
            hidden_ratio=hidden_ratio,
        )
        
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        models.append(model)
        print(f"Loaded fold {fold}: {ckpt_path}")
    
    print(f"Total models loaded: {len(models)}")
    return models


@torch.no_grad()
def predict_ensemble(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device,
    device_type: DeviceType = DeviceType.CUDA,
) -> np.ndarray:
    """Predict with ensemble of models (average predictions)."""
    all_preds = []
    
    # Get AMP settings for device
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    
    for x_left, x_right in tqdm(loader, desc="Predicting"):
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        
        batch_preds = []
        
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            for model in models:
                green, dead, clover, gdm, total = model(x_left, x_right)
                preds = torch.cat([green, dead, clover, gdm, total], dim=1)
                batch_preds.append(preds.float())
        
        # Average across models
        avg_preds = torch.stack(batch_preds, dim=0).mean(dim=0)
        all_preds.append(avg_preds.cpu().numpy())
    
    return np.concatenate(all_preds, axis=0)


@torch.no_grad()
def predict_with_tta(
    models: List[nn.Module],
    test_df: pd.DataFrame,
    image_dir: str,
    device: torch.device,
    device_type: DeviceType = DeviceType.CUDA,
    batch_size: int = 4,
    num_workers: int = 4,
) -> np.ndarray:
    """Predict with TTA (Test Time Augmentation)."""
    tta_transforms = get_tta_transforms(img_size=518)
    tta_preds = []
    
    # pin_memory only beneficial on CUDA
    pin_memory = device_type == DeviceType.CUDA
    
    for i, transform in enumerate(tta_transforms):
        print(f"\nTTA view {i+1}/{len(tta_transforms)}")
        
        dataset = TestDataset(test_df, image_dir, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        preds = predict_ensemble(models, loader, device, device_type)
        tta_preds.append(preds)
    
    # Average across TTA views
    return np.mean(tta_preds, axis=0)


def create_submission(
    preds: np.ndarray,
    test_df: pd.DataFrame,
    test_unique: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Create submission file in competition format."""
    target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    
    # Ensure non-negative predictions
    preds = np.maximum(preds, 0)
    
    # Create wide format predictions
    wide = pd.DataFrame({
        "image_path": test_unique["image_path"].values,
        "Dry_Green_g": preds[:, 0],
        "Dry_Dead_g": preds[:, 1],
        "Dry_Clover_g": preds[:, 2],
        "GDM_g": preds[:, 3],
        "Dry_Total_g": preds[:, 4],
    })
    
    # Convert to long format
    long_preds = wide.melt(
        id_vars=["image_path"],
        value_vars=target_cols,
        var_name="target_name",
        value_name="target",
    )
    
    # Merge with test data to get sample_id
    submission = pd.merge(
        test_df[["sample_id", "image_path", "target_name"]],
        long_preds,
        on=["image_path", "target_name"],
        how="left",
    )[["sample_id", "target"]]
    
    # Clean up
    submission["target"] = submission["target"].fillna(0).clip(lower=0)
    
    # Save
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head(10))
    
    return submission


def main():
    parser = argparse.ArgumentParser(description="Ensemble Inference (supports 1-N backbones)")
    parser.add_argument("--model-dir", type=str, nargs="+", required=True, 
                        help="Directory(s) with fold checkpoints. Each dir = 5 folds of one backbone.")
    parser.add_argument("--backbone", type=str, nargs="+", default=None,
                        help="Backbone name(s) for each model-dir. If single value, used for all.")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Weights for each ensemble (must match number of model-dirs). Default: equal weights.")
    parser.add_argument("--test-csv", type=str, default="./data/test.csv")
    parser.add_argument("--test-image-dir", type=str, default="./data/test")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true", help="Enable Test Time Augmentation")
    parser.add_argument(
        "--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help="Device type: cuda (NVIDIA GPU), mps (Apple Silicon), cpu. Auto-detected if not specified."
    )
    
    args = parser.parse_args()
    
    # Device setup
    device_type = DeviceType(args.device_type) if args.device_type else get_device_type()
    device = get_device(device_type)
    print(f"Device type: {device_type.value}")
    print(f"Device: {device}")
    print(f"Model dirs: {args.model_dir}")
    print(f"TTA: {args.tta}")
    
    # Set ensemble weights
    num_ensembles = len(args.model_dir)
    if args.weights is None:
        weights = [1.0 / num_ensembles] * num_ensembles
    else:
        if len(args.weights) != num_ensembles:
            raise ValueError(f"Number of weights ({len(args.weights)}) must match number of model dirs ({num_ensembles})")
        # Normalize weights
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
    
    # Handle backbone(s) - can be single or per-model-dir
    if args.backbone is None:
        backbones = ["vit_base_patch14_reg4_dinov2.lvd142m"] * num_ensembles
    elif len(args.backbone) == 1:
        backbones = args.backbone * num_ensembles
    elif len(args.backbone) == num_ensembles:
        backbones = args.backbone
    else:
        raise ValueError(f"Number of backbones ({len(args.backbone)}) must be 1 or match model dirs ({num_ensembles})")
    
    print(f"Ensemble weights: {weights}")
    print(f"Backbones: {backbones}")
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(args.test_csv)
    test_unique = test_df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    print(f"Total test samples: {len(test_df)}")
    print(f"Unique images: {len(test_unique)}")
    
    # Load all ensembles
    all_ensemble_preds = []
    
    for i, model_dir in enumerate(args.model_dir):
        print(f"\n{'='*50}")
        print(f"Ensemble {i+1}/{num_ensembles}: {model_dir}")
        print(f"Backbone: {backbones[i]}")
        print(f"Weight: {weights[i]:.4f}")
        print(f"{'='*50}")
        
        # Load models for this ensemble
        models = load_fold_models(
            model_dir,
            device,
            num_folds=args.num_folds,
            backbone=backbones[i],
        )
        
        if len(models) == 0:
            print(f"WARNING: No models loaded from {model_dir}, skipping...")
            continue
        
        # Predict
        print(f"\nRunning inference for ensemble {i+1}...")
        if args.tta:
            preds = predict_with_tta(
                models,
                test_unique,
                args.test_image_dir,
                device,
                device_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        else:
            transform = get_transforms(img_size=518)
            dataset = TestDataset(test_unique, args.test_image_dir, transform)
            # pin_memory only beneficial on CUDA
            pin_memory = device_type == DeviceType.CUDA
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            preds = predict_ensemble(models, loader, device, device_type)
        
        all_ensemble_preds.append((weights[i], preds))
        
        # Cleanup
        del models
        empty_cache(device_type)
    
    if len(all_ensemble_preds) == 0:
        raise RuntimeError("No predictions generated!")
    
    # Weighted average of ensembles
    print(f"\n{'='*50}")
    print("Combining ensembles...")
    print(f"{'='*50}")
    
    final_preds = np.zeros_like(all_ensemble_preds[0][1])
    for weight, preds in all_ensemble_preds:
        final_preds += weight * preds
    
    # Renormalize if some ensembles were skipped
    actual_weight_sum = sum(w for w, _ in all_ensemble_preds)
    if actual_weight_sum < 0.999:
        final_preds /= actual_weight_sum
    
    # Create submission
    print("\nCreating submission...")
    create_submission(final_preds, test_df, test_unique, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

