#!/usr/bin/env python3
"""
OOF Evaluation script for Ratio Model.

Usage:
    python -m src.eval_ratio_oof --model-dir ./outputs/ratio_XXXX --device mps
    python -m src.eval_ratio_oof --model-dir ./outputs/ratio_XXXX --tta --device cuda
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .models_ratio import build_ratio_model


class OOFDataset(Dataset):
    """Dataset for OOF evaluation."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: A.Compose,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        
        self.target_cols = [
            "Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"
        ]
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, os.path.basename(row["image_path"]))
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Split stereo
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        
        # Transform
        aug_left = self.transform(image=left)
        aug_right = self.transform(image=right)
        
        left_t = aug_left["image"]
        right_t = aug_right["image"]
        
        # Targets
        targets = torch.tensor(
            [row[col] for col in self.target_cols],
            dtype=torch.float32
        )
        
        return left_t, right_t, targets


def get_val_transform(img_size: int = 518) -> A.Compose:
    """Get validation transform."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 518) -> List[A.Compose]:
    """Get TTA transforms."""
    base = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return [
        A.Compose([A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA), *base]),
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA), *base]),
        A.Compose([A.VerticalFlip(p=1.0), A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA), *base]),
    ]


def load_model(
    checkpoint_path: str,
    backbone_name: str,
    device: torch.device,
    model_type: str = "softmax",
    grid: Tuple[int, int] = (2, 2),
    dropout: float = 0.2,
    hidden_ratio: float = 0.5,
    use_film: bool = True,
    use_attention_pool: bool = True,
) -> nn.Module:
    """Load trained model from checkpoint."""
    model = build_ratio_model(
        backbone_name=backbone_name,
        grid=grid,
        pretrained=False,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
        use_film=use_film,
        use_attention_pool=use_attention_pool,
        model_type=model_type,
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def predict_fold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Predicting",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict on a fold's validation set."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_ratios = []
    
    for batch in tqdm(loader, desc=desc, leave=False):
        x_left, x_right, targets = batch
        x_left = x_left.to(device)
        x_right = x_right.to(device)
        
        outputs = model(x_left, x_right, return_ratios=True)
        # SoftmaxRatioDINO returns 6 values, HierarchicalRatioDINO returns 7
        if len(outputs) == 6:
            green, dead, clover, gdm, total, ratios = outputs
        else:
            green, dead, clover, gdm, total, alive_ratio, green_ratio = outputs
            total_safe = total + 1e-8
            ratios = torch.cat([
                green / total_safe,
                dead / total_safe,
                clover / total_safe,
            ], dim=1)
        preds = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
        all_ratios.append(ratios.cpu().numpy())
    
    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_ratios, axis=0),
    )


@torch.no_grad()
def predict_fold_tta(
    model: nn.Module,
    df: pd.DataFrame,
    image_dir: str,
    transforms: List[A.Compose],
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 0,
    desc: str = "TTA Predicting",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict with TTA on a fold's validation set."""
    model.eval()
    
    all_view_preds = []
    all_view_ratios = []
    targets = None
    
    for i, transform in enumerate(transforms):
        ds = OOFDataset(df, image_dir, transform)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=device.type == "cuda"
        )
        
        preds_list = []
        targets_list = []
        ratios_list = []
        
        for batch in tqdm(loader, desc=f"{desc} view {i+1}/{len(transforms)}", leave=False):
            x_left, x_right, tgt = batch
            x_left = x_left.to(device)
            x_right = x_right.to(device)
            
            outputs = model(x_left, x_right, return_ratios=True)
            if len(outputs) == 6:
                green, dead, clover, gdm, total, ratios = outputs
            else:
                green, dead, clover, gdm, total, alive_ratio, green_ratio = outputs
                total_safe = total + 1e-8
                ratios = torch.cat([
                    green / total_safe,
                    dead / total_safe,
                    clover / total_safe,
                ], dim=1)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            
            preds_list.append(preds.cpu().numpy())
            targets_list.append(tgt.numpy())
            ratios_list.append(ratios.cpu().numpy())
        
        all_view_preds.append(np.concatenate(preds_list, axis=0))
        all_view_ratios.append(np.concatenate(ratios_list, axis=0))
        if targets is None:
            targets = np.concatenate(targets_list, axis=0)
    
    # Average predictions across TTA views
    avg_preds = np.mean(all_view_preds, axis=0)
    avg_ratios = np.mean(all_view_ratios, axis=0)
    
    return avg_preds, targets, avg_ratios


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    target_names: List[str] = ["Green", "Dead", "Clover", "GDM", "Total"],
    weights: List[float] = [0.1, 0.1, 0.1, 0.2, 0.5],
) -> Dict[str, float]:
    """Compute per-target and weighted R² metrics."""
    metrics = {}
    
    weighted_r2 = 0.0
    for i, (name, weight) in enumerate(zip(target_names, weights)):
        r2 = r2_score(targets[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
        mae = mean_absolute_error(targets[:, i], preds[:, i])
        
        metrics[f"r2_{name.lower()}"] = r2
        metrics[f"rmse_{name.lower()}"] = rmse
        metrics[f"mae_{name.lower()}"] = mae
        
        weighted_r2 += weight * r2
    
    metrics["weighted_r2"] = weighted_r2
    
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="OOF Evaluation for Ratio Model")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone name (auto-detect if not specified)")
    parser.add_argument("--model-type", type=str, default="softmax", choices=["softmax", "hierarchical"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta", action="store_true", help="Use TTA")
    parser.add_argument("--img-size", type=int, default=518)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load config
    config_path = os.path.join(args.model_dir, "results.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        cfg = config.get("config", {})
        backbone = args.backbone or cfg.get("backbone", "vit_base_patch14_reg4_dinov2.lvd142m")
        model_type = cfg.get("model_type", args.model_type)
        grid = tuple(cfg.get("grid", [2, 2]))
        dropout = cfg.get("dropout", 0.2)
        hidden_ratio = cfg.get("hidden_ratio", 0.5)
        use_film = cfg.get("use_film", True)
        use_attention_pool = cfg.get("use_attention_pool", True)
    else:
        backbone = args.backbone or "vit_base_patch14_reg4_dinov2.lvd142m"
        model_type = args.model_type
        grid = (2, 2)
        dropout = 0.2
        hidden_ratio = 0.5
        use_film = True
        use_attention_pool = True
    
    print("=" * 60)
    print("OOF Evaluation for Ratio Model")
    print("=" * 60)
    print(f"Model dir: {args.model_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"Backbone: {backbone}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"TTA: {args.tta}")
    print(f"Image size: {args.img_size}")
    print("=" * 60)
    
    # Load folds
    folds_path = os.path.join(args.model_dir, "folds.csv")
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"folds.csv not found in {args.model_dir}")
    
    df = pd.read_csv(folds_path)
    print(f"Loaded {len(df)} samples from folds.csv")
    print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")
    
    image_dir = os.path.join(args.data_dir, "train")
    
    # OOF prediction
    n_samples = len(df)
    oof_preds = np.zeros((n_samples, 5), dtype=np.float32)
    oof_targets = np.zeros((n_samples, 5), dtype=np.float32)
    oof_ratios = np.zeros((n_samples, 3), dtype=np.float32)
    
    folds = sorted(df["fold"].unique())
    
    for fold in folds:
        ckpt_path = os.path.join(args.model_dir, f"ratio_best_fold{fold}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for fold {fold}, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Fold {fold}: Loading model and predicting validation set")
        if args.tta:
            print(f"  TTA: Enabled (3 views: original, hflip, vflip)")
        print("=" * 60)
        
        # Load model
        model = load_model(
            ckpt_path, backbone, device,
            model_type=model_type,
            grid=grid,
            dropout=dropout,
            hidden_ratio=hidden_ratio,
            use_film=use_film,
            use_attention_pool=use_attention_pool,
        )
        print(f"  Model: FiLM={use_film}, AttnPool={use_attention_pool}")
        
        # Get validation indices
        val_mask = df["fold"] == fold
        val_indices = df.index[val_mask].tolist()
        val_df = df[val_mask].reset_index(drop=True)
        print(f"  Validation samples: {len(val_df)}")
        
        # Predict
        if args.tta:
            transforms = get_tta_transforms(args.img_size)
            preds, targets, ratios = predict_fold_tta(
                model, val_df, image_dir, transforms, device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                desc=f"Fold {fold}",
            )
        else:
            transform = get_val_transform(args.img_size)
            ds = OOFDataset(val_df, image_dir, transform)
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=device.type == "cuda"
            )
            preds, targets, ratios = predict_fold(model, loader, device, desc=f"Fold {fold}")
        
        # Store
        for i, idx in enumerate(val_indices):
            oof_preds[idx] = preds[i]
            oof_targets[idx] = targets[i]
            oof_ratios[idx] = ratios[i]
        
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("OOF Results")
    print("=" * 60)
    
    metrics = compute_metrics(oof_preds, oof_targets)
    
    for name in ["Green", "Dead", "Clover", "GDM", "Total"]:
        r2 = metrics[f"r2_{name.lower()}"]
        rmse = metrics[f"rmse_{name.lower()}"]
        mae = metrics[f"mae_{name.lower()}"]
        print(f"  {name:8s}: R² = {r2:7.4f}, RMSE = {rmse:7.2f}, MAE = {mae:7.2f}")
    
    print(f"\n  Weighted R²: {metrics['weighted_r2']:.4f}")
    
    # Ratio statistics
    print("\n" + "=" * 60)
    print("Predicted Ratio Statistics")
    print("=" * 60)
    print(f"  Green ratio:  mean={oof_ratios[:, 0].mean():.3f}, std={oof_ratios[:, 0].std():.3f}")
    print(f"  Dead ratio:   mean={oof_ratios[:, 1].mean():.3f}, std={oof_ratios[:, 1].std():.3f}")
    print(f"  Clover ratio: mean={oof_ratios[:, 2].mean():.3f}, std={oof_ratios[:, 2].std():.3f}")
    
    # Check constraint: ratios sum to 1
    ratio_sum = oof_ratios.sum(axis=1)
    print(f"\n  Ratio sum check: mean={ratio_sum.mean():.6f}, std={ratio_sum.std():.6f} (should be 1.0)")
    
    # Check constraint: Green + Dead + Clover = Total
    component_sum = oof_preds[:, 0] + oof_preds[:, 1] + oof_preds[:, 2]
    total_pred = oof_preds[:, 4]
    constraint_diff = np.abs(component_sum - total_pred)
    print(f"  G+D+C=T check: max_diff={constraint_diff.max():.6f}, mean_diff={constraint_diff.mean():.6f}")
    
    # Save OOF predictions
    oof_df = df[["sample_id_prefix", "fold"]].copy()
    oof_df["pred_green"] = oof_preds[:, 0]
    oof_df["pred_dead"] = oof_preds[:, 1]
    oof_df["pred_clover"] = oof_preds[:, 2]
    oof_df["pred_gdm"] = oof_preds[:, 3]
    oof_df["pred_total"] = oof_preds[:, 4]
    oof_df["true_green"] = oof_targets[:, 0]
    oof_df["true_dead"] = oof_targets[:, 1]
    oof_df["true_clover"] = oof_targets[:, 2]
    oof_df["true_gdm"] = oof_targets[:, 3]
    oof_df["true_total"] = oof_targets[:, 4]
    oof_df["ratio_green"] = oof_ratios[:, 0]
    oof_df["ratio_dead"] = oof_ratios[:, 1]
    oof_df["ratio_clover"] = oof_ratios[:, 2]
    
    oof_path = os.path.join(args.model_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved to: {oof_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.model_dir, "oof_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"OOF metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
