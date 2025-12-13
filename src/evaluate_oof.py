#!/usr/bin/env python3
"""
Out-of-Fold (OOF) Evaluation Script.

Each fold model predicts ONLY on its validation set (data it never saw during training).
This gives an unbiased estimate of model performance.

Usage:
    python -m src.evaluate_oof --model-dir outputs/20241208_123456
"""
import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import BiomassDataset, prepare_dataframe, create_folds, get_valid_transforms
from .models import build_model
from .inference import auto_detect_config, get_tta_transforms


def compute_weighted_r2(
    preds: np.ndarray, 
    targets: np.ndarray, 
    weights: List[float] = [0.1, 0.1, 0.1, 0.2, 0.5]
) -> float:
    """Compute globally weighted R² as per competition metric."""
    weights = np.array(weights)
    n_samples = preds.shape[0]
    
    w = np.tile(weights, n_samples)
    y = targets.flatten()
    y_hat = preds.flatten()
    
    y_bar_w = np.sum(w * y) / np.sum(w)
    ss_res = np.sum(w * (y - y_hat) ** 2)
    ss_tot = np.sum(w * (y - y_bar_w) ** 2)
    
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0
    
    return float(1.0 - (ss_res / ss_tot))

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² for a single target with edge-case handling."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    y_bar = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_bar) ** 2))
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_weighted_mse_loss(
    preds: np.ndarray, targets: np.ndarray, weights: List[float] = [0.1, 0.1, 0.1, 0.2, 0.5]
) -> float:
    """
    Weighted MSE loss matching training loss (see src/trainer.py:WeightedMSELoss).
    Returns scalar: sum_i w_i * mse_i / sum(w).
    """
    w = np.array(weights, dtype=np.float64)
    mse = np.mean((preds.astype(np.float64) - targets.astype(np.float64)) ** 2, axis=0)  # (5,)
    return float(np.sum(w * mse) / np.sum(w))


def compute_per_target_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
    target_weights: List[float],
) -> List[Dict[str, float]]:
    """Compute per-target metrics for tracking which labels are weak."""
    preds_f = preds.astype(np.float64)
    targets_f = targets.astype(np.float64)
    mse = np.mean((preds_f - targets_f) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_f - targets_f), axis=0)
    r2 = np.array([_safe_r2(targets_f[:, i], preds_f[:, i]) for i in range(preds_f.shape[1])], dtype=np.float64)
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


def _get_model_cfg(model_dir: str) -> dict:
    """
    Get model config for a model directory (prefers results.json).
    Returns keys: backbone, grid, img_size, dropout, hidden_ratio.
    """
    cfg = auto_detect_config(model_dir) or {}
    return {
        "backbone": cfg.get("backbone", "vit_base_patch14_reg4_dinov2.lvd142m"),
        "grid": tuple(cfg.get("grid", (2, 2))),
        "img_size": int(cfg.get("img_size", 518)),
        "dropout": float(cfg.get("dropout", 0.30)),
        "hidden_ratio": float(cfg.get("hidden_ratio", 0.25)),
    }


def load_model(checkpoint_path: str, device: torch.device, model_cfg: dict) -> nn.Module:
    """Load a trained model from checkpoint using provided model config."""
    model = build_model(
        backbone_name=model_cfg["backbone"],
        model_type="tiled_film",
        grid=tuple(model_cfg["grid"]),
        pretrained=False,
        dropout=float(model_cfg["dropout"]),
        hidden_ratio=float(model_cfg["hidden_ratio"]),
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def predict_fold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions for a fold's validation set."""
    all_preds = []
    all_targets = []
    
    for x_left, x_right, targets in tqdm(loader, desc="Predicting", leave=False):
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            green, dead, clover, gdm, total = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        all_preds.append(preds.float().cpu().numpy())
        all_targets.append(targets.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_targets)

def _parse_weights(weights: Optional[List[float]], n: int) -> np.ndarray:
    """Parse and normalize ensemble weights."""
    if weights is None or len(weights) == 0:
        w = np.ones(n, dtype=np.float64) / float(n)
        return w
    if len(weights) != n:
        raise ValueError(f"--weights length ({len(weights)}) must match number of model dirs ({n})")
    w = np.array(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("--weights must be non-negative")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("--weights sum must be > 0")
    return w / s


def evaluate_oof(
    model_dirs: List[str],
    data_path: str = "/root/workspace/data/train.csv",
    image_dir: str = "/root/workspace/data/train",
    num_folds: int = 5,
    batch_size: int = 8,
    num_workers: int = 4,
    cv_strategy: str = "group_date_state",
    seed: int = 18,
    weights: Optional[List[float]] = None,
    tta: bool = False,
    img_size: Optional[int] = None,
) -> Dict:
    """
    Evaluate all folds using Out-of-Fold predictions.
    
    Each fold model predicts on its validation set only.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model dirs: {model_dirs}")
    w = _parse_weights(weights, len(model_dirs))
    print(f"Ensemble weights: {w.tolist()}")

    # Auto-detect per-model configs (backbone/grid/etc.)
    model_cfgs = [_get_model_cfg(md) for md in model_dirs]
    if img_size is None:
        img_size = int(model_cfgs[0]["img_size"]) if model_cfgs else 518
    # Ensure compatible resize across ensembled model dirs
    bad = [(md, cfg["img_size"]) for md, cfg in zip(model_dirs, model_cfgs) if int(cfg["img_size"]) != int(img_size)]
    if bad:
        raise ValueError(
            f"Ensembled model-dirs have different img_size than requested img_size={img_size}: {bad}. "
            "Re-train to a common img_size or run OOF per-model-dir."
        )

    print("Model configs:")
    for md, cfg in zip(model_dirs, model_cfgs):
        print(f"  - {md}: backbone={cfg['backbone']} grid={cfg['grid']} img_size={img_size}")
    
    # Load and prepare data
    df = prepare_dataframe(data_path)
    df = create_folds(df, n_folds=num_folds, seed=seed, cv_strategy=cv_strategy)
    
    base_transform = get_valid_transforms(img_size=int(img_size))
    tta_transforms = get_tta_transforms(img_size=int(img_size)) if tta else None
    
    # Collect OOF predictions (ensemble)
    oof_preds = np.zeros((len(df), 5), dtype=np.float32)
    oof_targets = np.zeros((len(df), 5))
    fold_results = []
    
    for fold in range(num_folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold}")
        print(f"{'='*50}")

        # Get validation data for this fold
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        val_indices = df[df["fold"] == fold].index.tolist()
        print(f"  Validation samples: {len(val_df)}")
        
        # Loader is recreated per transform (for TTA) to keep implementation simple/explicit.

        # Predict per model dir, then blend
        per_model_preds: List[np.ndarray] = []
        targets: Optional[np.ndarray] = None
        for md, cfg in zip(model_dirs, model_cfgs):
            ckpt_path = os.path.join(md, f"tiled_film_best_model_fold{fold}.pth")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            model = load_model(ckpt_path, device, cfg)
            print(f"  Loaded: {ckpt_path}")

            if tta_transforms is None:
                val_ds = BiomassDataset(val_df, image_dir, base_transform, is_train=False)
                val_loader = DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                preds_i, targets_i = predict_fold(model, val_loader, device)
            else:
                view_preds: List[np.ndarray] = []
                targets_i = None
                for ti, tform in enumerate(tta_transforms):
                    if ti == 0:
                        print(f"    TTA views: {len(tta_transforms)}")
                    val_ds = BiomassDataset(val_df, image_dir, tform, is_train=False)
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                    p_view, t_view = predict_fold(model, val_loader, device)
                    view_preds.append(p_view)
                    if targets_i is None:
                        targets_i = t_view
                preds_i = np.mean(np.stack(view_preds, axis=0), axis=0)
                assert targets_i is not None

            per_model_preds.append(preds_i)
            if targets is None:
                targets = targets_i

            # Cleanup per model
            del model
            torch.cuda.empty_cache()

        assert targets is not None
        blended = np.zeros_like(per_model_preds[0], dtype=np.float64)
        for wi, pi in zip(w, per_model_preds):
            blended += float(wi) * pi.astype(np.float64)
        preds = blended.astype(np.float32)
        
        # Store OOF predictions
        oof_preds[val_indices] = preds
        oof_targets[val_indices] = targets
        
        # Per-fold metrics
        fold_r2 = compute_weighted_r2(preds, targets)
        fold_results.append({"fold": fold, "r2": fold_r2, "n_samples": len(val_df)})
        print(f"  Fold {fold} R²: {fold_r2:.4f}")
    
    # Overall OOF metrics
    print(f"\n{'='*50}")
    print("Overall OOF Results")
    print(f"{'='*50}")
    
    overall_r2 = compute_weighted_r2(oof_preds, oof_targets)
    overall_wmse = compute_weighted_mse_loss(oof_preds, oof_targets)
    
    # Per-target RMSE
    target_names = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    
    per_target = compute_per_target_metrics(oof_preds, oof_targets, target_names, target_weights)

    print("\nPer-Target Metrics:")
    print(f"{'Target':<15} {'W':<5} {'R2':<9} {'RMSE':<11} {'MAE':<11} {'MSE':<13} {'w*MSE':<13}")
    print("-" * 80)
    for row in per_target:
        print(
            f"{row['target']:<15} "
            f"{row['weight']:<5.2f} "
            f"{row['r2']:<9.4f} "
            f"{row['rmse']:<11.4f} "
            f"{row['mae']:<11.4f} "
            f"{row['mse']:<13.4f} "
            f"{row['weighted_mse_component']:<13.4f}"
        )
    
    print("\nFold Summary:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: R² = {r['r2']:.4f} (n={r['n_samples']})")
    
    fold_r2s = [r["r2"] for r in fold_results]
    print(f"\nCV Mean R²: {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")
    print(f"Overall OOF R²: {overall_r2:.4f}")
    print(f"Overall OOF weighted MSE (loss): {overall_wmse:.6f}")
    
    # Save OOF predictions
    oof_df = df.copy()
    for i, name in enumerate(target_names):
        oof_df[f"pred_{name}"] = oof_preds[:, i]
    
    # Save OOF predictions to first model_dir (or current working dir fallback)
    out_dir = model_dirs[0] if len(model_dirs) > 0 else "."
    oof_path = os.path.join(out_dir, "oof_predictions_ensemble.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved to: {oof_path}")
    
    return {
        "overall_r2": overall_r2,
        "overall_weighted_mse": overall_wmse,
        "cv_mean_r2": np.mean(fold_r2s),
        "cv_std_r2": np.std(fold_r2s),
        "fold_results": fold_results,
        "model_dirs": model_dirs,
        "weights": w.tolist(),
        "per_target_metrics": per_target,
    }


def main():
    parser = argparse.ArgumentParser(description="OOF Evaluation")
    parser.add_argument(
        "--model-dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories with fold checkpoints (ensemble if multiple).",
    )
    parser.add_argument("--data-path", type=str, default="/root/workspace/data/train.csv")
    parser.add_argument("--image-dir", type=str, default="/root/workspace/data/train")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=None, help="Resize size for eval (defaults to results.json img_size).")
    parser.add_argument("--tta", action="store_true", help="Enable TTA during OOF evaluation (mirrors src.inference.py).")
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default="group_date_state",
        choices=[
            "group_month",
            "group_date",
            "group_date_state",
            "group_date_state_bin",
            "stratified",
            "random",
        ],
        help="CV strategy used to generate folds (should match training). Note: group_date_state is month-grouped and state-stratified.",
    )
    parser.add_argument("--seed", type=int, default=18, help="Seed used to generate folds (should match training).")
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional non-negative ensemble weights (must match number of --model-dir). If omitted, uniform.",
    )
    
    args = parser.parse_args()
    
    results = evaluate_oof(
        model_dirs=args.model_dir,
        data_path=args.data_path,
        image_dir=args.image_dir,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cv_strategy=args.cv_strategy,
        seed=args.seed,
        weights=args.weights,
        tta=args.tta,
        img_size=args.img_size,
    )
    
    print(f"\n{'='*50}")
    print(f"Final OOF R²: {results['overall_r2']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

