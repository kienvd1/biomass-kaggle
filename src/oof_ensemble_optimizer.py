#!/usr/bin/env python3
"""
OOF Ensemble Weight Optimizer for DINOv3 Models.

This script matches exactly how inference is done:
1. Loads trained DINOv3 models from multiple directories
2. Generates OOF (out-of-fold) predictions with TTA (same as inference)
3. Optimizes ensemble weights using scipy.optimize
4. Outputs optimal weights and expected ensemble score

Usage:
    python src/oof_ensemble_optimizer.py \
        --model-dirs outputs/dinov3_grid2 outputs/dinov3_grid3 \
        --data-dir data \
        --output-dir outputs/ensemble_analysis \
        --tta-level default
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from device import DeviceType, get_device_and_type


# ============== Target Weights (Competition Metric) ==============
TARGET_WEIGHTS = np.array([0.30, 0.18, 0.18, 0.17, 0.17])  # Green, Dead, Clover, GDM, Total
TARGET_NAMES = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


# ============== Dataset (same as inference) ==============
class BiomassDataset(Dataset):
    """Dataset for OOF prediction - matches inference exactly."""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        image_dir: str, 
        transform: A.Compose,
        return_targets: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.return_targets = return_targets
        
        # Get unique images
        self.unique_df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
        
        # Create target lookup if needed
        if return_targets:
            self._build_target_lookup()
    
    def _build_target_lookup(self) -> None:
        """Build lookup from image_path to targets."""
        self.target_lookup: Dict[str, Dict[str, float]] = {}
        for _, row in self.df.iterrows():
            img_path = row["image_path"]
            if img_path not in self.target_lookup:
                self.target_lookup[img_path] = {}
            self.target_lookup[img_path][row["target_name"]] = row["target"]
    
    def __len__(self) -> int:
        return len(self.unique_df)
    
    def __getitem__(self, idx: int) -> Tuple:
        row = self.unique_df.iloc[idx]
        img_path = row["image_path"]
        
        # Load image
        full_path = os.path.join(self.image_dir, os.path.basename(img_path))
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Split stereo
        H, W = img.shape[:2]
        mid = W // 2
        img_left = img[:, :mid, :]
        img_right = img[:, mid:, :]
        
        # Apply transforms
        if self.transform:
            aug_left = self.transform(image=img_left)
            aug_right = self.transform(image=img_right)
            img_left = aug_left["image"]
            img_right = aug_right["image"]
        
        if self.return_targets:
            targets = np.array([
                self.target_lookup[img_path].get(name, 0.0) 
                for name in TARGET_NAMES
            ], dtype=np.float32)
            return img_left, img_right, targets, img_path
        
        return img_left, img_right, img_path


# ============== Transforms (same as inference) ==============
def get_val_transform(img_size: int) -> A.Compose:
    """Get validation transform."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int, level: str = "default") -> List[A.Compose]:
    """
    TTA transforms with configurable levels (matches inference).
    
    Args:
        img_size: Image size
        level: TTA level - "none", "light", "default", "heavy", "extreme"
    
    Returns:
        List of albumentations Compose transforms
    """
    # Base transform (always included)
    base = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    if level == "none":
        return [base]
    
    # Horizontal flip
    hflip = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    if level == "light":
        return [base, hflip]
    
    # Brightness/Contrast adjustment (slightly brighter)
    bright = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.RandomBrightnessContrast(brightness_limit=(0.08, 0.12), contrast_limit=(0.08, 0.12), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    if level == "default":
        return [base, hflip, bright]
    
    # Vertical flip
    vflip = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    if level == "heavy":
        return [base, hflip, bright, vflip]
    
    # Both flips
    hvflip = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Darker version
    dark = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.RandomBrightnessContrast(brightness_limit=(-0.12, -0.08), contrast_limit=(-0.08, -0.04), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    if level == "extreme":
        return [base, hflip, bright, vflip, hvflip, dark]
    
    # default fallback
    return [base, hflip, bright]


# ============== Model Loading (same as inference) ==============
def load_model_config(model_dir: str) -> Dict:
    """Load model config from results.json."""
    results_path = os.path.join(model_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        return results.get("config", {})
    return {}


def load_folds_info(model_dir: str) -> pd.DataFrame:
    """Load folds.csv from model directory."""
    folds_path = os.path.join(model_dir, "folds.csv")
    if os.path.exists(folds_path):
        return pd.read_csv(folds_path)
    raise FileNotFoundError(f"folds.csv not found in {model_dir}")


def find_checkpoints(model_dir: str) -> Dict[int, str]:
    """Find all fold checkpoints in model directory."""
    checkpoints = {}
    for f in os.listdir(model_dir):
        if f.startswith("dinov3_best_fold") and f.endswith(".pth"):
            fold_num = int(f.replace("dinov3_best_fold", "").replace(".pth", ""))
            checkpoints[fold_num] = os.path.join(model_dir, f)
    return checkpoints


def _strip_module_prefix(sd: dict) -> dict:
    """Remove 'module.' prefix from state dict keys (for DDP-trained models)."""
    if not sd:
        return sd
    keys = list(sd.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _filter_depth_model_keys(sd: dict) -> dict:
    """Filter out embedded depth model weights from old checkpoints."""
    if not sd:
        return sd
    filtered = {}
    for k, v in sd.items():
        if "._depth_model." not in k:
            filtered[k] = v
    return filtered


def load_dinov3_model(
    checkpoint_path: str,
    config: Dict,
    device: torch.device,
    depth_model_path: Optional[str] = None,
) -> nn.Module:
    """Load DINOv3Direct model from checkpoint (same as inference)."""
    # Import here to avoid circular imports
    from dinov3_models import DINOv3Direct
    
    # Extract config
    grid_val = config.get("grid", 2)
    grid = (grid_val, grid_val) if isinstance(grid_val, int) else tuple(grid_val)
    
    model = DINOv3Direct(
        grid=grid,
        pretrained=False,
        dropout=config.get("dropout", 0.3),
        hidden_ratio=config.get("hidden_ratio", 0.25),
        use_film=config.get("use_film", True),
        use_attention_pool=config.get("use_attention_pool", True),
        train_dead=config.get("train_dead", False),
        train_clover=config.get("train_clover", False),
        use_vegetation_indices=config.get("use_vegetation_indices", False),
        use_disparity=config.get("use_disparity", False),
        use_depth=config.get("use_depth", False),
        depth_model_size=config.get("depth_model_size", "small"),
        use_depth_attention=config.get("depth_attention", False),
        use_learnable_aug=config.get("use_learnable_aug", False),
        depth_model_path=depth_model_path,
    )
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _strip_module_prefix(state_dict)
    state_dict = _filter_depth_model_keys(state_dict)
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


# ============== OOF Prediction (same as inference) ==============
@torch.no_grad()
def predict_one_view(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run prediction for a single TTA view with fold ensemble (same as inference)."""
    all_preds = []
    all_targets = []
    all_paths = []
    
    for batch in tqdm(loader, desc="    Predicting", leave=False):
        x_left, x_right, targets, paths = batch
        x_left = x_left.to(device)
        x_right = x_right.to(device)
        
        # Fold ensemble (same as inference)
        fold_preds = []
        for model in models:
            outputs = model(x_left, x_right)
            # Handle both old (5 outputs) and new (6 outputs with aux_loss) models
            if len(outputs) == 6:
                green, dead, clover, gdm, total, _ = outputs
            else:
                green, dead, clover, gdm, total = outputs
            pred = torch.cat([green, dead, clover, gdm, total], dim=1)
            fold_preds.append(pred.float().cpu().numpy())
        
        # Average across folds
        avg_pred = np.mean(fold_preds, axis=0)
        all_preds.append(avg_pred)
        all_targets.append(targets.numpy())
        all_paths.extend(paths)
    
    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        all_paths,
    )


def run_inference_tta(
    models: List[nn.Module],
    val_df: pd.DataFrame,
    image_dir: str,
    img_size: int,
    device: torch.device,
    tta_level: str = "default",
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run inference with TTA (same as inference notebook)."""
    transforms = get_tta_transforms(img_size, level=tta_level)
    print(f"  TTA level: {tta_level} ({len(transforms)} views)")
    
    all_view_preds = []
    targets = None
    paths = None
    
    for i, transform in enumerate(transforms):
        print(f"    View {i+1}/{len(transforms)}...")
        
        # Create dataset with this transform
        ds = BiomassDataset(val_df, image_dir, transform, return_targets=True)
        
        # batch_size=1 as requested
        loader = DataLoader(
            ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
        )
        
        preds, tgt, img_paths = predict_one_view(models, loader, device)
        all_view_preds.append(preds)
        
        if targets is None:
            targets = tgt
            paths = img_paths
    
    # Average across TTA views
    final_preds = np.mean(all_view_preds, axis=0)
    return final_preds, targets, paths


def generate_oof_predictions(
    model_dir: str,
    data_dir: str,
    device: torch.device,
    tta_level: str = "default",
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate OOF predictions for all folds of a model.
    Uses TTA and batch_size=1 (same as inference).
    
    Returns:
        oof_preds: (N, 5) OOF predictions
        targets: (N, 5) ground truth targets
        image_paths: List of image paths
    """
    print(f"\n{'='*60}")
    print(f"Generating OOF predictions: {os.path.basename(model_dir)}")
    print(f"{'='*60}")
    
    # Load config
    config = load_model_config(model_dir)
    grid_val = config.get("grid", 2)
    img_size = config.get("img_size", 576)
    print(f"Grid: {grid_val}x{grid_val}, Image size: {img_size}")
    
    # Load folds info
    folds_df = load_folds_info(model_dir)
    print(f"Loaded folds.csv with {len(folds_df)} samples")
    
    # Load train data
    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    image_dir = os.path.join(data_dir, "train")
    
    # Find checkpoints
    checkpoints = find_checkpoints(model_dir)
    print(f"Found {len(checkpoints)} fold checkpoints: {list(checkpoints.keys())}")
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_dir}")
    
    # Collect OOF predictions
    all_oof_preds = {}  # image_path -> prediction
    all_targets = {}    # image_path -> targets
    
    for fold_num, ckpt_path in sorted(checkpoints.items()):
        print(f"\nFold {fold_num}:")
        
        # Get validation samples for this fold
        val_images = folds_df[folds_df["fold"] == fold_num]["image_path"].tolist()
        val_df = train_df[train_df["image_path"].isin(val_images)]
        
        if len(val_df) == 0:
            print(f"  Warning: No validation samples for fold {fold_num}")
            continue
        
        n_unique = len(val_df.drop_duplicates("image_path"))
        print(f"  Validation samples: {n_unique} images")
        
        # Load model for this fold only (same as inference uses all folds)
        # For OOF, we use only the fold that was NOT trained on this data
        model = load_dinov3_model(ckpt_path, config, device)
        print(f"  Loaded: {os.path.basename(ckpt_path)}")
        
        # Run inference with TTA (batch_size=1)
        preds, targets, paths = run_inference_tta(
            models=[model],  # Single model for this fold
            val_df=val_df,
            image_dir=image_dir,
            img_size=img_size,
            device=device,
            tta_level=tta_level,
            num_workers=num_workers,
        )
        
        # Store predictions
        for i, path in enumerate(paths):
            all_oof_preds[path] = preds[i]
            all_targets[path] = targets[i]
        
        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Combine all predictions in consistent order
    all_paths = sorted(all_oof_preds.keys())
    oof_preds = np.array([all_oof_preds[p] for p in all_paths])
    targets = np.array([all_targets[p] for p in all_paths])
    
    print(f"\nTotal OOF samples: {len(oof_preds)}")
    
    return oof_preds, targets, all_paths


# ============== Metrics ==============
def compute_weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Compute weighted R² (competition metric)."""
    r2_scores = {}
    for i, name in enumerate(TARGET_NAMES):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        r2_scores[name] = r2
    
    weighted_r2 = np.average(list(r2_scores.values()), weights=TARGET_WEIGHTS)
    return weighted_r2, r2_scores


# ============== Weight Optimization ==============
def optimize_weights(
    oof_preds_list: List[np.ndarray],
    targets: np.ndarray,
    model_names: List[str],
) -> Tuple[np.ndarray, float]:
    """
    Optimize ensemble weights using scipy.optimize.
    """
    n_models = len(oof_preds_list)
    
    def objective(w: np.ndarray) -> float:
        """Negative weighted R² (minimize)."""
        w = np.abs(w)
        w = w / (w.sum() + 1e-8)
        
        ensemble_pred = sum(pred * weight for pred, weight in zip(oof_preds_list, w))
        score, _ = compute_weighted_r2(targets, ensemble_pred)
        return -score
    
    # Initial: equal weights
    x0 = np.ones(n_models) / n_models
    
    # Bounds: 0 to 1
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    
    # Constraint: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8},
    )
    
    # Normalize
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()
    best_score = -result.fun
    
    return optimal_weights, best_score


def grid_search_weights_2models(
    oof_preds_list: List[np.ndarray],
    targets: np.ndarray,
    n_steps: int = 101,
) -> Tuple[np.ndarray, float]:
    """Simple grid search for 2 models."""
    assert len(oof_preds_list) == 2, "Grid search only supports 2 models"
    
    best_score = -np.inf
    best_w1 = 0.5
    
    for w1 in np.linspace(0, 1, n_steps):
        w2 = 1 - w1
        pred = oof_preds_list[0] * w1 + oof_preds_list[1] * w2
        score, _ = compute_weighted_r2(targets, pred)
        
        if score > best_score:
            best_score = score
            best_w1 = w1
    
    return np.array([best_w1, 1 - best_w1]), best_score


# ============== Main ==============
def main() -> Dict:
    parser = argparse.ArgumentParser(description="OOF Ensemble Weight Optimizer")
    parser.add_argument(
        "--model-dirs", 
        type=str, 
        nargs="+", 
        required=True,
        help="Model directories to ensemble",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory with train.csv and train/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ensemble_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--tta-level",
        type=str,
        default="default",
        choices=["none", "light", "default", "heavy", "extreme"],
        help="TTA level (default: default = base + hflip + bright)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu")
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device, _ = get_device_and_type()
    print(f"Device: {device}")
    print(f"TTA level: {args.tta_level}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate OOF predictions for each model
    model_names = [os.path.basename(d) for d in args.model_dirs]
    oof_preds_list = []
    targets = None
    image_paths = None
    
    for model_dir in args.model_dirs:
        preds, tgt, paths = generate_oof_predictions(
            model_dir,
            args.data_dir,
            device,
            tta_level=args.tta_level,
            num_workers=args.num_workers,
        )
        oof_preds_list.append(preds)
        
        if targets is None:
            targets = tgt
            image_paths = paths
        else:
            # Verify consistency
            assert np.allclose(targets, tgt), "Targets mismatch between models!"
    
    # Save OOF predictions
    for name, preds in zip(model_names, oof_preds_list):
        np.save(os.path.join(args.output_dir, f"oof_{name}.npy"), preds)
    np.save(os.path.join(args.output_dir, "targets.npy"), targets)
    with open(os.path.join(args.output_dir, "image_paths.json"), "w") as f:
        json.dump(image_paths, f)
    print(f"\nSaved OOF predictions to {args.output_dir}")
    
    # Compute individual model scores
    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL SCORES")
    print(f"{'='*60}")
    
    for name, preds in zip(model_names, oof_preds_list):
        score, per_target = compute_weighted_r2(targets, preds)
        print(f"\n{name}:")
        print(f"  Weighted R²: {score:.4f}")
        for tname, r2 in per_target.items():
            print(f"    {tname}: {r2:.4f}")
    
    # Optimize weights
    print(f"\n{'='*60}")
    print("WEIGHT OPTIMIZATION")
    print(f"{'='*60}")
    
    # Equal weights baseline
    equal_pred = sum(preds / len(oof_preds_list) for preds in oof_preds_list)
    equal_score, _ = compute_weighted_r2(targets, equal_pred)
    print(f"\nEqual weights baseline: {equal_score:.4f}")
    
    # Optimized weights
    if len(oof_preds_list) == 2:
        # Use grid search for 2 models (more thorough)
        optimal_weights, best_score = grid_search_weights_2models(oof_preds_list, targets)
    else:
        optimal_weights, best_score = optimize_weights(oof_preds_list, targets, model_names)
    
    print(f"\nOptimal weights:")
    for name, w in zip(model_names, optimal_weights):
        print(f"  {name}: {w:.4f} ({w*100:.1f}%)")
    print(f"\nOptimal ensemble score: {best_score:.4f}")
    print(f"Improvement over equal: {(best_score - equal_score)*100:.2f}%")
    
    # Save results
    results = {
        "model_dirs": args.model_dirs,
        "model_names": model_names,
        "tta_level": args.tta_level,
        "optimal_weights": optimal_weights.tolist(),
        "optimal_score": best_score,
        "equal_weights_score": equal_score,
        "individual_scores": {},
    }
    
    for name, preds in zip(model_names, oof_preds_list):
        score, per_target = compute_weighted_r2(targets, preds)
        results["individual_scores"][name] = {
            "weighted_r2": score,
            "per_target": per_target,
        }
    
    results_path = os.path.join(args.output_dir, "ensemble_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Print config for inference notebook
    print(f"\n{'='*60}")
    print("COPY TO INFERENCE NOTEBOOK")
    print(f"{'='*60}")
    print("MODEL_CONFIGS = [")
    for model_dir, name, w in zip(args.model_dirs, model_names, optimal_weights):
        config = load_model_config(model_dir)
        grid = config.get("grid", 2)
        img_size = config.get("img_size", 576)
        print(f"    ModelConfig(")
        print(f'        model_dir="{model_dir}",')
        print(f"        weight={w:.4f},")
        print(f"        grid=({grid}, {grid}),")
        print(f"        img_size={img_size},")
        print(f"    ),")
    print("]")
    
    return results


if __name__ == "__main__":
    main()
