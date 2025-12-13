#!/usr/bin/env python3
"""
Inference script for 5-Head DINOv2 model with post-processing.

Usage:
    python -m src.inference_5head \
        --model-dir outputs/5head_frozen \
        --output submission.csv \
        --always-correct-dead
"""
import argparse
import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .device import DeviceType, empty_cache, get_amp_settings, get_device, get_device_type
from .models_5head import DeadPostProcessor, build_5head_model


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
    """Inference transforms."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_5head_models(
    model_dir: str,
    device: torch.device,
    num_folds: int = 5,
) -> Tuple[List[nn.Module], dict]:
    """Load all fold models and config."""
    
    # Load config
    config_path = os.path.join(model_dir, "results.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            results = json.load(f)
        config = results.get("config", {})
    else:
        config = {}
    
    # Extract model config with defaults matching new architecture
    backbone = config.get("backbone", "vit_base_patch14_reg4_dinov2.lvd142m")
    grid = tuple(config.get("grid", [2, 2]))
    dropout = config.get("dropout", 0.2)
    hidden_ratio = config.get("hidden_ratio", 0.5)
    use_film = config.get("use_film", True)
    use_attention_pool = config.get("use_attention_pool", True)
    
    print(f"Loading 5-head models:")
    print(f"  backbone={backbone}")
    print(f"  grid={grid}")
    print(f"  use_film={use_film}, use_attention_pool={use_attention_pool}")
    
    models = []
    for fold in range(num_folds):
        ckpt_path = os.path.join(model_dir, f"5head_best_fold{fold}.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue
        
        model = build_5head_model(
            backbone_name=backbone,
            grid=grid,
            pretrained=False,
            dropout=dropout,
            hidden_ratio=hidden_ratio,
            use_film=use_film,
            use_attention_pool=use_attention_pool,
        )
        
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        models.append(model)
        print(f"Loaded fold {fold}")
    
    print(f"Total models: {len(models)}")
    return models, config


def tta_flip_batch(
    x_left: torch.Tensor, 
    x_right: torch.Tensor,
    flip_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply flip augmentation for TTA."""
    if flip_type == "none":
        return x_left, x_right
    elif flip_type == "hflip":
        return torch.flip(x_left, dims=[-1]), torch.flip(x_right, dims=[-1])
    elif flip_type == "vflip":
        return torch.flip(x_left, dims=[-2]), torch.flip(x_right, dims=[-2])
    elif flip_type == "hvflip":
        return torch.flip(x_left, dims=[-2, -1]), torch.flip(x_right, dims=[-2, -1])
    else:
        return x_left, x_right


@torch.no_grad()
def predict_ensemble(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device,
    device_type: DeviceType,
    postprocessor: Optional[DeadPostProcessor] = None,
    use_tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with ensemble, optional TTA, and optional post-processing.
    
    Args:
        models: List of trained models
        loader: DataLoader for test data
        device: Target device
        device_type: Device type enum
        postprocessor: Optional dead value post-processor
        use_tta: Enable test-time augmentation (4x slower but more robust)
    
    Returns:
        raw_preds: (N, 5) - raw ensemble predictions
        post_preds: (N, 5) - post-processed predictions
    """
    use_amp, autocast_device, amp_dtype = get_amp_settings(device_type)
    
    tta_flips = ["none", "hflip", "vflip", "hvflip"] if use_tta else ["none"]
    
    all_raw_preds = []
    all_post_preds = []
    
    desc = "Predicting (TTA)" if use_tta else "Predicting"
    for x_left, x_right in tqdm(loader, desc=desc):
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)
        
        batch_preds = []
        
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            for flip_type in tta_flips:
                x_l_aug, x_r_aug = tta_flip_batch(x_left, x_right, flip_type)
                
                for model in models:
                    green, dead, clover, gdm, total = model(x_l_aug, x_r_aug)
                    preds = torch.cat([green, dead, clover, gdm, total], dim=1)
                    batch_preds.append(preds.float())
        
        # Average across models and TTA augmentations
        avg_preds = torch.stack(batch_preds, dim=0).mean(dim=0)  # (B, 5)
        all_raw_preds.append(avg_preds.cpu())
        
        # Apply post-processing
        if postprocessor is not None:
            green = avg_preds[:, 0]
            dead = avg_preds[:, 1]
            clover = avg_preds[:, 2]
            gdm = avg_preds[:, 3]
            total = avg_preds[:, 4]
            
            green_p, dead_p, clover_p, gdm_p, total_p = postprocessor(
                green, dead, clover, gdm, total
            )
            post_preds = torch.stack([green_p, dead_p, clover_p, gdm_p, total_p], dim=1)
            all_post_preds.append(post_preds.cpu())
        else:
            all_post_preds.append(avg_preds.cpu())
    
    raw_preds = torch.cat(all_raw_preds, dim=0).numpy()
    post_preds = torch.cat(all_post_preds, dim=0).numpy()
    
    return raw_preds, post_preds


def create_submission(
    preds: np.ndarray,
    test_df: pd.DataFrame,
    test_unique: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Create submission file."""
    target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    
    # Ensure non-negative
    preds = np.maximum(preds, 0)
    
    # Wide format
    wide = pd.DataFrame({
        "image_path": test_unique["image_path"].values,
        "Dry_Green_g": preds[:, 0],
        "Dry_Dead_g": preds[:, 1],
        "Dry_Clover_g": preds[:, 2],
        "GDM_g": preds[:, 3],
        "Dry_Total_g": preds[:, 4],
    })
    
    # Long format
    long_preds = wide.melt(
        id_vars=["image_path"],
        value_vars=target_cols,
        var_name="target_name",
        value_name="target",
    )
    
    # Merge
    submission = pd.merge(
        test_df[["sample_id", "image_path", "target_name"]],
        long_preds,
        on=["image_path", "target_name"],
        how="left",
    )[["sample_id", "target"]]
    
    submission["target"] = submission["target"].fillna(0).clip(lower=0)
    
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"Shape: {submission.shape}")
    
    return submission


def main():
    parser = argparse.ArgumentParser(description="5-Head Model Inference with Post-Processing and TTA")
    
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory with 5-head model checkpoints")
    parser.add_argument("--test-csv", type=str, default="./data/test.csv")
    parser.add_argument("--test-image-dir", type=str, default="./data/test")
    parser.add_argument("--output", type=str, default="submission_5head.csv")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # TTA
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (4x slower, more robust)")
    
    # Post-processing options
    parser.add_argument("--correction-threshold", type=float, default=0.15,
                        help="Error threshold to trigger dead correction")
    parser.add_argument("--always-correct-dead", action="store_true",
                        help="Always derive dead = total - gdm")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable post-processing")
    
    # Device
    parser.add_argument("--device-type", type=str, default=None,
                        choices=["cuda", "mps", "cpu"])
    
    args = parser.parse_args()
    
    # Device
    device_type = DeviceType(args.device_type) if args.device_type else get_device_type()
    device = get_device(device_type)
    
    print("=" * 60)
    print("5-Head Model Inference")
    print("=" * 60)
    print(f"Device: {device} ({device_type.value})")
    print(f"Model dir: {args.model_dir}")
    print(f"TTA: {args.tta}")
    print(f"Post-processing: {not args.no_postprocess}")
    if not args.no_postprocess:
        print(f"  Always correct dead: {args.always_correct_dead}")
        print(f"  Correction threshold: {args.correction_threshold}")
    print("=" * 60)
    
    # Load models
    models, config = load_5head_models(args.model_dir, device, args.num_folds)
    
    if len(models) == 0:
        raise RuntimeError("No models loaded!")
    
    # Post-processor
    if args.no_postprocess:
        postprocessor = None
    else:
        postprocessor = DeadPostProcessor(
            correction_threshold=args.correction_threshold,
            always_correct=args.always_correct_dead,
        )
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(args.test_csv)
    test_unique = test_df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    print(f"Total samples: {len(test_df)}, Unique images: {len(test_unique)}")
    
    # Dataset & loader
    transform = get_transforms(img_size=518)
    dataset = TestDataset(test_unique, args.test_image_dir, transform)
    
    pin_memory = device_type == DeviceType.CUDA
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    
    # Predict
    print("\nRunning inference...")
    raw_preds, post_preds = predict_ensemble(
        models, loader, device, device_type, postprocessor,
        use_tta=args.tta,
    )
    
    # Compare raw vs post-processed
    if postprocessor is not None:
        diff = np.abs(raw_preds - post_preds)
        dead_diff = diff[:, 1]  # Dead column
        print(f"\nPost-processing statistics:")
        print(f"  Dead predictions modified: {(dead_diff > 0.01).sum()} / {len(dead_diff)}")
        print(f"  Mean dead correction: {dead_diff.mean():.4f}")
        print(f"  Max dead correction: {dead_diff.max():.4f}")
    
    # Create submission (use post-processed)
    print("\nCreating submission...")
    create_submission(post_preds, test_df, test_unique, args.output)
    
    # Cleanup
    del models
    empty_cache(device_type)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

