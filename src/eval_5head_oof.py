#!/usr/bin/env python3
"""
Out-of-Fold (OOF) Evaluation for 5-Head Model.

This script performs proper OOF evaluation:
- Each sample is predicted using only the model that was NOT trained on it
- Computes per-target and weighted R² metrics
- Applies Dead derivation fix: Dead = Total - GDM

Usage:
    python -m src.eval_5head_oof --model-dir ./outputs/5head_20251213_170453
    python -m src.eval_5head_oof --model-dir ./outputs/5head_20251213_170453 --derive-dead
"""
import argparse
import gc
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm


# ============== Dataset ==============

class OOFDataset(Dataset):
    """Dataset for OOF inference."""

    def __init__(self, df: pd.DataFrame, image_dir: str, transform: A.Compose) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.paths = self.df["image_path"].values
        self.targets = self.df[
            ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
        ].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)

        return left_t, right_t, targets


def get_val_transform(img_size: int = 518) -> A.Compose:
    """Validation transform (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============== Model ==============

def _infer_input_res(m) -> int:
    if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "img_size"):
        isz = m.patch_embed.img_size
        return int(isz if isinstance(isz, (int, float)) else isz[0])
    if hasattr(m, "img_size"):
        isz = m.img_size
        return int(isz if isinstance(isz, (int, float)) else isz[0])
    dc = getattr(m, "default_cfg", {}) or {}
    ins = dc.get("input_size", None)
    if ins:
        if isinstance(ins, (tuple, list)) and len(ins) >= 2:
            return int(ins[1])
        return int(ins if isinstance(ins, (int, float)) else 224)
    return 518


def _build_dino_by_name(name: str, pretrained: bool = False):
    m = timm.create_model(name, pretrained=pretrained, num_classes=0)
    feat = m.num_features
    input_res = _infer_input_res(m)
    return m, feat, input_res


def _make_edges(L: int, parts: int) -> List[Tuple[int, int]]:
    step = L // parts
    edges = []
    start = 0
    for _ in range(parts - 1):
        edges.append((start, start + step))
        start += step
    edges.append((start, L))
    return edges


class FiLM(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        hidden = max(64, in_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim * 2),
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.mlp(context)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma, beta


class AttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x.mean(dim=1, keepdim=True))
        k = self.key(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ x).squeeze(1)
        return out


NUM_STATES = 4
NUM_MONTHS = 10


class FiveHeadDINO(nn.Module):
    """5-Head DINOv2 model."""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        grid: Tuple[int, int] = (2, 2),
        pretrained: bool = False,
        dropout: float = 0.2,
        hidden_ratio: float = 0.5,
        use_film: bool = True,
        use_attention_pool: bool = True,
        use_aux_heads: bool = False,
    ) -> None:
        super().__init__()

        self.backbone, feat_dim, input_res = _build_dino_by_name(backbone_name, pretrained)
        self.used_backbone_name = backbone_name
        self.input_res = int(input_res)
        self.feat_dim = feat_dim
        self.grid = tuple(grid)
        self.use_film = use_film
        self.use_attention_pool = use_attention_pool
        self.use_aux_heads = use_aux_heads

        if use_film:
            self.film_left = FiLM(feat_dim)
            self.film_right = FiLM(feat_dim)

        if use_attention_pool:
            self.attn_pool_left = AttentionPooling(feat_dim)
            self.attn_pool_right = AttentionPooling(feat_dim)

        self.combined_dim = feat_dim * 2
        hidden_dim = max(64, int(self.combined_dim * hidden_ratio))
        self.hidden_dim = hidden_dim

        self.shared_proj = nn.Sequential(
            nn.LayerNorm(self.combined_dim),
            nn.Linear(self.combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        def _make_head(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(in_dim, 1),
            )

        self.head_green = _make_head(hidden_dim)
        self.head_dead = _make_head(hidden_dim)
        self.head_clover = _make_head(hidden_dim)
        self.head_gdm = _make_head(hidden_dim)
        self.head_total = _make_head(hidden_dim)

        if use_aux_heads:
            self.head_state = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, NUM_STATES),
            )
            self.head_month = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, NUM_MONTHS),
            )

        self.softplus = nn.Softplus(beta=1.0)

    def _collect_tiles(self, x: torch.Tensor) -> List[torch.Tensor]:
        _, C, H, W = x.shape
        r, c = self.grid
        rows = _make_edges(H, r)
        cols = _make_edges(W, c)
        tiles = []
        for rs, re in rows:
            for cs, ce in cols:
                xt = x[:, :, rs:re, cs:ce]
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res),
                                       mode="bilinear", align_corners=False)
                tiles.append(xt)
        return tiles

    def _extract_tiles_fused(self, x_left: torch.Tensor, x_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_left.size(0)
        tiles_left = self._collect_tiles(x_left)
        tiles_right = self._collect_tiles(x_right)
        num_tiles = len(tiles_left)

        all_tiles = torch.cat(tiles_left + tiles_right, dim=0)
        all_feats = self.backbone(all_tiles)

        total_tiles = 2 * num_tiles
        all_feats = all_feats.view(total_tiles, B, -1).permute(1, 0, 2)
        feats_left = all_feats[:, :num_tiles, :]
        feats_right = all_feats[:, num_tiles:, :]
        return feats_left, feats_right

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        tiles_left, tiles_right = self._extract_tiles_fused(x_left, x_right)

        ctx_left = tiles_left.mean(dim=1)
        ctx_right = tiles_right.mean(dim=1)

        if self.use_film:
            gamma_l, beta_l = self.film_left(ctx_right)
            gamma_r, beta_r = self.film_right(ctx_left)
            tiles_left = tiles_left * (1 + gamma_l.unsqueeze(1)) + beta_l.unsqueeze(1)
            tiles_right = tiles_right * (1 + gamma_r.unsqueeze(1)) + beta_r.unsqueeze(1)

        if self.use_attention_pool:
            f_l = self.attn_pool_left(tiles_left)
            f_r = self.attn_pool_right(tiles_right)
        else:
            f_l = tiles_left.mean(dim=1)
            f_r = tiles_right.mean(dim=1)

        f = torch.cat([f_l, f_r], dim=1)
        f = self.shared_proj(f)

        green = self.softplus(self.head_green(f))
        dead = self.softplus(self.head_dead(f))
        clover = self.softplus(self.head_clover(f))
        gdm = self.softplus(self.head_gdm(f))
        total = self.softplus(self.head_total(f))

        return green, dead, clover, gdm, total


def _strip_module_prefix(sd: dict) -> dict:
    if not sd:
        return sd
    keys = list(sd.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _detect_model_config(sd_keys: set) -> dict:
    return {
        "use_film": any(k.startswith("film_left.") or k.startswith("film_right.") for k in sd_keys),
        "use_attention_pool": any(k.startswith("attn_pool_left.") or k.startswith("attn_pool_right.") for k in sd_keys),
        "use_aux_heads": any(k.startswith("head_state.") or k.startswith("head_month.") for k in sd_keys),
    }


def load_model(
    path: str,
    backbone_name: str,
    device: torch.device,
    grid: Tuple[int, int] = (2, 2),
    dropout: float = 0.2,
    hidden_ratio: float = 0.5,
) -> FiveHeadDINO:
    """Load a 5-head model checkpoint."""
    try:
        raw_sd = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        raw_sd = torch.load(path, map_location=device)

    sd = _strip_module_prefix(raw_sd)
    detected = _detect_model_config(set(sd.keys()))

    model = FiveHeadDINO(
        backbone_name=backbone_name,
        grid=grid,
        pretrained=False,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
        use_film=detected["use_film"],
        use_attention_pool=detected["use_attention_pool"],
        use_aux_heads=detected["use_aux_heads"],
    )

    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    return model


# ============== Inference ==============

@torch.no_grad()
def predict_fold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference for a single fold."""
    model.eval()
    all_preds = []
    all_targets = []

    use_amp = device.type == "cuda"
    amp_device = "cuda" if use_amp else "cpu"

    for x_left, x_right, targets in tqdm(loader, desc="  Predicting", leave=False):
        x_left = x_left.to(device, non_blocking=True)
        x_right = x_right.to(device, non_blocking=True)

        with torch.amp.autocast(amp_device, enabled=use_amp):
            green, dead, clover, gdm, total = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            preds = torch.clamp(preds, min=0.0)

        all_preds.append(preds.float().cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    target_names: List[str] = ["Green", "Dead", "Clover", "GDM", "Total"],
    target_weights: List[float] = [0.1, 0.1, 0.1, 0.2, 0.5],
) -> Dict[str, float]:
    """Compute per-target and weighted metrics."""
    metrics = {}

    for i, name in enumerate(target_names):
        r2 = r2_score(targets[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
        mae = np.mean(np.abs(targets[:, i] - preds[:, i]))
        metrics[f"r2_{name.lower()}"] = r2
        metrics[f"rmse_{name.lower()}"] = rmse
        metrics[f"mae_{name.lower()}"] = mae

    # Weighted R²
    weighted_r2 = sum(
        w * metrics[f"r2_{name.lower()}"]
        for w, name in zip(target_weights, target_names)
    ) / sum(target_weights)
    metrics["weighted_r2"] = weighted_r2

    return metrics


def derive_dead_from_total_gdm(preds: np.ndarray) -> np.ndarray:
    """
    Derive Dead = Total - GDM.

    Since Total = Green + Dead + Clover and GDM = Green + Clover,
    then Dead = Total - GDM.

    Args:
        preds: (N, 5) array [Green, Dead, Clover, GDM, Total]

    Returns:
        Modified preds with Dead replaced by max(0, Total - GDM)
    """
    preds_fixed = preds.copy()
    total = preds[:, 4]
    gdm = preds[:, 3]
    dead_derived = np.maximum(0, total - gdm)
    preds_fixed[:, 1] = dead_derived
    return preds_fixed


def run_oof_evaluation(
    model_dir: str,
    data_dir: str,
    backbone_name: str,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 4,
    num_folds: int = 5,
    derive_dead: bool = False,
    grid: Tuple[int, int] = (2, 2),
    dropout: float = 0.2,
    hidden_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Run proper OOF evaluation.

    Each sample is predicted using only the fold model that was NOT trained on it.
    """
    # Load folds CSV
    folds_csv = os.path.join(model_dir, "folds.csv")
    if not os.path.exists(folds_csv):
        raise FileNotFoundError(f"folds.csv not found in {model_dir}")

    df = pd.read_csv(folds_csv)
    print(f"Loaded {len(df)} samples from folds.csv")
    print(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")

    image_dir = os.path.join(data_dir, "train")
    transform = get_val_transform(518)

    # Initialize OOF arrays
    n_samples = len(df)
    oof_preds = np.zeros((n_samples, 5), dtype=np.float32)
    oof_targets = np.zeros((n_samples, 5), dtype=np.float32)

    # Process each fold
    for fold in range(num_folds):
        ckpt_path = os.path.join(model_dir, f"5head_best_fold{fold}.pth")
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Fold {fold} checkpoint not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Fold {fold}: Loading model and predicting validation set")
        print(f"{'='*60}")

        # Load model
        model = load_model(
            ckpt_path, backbone_name, device,
            grid=grid, dropout=dropout, hidden_ratio=hidden_ratio
        )
        print(f"  Model: FiLM={model.use_film}, AttnPool={model.use_attention_pool}, AuxHeads={model.use_aux_heads}")

        # Get validation samples for this fold
        val_mask = df["fold"] == fold
        val_df = df[val_mask].reset_index(drop=True)
        val_indices = df.index[val_mask].tolist()

        print(f"  Validation samples: {len(val_df)}")

        # Create dataloader
        val_ds = OOFDataset(val_df, image_dir, transform)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=device.type == "cuda"
        )

        # Predict
        fold_preds, fold_targets = predict_fold(model, val_loader, device)

        # Store in OOF arrays
        for i, idx in enumerate(val_indices):
            oof_preds[idx] = fold_preds[i]
            oof_targets[idx] = fold_targets[i]

        # Cleanup
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Apply Dead derivation fix
    if derive_dead:
        print("\n" + "="*60)
        print("Applying Dead derivation fix: Dead = max(0, Total - GDM)")
        print("="*60)
        oof_preds_fixed = derive_dead_from_total_gdm(oof_preds)
    else:
        oof_preds_fixed = oof_preds

    return oof_preds, oof_preds_fixed, oof_targets, df


def main():
    parser = argparse.ArgumentParser(description="OOF Evaluation for 5-Head Model")
    parser.add_argument("--model-dir", type=str, required=True, help="Model directory with checkpoints and folds.csv")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--derive-dead", action="store_true", help="Derive Dead from Total - GDM")
    parser.add_argument("--grid", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("="*60)
    print("OOF Evaluation for 5-Head Model")
    print("="*60)
    print(f"Model dir: {args.model_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {device}")
    print(f"Derive Dead: {args.derive_dead}")
    print("="*60)

    # Run OOF evaluation
    oof_preds_raw, oof_preds_fixed, oof_targets, df = run_oof_evaluation(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        backbone_name=args.backbone,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_folds=args.num_folds,
        derive_dead=args.derive_dead,
        grid=tuple(args.grid),
        dropout=args.dropout,
        hidden_ratio=args.hidden_ratio,
    )

    # Compute metrics
    target_names = ["Green", "Dead", "Clover", "GDM", "Total"]
    target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]

    print("\n" + "="*60)
    print("OOF Results (Raw Predictions)")
    print("="*60)

    metrics_raw = compute_metrics(oof_preds_raw, oof_targets, target_names, target_weights)

    for name in target_names:
        r2 = metrics_raw[f"r2_{name.lower()}"]
        rmse = metrics_raw[f"rmse_{name.lower()}"]
        mae = metrics_raw[f"mae_{name.lower()}"]
        print(f"  {name:8s}: R² = {r2:7.4f}, RMSE = {rmse:7.2f}, MAE = {mae:7.2f}")

    print(f"\n  Weighted R²: {metrics_raw['weighted_r2']:.4f}")

    if args.derive_dead:
        print("\n" + "="*60)
        print("OOF Results (Dead = Total - GDM)")
        print("="*60)

        metrics_fixed = compute_metrics(oof_preds_fixed, oof_targets, target_names, target_weights)

        for name in target_names:
            r2 = metrics_fixed[f"r2_{name.lower()}"]
            rmse = metrics_fixed[f"rmse_{name.lower()}"]
            mae = metrics_fixed[f"mae_{name.lower()}"]
            improvement = ""
            if name == "Dead":
                r2_diff = r2 - metrics_raw["r2_dead"]
                improvement = f" ({'+' if r2_diff >= 0 else ''}{r2_diff:.4f})"
            print(f"  {name:8s}: R² = {r2:7.4f}, RMSE = {rmse:7.2f}, MAE = {mae:7.2f}{improvement}")

        print(f"\n  Weighted R²: {metrics_fixed['weighted_r2']:.4f} ({'+' if metrics_fixed['weighted_r2'] >= metrics_raw['weighted_r2'] else ''}{metrics_fixed['weighted_r2'] - metrics_raw['weighted_r2']:.4f})")

        # Compare Dead predictions
        print("\n" + "-"*40)
        print("Dead Prediction Comparison:")
        print(f"  Raw Dead R²:     {metrics_raw['r2_dead']:.4f}")
        print(f"  Derived Dead R²: {metrics_fixed['r2_dead']:.4f}")
        print(f"  Improvement:     {metrics_fixed['r2_dead'] - metrics_raw['r2_dead']:.4f}")

    # Save OOF predictions
    oof_df = df.copy()
    oof_df["pred_green"] = oof_preds_fixed[:, 0]
    oof_df["pred_dead"] = oof_preds_fixed[:, 1]
    oof_df["pred_clover"] = oof_preds_fixed[:, 2]
    oof_df["pred_gdm"] = oof_preds_fixed[:, 3]
    oof_df["pred_total"] = oof_preds_fixed[:, 4]

    # Also save raw dead for comparison
    if args.derive_dead:
        oof_df["pred_dead_raw"] = oof_preds_raw[:, 1]

    output_path = os.path.join(args.model_dir, "oof_predictions.csv")
    oof_df.to_csv(output_path, index=False)
    print(f"\nOOF predictions saved to: {output_path}")

    # Save metrics (convert numpy floats to Python floats for JSON)
    def to_python_float(d: Dict) -> Dict:
        return {k: float(v) for k, v in d.items()}

    metrics_output = {
        "raw": to_python_float(metrics_raw),
    }
    if args.derive_dead:
        metrics_output["derived_dead"] = to_python_float(metrics_fixed)

    metrics_path = os.path.join(args.model_dir, "oof_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"OOF metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
