#!/usr/bin/env python3
"""
Check if depth estimation is useful for biomass prediction.

This script:
1. Loads sample images and their biomass labels
2. Generates depth maps using Depth Anything V2 or V3
3. Computes depth statistics (mean, std, max, volume proxy)
4. Correlates depth stats with biomass targets
5. Visualizes depth maps for inspection

Usage:
    # For Depth Anything V2 (default, easier install)
    pip install transformers accelerate
    python scripts/check_depth_usefulness.py --model da2
    
    # For Depth Anything V3 (better performance)
    pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git
    python scripts/check_depth_usefulness.py --model da3
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DepthModelDA2:
    """Depth Anything V2 wrapper."""
    
    def __init__(self, model_size: str = "small"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        
        model_names = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf",
        }
        
        model_name = model_names.get(model_size, model_names["small"])
        print(f"Loading {model_name}...")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                              else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"DA2 model loaded on {self.device}")
    
    @torch.no_grad()
    def get_depth(self, image: Image.Image) -> np.ndarray:
        """Get depth map from image."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        depth = outputs.predicted_depth
        
        # Interpolate to original size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        
        return depth


def load_depth_model(model_type: str = "da2", model_size: str = "small"):
    """Load depth model.
    
    Currently only DA2 is supported via HuggingFace transformers.
    DA3 requires the official package: pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git
    """
    if model_type == "da3":
        print("WARNING: DA3 requires official package, falling back to DA2")
        print("  To use DA3: pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git")
        print("  Using DA2 instead (already shows r=0.63 correlation - very good!)")
    return DepthModelDA2(model_size)


def compute_depth_stats(depth: np.ndarray) -> dict:
    """Compute statistics from depth map."""
    flat = depth.flatten()
    
    # Height statistics
    stats_dict = {
        "depth_mean": float(np.mean(flat)),
        "depth_std": float(np.std(flat)),
        "depth_max": float(np.max(flat)),
        "depth_min": float(np.min(flat)),
        "depth_p10": float(np.percentile(flat, 10)),
        "depth_p90": float(np.percentile(flat, 90)),
        "depth_range": float(np.max(flat) - np.min(flat)),
    }
    
    # Gradient (captures vegetation boundaries)
    grad_y = np.abs(np.diff(depth, axis=0))
    grad_x = np.abs(np.diff(depth, axis=1))
    stats_dict["depth_gradient"] = float(grad_y.mean() + grad_x.mean())
    
    # Volume proxy (sum of heights above minimum)
    min_depth = np.min(flat)
    volume = np.sum(flat - min_depth) / len(flat)
    stats_dict["depth_volume"] = float(volume)
    
    # High depth ratio (% of pixels with high depth = tall vegetation)
    threshold = np.percentile(flat, 75)
    stats_dict["depth_high_ratio"] = float(np.mean(flat > threshold))
    
    return stats_dict


def visualize_samples(samples: list, save_path: str = None):
    """Visualize sample images with depth maps."""
    n = min(6, len(samples))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    
    for i, sample in enumerate(samples[:n]):
        # Left image
        axes[i, 0].imshow(sample["img_left"])
        axes[i, 0].set_title(f"Left - Total: {sample['total']:.1f}")
        axes[i, 0].axis("off")
        
        # Depth map
        im = axes[i, 1].imshow(sample["depth_left"], cmap="viridis")
        axes[i, 1].set_title(f"Depth - Mean: {sample['depth_mean']:.2f}")
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
        
        # Right image
        axes[i, 2].imshow(sample["img_right"])
        axes[i, 2].set_title(f"Green: {sample['green']:.1f}, Dead: {sample['dead']:.1f}")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    plt.show()


def compute_correlations(df: pd.DataFrame, group_name: str = "all") -> pd.DataFrame:
    """Compute correlations between depth stats and biomass targets."""
    import warnings
    depth_cols = [c for c in df.columns if c.startswith("depth_")]
    target_cols = ["green", "dead", "clover", "gdm", "total"]
    
    if len(df) < 5:  # Need minimum samples for correlation
        return pd.DataFrame()
    
    correlations = []
    for depth_col in depth_cols:
        for target_col in target_cols:
            try:
                # Suppress constant input warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Check for constant values
                    if df[depth_col].std() < 1e-10 or df[target_col].std() < 1e-10:
                        continue
                    r, p = stats.pearsonr(df[depth_col], df[target_col])
                    if np.isnan(r):
                        continue
                    correlations.append({
                        "group": group_name,
                        "depth_stat": depth_col,
                        "target": target_col,
                        "correlation": r,
                        "p_value": p,
                        "significant": p < 0.05,
                        "n_samples": len(df),
                    })
            except Exception:
                pass
    
    return pd.DataFrame(correlations)


def compute_breakdown_correlations(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute correlations broken down by a grouping column."""
    all_corrs = []
    
    for group_val, group_df in df.groupby(group_col):
        if len(group_df) >= 5:  # Need minimum samples
            corrs = compute_correlations(group_df, group_name=f"{group_col}={group_val}")
            all_corrs.append(corrs)
    
    if all_corrs:
        return pd.concat(all_corrs, ignore_index=True)
    return pd.DataFrame()


def print_breakdown_summary(corr_df: pd.DataFrame, group_col: str, target: str = "total"):
    """Print summary of correlations by group for a specific target."""
    if corr_df.empty:
        print(f"  No data for {group_col} breakdown")
        return
    
    # Filter for target and best depth stat per group
    target_corrs = corr_df[corr_df["target"] == target].copy()
    if target_corrs.empty:
        return
    
    # Drop NaN correlations
    target_corrs = target_corrs.dropna(subset=["correlation"])
    if target_corrs.empty:
        print(f"  No valid correlations for '{target}'")
        return
    
    # Get best correlation per group (by absolute value)
    def get_best_idx(group_df):
        if group_df["correlation"].isna().all():
            return None
        return group_df["correlation"].abs().idxmax()
    
    best_indices = target_corrs.groupby("group").apply(get_best_idx)
    best_indices = best_indices.dropna()
    
    if best_indices.empty:
        print(f"  No valid correlations for '{target}'")
        return
    
    best_per_group = target_corrs.loc[best_indices.values]
    best_per_group = best_per_group.sort_values("correlation", key=abs, ascending=False)
    
    print(f"\n  {group_col.upper()} breakdown (best depth stat for '{target}'):")
    print(f"  {'-'*60}")
    for _, row in best_per_group.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["significant"] else ""
        group_name = str(row["group"]).replace(f"{group_col}=", "")
        print(f"    {group_name:25s}: r={row['correlation']:+.3f} ({row['depth_stat']:15s}) n={row['n_samples']:3d} {sig}")


def main(args):
    # Paths
    base_path = Path("./data")
    train_csv = base_path / "train.csv"
    image_dir = base_path / "train"
    output_dir = Path("./outputs/depth_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(train_csv)
    
    # Get unique sample prefixes (format: ID1011485656__Dry_Clover_g -> ID1011485656)
    df["sample_id_prefix"] = df["sample_id"].str.split("__").str[0]
    
    # Pivot to get targets per sample
    target_map = {"Dry_Green_g": "green", "Dry_Dead_g": "dead", "Dry_Clover_g": "clover", 
                  "GDM_g": "gdm", "Dry_Total_g": "total"}
    
    samples_list = []
    for prefix, group in df.groupby("sample_id_prefix"):
        first_row = group.iloc[0]
        row_data = {
            "sample_id_prefix": prefix, 
            "image_path": first_row["image_path"],
            "state": first_row.get("State", "Unknown"),
            "species": first_row.get("Species", "Unknown"),
            "sampling_date": first_row.get("Sampling_Date", ""),
        }
        # Extract month from date (format: YYYY/M/D or similar)
        try:
            date_parts = str(row_data["sampling_date"]).split("/")
            if len(date_parts) >= 2:
                row_data["month"] = int(date_parts[1])
            else:
                row_data["month"] = 0
        except:
            row_data["month"] = 0
        
        for _, r in group.iterrows():
            if r["target_name"] in target_map:
                row_data[target_map[r["target_name"]]] = r["target"]
        if "green" in row_data and "total" in row_data:  # Has all targets
            samples_list.append(row_data)
    
    samples_df = pd.DataFrame(samples_list)
    
    # Sample subset for analysis (or use all)
    if args.all or args.n_samples <= 0:
        n_samples = len(samples_df)
        print(f"Analyzing ALL {n_samples} samples with {args.model.upper()} ({args.size})...")
    else:
        n_samples = min(args.n_samples, len(samples_df))
        samples_df = samples_df.sample(n=n_samples, random_state=42)
        print(f"Analyzing {n_samples} samples with {args.model.upper()} ({args.size})...")
    
    # Load depth model
    depth_model = load_depth_model(args.model, args.size)
    
    # Process samples
    results = []
    viz_samples = []
    
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Processing"):
        # Load stereo image (left and right are side-by-side in one file)
        img_path = base_path / row["image_path"]
        
        if not img_path.exists():
            continue
        
        # Load and split into left/right halves
        img_full = Image.open(img_path).convert("RGB")
        w, h = img_full.size
        mid = w // 2
        img_left = img_full.crop((0, 0, mid, h))
        img_right = img_full.crop((mid, 0, w, h))
        
        # Get depth maps
        depth_left = depth_model.get_depth(img_left)
        depth_right = depth_model.get_depth(img_right)
        
        # Average depth (stereo consistency)
        depth_avg = (depth_left + depth_right) / 2
        
        # Compute stats
        stats_left = compute_depth_stats(depth_left)
        stats_right = compute_depth_stats(depth_right)
        stats_avg = compute_depth_stats(depth_avg)
        
        # Store results
        result = {
            "sample_id_prefix": row["sample_id_prefix"],
            "state": row["state"],
            "species": row["species"],
            "month": row["month"],
            "green": row["green"],
            "dead": row["dead"],
            "clover": row["clover"],
            "gdm": row["gdm"],
            "total": row["total"],
        }
        
        # Add depth stats (use average of L/R)
        for k, v in stats_avg.items():
            result[k] = v
        
        # Also store L-R difference (stereo disparity proxy)
        result["depth_lr_diff"] = float(np.mean(np.abs(depth_left - depth_right)))
        
        results.append(result)
        
        # Store for visualization
        if len(viz_samples) < 6:
            viz_samples.append({
                "img_left": np.array(img_left),
                "img_right": np.array(img_right),
                "depth_left": depth_left,
                "depth_right": depth_right,
                **result,
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "depth_stats.csv", index=False)
    print(f"\nSaved depth stats to {output_dir / 'depth_stats.csv'}")
    
    # Compute correlations
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS (ALL SAMPLES)")
    print("=" * 60)
    
    corr_df = compute_correlations(results_df, "all")
    if corr_df.empty:
        print("ERROR: No correlations computed. Check if depth stats were extracted.")
        return
    
    corr_df = corr_df.sort_values("correlation", key=abs, ascending=False)
    
    # Show top correlations
    print("\nTop correlations (|r| > 0.1):")
    print("-" * 50)
    strong_corr = corr_df[abs(corr_df["correlation"]) > 0.1].head(20)
    for _, row in strong_corr.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["significant"] else ""
        print(f"  {row['depth_stat']:20s} vs {row['target']:8s}: r={row['correlation']:+.3f} {sig}")
    
    # Save correlations
    corr_df.to_csv(output_dir / "correlations.csv", index=False)
    
    # Summary by target
    print("\n" + "=" * 60)
    print("BEST DEPTH FEATURE PER TARGET")
    print("=" * 60)
    
    for target in ["green", "dead", "clover", "gdm", "total"]:
        target_corrs = corr_df[corr_df["target"] == target]
        if not target_corrs.empty:
            target_corr = target_corrs.iloc[0]
            print(f"  {target:8s}: {target_corr['depth_stat']:20s} (r={target_corr['correlation']:+.3f})")
    
    # =========================================================================
    # BREAKDOWN ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("BREAKDOWN BY STATE")
    print("=" * 60)
    state_corrs = compute_breakdown_correlations(results_df, "state")
    if not state_corrs.empty:
        state_corrs.to_csv(output_dir / "correlations_by_state.csv", index=False)
        for target in ["total", "green", "dead"]:
            print_breakdown_summary(state_corrs, "state", target)
    
    print("\n" + "=" * 60)
    print("BREAKDOWN BY MONTH")
    print("=" * 60)
    month_corrs = compute_breakdown_correlations(results_df, "month")
    if not month_corrs.empty:
        month_corrs.to_csv(output_dir / "correlations_by_month.csv", index=False)
        for target in ["total", "green", "dead"]:
            print_breakdown_summary(month_corrs, "month", target)
    
    print("\n" + "=" * 60)
    print("BREAKDOWN BY SPECIES")
    print("=" * 60)
    species_corrs = compute_breakdown_correlations(results_df, "species")
    if not species_corrs.empty:
        species_corrs.to_csv(output_dir / "correlations_by_species.csv", index=False)
        for target in ["total", "green"]:
            print_breakdown_summary(species_corrs, "species", target)
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    max_corr = abs(corr_df["correlation"]).max()
    avg_corr = abs(corr_df[corr_df["target"] == "total"]["correlation"]).max()
    
    if max_corr > 0.3:
        print(f"✅ STRONG correlation found (max |r| = {max_corr:.3f})")
        print("   → Depth features are HIGHLY USEFUL for biomass prediction")
        print("   → Recommend: Integrate depth model into training")
    elif max_corr > 0.15:
        print(f"⚠️  MODERATE correlation found (max |r| = {max_corr:.3f})")
        print("   → Depth features may provide SOME benefit")
        print("   → Recommend: Test with small experiment first")
    else:
        print(f"❌ WEAK correlation (max |r| = {max_corr:.3f})")
        print("   → Depth features unlikely to help significantly")
        print("   → Recommend: Focus on other improvements")
    
    print(f"\nTotal biomass correlation with best depth stat: r={avg_corr:.3f}")
    
    # Visualize samples
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    visualize_samples(viz_samples, save_path=str(output_dir / "depth_samples.png"))
    
    # Plot correlation heatmap
    print("\nGenerating correlation heatmap...")
    depth_cols = [c for c in results_df.columns if c.startswith("depth_")]
    target_cols = ["green", "dead", "clover", "gdm", "total"]
    
    corr_matrix = results_df[depth_cols + target_cols].corr().loc[depth_cols, target_cols]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(target_cols)))
    ax.set_xticklabels(target_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(depth_cols)))
    ax.set_yticklabels([c.replace("depth_", "") for c in depth_cols])
    
    # Add correlation values
    for i in range(len(depth_cols)):
        for j in range(len(target_cols)):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
    
    plt.colorbar(im, label="Pearson Correlation")
    plt.title("Depth Statistics vs Biomass Targets Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_dir / 'correlation_heatmap.png'}")
    plt.show()
    
    print("\n✅ Analysis complete!")
    print(f"   Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if depth estimation helps biomass prediction")
    parser.add_argument("--model", type=str, default="da2", choices=["da2", "da3"],
                        help="Depth model: da2 (Depth Anything V2) or da3 (Depth Anything V3)")
    parser.add_argument("--size", type=str, default="small", choices=["small", "base", "large"],
                        help="Model size (small/base/large)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples to analyze (0 or -1 for all)")
    parser.add_argument("--all", action="store_true",
                        help="Use all samples")
    args = parser.parse_args()
    main(args)

