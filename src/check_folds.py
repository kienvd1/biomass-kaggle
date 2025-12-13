#!/usr/bin/env python3
"""
Check fold distribution balance.

Usage:
    python -m src.check_folds
    python -m src.check_folds --csv /path/to/train.csv --cv-strategy group_month
"""
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

from .dataset import create_folds, prepare_dataframe


def check_fold_balance(
    df: pd.DataFrame,
    target_cols: List[str] = None,
    group_cols: List[str] = None,
) -> Dict:
    """
    Check fold distribution balance.
    
    Args:
        df: DataFrame with 'fold' column
        target_cols: Target columns to check distribution
        group_cols: Grouping columns to check distribution (e.g., State, Month)
    
    Returns:
        Dictionary with balance statistics
    """
    if target_cols is None:
        target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    
    if group_cols is None:
        group_cols = ["State", "Sampling_Date_Month"]
    
    n_folds = df["fold"].nunique()
    folds = sorted(df["fold"].unique())
    
    results = {
        "n_folds": n_folds,
        "folds": folds,
        "sample_counts": {},
        "target_stats": {},
        "group_distributions": {},
    }
    
    print("=" * 70)
    print("FOLD BALANCE CHECK")
    print("=" * 70)
    
    # 1. Train/Val sample counts per fold
    print("\n1. TRAIN/VAL SAMPLE COUNTS PER FOLD")
    print("-" * 70)
    counts = df["fold"].value_counts().sort_index()
    total = len(df)
    expected_val = total / n_folds
    expected_train = total - expected_val
    
    print(f"{'Fold':<6} {'Train':>8} {'Val':>8} {'Train%':>8} {'Val%':>8} {'Train/Val Ratio':>16}")
    print("-" * 70)
    
    for fold in folds:
        val_count = counts[fold]
        train_count = total - val_count
        train_pct = train_count / total * 100
        val_pct = val_count / total * 100
        ratio = train_count / val_count if val_count > 0 else 0
        
        results["sample_counts"][fold] = {
            "train": train_count,
            "val": val_count,
            "train_pct": train_pct,
            "val_pct": val_pct,
            "ratio": ratio,
        }
        print(f"Fold {fold:<3} {train_count:>8} {val_count:>8} {train_pct:>7.1f}% {val_pct:>7.1f}% {ratio:>16.2f}")
    
    print("-" * 70)
    print(f"Total samples: {total}")
    print(f"Expected: Train={expected_train:.0f} ({expected_train/total*100:.1f}%), Val={expected_val:.0f} ({expected_val/total*100:.1f}%)")
    
    # 2. Target distribution per fold
    print("\n2. TARGET DISTRIBUTION PER FOLD")
    print("-" * 70)
    
    # Filter to only existing columns
    existing_targets = [c for c in target_cols if c in df.columns]
    
    for target in existing_targets:
        print(f"\n{target}:")
        global_mean = df[target].mean()
        global_std = df[target].std()
        
        results["target_stats"][target] = {"global_mean": global_mean, "global_std": global_std, "folds": {}}
        
        fold_means = []
        fold_stds = []
        
        for fold in folds:
            fold_df = df[df["fold"] == fold]
            mean = fold_df[target].mean()
            std = fold_df[target].std()
            median = fold_df[target].median()
            fold_means.append(mean)
            fold_stds.append(std)
            
            diff_pct = (mean - global_mean) / global_mean * 100 if global_mean != 0 else 0
            results["target_stats"][target]["folds"][fold] = {"mean": mean, "std": std, "median": median}
            
            print(f"  Fold {fold}: mean={mean:8.2f} (±{std:7.2f}) | median={median:8.2f} | diff={diff_pct:+5.1f}%")
        
        # Check variance across folds
        cv_of_means = np.std(fold_means) / np.mean(fold_means) * 100 if np.mean(fold_means) != 0 else 0
        print(f"  Global:  mean={global_mean:8.2f} (±{global_std:7.2f}) | CV of fold means: {cv_of_means:.2f}%")
    
    # 3. Group distributions per fold (e.g., State, Month)
    print("\n3. GROUP DISTRIBUTIONS PER FOLD")
    print("-" * 70)
    
    existing_groups = [c for c in group_cols if c in df.columns]
    
    for group_col in existing_groups:
        print(f"\n{group_col}:")
        global_dist = df[group_col].value_counts(normalize=True).sort_index()
        
        results["group_distributions"][group_col] = {"global": global_dist.to_dict(), "folds": {}}
        
        # Create comparison table
        dist_table = pd.DataFrame(index=sorted(df[group_col].unique()))
        dist_table["Global"] = df[group_col].value_counts(normalize=True).sort_index() * 100
        
        for fold in folds:
            fold_df = df[df["fold"] == fold]
            fold_dist = fold_df[group_col].value_counts(normalize=True).sort_index() * 100
            dist_table[f"Fold {fold}"] = fold_dist
            results["group_distributions"][group_col]["folds"][fold] = fold_dist.to_dict()
        
        # Fill NaN with 0
        dist_table = dist_table.fillna(0)
        
        print(dist_table.round(1).to_string())
    
    # 4. Month overlap check (for group_month strategy)
    if "Sampling_Date_Month" in df.columns:
        print("\n4. MONTH OVERLAP CHECK")
        print("-" * 70)
        
        for fold in folds:
            train_df = df[df["fold"] != fold]
            val_df = df[df["fold"] == fold]
            
            train_months = set(train_df["Sampling_Date_Month"].unique())
            val_months = set(val_df["Sampling_Date_Month"].unique())
            overlap = train_months & val_months
            
            train_months_sorted = sorted([int(m) for m in train_months])
            val_months_sorted = sorted([int(m) for m in val_months])
            overlap_sorted = sorted([int(m) for m in overlap]) if overlap else []
            
            status = "⚠️  OVERLAP" if overlap else "✓ No overlap"
            print(f"Fold {fold}: Train months: {train_months_sorted}")
            print(f"        Val months:   {val_months_sorted}")
            print(f"        {status}: {overlap_sorted if overlap else 'None'}")
            print()
    
    # 5. Summary metrics
    print("\n5. BALANCE SUMMARY")
    print("-" * 70)
    
    # Sample count balance (coefficient of variation)
    val_counts = [results["sample_counts"][f]["val"] for f in folds]
    train_counts = [results["sample_counts"][f]["train"] for f in folds]
    val_cv = np.std(val_counts) / np.mean(val_counts) * 100 if np.mean(val_counts) != 0 else 0
    
    print(f"Val sample count CV: {val_cv:.2f}% (lower is better, <5% is good)")
    print(f"Val size range: {min(val_counts)} - {max(val_counts)} samples")
    print(f"Train size range: {min(train_counts)} - {max(train_counts)} samples")
    
    # Target mean balance
    for target in existing_targets:
        fold_means = [results["target_stats"][target]["folds"][f]["mean"] for f in folds]
        global_mean = results["target_stats"][target]["global_mean"]
        max_diff = max(abs(m - global_mean) / global_mean * 100 for m in fold_means) if global_mean != 0 else 0
        print(f"{target} max diff from global: {max_diff:.1f}% (lower is better, <10% is good)")
    
    print("\n" + "=" * 70)
    
    return results


def search_best_seeds(
    df_base: pd.DataFrame,
    n_folds: int = 5,
    cv_strategy: str = "group_month",
    num_bins: int = 4,
    seed_range: tuple = (1, 100),
    target_cols: List[str] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Search for best seeds based on fold balance metrics.
    
    Args:
        df_base: Base DataFrame (without folds)
        n_folds: Number of folds
        cv_strategy: CV strategy
        num_bins: Number of bins for stratification
        seed_range: (start, end) range of seeds to search
        target_cols: Target columns to evaluate
        top_k: Number of top seeds to return
    
    Returns:
        DataFrame with seed scores sorted by best
    """
    if target_cols is None:
        target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    
    existing_targets = [c for c in target_cols if c in df_base.columns]
    
    print("=" * 70)
    print("SEED SEARCH")
    print("=" * 70)
    print(f"Searching seeds {seed_range[0]} to {seed_range[1]}...")
    print(f"CV Strategy: {cv_strategy}, Folds: {n_folds}, Bins: {num_bins}")
    print()
    
    results = []
    
    for seed in range(seed_range[0], seed_range[1] + 1):
        # Create folds with this seed (suppress output)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            df = create_folds(
                df_base.copy(),
                n_folds=n_folds,
                seed=seed,
                cv_strategy=cv_strategy,
                num_bins=num_bins,
            )
        finally:
            sys.stdout = old_stdout
        
        folds = sorted(df["fold"].unique())
        total = len(df)
        
        # Calculate metrics
        # 1. Val size CV (lower is better)
        val_counts = [len(df[df["fold"] == f]) for f in folds]
        val_cv = np.std(val_counts) / np.mean(val_counts) * 100 if np.mean(val_counts) != 0 else 999
        
        # 2. Target mean CV across folds (lower is better)
        target_cvs = []
        target_max_diffs = []
        for target in existing_targets:
            global_mean = df[target].mean()
            fold_means = [df[df["fold"] == f][target].mean() for f in folds]
            
            if np.mean(fold_means) != 0:
                cv = np.std(fold_means) / np.mean(fold_means) * 100
            else:
                cv = 0
            target_cvs.append(cv)
            
            if global_mean != 0:
                max_diff = max(abs(m - global_mean) / global_mean * 100 for m in fold_means)
            else:
                max_diff = 0
            target_max_diffs.append(max_diff)
        
        avg_target_cv = np.mean(target_cvs)
        max_target_diff = max(target_max_diffs)
        
        # 3. Month overlap check (0 is best)
        month_overlaps = 0
        if "Sampling_Date_Month" in df.columns:
            for fold in folds:
                train_months = set(df[df["fold"] != fold]["Sampling_Date_Month"].unique())
                val_months = set(df[df["fold"] == fold]["Sampling_Date_Month"].unique())
                month_overlaps += len(train_months & val_months)
        
        # Combined score (lower is better)
        # Weight: val_cv (30%) + avg_target_cv (40%) + max_target_diff (20%) + month_overlaps (10%)
        score = val_cv * 0.3 + avg_target_cv * 0.4 + max_target_diff * 0.2 + month_overlaps * 10 * 0.1
        
        results.append({
            "seed": seed,
            "score": score,
            "val_cv": val_cv,
            "target_cv": avg_target_cv,
            "max_diff": max_target_diff,
            "month_overlaps": month_overlaps,
            "val_min": min(val_counts),
            "val_max": max(val_counts),
        })
        
        # Progress
        if seed % 20 == 0:
            print(f"  Processed seed {seed}...")
    
    # Sort by score
    results_df = pd.DataFrame(results).sort_values("score")
    
    print("\n" + "=" * 70)
    print(f"TOP {top_k} BEST SEEDS")
    print("=" * 70)
    print(f"{'Seed':>6} {'Score':>8} {'Val CV':>8} {'Tgt CV':>8} {'MaxDiff':>8} {'Overlap':>8} {'Val Range':>12}")
    print("-" * 70)
    
    for _, row in results_df.head(top_k).iterrows():
        print(f"{row['seed']:>6} {row['score']:>8.2f} {row['val_cv']:>7.2f}% {row['target_cv']:>7.2f}% "
              f"{row['max_diff']:>7.1f}% {row['month_overlaps']:>8} {row['val_min']:>5}-{row['val_max']:<5}")
    
    print("\n" + "-" * 70)
    print("WORST 5 SEEDS (for comparison)")
    print("-" * 70)
    for _, row in results_df.tail(5).iterrows():
        print(f"{row['seed']:>6} {row['score']:>8.2f} {row['val_cv']:>7.2f}% {row['target_cv']:>7.2f}% "
              f"{row['max_diff']:>7.1f}% {row['month_overlaps']:>8} {row['val_min']:>5}-{row['val_max']:<5}")
    
    best_seed = int(results_df.iloc[0]["seed"])
    print(f"\n✓ BEST SEED: {best_seed} (score: {results_df.iloc[0]['score']:.2f})")
    print("=" * 70)
    
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Check fold distribution balance")
    parser.add_argument("--csv", type=str, default="/root/workspace/data/train.csv", help="Path to train.csv")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=18, help="Random seed")
    parser.add_argument("--num-bins", type=int, default=4, help="Number of bins for stratification")
    parser.add_argument(
        "--cv-strategy", type=str, default="group_month",
        choices=["group_month", "group_date", "stratified", "random"],
        help="CV strategy"
    )
    parser.add_argument("--search-seeds", action="store_true", help="Search for best seeds")
    parser.add_argument("--seed-start", type=int, default=1, help="Start of seed range")
    parser.add_argument("--seed-end", type=int, default=100, help="End of seed range")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top seeds to show")
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.csv}")
    
    # Load and prepare data
    df_base = prepare_dataframe(args.csv)
    
    if args.search_seeds:
        # Search for best seeds
        results_df = search_best_seeds(
            df_base,
            n_folds=args.n_folds,
            cv_strategy=args.cv_strategy,
            num_bins=args.num_bins,
            seed_range=(args.seed_start, args.seed_end),
            top_k=args.top_k,
        )
        
        # Show detailed analysis for best seed
        best_seed = int(results_df.iloc[0]["seed"])
        print(f"\nDetailed analysis for best seed ({best_seed}):\n")
        
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        df = create_folds(df_base, n_folds=args.n_folds, seed=best_seed, 
                          cv_strategy=args.cv_strategy, num_bins=args.num_bins)
        sys.stdout = old_stdout
        
        check_fold_balance(df)
    else:
        # Single seed check
        print(f"CV Strategy: {args.cv_strategy}")
        print(f"Folds: {args.n_folds}, Bins: {args.num_bins}, Seed: {args.seed}")
        print()
        
        df = create_folds(
            df_base, 
            n_folds=args.n_folds, 
            seed=args.seed, 
            cv_strategy=args.cv_strategy,
            num_bins=args.num_bins,
        )
        
        check_fold_balance(df)


if __name__ == "__main__":
    main()

