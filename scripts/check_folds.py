#!/usr/bin/env python3
"""
Check fold statistics for different CV strategies.

Usage:
    python scripts/check_folds.py
    python scripts/check_folds.py --strategy group_location
    python scripts/check_folds.py --strategy group_date_state
"""
import argparse
import sys
sys.path.insert(0, '.')

import pandas as pd
from src.dataset import prepare_dataframe, create_folds


def analyze_folds(df: pd.DataFrame, n_folds: int = 5) -> None:
    """Analyze fold distribution statistics."""
    
    print("\n" + "="*70)
    print("FOLD STATISTICS")
    print("="*70)
    
    total = len(df)
    
    # Basic counts
    print(f"\nTotal samples: {total}")
    print(f"Unique groups (sample_id_prefix): {df['sample_id_prefix'].nunique()}")
    
    # Overall distributions
    print(f"\n--- Overall Distribution ---")
    print(f"States: {df['State'].value_counts().to_dict()}")
    if 'Species' in df.columns:
        print(f"Species (top 5): {df['Species'].value_counts().head().to_dict()}")
    if 'Sampling_Date_Month' in df.columns:
        months = sorted(df['Sampling_Date_Month'].unique())
        print(f"Months: {months}")
    
    # Per-fold analysis
    print(f"\n--- Per-Fold Analysis ---")
    print(f"{'Fold':<6} {'Train':<8} {'Val':<6} {'Val%':<8} {'States in Val':<40}")
    print("-" * 70)
    
    for fold in range(n_folds):
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        
        val_pct = 100 * len(val_df) / total
        val_states = val_df['State'].value_counts().to_dict()
        
        print(f"{fold:<6} {len(train_df):<8} {len(val_df):<6} {val_pct:<8.1f} {str(val_states):<40}")
    
    # State distribution check
    print(f"\n--- State Distribution Across Folds ---")
    state_fold_matrix = pd.crosstab(df['fold'], df['State'])
    print(state_fold_matrix)
    
    # Check for imbalance
    print(f"\n--- Balance Check ---")
    for state in df['State'].unique():
        state_folds = df[df['State'] == state]['fold'].value_counts().sort_index()
        if len(state_folds) < n_folds:
            missing = set(range(n_folds)) - set(state_folds.index)
            print(f"⚠️  {state}: Missing from folds {missing}")
        else:
            min_count = state_folds.min()
            max_count = state_folds.max()
            ratio = max_count / max(min_count, 1)
            if ratio > 2:
                print(f"⚠️  {state}: Imbalanced (min={min_count}, max={max_count}, ratio={ratio:.1f}x)")
            else:
                print(f"✓  {state}: Balanced (min={min_count}, max={max_count})")
    
    # Species distribution if available
    if 'Species' in df.columns:
        print(f"\n--- Species Distribution Across Folds ---")
        species_fold_matrix = pd.crosstab(df['fold'], df['Species'])
        # Show top 5 species
        top_species = df['Species'].value_counts().head(5).index
        print(species_fold_matrix[top_species])
    
    # Month/Season distribution
    if 'Sampling_Date_Month' in df.columns:
        print(f"\n--- Month Distribution Across Folds ---")
        month_fold_matrix = pd.crosstab(df['fold'], df['Sampling_Date_Month'])
        print(month_fold_matrix)


def main():
    parser = argparse.ArgumentParser(description="Check fold statistics")
    parser.add_argument("--data", type=str, default="data/train.csv",
                        help="Path to training CSV")
    parser.add_argument("--strategy", type=str, default="group_location",
                        choices=["group_location", "group_date_state", "group_date_state_bin", 
                                 "group_month", "group_date", "stratified", "random"],
                        help="CV strategy to test")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--use-existing", type=str, default=None,
                        help="Use existing fold CSV instead of creating new folds")
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"CV Strategy Analysis: {args.strategy}")
    print("="*70)
    
    if args.use_existing:
        print(f"\nLoading existing folds from: {args.use_existing}")
        df = pd.read_csv(args.use_existing)
    else:
        print(f"\nLoading data from: {args.data}")
        df = prepare_dataframe(args.data)
        
        print(f"\nCreating {args.n_folds} folds with strategy: {args.strategy}")
        df = create_folds(df, n_folds=args.n_folds, seed=args.seed, cv_strategy=args.strategy)
    
    analyze_folds(df, n_folds=args.n_folds)
    
    # Save if new folds created
    if not args.use_existing:
        output_path = f"data/trainfold_{args.strategy}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved folds to: {output_path}")


if __name__ == "__main__":
    main()


