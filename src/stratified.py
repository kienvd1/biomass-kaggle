class DataPreparator:
    """
    Class responsible for data loading and preprocessing.
    
    Main functions:
    - Load and pivot CSV
    - Stratified K-Fold splitting
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.df_wide: Optional[pd.DataFrame] = None
        
    def load_and_pivot(self) -> pd.DataFrame:
        """
        Load CSV and convert from long format to wide format.
        
        Returns:
            Wide-format DataFrame (each row is one image, each column is one target)
            
        Why not: Using pivot() instead of pivot_table()
            → Image paths are guaranteed to be unique
        """
        print(f"Loading CSV: {self.config.train_csv}")
        
        try:
            df_long = pd.read_csv(self.config.train_csv)
            print(f"Long format: {len(df_long)} rows")
            df_long[['sample_id_prefix', 'sample_id_suffix']] = df_long.sample_id.str.split('__', expand=True)
            # Pivot transformation: image_path × target_name → values
            cols = ['sample_id_prefix', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
            df_wide = df_long.groupby(cols).apply(lambda df: df.set_index('target_name').target)
            df_wide.reset_index(inplace=True)
            df_wide.columns.name = None
            df_wide['Sampling_Date_Month'] = df_wide.Sampling_Date.apply(lambda x: x.split('/')[1].strip())
            print(f"Wide format: {len(df_wide)} rows × {len(df_wide.columns)} columns")
            print(f"\nFirst 5 rows:\n{df_wide.head()}\n")
            
            self.df_wide = df_wide
            return df_wide
            
        except FileNotFoundError:
            print(f"Error: {self.config.train_csv} not found")
            # Return dummy DataFrame on error (prevent downstream crashes)
            return pd.DataFrame(columns=['image_path'] + self.config.all_target_cols)
    
    def create_stratified_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign fold numbers using KFold or StratifiedGroupKFold based on config.
        """
        df = df.copy()
        df['fold'] = -1

        # Assuming self.config.cv_strategy exists, otherwise default to 'groupby_Sampling_Date'
        cv_strategy = getattr(self.config, 'cv_strategy', 'groupby_Sampling_Date')

        num_bins = 5
        print(f"Stratifying Dry_Total_g into {num_bins} bins")
        
        df['total_bin'] = pd.cut(
            df['Dry_Total_g'], 
            bins=num_bins, 
            labels=False,
            duplicates='drop'  # Remove duplicate edges
        )
        
        if cv_strategy == 'groupby_Sampling_Date':
            print(f"\nPreparing {self.config.n_folds}-Fold StratifiedGroupKFold (Group: Date, Stratify: State)...")
            kfold = StratifiedGroupKFold(
                n_splits=self.config.n_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )
        else:
            print(f"\nPreparing {self.config.n_folds}-Fold Standard KFold (Random)...")
            kfold = KFold(
                n_splits=self.config.n_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )

        split_gen = kfold.split(
            X=df, 
            y=df['total_bin'], 
            groups=df['Sampling_Date']
        )

        for i, (trn_idx, val_idx) in enumerate(split_gen):
            df.loc[val_idx, 'fold'] = i
            
            # Logging logic (from your snippet)
            trn_df = df.iloc[trn_idx]
            val_df = df.iloc[val_idx]
            
            # Convert months to int for sorting
            trn_months = sorted(list(set(int(x) for x in trn_df.Sampling_Date_Month.unique())))
            val_months = sorted(list(set(int(x) for x in val_df.Sampling_Date_Month.unique())))
            
            print(f'Fold {i}: trn({trn_df.shape[0]}) -> val({val_df.shape[0]}) | '
                  f'Months: {trn_months} -> {val_months}')

        print("\nFold distribution:")
        print(df['fold'].value_counts().sort_index())

        self.df_wide = df
        return df