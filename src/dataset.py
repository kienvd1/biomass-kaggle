"""Dataset classes for CSIRO Biomass training."""
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Fast JPEG loading with TurboJPEG (optional)
try:
    from turbojpeg import TurboJPEG
    _jpeg = TurboJPEG()
    _USE_TURBOJPEG = True
except ImportError:
    _USE_TURBOJPEG = False


class BiomassDataset(Dataset):
    """Dataset for biomass prediction from stereo image pairs."""

    # Label mappings for auxiliary heads (must match models_5head.py)
    STATE_LABELS: Dict[str, int] = {"NSW": 0, "Tas": 1, "Vic": 2, "WA": 3}
    MONTH_LABELS: Dict[int, int] = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}

    # Species labels - grouped by similarity for better learning
    SPECIES_LABELS: Dict[str, int] = {
        # Clover-dominant (usually high clover, low/zero dead in WA)
        "Clover": 0,
        "WhiteClover": 0,
        "SubcloverLosa": 0,
        "SubcloverDalkeith": 0,
        # Ryegrass types
        "Ryegrass": 1,
        "Ryegrass_Clover": 2,
        # Phalaris types
        "Phalaris": 3,
        "Phalaris_Clover": 4,
        "Phalaris_Ryegrass_Clover": 4,
        "Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass": 4,
        "Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed": 4,
        # Other grasses (often zero clover)
        "Fescue": 5,
        "Fescue_CrumbWeed": 5,
        "Lucerne": 6,
        "Mixed": 7,
    }
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[A.Compose] = None,
        is_train: bool = True,
        cache_images: bool = False,
        return_aux_labels: bool = False,
    ) -> None:
        """
        Args:
            df: DataFrame with columns [image_path, Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
            image_dir: Directory containing images
            transform: Albumentations transform
            is_train: Whether this is training data
            cache_images: If True, cache all images in RAM (use when you have >16GB free RAM)
            return_aux_labels: If True, return State and Month labels for auxiliary heads
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        self.cache_images = cache_images
        self.return_aux_labels = return_aux_labels
        self._cache: Dict[int, np.ndarray] = {}
        
        self.paths = self.df["image_path"].values
        self.targets = self.df[
            ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
        ].values.astype(np.float32)
        
        # Prepare auxiliary labels if needed
        if self.return_aux_labels:
            # State labels
            if "State" in self.df.columns:
                self.state_labels = self.df["State"].map(self.STATE_LABELS).values.astype(np.int64)
            else:
                self.state_labels = np.zeros(len(self.df), dtype=np.int64)

            # Month labels (from Sampling_Date_Month column)
            if "Sampling_Date_Month" in self.df.columns:
                self.month_labels = self.df["Sampling_Date_Month"].astype(int).map(self.MONTH_LABELS).values.astype(np.int64)
            else:
                self.month_labels = np.zeros(len(self.df), dtype=np.int64)

            # Species labels
            if "Species" in self.df.columns:
                self.species_labels = self.df["Species"].map(self.SPECIES_LABELS).fillna(7).values.astype(np.int64)
            else:
                self.species_labels = np.zeros(len(self.df), dtype=np.int64)
        
        # Pre-cache all images if enabled
        if self.cache_images:
            print(f"Caching {len(self.paths)} images in RAM...")
            for idx in range(len(self.paths)):
                self._load_image(idx)
            print(f"Cached {len(self._cache)} images")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_image(self, idx: int) -> np.ndarray:
        """Load image from disk or cache. Uses TurboJPEG if available for speed."""
        if idx in self._cache:
            return self._cache[idx]

        filename = os.path.basename(self.paths[idx])
        full_path = os.path.join(self.image_dir, filename)

        img = None

        # Try TurboJPEG first (faster for JPEG files)
        if _USE_TURBOJPEG and full_path.lower().endswith(('.jpg', '.jpeg')):
            try:
                with open(full_path, 'rb') as f:
                    img = _jpeg.decode(f.read())  # Returns RGB directly
            except Exception:
                img = None

        # Fallback to OpenCV
        if img is None:
            img = cv2.imread(full_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create blank image if loading failed
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)

        if self.cache_images:
            self._cache[idx] = img
        return img
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            left_tensor: Left image tensor (C, H, W)
            right_tensor: Right image tensor (C, H, W)
            targets: Target values tensor (5,)
            state_label: (optional) State label (int) if return_aux_labels=True
            month_label: (optional) Month label (int) if return_aux_labels=True
            species_label: (optional) Species label (int) if return_aux_labels=True
        """
        img = self._load_image(idx)

        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]

        if self.transform:
            # Apply same spatial transform to both views
            if self.is_train:
                replay = self.transform(image=left)
                left_t = replay["image"]
                right_t = self.transform.replay(replay["replay"], image=right)["image"]
            else:
                left_t = self.transform(image=left)["image"]
                right_t = self.transform(image=right)["image"]
        else:
            left_t = torch.from_numpy(left.transpose(2, 0, 1)).float() / 255.0
            right_t = torch.from_numpy(right.transpose(2, 0, 1)).float() / 255.0

        targets = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.return_aux_labels:
            state_label = torch.tensor(self.state_labels[idx], dtype=torch.long)
            month_label = torch.tensor(self.month_labels[idx], dtype=torch.long)
            species_label = torch.tensor(self.species_labels[idx], dtype=torch.long)
            return left_t, right_t, targets, state_label, month_label, species_label

        return left_t, right_t, targets


def get_train_transforms(img_size: int = 518, aug_prob: float = 0.5) -> A.ReplayCompose:
    """Get training augmentations with replay support for consistent stereo augmentation."""
    return A.ReplayCompose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=aug_prob),
        A.VerticalFlip(p=aug_prob),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT_101,
            p=aug_prob,
        ),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=aug_prob),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size: int = 518) -> A.Compose:
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Convert long-format CSV to wide format for training.
    
    Args:
        csv_path: Path to train.csv in long format
        
    Returns:
        Wide-format DataFrame with one row per image
    """
    print(f"Loading CSV: {csv_path}")
    df_long = pd.read_csv(csv_path)
    print(f"Long format: {len(df_long)} rows")
    
    # Extract sample_id prefix
    df_long[["sample_id_prefix", "sample_id_suffix"]] = df_long["sample_id"].str.split("__", expand=True)
    
    # Pivot to wide format using groupby (more robust)
    index_cols = ["sample_id_prefix", "image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]
    df_wide = df_long.groupby(index_cols).apply(
        lambda x: x.set_index("target_name")["target"]
    ).reset_index()
    df_wide.columns.name = None
    
    # Extract month from Sampling_Date (format: "DD/MM/YYYY")
    df_wide["Sampling_Date_Month"] = df_wide["Sampling_Date"].apply(
        lambda x: str(x).split("/")[1].strip()
    )
    
    # Ensure all target columns exist
    target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    for col in target_cols:
        if col not in df_wide.columns:
            df_wide[col] = 0.0
    
    print(f"Wide format: {len(df_wide)} rows × {len(df_wide.columns)} columns")
    
    return df_wide


def create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 18,
    cv_strategy: str = "group_date_state",
    num_bins: int = 4,
) -> pd.DataFrame:
    """
    Create folds based on specified strategy.
    
    Args:
        df: Wide-format DataFrame (must have Sampling_Date_Month for month grouping; Sampling_Date for date grouping)
        n_folds: Number of folds
        seed: Random seed
        num_bins: Number of bins for target stratification (default: 4)
        cv_strategy: One of:
            - "group_month": StratifiedGroupKFold grouped by month, stratified by Dry_Total_g bins (default)
            - "group_date": StratifiedGroupKFold grouped by Sampling_Date, stratified by Dry_Total_g bins
            - "group_date_state": StratifiedGroupKFold grouped by Sampling_Date_Month, stratified by State
            - "group_date_state_bin": StratifiedGroupKFold grouped by Sampling_Date_Month, stratified by State × Dry_Total_g bin
            - "stratified": StratifiedKFold on Dry_Total_g bins only
            - "random": Standard KFold (random, no stratification)
        
    Returns:
        DataFrame with 'fold' column added (0-indexed: 0, 1, 2, 3, 4)
    """
    from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold
    
    df = df.copy()
    df["fold"] = -1
    
    # Create bins for stratification based on Dry_Total_g (used by several strategies)
    print(f"Stratifying Dry_Total_g into {num_bins} bins (for bin-based strategies)")
    df["target_bin"] = pd.qcut(df["Dry_Total_g"], q=num_bins, labels=False, duplicates="drop")
    
    if cv_strategy == "group_month":
        # StratifiedGroupKFold: samples from same month stay together, stratified by target bins
        if "Sampling_Date_Month" not in df.columns:
            print("WARNING: Sampling_Date_Month not found, falling back to stratified")
            cv_strategy = "stratified"
        else:
            unique_months = sorted(set(int(m) for m in df["Sampling_Date_Month"].unique()))
            print(f"\nUsing StratifiedGroupKFold (group=month, stratify=target_bin)")
            print(f"Found {len(unique_months)} unique months: {unique_months}")
            
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for fold_idx, (train_idx, val_idx) in enumerate(
                sgkf.split(df, df["target_bin"], groups=df["Sampling_Date_Month"])
            ):
                df.loc[df.index[val_idx], "fold"] = fold_idx
                
                train_months = sorted(set(int(m) for m in df.iloc[train_idx]["Sampling_Date_Month"].unique()))
                val_months = sorted(set(int(m) for m in df.iloc[val_idx]["Sampling_Date_Month"].unique()))
                print(f"Fold {fold_idx}: train={len(train_idx)} (months: {train_months}) -> "
                      f"val={len(val_idx)} (months: {val_months})")
    
    if cv_strategy == "group_date":
        # StratifiedGroupKFold: samples from same date stay together, stratified by target bins
        if "Sampling_Date" not in df.columns:
            print("WARNING: Sampling_Date not found, falling back to stratified")
            cv_strategy = "stratified"
        else:
            print(f"\nUsing StratifiedGroupKFold (group=Sampling_Date, stratify=target_bin)")
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for fold_idx, (train_idx, val_idx) in enumerate(
                sgkf.split(df, df["target_bin"], groups=df["Sampling_Date"])
            ):
                df.loc[df.index[val_idx], "fold"] = fold_idx
                
                train_dates = df.iloc[train_idx]["Sampling_Date"].nunique()
                val_dates = df.iloc[val_idx]["Sampling_Date"].nunique()
                train_months = sorted(set(int(m) for m in df.iloc[train_idx]["Sampling_Date_Month"].unique())) if "Sampling_Date_Month" in df.columns else []
                val_months = sorted(set(int(m) for m in df.iloc[val_idx]["Sampling_Date_Month"].unique())) if "Sampling_Date_Month" in df.columns else []
                print(f"Fold {fold_idx}: train={len(train_idx)} ({train_dates} dates, months: {train_months}) -> "
                      f"val={len(val_idx)} ({val_dates} dates, months: {val_months})")

    if cv_strategy in ("group_date_state", "group_date_state_bin"):
        # Split variant:
        # - group by month (Sampling_Date_Month)
        # - stratify by State (and optionally State × target bin)
        if "Sampling_Date_Month" not in df.columns:
            print("WARNING: Sampling_Date_Month not found, falling back to stratified")
            cv_strategy = "stratified"
        elif "State" not in df.columns:
            print("WARNING: State not found, falling back to group_month")
            cv_strategy = "group_month"
        else:
            unique_months = sorted(set(int(m) for m in df["Sampling_Date_Month"].unique()))
            print("\nUsing StratifiedGroupKFold (group=month, stratify=State)"
                  + (" × Dry_Total_g bin" if cv_strategy == "group_date_state_bin" else ""))
            print(f"Found {len(unique_months)} unique months: {unique_months}")
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            if cv_strategy == "group_date_state_bin":
                y = df["State"].astype(str) + "__" + df["target_bin"].astype(str)
            else:
                y = df["State"]

            for fold_idx, (train_idx, val_idx) in enumerate(
                sgkf.split(df, y, groups=df["Sampling_Date_Month"])
            ):
                df.loc[df.index[val_idx], "fold"] = fold_idx

                train_months = sorted(set(int(m) for m in df.iloc[train_idx]["Sampling_Date_Month"].unique()))
                val_months = sorted(set(int(m) for m in df.iloc[val_idx]["Sampling_Date_Month"].unique()))
                train_states = df.iloc[train_idx]["State"].value_counts().to_dict()
                val_states = df.iloc[val_idx]["State"].value_counts().to_dict()
                print(
                    f"Fold {fold_idx}: train={len(train_idx)} (months: {train_months}) -> "
                    f"val={len(val_idx)} (months: {val_months}) | "
                    f"State dist train={train_states} val={val_states}"
                )
    
    if cv_strategy == "stratified":
        # Standard StratifiedKFold on target bins only
        print(f"\nUsing StratifiedKFold (stratify=target_bin)")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["target_bin"])):
            df.loc[df.index[val_idx], "fold"] = fold_idx
            print(f"Fold {fold_idx}: train={len(train_idx)} -> val={len(val_idx)}")
    
    elif cv_strategy == "random":
        # Standard random KFold (no stratification)
        print(f"\nUsing KFold (random)")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[df.index[val_idx], "fold"] = fold_idx
            print(f"Fold {fold_idx}: train={len(train_idx)} -> val={len(val_idx)}")
    
    # Print fold distribution
    print("\nFold distribution:")
    print(df["fold"].value_counts().sort_index())
    
    df = df.drop(columns=["target_bin"], errors="ignore")
    
    return df

