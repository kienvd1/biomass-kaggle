"""Dataset classes for CSIRO Biomass training."""
import os
import random
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
        use_log_target: bool = False,
        stereo_swap_prob: float = 0.0,
        photometric_transform: Optional[A.Compose] = None,
        photometric_left_only: bool = False,
        photometric_right_only: bool = False,
        mixup_prob: float = 0.0,
        mixup_alpha: float = 0.4,
        cutmix_prob: float = 0.0,
        cutmix_alpha: float = 1.0,
        mix_same_context: bool = True,
    ) -> None:
        """
        Args:
            df: DataFrame with columns [image_path, Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
            image_dir: Directory containing images
            transform: Albumentations transform (geometric + normalize for stereo-correct mode)
            is_train: Whether this is training data
            cache_images: If True, cache all images in RAM (use when you have >16GB free RAM)
            return_aux_labels: If True, return State and Month labels for auxiliary heads
            use_log_target: If True, apply log1p to targets (for long-tail distributions)
            stereo_swap_prob: Probability of swapping left/right images (0.0 to disable)
            photometric_transform: Separate photometric transforms applied independently per view
            photometric_left_only: If True, only apply photometric to left view
            photometric_right_only: If True, only apply photometric to right view
            mixup_prob: Probability of applying MixUp (0.0 to disable)
            mixup_alpha: Beta distribution alpha for MixUp lambda sampling
            cutmix_prob: Probability of applying CutMix (0.0 to disable)
            cutmix_alpha: Beta distribution alpha for CutMix lambda sampling
            mix_same_context: If True, only mix samples with same species AND month
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        self.cache_images = cache_images
        self.return_aux_labels = return_aux_labels
        self.use_log_target = use_log_target
        self.stereo_swap_prob = stereo_swap_prob
        self.photometric_transform = photometric_transform
        self.photometric_left_only = photometric_left_only
        self.photometric_right_only = photometric_right_only
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
        self.cutmix_alpha = cutmix_alpha
        self.mix_same_context = mix_same_context
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
                # Map months to class indices, fill unmapped months with 0
                self.month_labels = self.df["Sampling_Date_Month"].astype(int).map(self.MONTH_LABELS).fillna(0).values.astype(np.int64)
            else:
                self.month_labels = np.zeros(len(self.df), dtype=np.int64)

            # Species labels
            if "Species" in self.df.columns:
                self.species_labels = self.df["Species"].map(self.SPECIES_LABELS).fillna(7).values.astype(np.int64)
            else:
                self.species_labels = np.zeros(len(self.df), dtype=np.int64)
            
            # Ground-truth NDVI (Pre_GSHH_NDVI) - range typically 0.16-0.91
            if "Pre_GSHH_NDVI" in self.df.columns:
                self.ndvi_values = self.df["Pre_GSHH_NDVI"].fillna(0.5).values.astype(np.float32)
            else:
                self.ndvi_values = np.full(len(self.df), 0.5, dtype=np.float32)
            
            # Average height in cm (Height_Ave_cm) - range typically 1-70 cm
            if "Height_Ave_cm" in self.df.columns:
                self.height_values = self.df["Height_Ave_cm"].fillna(10.0).values.astype(np.float32)
            else:
                self.height_values = np.full(len(self.df), 10.0, dtype=np.float32)
        
        # Build context index for constrained MixUp/CutMix (same species + month + state)
        self._context_index: Dict[Tuple[int, int, int], List[int]] = {}
        if self.is_train and (self.mixup_prob > 0 or self.cutmix_prob > 0) and self.mix_same_context:
            self._build_context_index()
        
        # Pre-cache all images if enabled
        if self.cache_images:
            print(f"Caching {len(self.paths)} images in RAM...")
            for idx in range(len(self.paths)):
                self._load_image(idx)
            print(f"Cached {len(self._cache)} images")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _build_context_index(self) -> None:
        """Build index mapping (species_label, month_label, state_label) -> list of sample indices.
        
        Used for constrained MixUp/CutMix to only mix samples with same species, month, AND state.
        """
        # Get species, month, and state for all samples
        if "Species" in self.df.columns:
            species = self.df["Species"].map(self.SPECIES_LABELS).fillna(7).values.astype(int)
        else:
            species = np.zeros(len(self.df), dtype=int)
        
        if "Sampling_Date_Month" in self.df.columns:
            month = self.df["Sampling_Date_Month"].astype(int).map(self.MONTH_LABELS).fillna(0).values.astype(int)
        else:
            month = np.zeros(len(self.df), dtype=int)
        
        if "State" in self.df.columns:
            state = self.df["State"].map(self.STATE_LABELS).fillna(0).values.astype(int)
        else:
            state = np.zeros(len(self.df), dtype=int)
        
        # Build index with (species, month, state) as key
        for idx in range(len(self.df)):
            key = (int(species[idx]), int(month[idx]), int(state[idx]))
            if key not in self._context_index:
                self._context_index[key] = []
            self._context_index[key].append(idx)
        
        # Log statistics
        valid_groups = {k: v for k, v in self._context_index.items() if len(v) > 1}
        print(f"MixUp/CutMix context index: {len(valid_groups)} groups (species×month×state) with 2+ samples "
              f"(total {sum(len(v) for v in valid_groups.values())} samples eligible)")
    
    def _get_mix_partner(self, idx: int) -> Optional[int]:
        """Get a random sample index with same species, month, AND state for mixing."""
        if not self.mix_same_context:
            # Random partner from entire dataset
            partner = random.randint(0, len(self.df) - 1)
            return partner if partner != idx else None
        
        # Get species, month, and state for current sample
        row = self.df.iloc[idx]
        
        if "Species" in self.df.columns:
            species = self.SPECIES_LABELS.get(row.get("Species", "Mixed"), 7)
        else:
            species = 0
        
        if "Sampling_Date_Month" in self.df.columns:
            month_val = int(row["Sampling_Date_Month"])
            month = self.MONTH_LABELS.get(month_val, 0)
        else:
            month = 0
        
        if "State" in self.df.columns:
            state = self.STATE_LABELS.get(row.get("State", "NSW"), 0)
        else:
            state = 0
        
        key = (species, month, state)
        candidates = self._context_index.get(key, [])
        
        # Need at least 2 samples (current + 1 partner)
        if len(candidates) < 2:
            return None
        
        # Pick random partner (excluding self)
        candidates = [c for c in candidates if c != idx]
        if not candidates:
            return None
        
        return random.choice(candidates)
    
    def _rand_bbox(self, h: int, w: int, lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        
        # Center of the box
        cy = random.randint(0, h)
        cx = random.randint(0, w)
        
        # Box coordinates (clipped to image bounds)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(h, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(w, cx + cut_w // 2)
        
        return y1, y2, x1, x2
    
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

        # Stereo swap augmentation: randomly swap left/right views
        # This works because FiLM/attention pooling don't assume L/R order
        if self.is_train and self.stereo_swap_prob > 0 and random.random() < self.stereo_swap_prob:
            left, right = right, left

        if self.transform:
            if self.is_train:
                # Apply same geometric transform to both views (via replay)
                replay = self.transform(image=left)
                left_t = replay["image"]
                right_t = self.transform.replay(replay["replay"], image=right)["image"]
                
                # Apply photometric transforms (optionally independent per view)
                if self.photometric_transform is not None:
                    # Convert back to numpy for photometric aug
                    left_np = left_t.permute(1, 2, 0).numpy()
                    right_np = right_t.permute(1, 2, 0).numpy()
                    
                    # Apply photometric transforms based on mode
                    if self.photometric_left_only:
                        # Only apply to left
                        left_t = self.photometric_transform(image=left_np)["image"]
                        right_t = torch.from_numpy(right_np.transpose(2, 0, 1))
                    elif self.photometric_right_only:
                        # Only apply to right
                        left_t = torch.from_numpy(left_np.transpose(2, 0, 1))
                        right_t = self.photometric_transform(image=right_np)["image"]
                    else:
                        # Apply independently to both (default)
                        left_t = self.photometric_transform(image=left_np)["image"]
                        right_t = self.photometric_transform(image=right_np)["image"]
            else:
                left_t = self.transform(image=left)["image"]
                right_t = self.transform(image=right)["image"]
        else:
            left_t = torch.from_numpy(left.transpose(2, 0, 1)).float() / 255.0
            right_t = torch.from_numpy(right.transpose(2, 0, 1)).float() / 255.0

        targets_raw = self.targets[idx].copy()
        
        # MixUp/CutMix augmentation (training only, same species/month/state)
        mix_lambda = 1.0  # Default: no mixing
        if self.is_train:
            do_mixup = self.mixup_prob > 0 and random.random() < self.mixup_prob
            do_cutmix = self.cutmix_prob > 0 and random.random() < self.cutmix_prob
            
            if do_mixup or do_cutmix:
                partner_idx = self._get_mix_partner(idx)
                
                if partner_idx is not None:
                    # Load and transform partner image
                    partner_img = self._load_image(partner_idx)
                    ph, pw, _ = partner_img.shape
                    pmid = pw // 2
                    partner_left = partner_img[:, :pmid]
                    partner_right = partner_img[:, pmid:]
                    
                    if self.transform:
                        preplay = self.transform(image=partner_left)
                        partner_left_t = preplay["image"]
                        partner_right_t = self.transform.replay(preplay["replay"], image=partner_right)["image"]
                        
                        if self.photometric_transform is not None:
                            pl_np = partner_left_t.permute(1, 2, 0).numpy()
                            pr_np = partner_right_t.permute(1, 2, 0).numpy()
                            partner_left_t = self.photometric_transform(image=pl_np)["image"]
                            partner_right_t = self.photometric_transform(image=pr_np)["image"]
                    else:
                        partner_left_t = torch.from_numpy(partner_left.transpose(2, 0, 1)).float() / 255.0
                        partner_right_t = torch.from_numpy(partner_right.transpose(2, 0, 1)).float() / 255.0
                    
                    partner_targets = self.targets[partner_idx].copy()
                    
                    if do_cutmix:
                        # CutMix: paste rectangular region from partner
                        mix_lambda = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                        _, h_t, w_t = left_t.shape
                        y1, y2, x1, x2 = self._rand_bbox(h_t, w_t, mix_lambda)
                        
                        # Apply same cutout region to both L/R images
                        left_t[:, y1:y2, x1:x2] = partner_left_t[:, y1:y2, x1:x2]
                        right_t[:, y1:y2, x1:x2] = partner_right_t[:, y1:y2, x1:x2]
                        
                        # Adjust lambda based on actual area
                        mix_lambda = 1 - ((y2 - y1) * (x2 - x1)) / (h_t * w_t)
                    else:
                        # MixUp: weighted average of images
                        mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                        left_t = mix_lambda * left_t + (1 - mix_lambda) * partner_left_t
                        right_t = mix_lambda * right_t + (1 - mix_lambda) * partner_right_t
                    
                    # Mix targets with same lambda
                    targets_raw = mix_lambda * targets_raw + (1 - mix_lambda) * partner_targets
        
        # Apply log1p transformation for long-tail target distribution
        if self.use_log_target:
            targets_raw = np.log1p(targets_raw)
        
        targets = torch.tensor(targets_raw, dtype=torch.float32)

        if self.return_aux_labels:
            # For constrained mixing, aux labels are the same (same species/month/state)
            state_label = torch.tensor(self.state_labels[idx], dtype=torch.long)
            month_label = torch.tensor(self.month_labels[idx], dtype=torch.long)
            species_label = torch.tensor(self.species_labels[idx], dtype=torch.long)
            # Ground-truth NDVI and Height for auxiliary regression
            ndvi_target = torch.tensor(self.ndvi_values[idx], dtype=torch.float32)
            height_target = torch.tensor(self.height_values[idx], dtype=torch.float32)
            return left_t, right_t, targets, state_label, month_label, species_label, ndvi_target, height_target

        return left_t, right_t, targets


def get_train_transforms(
    img_size: int = 518, 
    aug_prob: float = 0.5, 
    strong: bool = False,
    no_lighting: bool = False,
    use_perspective_jitter: bool = False,
    use_border_mask: bool = False,
) -> A.ReplayCompose:
    """Get training augmentations with replay support for consistent stereo augmentation.
    
    Args:
        img_size: Target image size
        aug_prob: Base augmentation probability
        strong: If True, use stronger augmentations from dinov3-5tar.ipynb notebook
        use_perspective_jitter: If True, add perspective warp to simulate annotation noise
        use_border_mask: If True, add border masking to prevent frame-based shortcuts
    
    NOTE: This applies the SAME photometric transforms to both views (via replay),
    which may allow the model to "cheat" by matching pixel noise. For stereo-correct
    augmentations, use get_stereo_correct_transforms() instead.
    """
    transforms = [
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=aug_prob),
        A.VerticalFlip(p=aug_prob),
    ]
    
    # Perspective jitter: simulate corner-annotation / warp noise
    # The dataset has manual corner annotations + perspective normalization
    # This helps the model be robust to imperfect normalization
    if use_perspective_jitter:
        transforms.insert(0, A.Perspective(
            scale=(0.02, 0.08),  # Small perspective distortion
            keep_size=True,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ))
    
    # Border mask: randomly mask thin borders to prevent frame-based shortcuts
    # The dataset has metal frame artifacts that could be spurious features
    if use_border_mask:
        transforms.append(A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.02, 0.05),  # Thin strips
            hole_width_range=(0.9, 1.0),  # Full width
            fill=0,
            p=0.2,
        ))
        transforms.append(A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.9, 1.0),  # Full height
            hole_width_range=(0.02, 0.05),  # Thin strips
            fill=0,
            p=0.2,
        ))
    
    if strong:
        # Strong augmentations from dinov3-5tar.ipynb
        transforms.append(A.RandomRotate90(p=aug_prob))
        
        # Lighting augmentations (skip if no_lighting=True)
        if not no_lighting:
            transforms.extend([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.75),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.CLAHE(p=0.2),
            ])
        
        transforms.extend([
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.05, 0.05),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.1, 0.2),
                hole_width_range=(0.1, 0.2),
                fill=0,
                p=0.3,
            ),
            A.MotionBlur(p=0.1),
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3)),
                A.GaussNoise(std_range=(0.05, 0.1), mean_range=(0, 0), per_channel=True),
            ], p=0.3),
        ])
    else:
        # Standard augmentations
        transforms.extend([
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
        ])
    
    # Common normalization and tensor conversion
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return A.ReplayCompose(transforms)


def get_pad_to_square_transforms(img_size: int = 518) -> A.ReplayCompose:
    """
    Pad-to-Square transform that preserves aspect ratio.
    
    The quadrat images are 70cm × 30cm (7:3 aspect ratio).
    Squashing to 1:1 destroys texture frequency (density patterns).
    
    This pads the image to square instead of resizing, preserving
    the biomass density visual cues.
    
    Args:
        img_size: Target square size
    """
    return A.ReplayCompose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            position="center",
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_stereo_geometric_transforms(img_size: int = 518, aug_prob: float = 0.5, strong: bool = False) -> A.ReplayCompose:
    """Get GEOMETRIC-only transforms with replay support for stereo pairs.
    
    These transforms are applied identically to both L/R views to preserve stereo geometry.
    Does NOT include normalization or ToTensorV2 - those are handled separately.
    
    Args:
        img_size: Target image size
        aug_prob: Base augmentation probability
        strong: If True, use stronger augmentations from dinov3-5tar.ipynb
    """
    transforms = [
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=aug_prob),
        A.VerticalFlip(p=aug_prob),
    ]
    
    if strong:
        # Strong geometric augmentations
        transforms.extend([
            A.RandomRotate90(p=aug_prob),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.05, 0.05),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.1, 0.2),
                hole_width_range=(0.1, 0.2),
                fill=0,
                p=0.3,
            ),
        ])
    else:
        # Standard geometric augmentations
        transforms.extend([
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_REFLECT_101,
                p=aug_prob,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.3,
            ),
        ])
    
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return A.ReplayCompose(transforms)


def get_stereo_photometric_transforms(aug_prob: float = 0.5, strong: bool = False) -> A.Compose:
    """Get PHOTOMETRIC-only transforms applied INDEPENDENTLY to each stereo view.
    
    This prevents the model from "cheating" by matching identical noise/blur patterns
    between left and right images.
    
    Args:
        aug_prob: Base augmentation probability
        strong: If True, use stronger augmentations from dinov3-5tar.ipynb
    
    Input/output: normalized tensor (C, H, W) as numpy array.
    """
    if strong:
        # Strong photometric augmentations from notebook
        return A.Compose([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.75,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3,
            ),
            A.CLAHE(p=0.2),
            A.MotionBlur(p=0.1),
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3)),
                A.GaussNoise(std_range=(0.05, 0.1), mean_range=(0, 0), per_channel=True),
            ], p=0.3),
            ToTensorV2(),
        ])
    else:
        # Standard photometric augmentations
        return A.Compose([
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            ], p=aug_prob),
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
            - "group_location": RECOMMENDED - GroupKFold by sample_id_prefix (location+date), 
                               stratified by State×Season×Species. Best for preventing data leakage
                               and ensuring balanced green/dead/clover distribution across folds.
            - "group_month": StratifiedGroupKFold grouped by month, stratified by Dry_Total_g bins
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
    
    elif cv_strategy == "group_location":
        # BEST STRATEGY based on dataset paper:
        # - Group by sample_id_prefix (location + date) to prevent leakage
        # - Stratify by State × Season × Species_group to ensure distribution balance
        # Paper: 19 locations, 4 states, 3 years, 6 major species
        # "manual sorting into: green, dead, clover" - species affects these ratios!
        if "sample_id_prefix" not in df.columns:
            print("WARNING: sample_id_prefix not found, falling back to group_date_state")
            cv_strategy = "group_date_state"
        elif "State" not in df.columns:
            print("WARNING: State not found, falling back to stratified")
            cv_strategy = "stratified"
        else:
            stratify_parts = [df["State"].astype(str)]
            stratify_name = "State"
            
            # Add Season from month (paper spans all seasons)
            if "Sampling_Date_Month" in df.columns:
                month_to_season = {
                    12: "Summer", 1: "Summer", 2: "Summer",  # Dec-Feb (Southern Hemisphere)
                    3: "Autumn", 4: "Autumn", 5: "Autumn",   # Mar-May
                    6: "Winter", 7: "Winter", 8: "Winter",   # Jun-Aug
                    9: "Spring", 10: "Spring", 11: "Spring"  # Sep-Nov
                }
                df["_Season"] = df["Sampling_Date_Month"].astype(int).map(month_to_season)
                stratify_parts.append(df["_Season"].astype(str))
                stratify_name += "×Season"
            
            # Add Species group (paper: "Species: Pasture species by biomass")
            # Different species have very different green/dead/clover ratios
            if "Species" in df.columns:
                # Simplify species to major groups to avoid too many strata
                species_to_group = {
                    "Clover": "Clover", "WhiteClover": "Clover", 
                    "SubcloverLosa": "Clover", "SubcloverDalkeith": "Clover",
                    "Ryegrass": "Ryegrass", "Ryegrass_Clover": "Ryegrass",
                    "Phalaris": "Phalaris", "Phalaris_Clover": "Phalaris",
                    "Phalaris_Ryegrass_Clover": "Phalaris",
                    "Fescue": "Other", "Fescue_CrumbWeed": "Other",
                    "Lucerne": "Other",
                }
                df["_SpeciesGroup"] = df["Species"].map(lambda x: species_to_group.get(x, "Other"))
                stratify_parts.append(df["_SpeciesGroup"].astype(str))
                stratify_name += "×Species"
            
            # Combine stratification columns
            stratify_col = stratify_parts[0]
            for part in stratify_parts[1:]:
                stratify_col = stratify_col + "_" + part
            
            n_groups = df["sample_id_prefix"].nunique()
            n_strata = stratify_col.nunique()
            print(f"\nUsing StratifiedGroupKFold (group=sample_id_prefix, stratify={stratify_name})")
            print(f"Found {n_groups} unique location+date groups, {n_strata} strata")
            
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for fold_idx, (train_idx, val_idx) in enumerate(
                sgkf.split(df, stratify_col, groups=df["sample_id_prefix"])
            ):
                df.loc[df.index[val_idx], "fold"] = fold_idx
                
                train_states = df.iloc[train_idx]["State"].value_counts().to_dict()
                val_states = df.iloc[val_idx]["State"].value_counts().to_dict()
                train_groups = df.iloc[train_idx]["sample_id_prefix"].nunique()
                val_groups = df.iloc[val_idx]["sample_id_prefix"].nunique()
                print(
                    f"Fold {fold_idx}: train={len(train_idx)} ({train_groups} groups) -> "
                    f"val={len(val_idx)} ({val_groups} groups) | "
                    f"States: train={train_states} val={val_states}"
                )
            
            # Cleanup temp columns
            df = df.drop(columns=["_Season", "_SpeciesGroup"], errors="ignore")
    
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

