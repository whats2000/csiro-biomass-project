"""PyTorch Dataset for biomass prediction (Step 6)."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .transforms import load_image

logger = logging.getLogger(__name__)


class BiomassDataset(Dataset):
    """Per-image dataset for biomass prediction.
    
    Returns:
    - image tensor
    - tabular features tensor
    - target tensor (3 base components: Dry_Green_g, Dry_Dead_g, Dry_Clover_g)
    - metadata dict (image_id, fold, etc.)
    
    This avoids duplicated image reads for the 5 long-format rows.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        categorical_features: List[str] = None,
        continuous_features: List[str] = None,
        cat_encoders: Optional[Dict] = None,
        continuous_stats: Optional[Dict] = None,
        base_targets: List[str] = None,
        is_test: bool = False,
    ):
        """Initialize dataset.
        
        Args:
            df: Image-level dataframe (one row per image)
            transform: Albumentations transform
            categorical_features: List of categorical feature names
            continuous_features: List of continuous feature names
            cat_encoders: Dict mapping category name to label encoding dict
            continuous_stats: Dict with mean/std for each continuous feature
            base_targets: List of base target names to predict
            is_test: Whether this is test data (no targets)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.categorical_features = categorical_features or []
        self.continuous_features = continuous_features or []
        self.cat_encoders = cat_encoders or {}
        self.continuous_stats = continuous_stats or {}
        self.base_targets = base_targets or ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
        self.is_test = is_test
        
        # Build categorical encoders if not provided
        if not self.cat_encoders and not is_test:
            self._build_encoders()
        
        logger.info(f"Dataset initialized with {len(self)} images")
    
    def _build_encoders(self):
        """Build label encoders for categorical features."""
        for feat in self.categorical_features:
            if feat in self.df.columns:
                unique_vals = sorted(self.df[feat].dropna().unique())
                self.cat_encoders[feat] = {val: idx for idx, val in enumerate(unique_vals)}
                logger.info(f"Encoded {feat}: {len(unique_vals)} categories")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get one sample.
        
        Returns:
            Tuple of (image, tabular_features, targets, metadata)
        """
        row = self.df.iloc[idx]
        
        # Load and transform image
        image = load_image(row["image_path"])
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Prepare tabular features
        tabular_features = []
        
        # Categorical features (encoded as indices)
        for feat in self.categorical_features:
            if feat in self.df.columns:
                val = row[feat]
                encoded = self.cat_encoders[feat].get(val, 0)  # Default to 0 if unknown
                tabular_features.append(encoded)
            else:
                # Missing feature - use default encoding (0)
                tabular_features.append(0)
        
        # Continuous features (normalized)
        for feat in self.continuous_features:
            if feat in self.df.columns:
                val = row[feat]
                if feat in self.continuous_stats:
                    mean = self.continuous_stats[feat]["mean"]
                    std = self.continuous_stats[feat]["std"]
                    val = (val - mean) / (std + 1e-8)
                tabular_features.append(val)
            else:
                # Missing feature - use default value (0.0, which is the normalized mean)
                tabular_features.append(0.0)
        
        tabular_tensor = torch.tensor(tabular_features, dtype=torch.float32)
        
        # Prepare targets (3 base components)
        if not self.is_test:
            targets = []
            for target_name in self.base_targets:
                if target_name in row:
                    targets.append(row[target_name])
                else:
                    targets.append(0.0)  # Fallback
            target_tensor = torch.tensor(targets, dtype=torch.float32)
        else:
            target_tensor = torch.zeros(len(self.base_targets), dtype=torch.float32)
        
        # Metadata
        metadata = {
            "image_id": row["image_id"],
            "image_path": row["image_path"],
        }
        if "fold" in row:
            metadata["fold"] = row["fold"]
        
        return image, tabular_tensor, target_tensor, metadata


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_transform,
    val_transform,
    config,
    cat_encoders: Dict,
    continuous_stats: Dict,
):
    """Create train and validation dataloaders.
    
    Args:
        train_df: Training image-level dataframe
        val_df: Validation image-level dataframe
        train_transform: Training transforms
        val_transform: Validation transforms
        config: Config object
        cat_encoders: Categorical encoders
        continuous_stats: Continuous feature statistics
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = BiomassDataset(
        df=train_df,
        transform=train_transform,
        categorical_features=config.categorical_features,
        continuous_features=config.continuous_features,
        cat_encoders=cat_encoders,
        continuous_stats=continuous_stats,
        base_targets=config.base_targets,
        is_test=False,
    )
    
    val_dataset = BiomassDataset(
        df=val_df,
        transform=val_transform,
        categorical_features=config.categorical_features,
        continuous_features=config.continuous_features,
        cat_encoders=cat_encoders,
        continuous_stats=continuous_stats,
        base_targets=config.base_targets,
        is_test=False,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader
