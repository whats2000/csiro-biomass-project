"""Cross-validation splitting (Step 5)."""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import logging

logger = logging.getLogger(__name__)


def create_folds(
    image_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    group_col: str = "image_id",
) -> pd.DataFrame:
    """Create fold assignments using GroupKFold.
    
    Implements leakage-safe CV by splitting on image_id level.
    Never split by long-format rows to avoid leakage.
    
    Args:
        image_df: Image-level dataframe (one row per image)
        n_folds: Number of folds
        seed: Random seed
        group_col: Column to group by (default: image_id)
        
    Returns:
        DataFrame with 'fold' column added
    """
    df = image_df.copy()
    df["fold"] = -1
    
    # Use GroupKFold to ensure same image_id doesn't appear in train and val
    gkf = GroupKFold(n_splits=n_folds)
    
    # Create groups
    groups = df[group_col].values
    
    # If we want stratification by Dry_Total_g, we can bin it
    # For now, use simple GroupKFold
    X = df.index.values
    y = np.zeros(len(df))  # Dummy y for compatibility
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        df.loc[val_idx, "fold"] = fold
    
    # Verify all images got assigned
    assert (df["fold"] >= 0).all(), "Some images not assigned to folds"
    
    # Log fold statistics
    logger.info("\n=== Fold Statistics ===")
    fold_counts = df.groupby("fold").size()
    logger.info(f"Images per fold:\n{fold_counts}")
    
    # Check for balance in key features
    if "State" in df.columns:
        state_dist = df.groupby(["fold", "State"]).size().unstack(fill_value=0)
        logger.info(f"\nState distribution per fold:\n{state_dist}")
    
    if "Dry_Total_g" in df.columns:
        target_stats = df.groupby("fold")["Dry_Total_g"].agg(["mean", "std", "min", "max"])
        logger.info(f"\nDry_Total_g statistics per fold:\n{target_stats}")
    
    return df
