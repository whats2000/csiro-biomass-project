"""Data loading and validation (Steps 1-2)."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_and_validate_data(
    train_csv: Path,
    test_csv: Path,
    sample_submission_csv: Path,
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate train, test, and submission CSVs.
    
    Implements Step 2 from AGENTS.md:
    - Load CSVs
    - Create image_id by splitting sample_id
    - Validate schema and data integrity
    - Parse dates
    
    Args:
        train_csv: Path to train.csv
        test_csv: Path to test.csv
        sample_submission_csv: Path to sample_submission.csv
        data_dir: Base data directory for image paths
        
    Returns:
        Tuple of (train_df, test_df, sample_sub_df)
    """
    logger.info("Loading CSV files...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    sample_sub_df = pd.read_csv(sample_submission_csv)
    
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Create image_id by splitting sample_id at '__'
    train_df["image_id"] = train_df["sample_id"].str.split("__").str[0]
    test_df["image_id"] = test_df["sample_id"].str.split("__").str[0]
    
    # Validate target_name values
    allowed_targets = {"Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"}
    train_targets = set(train_df["target_name"].unique())
    test_targets = set(test_df["target_name"].unique())
    
    assert train_targets == allowed_targets, f"Train targets mismatch: {train_targets}"
    assert test_targets == allowed_targets, f"Test targets mismatch: {test_targets}"
    logger.info("✓ Target names validated")
    
    # Validate each image_id has exactly 5 rows
    train_counts = train_df.groupby("image_id").size()
    assert (train_counts == 5).all(), "Not all train images have 5 target rows"
    
    test_counts = test_df.groupby("image_id").size()
    assert (test_counts == 5).all(), "Not all test images have 5 target rows"
    logger.info("✓ Each image has exactly 5 target rows")
    
    # Validate target column exists in train
    assert "target" in train_df.columns, "Target column missing in train"
    assert not train_df["target"].isna().any(), "Missing target values in train"
    logger.info("✓ Target values present in train")
    
    # Validate test sample_id matches submission
    test_ids = set(test_df["sample_id"])
    sub_ids = set(sample_sub_df["sample_id"])
    assert test_ids == sub_ids, f"Sample ID mismatch: {len(test_ids)} vs {len(sub_ids)}"
    logger.info("✓ Test sample_ids match submission")
    
    # Check for missing values in critical fields
    critical_fields = ["sample_id", "image_path", "target_name"]
    for field in critical_fields:
        assert not train_df[field].isna().any(), f"Missing values in train.{field}"
        assert not test_df[field].isna().any(), f"Missing values in test.{field}"
    logger.info("✓ No missing values in critical fields")
    
    # Parse Sampling_Date
    if "Sampling_Date" in train_df.columns:
        train_df["Sampling_Date_str"] = train_df["Sampling_Date"]
        train_df["Sampling_Date"] = pd.to_datetime(train_df["Sampling_Date"], errors="coerce")
        train_df["month"] = train_df["Sampling_Date"].dt.month
        train_df["year"] = train_df["Sampling_Date"].dt.year
        train_df["day_of_year"] = train_df["Sampling_Date"].dt.dayofyear
        
        # Create season (Southern Hemisphere)
        def get_season(month):
            if month in [12, 1, 2]:
                return "Summer"
            elif month in [3, 4, 5]:
                return "Autumn"
            elif month in [6, 7, 8]:
                return "Winter"
            else:
                return "Spring"
        
        train_df["season"] = train_df["month"].apply(get_season)
        logger.info("✓ Parsed Sampling_Date and extracted temporal features")
    
    # Make image paths absolute
    train_df["image_path"] = train_df["image_path"].apply(lambda x: str(data_dir / x))
    test_df["image_path"] = test_df["image_path"].apply(lambda x: str(data_dir / x))
    
    # Dataset statistics
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Unique images - Train: {train_df['image_id'].nunique()}, Test: {test_df['image_id'].nunique()}")
    
    if "State" in train_df.columns:
        logger.info(f"\nState distribution:\n{train_df.groupby('State')['image_id'].nunique()}")
    
    if "Species" in train_df.columns:
        logger.info(f"\nSpecies distribution:\n{train_df.groupby('Species')['image_id'].nunique()}")
    
    if "season" in train_df.columns:
        logger.info(f"\nSeason distribution:\n{train_df.groupby('season')['image_id'].nunique()}")
    
    return train_df, test_df, sample_sub_df


def create_image_level_dataframe(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format dataframe to image-level (wide) format.
    
    Each image_id becomes one row with all 5 targets as columns.
    This is useful for:
    - Creating per-image datasets
    - Computing derived targets
    - Fold splitting
    
    Args:
        long_df: Long format dataframe with one row per (image, target_name)
        
    Returns:
        Wide format dataframe with one row per image
    """
    # Pivot targets to columns
    target_cols = long_df.pivot(
        index="image_id",
        columns="target_name",
        values="target"
    ).reset_index()
    
    # Get metadata (same for all 5 rows of each image)
    metadata_cols = ["image_id", "image_path"]
    optional_cols = ["State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm", 
                     "Sampling_Date", "month", "year", "season", "day_of_year"]
    
    for col in optional_cols:
        if col in long_df.columns:
            metadata_cols.append(col)
    
    metadata = long_df[metadata_cols].drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    
    # Merge metadata with targets
    image_df = metadata.merge(target_cols, on="image_id", how="left")
    
    # Validate derived target relationships (if targets exist)
    if "Dry_Green_g" in image_df.columns:
        # Check GDM_g ≈ Dry_Green_g + Dry_Clover_g
        if "GDM_g" in image_df.columns:
            expected_gdm = image_df["Dry_Green_g"] + image_df["Dry_Clover_g"]
            gdm_diff = (image_df["GDM_g"] - expected_gdm).abs()
            logger.info(f"GDM_g deviation: mean={gdm_diff.mean():.2f}, max={gdm_diff.max():.2f}")
        
        # Check Dry_Total_g ≈ Dry_Green_g + Dry_Dead_g + Dry_Clover_g
        if "Dry_Total_g" in image_df.columns:
            expected_total = image_df["Dry_Green_g"] + image_df["Dry_Dead_g"] + image_df["Dry_Clover_g"]
            total_diff = (image_df["Dry_Total_g"] - expected_total).abs()
            logger.info(f"Dry_Total_g deviation: mean={total_diff.mean():.2f}, max={total_diff.max():.2f}")
    
    return image_df


def get_tabular_statistics(df: pd.DataFrame, continuous_features: list) -> Dict[str, Any]:
    """Compute mean and std for continuous features (for normalization).
    
    Args:
        df: Dataframe with continuous features
        continuous_features: List of continuous feature names
        
    Returns:
        Dictionary with 'mean' and 'std' for each feature
    """
    stats = {}
    for feat in continuous_features:
        if feat in df.columns:
            stats[feat] = {
                "mean": df[feat].mean(),
                "std": df[feat].std()
            }
    return stats
