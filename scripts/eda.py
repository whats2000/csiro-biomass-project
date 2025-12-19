"""Exploratory Data Analysis (Step 3)."""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from biomass.utils import Config
from biomass.data import load_and_validate_data, create_image_level_dataframe

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def analyze_target_distributions(train_img_df: pd.DataFrame, config: Config):
    """Analyze target distributions."""
    logger.info("\n=== Target Distributions ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, target in enumerate(config.all_targets):
        if target in train_img_df.columns:
            ax = axes[idx]
            values = train_img_df[target].dropna()
            
            # Summary statistics
            logger.info(f"\n{target}:")
            logger.info(f"  Mean: {values.mean():.2f}")
            logger.info(f"  Std: {values.std():.2f}")
            logger.info(f"  Min: {values.min():.2f}, Max: {values.max():.2f}")
            logger.info(f"  Median: {values.median():.2f}")
            logger.info(f"  Skew: {values.skew():.2f}")
            
            # Plot distribution
            ax.hist(values, bins=50, alpha=0.7, edgecolor="black")
            ax.set_title(f"{target} Distribution")
            ax.set_xlabel(target)
            ax.set_ylabel("Frequency")
            ax.axvline(values.mean(), color="red", linestyle="--", label=f"Mean: {values.mean():.1f}")
            ax.legend()
    
    # Remove extra subplot
    if len(config.all_targets) < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "target_distributions.png", dpi=150)
    logger.info(f"\nSaved plot to {config.output_dir / 'target_distributions.png'}")
    plt.close()


def validate_target_relationships(train_img_df: pd.DataFrame):
    """Validate derived target relationships."""
    logger.info("\n=== Target Relationship Validation ===")
    
    # GDM_g ≈ Dry_Green_g + Dry_Clover_g
    expected_gdm = train_img_df["Dry_Green_g"] + train_img_df["Dry_Clover_g"]
    gdm_diff = (train_img_df["GDM_g"] - expected_gdm).abs()
    logger.info(f"\nGDM_g deviation from (Dry_Green_g + Dry_Clover_g):")
    logger.info(f"  Mean: {gdm_diff.mean():.2f}, Max: {gdm_diff.max():.2f}")
    logger.info(f"  % within 1g: {(gdm_diff < 1).mean()*100:.1f}%")
    
    # Dry_Total_g ≈ Dry_Green_g + Dry_Dead_g + Dry_Clover_g
    expected_total = train_img_df["Dry_Green_g"] + train_img_df["Dry_Dead_g"] + train_img_df["Dry_Clover_g"]
    total_diff = (train_img_df["Dry_Total_g"] - expected_total).abs()
    logger.info(f"\nDry_Total_g deviation from sum of base components:")
    logger.info(f"  Mean: {total_diff.mean():.2f}, Max: {total_diff.max():.2f}")
    logger.info(f"  % within 1g: {(total_diff < 1).mean()*100:.1f}%")


def analyze_covariates(train_img_df: pd.DataFrame, config: Config):
    """Analyze covariate distributions and correlations."""
    logger.info("\n=== Covariate Analysis ===")
    
    # Continuous features
    for feat in config.continuous_features:
        if feat in train_img_df.columns:
            values = train_img_df[feat].dropna()
            logger.info(f"\n{feat}:")
            logger.info(f"  Range: [{values.min():.3f}, {values.max():.3f}]")
            logger.info(f"  Mean: {values.mean():.3f} ± {values.std():.3f}")
    
    # Categorical features
    for feat in config.categorical_features:
        if feat in train_img_df.columns:
            logger.info(f"\n{feat} distribution:")
            logger.info(train_img_df[feat].value_counts())
    
    # Correlations with targets
    logger.info("\n=== Feature Correlations with Targets ===")
    corr_features = config.continuous_features + config.all_targets
    corr_features = [f for f in corr_features if f in train_img_df.columns]
    
    corr_matrix = train_img_df[corr_features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature-Target Correlation Matrix")
    plt.tight_layout()
    plt.savefig(config.output_dir / "correlation_matrix.png", dpi=150)
    logger.info(f"Saved correlation matrix to {config.output_dir / 'correlation_matrix.png'}")
    plt.close()
    
    # Print correlations with Dry_Total_g
    if "Dry_Total_g" in train_img_df.columns:
        logger.info("\nCorrelations with Dry_Total_g:")
        for feat in config.continuous_features:
            if feat in train_img_df.columns:
                corr = train_img_df[[feat, "Dry_Total_g"]].corr().iloc[0, 1]
                logger.info(f"  {feat}: {corr:.3f}")


def analyze_seasonality(train_img_df: pd.DataFrame, config: Config):
    """Analyze seasonal patterns."""
    if "season" not in train_img_df.columns:
        return
    
    logger.info("\n=== Seasonality Analysis ===")
    
    # Target stats by season
    season_stats = train_img_df.groupby("season")["Dry_Total_g"].agg(["count", "mean", "std"])
    logger.info(f"\nDry_Total_g by season:\n{season_stats}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    train_img_df.boxplot(column="Dry_Total_g", by="season", ax=ax)
    ax.set_title("Dry_Total_g by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Dry_Total_g (grams)")
    plt.suptitle("")  # Remove default title
    plt.tight_layout()
    plt.savefig(config.output_dir / "seasonality.png", dpi=150)
    logger.info(f"Saved seasonality plot to {config.output_dir / 'seasonality.png'}")
    plt.close()


def main():
    """Run EDA."""
    config = Config()
    config.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    train_df, test_df, sample_sub_df = load_and_validate_data(
        train_csv=config.train_csv,
        test_csv=config.test_csv,
        sample_submission_csv=config.sample_submission_csv,
        data_dir=config.data_dir,
    )
    
    # Create image-level dataframe
    train_img_df = create_image_level_dataframe(train_df)
    
    logger.info(f"\nImage-level data shape: {train_img_df.shape}")
    logger.info(f"Columns: {list(train_img_df.columns)}")
    
    # Run analyses
    analyze_target_distributions(train_img_df, config)
    validate_target_relationships(train_img_df)
    analyze_covariates(train_img_df, config)
    analyze_seasonality(train_img_df, config)
    
    logger.info("\n✓ EDA completed!")


if __name__ == "__main__":
    main()
