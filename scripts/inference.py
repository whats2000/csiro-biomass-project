"""Inference script to generate submission.csv (Step 9)."""

import sys
from pathlib import Path
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from biomass.utils import Config, set_seed
from biomass.data import (
    load_and_validate_data,
    create_image_level_dataframe,
    get_tabular_statistics,
    get_val_transforms,
    BiomassDataset,
)
from biomass.models import get_vision_backbone, BiomassPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_trained_model(checkpoint_path: Path, config: Config):
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Config object
        
    Returns:
        Loaded model
    """
    # Create model architecture
    backbone, embedding_dim = get_vision_backbone(
        model_name=config.backbone,
        pretrained=False,  # Will load from checkpoint
        freeze=config.freeze_backbone,
    )
    
    num_cat = len(config.categorical_features)
    num_cont = len(config.continuous_features)
    
    model = BiomassPredictor(
        vision_backbone=backbone,
        vision_embedding_dim=embedding_dim,
        num_categorical_features=num_cat,
        categorical_embedding_dim=16,
        num_continuous_features=num_cont,
        hidden_dim=256,
        dropout=0.3,
        num_outputs=len(config.base_targets),
        freeze_backbone=config.freeze_backbone,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint['epoch']}, Val R²: {checkpoint.get('val_r2', 'N/A')}")
    
    return model


@torch.no_grad()
def predict_test(
    model,
    test_loader,
    device: str,
    config: Config,
) -> pd.DataFrame:
    """Generate predictions for test set.
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device
        config: Config object
        
    Returns:
        DataFrame with predictions for all 5 targets
    """
    model.eval()
    
    all_image_ids = []
    all_predictions = []
    
    for batch in tqdm(test_loader, desc="Predicting"):
        images, tabular, _, metadata = batch
        images = images.to(device)
        tabular = tabular.to(device)
        
        # Get predictions for all 5 targets
        pred_dict = model.predict_all_targets(images, tabular)
        all_preds = pred_dict["all"].cpu().numpy()  # [batch, 5]
        
        # Apply non-negativity constraint
        all_preds = np.maximum(all_preds, 0.0)
        
        all_predictions.append(all_preds)
        all_image_ids.extend(metadata["image_id"])
    
    # Concatenate all predictions
    all_predictions = np.vstack(all_predictions)  # [n_images, 5]
    
    # Create dataframe
    pred_records = []
    for img_id, preds in zip(all_image_ids, all_predictions):
        for target_name, pred_val in zip(config.all_targets, preds):
            pred_records.append({
                "image_id": img_id,
                "target_name": target_name,
                "prediction": pred_val,
            })
    
    pred_df = pd.DataFrame(pred_records)
    
    return pred_df


def create_submission(
    pred_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    output_path: Path,
):
    """Create submission.csv in the correct format.
    
    Args:
        pred_df: Predictions in long format
        sample_submission_df: Sample submission for alignment
        output_path: Path to save submission.csv
    """
    # Create sample_id
    pred_df["sample_id"] = pred_df["image_id"] + "__" + pred_df["target_name"]
    
    # Select and rename columns
    submission = pred_df[["sample_id", "prediction"]].rename(columns={"prediction": "target"})
    
    # Align with sample submission
    submission = sample_submission_df[["sample_id"]].merge(
        submission,
        on="sample_id",
        how="left",
    )
    
    # Check for missing predictions
    missing = submission["target"].isna().sum()
    if missing > 0:
        logger.warning(f"Missing {missing} predictions! Filling with 0.")
        submission["target"] = submission["target"].fillna(0.0)
    
    # Verify format
    assert len(submission) == len(sample_submission_df), "Submission length mismatch"
    assert set(submission.columns) == {"sample_id", "target"}, "Submission columns mismatch"
    
    # Save
    submission.to_csv(output_path, index=False)
    logger.info(f"✓ Saved submission to {output_path}")
    logger.info(f"  Rows: {len(submission)}")
    logger.info(f"  Target range: [{submission['target'].min():.2f}, {submission['target'].max():.2f}]")


def main():
    """Main inference function."""
    config = Config()
    set_seed(config.seed)
    
    logger.info(f"Using device: {config.device}")
    
    # Load data
    train_df, test_df, sample_sub_df = load_and_validate_data(
        train_csv=config.train_csv,
        test_csv=config.test_csv,
        sample_submission_csv=config.sample_submission_csv,
        data_dir=config.data_dir,
    )
    
    # Get train statistics for normalization
    train_img_df = create_image_level_dataframe(train_df)
    continuous_stats = get_tabular_statistics(train_img_df, config.continuous_features)
    
    # Build categorical encoders from full training data
    cat_encoders = {}
    for feat in config.categorical_features:
        if feat in train_img_df.columns:
            unique_vals = sorted(train_img_df[feat].dropna().unique())
            cat_encoders[feat] = {val: idx for idx, val in enumerate(unique_vals)}
    
    # Create test dataset
    test_img_df = create_image_level_dataframe(test_df)
    test_transform = get_val_transforms(config.image_size)
    
    test_dataset = BiomassDataset(
        df=test_img_df,
        transform=test_transform,
        categorical_features=config.categorical_features,
        continuous_features=config.continuous_features,
        cat_encoders=cat_encoders,
        continuous_stats=continuous_stats,
        base_targets=config.base_targets,
        is_test=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Load the trained model
    model_path = config.output_dir / "model_best.pth"
    if not model_path.exists():
        logger.error(f"No trained model found at {model_path}! Run train.py first.")
        return
    
    model = load_trained_model(model_path, config)
    
    # Generate predictions
    pred_df = predict_test(
        model=model,
        test_loader=test_loader,
        device=config.device,
        config=config,
    )
    
    # Create submission
    submission_path = config.output_dir / "submission.csv"
    create_submission(
        pred_df=pred_df,
        sample_submission_df=sample_sub_df,
        output_path=submission_path,
    )
    
    logger.info("\n✓ Inference completed successfully!")


if __name__ == "__main__":
    main()
