"""Main training script without cross-validation."""

import sys
import argparse
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.amp import GradScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from biomass.utils import Config, set_seed
from biomass.data import (
    load_and_validate_data,
    create_image_level_dataframe,
    get_tabular_statistics,
    get_train_transforms,
    get_val_transforms,
    create_dataloaders,
)
from biomass.training import train_one_epoch, evaluate, create_optimizer
from biomass.models import get_vision_backbone, get_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_model(config: Config):
    """Train and evaluate a single model.
    
    Args:
        config: Configuration object
        
    Returns:
        Dict with training results
    """
    logger.info(f"\n{'='*60}\nTraining Model\n{'='*60}")
    
    # Load data
    train_df, test_df, sample_sub_df = load_and_validate_data(
        train_csv=config.train_csv,
        test_csv=config.test_csv,
        sample_submission_csv=config.sample_submission_csv,
        data_dir=config.data_dir,
    )
    
    # Create image-level dataframe
    train_img_df = create_image_level_dataframe(train_df)
    
    # Split into train and validation (80/20 split)
    train_data, val_data = train_test_split(
        train_img_df, 
        test_size=0.2, 
        random_state=config.seed,
        shuffle=True
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    logger.info(f"Train: {len(train_data)} images, Val: {len(val_data)} images")
    
    # Get tabular statistics from training data only
    continuous_stats = get_tabular_statistics(train_data, config.continuous_features)
    
    # Build categorical encoders from training data
    cat_encoders = {}
    for feat in config.categorical_features:
        if feat in train_data.columns:
            unique_vals = sorted(train_data[feat].dropna().unique())
            cat_encoders[feat] = {val: idx for idx, val in enumerate(unique_vals)}
    
    # Create transforms
    train_transform = get_train_transforms(config.image_size)
    val_transform = get_val_transforms(config.image_size)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df=train_data,
        val_df=val_data,
        train_transform=train_transform,
        val_transform=val_transform,
        config=config,
        cat_encoders=cat_encoders,
        continuous_stats=continuous_stats,
    )
    
    # Create model
    num_cat = len(config.categorical_features)
    num_cont = len(config.continuous_features)
    
    logger.info(f"Creating model: {config.model_name}")
    
    # DINO model creates its own backbone internally
    if config.model_name.lower() in ['dino_vision', 'dino', 'vision_only']:
        model = get_model(
            model_name=config.model_name,
            vision_backbone=None,  # DINO creates its own
            vision_embedding_dim=768,  # DINO default
            num_categorical_features=num_cat,
            categorical_embedding_dim=config.categorical_embedding_dim,
            num_continuous_features=num_cont,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            num_outputs=len(config.base_targets),
            freeze_backbone=config.freeze_backbone,
        )
    else:
        # Other models need a separate backbone
        backbone, embedding_dim = get_vision_backbone(
            model_name=config.backbone,
            pretrained=config.pretrained,
            freeze=config.freeze_backbone,
        )
        
        model = get_model(
            model_name=config.model_name,
            vision_backbone=backbone,
            vision_embedding_dim=embedding_dim,
            num_categorical_features=num_cat,
            categorical_embedding_dim=config.categorical_embedding_dim,
            num_continuous_features=num_cont,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            num_outputs=len(config.base_targets),
            freeze_backbone=config.freeze_backbone,
            # Additional arguments for advanced models
            use_cross_attention=config.use_cross_attention,
            use_transformer_blocks=config.use_transformer_blocks,
            num_transformer_layers=config.num_transformer_layers,
        )
    
    model = model.to(config.device)
    
    # Create optimizer and loss
    optimizer = create_optimizer(model, config)
    criterion = nn.SmoothL1Loss()  # Huber loss
    
    # AMP scaler
    scaler = GradScaler('cuda') if config.use_amp else None
    
    # Training loop
    best_val_r2 = -np.inf
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    # Prepare validation long-format data
    val_df_long = train_df[train_df["image_id"].isin(val_data["image_id"])].copy()
    
    for epoch in range(config.epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=config.device,
            use_amp=config.use_amp,
            scaler=scaler,
        )
        
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=config.device,
            config=config,
            val_df_long=val_df_long,
        )
        
        val_r2 = val_metrics["weighted_r2"]
        
        # Early stopping
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            save_path = config.output_dir / "model_best.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_r2": val_r2,
                "config": config,
            }, save_path)
            logger.info(f"✓ Saved best model (R²={val_r2:.4f}) to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"\nTraining completed - Best Val R²: {best_val_r2:.4f} at epoch {best_epoch+1}")
    
    # Save validation predictions
    val_predictions_df = val_metrics["predictions"]
    val_predictions_df.to_csv(config.output_dir / "val_predictions.csv", index=False)
    logger.info(f"Saved validation predictions to {config.output_dir / 'val_predictions.csv'}")
    
    return {
        "best_val_r2": best_val_r2,
        "best_epoch": best_epoch,
    }


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train biomass prediction model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file (e.g., configs/train_simple_concat.yaml)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name override (simple_concat, multimodal_attention)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory override'
    )
    args = parser.parse_args()
    
    # Setup
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = Config.from_yaml(args.config)
    else:
        logger.info("Using default config")
        config = Config()
    
    # Apply command-line overrides
    if args.model:
        config.model_name = args.model
        logger.info(f"Model override: {args.model}")
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        logger.info(f"Output directory override: {args.output_dir}")
    
    set_seed(config.seed)
    
    logger.info(f"Using device: {config.device}")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Backbone: {config.backbone}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Save config to output directory
    config_save_path = config.output_dir / "config.yaml"
    config.save_yaml(config_save_path)
    logger.info(f"Saved config to: {config_save_path}")
    
    # Train the model
    result = train_model(config)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Best Validation R²: {result['best_val_r2']:.4f}")
    logger.info(f"Best Epoch: {result['best_epoch'] + 1}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
