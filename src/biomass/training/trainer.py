"""Training and evaluation loops (Step 8)."""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from .metrics import expand_predictions_to_long_format, evaluate_predictions

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: str,
    use_amp: bool = True,
    scaler: GradScaler = None,
) -> Dict:
    """Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP
        
    Returns:
        Dict with training metrics
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        images, tabular, targets, metadata = batch
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        if use_amp and scaler is not None:
            with autocast('cuda'):
                predictions = model(images, tabular)
                loss = criterion(predictions, targets)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images, tabular)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        "train_loss": avg_loss,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: str,
    config,
    val_df_long: pd.DataFrame,
) -> Dict:
    """Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        device: Device
        config: Config object with target info
        val_df_long: Long-format validation dataframe with ground truth
        
    Returns:
        Dict with evaluation metrics and predictions
    """
    model.eval()
    
    all_image_ids = []
    all_predictions = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    
    for batch in pbar:
        images, tabular, targets, metadata = batch
        images = images.to(device)
        tabular = tabular.to(device)
        
        # Get predictions for all 5 targets
        pred_dict = model.predict_all_targets(images, tabular)
        all_preds = pred_dict["all"].cpu().numpy()  # [batch, 5]
        
        all_predictions.append(all_preds)
        all_image_ids.extend(metadata["image_id"])
    
    # Concatenate all predictions
    all_predictions = np.vstack(all_predictions)  # [n_images, 5]
    
    # Expand to long format
    pred_df = expand_predictions_to_long_format(
        image_ids=all_image_ids,
        predictions=all_predictions,
        target_names=config.all_targets,
    )
    
    # Evaluate against ground truth
    metrics = evaluate_predictions(
        true_df=val_df_long,
        pred_df=pred_df,
        target_weights=config.target_weights,
        all_targets=config.all_targets,
    )
    
    logger.info(f"Validation - Weighted R²: {metrics['weighted_r2']:.4f}")
    for target, r2 in metrics["per_target_r2"].items():
        logger.info(f"  {target}: {r2:.4f}")
    
    return {
        **metrics,
        "predictions": pred_df,
    }


def create_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Create optimizer with separate parameter groups for backbone and head.
    
    Args:
        model: Model
        config: Config object
        
    Returns:
        Optimizer
    """
    # Separate backbone and head parameters
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "vision_backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    # If backbone is frozen, only optimize head
    if config.freeze_backbone or len(backbone_params) == 0:
        optimizer = torch.optim.AdamW(
            head_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        logger.info(f"Optimizer created: AdamW (head only), lr={config.learning_rate}")
    else:
        # Two parameter groups with different LRs
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": config.learning_rate * 0.1},
            {"params": head_params, "lr": config.learning_rate},
        ], weight_decay=config.weight_decay)
        logger.info(f"Optimizer created: AdamW (2 groups), head_lr={config.learning_rate}, "
                   f"backbone_lr={config.learning_rate * 0.1}")
    
    return optimizer
