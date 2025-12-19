"""Evaluation metrics (Step 8)."""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def compute_weighted_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    weights: Dict[str, float],
) -> float:
    """Compute global weighted R² metric.
    
    This is the competition metric:
    - Compute weighted residuals and weighted total variance
    - Return single R² across all rows
    
    Args:
        y_true: True values [n_samples, n_targets] in long format
        y_pred: Predicted values [n_samples, n_targets] in long format
        target_names: List of target names corresponding to columns
        weights: Dict mapping target_name to weight
        
    Returns:
        Global weighted R²
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Flatten if needed
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    
    assert len(y_true) == len(y_pred), "Length mismatch"
    
    # Build weight array (repeat weight for each target occurrence)
    n_samples_per_target = len(y_true) // len(target_names)
    weight_array = []
    for target in target_names:
        weight_array.extend([weights[target]] * n_samples_per_target)
    weight_array = np.array(weight_array)
    
    # Weighted residuals
    weighted_residuals = weight_array * (y_true - y_pred) ** 2
    ss_res = np.sum(weighted_residuals)
    
    # Weighted total variance
    weighted_mean = np.sum(weight_array * y_true) / np.sum(weight_array)
    weighted_total_var = weight_array * (y_true - weighted_mean) ** 2
    ss_tot = np.sum(weighted_total_var)
    
    # R²
    if ss_tot < 1e-8:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    
    return r2


def compute_per_target_r2(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute R² per target (for diagnostic purposes).
    
    Args:
        y_true_dict: Dict mapping target name to true values
        y_pred_dict: Dict mapping target name to predicted values
        
    Returns:
        Dict mapping target name to R²
    """
    r2_scores = {}
    
    for target_name in y_true_dict.keys():
        y_true = np.array(y_true_dict[target_name])
        y_pred = np.array(y_pred_dict[target_name])
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        
        if ss_tot < 1e-8:
            r2 = 0.0
        else:
            r2 = 1 - (ss_res / ss_tot)
        
        r2_scores[target_name] = r2
    
    return r2_scores


def expand_predictions_to_long_format(
    image_ids: List[str],
    predictions: np.ndarray,
    target_names: List[str],
) -> pd.DataFrame:
    """Expand image-level predictions to long format for metric computation.
    
    Args:
        image_ids: List of image IDs
        predictions: Array of predictions [n_images, n_targets]
        target_names: List of target names (5 targets)
        
    Returns:
        Long-format dataframe with columns [image_id, target_name, prediction]
    """
    records = []
    
    for img_id, pred_row in zip(image_ids, predictions):
        for target_name, pred_val in zip(target_names, pred_row):
            records.append({
                "image_id": img_id,
                "target_name": target_name,
                "prediction": pred_val,
            })
    
    df = pd.DataFrame(records)
    return df


def evaluate_predictions(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_weights: Dict[str, float],
    all_targets: List[str],
) -> Dict:
    """Evaluate predictions against ground truth.
    
    Args:
        true_df: Long-format dataframe with ground truth
        pred_df: Long-format dataframe with predictions
        target_weights: Weight for each target
        all_targets: List of all target names
        
    Returns:
        Dict with evaluation metrics
    """
    # Merge predictions with ground truth
    merged = true_df.merge(
        pred_df,
        on=["image_id", "target_name"],
        how="inner",
        suffixes=("_true", "_pred"),
    )
    
    # Extract true and predicted values
    y_true = merged["target"].values
    y_pred = merged["prediction"].values
    
    # Compute global weighted R²
    weighted_r2 = compute_weighted_r2(
        y_true=y_true,
        y_pred=y_pred,
        target_names=all_targets,
        weights=target_weights,
    )
    
    # Compute per-target R²
    y_true_dict = {}
    y_pred_dict = {}
    for target in all_targets:
        mask = merged["target_name"] == target
        y_true_dict[target] = merged.loc[mask, "target"].values
        y_pred_dict[target] = merged.loc[mask, "prediction"].values
    
    per_target_r2 = compute_per_target_r2(y_true_dict, y_pred_dict)
    
    return {
        "weighted_r2": weighted_r2,
        "per_target_r2": per_target_r2,
    }
