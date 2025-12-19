"""Fusion model: vision + tabular → biomass predictions (Step 7)."""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class BiomassPredictor(nn.Module):
    """Combined model: vision backbone + tabular fusion head.
    
    Predicts 3 base components (Dry_Green_g, Dry_Dead_g, Dry_Clover_g).
    Derived targets (GDM_g, Dry_Total_g) computed via summation.
    """
    
    def __init__(
        self,
        vision_backbone: nn.Module,
        vision_embedding_dim: int,
        num_categorical_features: int,
        categorical_embedding_dim: int,
        num_continuous_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_outputs: int = 3,  # 3 base components
        freeze_backbone: bool = True,
    ):
        """Initialize fusion model.
        
        Args:
            vision_backbone: Pre-trained vision model
            vision_embedding_dim: Dimension of vision embeddings
            num_categorical_features: Number of categorical features
            categorical_embedding_dim: Embedding dimension for each categorical
            num_continuous_features: Number of continuous features
            hidden_dim: Hidden dimension for fusion MLP
            dropout: Dropout rate
            num_outputs: Number of outputs (3 base components)
            freeze_backbone: Whether to freeze vision backbone
        """
        super().__init__()
        
        self.vision_backbone = vision_backbone
        self.freeze_backbone = freeze_backbone
        self.num_categorical_features = num_categorical_features
        self.num_continuous_features = num_continuous_features
        
        if freeze_backbone:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            self.vision_backbone.eval()
        
        # Categorical embeddings for better performance
        if num_categorical_features > 0:
            # Each categorical feature gets its own embedding
            # We'll use ModuleList to handle multiple categorical features
            # For now, use a single projection layer for all categorical features
            self.cat_projection = nn.Sequential(
                nn.Linear(num_categorical_features, categorical_embedding_dim),
                nn.ReLU(),
            )
            cat_dim = categorical_embedding_dim
        else:
            self.cat_projection = None
            cat_dim = 0
        
        # Fusion head
        fusion_input_dim = vision_embedding_dim + cat_dim + num_continuous_features
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_outputs),
        )
        
        logger.info(f"Fusion model created: vision_dim={vision_embedding_dim}, "
                   f"cat_features={num_categorical_features}→{cat_dim}, "
                   f"cont_features={num_continuous_features}, fusion_dim={fusion_input_dim}")
    
    def forward(self, image, tabular_features):
        """Forward pass.
        
        Args:
            image: Image tensor [batch, 3, H, W]
            tabular_features: Tabular features [batch, num_features]
                First num_categorical_features are categorical (as continuous indices/one-hot)
                Remaining are continuous (normalized)
        
        Returns:
            Predictions for 3 base components [batch, 3]
        """
        # Extract vision embeddings
        if self.freeze_backbone:
            with torch.no_grad():
                vision_emb = self.vision_backbone(image)
        else:
            vision_emb = self.vision_backbone(image)
        
        # Split tabular features into categorical and continuous
        if self.num_categorical_features > 0 and tabular_features.shape[1] >= self.num_categorical_features:
            cat_features = tabular_features[:, :self.num_categorical_features]
            cont_features = tabular_features[:, self.num_categorical_features:]
            
            # Project categorical features
            if self.cat_projection is not None:
                cat_emb = self.cat_projection(cat_features)
                # Concatenate: vision + categorical embeddings + continuous
                fused = torch.cat([vision_emb, cat_emb, cont_features], dim=1)
            else:
                # Concatenate: vision + categorical + continuous
                fused = torch.cat([vision_emb, cat_features, cont_features], dim=1)
        else:
            # No categorical features or all features are continuous
            fused = torch.cat([vision_emb, tabular_features], dim=1)
        
        # Predict base components
        base_predictions = self.fusion_head(fused)
        
        return base_predictions
    
    def predict_all_targets(self, image, tabular_features):
        """Predict all 5 targets (3 base + 2 derived).
        
        Returns:
            Dictionary with predictions for all 5 targets
        """
        # Get base predictions
        base = self.forward(image, tabular_features)
        
        # Extract base components
        dry_green = base[:, 0:1]
        dry_dead = base[:, 1:2]
        dry_clover = base[:, 2:3]
        
        # Derive GDM and Total
        gdm = dry_green + dry_clover
        dry_total = dry_green + dry_dead + dry_clover
        
        # Concatenate all
        all_predictions = torch.cat([dry_green, dry_dead, dry_clover, gdm, dry_total], dim=1)
        
        return {
            "base": base,
            "all": all_predictions,
            "Dry_Green_g": dry_green,
            "Dry_Dead_g": dry_dead,
            "Dry_Clover_g": dry_clover,
            "GDM_g": gdm,
            "Dry_Total_g": dry_total,
        }
