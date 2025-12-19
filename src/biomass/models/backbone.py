"""Vision backbone models (Step 7A)."""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)


def get_vision_backbone(
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    freeze: bool = True,
) -> tuple:
    """Get pre-trained vision backbone.
    
    Args:
        model_name: Model architecture name (from timm)
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze backbone parameters
        
    Returns:
        Tuple of (model, embedding_dim)
    """
    logger.info(f"Loading backbone: {model_name} (pretrained={pretrained}, freeze={freeze})")
    
    # Load model from timm
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,  # Remove classification head, get embeddings
        global_pool="avg",  # Average pooling
    )
    
    # Get embedding dimension
    if hasattr(model, "num_features"):
        embedding_dim = model.num_features
    else:
        # Try to infer from forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = model(dummy_input)
        embedding_dim = dummy_output.shape[1]
    
    logger.info(f"Backbone embedding dimension: {embedding_dim}")
    
    # Freeze if requested
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        logger.info("Backbone frozen")
    
    return model, embedding_dim


class VisionBackboneWrapper(nn.Module):
    """Wrapper for vision backbone with optional embedding caching."""
    
    def __init__(
        self,
        backbone: nn.Module,
        freeze: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze = freeze
        
        if freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Forward pass through backbone."""
        if self.freeze:
            with torch.no_grad():
                embeddings = self.backbone(x)
        else:
            embeddings = self.backbone(x)
        return embeddings
