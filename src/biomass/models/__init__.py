"""Model architectures for biomass prediction.

Available models:
- SimpleConcatFusion: Basic concatenation-based fusion (baseline)
- MultiModalAttentionFusion: Enhanced fusion with cross-attention
- DINOPyramidMamba: Advanced DINO with pyramid pooling and Mamba blocks
"""

from .backbone import get_vision_backbone
from .simple_concat_fusion import BiomassPredictor as SimpleConcatFusion
from .multimodal_attention_fusion import EnhancedBiomassPredictor as MultiModalAttentionFusion

# Keep backward compatibility
BiomassPredictor = SimpleConcatFusion


def get_model(
    model_name: str,
    vision_backbone,
    vision_embedding_dim: int,
    num_categorical_features: int = 2,
    categorical_embedding_dim: int = 16,
    num_continuous_features: int = 2,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    num_outputs: int = 3,
    freeze_backbone: bool = True,
    **kwargs
):
    """Factory function to create different model architectures.
    
    Args:
        model_name: Name of the model architecture
            - 'simple_concat': SimpleConcatFusion (baseline)
            - 'multimodal_attention': MultiModalAttentionFusion
            - 'attention': Alias for multimodal_attention
        vision_backbone: Pre-trained vision model
        vision_embedding_dim: Dimension of vision embeddings
        num_categorical_features: Number of categorical features
        categorical_embedding_dim: Embedding dimension for categoricals
        num_continuous_features: Number of continuous features
        hidden_dim: Hidden dimension for fusion layers
        dropout: Dropout rate
        num_outputs: Number of outputs (3 base components)
        freeze_backbone: Whether to freeze vision backbone
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model instance
        
    Examples:
        >>> # Simple baseline model
        >>> model = get_model('simple_concat', backbone, 1280)
        
        >>> # Advanced attention model
        >>> model = get_model(
        ...     'multimodal_attention',
        ...     backbone,
        ...     1280,
        ...     use_cross_attention=True,
        ...     use_transformer_blocks=True,
        ...     num_transformer_layers=2
        ... )
    """
    model_name = model_name.lower()
    
    # Common arguments for all models
    common_args = {
        "vision_backbone": vision_backbone,
        "vision_embedding_dim": vision_embedding_dim,
        "num_categorical_features": num_categorical_features,
        "categorical_embedding_dim": categorical_embedding_dim,
        "num_continuous_features": num_continuous_features,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "num_outputs": num_outputs,
        "freeze_backbone": freeze_backbone,
    }
    
    if model_name in ['simple_concat', 'simple', 'baseline', 'concat']:
        return SimpleConcatFusion(**common_args)
    
    elif model_name in ['multimodal_attention', 'attention', 'enhanced', 'cross_attention']:
        # Default attention parameters
        attention_args = {
            'use_cross_attention': kwargs.get('use_cross_attention', True),
            'use_transformer_blocks': kwargs.get('use_transformer_blocks', True),
            'num_transformer_layers': kwargs.get('num_transformer_layers', 2),
        }
        return MultiModalAttentionFusion(**common_args, **attention_args)
    
    elif model_name in ['dino_vision', 'dino', 'vision_only']:
        # DINO vision-only model (uses dino_pyramid_mamba factory)
        from .dino_pyramid_mamba import create_enhanced_dino_model
        dropout = kwargs.get('dropout', common_args.get('dropout', 0.1))
        hidden_ratio = kwargs.get('hidden_ratio', 0.35)
        use_enhanced_pyramid = kwargs.get('use_enhanced_pyramid', True)
        use_enhanced_cross_fusion = kwargs.get('use_enhanced_cross_fusion', True)
        return create_enhanced_dino_model(
            dropout=dropout,
            hidden_ratio=hidden_ratio,
            use_enhanced_pyramid=use_enhanced_pyramid,
            use_enhanced_cross_fusion=use_enhanced_cross_fusion
        )
    
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Available models: 'simple_concat', 'multimodal_attention', 'dino_vision'"
        )


__all__ = [
    "get_vision_backbone",
    "get_model",
    "BiomassPredictor",  # For backward compatibility
    "SimpleConcatFusion",
    "MultiModalAttentionFusion",
]
