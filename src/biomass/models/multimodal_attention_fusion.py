"""Enhanced Fusion Model with Advanced Techniques for Performance Boost."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """x: [batch, channels, seq_len] or [batch, channels]"""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            squeeze_flag = True
        else:
            squeeze_flag = False
            
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        result = x * y.expand_as(x)
        
        return result.squeeze(-1) if squeeze_flag else result


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for feature refinement."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: [batch, seq_len, dim]"""
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and residual connections."""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention between vision and tabular features."""
    
    def __init__(self, vision_dim: int, tabular_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = vision_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_vision = nn.Linear(vision_dim, vision_dim)
        self.k_tabular = nn.Linear(tabular_dim, vision_dim)
        self.v_tabular = nn.Linear(tabular_dim, vision_dim)
        
        self.q_tabular = nn.Linear(tabular_dim, vision_dim)
        self.k_vision = nn.Linear(vision_dim, vision_dim)
        self.v_vision = nn.Linear(vision_dim, vision_dim)
        
        self.proj_vision = nn.Linear(vision_dim, vision_dim)
        self.proj_tabular = nn.Linear(vision_dim, tabular_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, vision_feat, tabular_feat):
        """
        vision_feat: [B, vision_dim]
        tabular_feat: [B, tabular_dim]
        """
        B = vision_feat.shape[0]
        
        # Add sequence dimension
        vision_feat = vision_feat.unsqueeze(1)  # [B, 1, vision_dim]
        tabular_feat = tabular_feat.unsqueeze(1)  # [B, 1, tabular_dim]
        
        # Vision attends to tabular
        q_v = self.q_vision(vision_feat).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_t = self.k_tabular(tabular_feat).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_t = self.v_tabular(tabular_feat).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_vt = (q_v @ k_t.transpose(-2, -1)) * self.scale
        attn_vt = attn_vt.softmax(dim=-1)
        attn_vt = self.dropout(attn_vt)
        
        vision_enhanced = (attn_vt @ v_t).transpose(1, 2).reshape(B, 1, -1)
        vision_enhanced = self.proj_vision(vision_enhanced).squeeze(1)
        
        # Tabular attends to vision
        q_t = self.q_tabular(tabular_feat).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_v = self.k_vision(vision_feat).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_v = self.v_vision(vision_feat).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_tv = (q_t @ k_v.transpose(-2, -1)) * self.scale
        attn_tv = attn_tv.softmax(dim=-1)
        attn_tv = self.dropout(attn_tv)
        
        tabular_enhanced = (attn_tv @ v_v).transpose(1, 2).reshape(B, 1, -1)
        tabular_enhanced = self.proj_tabular(tabular_enhanced).squeeze(1)
        
        return vision_enhanced, tabular_enhanced


class AdaptiveFeatureFusion(nn.Module):
    """Adaptive gating mechanism for feature fusion."""
    
    def __init__(self, vision_dim: int, tabular_dim: int):
        super().__init__()
        total_dim = vision_dim + tabular_dim
        
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim // 2, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, vision_feat, tabular_feat):
        """Adaptively weight vision and tabular features."""
        concat = torch.cat([vision_feat, tabular_feat], dim=1)
        weights = self.gate(concat)
        
        # Apply weights
        vision_weighted = vision_feat * weights[:, 0:1]
        tabular_weighted = tabular_feat * weights[:, 1:2]
        
        return torch.cat([vision_weighted, tabular_weighted], dim=1), weights


class EnhancedBiomassPredictor(nn.Module):
    """Enhanced fusion model with advanced attention and gating mechanisms.
    
    Key improvements:
    1. Cross-attention between vision and tabular features
    2. SE blocks for channel attention
    3. Transformer blocks for feature refinement
    4. Adaptive gating for multi-modal fusion
    5. Residual connections throughout
    6. Multi-scale feature processing
    """
    
    def __init__(
        self,
        vision_backbone: nn.Module,
        vision_embedding_dim: int,
        num_categorical_features: int,
        categorical_embedding_dim: int,
        num_continuous_features: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_outputs: int = 3,
        freeze_backbone: bool = True,
        use_cross_attention: bool = True,
        use_transformer_blocks: bool = True,
        num_transformer_layers: int = 2,
    ):
        """Initialize enhanced fusion model.
        
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
            use_cross_attention: Use cross-attention between modalities
            use_transformer_blocks: Use transformer blocks for refinement
            num_transformer_layers: Number of transformer layers
        """
        super().__init__()
        
        self.vision_backbone = vision_backbone
        self.freeze_backbone = freeze_backbone
        self.num_categorical_features = num_categorical_features
        self.num_continuous_features = num_continuous_features
        self.use_cross_attention = use_cross_attention
        self.use_transformer_blocks = use_transformer_blocks
        
        if freeze_backbone:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            self.vision_backbone.eval()
        
        # Enhanced categorical embeddings with attention
        if num_categorical_features > 0:
            self.cat_embedding = nn.Sequential(
                nn.Linear(num_categorical_features, categorical_embedding_dim),
                nn.LayerNorm(categorical_embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )
            cat_dim = categorical_embedding_dim
        else:
            self.cat_embedding = None
            cat_dim = 0
        
        # Continuous feature processing
        if num_continuous_features > 0:
            self.cont_processing = nn.Sequential(
                nn.Linear(num_continuous_features, num_continuous_features * 2),
                nn.LayerNorm(num_continuous_features * 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(num_continuous_features * 2, num_continuous_features * 2),
            )
            cont_dim = num_continuous_features * 2
        else:
            self.cont_processing = None
            cont_dim = 0
        
        tabular_dim = cat_dim + cont_dim
        
        # Vision feature enhancement
        self.vision_enhancement = nn.Sequential(
            nn.Linear(vision_embedding_dim, vision_embedding_dim),
            nn.LayerNorm(vision_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # SE block for vision features
        self.vision_se = SEBlock(vision_embedding_dim, reduction=16)
        
        # Tabular feature enhancement
        if tabular_dim > 0:
            self.tabular_enhancement = nn.Sequential(
                nn.Linear(tabular_dim, tabular_dim),
                nn.LayerNorm(tabular_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            self.tabular_se = SEBlock(tabular_dim, reduction=8)
        else:
            self.tabular_enhancement = None
            self.tabular_se = None
        
        # Cross-attention fusion
        if use_cross_attention and tabular_dim > 0:
            self.cross_attention = CrossAttentionFusion(
                vision_embedding_dim, 
                tabular_dim,
                num_heads=4,
                dropout=dropout
            )
        else:
            self.cross_attention = None
        
        # Adaptive fusion gate
        if tabular_dim > 0:
            self.adaptive_fusion = AdaptiveFeatureFusion(vision_embedding_dim, tabular_dim)
            fusion_input_dim = vision_embedding_dim + tabular_dim
        else:
            self.adaptive_fusion = None
            fusion_input_dim = vision_embedding_dim
        
        # Transformer blocks for feature refinement
        if use_transformer_blocks:
            # Calculate appropriate number of heads based on fusion_input_dim
            transformer_heads = min(8, fusion_input_dim // 64)
            # Ensure dimension is divisible by number of heads
            while fusion_input_dim % transformer_heads != 0 and transformer_heads > 1:
                transformer_heads -= 1
            
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(
                    fusion_input_dim,
                    num_heads=transformer_heads,
                    mlp_ratio=4.0,
                    dropout=dropout
                )
                for _ in range(num_transformer_layers)
            ])
        else:
            self.transformer_blocks = None
        
        # Enhanced fusion head with residual connections
        self.fusion_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ),
            nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            ),
        ])
        
        # Separate prediction heads for each component (better than shared head)
        head_input_dim = hidden_dim // 4
        
        self.head_green = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(head_input_dim // 2, 1),
        )
        
        self.head_dead = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(head_input_dim // 2, 1),
        )
        
        self.head_clover = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(head_input_dim // 2, 1),
        )
        
        # Optional: Auxiliary prediction head from intermediate features
        self.aux_predictor = nn.Linear(hidden_dim, num_outputs)
        
        logger.info(f"Enhanced Fusion Model created:")
        logger.info(f"  Vision dim: {vision_embedding_dim}")
        logger.info(f"  Tabular dim: {tabular_dim} (cat={cat_dim}, cont={cont_dim})")
        logger.info(f"  Fusion dim: {fusion_input_dim}")
        logger.info(f"  Cross-attention: {use_cross_attention}")
        logger.info(f"  Transformer blocks: {num_transformer_layers if use_transformer_blocks else 0}")
    
    def forward(
        self, 
        image, 
        tabular_features,
        return_intermediate: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Forward pass with enhanced feature fusion.
        
        Args:
            image: Image tensor [batch, 3, H, W]
            tabular_features: Tabular features [batch, num_features]
            return_intermediate: Return intermediate features for analysis
        
        Returns:
            Predictions for 3 base components [batch, 3]
            or dict with predictions and intermediate features
        """
        intermediates = {}
        
        # Extract vision embeddings
        if self.freeze_backbone:
            with torch.no_grad():
                vision_emb = self.vision_backbone(image)
        else:
            vision_emb = self.vision_backbone(image)
        
        # Enhance vision features
        vision_feat = self.vision_enhancement(vision_emb)
        vision_feat = vision_feat + self.vision_se(vision_feat)  # Residual + SE
        intermediates['vision_enhanced'] = vision_feat
        
        # Process tabular features
        if self.num_categorical_features > 0 and tabular_features.shape[1] >= self.num_categorical_features:
            cat_features = tabular_features[:, :self.num_categorical_features]
            cont_features = tabular_features[:, self.num_categorical_features:]
            
            # Process categorical
            cat_emb = self.cat_embedding(cat_features) if self.cat_embedding else cat_features
            
            # Process continuous
            if self.cont_processing and cont_features.shape[1] > 0:
                cont_emb = self.cont_processing(cont_features)
            else:
                cont_emb = cont_features
            
            # Combine tabular features
            tabular_feat = torch.cat([cat_emb, cont_emb], dim=1)
        else:
            # All continuous
            if self.cont_processing:
                tabular_feat = self.cont_processing(tabular_features)
            else:
                tabular_feat = tabular_features
        
        # Enhance tabular features
        if self.tabular_enhancement is not None:
            tabular_feat = self.tabular_enhancement(tabular_feat)
            tabular_feat = tabular_feat + self.tabular_se(tabular_feat)  # Residual + SE
        
        intermediates['tabular_enhanced'] = tabular_feat
        
        # Cross-attention between modalities
        if self.cross_attention is not None:
            vision_cross, tabular_cross = self.cross_attention(vision_feat, tabular_feat)
            vision_feat = vision_feat + vision_cross  # Residual
            tabular_feat = tabular_feat + tabular_cross  # Residual
            intermediates['cross_attention'] = True
        
        # Adaptive fusion
        if self.adaptive_fusion is not None:
            fused, fusion_weights = self.adaptive_fusion(vision_feat, tabular_feat)
            intermediates['fusion_weights'] = fusion_weights
        else:
            fused = vision_feat
        
        # Transformer refinement
        if self.transformer_blocks is not None:
            fused_with_seq = fused.unsqueeze(1)  # Add sequence dimension
            for block in self.transformer_blocks:
                fused_with_seq = block(fused_with_seq)
            fused = fused_with_seq.squeeze(1)
        
        # Fusion head with residuals
        x = fused
        for i, layer in enumerate(self.fusion_head):
            x_new = layer(x)
            # Add residual connection where dimensions match
            if i == 0:
                x = x_new
                intermediates['fusion_layer_0'] = x
                # Auxiliary prediction from intermediate features
                aux_pred = self.aux_predictor(x)
                intermediates['aux_predictions'] = aux_pred
            else:
                x = x_new
        
        # Separate predictions for each component
        dry_green = self.head_green(x)
        dry_dead = self.head_dead(x)
        dry_clover = self.head_clover(x)
        
        # Combine predictions
        base_predictions = torch.cat([dry_green, dry_dead, dry_clover], dim=1)
        
        if return_intermediate:
            intermediates['base_predictions'] = base_predictions
            return intermediates
        
        return base_predictions
    
    def predict_all_targets(
        self, 
        image, 
        tabular_features,
        return_intermediate: bool = False
    ):
        """Predict all 5 targets (3 base + 2 derived).
        
        Returns:
            Dictionary with predictions for all 5 targets
        """
        # Get base predictions (and optionally intermediates)
        if return_intermediate:
            intermediates = self.forward(image, tabular_features, return_intermediate=True)
            base = intermediates['base_predictions']
        else:
            base = self.forward(image, tabular_features, return_intermediate=False)
            intermediates = {}
        
        # Extract base components
        dry_green = base[:, 0:1]
        dry_dead = base[:, 1:2]
        dry_clover = base[:, 2:3]
        
        # Derive GDM and Total
        gdm = dry_green + dry_clover
        dry_total = dry_green + dry_dead + dry_clover
        
        # Concatenate all
        all_predictions = torch.cat([dry_green, dry_dead, dry_clover, gdm, dry_total], dim=1)
        
        result = {
            "base": base,
            "all": all_predictions,
            "Dry_Green_g": dry_green,
            "Dry_Dead_g": dry_dead,
            "Dry_Clover_g": dry_clover,
            "GDM_g": gdm,
            "Dry_Total_g": dry_total,
        }
        
        if return_intermediate:
            result['intermediates'] = intermediates
        
        return result
