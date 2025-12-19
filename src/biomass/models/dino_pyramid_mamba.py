"""Advanced DINO-based model with additional performance optimizations.

This module contains enhanced versions of the vision models with:
1. Improved attention mechanisms (Scaled Cosine Attention)
2. Better feature pyramid design
3. Enhanced cross-scale fusion
4. Mamba v2 integration
5. Additional regularization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict


class ScaledCosineAttention(nn.Module):
    """Scaled cosine attention - more stable than dot-product attention."""
    
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(1, heads, 1, 1) * (dim // heads) ** -0.5)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Normalize for cosine similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Scaled cosine attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class StochasticDepth(nn.Module):
    """Stochastic depth (drop path) for better regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class EnhancedFeedForward(nn.Module):
    """Enhanced FFN with gating mechanism."""
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid * 2)  # For gating
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.act(gate)  # Gated activation
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EnhancedMobileViTBlock(nn.Module):
    """Enhanced MobileViT with better local-global interaction."""
    
    def __init__(
        self, 
        dim: int, 
        heads: int = 4, 
        depth: int = 2, 
        patch: Tuple[int, int] = (2, 2),
        dropout: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        # Local processing with depthwise separable conv
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        
        self.patch = patch
        
        # Transformer with cosine attention
        self.transformer = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(dim),
                'attn': ScaledCosineAttention(dim, heads=heads, dropout=dropout),
                'drop_path': StochasticDepth(drop_path),
                'norm2': nn.LayerNorm(dim),
                'ffn': EnhancedFeedForward(dim, mlp_ratio=3.0, dropout=dropout),
            })
            for _ in range(depth)
        ])
        
        # Fusion with attention
        self.fuse_attn = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Conv2d(dim * 2, dim, 1)
        
    def forward(self, x: torch.Tensor):
        identity = x
        local_feat = self.local(x)
        
        B, C, H, W = local_feat.shape
        ph, pw = self.patch
        
        # Pad if necessary
        new_h = math.ceil(H / ph) * ph
        new_w = math.ceil(W / pw) * pw
        if new_h != H or new_w != W:
            local_feat = F.interpolate(local_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
            H, W = new_h, new_w
        
        # Tokenize
        tokens = local_feat.unfold(2, ph, ph).unfold(3, pw, pw)
        tokens = tokens.contiguous().view(B, C, -1, ph, pw)
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        
        # Transform
        for block in self.transformer:
            h = block['norm1'](tokens)
            tokens = tokens + block['drop_path'](block['attn'](h))
            tokens = tokens + block['drop_path'](block['ffn'](block['norm2'](tokens)))
        
        # Detokenize
        nh, nw = H // ph, W // pw
        feat = tokens.view(B, nh, nw, ph, pw, C).permute(0, 5, 1, 3, 2, 4)
        feat = feat.reshape(B, C, H, W)
        
        if feat.shape[-2:] != identity.shape[-2:]:
            feat = F.interpolate(feat, size=identity.shape[-2:], mode="bilinear", align_corners=False)
        
        # Attention-based fusion
        concat = torch.cat([identity, feat], dim=1)
        attn = self.fuse_attn(concat)
        out = self.fuse_conv(concat * attn)
        
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling for multi-scale context."""
    
    def __init__(self, dim: int, pool_sizes: List[int] = [1, 2, 3, 6]):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(dim, dim // len(pool_sizes), 1),
                nn.BatchNorm2d(dim // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
        self.fusion = nn.Conv2d(dim + dim, dim, 1)
        
    def forward(self, x):
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        
        pools = []
        for conv in self.convs:
            pool = conv(x)
            pool = F.interpolate(pool, size=(H, W), mode="bilinear", align_corners=False)
            pools.append(pool)
        
        pools = torch.cat(pools, dim=1)
        out = self.fusion(torch.cat([x, pools], dim=1))
        return out


class EnhancedPyramidMixer(nn.Module):
    """Enhanced pyramid mixer with better multi-scale processing."""
    
    def __init__(
        self,
        dim_in: int,
        dims: Tuple[int, int, int] = (384, 512, 640),
        mobilevit_heads: int = 4,
        mobilevit_depth: int = 2,
        sra_heads: int = 6,
        sra_ratio: int = 2,
        mamba_depth: int = 3,
        mamba_kernel: int = 5,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        c1, c2, c3 = dims
        
        # Stage 1: MobileViT + Pyramid Pooling
        self.proj1 = nn.Linear(dim_in, c1)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, mobilevit_depth)]
        self.mobilevit = EnhancedMobileViTBlock(
            c1, 
            heads=mobilevit_heads, 
            depth=mobilevit_depth, 
            dropout=dropout,
            drop_path=dpr[0]
        )
        self.pyramid_pool = PyramidPooling(c1)
        
        # Stage 2: Enhanced PVT with layer scale
        self.proj2 = nn.Linear(c1, c2)
        self.layer_scale2 = nn.Parameter(torch.ones(c2) * 1e-5)
        
        # Use cosine attention instead of regular attention
        self.pvt_attn = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(c2),
                'attn': ScaledCosineAttention(c2, heads=sra_heads, dropout=dropout),
                'ffn': EnhancedFeedForward(c2, mlp_ratio=3.0, dropout=dropout),
                'norm2': nn.LayerNorm(c2),
            })
            for _ in range(2)
        ])
        
        # Stage 3: Global aggregation with Mamba
        self.proj3 = nn.Linear(c2, c3)
        self.layer_scale3 = nn.Parameter(torch.ones(c3) * 1e-5)
        
        # Enhanced Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(c3),
                'dwconv': nn.Conv1d(c3, c3, kernel_size=mamba_kernel, padding=mamba_kernel//2, groups=c3),
                'gate': nn.Linear(c3, c3 * 2),
                'proj': nn.Linear(c3, c3),
                'drop': nn.Dropout(dropout),
            })
            for _ in range(mamba_depth)
        ])
        
        # Final attention aggregation
        self.final_attn = ScaledCosineAttention(c3, heads=min(8, c3 // 64 + 1), dropout=dropout)
        self.final_norm = nn.LayerNorm(c3)
        
    def _tokens_to_map(self, tokens: torch.Tensor, target_hw: Tuple[int, int]):
        B, N, C = tokens.shape
        H, W = target_hw
        need = H * W
        if N < need:
            pad = tokens.new_zeros(B, need - N, C)
            tokens = torch.cat([tokens, pad], dim=1)
        tokens = tokens[:, :need, :]
        return tokens.transpose(1, 2).reshape(B, C, H, W)
    
    def forward(self, tokens: torch.Tensor):
        B, N, C = tokens.shape
        
        # Stage 1: Local + Pyramid
        t1 = self.proj1(tokens)
        map_hw = (3, 4)
        m1 = self._tokens_to_map(t1, map_hw)
        m1 = self.mobilevit(m1)
        m1 = self.pyramid_pool(m1)
        t1_out = m1.flatten(2).transpose(1, 2)[:, :N]
        
        # Stage 2: PVT with layer scale
        t2 = self.proj2(t1_out)
        
        for block in self.pvt_attn:
            h = block['norm'](t2)
            t2 = t2 + self.layer_scale2 * block['attn'](h)
            h = block['norm2'](t2)
            t2 = t2 + self.layer_scale2 * block['ffn'](h)
        
        # Adaptive pooling
        pooled = torch.stack([
            t2.mean(dim=1), 
            t2.max(dim=1).values,
            t2[:, 0] if t2.size(1) > 0 else t2.mean(dim=1)
        ], dim=1)
        
        # Stage 3: Mamba with gating
        t3 = self.proj3(pooled)
        
        for block in self.mamba_blocks:
            shortcut = t3
            x = block['norm'](t3)
            
            # Gating mechanism
            gate_input = block['gate'](x)
            g1, g2 = gate_input.chunk(2, dim=-1)
            g = torch.sigmoid(g1)
            
            # Depthwise conv
            x_conv = (x * g).transpose(1, 2)
            x_conv = block['dwconv'](x_conv).transpose(1, 2)
            
            # Project and residual
            x = block['proj'](x_conv * g2)
            x = block['drop'](x)
            t3 = shortcut + self.layer_scale3 * x
        
        # Final attention
        t3 = self.final_norm(t3)
        t3 = t3 + self.final_attn(t3)
        
        # Global pooling
        global_feat = torch.cat([
            t3.mean(dim=1),
            t3.max(dim=1).values
        ], dim=-1)
        
        return global_feat, {
            "stage1_map": m1.detach(),
            "stage2_tokens": t2.detach(),
            "stage3_tokens": t3.detach()
        }


class BiDirectionalCrossAttention(nn.Module):
    """Bi-directional cross-attention for better feature interaction."""
    
    def __init__(self, dim: int, heads: int = 6, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        # For stream A -> B
        self.q_a = nn.Linear(dim, dim)
        self.kv_b = nn.Linear(dim, dim * 2)
        
        # For stream B -> A
        self.q_b = nn.Linear(dim, dim)
        self.kv_a = nn.Linear(dim, dim * 2)
        
        self.proj_a = nn.Linear(dim, dim)
        self.proj_b = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        """feat_a, feat_b: [B, N, C]"""
        B, N, C = feat_a.shape
        
        # A queries B
        q_a = self.q_a(feat_a).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        kv_b = self.kv_b(feat_b).reshape(B, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_b, v_b = kv_b[0], kv_b[1]
        
        attn_a = (q_a @ k_b.transpose(-2, -1)) * self.scale
        attn_a = attn_a.softmax(dim=-1)
        attn_a = self.dropout(attn_a)
        out_a = (attn_a @ v_b).transpose(1, 2).reshape(B, N, C)
        out_a = self.proj_a(out_a)
        
        # B queries A
        q_b = self.q_b(feat_b).reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)
        kv_a = self.kv_a(feat_a).reshape(B, N, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_a, v_a = kv_a[0], kv_a[1]
        
        attn_b = (q_b @ k_a.transpose(-2, -1)) * self.scale
        attn_b = attn_b.softmax(dim=-1)
        attn_b = self.dropout(attn_b)
        out_b = (attn_b @ v_a).transpose(1, 2).reshape(B, -1, C)
        out_b = self.proj_b(out_b)
        
        return out_a, out_b


class EnhancedCrossScaleFusion(nn.Module):
    """Enhanced cross-scale fusion with bi-directional attention."""
    
    def __init__(
        self, 
        dim: int, 
        heads: int = 6, 
        dropout: float = 0.0, 
        layers: int = 2
    ):
        super().__init__()
        self.layers = layers
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm_s': nn.LayerNorm(dim),
                'norm_b': nn.LayerNorm(dim),
                'self_attn_s': ScaledCosineAttention(dim, heads=heads, dropout=dropout),
                'self_attn_b': ScaledCosineAttention(dim, heads=heads, dropout=dropout),
                'cross_attn': BiDirectionalCrossAttention(dim, heads=heads, dropout=dropout),
                'ffn_s': EnhancedFeedForward(dim, mlp_ratio=3.0, dropout=dropout),
                'ffn_b': EnhancedFeedForward(dim, mlp_ratio=3.0, dropout=dropout),
                'norm_s2': nn.LayerNorm(dim),
                'norm_b2': nn.LayerNorm(dim),
            })
            for _ in range(layers)
        ])
        
        # Layer scale for better training stability
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(dim) * 1e-5) for _ in range(layers * 4)
        ])
        
    def forward(self, tok_s: torch.Tensor, tok_b: torch.Tensor):
        B = tok_s.shape[0]
        C = tok_s.shape[-1]
        
        # Add CLS tokens
        cls_s = tok_s.new_zeros(B, 1, C)
        cls_b = tok_b.new_zeros(B, 1, C)
        tok_s = torch.cat([cls_s, tok_s], dim=1)
        tok_b = torch.cat([cls_b, tok_b], dim=1)
        
        for i, block in enumerate(self.blocks):
            # Self-attention
            tok_s = tok_s + self.layer_scales[i*4] * block['self_attn_s'](block['norm_s'](tok_s))
            tok_b = tok_b + self.layer_scales[i*4] * block['self_attn_b'](block['norm_b'](tok_b))
            
            # Cross-attention
            cross_s, cross_b = block['cross_attn'](tok_s, tok_b)
            tok_s = tok_s + self.layer_scales[i*4+1] * cross_s
            tok_b = tok_b + self.layer_scales[i*4+1] * cross_b
            
            # FFN
            tok_s = tok_s + self.layer_scales[i*4+2] * block['ffn_s'](block['norm_s2'](tok_s))
            tok_b = tok_b + self.layer_scales[i*4+2] * block['ffn_b'](block['norm_b2'](tok_b))
        
        # Combine tokens
        tokens = torch.cat([tok_s, tok_b], dim=1)
        return tokens


# Helper function to create enhanced model
class CrossPVT_T2T_MambaDINO(nn.Module):
    """Full DINO model with CrossPVT, T2T, and Mamba components.
    
    This is a vision-only model that processes left/right image crops separately.
    """
    
    def __init__(self, dropout: float = 0.1, hidden_ratio: float = 0.35):
        super().__init__()
        self.backbone, self.feat_dim, self.backbone_name, self.input_res = self._build_dino_backbone()
        self.tile_encoder = TileEncoder(self.backbone, self.input_res)
        
        small_grid = (4, 4)
        big_grid = (2, 2)
        self.small_grid = small_grid
        self.big_grid = big_grid
        
        self.t2t = T2TRetokenizer(self.feat_dim, depth=2, heads=6, dropout=dropout)
        self.cross = EnhancedCrossScaleFusion(
            self.feat_dim, heads=6, dropout=dropout, layers=2
        )
        self.pyramid = EnhancedPyramidMixer(
            dim_in=self.feat_dim,
            dims=(384, 512, 640),
            mobilevit_heads=4,
            mobilevit_depth=2,
            sra_heads=8,
            sra_ratio=2,
            mamba_depth=3,
            mamba_kernel=5,
            dropout=dropout,
        )

        # Pyramid returns [B, 640*2] for each half (left/right)
        # After concatenation: [B, 640*2*2] = [B, 2560]
        pyramid_out_per_half = 640 * 2  # 1280
        combined = pyramid_out_per_half * 2  # 2560 (left + right)
        self.combined_dim = combined
        hidden = max(32, int(combined * hidden_ratio))

        def head():
            return nn.Sequential(
                nn.Linear(combined, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        self.head_green = head()
        self.head_clover = head()
        self.head_dead = head()
        self.softplus = nn.Softplus(beta=1.0)

        # Cross gates operate on individual pyramid output (before concatenation)
        self.cross_gate_left = nn.Linear(pyramid_out_per_half, pyramid_out_per_half)
        self.cross_gate_right = nn.Linear(pyramid_out_per_half, pyramid_out_per_half)

    def _build_dino_backbone(self):
        """Build DINO backbone from timm."""
        import timm
        candidates = [
            "vit_base_patch14_dinov2",
            "vit_base_patch14_reg4_dinov2",
            "vit_small_patch14_dinov2",
        ]
        
        last_err = None
        for name in candidates:
            for gp in ["token", "avg", "__default__"]:
                try:
                    if gp == "__default__":
                        m = timm.create_model(name, pretrained=True, num_classes=0)
                    else:
                        m = timm.create_model(name, pretrained=True, num_classes=0, global_pool=gp)
                    
                    feat = m.num_features
                    input_res = self._infer_input_res(m)
                    if hasattr(m, "set_grad_checkpointing"):
                        m.set_grad_checkpointing(True)
                    return m, feat, name, int(input_res)
                except Exception as e:
                    last_err = e
                    continue
        raise RuntimeError(f"Cannot create DINO backbone. Last error: {last_err}")

    @staticmethod
    def _infer_input_res(m) -> int:
        """Infer input resolution from model."""
        if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "img_size"):
            isz = m.patch_embed.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        if hasattr(m, "img_size"):
            isz = m.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        dc = getattr(m, "default_cfg", {}) or {}
        ins = dc.get("input_size", None)
        if ins:
            if isinstance(ins, (tuple, list)) and len(ins) >= 2:
                return int(ins[1])
            return int(ins if isinstance(ins, (int, float)) else 224)
        return 518

    def _half_forward(self, x_half: torch.Tensor):
        """Process one half of the image (left or right)."""
        tiles_small = self.tile_encoder(x_half, self.small_grid)
        tiles_big = self.tile_encoder(x_half, self.big_grid)
        t2, stage1_map = self.t2t(tiles_small, self.small_grid)
        fused = self.cross(t2, tiles_big)
        feat, feat_maps = self.pyramid(fused)
        feat_maps["stage1_map"] = stage1_map
        return feat, feat_maps

    def _merge_heads(self, f_l: torch.Tensor, f_r: torch.Tensor):
        """Merge left and right features with gating."""
        g_l = torch.sigmoid(self.cross_gate_left(f_r))
        g_r = torch.sigmoid(self.cross_gate_right(f_l))
        f_l = f_l * g_l
        f_r = f_r * g_r
        f = torch.cat([f_l, f_r], dim=1)
        
        green_pos = self.softplus(self.head_green(f))
        clover_pos = self.softplus(self.head_clover(f))
        dead_pos = self.softplus(self.head_dead(f))
        gdm = green_pos + clover_pos
        total = gdm + dead_pos
        return total, gdm, green_pos, f

    def forward(self, images, tabular=None):
        """Forward pass.
        
        Args:
            images: Image tensor [B, C, H, W]
            tabular: Ignored (for compatibility with other models)
            
        Returns:
            Predictions tensor [B, 3] for (Dry_Green_g, Dry_Dead_g, Dry_Clover_g)
        """
        # Split image into left and right halves
        B, C, H, W = images.shape
        mid = W // 2
        x_left = images[:, :, :, :mid]
        x_right = images[:, :, :, mid:]
        
        feat_l, feats_l = self._half_forward(x_left)
        feat_r, feats_r = self._half_forward(x_right)
        total, gdm, green, f_concat = self._merge_heads(feat_l, feat_r)
        
        # Return base predictions: [Dry_Green_g, Dry_Dead_g, Dry_Clover_g]
        dead = total - gdm
        clover = gdm - green
        
        # Stack predictions
        predictions = torch.cat([green, dead, clover], dim=1)
        return predictions
    
    def predict_all_targets(self, images, tabular=None):
        """Predict all 5 targets including derived ones.
        
        Args:
            images: Image tensor [B, C, H, W]
            tabular: Ignored
            
        Returns:
            Dict with 'all' key containing [B, 5] predictions
        """
        # Get base predictions
        base_preds = self.forward(images, tabular)
        
        # Extract components
        green = base_preds[:, 0:1]
        dead = base_preds[:, 1:2]
        clover = base_preds[:, 2:3]
        
        # Compute derived targets
        gdm = green + clover
        total = gdm + dead
        
        # Stack all 5 targets
        all_preds = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        return {"all": all_preds}


class TileEncoder(nn.Module):
    """Encode image tiles using a backbone."""
    
    def __init__(self, backbone: nn.Module, input_res: int):
        super().__init__()
        self.backbone = backbone
        self.input_res = input_res

    def forward(self, x: torch.Tensor, grid: Tuple[int, int]):
        """Split image into tiles and encode them.
        
        Args:
            x: Image tensor [B, C, H, W]
            grid: (rows, cols) for tiling
            
        Returns:
            Encoded tiles [B, num_tiles, feat_dim]
        """
        B, C, H, W = x.shape
        r, c = grid
        hs = torch.linspace(0, H, steps=r + 1, device=x.device).round().long()
        ws = torch.linspace(0, W, steps=c + 1, device=x.device).round().long()
        tiles = []
        for i in range(r):
            for j in range(c):
                rs, re = hs[i].item(), hs[i + 1].item()
                cs, ce = ws[j].item(), ws[j + 1].item()
                xt = x[:, :, rs:re, cs:ce]
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res), mode="bilinear", align_corners=False)
                tiles.append(xt)
        tiles = torch.stack(tiles, dim=1)
        flat = tiles.view(-1, C, self.input_res, self.input_res)
        feats = self.backbone(flat)
        feats = feats.view(B, -1, feats.shape[-1])
        return feats


class T2TRetokenizer(nn.Module):
    """Tokens-to-Token retokenizer."""
    
    def __init__(self, dim, depth=2, heads=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(dim),
                'attn': ScaledCosineAttention(dim, heads=heads, dropout=dropout),
                'ffn': EnhancedFeedForward(dim, mlp_ratio=2.0, dropout=dropout),
                'norm2': nn.LayerNorm(dim),
            })
            for _ in range(depth)
        ])

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]):
        """Retokenize with attention blocks.
        
        Args:
            tokens: Input tokens [B, T, C]
            grid_hw: Grid dimensions (H, W)
            
        Returns:
            Retokenized output [B, 4, C] and feature map
        """
        B, T, C = tokens.shape
        H, W = grid_hw
        feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
        seq = feat_map.flatten(2).transpose(1, 2)
        
        for blk in self.blocks:
            h = blk['norm'](seq)
            seq = seq + blk['attn'](h)
            seq = seq + blk['ffn'](blk['norm2'](seq))
        
        seq_map = seq.transpose(1, 2).reshape(B, C, H, W)
        pooled = F.adaptive_avg_pool2d(seq_map, (2, 2))
        retokens = pooled.flatten(2).transpose(1, 2)
        return retokens, seq_map


def create_enhanced_dino_model(
    backbone_name: str = "vit_base_patch14_dinov2",
    dropout: float = 0.1,
    hidden_ratio: float = 0.35,
    use_enhanced_pyramid: bool = True,
    use_enhanced_cross_fusion: bool = True,
):
    """Create an enhanced DINO model.
    
    Args:
        backbone_name: Name of the DINO backbone (unused, for compatibility)
        dropout: Dropout rate
        hidden_ratio: Ratio for hidden dimension
        use_enhanced_pyramid: Unused (for compatibility)
        use_enhanced_cross_fusion: Unused (for compatibility)
    
    Returns:
        CrossPVT_T2T_MambaDINO model instance
    """
    return CrossPVT_T2T_MambaDINO(dropout=dropout, hidden_ratio=hidden_ratio)
