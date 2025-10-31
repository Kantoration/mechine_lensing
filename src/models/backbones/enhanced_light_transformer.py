#!/usr/bin/env python3
"""
enhanced_light_transformer.py
============================
Enhanced Light Transformer with advanced attention mechanisms for lensing.

Key Features:
- Integration with lensing-specific attention mechanisms
- Multi-scale processing within the transformer
- Physics-informed attention priors
- Adaptive attention based on image characteristics
- Enhanced regularization and training stability

Usage:
    from models.backbones.enhanced_light_transformer import EnhancedLightTransformerBackbone
"""

from __future__ import annotations

import logging
import math
from typing import Tuple, Optional, Literal, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from .light_transformer import DropPath, PatchEmbedding, MultiHeadSelfAttention
from ..attention.lensing_attention import (
    ArcAwareAttention,
    MultiScaleAttention,
    AdaptiveAttention,
)

logger = logging.getLogger(__name__)


class EnhancedLightTransformerBackbone(nn.Module):
    """
    Enhanced Light Transformer with advanced attention mechanisms for gravitational lensing.

    This backbone combines the efficiency of CNN feature extraction with the expressiveness
    of transformer attention, specifically enhanced for lensing feature detection.

    Key enhancements over the base Light Transformer:
    - Arc-aware attention for lensing arc detection
    - Multi-scale attention for different arc sizes
    - Adaptive attention based on image characteristics
    - Physics-informed attention priors
    - Enhanced regularization and training stability
    """

    def __init__(
        self,
        in_ch: int = 3,
        pretrained: bool = True,
        cnn_stage: Literal["layer2", "layer3"] = "layer3",
        patch_size: int = 2,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 2.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
        pos_drop: float = 0.1,
        drop_path_max: float = 0.1,
        pooling: Literal["avg", "attn", "cls"] = "avg",
        freeze_until: Literal["none", "layer2", "layer3"] = "none",
        max_tokens: int = 256,
        attention_type: Literal[
            "standard", "arc_aware", "multi_scale", "adaptive"
        ] = "adaptive",
        attention_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize enhanced light transformer backbone.

        Args:
            in_ch: Number of input channels
            pretrained: Whether to use pretrained CNN weights
            cnn_stage: CNN stage to use for feature extraction
            patch_size: Size of patches for transformer
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: MLP hidden dimension ratio
            attn_drop: Attention dropout probability
            proj_drop: Projection dropout probability
            pos_drop: Positional embedding dropout probability
            drop_path_max: Maximum DropPath probability
            pooling: Pooling strategy
            freeze_until: Freezing schedule
            max_tokens: Maximum number of tokens
            attention_type: Type of attention mechanism
            attention_config: Configuration for attention mechanism
        """
        super().__init__()

        self.in_ch = in_ch
        self.pretrained = pretrained
        self.cnn_stage = cnn_stage
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pooling = pooling
        self.max_tokens = max_tokens
        self.attention_type = attention_type

        # Default attention configuration
        if attention_config is None:
            attention_config = {}

        # CNN feature extractor (ResNet-18 backbone)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        # Adapt first layer for multi-channel inputs
        if resnet.conv1.in_channels != in_ch:
            self._adapt_first_layer(resnet)

        # Build CNN features up to specified stage
        if cnn_stage == "layer2":
            self.cnn_features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
            )
            cnn_feature_dim = 128  # ResNet-18 layer2 output channels
        elif cnn_stage == "layer3":
            self.cnn_features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
            )
            cnn_feature_dim = 256  # ResNet-18 layer3 output channels
        else:
            raise ValueError(f"Unsupported cnn_stage: {cnn_stage}")

        # Apply freezing schedule
        self._apply_freezing(freeze_until)

        # Patch embedding from CNN features
        self.patch_embed = PatchEmbedding(cnn_feature_dim, patch_size, embed_dim)

        # Dynamic positional embeddings with adaptive sizing
        initial_patches = min(64, max_tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, initial_patches, embed_dim))
        self.pos_drop = nn.Dropout(pos_drop)

        # CLS token for CLS pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Attention pooling query
        if pooling == "attn":
            self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Enhanced transformer blocks with specialized attention
        self.transformer_blocks = self._create_enhanced_blocks(
            embed_dim,
            num_heads,
            num_layers,
            mlp_ratio,
            attn_drop,
            proj_drop,
            drop_path_max,
            attention_type,
            attention_config,
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Feature dimension for head
        self.feature_dim = embed_dim

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Enhanced Light Transformer: {attention_type} attention, {num_layers} layers, {embed_dim}D"
        )

    def _create_enhanced_blocks(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        attn_drop: float,
        proj_drop: float,
        drop_path_max: float,
        attention_type: str,
        attention_config: Dict[str, Any],
    ) -> nn.ModuleList:
        """Create enhanced transformer blocks with specialized attention."""
        blocks = nn.ModuleList()
        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_max, num_layers)
        ]

        for i in range(num_layers):
            # Create attention mechanism based on type
            if attention_type == "standard":
                attention = MultiHeadSelfAttention(
                    embed_dim, num_heads, attn_drop, proj_drop
                )
            elif attention_type == "arc_aware":
                attention = ArcAwareAttention(
                    embed_dim,
                    num_heads,
                    arc_prior_strength=attention_config.get("arc_prior_strength", 0.1),
                    curvature_sensitivity=attention_config.get(
                        "curvature_sensitivity", 1.0
                    ),
                )
            elif attention_type == "multi_scale":
                attention = MultiScaleAttention(
                    embed_dim,
                    num_heads,
                    scales=attention_config.get("scales", [1, 2, 4]),
                    fusion_method=attention_config.get("fusion_method", "weighted_sum"),
                )
            elif attention_type == "adaptive":
                attention = AdaptiveAttention(
                    embed_dim,
                    num_heads,
                    adaptation_layers=attention_config.get("adaptation_layers", 2),
                )
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")

            # Create enhanced transformer block
            block = EnhancedTransformerBlock(
                embed_dim=embed_dim,
                attention=attention,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop,
                drop_path1=drop_path_rates[i],
                drop_path2=drop_path_rates[i],
                attention_type=attention_type,
            )
            blocks.append(block)

        return blocks

    def _adapt_first_layer(self, resnet: nn.Module) -> None:
        """Adapt first layer for multi-channel inputs with norm-preserving initialization."""
        original_conv = resnet.conv1
        new_conv = nn.Conv2d(
            self.in_ch,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        if self.in_ch == 3:
            # Direct copy for RGB
            new_conv.weight.data = original_conv.weight.data
        else:
            # Norm-preserving initialization for multi-channel
            with torch.no_grad():
                # Average RGB weights and scale by 3/in_ch
                rgb_weights = original_conv.weight.data  # [out_ch, 3, H, W]
                avg_weights = rgb_weights.mean(dim=1, keepdim=True)  # [out_ch, 1, H, W]
                scale_factor = 3.0 / self.in_ch
                new_conv.weight.data = (
                    avg_weights.expand(-1, self.in_ch, -1, -1) * scale_factor
                )

        if original_conv.bias is not None:
            new_conv.bias.data = original_conv.bias.data

        resnet.conv1 = new_conv

    def _apply_freezing(self, freeze_until: str) -> None:
        """Apply progressive freezing schedule."""
        if freeze_until == "none":
            return

        # Freeze early layers
        for name, param in self.cnn_features.named_parameters():
            if freeze_until == "layer2" and "layer2" in name:
                break
            elif freeze_until == "layer3" and "layer3" in name:
                break
            param.requires_grad = False

        logger.info(f"Frozen CNN layers up to {freeze_until}")

    def _init_weights(self) -> None:
        """Initialize weights with astronomical data considerations."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize transformer blocks
        for block in self.transformer_blocks:
            if hasattr(block, "norm1"):
                nn.init.constant_(block.norm1.weight, 1.0)
                nn.init.constant_(block.norm1.bias, 0.0)
            if hasattr(block, "norm2"):
                nn.init.constant_(block.norm2.weight, 1.0)
                nn.init.constant_(block.norm2.bias, 0.0)

    def _interpolate_pos_embed(
        self, pos_embed: torch.Tensor, N: int, H: int, W: int
    ) -> torch.Tensor:
        """Interpolate positional embeddings for different input sizes."""
        if N == pos_embed.shape[1]:
            return pos_embed

        # Reshape to 2D grid
        old_N = pos_embed.shape[1]
        old_H = old_W = int(math.sqrt(old_N))

        if old_H * old_W != old_N:
            # Handle non-square case
            pos_embed = pos_embed[:, : old_H * old_H]  # Truncate to square
            old_N = old_H * old_H

        pos_embed_2d = pos_embed.reshape(1, old_H, old_W, -1).permute(0, 3, 1, 2)

        # Interpolate to new size
        pos_embed_2d = F.interpolate(
            pos_embed_2d, size=(H, W), mode="bicubic", align_corners=False
        )

        # Reshape back to sequence
        pos_embed = pos_embed_2d.permute(0, 2, 3, 1).reshape(1, H * W, -1)

        return pos_embed

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pool features using specified strategy."""
        if self.pooling == "avg":
            # Global average pooling
            pooled = x.mean(dim=1)
        elif self.pooling == "attn":
            # Attention pooling
            B, N, C = x.shape
            query = self.pool_query.expand(B, -1, -1)
            attn_weights = torch.softmax(torch.bmm(query, x.transpose(1, 2)), dim=-1)
            pooled = torch.bmm(attn_weights, x).squeeze(1)
        elif self.pooling == "cls":
            # CLS token pooling
            pooled = x[:, 0]  # First token is CLS token
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return pooled

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through enhanced light transformer backbone.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Tuple of (global_features, attention_info)
        """
        B, C, H, W = x.shape

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Input shape: {x.shape}")

        # CNN feature extraction
        cnn_features = self.cnn_features(x)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"CNN features shape: {cnn_features.shape}")

        # Convert to patch embeddings
        patch_embeddings, Hp, Wp = self.patch_embed(cnn_features)
        B, N, _ = patch_embeddings.shape

        # Adaptive token management
        if N > self.max_tokens:
            optimal_patch_size = int(np.sqrt(N / self.max_tokens)) + 1
            suggested_cnn_stage = "layer3" if self.cnn_stage == "layer2" else "layer3"

            error_msg = (
                f"Token count {N} exceeds maximum {self.max_tokens}. "
                f"Current config: patch_size={self.patch_size}, cnn_stage='{self.cnn_stage}'. "
                f"Suggested fixes: "
                f"1) Increase patch_size to {optimal_patch_size} "
                f"2) Use deeper cnn_stage='{suggested_cnn_stage}' "
                f"3) Increase max_tokens to {N} "
                f"4) Reduce input image size"
            )

            logger.error(error_msg)
            raise ValueError(error_msg)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Patch embeddings shape: {patch_embeddings.shape}, grid: {Hp}x{Wp}"
            )

        # Add interpolated positional embeddings
        pos_embed = self._interpolate_pos_embed(self.pos_embed, N, Hp, Wp)
        x = self.pos_drop(patch_embeddings + pos_embed)

        # Add CLS token if using CLS pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Apply enhanced transformer blocks
        attention_info = {}
        for i, block in enumerate(self.transformer_blocks):
            x, block_info = block(x, Hp, Wp)
            attention_info[f"block_{i}"] = block_info

        # Pool features using specified strategy
        pooled = self._pool_features(x)

        # Final normalization
        x = self.norm(pooled)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Output features shape: {x.shape}, pooling: {self.pooling}")

        return x, attention_info

    def get_feature_dim(self) -> int:
        """Get the dimension of output features."""
        return self.feature_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        return {
            "architecture": "Enhanced Light Transformer with Lensing Attention",
            "attention_type": self.attention_type,
            "input_channels": self.in_ch,
            "cnn_stage": self.cnn_stage,
            "patch_size": self.patch_size,
            "feature_dim": self.feature_dim,
            "embed_dim": self.embed_dim,
            "num_layers": len(self.transformer_blocks),
            "pooling": self.pooling,
            "pretrained": self.pretrained,
            "num_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


class EnhancedTransformerBlock(nn.Module):
    """
    Enhanced transformer block with specialized attention mechanisms.
    """

    def __init__(
        self,
        embed_dim: int,
        attention: nn.Module,
        mlp_ratio: float = 2.0,
        proj_drop: float = 0.1,
        drop_path1: float = 0.0,
        drop_path2: float = 0.0,
        attention_type: str = "standard",
    ):
        """
        Initialize enhanced transformer block.

        Args:
            embed_dim: Embedding dimension
            attention: Attention mechanism
            mlp_ratio: MLP hidden dimension ratio
            proj_drop: Projection dropout probability
            drop_path1: DropPath probability for attention branch
            drop_path2: DropPath probability for MLP branch
            attention_type: Type of attention mechanism
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.attention = attention
        self.attention_type = attention_type

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # DropPath for regularization
        self.drop_path1 = DropPath(drop_path1)
        self.drop_path2 = DropPath(drop_path2)

        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(proj_drop),
        )

    def forward(
        self, x: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced transformer block forward pass.

        Args:
            x: Input embeddings [B, N, embed_dim]
            H: Height of spatial grid (for specialized attention)
            W: Width of spatial grid (for specialized attention)

        Returns:
            Tuple of (output_features, attention_info)
        """
        # Self-attention with residual connection and DropPath
        if self.attention_type in ["arc_aware", "multi_scale", "adaptive"]:
            # Specialized attention mechanisms
            attn_out = self.attention(self.norm1(x), H, W)
            if isinstance(attn_out, tuple):
                x_attn, attention_info = attn_out
            else:
                x_attn = attn_out
                attention_info = {}
        else:
            # Standard attention
            x_attn = self.attention(self.norm1(x))
            attention_info = {}

        x = x + self.drop_path1(x_attn)

        # MLP with residual connection and DropPath
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x, attention_info


def create_enhanced_light_transformer_backbone(
    in_ch: int = 3, pretrained: bool = True, attention_type: str = "adaptive", **kwargs
) -> Tuple[EnhancedLightTransformerBackbone, int]:
    """
    Factory function to create enhanced light transformer backbone.

    Args:
        in_ch: Number of input channels
        pretrained: Whether to use pretrained weights
        attention_type: Type of attention mechanism
        **kwargs: Additional arguments

    Returns:
        Tuple of (backbone, feature_dim)
    """
    backbone = EnhancedLightTransformerBackbone(
        in_ch=in_ch, pretrained=pretrained, attention_type=attention_type, **kwargs
    )

    return backbone, backbone.get_feature_dim()


def get_enhanced_light_transformer_info() -> Dict[str, Any]:
    """Get enhanced light transformer architecture information."""
    return {
        "input_size": 112,  # Recommended input size
        "description": "Enhanced Light Transformer with Lensing-Specific Attention",
        "default_feature_dim": 256,
        "parameter_count": "~3-6M parameters (configurable)",
        "attention_types": ["standard", "arc_aware", "multi_scale", "adaptive"],
        "strengths": [
            "Physics-informed attention for lensing arc detection",
            "Multi-scale attention for different arc sizes",
            "Adaptive attention based on image characteristics",
            "Enhanced regularization and training stability",
            "Interpretable attention maps for analysis",
            "Dynamic positional embeddings for flexible input sizes",
        ],
        "recommended_configs": {
            "fast": {
                "cnn_stage": "layer2",
                "patch_size": 2,
                "embed_dim": 128,
                "num_layers": 3,
                "attention_type": "standard",
            },
            "balanced": {
                "cnn_stage": "layer3",
                "patch_size": 2,
                "embed_dim": 256,
                "num_layers": 4,
                "attention_type": "arc_aware",
            },
            "quality": {
                "cnn_stage": "layer3",
                "patch_size": 1,
                "embed_dim": 384,
                "num_layers": 6,
                "attention_type": "adaptive",
            },
        },
    }
