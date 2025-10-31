#!/usr/bin/env python3
"""
Enhanced Light Transformer backbone for gravitational lens classification.

This module implements a robust hybrid CNN-Transformer architecture with:
- Dynamic positional embeddings with bicubic interpolation
- Configurable CNN stage and patch size for token control
- Advanced regularization (DropPath, projection dropout, attention dropout)
- Norm-preserving multi-channel weight initialization
- Flexible pooling strategies (avg/attention/CLS)
- Progressive layer freezing schedules
- Production-ready robustness across input sizes and channel counts

The architecture combines CNN inductive biases with transformer expressiveness
while maintaining computational efficiency for astronomical image analysis.
"""

from __future__ import annotations

import logging
from typing import Tuple, Literal, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) regularization.

    Randomly drops entire residual branches during training to improve
    regularization and reduce overfitting in deep networks.

    References:
        - Huang et al. (2016). Deep Networks with Stochastic Depth
        - Larsson et al. (2016). FractalNet: Ultra-Deep Neural Networks without Residuals
    """

    def __init__(self, p: float = 0.0) -> None:
        """
        Initialize DropPath module.

        Args:
            p: Drop probability. 0.0 means no dropping.
        """
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth to input tensor.

        Args:
            x: Input tensor of shape [B, ...]

        Returns:
            Output tensor with same shape as input
        """
        if self.p == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.p
        # Create random tensor with same batch dimension, broadcast to other dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        # Scale by keep_prob to maintain expected value
        return x * random_tensor / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"


class PatchEmbedding(nn.Module):
    """
    Convert CNN feature maps to patch embeddings for transformer processing.

    This module takes CNN features and converts them to a sequence of patch
    embeddings that can be processed by transformer blocks.
    """

    def __init__(self, feature_dim: int, patch_size: int, embed_dim: int):
        """
        Initialize patch embedding layer.

        Args:
            feature_dim: Input feature dimension from CNN
            patch_size: Size of patches to extract
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Project CNN features to embedding dimension
        self.projection = nn.Conv2d(
            feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Convert feature maps to patch embeddings.

        Args:
            x: CNN features [B, feature_dim, H, W]

        Returns:
            Tuple of (patch_embeddings [B, N, embed_dim], H_patches, W_patches)
        """
        # Project to embeddings: [B, embed_dim, H//patch_size, W//patch_size]
        x = self.projection(x)

        # Get patch grid dimensions
        B, C, Hp, Wp = x.shape

        # Flatten spatial dimensions: [B, embed_dim, N] -> [B, N, embed_dim]
        x = x.view(B, C, Hp * Wp).transpose(1, 2)

        # Apply layer norm
        x = self.norm(x)

        return x, Hp, Wp


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with enhanced dropout and regularization.

    Includes attention dropout, projection dropout, and proper initialization
    for astronomical feature processing.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
    ):
        """
        Initialize multi-head self-attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            attn_drop: Attention dropout probability
            proj_drop: Projection dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize with astronomical data in mind
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for astronomical feature patterns."""
        # Use smaller initialization for stability with astronomical data
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.8)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.8)
        nn.init.constant_(self.qkv.bias, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-head self-attention forward pass.

        Args:
            x: Input embeddings [B, N, embed_dim]

        Returns:
            Attended features [B, N, embed_dim]
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with advanced regularization.

    Includes layer normalization, multi-head attention, MLP, and
    stochastic depth (DropPath) for improved training stability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
        drop_path1: float = 0.0,
        drop_path2: float = 0.0,
    ):
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            attn_drop: Attention dropout
            proj_drop: Projection dropout
            drop_path1: DropPath probability for attention branch
            drop_path2: DropPath probability for MLP branch
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop, proj_drop)
        self.drop_path1 = DropPath(drop_path1)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),  # GELU works better than ReLU for transformers
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(proj_drop),
        )
        self.drop_path2 = DropPath(drop_path2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer block forward pass with residual connections and DropPath."""
        # Self-attention with residual connection and DropPath
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # MLP with residual connection and DropPath
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class LightTransformerBackbone(nn.Module):
    """
    Enhanced light transformer backbone with production-ready features.

    This architecture combines CNN feature extraction with transformer processing,
    featuring dynamic positional embeddings, configurable token counts, advanced
    regularization, and flexible pooling strategies.

    Key improvements:
    - Dynamic positional embeddings with bicubic interpolation
    - Configurable CNN stage and patch size for token control
    - DropPath regularization and enhanced dropout
    - Norm-preserving multi-channel initialization
    - Multiple pooling strategies (avg/attention/CLS)
    - Progressive layer freezing

    Architecture inspired by:
    - DeiT (Data-efficient Image Transformers)
    - Hybrid CNN-Transformer architectures
    - Bologna Lens Challenge winning approaches
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
    ):
        """
        Initialize enhanced light transformer backbone.

        Args:
            in_ch: Number of input channels
            pretrained: Whether to use pretrained CNN weights
            cnn_stage: CNN stage to extract features from ("layer2" or "layer3")
            patch_size: Patch size for tokenization
            embed_dim: Transformer embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: MLP hidden dimension ratio
            attn_drop: Attention dropout probability
            proj_drop: Projection dropout probability
            pos_drop: Positional embedding dropout probability
            drop_path_max: Maximum DropPath probability (linearly scheduled)
            pooling: Pooling strategy ("avg", "attn", or "cls")
            freeze_until: CNN layers to freeze ("none", "layer2", or "layer3")
            max_tokens: Maximum number of tokens allowed (for memory management)
        """
        super().__init__()

        self.in_ch = in_ch
        self.pretrained = pretrained
        self.cnn_stage = cnn_stage
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pooling = pooling

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
        # Initialize for reasonable default, will interpolate as needed
        self.max_tokens = max_tokens
        initial_patches = min(64, max_tokens)  # Conservative initialization
        self.pos_embed = nn.Parameter(torch.zeros(1, initial_patches, embed_dim))
        self.pos_drop = nn.Dropout(pos_drop)

        # CLS token for CLS pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Attention pooling query
        if pooling == "attn":
            self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder layers with progressive DropPath
        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_max, num_layers)
        ]
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path1=drop_path_rates[i],
                    drop_path2=drop_path_rates[i],
                )
                for i in range(num_layers)
            ]
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Feature dimension for downstream heads
        self.feature_dim = embed_dim

        # Initialize positional embeddings and other parameters
        self._init_weights()

        logger.info(
            f"Created Enhanced Light Transformer: in_ch={in_ch}, "
            f"cnn_stage={cnn_stage}, patch_size={patch_size}, "
            f"embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}, "
            f"pooling={pooling}, feature_dim={self.feature_dim}"
        )

    def _adapt_first_layer(self, resnet: nn.Module) -> None:
        """Adapt ResNet first layer for multi-channel inputs with norm-preserving scaling."""
        old_conv = resnet.conv1

        new_conv = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            if self.pretrained and old_conv.weight is not None:
                # Norm-preserving multi-channel initialization
                avg_weights = old_conv.weight.data.mean(
                    dim=1, keepdim=True
                )  # [out, 1, H, W]
                scale = 3.0 / float(self.in_ch)  # Preserve activation magnitude
                new_weights = avg_weights.repeat(1, self.in_ch, 1, 1) * scale
                new_conv.weight.copy_(new_weights)

                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

                logger.debug(
                    f"Adapted CNN first layer: {old_conv.in_channels} -> {self.in_ch} "
                    f"channels with scale={scale:.3f}"
                )

        resnet.conv1 = new_conv

    def _apply_freezing(self, freeze_until: str) -> None:
        """Apply progressive freezing schedule to CNN layers."""
        if freeze_until == "none":
            return

        # Get the layers to freeze
        if freeze_until == "layer2":
            # Freeze conv1, bn1, maxpool, layer1, layer2
            freeze_modules = self.cnn_features[
                :5
            ]  # conv1, bn1, relu, maxpool, layer1, layer2
        elif freeze_until == "layer3":
            # Freeze conv1, bn1, maxpool, layer1, layer2, layer3
            freeze_modules = self.cnn_features[:6]  # Everything up to layer3
        else:
            raise ValueError(f"Invalid freeze_until: {freeze_until}")

        # Freeze parameters but keep LayerNorm trainable
        frozen_params = 0
        for module in freeze_modules:
            for param in module.parameters():
                param.requires_grad_(False)
                frozen_params += param.numel()

        logger.info(f"Froze {frozen_params:,} parameters up to {freeze_until}")

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize other parameters
        def _init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_fn)

    def _interpolate_pos_embed(
        self, pos: torch.Tensor, N: int, H: int, W: int
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings to match current patch grid size.

        Args:
            pos: Positional embeddings [1, Pmax, C]
            N: Number of current patches (H * W)
            H: Height of patch grid
            W: Width of patch grid

        Returns:
            Interpolated positional embeddings [1, N, C]
        """
        C = pos.shape[-1]
        Pmax = pos.shape[1]
        side = int(Pmax**0.5)

        # Reshape to 2D grid and interpolate
        pos2d = (
            pos[:, : side * side, :].reshape(1, side, side, C).permute(0, 3, 1, 2)
        )  # [1,C,S,S]
        pos2d = F.interpolate(
            pos2d, size=(H, W), mode="bicubic", align_corners=False
        )  # [1,C,H,W]
        posN = pos2d.permute(0, 2, 3, 1).reshape(1, H * W, C)  # [1, H*W, C]

        return posN[:, :N, :]  # [1, N, C]

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool transformer features using the specified pooling strategy.

        Args:
            x: Transformer features [B, N, C] (or [B, N+1, C] for CLS)

        Returns:
            Pooled features [B, C]
        """
        B, N, C = x.shape

        if self.pooling == "avg":
            # Average pooling over all tokens
            pooled = x.mean(dim=1)  # [B, C]

        elif self.pooling == "attn":
            # Attention-based pooling
            q = self.pool_query.expand(B, -1, -1)  # [B, 1, C]
            attn = torch.softmax(
                (q @ x.transpose(1, 2)) / (C**0.5), dim=-1
            )  # [B, 1, N]
            pooled = (attn @ x).squeeze(1)  # [B, C]

        elif self.pooling == "cls":
            # Use CLS token (first token)
            pooled = x[:, 0]  # [B, C]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced light transformer backbone.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Global features [B, feature_dim]
        """
        B, C, H, W = x.shape

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Input shape: {x.shape}")

        # CNN feature extraction: [B, C, H, W] -> [B, feature_dim, H', W']
        cnn_features = self.cnn_features(x)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"CNN features shape: {cnn_features.shape}")

        # Convert to patch embeddings: [B, feature_dim, H', W'] -> [B, N, embed_dim]
        patch_embeddings, Hp, Wp = self.patch_embed(cnn_features)
        B, N, _ = patch_embeddings.shape

        # Adaptive token management for memory efficiency
        if N > self.max_tokens:
            # Calculate optimal patch size for current input
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
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, embed_dim]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Pool features using specified strategy
        pooled = self._pool_features(x)

        # Final normalization
        x = self.norm(pooled)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Output features shape: {x.shape}, pooling: {self.pooling}")

        return x

    def get_feature_dim(self) -> int:
        """Get the dimension of output features."""
        return self.feature_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        return {
            "architecture": "Enhanced Light Transformer (CNN + Self-Attention)",
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


def create_light_transformer_backbone(
    in_ch: int = 3, pretrained: bool = True, **kwargs
) -> Tuple[LightTransformerBackbone, int]:
    """
    Factory function to create enhanced light transformer backbone.

    Args:
        in_ch: Number of input channels
        pretrained: Whether to use pretrained CNN weights
        **kwargs: Additional arguments for backbone configuration

    Returns:
        Tuple of (backbone_model, feature_dimension)
    """
    backbone = LightTransformerBackbone(in_ch=in_ch, pretrained=pretrained, **kwargs)
    return backbone, backbone.get_feature_dim()


def get_light_transformer_info() -> Dict[str, Any]:
    """Get enhanced light transformer architecture information."""
    return {
        "input_size": 112,  # Recommended input size
        "description": "Enhanced Light Transformer: CNN features + Self-Attention with advanced regularization",
        "default_feature_dim": 256,
        "parameter_count": "~2-4M parameters (configurable)",
        "strengths": [
            "Dynamic positional embeddings for flexible input sizes",
            "Configurable token count and CNN stage",
            "Advanced regularization (DropPath, enhanced dropout)",
            "Multiple pooling strategies",
            "Norm-preserving multi-channel initialization",
            "Progressive layer freezing support",
        ],
        "recommended_configs": {
            "fast": {
                "cnn_stage": "layer2",
                "patch_size": 2,
                "embed_dim": 128,
                "num_layers": 3,
            },
            "balanced": {
                "cnn_stage": "layer3",
                "patch_size": 2,
                "embed_dim": 256,
                "num_layers": 4,
            },
            "quality": {
                "cnn_stage": "layer3",
                "patch_size": 1,
                "embed_dim": 384,
                "num_layers": 6,
            },
        },
    }
