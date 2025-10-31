#!/usr/bin/env python3
"""
Vision Transformer (ViT) backbone implementation for gravitational lens classification.

This module provides a ViT-B/16 backbone with support for arbitrary input channel counts
by averaging ImageNet pretrained weights across channels.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

logger = logging.getLogger(__name__)


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone with multi-channel input support.

    This implementation uses ViT-B/16 as the base architecture and adapts
    the first convolutional layer to support arbitrary input channel counts
    by averaging pretrained ImageNet weights.

    Features:
    - Supports arbitrary input channels (e.g., 3 for RGB, 5 for multi-band)
    - Preserves pretrained weights when possible
    - Returns feature embeddings before final classification layer
    """

    def __init__(self, in_ch: int = 3, pretrained: bool = True) -> None:
        """
        Initialize ViT backbone.

        Args:
            in_ch: Number of input channels
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        self.in_ch = in_ch
        self.pretrained = pretrained

        # Load pretrained ViT-B/16
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # Adapt first layer for multi-channel inputs
        if self.vit.conv_proj.in_channels != in_ch:
            self._adapt_first_layer()

        # Remove the final classification head - we only want features
        self.feature_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()  # Remove classification head

        logger.info(
            f"Created ViT backbone: in_ch={in_ch}, pretrained={pretrained}, "
            f"feature_dim={self.feature_dim}"
        )

    def _adapt_first_layer(self) -> None:
        """
        Adapt the first convolutional layer for arbitrary input channels.

        For multi-channel inputs, we average the pretrained RGB weights
        across channels to initialize the new layer. This preserves
        pretrained knowledge while supporting new input modalities.
        """
        old_conv = self.vit.conv_proj

        # Create new convolutional layer with desired input channels
        new_conv = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights by averaging across input channels
        with torch.no_grad():
            if self.pretrained and old_conv.weight is not None:
                # Average RGB weights across channels and replicate
                old_weights = old_conv.weight.data  # Shape: [out_ch, 3, H, W]
                avg_weights = old_weights.mean(dim=1, keepdim=True)  # [out_ch, 1, H, W]
                new_weights = avg_weights.repeat(
                    1, self.in_ch, 1, 1
                )  # [out_ch, in_ch, H, W]
                new_conv.weight.copy_(new_weights)

                # Copy bias if it exists
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

                logger.info(
                    f"Adapted ViT first layer: {old_conv.in_channels} -> {self.in_ch} channels"
                )
            else:
                # Standard initialization for non-pretrained models
                nn.init.kaiming_normal_(
                    new_conv.weight, mode="fan_out", nonlinearity="relu"
                )
                if new_conv.bias is not None:
                    nn.init.constant_(new_conv.bias, 0)

        # Replace the original layer
        self.vit.conv_proj = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT backbone.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Feature embeddings of shape [B, feature_dim]
        """
        # Ensure input has correct number of channels
        if x.shape[1] != self.in_ch:
            raise ValueError(f"Expected {self.in_ch} input channels, got {x.shape[1]}")

        # Forward through ViT (returns CLS token embedding)
        features = self.vit(x)  # Shape: [B, feature_dim]

        return features

    def get_feature_dim(self) -> int:
        """Get the dimension of output features."""
        return self.feature_dim

    def get_model_info(self) -> dict:
        """Get information about the model architecture."""
        return {
            "architecture": "ViT-B/16",
            "input_channels": self.in_ch,
            "feature_dim": self.feature_dim,
            "pretrained": self.pretrained,
            "num_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


def create_vit_backbone(
    in_ch: int = 3, pretrained: bool = True
) -> Tuple[ViTBackbone, int]:
    """
    Factory function to create ViT backbone.

    Args:
        in_ch: Number of input channels
        pretrained: Whether to use pretrained weights

    Returns:
        Tuple of (backbone_model, feature_dimension)
    """
    backbone = ViTBackbone(in_ch=in_ch, pretrained=pretrained)
    return backbone, backbone.get_feature_dim()


def get_vit_info() -> dict:
    """Get ViT architecture information."""
    return {
        "input_size": 224,  # Standard ViT input size
        "patch_size": 16,
        "description": "Vision Transformer Base with 16x16 patches",
        "default_feature_dim": 768,
    }
