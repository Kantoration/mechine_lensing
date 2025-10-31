#!/usr/bin/env python3
"""
ResNet backbone implementations for gravitational lens classification.

This module provides ResNet backbones with support for arbitrary input channel counts
by averaging ImageNet pretrained weights across channels.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights

logger = logging.getLogger(__name__)


class ResNetBackbone(nn.Module):
    """
    ResNet backbone with multi-channel input support.

    This implementation adapts ResNet architectures to support arbitrary
    input channel counts by averaging pretrained ImageNet weights.

    Features:
    - Supports ResNet-18 and ResNet-34
    - Supports arbitrary input channels
    - Preserves pretrained weights when possible
    - Returns feature embeddings before final classification layer
    """

    def __init__(
        self, arch: str = "resnet18", in_ch: int = 3, pretrained: bool = True
    ) -> None:
        """
        Initialize ResNet backbone.

        Args:
            arch: Architecture name ('resnet18' or 'resnet34')
            in_ch: Number of input channels
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        self.arch = arch
        self.in_ch = in_ch
        self.pretrained = pretrained

        # Create base model
        if arch == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif arch == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet34(weights=weights)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported ResNet architecture: {arch}")

        # Adapt first layer for multi-channel inputs
        if self.resnet.conv1.in_channels != in_ch:
            self._adapt_first_layer()

        # Remove the final classification layer - we only want features
        self.resnet.fc = nn.Identity()

        logger.info(
            f"Created ResNet backbone: arch={arch}, in_ch={in_ch}, "
            f"pretrained={pretrained}, feature_dim={self.feature_dim}"
        )

    def _adapt_first_layer(self) -> None:
        """
        Adapt the first convolutional layer for arbitrary input channels.

        For multi-channel inputs, we average the pretrained RGB weights
        across channels to initialize the new layer.
        """
        old_conv = self.resnet.conv1

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
                    f"Adapted ResNet first layer: {old_conv.in_channels} -> {self.in_ch} channels"
                )
            else:
                # Standard initialization for non-pretrained models
                nn.init.kaiming_normal_(
                    new_conv.weight, mode="fan_out", nonlinearity="relu"
                )
                if new_conv.bias is not None:
                    nn.init.constant_(new_conv.bias, 0)

        # Replace the original layer
        self.resnet.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet backbone.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Feature embeddings of shape [B, feature_dim]
        """
        # Ensure input has correct number of channels
        if x.shape[1] != self.in_ch:
            raise ValueError(f"Expected {self.in_ch} input channels, got {x.shape[1]}")

        # Forward through ResNet (returns features before classification)
        features = self.resnet(x)  # Shape: [B, feature_dim]

        return features

    def get_feature_dim(self) -> int:
        """Get the dimension of output features."""
        return self.feature_dim

    def get_model_info(self) -> dict:
        """Get information about the model architecture."""
        return {
            "architecture": self.arch,
            "input_channels": self.in_ch,
            "feature_dim": self.feature_dim,
            "pretrained": self.pretrained,
            "num_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


def create_resnet_backbone(
    arch: str = "resnet18", in_ch: int = 3, pretrained: bool = True
) -> Tuple[ResNetBackbone, int]:
    """
    Factory function to create ResNet backbone.

    Args:
        arch: Architecture name ('resnet18' or 'resnet34')
        in_ch: Number of input channels
        pretrained: Whether to use pretrained weights

    Returns:
        Tuple of (backbone_model, feature_dimension)
    """
    backbone = ResNetBackbone(arch=arch, in_ch=in_ch, pretrained=pretrained)
    return backbone, backbone.get_feature_dim()


def get_resnet_info(arch: str) -> dict:
    """Get ResNet architecture information."""
    resnet_configs = {
        "resnet18": {
            "input_size": 64,  # Recommended for lens classification
            "description": "ResNet-18 Convolutional Neural Network",
            "feature_dim": 512,
        },
        "resnet34": {
            "input_size": 64,  # Recommended for lens classification
            "description": "ResNet-34 Convolutional Neural Network (Deeper)",
            "feature_dim": 512,
        },
    }

    if arch not in resnet_configs:
        raise ValueError(f"Unknown ResNet architecture: {arch}")

    return resnet_configs[arch]
