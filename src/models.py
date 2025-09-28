#!/usr/bin/env python3
"""
models.py
=========
Central model factory for gravitational lens classification.

Supports both CNN (ResNet-18) and Vision Transformer (ViT-B/16) architectures
with unified interface for training and evaluation.

Key Features:
- Unified model creation interface
- Pretrained weight loading
- Automatic final layer adaptation for binary classification
- Parameter counting and model summaries
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

logger = logging.getLogger(__name__)

# Supported architectures
SUPPORTED_ARCHITECTURES = {
    'resnet18': {
        'model_fn': models.resnet18,
        'weights': ResNet18_Weights.DEFAULT,
        'input_size': 64,  # Recommended input size
        'description': 'ResNet-18 Convolutional Neural Network'
    },
    'resnet34': {
        'model_fn': models.resnet34,
        'weights': models.ResNet34_Weights.DEFAULT,
        'input_size': 64,  # Same as ResNet-18 but deeper
        'description': 'ResNet-34 Convolutional Neural Network (Deeper)'
    },
    'vit_b_16': {
        'model_fn': models.vit_b_16,
        'weights': ViT_B_16_Weights.DEFAULT,
        'input_size': 224,  # Required input size for ViT
        'description': 'Vision Transformer Base with 16x16 patches'
    }
}


class LensClassifier(nn.Module):
    """
    Unified wrapper for different architectures with binary classification head.
    
    This wrapper provides a consistent interface regardless of the underlying
    architecture (ResNet, ViT, etc.), making it easy to switch between models.
    """
    
    def __init__(
        self, 
        arch: str, 
        num_classes: int = 1,  # Binary classification (sigmoid output)
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        """
        Initialize lens classifier with specified architecture.
        
        Args:
            arch: Architecture name ('resnet18' or 'vit_b_16')
            num_classes: Number of output classes (1 for binary classification)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture '{arch}' not supported. "
                f"Choose from: {list(SUPPORTED_ARCHITECTURES.keys())}"
            )
        
        self.arch = arch
        self.num_classes = num_classes
        
        # Get model configuration
        model_config = SUPPORTED_ARCHITECTURES[arch]
        model_fn = model_config['model_fn']
        weights = model_config['weights'] if pretrained else None
        
        # Create backbone model
        self.backbone = model_fn(weights=weights)
        
        # Adapt final layer for binary classification
        self._adapt_classifier_head(dropout_rate)
        
        # Log model creation
        param_count = self._count_parameters()
        logger.info(f"Created {model_config['description']} with {param_count:,} parameters")
        if pretrained:
            logger.info(f"Loaded pretrained weights: {weights}")
    
    def _adapt_classifier_head(self, dropout_rate: float) -> None:
        """Adapt the final classification layer for binary classification."""
        if self.arch in ['resnet18', 'resnet34']:
            # ResNet: Replace fc layer
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, self.num_classes)
            )
            
        elif self.arch == 'vit_b_16':
            # ViT: Replace heads.head layer
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, self.num_classes)
            )
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.backbone(x)
    
    def get_input_size(self) -> int:
        """Get recommended input size for this architecture."""
        return SUPPORTED_ARCHITECTURES[self.arch]['input_size']


def build_model(
    arch: str, 
    num_classes: int = 1, 
    pretrained: bool = True,
    dropout_rate: float = 0.5
) -> LensClassifier:
    """
    Factory function to build lens classification models.
    
    Args:
        arch: Architecture name ('resnet18' or 'vit_b_16')
        num_classes: Number of output classes (1 for binary classification)
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for regularization
        
    Returns:
        LensClassifier instance
        
    Raises:
        ValueError: If architecture is not supported
        
    Example:
        >>> # Create ResNet-18 model
        >>> model = build_model('resnet18', pretrained=True)
        >>> 
        >>> # Create ViT model
        >>> model = build_model('vit_b_16', pretrained=True)
    """
    return LensClassifier(
        arch=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


def get_model_info(arch: str) -> dict:
    """
    Get information about a specific architecture.
    
    Args:
        arch: Architecture name
        
    Returns:
        Dictionary with model information
    """
    if arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Architecture '{arch}' not supported")
    
    return SUPPORTED_ARCHITECTURES[arch].copy()


def list_available_architectures() -> list:
    """List all available architectures."""
    return list(SUPPORTED_ARCHITECTURES.keys())


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Available architectures:")
    for arch in list_available_architectures():
        info = get_model_info(arch)
        print(f"  {arch}: {info['description']} (input size: {info['input_size']})")
    
    print("\nTesting model creation:")
    
    # Test ResNet-18
    try:
        resnet = build_model('resnet18', pretrained=True)
        print(f"✅ ResNet-18: {resnet._count_parameters():,} parameters")
        print(f"   Input size: {resnet.get_input_size()}")
        
        # Test forward pass
        x = torch.randn(1, 3, 64, 64)
        output = resnet(x)
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ ResNet-18 failed: {e}")
    
    # Test ViT-B/16
    try:
        vit = build_model('vit_b_16', pretrained=True)
        print(f"✅ ViT-B/16: {vit._count_parameters():,} parameters")
        print(f"   Input size: {vit.get_input_size()}")
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = vit(x)
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ ViT-B/16 failed: {e}")
