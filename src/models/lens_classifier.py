#!/usr/bin/env python3
"""
Unified lens classifier wrapper for different architectures.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LensClassifier(nn.Module):
    """
    Unified wrapper for different architectures with binary classification head.
    
    This wrapper provides a consistent interface regardless of the underlying
    architecture (ResNet, ViT, etc.), making it easy to switch between models.
    """
    
    def __init__(
        self, 
        arch: str, 
        backbone: nn.Module,
        num_classes: int = 1,  # Binary classification (sigmoid output)
        dropout_rate: float = 0.5
    ):
        """
        Initialize lens classifier with specified architecture.
        
        Args:
            arch: Architecture name ('resnet18' or 'vit_b_16')
            backbone: Pre-configured backbone model
            num_classes: Number of output classes (1 for binary classification)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.arch = arch
        self.num_classes = num_classes
        self.backbone = backbone
        
        # Adapt final layer for binary classification
        self._adapt_classifier_head(dropout_rate)
        
        # Log model creation
        param_count = self._count_parameters()
        logger.info(f"Created {arch} classifier with {param_count:,} parameters")
    
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
        else:
            raise ValueError(f"Unknown architecture adaptation for: {self.arch}")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Return architecture information."""
        return {
            'architecture': self.arch,
            'num_classes': self.num_classes,
            'num_parameters': self._count_parameters(),
        }
