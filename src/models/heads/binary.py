#!/usr/bin/env python3
"""
Binary classification heads for gravitational lens classification.

This module provides classification heads that convert feature embeddings
into binary classification logits with Monte Carlo dropout support.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BinaryHead(nn.Module):
    """
    Binary classification head with Monte Carlo dropout support.
    
    This head applies dropout followed by a linear layer to convert
    feature embeddings into binary classification logits. The dropout
    can be used during inference for uncertainty estimation via
    Monte Carlo sampling.
    
    Features:
    - Configurable dropout rate for uncertainty estimation
    - Single output for binary classification (use with BCEWithLogitsLoss)
    - Supports both training and MC-dropout inference modes
    """
    
    def __init__(self, in_dim: int, p: float = 0.2) -> None:
        """
        Initialize binary classification head.
        
        Args:
            in_dim: Input feature dimension
            p: Dropout probability (0.0 to disable dropout)
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.dropout_p = p
        
        # Dropout layer for regularization and uncertainty estimation
        self.dropout = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        
        # Final classification layer
        self.fc = nn.Linear(in_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Created binary head: in_dim={in_dim}, dropout_p={p}")
    
    def _init_weights(self) -> None:
        """Initialize layer weights using best practices."""
        # Xavier/Glorot initialization for linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            features: Feature embeddings of shape [B, in_dim]
            
        Returns:
            Logits of shape [B] (squeezed from [B, 1])
        """
        # Apply dropout and linear layer
        x = self.dropout(features)
        logits = self.fc(x)  # Shape: [B, 1]
        
        # Squeeze to [B] for compatibility with BCEWithLogitsLoss
        return logits.squeeze(1)
    
    def mc_forward(self, features: torch.Tensor, mc_samples: int = 20) -> torch.Tensor:
        """
        Monte Carlo forward pass for uncertainty estimation.
        
        This method performs multiple forward passes with dropout enabled
        to estimate predictive uncertainty. The model should be in training
        mode to enable dropout during inference.
        
        Args:
            features: Feature embeddings of shape [B, in_dim]
            mc_samples: Number of Monte Carlo samples
            
        Returns:
            Logits of shape [mc_samples, B]
        """
        # Ensure dropout is enabled
        training_mode = self.training
        self.train()  # Enable dropout
        
        mc_logits = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.forward(features)
                mc_logits.append(logits)
        
        # Restore original training mode
        self.train(training_mode)
        
        # Stack MC samples: [mc_samples, B]
        return torch.stack(mc_logits, dim=0)
    
    def get_uncertainty(self, features: torch.Tensor, mc_samples: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate predictive uncertainty using Monte Carlo dropout.
        
        Args:
            features: Feature embeddings of shape [B, in_dim]
            mc_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean_probabilities, uncertainty_variance) both of shape [B]
        """
        # Get MC samples
        mc_logits = self.mc_forward(features, mc_samples)  # [mc_samples, B]
        
        # Convert to probabilities
        mc_probs = torch.sigmoid(mc_logits)  # [mc_samples, B]
        
        # Compute statistics
        mean_probs = mc_probs.mean(dim=0)  # [B]
        var_probs = mc_probs.var(dim=0, unbiased=False)  # [B]
        
        return mean_probs, var_probs
    
    def get_head_info(self) -> dict:
        """Get information about the classification head."""
        return {
            'type': 'binary_classification',
            'input_dim': self.in_dim,
            'output_dim': 1,
            'dropout_p': self.dropout_p,
            'num_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class MultiClassHead(nn.Module):
    """
    Multi-class classification head (for future extensions).
    
    This head supports multi-class classification problems while
    maintaining the same interface as BinaryHead.
    """
    
    def __init__(self, in_dim: int, num_classes: int, p: float = 0.2) -> None:
        """
        Initialize multi-class classification head.
        
        Args:
            in_dim: Input feature dimension
            num_classes: Number of output classes
            p: Dropout probability
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.dropout_p = p
        
        self.dropout = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)
        
        self._init_weights()
        
        logger.info(f"Created multi-class head: in_dim={in_dim}, "
                   f"num_classes={num_classes}, dropout_p={p}")
    
    def _init_weights(self) -> None:
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            features: Feature embeddings of shape [B, in_dim]
            
        Returns:
            Logits of shape [B, num_classes]
        """
        x = self.dropout(features)
        logits = self.fc(x)
        return logits
    
    def get_head_info(self) -> dict:
        """Get information about the classification head."""
        return {
            'type': 'multi_class_classification',
            'input_dim': self.in_dim,
            'output_dim': self.num_classes,
            'dropout_p': self.dropout_p,
            'num_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_binary_head(in_dim: int, p: float = 0.2) -> BinaryHead:
    """
    Factory function to create binary classification head.
    
    Args:
        in_dim: Input feature dimension
        p: Dropout probability
        
    Returns:
        Binary classification head
    """
    return BinaryHead(in_dim=in_dim, p=p)


def create_multiclass_head(in_dim: int, num_classes: int, p: float = 0.2) -> MultiClassHead:
    """
    Factory function to create multi-class classification head.
    
    Args:
        in_dim: Input feature dimension
        num_classes: Number of output classes
        p: Dropout probability
        
    Returns:
        Multi-class classification head
    """
    return MultiClassHead(in_dim=in_dim, num_classes=num_classes, p=p)
