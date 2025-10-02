#!/usr/bin/env python3
"""
physics_regularized_attention.py
================================
Physics-regularized attention mechanisms with learnable kernels.

Key Features:
- Learnable physics-inspired kernels with regularization
- Physics-constrained loss functions
- End-to-end learning with physics priors
- Interpretable kernel evolution during training

Usage:
    from models.attention.physics_regularized_attention import PhysicsRegularizedAttention
"""

from __future__ import annotations

import logging
import math
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PhysicsRegularizedAttention(nn.Module):
    """
    Physics-regularized attention with learnable kernels.
    
    This module learns physics-inspired attention patterns end-to-end while
    maintaining interpretability through regularization constraints.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        physics_weight: float = 0.1,
        kernel_size: int = 3,
        num_kernels: int = 4
    ):
        """
        Initialize physics-regularized attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            physics_weight: Weight for physics regularization
            kernel_size: Size of learnable kernels
            num_kernels: Number of different kernel types
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.physics_weight = physics_weight
        
        # Standard attention projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable physics-inspired kernels
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, kernel_size, kernel_size) * 0.1)
            for _ in range(num_kernels)
        ])
        
        # Kernel type indicators (for regularization)
        self.kernel_types = ['arc', 'curvature', 'radial', 'tangential']
        
        # Physics constraints
        self.physics_constraints = self._create_physics_constraints()
        
        # Dropout
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)
        
        # Initialize kernels with physics-informed patterns
        self._init_physics_kernels()
    
    def _create_physics_constraints(self) -> Dict[str, torch.Tensor]:
        """Create physics constraints for kernel regularization."""
        constraints = {}
        
        # Arc constraint: should detect curved structures
        constraints['arc'] = torch.tensor([
            [[[-1, -1, -1],
              [ 2,  2,  2],
              [-1, -1, -1]]]
        ], dtype=torch.float32)
        
        # Curvature constraint: should detect curvature changes
        constraints['curvature'] = torch.tensor([
            [[[ 0,  0, -1,  0,  0],
              [ 0, -1,  2, -1,  0],
              [-1,  2,  4,  2, -1],
              [ 0, -1,  2, -1,  0],
              [ 0,  0, -1,  0,  0]]]
        ], dtype=torch.float32)
        
        # Radial constraint: should detect radial patterns
        constraints['radial'] = torch.tensor([
            [[[ 0, -1,  0],
              [-1,  4, -1],
              [ 0, -1,  0]]]
        ], dtype=torch.float32)
        
        # Tangential constraint: should detect tangential patterns
        constraints['tangential'] = torch.tensor([
            [[[-1,  0, -1],
              [ 0,  4,  0],
              [-1,  0, -1]]]
        ], dtype=torch.float32)
        
        return constraints
    
    def _init_physics_kernels(self):
        """Initialize kernels with physics-informed patterns."""
        for i, kernel in enumerate(self.kernels):
            kernel_type = self.kernel_types[i]
            if kernel_type in self.physics_constraints:
                # Initialize with physics constraint
                constraint = self.physics_constraints[kernel_type]
                if constraint.shape == kernel.shape:
                    kernel.data = constraint
                else:
                    # Resize constraint to match kernel size
                    constraint_resized = F.interpolate(
                        constraint.unsqueeze(0), 
                        size=(kernel.shape[-2], kernel.shape[-1]), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    kernel.data = constraint_resized
            else:
                # Random initialization with small values
                nn.init.normal_(kernel, std=0.1)
    
    def _compute_physics_regularization(self) -> torch.Tensor:
        """Compute physics regularization loss for kernels."""
        reg_loss = 0.0
        
        for i, kernel in enumerate(self.kernels):
            kernel_type = self.kernel_types[i]
            
            if kernel_type in self.physics_constraints:
                constraint = self.physics_constraints[kernel_type]
                
                # Resize constraint to match kernel size
                if constraint.shape != kernel.shape:
                    constraint_resized = F.interpolate(
                        constraint.unsqueeze(0), 
                        size=(kernel.shape[-2], kernel.shape[-1]), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                else:
                    constraint_resized = constraint
                
                # L2 regularization towards physics constraint
                reg_loss += F.mse_loss(kernel, constraint_resized)
            
            # Additional regularization: encourage sparsity
            reg_loss += 0.01 * torch.norm(kernel, p=1)
        
        return reg_loss
    
    def _compute_physics_attention_prior(
        self, 
        x: torch.Tensor, 
        H: int, 
        W: int
    ) -> torch.Tensor:
        """
        Compute physics-informed attention prior using learnable kernels.
        
        Args:
            x: Input features [B, N, embed_dim]
            H: Height of spatial grid
            W: Width of spatial grid
            
        Returns:
            Physics attention prior [B, N, N]
        """
        B, N, C = x.shape
        
        # Reshape to spatial format
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply learnable kernels
        kernel_outputs = []
        for kernel in self.kernels:
            # Apply kernel to each channel
            kernel_out = F.conv2d(
                x_spatial, 
                kernel.expand(C, -1, -1, -1), 
                padding=kernel.shape[-1]//2, 
                groups=C
            )
            kernel_outputs.append(kernel_out)
        
        # Combine kernel outputs
        combined_features = torch.stack(kernel_outputs, dim=1)  # [B, num_kernels, C, H, W]
        combined_features = combined_features.mean(dim=1)  # [B, C, H, W]
        
        # Compute attention prior
        # Use mean across channels for attention computation
        attention_map = combined_features.mean(dim=1)  # [B, H, W]
        attention_map = torch.sigmoid(attention_map)  # [B, H, W]
        
        # Reshape to sequence format
        attention_map = attention_map.reshape(B, H * W)  # [B, N]
        
        # Create attention prior matrix
        attention_prior = torch.bmm(
            attention_map.unsqueeze(2), 
            attention_map.unsqueeze(1)
        )  # [B, N, N]
        
        return attention_prior
    
    def forward(
        self, 
        x: torch.Tensor, 
        H: Optional[int] = None, 
        W: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with physics-regularized attention.
        
        Args:
            x: Input embeddings [B, N, embed_dim]
            H: Height of spatial grid
            W: Width of spatial grid
            
        Returns:
            Tuple of (attended_features, attention_weights, physics_reg_loss)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Compute standard attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Add physics-informed prior if spatial dimensions available
        if H is not None and W is not None and H * W == N:
            physics_prior = self._compute_physics_attention_prior(x, H, W)  # [B, N, N]
            # Broadcast physics prior to all heads
            physics_prior = physics_prior.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn + self.physics_weight * physics_prior
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Compute physics regularization loss
        physics_reg_loss = self._compute_physics_regularization()
        
        # Return attended features, attention weights, and regularization loss
        attention_weights = attn.mean(dim=1)  # Average across heads [B, N, N]
        
        return x, attention_weights, physics_reg_loss


class AdaptivePhysicsAttention(nn.Module):
    """
    Adaptive physics attention that learns when to apply physics priors.
    
    This module learns to adaptively apply physics constraints based on
    image characteristics and training progress.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        adaptation_layers: int = 2
    ):
        """
        Initialize adaptive physics attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            adaptation_layers: Number of adaptation layers
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Physics-regularized attention
        self.physics_attention = PhysicsRegularizedAttention(embed_dim, num_heads)
        
        # Standard attention
        self.standard_attention = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Adaptation network
        self.adaptation_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 2),  # weights for physics vs standard
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        H: Optional[int] = None, 
        W: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive physics attention.
        
        Args:
            x: Input embeddings [B, N, embed_dim]
            H: Height of spatial grid
            W: Width of spatial grid
            
        Returns:
            Tuple of (adapted_features, adaptation_info)
        """
        B, N, C = x.shape
        
        # Analyze input characteristics
        global_features = x.mean(dim=1)  # [B, embed_dim]
        adaptation_weights = self.adaptation_net(global_features)  # [B, 2]
        
        # Apply physics-regularized attention
        x_physics, physics_attn, physics_reg_loss = self.physics_attention(x, H, W)
        
        # Apply standard attention
        x_std, std_attn = self.standard_attention(x, x, x)
        
        # Adaptive fusion
        physics_weight = adaptation_weights[:, 0:1].unsqueeze(-1)  # [B, 1, 1]
        std_weight = adaptation_weights[:, 1:2].unsqueeze(-1)  # [B, 1, 1]
        
        x_fused = physics_weight * x_physics + std_weight * x_std
        
        # Final projection
        output = self.output_proj(x_fused)
        
        # Collect adaptation information
        adaptation_info = {
            'adaptation_weights': adaptation_weights,
            'physics_attention': physics_attn,
            'standard_attention': std_attn,
            'physics_reg_loss': physics_reg_loss
        }
        
        return output, adaptation_info


def create_physics_regularized_attention(
    attention_type: str = "physics_regularized",
    embed_dim: int = 256,
    num_heads: int = 4,
    **kwargs
) -> nn.Module:
    """
    Factory function to create physics-regularized attention modules.
    
    Args:
        attention_type: Type of attention mechanism
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        **kwargs: Additional arguments
        
    Returns:
        Attention module
    """
    if attention_type == "physics_regularized":
        return PhysicsRegularizedAttention(embed_dim, num_heads, **kwargs)
    elif attention_type == "adaptive_physics":
        return AdaptivePhysicsAttention(embed_dim, num_heads, **kwargs)
    else:
        raise ValueError(f"Unknown physics attention type: {attention_type}")


def analyze_kernel_evolution(
    model: PhysicsRegularizedAttention,
    save_path: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Analyze the evolution of physics kernels during training.
    
    Args:
        model: Physics-regularized attention model
        save_path: Optional path to save analysis
        
    Returns:
        Dictionary with kernel analysis
    """
    analysis = {}
    
    for i, kernel in enumerate(model.kernels):
        kernel_type = model.kernel_types[i]
        
        # Compute kernel statistics
        kernel_stats = {
            'mean': kernel.mean().item(),
            'std': kernel.std().item(),
            'min': kernel.min().item(),
            'max': kernel.max().item(),
            'norm': torch.norm(kernel).item()
        }
        
        analysis[f'kernel_{i}_{kernel_type}'] = kernel_stats
    
    # Compute physics constraint alignment
    physics_alignment = {}
    for i, kernel in enumerate(model.kernels):
        kernel_type = model.kernel_types[i]
        if kernel_type in model.physics_constraints:
            constraint = model.physics_constraints[kernel_type]
            
            # Resize constraint to match kernel size
            if constraint.shape != kernel.shape:
                constraint_resized = F.interpolate(
                    constraint.unsqueeze(0), 
                    size=(kernel.shape[-2], kernel.shape[-1]), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            else:
                constraint_resized = constraint
            
            # Compute alignment (cosine similarity)
            alignment = F.cosine_similarity(
                kernel.flatten(), 
                constraint_resized.flatten(), 
                dim=0
            ).item()
            
            physics_alignment[f'kernel_{i}_{kernel_type}'] = alignment
    
    analysis['physics_alignment'] = physics_alignment
    
    if save_path:
        import matplotlib.pyplot as plt
        
        # Plot kernel evolution
        fig, axes = plt.subplots(2, len(model.kernels), figsize=(4*len(model.kernels), 8))
        
        for i, kernel in enumerate(model.kernels):
            kernel_type = model.kernel_types[i]
            
            # Plot current kernel
            axes[0, i].imshow(kernel.detach().cpu().numpy()[0, 0], cmap='RdBu_r')
            axes[0, i].set_title(f'Learned {kernel_type} kernel')
            axes[0, i].axis('off')
            
            # Plot physics constraint
            if kernel_type in model.physics_constraints:
                constraint = model.physics_constraints[kernel_type]
                if constraint.shape != kernel.shape:
                    constraint_resized = F.interpolate(
                        constraint.unsqueeze(0), 
                        size=(kernel.shape[-2], kernel.shape[-1]), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                else:
                    constraint_resized = constraint
                
                axes[1, i].imshow(constraint_resized.detach().cpu().numpy()[0, 0], cmap='RdBu_r')
                axes[1, i].set_title(f'Physics {kernel_type} constraint')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    return analysis
