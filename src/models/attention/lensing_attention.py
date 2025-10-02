#!/usr/bin/env python3
"""
lensing_attention.py
===================
Explicit attention mechanisms for gravitational lensing feature detection.

Key Features:
- Arc-aware attention for detecting lensing arcs
- Multi-scale attention for different arc sizes
- Physics-informed attention priors
- Adaptive attention based on image characteristics
- Interpretable attention visualization

Usage:
    from models.attention.lensing_attention import ArcAwareAttention, MultiScaleAttention
"""

from __future__ import annotations

import logging
import math
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ArcAwareAttention(nn.Module):
    """
    Attention mechanism specifically designed for gravitational lensing arc detection.
    
    This module implements physics-informed attention that:
    - Focuses on curved structures (potential arcs)
    - Uses radial and tangential attention patterns
    - Adapts to different arc orientations and curvatures
    - Provides interpretable attention maps
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        arc_prior_strength: float = 0.1,
        curvature_sensitivity: float = 1.0
    ):
        """
        Initialize arc-aware attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            arc_prior_strength: Strength of arc detection prior
            curvature_sensitivity: Sensitivity to curvature patterns
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Standard attention projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Arc-specific attention components
        self.arc_prior_strength = arc_prior_strength
        self.curvature_sensitivity = curvature_sensitivity
        
        # Learnable arc detection filters
        self.arc_detector = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.curvature_detector = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        
        # Initialize arc detection filters with physics-informed patterns
        self._init_arc_filters()
        
        # Dropout for regularization
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)
        
    def _init_arc_filters(self):
        """Initialize filters with physics-informed arc detection patterns."""
        # Arc detector: emphasizes curved structures
        arc_kernel = torch.tensor([
            [[[-1, -1, -1],
              [ 2,  2,  2],
              [-1, -1, -1]]]
        ], dtype=torch.float32)
        self.arc_detector.weight.data = arc_kernel
        
        # Curvature detector: emphasizes curvature changes
        curvature_kernel = torch.tensor([
            [[[ 0,  0, -1,  0,  0],
              [ 0, -1,  2, -1,  0],
              [-1,  2,  4,  2, -1],
              [ 0, -1,  2, -1,  0],
              [ 0,  0, -1,  0,  0]]]
        ], dtype=torch.float32)
        self.curvature_detector.weight.data = curvature_kernel
        
    def _compute_arc_attention_prior(
        self, 
        x: torch.Tensor, 
        H: int, 
        W: int
    ) -> torch.Tensor:
        """
        Compute physics-informed attention prior for arc detection.
        
        Args:
            x: Input features [B, N, embed_dim]
            H: Height of spatial grid
            W: Width of spatial grid
            
        Returns:
            Arc attention prior [B, N, N]
        """
        B, N, C = x.shape
        
        # Reshape to spatial format for convolution
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Compute arc and curvature features
        # Use mean across channels for arc detection
        x_mean = x_spatial.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        arc_features = self.arc_detector(x_mean)  # [B, 1, H, W]
        curvature_features = self.curvature_detector(x_mean)  # [B, 1, H, W]
        
        # Combine arc and curvature information
        arc_prior = torch.sigmoid(arc_features + curvature_features)  # [B, 1, H, W]
        
        # Reshape back to sequence format
        arc_prior = arc_prior.reshape(B, H * W)  # [B, N]
        
        # Create attention prior matrix
        # Higher attention for positions with strong arc features
        attention_prior = torch.outer(arc_prior, arc_prior)  # [B, N, N]
        
        # Normalize to prevent overwhelming the learned attention
        attention_prior = attention_prior * self.arc_prior_strength
        
        return attention_prior
    
    def forward(
        self, 
        x: torch.Tensor, 
        H: Optional[int] = None, 
        W: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with arc-aware attention.
        
        Args:
            x: Input embeddings [B, N, embed_dim]
            H: Height of spatial grid (for arc prior computation)
            W: Width of spatial grid (for arc prior computation)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Compute standard attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Add arc-aware prior if spatial dimensions are available
        if H is not None and W is not None and H * W == N:
            arc_prior = self._compute_arc_attention_prior(x, H, W)  # [B, N, N]
            # Broadcast arc prior to all heads
            arc_prior = arc_prior.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn + arc_prior
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Return attended features and attention weights for visualization
        attention_weights = attn.mean(dim=1)  # Average across heads [B, N, N]
        
        return x, attention_weights


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for detecting lensing features at different scales.
    
    This module processes features at multiple scales to capture:
    - Large-scale lensing arcs
    - Small-scale lensing features
    - Multi-scale galaxy structures
    - Scale-invariant lensing patterns
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        scales: list = [1, 2, 4],
        fusion_method: str = "weighted_sum"
    ):
        """
        Initialize multi-scale attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            scales: List of scale factors for multi-scale processing
            fusion_method: Method to fuse multi-scale features
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scales = scales
        self.fusion_method = fusion_method
        
        # Multi-scale attention modules
        self.scale_attentions = nn.ModuleList([
            ArcAwareAttention(embed_dim, num_heads) for _ in scales
        ])
        
        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in scales
        ])
        
        # Feature fusion
        if fusion_method == "weighted_sum":
            self.fusion_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        elif fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(embed_dim, num_heads)
        elif fusion_method == "mlp":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim * len(scales), embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def _apply_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Apply scale transformation to input features.
        
        Args:
            x: Input features [B, N, embed_dim]
            scale: Scale factor
            
        Returns:
            Scaled features [B, N', embed_dim]
        """
        B, N, C = x.shape
        
        if scale == 1:
            return x
        
        # Reshape to spatial format
        H = W = int(math.sqrt(N))
        if H * W != N:
            # If not a perfect square, use adaptive pooling
            x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
            new_size = max(1, H // scale)
            x_scaled = F.adaptive_avg_pool2d(x_spatial, (new_size, new_size))
            x_scaled = x_scaled.reshape(B, C, -1).transpose(1, 2)
            return x_scaled
        
        # Perfect square case
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply scale transformation
        if scale > 1:
            # Downsample
            new_size = max(1, H // scale)
            x_scaled = F.adaptive_avg_pool2d(x_spatial, (new_size, new_size))
        else:
            # Upsample
            new_size = H * abs(scale)
            x_scaled = F.interpolate(x_spatial, size=(new_size, new_size), mode='bilinear', align_corners=False)
        
        # Reshape back to sequence format
        x_scaled = x_scaled.reshape(B, C, -1).transpose(1, 2)
        
        return x_scaled
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-scale attention.
        
        Args:
            x: Input embeddings [B, N, embed_dim]
            
        Returns:
            Tuple of (fused_features, attention_maps)
        """
        B, N, C = x.shape
        
        # Process at multiple scales
        scale_features = []
        attention_maps = {}
        
        for i, scale in enumerate(self.scales):
            # Apply scale transformation
            x_scaled = self._apply_scale(x, scale)
            
            # Apply scale-specific attention
            x_attended, attn_weights = self.scale_attentions[i](x_scaled)
            
            # Project scale-specific features
            x_projected = self.scale_projections[i](x_attended)
            
            # Upsample back to original resolution if needed
            if scale != 1:
                x_projected = self._apply_scale(x_projected, 1 // scale)
            
            scale_features.append(x_projected)
            attention_maps[f'scale_{scale}'] = attn_weights
        
        # Fuse multi-scale features
        if self.fusion_method == "weighted_sum":
            # Weighted sum of scale features
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, scale_features))
            
        elif self.fusion_method == "attention":
            # Attention-based fusion
            # Stack features: [B, N, embed_dim * num_scales]
            stacked = torch.cat(scale_features, dim=-1)
            # Use attention to select and combine features
            fused, _ = self.fusion_attention(stacked, stacked, stacked)
            
        elif self.fusion_method == "mlp":
            # MLP-based fusion
            stacked = torch.cat(scale_features, dim=-1)
            fused = self.fusion_mlp(stacked)
        
        # Final output projection
        output = self.output_proj(fused)
        
        return output, attention_maps


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism that adjusts based on image characteristics.
    
    This module:
    - Analyzes image properties (brightness, contrast, structure)
    - Adapts attention patterns accordingly
    - Provides different attention strategies for different image types
    - Learns to focus on relevant features automatically
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        adaptation_layers: int = 2
    ):
        """
        Initialize adaptive attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            adaptation_layers: Number of layers for adaptation
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Image characteristic analysis
        self.image_analyzer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 4),  # brightness, contrast, structure, complexity
            nn.Sigmoid()
        )
        
        # Adaptive attention strategies
        self.arc_attention = ArcAwareAttention(embed_dim, num_heads)
        self.standard_attention = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Adaptation network
        self.adaptation_net = nn.Sequential(
            nn.Linear(4, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 2),  # weights for arc vs standard attention
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive attention.
        
        Args:
            x: Input embeddings [B, N, embed_dim]
            
        Returns:
            Tuple of (adapted_features, adaptation_info)
        """
        B, N, C = x.shape
        
        # Analyze image characteristics
        # Use mean pooling to get global image features
        global_features = x.mean(dim=1)  # [B, embed_dim]
        image_chars = self.image_analyzer(global_features)  # [B, 4]
        
        # Compute adaptation weights
        adaptation_weights = self.adaptation_net(image_chars)  # [B, 2]
        
        # Apply different attention strategies
        # Arc-aware attention
        x_arc, arc_attn = self.arc_attention(x)
        
        # Standard attention
        x_std, std_attn = self.standard_attention(x, x, x)
        
        # Adaptive fusion
        arc_weight = adaptation_weights[:, 0:1].unsqueeze(-1)  # [B, 1, 1]
        std_weight = adaptation_weights[:, 1:2].unsqueeze(-1)  # [B, 1, 1]
        
        x_fused = arc_weight * x_arc + std_weight * x_std
        
        # Final projection
        output = self.output_proj(x_fused)
        
        # Collect adaptation information
        adaptation_info = {
            'image_characteristics': image_chars,
            'adaptation_weights': adaptation_weights,
            'arc_attention': arc_attn,
            'standard_attention': std_attn
        }
        
        return output, adaptation_info


def create_lensing_attention(
    attention_type: str = "arc_aware",
    embed_dim: int = 256,
    num_heads: int = 4,
    **kwargs
) -> nn.Module:
    """
    Factory function to create lensing attention modules.
    
    Args:
        attention_type: Type of attention mechanism
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        **kwargs: Additional arguments for specific attention types
        
    Returns:
        Attention module
    """
    if attention_type == "arc_aware":
        return ArcAwareAttention(embed_dim, num_heads, **kwargs)
    elif attention_type == "multi_scale":
        return MultiScaleAttention(embed_dim, num_heads, **kwargs)
    elif attention_type == "adaptive":
        return AdaptiveAttention(embed_dim, num_heads, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def visualize_attention_maps(
    attention_weights: torch.Tensor,
    input_shape: Tuple[int, int],
    save_path: Optional[str] = None
) -> torch.Tensor:
    """
    Visualize attention maps for interpretability.
    
    Args:
        attention_weights: Attention weights [B, N, N] or [N, N]
        input_shape: Shape of input spatial grid (H, W)
        save_path: Optional path to save visualization
        
    Returns:
        Visualization tensor
    """
    if attention_weights.dim() == 3:
        # Take mean across batch
        attention_weights = attention_weights.mean(dim=0)
    
    H, W = input_shape
    N = H * W
    
    # Reshape attention weights to spatial format
    attn_spatial = attention_weights[:N, :N].reshape(N, H, W)
    
    # Create visualization
    # Average attention across query positions
    attn_vis = attn_spatial.mean(dim=0)  # [H, W]
    
    # Normalize for visualization
    attn_vis = (attn_vis - attn_vis.min()) / (attn_vis.max() - attn_vis.min() + 1e-8)
    
    if save_path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(attn_vis.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Attention Map Visualization')
        plt.savefig(save_path)
        plt.close()
    
    return attn_vis
