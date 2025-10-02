#!/usr/bin/env python3
"""
Model registry for ensemble creation and management.

This module provides a centralized registry for creating backbone-head pairs
for different model architectures, supporting both ResNet and ViT models
with arbitrary input channel counts.
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, Optional

import torch.nn as nn

from ..backbones.resnet import ResNetBackbone
from ..backbones.vit import ViTBackbone
from ..backbones.enhanced_light_transformer import EnhancedLightTransformerBackbone
from ..backbones.light_transformer import LightTransformerBackbone
from ..heads.binary import BinaryHead

logger = logging.getLogger(__name__)


# Registry of available model architectures
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    'resnet18': {
        'backbone_class': ResNetBackbone,
        'backbone_kwargs': {'arch': 'resnet18'},
        'feature_dim': 512,
        'input_size': 64,
        'description': 'ResNet-18 Convolutional Neural Network'
    },
    'resnet34': {
        'backbone_class': ResNetBackbone,
        'backbone_kwargs': {'arch': 'resnet34'},
        'feature_dim': 512,
        'input_size': 64,
        'description': 'ResNet-34 Convolutional Neural Network (Deeper)'
    },
    'vit_b16': {
        'backbone_class': ViTBackbone,
        'backbone_kwargs': {},
        'feature_dim': 768,
        'input_size': 224,
        'description': 'Vision Transformer Base with 16x16 patches'
    },
    'light_transformer': {
        'backbone_class': LightTransformerBackbone,
        'backbone_kwargs': {
            'max_tokens': 256
        },
        'feature_dim': 256,
        'input_size': 112,
        'description': 'Light Transformer: CNN features + Self-Attention (2M params)'
    },
    'trans_enc_s': {
        'backbone_class': LightTransformerBackbone,
        'backbone_kwargs': {
            'cnn_stage': 'layer3',
            'patch_size': 2,
            'embed_dim': 256,
            'num_heads': 4,
            'num_layers': 4,
            'mlp_ratio': 2.0,
            'attn_drop': 0.0,
            'proj_drop': 0.1,
            'pos_drop': 0.1,
            'drop_path_max': 0.1,
            'pooling': 'avg',
            'freeze_until': 'none',
            'max_tokens': 256
        },
        'feature_dim': 256,
        'input_size': 112,
        'description': 'Enhanced Light Transformer: Production-ready CNN+Transformer with advanced regularization'
    },
    'enhanced_light_transformer_arc_aware': {
        'backbone_class': EnhancedLightTransformerBackbone,
        'backbone_kwargs': {
            'cnn_stage': 'layer3',
            'patch_size': 2,
            'embed_dim': 256,
            'num_heads': 4,
            'num_layers': 4,
            'attention_type': 'arc_aware',
            'attention_config': {
                'arc_prior_strength': 0.1,
                'curvature_sensitivity': 1.0
            }
        },
        'feature_dim': 256,
        'input_size': 112,
        'description': 'Enhanced Light Transformer with arc-aware attention for gravitational lensing detection'
    },
    'enhanced_light_transformer_multi_scale': {
        'backbone_class': EnhancedLightTransformerBackbone,
        'backbone_kwargs': {
            'cnn_stage': 'layer3',
            'patch_size': 2,
            'embed_dim': 256,
            'num_heads': 4,
            'num_layers': 4,
            'attention_type': 'multi_scale',
            'attention_config': {
                'scales': [1, 2, 4],
                'fusion_method': 'weighted_sum'
            }
        },
        'feature_dim': 256,
        'input_size': 112,
        'description': 'Enhanced Light Transformer with multi-scale attention for different arc sizes'
    },
    'enhanced_light_transformer_adaptive': {
        'backbone_class': EnhancedLightTransformerBackbone,
        'backbone_kwargs': {
            'cnn_stage': 'layer3',
            'patch_size': 2,
            'embed_dim': 256,
            'num_heads': 4,
            'num_layers': 4,
            'attention_type': 'adaptive',
            'attention_config': {
                'adaptation_layers': 2
            }
        },
        'feature_dim': 256,
        'input_size': 112,
        'description': 'Enhanced Light Transformer with adaptive attention based on image characteristics'
    }
}


def make_model(
    name: str, 
    bands: int = 3, 
    pretrained: bool = True,
    dropout_p: float = 0.2
) -> Tuple[nn.Module, nn.Module, int]:
    """
    Create a backbone-head pair for the specified architecture.
    
    Args:
        name: Model architecture name ('resnet18', 'resnet34', 'vit_b16', 'light_transformer', 'trans_enc_s')
        bands: Number of input channels/bands
        pretrained: Whether to use pretrained weights
        dropout_p: Dropout probability for the classification head
        
    Returns:
        Tuple of (backbone, head, feature_dim)
        
    Raises:
        ValueError: If the architecture name is not supported
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model architecture '{name}'. Available: {available}")
    
    # Get model configuration
    config = MODEL_REGISTRY[name]
    backbone_class = config['backbone_class']
    backbone_kwargs = config['backbone_kwargs'].copy()
    feature_dim = config['feature_dim']
    
    # Create backbone with multi-channel support
    backbone_kwargs.update({
        'in_ch': bands,
        'pretrained': pretrained
    })
    backbone = backbone_class(**backbone_kwargs)
    
    # Create binary classification head
    head = BinaryHead(in_dim=feature_dim, p=dropout_p)
    
    logger.info(f"Created model pair: {name} with {bands} bands, "
               f"pretrained={pretrained}, dropout_p={dropout_p}")
    
    return backbone, head, feature_dim


def get_model_info(name: str) -> Dict[str, Any]:
    """
    Get information about a model architecture.
    
    Args:
        name: Model architecture name
        
    Returns:
        Dictionary containing model information
        
    Raises:
        ValueError: If the architecture name is not supported
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model architecture '{name}'. Available: {available}")
    
    return MODEL_REGISTRY[name].copy()


def list_available_models() -> list[str]:
    """
    List all available model architectures.
    
    Returns:
        List of available model architecture names
    """
    return list(MODEL_REGISTRY.keys())


def get_recommended_input_size(name: str) -> int:
    """
    Get the recommended input image size for a model architecture.
    
    Args:
        name: Model architecture name
        
    Returns:
        Recommended input image size (height/width)
        
    Raises:
        ValueError: If the architecture name is not supported
    """
    info = get_model_info(name)
    return info['input_size']


def register_model(
    name: str,
    backbone_class: type,
    backbone_kwargs: Dict[str, Any],
    feature_dim: int,
    input_size: int,
    description: str
) -> None:
    """
    Register a new model architecture.
    
    This function allows extending the registry with new architectures
    without modifying the core registry code.
    
    Args:
        name: Unique name for the architecture
        backbone_class: Backbone class (must accept in_ch and pretrained kwargs)
        backbone_kwargs: Additional kwargs for backbone initialization
        feature_dim: Output feature dimension of the backbone
        input_size: Recommended input image size
        description: Human-readable description
        
    Raises:
        ValueError: If the name is already registered
    """
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is already registered")
    
    MODEL_REGISTRY[name] = {
        'backbone_class': backbone_class,
        'backbone_kwargs': backbone_kwargs,
        'feature_dim': feature_dim,
        'input_size': input_size,
        'description': description
    }
    
    logger.info(f"Registered new model architecture: {name}")


def create_ensemble_members(
    architectures: list[str],
    bands: int = 3,
    pretrained: bool = True,
    dropout_p: float = 0.2
) -> list[Tuple[nn.Module, nn.Module]]:
    """
    Create multiple backbone-head pairs for ensemble learning.
    
    Args:
        architectures: List of architecture names
        bands: Number of input channels/bands
        pretrained: Whether to use pretrained weights
        dropout_p: Dropout probability for classification heads
        
    Returns:
        List of (backbone, head) tuples
    """
    members = []
    
    for arch in architectures:
        backbone, head, _ = make_model(
            name=arch,
            bands=bands,
            pretrained=pretrained,
            dropout_p=dropout_p
        )
        members.append((backbone, head))
    
    logger.info(f"Created ensemble with {len(members)} members: {architectures}")
    
    return members


def validate_ensemble_compatibility(architectures: list[str], bands: int) -> None:
    """
    Validate that all architectures are compatible for ensemble learning.
    
    This function checks that all architectures can handle the specified
    number of input bands and are suitable for ensemble combination.
    
    Args:
        architectures: List of architecture names
        bands: Number of input channels/bands
        
    Raises:
        ValueError: If architectures are incompatible
    """
    if not architectures:
        raise ValueError("At least one architecture must be specified")
    
    # Check all architectures exist
    for arch in architectures:
        if arch not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown architecture '{arch}'. Available: {available}")
    
    # Validate bands
    if bands < 1:
        raise ValueError(f"Number of bands must be positive, got {bands}")
    
    # Log compatibility check
    logger.info(f"Validated ensemble compatibility: {architectures} with {bands} bands")


# Convenience functions for common ensemble configurations
def create_resnet_vit_ensemble(bands: int = 3, pretrained: bool = True) -> list[Tuple[nn.Module, nn.Module]]:
    """Create a ResNet-18 + ViT-B/16 ensemble."""
    return create_ensemble_members(['resnet18', 'vit_b16'], bands=bands, pretrained=pretrained)


def create_resnet_ensemble(bands: int = 3, pretrained: bool = True) -> list[Tuple[nn.Module, nn.Module]]:
    """Create a ResNet-18 + ResNet-34 ensemble."""
    return create_ensemble_members(['resnet18', 'resnet34'], bands=bands, pretrained=pretrained)


def create_physics_informed_ensemble(bands: int = 3, pretrained: bool = True) -> list[Tuple[nn.Module, nn.Module]]:
    """Create an ensemble with physics-informed attention mechanisms."""
    architectures = [
        'resnet18',  # Baseline CNN
        'enhanced_light_transformer_arc_aware',  # Arc detection
        'enhanced_light_transformer_multi_scale',  # Multi-scale features
        'enhanced_light_transformer_adaptive'  # Adaptive attention
    ]
    return create_ensemble_members(architectures, bands=bands, pretrained=pretrained)


def create_comprehensive_ensemble(bands: int = 3, pretrained: bool = True) -> list[Tuple[nn.Module, nn.Module]]:
    """Create a comprehensive ensemble combining traditional and physics-informed models."""
    architectures = [
        'resnet18',  # Fast CNN baseline
        'resnet34',  # Deeper CNN
        'vit_b16',   # Transformer baseline
        'enhanced_light_transformer_arc_aware',  # Physics-informed arc detection
        'enhanced_light_transformer_adaptive'   # Adaptive physics attention
    ]
    return create_ensemble_members(architectures, bands=bands, pretrained=pretrained)
