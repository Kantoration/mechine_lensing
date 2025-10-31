#!/usr/bin/env python3
"""
Model registry for ensemble creation and management.

This module provides a centralized registry for creating backbone-head pairs
for different model architectures, supporting both ResNet and ViT models
with arbitrary input channel counts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import torch
import torch.nn as nn

from ..backbones.resnet import ResNetBackbone
from ..backbones.vit import ViTBackbone
from ..backbones.enhanced_light_transformer import EnhancedLightTransformerBackbone
from ..backbones.light_transformer import LightTransformerBackbone
from ..heads.binary import BinaryHead

logger = logging.getLogger(__name__)


@dataclass
class ModelContract:
    """
    Physics-aware model contract specifying input/output requirements.
    
    This ensures consistent handling of bands, normalization, pixel scales,
    and task types across ensemble members.
    
    References:
    - ML4SCI collaboration (ArXiv:2308.15738, 2023). "Reliable Scientific Machine Learning:
      Standards and Contracts." Factories with explicit, typed contracts enable reproducible,
      robust ML science.
    - Lanusse, F. et al. (2018). "CMU Deep Lens: CNNs for Strong Lens Modeling." ApJS.
      Physics parameters, pixel scales, and band order must be propagated for valid science.
    
    Modern science-facing ML frameworks enforce "contracts" recording band order, pixel scale,
    units, normalization, and input type for each model. This is essential for valid cross-model
    fusion, reproducibility, and correct physics interpretation.
    """
    name: str
    bands: List[str]  # e.g., ['g','r','i'] or ['u','g','r','i','z']
    input_size: int
    normalization: Dict[str, Dict[str, float]]  # {band: {mean, std}}
    pixel_scale_arcsec: Optional[float] = None
    dx: Optional[float] = None  # radians
    dy: Optional[float] = None  # radians
    sigma_crit_policy: Optional[str] = None  # 'dimensionless'|'physical'
    task_type: str = "classification"  # 'classification'|'regression_kappa'|'regression_psi'|'regression_alpha'
    input_type: str = "image"  # 'image'|'image+kappa'|'full_maps'
    
    def __post_init__(self):
        """Derive dx/dy from pixel_scale_arcsec if not provided."""
        if self.dx is None and self.dy is None and self.pixel_scale_arcsec is not None:
            step_rad = self.pixel_scale_arcsec * (math.pi / 180.0 / 3600.0)
            self.dx = step_rad
            self.dy = step_rad


# Architecture name aliases for backward compatibility
ALIASES = {
    'vit_b16': 'vit_b_16',
    'ViT-B/16': 'vit_b_16',
    'vit_b_16': 'vit_b_16'  # Already correct
}


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
    'vit_b_16': {
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
    ,
    'lens_gnn': {
        'backbone_class': None,  # LensGNN doesn't use CNN backbone pattern
        'backbone_kwargs': {
            'node_dim': 128,
            'hidden_dim': 128,
            'mp_layers': 4,
            'heads': 4,
            'uncertainty': 'heteroscedastic'
        },
        'feature_dim': 0,  # N/A for graph model
        'input_size': 224,
        'description': 'LensGNN: physics-informed graph model producing latent κ/ψ/α maps'
    }
}


def make_model(
    name: str,
    bands: int = 3,
    bands_list: Optional[List[str]] = None,  # Explicit band names ['g','r','i']
    pretrained: bool = True,
    dropout_p: float = 0.2,
    normalization: Optional[Dict[str, Dict[str, float]]] = None,
    pixel_scale_arcsec: Optional[float] = None,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    sigma_crit_policy: Optional[str] = "dimensionless",
    task_type: str = "classification",
    input_type: str = "image",
) -> Tuple[nn.Module, Optional[nn.Module], int, ModelContract]:
    """
    Create a backbone-head pair for the specified architecture with physics-aware contract.
    
    Args:
        name: Model architecture name
        bands: Number of input channels/bands (deprecated, use bands_list)
        bands_list: Explicit band names, e.g., ['g','r','i']
        pretrained: Whether to use pretrained weights
        dropout_p: Dropout probability for the classification head
        normalization: Per-band normalization stats {band: {mean, std}}
        pixel_scale_arcsec: Pixel scale in arcseconds
        dx, dy: Grid spacing in radians (derived from pixel_scale_arcsec if not provided)
        sigma_crit_policy: 'dimensionless' or 'physical' for κ units
        task_type: 'classification', 'regression_kappa', 'regression_psi', 'regression_alpha'
        input_type: 'image', 'image+kappa', 'full_maps'
        
    Returns:
        Tuple of (backbone_or_model, head, feature_dim, contract)
        For LensGNN: (model, None, 0, contract)
    """
    # Resolve architecture aliases for backward compatibility
    name = ALIASES.get(name, name)

    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model architecture '{name}'. Available: {available}")

    # Create model contract
    if bands_list is None:
        # Fallback: generate band names from count
        bands_list = [f'band_{i}' for i in range(bands)]
    
    contract = ModelContract(
        name=name,
        bands=bands_list,
        input_size=MODEL_REGISTRY[name]['input_size'],
        normalization=normalization or {},
        pixel_scale_arcsec=pixel_scale_arcsec,
        dx=dx,
        dy=dy,
        sigma_crit_policy=sigma_crit_policy,
        task_type=task_type,
        input_type=input_type
    )

    # Special handling for LensGNN (non-CNN pattern)
    if name == "lens_gnn":
        from mlensing.gnn.lens_gnn import LensGNN
        
        # Ensure dx/dy are available (derived from contract)
        if contract.dx is None or contract.dy is None:
            if contract.pixel_scale_arcsec is None:
                raise ValueError("LensGNN requires dx/dy or pixel_scale_arcsec")
            contract.__post_init__()  # Derive dx/dy
        
        # Get LensGNN kwargs from registry
        gnn_kwargs = MODEL_REGISTRY[name]['backbone_kwargs'].copy()
        model = LensGNN(
            node_dim=gnn_kwargs.get('node_dim', 128),
            hidden_dim=gnn_kwargs.get('hidden_dim', 128),
            mp_layers=gnn_kwargs.get('mp_layers', 4),
            heads=gnn_kwargs.get('heads', 4),
            uncertainty=gnn_kwargs.get('uncertainty', 'heteroscedastic')
        )
        model.contract = contract  # Attach contract for downstream ops
        
        logger.info(f"Created LensGNN with dx={contract.dx:.6e}, dy={contract.dy:.6e}")
        return model, None, 0, contract

    # Standard CNN/ViT models
    config = MODEL_REGISTRY[name]
    backbone_class = config['backbone_class']
    backbone_kwargs = config['backbone_kwargs'].copy()
    feature_dim = config['feature_dim']
    
    # Create backbone with multi-channel support
    backbone_kwargs.update({
        'in_ch': len(contract.bands),
        'pretrained': pretrained
    })
    backbone = backbone_class(**backbone_kwargs)
    backbone.contract = contract  # Attach contract
    
    # Create binary classification head (or regression head if task_type requires)
    if task_type == "classification":
        head = BinaryHead(in_dim=feature_dim, p=dropout_p)
    else:
        # Regression tasks don't use classification head
        head = None
    
    logger.info(f"Created model pair: {name} with bands={contract.bands}, "
               f"pretrained={pretrained}, dropout_p={dropout_p}, task_type={task_type}")
    
    return backbone, head, feature_dim, contract


# Backward compatibility: wrapper that returns old signature
def make_model_legacy(
    name: str,
    bands: int = 3,
    pretrained: bool = True,
    dropout_p: float = 0.2
) -> Tuple[nn.Module, nn.Module, int]:
    """
    Legacy wrapper for make_model that returns old signature without contract.
    
    This maintains backward compatibility for existing code that expects
    (backbone, head, feature_dim) tuple.
    """
    backbone, head, feature_dim, _ = make_model(
        name=name,
        bands=bands,
        pretrained=pretrained,
        dropout_p=dropout_p
    )
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
    return create_ensemble_members(['resnet18', 'vit_b_16'], bands=bands, pretrained=pretrained)


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
        'vit_b_16',   # Transformer baseline
        'enhanced_light_transformer_arc_aware',  # Physics-informed arc detection
        'enhanced_light_transformer_adaptive'   # Adaptive physics attention
    ]
    return create_ensemble_members(architectures, bands=bands, pretrained=pretrained)
