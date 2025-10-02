"""
Models package for gravitational lens classification.

This package provides a modular architecture for different model components:
- backbones: Feature extraction networks (ResNet, ViT)
- heads: Classification heads and output layers
- ensemble: Ensemble methods and model combination
- unified_factory: Single entry point for all model creation
"""

# Legacy factory removed - use unified_factory instead
from .lens_classifier import LensClassifier
from .unified_factory import (
    ModelConfig, UnifiedModelFactory,
    create_model, create_model_from_config_file,
    list_available_models, get_model_info,
    build_model  # Backward compatibility
)
from .ensemble import (
    make_model, get_model_info as get_ensemble_model_info, list_available_models as list_ensemble_models,
    UncertaintyWeightedEnsemble, create_uncertainty_weighted_ensemble
)

__all__ = [
    # Unified factory (recommended)
    'ModelConfig',
    'UnifiedModelFactory', 
    'create_model',
    'create_model_from_config_file',
    'list_available_models',
    'get_model_info',
    'build_model',  # Backward compatibility
    
    # Legacy compatibility (removed deprecated factory) 
    'LensClassifier',
    'make_model',
    'get_ensemble_model_info', 
    'list_ensemble_models',
    'UncertaintyWeightedEnsemble',
    'create_uncertainty_weighted_ensemble'
]
