"""
Models package for gravitational lens classification.

This package provides a modular architecture for different model components:
- backbones: Feature extraction networks (ResNet, ViT)
- heads: Classification heads and output layers
- ensemble: Ensemble methods and model combination
"""

from .factory import build_model, list_available_architectures
from .lens_classifier import LensClassifier
from .ensemble import (
    make_model, get_model_info, list_available_models,
    UncertaintyWeightedEnsemble, create_uncertainty_weighted_ensemble
)

__all__ = [
    'build_model',
    'list_available_architectures', 
    'LensClassifier',
    'make_model',
    'get_model_info', 
    'list_available_models',
    'UncertaintyWeightedEnsemble',
    'create_uncertainty_weighted_ensemble'
]
