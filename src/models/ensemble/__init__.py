"""
Ensemble methods for combining multiple models.
"""

from .ensemble_classifier import EnsembleClassifier
from .weighted import UncertaintyWeightedEnsemble, SimpleEnsemble, create_uncertainty_weighted_ensemble
from .registry import (
    make_model, get_model_info, list_available_models, 
    create_ensemble_members, create_resnet_vit_ensemble
)

__all__ = [
    'EnsembleClassifier',
    'UncertaintyWeightedEnsemble',
    'SimpleEnsemble',
    'create_uncertainty_weighted_ensemble',
    'make_model',
    'get_model_info',
    'list_available_models',
    'create_ensemble_members',
    'create_resnet_vit_ensemble'
]