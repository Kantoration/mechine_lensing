"""
Ensemble methods for combining multiple models.
"""

from .ensemble_classifier import EnsembleClassifier
from .weighted import UncertaintyWeightedEnsemble, SimpleEnsemble, create_uncertainty_weighted_ensemble
from .physics_informed_ensemble import PhysicsInformedEnsemble, create_physics_informed_ensemble_from_config
from .registry import (
    make_model, get_model_info, list_available_models, 
    create_ensemble_members, create_resnet_vit_ensemble,
    create_physics_informed_ensemble, create_comprehensive_ensemble
)

__all__ = [
    'EnsembleClassifier',
    'UncertaintyWeightedEnsemble',
    'SimpleEnsemble',
    'create_uncertainty_weighted_ensemble',
    'PhysicsInformedEnsemble',
    'create_physics_informed_ensemble_from_config',
    'make_model',
    'get_model_info',
    'list_available_models',
    'create_ensemble_members',
    'create_resnet_vit_ensemble',
    'create_physics_informed_ensemble',
    'create_comprehensive_ensemble'
]