"""
Analysis utilities for gravitational lens classification.

This module provides post-hoc analysis tools for uncertainty quantification,
active learning, and model diagnostics without requiring trainable parameters.
"""

from .aleatoric import (
    AleatoricIndicators,
    compute_indicators,
    compute_indicators_with_targets,
    tta_indicators,
    selection_scores,
    topk_indices,
    ensemble_disagreement
)

__all__ = [
    'AleatoricIndicators',
    'compute_indicators', 
    'compute_indicators_with_targets',
    'tta_indicators',
    'selection_scores',
    'topk_indices',
    'ensemble_disagreement'
]
