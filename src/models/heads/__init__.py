"""
Classification heads for gravitational lens classification.
"""

from .binary import BinaryHead, MultiClassHead, create_binary_head, create_multiclass_head
from .aleatoric import AleatoricBinaryHead, AleatoricLoss, create_aleatoric_head, analyze_aleatoric_uncertainty

__all__ = [
    'BinaryHead',
    'MultiClassHead', 
    'create_binary_head',
    'create_multiclass_head',
    'AleatoricBinaryHead',
    'AleatoricLoss',
    'create_aleatoric_head',
    'analyze_aleatoric_uncertainty'
]
