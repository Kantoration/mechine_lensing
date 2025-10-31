"""
Model interfaces for physics-informed gravitational lensing detection.
"""

from .physics_capable import (
    PhysicsInfo,
    PhysicsCapable,
    PhysicsAnalyzer,
    PhysicsInformedModule,
    PhysicsWrapper,
    make_physics_capable,
    is_physics_capable,
)

__all__ = [
    "PhysicsInfo",
    "PhysicsCapable",
    "PhysicsAnalyzer",
    "PhysicsInformedModule",
    "PhysicsWrapper",
    "make_physics_capable",
    "is_physics_capable",
]
