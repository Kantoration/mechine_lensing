#!/usr/bin/env python3
"""
Physics Capability Interface
============================

Defines interfaces for models that support physics-informed operations
for gravitational lensing detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Protocol, Optional, runtime_checkable, TypedDict
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PhysicsInfo(TypedDict, total=False):
    """
    Typed dictionary for physics information returned by physics-capable models.
    
    All fields are optional to allow flexibility in implementation.
    """
    physics_reg_loss: torch.Tensor
    attention_maps: Dict[str, torch.Tensor]
    physics_consistency: Dict[str, float]


@runtime_checkable
class PhysicsCapable(Protocol):
    """
    Protocol for models that can provide physics-informed analysis.
    
    Models implementing this protocol can return physics regularization
    losses and attention maps along with their predictions.
    
    This protocol is runtime-checkable, enabling isinstance() checks.
    """
    
    def forward_with_physics(self, x: torch.Tensor) -> Tuple[torch.Tensor, PhysicsInfo]:
        """
        Forward pass that returns both predictions and physics information.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of:
                - predictions: Model logits [B] (standardized shape and semantics)
                - physics_info: PhysicsInfo dictionary containing:
                    - 'physics_reg_loss': Physics regularization loss (Tensor)
                    - 'attention_maps': Dict of attention maps (optional)
                    - 'physics_consistency': Physics consistency metrics (optional)
        """
        ...
    
    @property
    def supports_physics_info(self) -> bool:
        """Return True if this model supports physics-informed analysis."""
        ...


class PhysicsAnalyzer(Protocol):
    """
    Protocol for physics analyzers that can extract physics information
    from model inputs and predictions.
    """
    
    def __call__(self, x: torch.Tensor, y_pred: torch.Tensor) -> PhysicsInfo:
        """
        Analyze physics properties of input and predictions.
        
        Args:
            x: Input tensor [B, C, H, W]
            y_pred: Model predictions/logits [B] or [B, 1]
            
        Returns:
            PhysicsInfo dictionary with analysis results
        """
        ...


class PhysicsInformedModule(nn.Module, ABC):
    """
    Abstract base class for physics-informed neural network modules.
    
    Provides a concrete implementation framework for models that need
    to incorporate gravitational lensing physics constraints.
    """
    
    def __init__(self):
        super().__init__()
        self._physics_weight = 0.1
        self._enable_physics_analysis = True
    
    @property
    def supports_physics_info(self) -> bool:
        """Return True since this is a physics-informed module."""
        return True
    
    @property
    def physics_weight(self) -> float:
        """Get the physics regularization weight."""
        return self._physics_weight
    
    @physics_weight.setter
    def physics_weight(self, value: float):
        """Set the physics regularization weight."""
        if value < 0:
            raise ValueError("Physics weight must be non-negative")
        self._physics_weight = value
    
    @property
    def enable_physics_analysis(self) -> bool:
        """Check if physics analysis is enabled."""
        return self._enable_physics_analysis
    
    @enable_physics_analysis.setter
    def enable_physics_analysis(self, value: bool):
        """Enable or disable physics analysis."""
        self._enable_physics_analysis = value
    
    @abstractmethod
    def forward_with_physics(self, x: torch.Tensor) -> Tuple[torch.Tensor, PhysicsInfo]:
        """
        Abstract method for physics-informed forward pass.
        
        Must be implemented by subclasses to provide physics information.
        Returns logits [B] and physics information.
        """
        pass
    
    @abstractmethod
    def forward_without_physics(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for fast forward pass without physics analysis.
        
        Must be implemented by subclasses for performance when physics
        analysis is not needed. Returns logits [B].
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass that conditionally runs physics analysis.
        
        Uses enable_physics_analysis flag to determine whether to run
        expensive physics computations.
        """
        if self._enable_physics_analysis:
            predictions, _ = self.forward_with_physics(x)
            return predictions
        else:
            return self.forward_without_physics(x)
    
    def get_physics_info(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Get only the physics information for input x.
        
        Args:
            x: Input tensor
            
        Returns:
            Physics information dictionary
        """
        _, physics_info = self.forward_with_physics(x)
        return physics_info
    
    def compute_physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute only the physics regularization loss.
        
        Args:
            x: Input tensor
            
        Returns:
            Physics regularization loss tensor (same device/dtype as input)
        """
        physics_info = self.get_physics_info(x)
        loss = physics_info.get('physics_reg_loss', None)
        if loss is None:
            return torch.zeros((), device=x.device, dtype=torch.float32)
        return torch.as_tensor(loss, device=x.device, dtype=torch.float32)


class PhysicsWrapper(nn.Module):
    """
    Wrapper to add physics capability to existing models.
    
    This allows existing models to be made physics-aware without
    modifying their original implementation.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        physics_analyzer: Optional[PhysicsAnalyzer] = None,
        physics_weight: float = 0.1
    ):
        """
        Initialize physics wrapper.
        
        Args:
            base_model: The base model to wrap
            physics_analyzer: Optional physics analyzer following PhysicsAnalyzer protocol
            physics_weight: Weight for physics regularization (applied to physics_reg_loss)
        """
        super().__init__()
        self.base_model = base_model
        self.physics_analyzer = physics_analyzer
        self.physics_weight = float(physics_weight)
    
    @property
    def supports_physics_info(self) -> bool:
        """Return True since this wrapper adds physics capability."""
        return True
    
    def forward_with_physics(self, x: torch.Tensor) -> Tuple[torch.Tensor, PhysicsInfo]:
        """
        Forward pass with physics analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (logits [B], physics_info)
        """
        # Get base model predictions (assume logits)
        predictions = self.base_model(x)
        
        # Initialize physics info with safe tensor creation
        physics_info: PhysicsInfo = {
            'physics_reg_loss': torch.zeros((), device=x.device, dtype=torch.float32)
        }
        
        # If physics analyzer is available, use it
        if self.physics_analyzer is not None:
            try:
                # Pass both input and predictions to analyzer
                analysis = self.physics_analyzer(x, predictions)
                
                # Apply physics weight to regularization loss if present
                if 'physics_reg_loss' in analysis:
                    scaled_loss = torch.as_tensor(
                        analysis['physics_reg_loss'], 
                        device=x.device, 
                        dtype=torch.float32
                    ) * self.physics_weight
                    analysis['physics_reg_loss'] = scaled_loss
                
                physics_info.update(analysis)
            except Exception as e:
                logger.warning("Physics analysis failed", exc_info=e)
        
        return predictions, physics_info
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.base_model(x)


def make_physics_capable(
    model: nn.Module, 
    physics_analyzer: Optional[PhysicsAnalyzer] = None,
    physics_weight: float = 0.1
) -> PhysicsCapable:
    """
    Convert a regular model to physics-capable.
    
    Args:
        model: Base model to convert
        physics_analyzer: Optional physics analyzer following PhysicsAnalyzer protocol
        physics_weight: Physics regularization weight
        
    Returns:
        Physics-capable model
    """
    if is_physics_capable(model):
        # Already physics-capable
        return model  # type: ignore
    else:
        # Wrap with physics capability
        return PhysicsWrapper(model, physics_analyzer, physics_weight)


def is_physics_capable(model: nn.Module) -> bool:
    """
    Check if a model supports physics-informed operations using structural typing.
    
    Args:
        model: Model to check
        
    Returns:
        True if model supports physics operations
    """
    # Use runtime-checkable protocol for structural typing
    if isinstance(model, PhysicsCapable):
        return True
    
    # Fallback to attribute checking for legacy compatibility
    return (hasattr(model, 'supports_physics_info') and 
            getattr(model, 'supports_physics_info', False))
