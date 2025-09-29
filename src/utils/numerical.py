#!/usr/bin/env python3
"""
Numerical stability utilities for deep learning.

Provides robust implementations of common operations that can cause
numerical instabilities in production ML systems.
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

def clamp_probs(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Clamp probabilities to prevent log(0) and other numerical issues.
    
    Args:
        probs: Probability tensor
        eps: Small epsilon for clamping bounds
        
    Returns:
        Clamped probabilities in [eps, 1-eps]
    """
    return torch.clamp(probs, min=eps, max=1.0 - eps)

def clamp_variances(variances: torch.Tensor, min_var: float = 1e-3, max_var: float = 1e3) -> torch.Tensor:
    """
    Clamp variances to prevent numerical instabilities in inverse-variance weighting.
    
    Args:
        variances: Variance tensor
        min_var: Minimum allowed variance
        max_var: Maximum allowed variance
        
    Returns:
        Clamped variances in [min_var, max_var]
    """
    return torch.clamp(variances, min=min_var, max=max_var)

def stable_log_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log-sigmoid computation.
    
    Args:
        logits: Input logits
        
    Returns:
        log(sigmoid(logits)) computed stably
    """
    # Use the identity: log(sigmoid(x)) = -softplus(-x)
    return -torch.nn.functional.softplus(-logits)

def stable_log_one_minus_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log(1 - sigmoid(x)) computation.
    
    Args:
        logits: Input logits
        
    Returns:
        log(1 - sigmoid(logits)) computed stably
    """
    # Use the identity: log(1 - sigmoid(x)) = -softplus(x)
    return -torch.nn.functional.softplus(logits)

def inverse_variance_weights(
    variances: torch.Tensor, 
    eps: float = 1e-3,
    max_weight_ratio: float = 1e3
) -> torch.Tensor:
    """
    Compute numerically stable inverse-variance weights.
    
    Args:
        variances: Variance tensor of shape [num_members, batch_size, ...]
        eps: Minimum variance for stability
        max_weight_ratio: Maximum ratio between largest and smallest weight
        
    Returns:
        Normalized weights that sum to 1 along the first dimension
    """
    # Clamp variances for stability
    safe_vars = torch.clamp(variances, min=eps)
    
    # Compute inverse variances with clamping
    inv_vars = 1.0 / safe_vars
    inv_vars = torch.clamp(inv_vars, max=1.0 / eps * max_weight_ratio)
    
    # Normalize to sum to 1
    weights = inv_vars / inv_vars.sum(dim=0, keepdim=True)
    
    return weights

def ensemble_logit_fusion(
    logits_list: list[torch.Tensor],
    variances_list: list[torch.Tensor],
    temperatures: Optional[list[float]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse ensemble predictions in logit space using inverse-variance weighting.
    
    This is the correct approach for ensemble fusion as it assumes Gaussian
    distributions in logit space rather than probability space.
    
    Args:
        logits_list: List of logit tensors from ensemble members
        variances_list: List of variance tensors (epistemic uncertainty)
        temperatures: Optional per-member temperature scaling
        
    Returns:
        Tuple of (fused_logits, fused_variance)
    """
    if len(logits_list) != len(variances_list):
        raise ValueError("Number of logits and variances must match")
    
    # Apply temperature scaling if provided
    if temperatures is not None:
        if len(temperatures) != len(logits_list):
            raise ValueError("Number of temperatures must match number of members")
        logits_list = [logits / temp for logits, temp in zip(logits_list, temperatures)]
    
    # Stack tensors: [num_members, batch_size, ...]
    logits_stack = torch.stack(logits_list, dim=0)
    vars_stack = torch.stack(variances_list, dim=0)
    
    # Compute inverse-variance weights
    weights = inverse_variance_weights(vars_stack)
    
    # Fuse in logit space
    fused_logits = (weights * logits_stack).sum(dim=0)
    
    # Fused variance (inverse of sum of inverse variances)
    inv_vars = 1.0 / torch.clamp(vars_stack, min=1e-3)
    fused_variance = 1.0 / inv_vars.sum(dim=0)
    
    return fused_logits, fused_variance
