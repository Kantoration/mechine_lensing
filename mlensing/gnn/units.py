from __future__ import annotations

from typing import Dict, Any
import torch

Tensor = torch.Tensor


def enforce_sigma_crit_policy(kappa: Tensor, meta: Dict[str, Any], policy: str) -> Tensor:
    """
    Enforce Σcrit (critical surface density) normalization policy.
    
    References:
    - Gruen, D., & Brimioulle, F. (2015). "Cluster lensing in the CLASH survey." ApJS.
    - Bosch, J. et al. (2018). "The Hyper Suprime-Cam software pipeline." PASJ.
      Cluster lensing requires explicit Σcrit handling and pixel-to-radian propagation
      for physical mapping and cross-survey compatibility.
    
    This ensures consistent units across models:
    - 'dimensionless': κ is already in dimensionless units (Σ/Σcrit = 1)
    - 'physical': κ is in physical units, multiply by sigma_crit to normalize
    
    Args:
        kappa: Convergence map [B, 1, H, W] or [H, W]
        meta: Metadata dict containing 'sigma_crit' if policy='physical'
        policy: 'dimensionless' or 'physical'
        
    Returns:
        Normalized κ map in dimensionless units
        
    Raises:
        ValueError: If policy is unknown or sigma_crit missing for physical policy
    """
    if policy == 'dimensionless':
        # κ is already dimensionless (no conversion needed)
        # Verify sigma_crit is 1.0 if present (sanity check)
        if 'sigma_crit' in meta:
            sigma_crit = meta['sigma_crit']
            if isinstance(sigma_crit, (float, int)):
                if abs(sigma_crit - 1.0) > 1e-6:
                    raise ValueError(
                        f"dimensionless policy requires sigma_crit=1.0, got {sigma_crit}"
                    )
        return kappa
    
    elif policy == 'physical':
        # Convert from physical units (Σ) to dimensionless (Σ/Σcrit)
        if 'sigma_crit' not in meta:
            raise ValueError(
                "sigma_crit missing in meta for physical policy. "
                "Required for conversion: κ_physical → κ_dimensionless"
            )
        
        sigma_crit = meta['sigma_crit']
        if isinstance(sigma_crit, torch.Tensor):
            # Ensure broadcastable shape
            while sigma_crit.ndim < kappa.ndim:
                sigma_crit = sigma_crit.unsqueeze(-1)
        else:
            sigma_crit = float(sigma_crit)
        
        # κ_dimensionless = κ_physical / Σcrit
        return kappa / sigma_crit
    
    else:
        raise ValueError(f"Unknown Σcrit policy: {policy}. Must be 'dimensionless' or 'physical'")

