#!/usr/bin/env python3
"""
Physics-Informed Ensemble for Gravitational Lensing Detection
============================================================

This module implements ensemble methods that specifically leverage 
physics-informed attention mechanisms for improved gravitational
lensing detection.

Key Features:
- Integration of physics regularization losses
- Attention map visualization and analysis
- Physics-aware uncertainty estimation
- Adaptive weighting based on physics consistency
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import make_model, get_model_info, ModelContract
from ..interfaces.physics_capable import is_physics_capable, PhysicsInfo

logger = logging.getLogger(__name__)


class PhysicsInformedEnsemble(nn.Module):
    """
    Physics-informed ensemble that leverages gravitational lensing physics
    for improved detection and uncertainty estimation.
    
    This ensemble specifically handles:
    - Physics regularization losses from attention mechanisms
    - Physics-based confidence weighting
    - Attention map analysis for interpretability
    - Adaptive fusion based on physics consistency
    """
    
    def __init__(
        self,
        member_configs: List[Dict[str, Any]],
        physics_weight: float = 0.1,
        uncertainty_estimation: bool = True,
        attention_analysis: bool = True,
        physics_model_indicators: Optional[List[str]] = None,
        mc_samples: int = 10
    ):
        """
        Initialize physics-informed ensemble.
        
        Args:
            member_configs: List of member configuration dictionaries
            physics_weight: Weight for physics regularization losses
            uncertainty_estimation: Whether to estimate physics-based uncertainty
            attention_analysis: Whether to perform attention map analysis
            physics_model_indicators: List of strings to identify physics-informed models
                                     If None, defaults to ['enhanced_light_transformer']
            mc_samples: Number of Monte Carlo samples for uncertainty estimation
        """
        super().__init__()
        
        self.physics_weight = physics_weight
        self.uncertainty_estimation = uncertainty_estimation
        self.attention_analysis = attention_analysis
        
        # Configure physics model detection
        if physics_model_indicators is None:
            self.physics_model_indicators = ['enhanced_light_transformer']
        else:
            self.physics_model_indicators = physics_model_indicators
        
        # Monte Carlo sampling configuration
        self.mc_samples = mc_samples
        self.mc_dropout_p = 0.2  # Configurable MC dropout rate
        
        # Create ensemble members
        self.members = nn.ModuleList()
        self.member_names = []
        self.member_input_sizes = {}
        self.member_has_physics = {}
        
        for i, config in enumerate(member_configs):
            name = config['name']
            bands = config.get('bands', 3)
            pretrained = config.get('pretrained', True)
            dropout_p = config.get('dropout_p', 0.2)
            
            # Create model
            backbone, head, feature_dim = make_model(
                name=name,
                bands=bands,
                pretrained=pretrained,
                dropout_p=dropout_p
            )
            
            # Combine into single model
            model = nn.Sequential(backbone, head)
            self.members.append(model)
            
            # Store member metadata
            self.member_names.append(name)
            model_info = get_model_info(name)
            self.member_input_sizes[name] = model_info['input_size']
            
            # Check if member has physics-informed components
            # First check using capability interface, then fallback to name matching
            if is_physics_capable(model):
                self.member_has_physics[name] = True
            else:
                self.member_has_physics[name] = any(
                    indicator in name for indicator in self.physics_model_indicators
                )
            
            logger.info(f"Added ensemble member {i+1}/{len(member_configs)}: {name} "
                       f"(physics={self.member_has_physics[name]})")
        
        # Physics-aware weighting network
        if uncertainty_estimation:
            # Input: logits + uncertainties + physics_losses = 3 * num_members
            self.physics_weighting_net = nn.Sequential(
                nn.Linear(len(member_configs) * 3, 64),  # logits + uncertainties + physics features
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, len(member_configs)),
                nn.Softmax(dim=-1)
            )
        
        logger.info(f"Created physics-informed ensemble with {len(member_configs)} members")
    
    @staticmethod
    def _resize_maps(
        sample: Dict[str, torch.Tensor],
        target_h: int,
        target_w: int,
        contract: ModelContract
    ) -> Dict[str, torch.Tensor]:
        """
        Area-preserving resize for physics maps (κ, ψ, α).
        
        References:
        - Gruen, D., & Brimioulle, F. (2015). "Cluster lensing in the CLASH survey." ApJS.
        - Bosch, J. et al. (2018). "The Hyper Suprime-Cam software pipeline." PASJ.
          Best practice from cosmological weak lensing: mass conservation in coarse-graining
          requires average pooling for κ, never bilinear; compute α from ψ after new pool.
        
        Rules:
        - κ, ψ: Use adaptive average pooling (area-preserving, mass-conserving)
        - α: Recompute from ψ via gradient2d at new scale (preserves α-ψ consistency)
          OR average-pool if no ψ available
        
        Args:
            sample: Dict with 'kappa', 'psi', 'alpha' keys
            target_h, target_w: Target spatial dimensions
            contract: ModelContract with dx/dy for gradient computation
            
        Returns:
            Dict with resized maps
        """
        from mlensing.gnn.physics_ops import gradient2d
        
        out = {}
        
        # Resize κ using area-preserving pooling
        if 'kappa' in sample:
            kappa = sample['kappa']
            if kappa.ndim == 3:  # [C, H, W]
                kappa = kappa.unsqueeze(0)  # Add batch dim
            out['kappa'] = F.adaptive_avg_pool2d(kappa, (target_h, target_w)).squeeze(0)
        
        # Resize ψ and recompute α from resized ψ
        if 'psi' in sample:
            psi = sample['psi']
            if psi.ndim == 3:
                psi = psi.unsqueeze(0)
            psi_resized = F.adaptive_avg_pool2d(psi, (target_h, target_w)).squeeze(0)
            out['psi'] = psi_resized
            
            # Recompute α from resized ψ (preserves α-ψ consistency)
            if contract.dx is not None and contract.dy is not None:
                # Compute scale factor for new grid
                H_old, W_old = psi.shape[-2:] if psi.ndim == 3 else sample['psi'].shape[-2:]
                scale_x = contract.dx * (W_old / target_w)
                scale_y = contract.dy * (H_old / target_h)
                
                # Add batch and channel dims for gradient2d
                psi_4d = psi_resized.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                gx, gy = gradient2d(psi_4d, dx=scale_x, dy=scale_y)
                out['alpha_from_psi'] = torch.cat([gx, gy], dim=1).squeeze(0)  # [2, H, W]
        
        # Direct α resize (fallback if no ψ)
        if 'alpha' in sample and 'psi' not in sample:
            alpha = sample['alpha']
            if alpha.ndim == 3:  # [2, H, W]
                alpha = alpha.unsqueeze(0)  # [1, 2, H, W]
            # Average-pool each component independently
            out['alpha'] = F.adaptive_avg_pool2d(alpha, (target_h, target_w)).squeeze(0)
        
        return out
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Forward pass through physics-informed ensemble.
        
        Args:
            inputs: Dictionary of {model_name: input_tensor} pairs
            
        Returns:
            Dictionary containing:
                - 'prediction': Final ensemble predictions [B] (probabilities)
                - 'ensemble_logit': Fused ensemble logits [B] 
                - 'member_logits': Individual member logits [B, M]
                - 'member_predictions': Individual member predictions [B, M] (probabilities)
                - 'member_uncertainties': Uncertainty estimates [B, M]
                - 'ensemble_weights': Ensemble fusion weights [B, M]
                - 'physics_loss': Total physics regularization loss (scalar)
                - 'member_physics_losses': Per-member physics losses [B, M]
                - 'attention_maps': Attention visualizations (if enabled)
        """
        batch_size = next(iter(inputs.values())).size(0)
        device = next(iter(inputs.values())).device
        
        # Collect logits, uncertainties, and physics information
        member_logits = []
        logit_uncertainties = []
        physics_losses = []
        attention_maps = {}
        
        # Helper to create tensor fingerprint for cache isolation
        # References:
        # [1] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). "Simple and Scalable Predictive
        #     Uncertainty Estimation using Deep Ensembles." NeurIPS. Ensembles require controlled
        #     member routing and independent stochasticity per member.
        # [2] Breiman, L. (1996). "Bagging predictors." Machine Learning.
        def _tensor_fingerprint(t: torch.Tensor) -> Tuple[int, int, int, int]:
            """Create immutable fingerprint: batch, channels, height, width."""
            return (t.shape[0], t.shape[1], t.shape[2], t.shape[3])
        
        # Cache for resized tensors - keyed by member + target size + fingerprint
        # This prevents wrong tensor reuse across different members/batches, which would undermine
        # uncertainty estimation, calibration, and fusion quality. See [1][2] for ensemble correctness.
        resized_cache: Dict[Tuple[str, int, int, Tuple[int, int, int, int]], torch.Tensor] = {}
        
        for i, (name, model) in enumerate(zip(self.member_names, self.members)):
            # Get model input - enforce explicit routing
            if name not in inputs:
                raise KeyError(f"Missing input for ensemble member '{name}'. "
                             f"Available inputs: {list(inputs.keys())}. "
                             f"Ensure all ensemble members have corresponding inputs.")
            
            x = inputs[name]
            
            # Resize if needed, with robust size handling
            target_size = self.member_input_sizes[name]
            if isinstance(target_size, (tuple, list)):
                target_h, target_w = target_size
            else:
                target_h = target_w = int(target_size)
            
            if x.shape[-2:] != (target_h, target_w):
                # Cache key includes member name and tensor fingerprint to prevent cross-member reuse
                cache_key = (name, target_h, target_w, _tensor_fingerprint(x))
                if cache_key not in resized_cache:
                    resized_cache[cache_key] = torch.nn.functional.interpolate(
                        x, size=(target_h, target_w), 
                        mode='bilinear', align_corners=False, antialias=True
                    )
                x = resized_cache[cache_key]
            
            # Forward pass through model - collect LOGITS
            if self.member_has_physics[name]:
                # Physics-informed model
                if hasattr(model, 'forward_with_physics_logits'):
                    # Use physics-capable interface for logits
                    logits, extra_info = model.forward_with_physics_logits(x)
                elif hasattr(model, 'forward_with_physics'):
                    # Fallback: assume forward_with_physics returns logits (needs verification)
                    logits, extra_info = model.forward_with_physics(x)
                else:
                    # Custom extraction - returns logits now
                    logits, extra_info = self._forward_physics_logits(model, x)
                
                loss_val = extra_info.get('physics_reg_loss', None)
                if loss_val is None:
                    loss_tensor = torch.zeros([], device=device, dtype=torch.float32)
                else:
                    loss_tensor = torch.as_tensor(loss_val, device=device, dtype=torch.float32)
                
                # Normalize to [B] shape for consistent per-sample handling
                if loss_tensor.dim() == 0:
                    loss_vec = loss_tensor.expand(batch_size)  # [B]
                elif loss_tensor.dim() == 1 and loss_tensor.size(0) == batch_size:
                    loss_vec = loss_tensor  # [B]
                else:
                    raise ValueError(f"physics_reg_loss must be scalar or shape [B], got {loss_tensor.shape}")
                
                physics_losses.append(loss_vec)
                if self.attention_analysis:
                    attention_maps[name] = extra_info.get('attention_maps', {})
            else:
                # Standard model - get logits (no sigmoid)
                logits = model(x)
                # Add zero physics loss for standard models (expanded to [B])
                physics_losses.append(torch.zeros(batch_size, device=device, dtype=torch.float32))
            
            # Safe tensor flattening to [batch_size] - logits
            member_logits.append(self._safe_flatten_prediction(logits))
            
            # Estimate uncertainty using Monte Carlo dropout if enabled - on LOGITS
            if self.uncertainty_estimation:
                logit_uncertainty = self._estimate_uncertainty_logits(model, x, num_samples=self.mc_samples)
                logit_uncertainties.append(logit_uncertainty)
        
        # Stack logits and uncertainties
        logits = torch.stack(member_logits, dim=1)  # [B, num_members]
        if logit_uncertainties:
            uncertainties = torch.stack(logit_uncertainties, dim=1)  # [B, num_members]
        else:
            uncertainties = torch.full_like(logits, 0.1)
        
        # Stack physics losses to [B, M] shape
        member_physics_losses = torch.stack(physics_losses, dim=1)  # [B, M]
        
        # Physics-aware ensemble fusion in logit space
        if self.uncertainty_estimation:
            weights = self._compute_physics_weights_logits(logits, uncertainties, member_physics_losses)
        else:
            weights = torch.ones(batch_size, len(self.members), device=device) / len(self.members)
        
        # Weighted ensemble fusion in logit space
        fused_logit = torch.sum(logits * weights, dim=1)  # [B]
        
        # Clamp fused logits for numerical stability
        fused_logit = fused_logit.clamp(-40, 40)
        
        # Apply sigmoid only once at the end
        ensemble_pred = torch.sigmoid(fused_logit).clamp(1e-6, 1-1e-6)
        
        # Aggregate physics losses (safe tensor operations)
        total_physics_loss = member_physics_losses.mean(dim=0).sum() * self.physics_weight
        
        # Prepare output
        output = {
            'prediction': ensemble_pred,
            'ensemble_logit': fused_logit,
            'member_logits': logits,
            'member_predictions': torch.sigmoid(logits).clamp(1e-6, 1-1e-6),  # For backward compatibility
            'member_uncertainties': uncertainties,
            'ensemble_weights': weights,
            'physics_loss': total_physics_loss,
            'member_physics_losses': member_physics_losses  # [B, M]
        }
        
        if self.attention_analysis:
            output['attention_maps'] = attention_maps
        
        return output
    
    def _forward_physics_model(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through physics-informed model with extra information extraction."""
        # Check if model has enhanced transformer backbone with physics attention
        if hasattr(model, '__getitem__') and len(model) >= 1:
            backbone = model[0]  # First element should be backbone
            head = model[1] if len(model) > 1 else None
            
            # Check if backbone has physics-informed attention
            if hasattr(backbone, 'transformer_blocks'):
                # Enhanced Light Transformer with physics attention
                features = backbone(x)
                
                # Extract physics information from transformer blocks
                physics_reg_loss = torch.tensor(0.0, device=x.device)
                attention_maps = {}
                
                # Collect physics regularization losses from attention mechanisms
                for i, block in enumerate(backbone.transformer_blocks):
                    if hasattr(block, 'attention') and hasattr(block.attention, 'forward'):
                        # Try to extract physics information from attention
                        try:
                            # This assumes the attention mechanism returns physics info
                            # In practice, you'd need to modify the attention forward method
                            if hasattr(block.attention, 'get_physics_info'):
                                block_physics_info = block.attention.get_physics_info()
                                if 'physics_reg_loss' in block_physics_info:
                                    physics_reg_loss += block_physics_info['physics_reg_loss']
                                if 'attention_maps' in block_physics_info:
                                    attention_maps[f'block_{i}'] = block_physics_info['attention_maps']
                        except Exception as e:
                            logger.debug(f"Could not extract physics info from block {i}: {e}")
                
                # Apply classification head if present
                if head is not None:
                    pred = torch.sigmoid(head(features))
                else:
                    pred = torch.sigmoid(features)
                
                extra_info = {
                    'physics_reg_loss': physics_reg_loss,
                    'attention_maps': attention_maps
                }
                
                return pred, extra_info
        
        # Fallback: standard forward pass for non-physics models
        pred = torch.sigmoid(model(x))
        
        extra_info = {
            'physics_reg_loss': torch.tensor(0.0, device=x.device),
            'attention_maps': {}
        }
        
        return pred, extra_info
    
    def _forward_physics_logits(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, PhysicsInfo]:
        """
        Forward pass through physics-informed model returning logits.
        
        This is the preferred method as it returns logits instead of probabilities,
        enabling proper logit-space ensemble fusion.
        """
        # Check if model has enhanced transformer backbone with physics attention
        if hasattr(model, '__getitem__') and len(model) >= 1:
            backbone = model[0]  # First element should be backbone
            head = model[1] if len(model) > 1 else None
            
            # Check if backbone has physics-informed attention
            if hasattr(backbone, 'transformer_blocks'):
                # Enhanced Light Transformer with physics attention
                features = backbone(x)
                
                # Extract physics information from transformer blocks
                physics_reg_loss = torch.zeros([], device=x.device, dtype=torch.float32)
                attention_maps = {}
                
                # Collect physics regularization losses from attention mechanisms
                for i, block in enumerate(getattr(backbone, 'transformer_blocks', [])):
                    if hasattr(block, 'attention') and hasattr(block.attention, 'get_physics_info'):
                        try:
                            # Extract physics information from attention
                            block_physics_info = block.attention.get_physics_info()
                            if 'physics_reg_loss' in block_physics_info:
                                loss_val = block_physics_info['physics_reg_loss']
                                physics_reg_loss = physics_reg_loss + torch.as_tensor(
                                    loss_val, device=x.device, dtype=torch.float32
                                )
                            if 'attention_maps' in block_physics_info:
                                attention_maps[f'block_{i}'] = block_physics_info['attention_maps']
                        except Exception as e:
                            logger.debug(f"Could not extract physics info from block {i}: {e}")
                
                # Apply classification head if present - return LOGITS
                if head is not None:
                    logits = head(features)  # No sigmoid here!
                else:
                    logits = features  # Assume features are logit-ready
                
                extra_info: PhysicsInfo = {
                    'physics_reg_loss': physics_reg_loss,
                    'attention_maps': attention_maps
                }
                
                return logits, extra_info
        
        # Fallback: standard forward pass for non-physics models - return LOGITS
        logits = model(x)  # No sigmoid here!
        
        extra_info: PhysicsInfo = {
            'physics_reg_loss': torch.zeros([], device=x.device, dtype=torch.float32),
            'attention_maps': {}
        }
        
        return logits, extra_info
    
    # Removed _estimate_uncertainty - use _estimate_uncertainty_logits instead
    
    def _enable_mc_dropout(self, model: nn.Module):
        """
        Enable dropout layers while keeping BatchNorm in eval mode.
        
        This is required for proper Monte Carlo dropout: dropout must be active
        to create stochasticity, but BN must remain frozen to preserve running stats.
        
        References:
        - Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation." ICML.
        - Kendall, A., & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning
          for Computer Vision?" NeurIPS.
        
        Note: Simply applying functional dropout to logits (without enabling internal dropout)
        is a placebo that doesn't capture epistemic uncertainty. True MC-dropout requires
        activating all internal Dropout layers while keeping BN in eval mode.
        """
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()  # Activate dropout
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()   # Keep BN frozen to avoid running stats drift
    
    def _estimate_uncertainty_logits(self, model: nn.Module, x: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """
        Estimate predictive uncertainty using Monte Carlo dropout on logits.
        
        This properly enables internal dropout layers (not just logit-level dropout)
        while keeping BatchNorm frozen, enabling true epistemic uncertainty estimation.
        
        Args:
            model: Model to estimate uncertainty for
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Standard deviation of logits across MC samples [batch_size]
        """
        prev_mode = model.training
        model.eval()  # Start in eval mode to freeze BN stats
        self._enable_mc_dropout(model)  # Enable dropout layers for stochasticity
        
        logit_samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                # Forward pass with dropout active (stochastic internal activations)
                logits = model(x)
                logit_samples.append(self._safe_flatten_prediction(logits))
        
        logit_samples = torch.stack(logit_samples, dim=0)  # [num_samples, batch_size]
        logit_uncertainty = torch.std(logit_samples, dim=0, unbiased=False)  # [batch_size]
        
        # Restore original training mode
        model.train(prev_mode)
        return logit_uncertainty
    
    def _compute_physics_weights(self, predictions: torch.Tensor, uncertainties: torch.Tensor) -> torch.Tensor:
        """Compute physics-aware ensemble weights."""
        if hasattr(self, 'physics_weighting_net'):
            # Combine predictions and uncertainties
            features = torch.cat([predictions, uncertainties], dim=-1)  # [B, 2*num_members]
            weights = self.physics_weighting_net(features)  # [B, num_members]
        else:
            # Simple inverse-uncertainty weighting with stability
            eps = 1e-3
            u_clamped = uncertainties.clamp_min(eps)
            inv_uncertainties = (1.0 / u_clamped).clamp(1e-6, 1e6)
            weights = inv_uncertainties / torch.sum(inv_uncertainties, dim=1, keepdim=True)
        
        return weights
    
    def _compute_physics_weights_logits(
        self, 
        logits: torch.Tensor, 
        uncertainties: torch.Tensor,
        physics_losses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics-aware ensemble weights using logits and actual physics features.
        
        Args:
            logits: Member logits [B, M]
            uncertainties: Logit uncertainties [B, M]  
            physics_losses: Physics losses [B, M] (per-sample, per-member)
            
        Returns:
            Ensemble weights [B, M]
        """
        B, M = logits.shape
        eps = 1e-3
        
        # Clamp uncertainties for numerical stability
        u_clamped = uncertainties.clamp_min(eps)
        
        if hasattr(self, 'physics_weighting_net'):
            # Use neural network for physics-aware weighting
            # Features: logits + uncertainties + physics losses
            features = torch.cat([logits, u_clamped, physics_losses], dim=-1)  # [B, 3*M]
            # Optionally detach features to prevent gradients from flowing to weighting network
            # features = features.detach()
            weights = self.physics_weighting_net(features)
        else:
            # Fallback: inverse-uncertainty weighting with physics penalty
            # Higher physics loss → lower weight
            physics_penalty = (1.0 / (1.0 + physics_losses)).clamp_(1e-3, 1e3)  # [B, M]
            inv_uncertainties = (1.0 / u_clamped).clamp_(1e-6, 1e6) * physics_penalty
            weights = inv_uncertainties / inv_uncertainties.sum(dim=1, keepdim=True)
        
        return weights
    
    def get_physics_analysis(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Perform detailed physics analysis of ensemble predictions.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary containing physics analysis results
        """
        with torch.no_grad():
            output = self.forward(inputs)
        
        analysis = {
            'ensemble_prediction': output['prediction'].cpu().numpy(),
            'ensemble_logit': output['ensemble_logit'].cpu().numpy(),
            'member_logits': output['member_logits'].cpu().numpy(),
            'member_predictions': output['member_predictions'].cpu().numpy(),
            'member_uncertainties': output['member_uncertainties'].cpu().numpy(),
            'member_physics_losses': output['member_physics_losses'].cpu().numpy(),
            'ensemble_weights': output['ensemble_weights'].cpu().numpy(),
            'physics_loss': output['physics_loss'].item()
        }
        
        if 'attention_maps' in output:
            analysis['attention_maps'] = {
                name: {k: v.cpu().numpy() for k, v in maps.items()}
                for name, maps in output['attention_maps'].items()
            }
        
        # Physics consistency metrics
        analysis['physics_consistency'] = self._compute_physics_consistency(output)
        
        return analysis
    
    def _compute_physics_consistency(self, output: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute physics consistency metrics for the ensemble."""
        predictions = output['member_predictions']  # These are probabilities for consistency analysis
        
        # Compute prediction variance as a measure of consistency
        pred_variance = torch.var(predictions, dim=1).mean().item()
        
        # Compute correlation between physics-informed and traditional models
        physics_indices = [i for i, name in enumerate(self.member_names) 
                          if self.member_has_physics[name]]
        traditional_indices = [i for i, name in enumerate(self.member_names) 
                              if not self.member_has_physics[name]]
        
        if physics_indices and traditional_indices:
            physics_preds = predictions[:, physics_indices].mean(dim=1)
            traditional_preds = predictions[:, traditional_indices].mean(dim=1)
            
            # Compute safe correlation (handle constant vectors)
            correlation = self._safe_correlation(physics_preds, traditional_preds)
        else:
            correlation = 1.0
        
        return {
            'prediction_variance': pred_variance,
            'physics_traditional_correlation': correlation,
            'physics_loss': output['physics_loss'].item()
        }
    
    def _safe_correlation(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
        """
        Compute correlation safely, handling constant vectors and numerical issues.
        
        Args:
            a, b: Input tensors
            eps: Small value to prevent division by zero
            
        Returns:
            Correlation coefficient (float)
        """
        # Center the vectors
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        # Compute standard deviations
        a_std = a_centered.std()
        b_std = b_centered.std()
        
        # Handle constant vectors (std ≈ 0)
        denominator = (a_std * b_std).clamp_min(eps)
        
        # Compute correlation
        correlation = (a_centered * b_centered).mean() / denominator
        
        return correlation.item()
    
    def _safe_flatten_prediction(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Safely flatten prediction tensor to [batch_size] shape.
        
        Handles various output shapes: [B], [B,1], [B,1,1], etc.
        """
        if tensor.dim() == 1:
            return tensor  # Already [B]
        elif tensor.dim() == 2 and tensor.size(1) == 1:
            return tensor.squeeze(1)  # [B,1] -> [B]
        else:
            # General case: flatten to [B] 
            return tensor.view(tensor.size(0), -1).squeeze(1)


def create_physics_informed_ensemble_from_config(config_path: str) -> PhysicsInformedEnsemble:
    """
    Create physics-informed ensemble from configuration file.
    
    Args:
        config_path: Path to ensemble configuration YAML file
        
    Returns:
        Configured PhysicsInformedEnsemble instance
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ensemble_config = config.get('ensemble', {})
    member_configs = config.get('members', [])
    
    return PhysicsInformedEnsemble(
        member_configs=member_configs,
        physics_weight=ensemble_config.get('physics_weight', 0.1),
        uncertainty_estimation=ensemble_config.get('uncertainty_estimation', True),
        attention_analysis=ensemble_config.get('attention_analysis', True),
        physics_model_indicators=ensemble_config.get('physics_model_indicators'),
        mc_samples=ensemble_config.get('mc_samples', 10)
    )
