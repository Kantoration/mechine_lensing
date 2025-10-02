#!/usr/bin/env python3
"""
realistic_lens_models.py
========================
Support for realistic lens models beyond point mass approximation.

Key Features:
- Singular Isothermal Ellipsoid (SIE) model
- Navarro-Frenk-White (NFW) profile
- Composite lens models
- Critical curve and caustic computation
- Realistic deflection angle calculations

Usage:
    from validation.realistic_lens_models import SIELensModel, NFWLensModel, CompositeLensModel
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import optimize, special
from scipy.integrate import quad

logger = logging.getLogger(__name__)


class SIELensModel:
    """
    Singular Isothermal Ellipsoid (SIE) lens model.
    
    This is the most commonly used model for galaxy-scale lenses,
    providing a good balance between realism and computational tractability.
    """
    
    def __init__(
        self,
        einstein_radius: float,
        ellipticity: float = 0.0,
        position_angle: float = 0.0,
        center: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        Initialize SIE lens model.
        
        Args:
            einstein_radius: Einstein radius in arcsec
            ellipticity: Ellipticity (0 = circular, 1 = highly elliptical)
            position_angle: Position angle in radians
            center: Lens center (x, y) in arcsec
        """
        self.einstein_radius = einstein_radius
        self.ellipticity = ellipticity
        self.position_angle = position_angle
        self.center = center
        
        # Precompute rotation matrix
        cos_pa = np.cos(position_angle)
        sin_pa = np.sin(position_angle)
        self.rotation_matrix = np.array([[cos_pa, -sin_pa], [sin_pa, cos_pa]])
        self.inv_rotation_matrix = np.array([[cos_pa, sin_pa], [-sin_pa, cos_pa]])
        
        logger.info(f"SIE lens model: θ_E={einstein_radius:.3f}, e={ellipticity:.3f}")
    
    def deflection_angle(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute deflection angles for SIE model.
        
        Args:
            x, y: Image plane coordinates in arcsec
            
        Returns:
            Tuple of (α_x, α_y) deflection angles
        """
        # Translate to lens center
        x_centered = x - self.center[0]
        y_centered = y - self.center[1]
        
        # Rotate to principal axes
        coords = np.stack([x_centered, y_centered], axis=-1)
        coords_rot = np.dot(coords, self.rotation_matrix.T)
        x_rot, y_rot = coords_rot[..., 0], coords_rot[..., 1]
        
        # Compute deflection in principal axes
        q = 1 - self.ellipticity  # Axis ratio
        r = np.sqrt(x_rot**2 + (y_rot/q)**2)
        
        # Avoid division by zero
        r = np.maximum(r, 1e-10)
        
        # SIE deflection formula
        alpha_x_rot = self.einstein_radius * x_rot / r
        alpha_y_rot = self.einstein_radius * y_rot / (q * r)
        
        # Rotate back to image plane
        alpha_rot = np.stack([alpha_x_rot, alpha_y_rot], axis=-1)
        alpha = np.dot(alpha_rot, self.inv_rotation_matrix.T)
        
        return alpha[..., 0], alpha[..., 1]
    
    def convergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute convergence (surface mass density) for SIE model.
        
        Args:
            x, y: Image plane coordinates in arcsec
            
        Returns:
            Convergence κ
        """
        # Translate and rotate
        x_centered = x - self.center[0]
        y_centered = y - self.center[1]
        
        coords = np.stack([x_centered, y_centered], axis=-1)
        coords_rot = np.dot(coords, self.rotation_matrix.T)
        x_rot, y_rot = coords_rot[..., 0], coords_rot[..., 1]
        
        # SIE convergence
        q = 1 - self.ellipticity
        r = np.sqrt(x_rot**2 + (y_rot/q)**2)
        r = np.maximum(r, 1e-10)
        
        kappa = self.einstein_radius / (2 * r)
        
        return kappa
    
    def critical_curve(self, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute critical curve for SIE model.
        
        Args:
            resolution: Number of points along critical curve
            
        Returns:
            Tuple of (x_crit, y_crit) critical curve coordinates
        """
        # For SIE, critical curve is approximately elliptical
        q = 1 - self.ellipticity
        
        # Generate elliptical critical curve
        theta = np.linspace(0, 2*np.pi, resolution)
        r_crit = self.einstein_radius
        
        x_ellipse = r_crit * np.cos(theta)
        y_ellipse = r_crit * q * np.sin(theta)
        
        # Rotate and translate
        coords = np.stack([x_ellipse, y_ellipse], axis=-1)
        coords_rot = np.dot(coords, self.rotation_matrix.T)
        
        x_crit = coords_rot[:, 0] + self.center[0]
        y_crit = coords_rot[:, 1] + self.center[1]
        
        return x_crit, y_crit
    
    def caustic(self, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute caustic curve for SIE model.
        
        Args:
            resolution: Number of points along caustic
            
        Returns:
            Tuple of (x_caustic, y_caustic) caustic coordinates
        """
        # Get critical curve
        x_crit, y_crit = self.critical_curve(resolution)
        
        # Compute deflection angles at critical curve
        alpha_x, alpha_y = self.deflection_angle(x_crit, y_crit)
        
        # Caustic is source plane image of critical curve
        x_caustic = x_crit - alpha_x
        y_caustic = y_crit - alpha_y
        
        return x_caustic, y_caustic


class NFWLensModel:
    """
    Navarro-Frenk-White (NFW) lens model for cluster-scale lenses.
    
    This model is more realistic for massive galaxy clusters
    but computationally more expensive than SIE.
    """
    
    def __init__(
        self,
        scale_radius: float,
        concentration: float = 5.0,
        center: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        Initialize NFW lens model.
        
        Args:
            scale_radius: Scale radius in arcsec
            concentration: Concentration parameter
            center: Lens center (x, y) in arcsec
        """
        self.scale_radius = scale_radius
        self.concentration = concentration
        self.center = center
        
        # Compute virial radius
        self.virial_radius = concentration * scale_radius
        
        # Precompute normalization
        self._compute_normalization()
        
        logger.info(f"NFW lens model: r_s={scale_radius:.3f}, c={concentration:.3f}")
    
    def _compute_normalization(self):
        """Compute normalization constant for NFW profile."""
        c = self.concentration
        
        # NFW normalization
        self.norm_factor = 1.0 / (np.log(1 + c) - c / (1 + c))
    
    def deflection_angle(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute deflection angles for NFW model.
        
        Args:
            x, y: Image plane coordinates in arcsec
            
        Returns:
            Tuple of (α_x, α_y) deflection angles
        """
        # Translate to lens center
        x_centered = x - self.center[0]
        y_centered = y - self.center[1]
        
        # Radial distance
        r = np.sqrt(x_centered**2 + y_centered**2)
        r = np.maximum(r, 1e-10)
        
        # Dimensionless radius
        s = r / self.scale_radius
        
        # NFW deflection formula
        alpha_mag = self._nfw_deflection_magnitude(s)
        
        # Directional components
        alpha_x = alpha_mag * x_centered / r
        alpha_y = alpha_mag * y_centered / r
        
        return alpha_x, alpha_y
    
    def _nfw_deflection_magnitude(self, s: np.ndarray) -> np.ndarray:
        """Compute NFW deflection magnitude."""
        # NFW deflection formula
        alpha_mag = np.zeros_like(s)
        
        # For s < 1
        mask1 = s < 1
        if np.any(mask1):
            s1 = s[mask1]
            alpha_mag[mask1] = (2 * self.scale_radius * self.norm_factor / 
                              (s1**2 - 1) * (1 - 2 / np.sqrt(1 - s1**2) * 
                              np.arctanh(np.sqrt((1 - s1) / (1 + s1)))))
        
        # For s = 1
        mask2 = np.abs(s - 1) < 1e-10
        if np.any(mask2):
            alpha_mag[mask2] = 2 * self.scale_radius * self.norm_factor / 3
        
        # For s > 1
        mask3 = s > 1
        if np.any(mask3):
            s3 = s[mask3]
            alpha_mag[mask3] = (2 * self.scale_radius * self.norm_factor / 
                              (s3**2 - 1) * (1 - 2 / np.sqrt(s3**2 - 1) * 
                              np.arctan(np.sqrt((s3 - 1) / (s3 + 1)))))
        
        return alpha_mag
    
    def convergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute convergence for NFW model.
        
        Args:
            x, y: Image plane coordinates in arcsec
            
        Returns:
            Convergence κ
        """
        # Translate to lens center
        x_centered = x - self.center[0]
        y_centered = y - self.center[1]
        
        # Radial distance
        r = np.sqrt(x_centered**2 + y_centered**2)
        r = np.maximum(r, 1e-10)
        
        # Dimensionless radius
        s = r / self.scale_radius
        
        # NFW convergence
        kappa = self._nfw_convergence(s)
        
        return kappa
    
    def _nfw_convergence(self, s: np.ndarray) -> np.ndarray:
        """Compute NFW convergence."""
        # NFW convergence formula
        kappa = np.zeros_like(s)
        
        # For s < 1
        mask1 = s < 1
        if np.any(mask1):
            s1 = s[mask1]
            kappa[mask1] = (2 * self.norm_factor / (s1**2 - 1) * 
                           (1 - 2 / np.sqrt(1 - s1**2) * 
                            np.arctanh(np.sqrt((1 - s1) / (1 + s1)))))
        
        # For s = 1
        mask2 = np.abs(s - 1) < 1e-10
        if np.any(mask2):
            kappa[mask2] = 2 * self.norm_factor / 3
        
        # For s > 1
        mask3 = s > 1
        if np.any(mask3):
            s3 = s[mask3]
            kappa[mask3] = (2 * self.norm_factor / (s3**2 - 1) * 
                           (1 - 2 / np.sqrt(s3**2 - 1) * 
                            np.arctan(np.sqrt((s3 - 1) / (s3 + 1)))))
        
        return kappa


class CompositeLensModel:
    """
    Composite lens model combining multiple components.
    
    This allows for realistic modeling of complex lens systems
    with multiple galaxies, clusters, and external shear.
    """
    
    def __init__(self, components: List[Any], external_shear: Tuple[float, float] = (0.0, 0.0)):
        """
        Initialize composite lens model.
        
        Args:
            components: List of lens model components
            external_shear: External shear (γ1, γ2)
        """
        self.components = components
        self.external_shear = external_shear
        
        logger.info(f"Composite lens model with {len(components)} components")
    
    def deflection_angle(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute total deflection angles from all components.
        
        Args:
            x, y: Image plane coordinates in arcsec
            
        Returns:
            Tuple of (α_x, α_y) total deflection angles
        """
        alpha_x_total = np.zeros_like(x)
        alpha_y_total = np.zeros_like(y)
        
        # Sum contributions from all components
        for component in self.components:
            alpha_x, alpha_y = component.deflection_angle(x, y)
            alpha_x_total += alpha_x
            alpha_y_total += alpha_y
        
        # Add external shear
        if self.external_shear[0] != 0 or self.external_shear[1] != 0:
            gamma1, gamma2 = self.external_shear
            
            # External shear deflection
            alpha_x_shear = gamma1 * x + gamma2 * y
            alpha_y_shear = gamma2 * x - gamma1 * y
            
            alpha_x_total += alpha_x_shear
            alpha_y_total += alpha_y_shear
        
        return alpha_x_total, alpha_y_total
    
    def convergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute total convergence from all components.
        
        Args:
            x, y: Image plane coordinates in arcsec
            
        Returns:
            Total convergence κ
        """
        kappa_total = np.zeros_like(x)
        
        # Sum contributions from all components
        for component in self.components:
            if hasattr(component, 'convergence'):
                kappa_total += component.convergence(x, y)
        
        return kappa_total
    
    def critical_curve(self, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute critical curve for composite model.
        
        Args:
            resolution: Number of points along critical curve
            
        Returns:
            Tuple of (x_crit, y_crit) critical curve coordinates
        """
        # For composite models, critical curve must be found numerically
        # This is a simplified approach - in practice, more sophisticated
        # methods are needed for complex composite models
        
        # Create grid around lens center
        x_range = np.linspace(-10, 10, resolution)
        y_range = np.linspace(-10, 10, resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Compute convergence
        kappa = self.convergence(X, Y)
        
        # Find critical curve (where kappa = 1)
        from skimage import measure
        contours = measure.find_contours(kappa, 1.0)
        
        if len(contours) > 0:
            # Use the largest contour
            largest_contour = max(contours, key=len)
            x_crit = x_range[0] + largest_contour[:, 1] * (x_range[-1] - x_range[0]) / (resolution - 1)
            y_crit = y_range[0] + largest_contour[:, 0] * (y_range[-1] - y_range[0]) / (resolution - 1)
        else:
            # Fallback to circular critical curve
            theta = np.linspace(0, 2*np.pi, resolution)
            r_crit = 2.0  # Default radius
            x_crit = r_crit * np.cos(theta)
            y_crit = r_crit * np.sin(theta)
        
        return x_crit, y_crit


class RealisticLensValidator:
    """
    Validator for realistic lens models with proper physics.
    
    This validator extends the basic lensing metrics to work with
    realistic lens models instead of just point mass approximation.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize realistic lens validator.
        
        Args:
            device: Device for computations
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Realistic lens validator initialized on {self.device}")
    
    def validate_einstein_radius_realistic(
        self,
        attention_maps: torch.Tensor,
        ground_truth_lens_models: List[Any],
        pixel_scale: float = 0.1
    ) -> Dict[str, float]:
        """
        Validate Einstein radius estimation using realistic lens models.
        
        Args:
            attention_maps: Attention maps [B, H, W]
            ground_truth_lens_models: List of ground truth lens models
            pixel_scale: Pixel scale in arcsec/pixel
            
        Returns:
            Dictionary with realistic Einstein radius validation metrics
        """
        attn_np = attention_maps.detach().cpu().numpy()
        
        metrics = {
            'realistic_einstein_radius_mae': [],
            'realistic_einstein_radius_rmse': [],
            'realistic_einstein_radius_relative_error': [],
            'critical_curve_overlap': [],
            'caustic_overlap': []
        }
        
        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]
            lens_model = ground_truth_lens_models[i]
            
            # Estimate Einstein radius from attention map
            estimated_radius = self._estimate_einstein_radius_realistic(attn_map, pixel_scale)
            
            # Get true Einstein radius from lens model
            if hasattr(lens_model, 'einstein_radius'):
                true_radius = lens_model.einstein_radius
            else:
                # For composite models, estimate from critical curve
                x_crit, y_crit = lens_model.critical_curve()
                true_radius = np.mean(np.sqrt(x_crit**2 + y_crit**2))
            
            if estimated_radius is not None and true_radius > 0:
                # Compute metrics
                mae = abs(estimated_radius - true_radius)
                rmse = (estimated_radius - true_radius) ** 2
                relative_error = mae / true_radius
                
                metrics['realistic_einstein_radius_mae'].append(mae)
                metrics['realistic_einstein_radius_rmse'].append(rmse)
                metrics['realistic_einstein_radius_relative_error'].append(relative_error)
                
                # Compute critical curve overlap
                overlap = self._compute_critical_curve_overlap(attn_map, lens_model, pixel_scale)
                metrics['critical_curve_overlap'].append(overlap)
                
                # Compute caustic overlap
                caustic_overlap = self._compute_caustic_overlap(attn_map, lens_model, pixel_scale)
                metrics['caustic_overlap'].append(caustic_overlap)
        
        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0.0
        
        return metrics
    
    def _estimate_einstein_radius_realistic(
        self, 
        attention_map: np.ndarray, 
        pixel_scale: float
    ) -> Optional[float]:
        """
        Estimate Einstein radius using realistic lens model approach.
        
        Args:
            attention_map: Attention map [H, W]
            pixel_scale: Pixel scale in arcsec/pixel
            
        Returns:
            Estimated Einstein radius in arcsec
        """
        # Find lens center
        center_y, center_x = np.unravel_index(np.argmax(attention_map), attention_map.shape)
        
        # Convert to arcsec
        center_x_arcsec = center_x * pixel_scale
        center_y_arcsec = center_y * pixel_scale
        
        # Create coordinate grid
        H, W = attention_map.shape
        y_coords, x_coords = np.ogrid[:H, :W]
        x_arcsec = x_coords * pixel_scale
        y_arcsec = y_coords * pixel_scale
        
        # Find critical curve (where attention is high)
        threshold = 0.7 * attention_map.max()
        critical_mask = attention_map > threshold
        
        if critical_mask.sum() == 0:
            return None
        
        # Compute radial distances from center
        r = np.sqrt((x_arcsec - center_x_arcsec)**2 + (y_arcsec - center_y_arcsec)**2)
        
        # Find average radius of critical curve
        critical_radii = r[critical_mask]
        einstein_radius = np.mean(critical_radii)
        
        return einstein_radius
    
    def _compute_critical_curve_overlap(
        self,
        attention_map: np.ndarray,
        lens_model: Any,
        pixel_scale: float
    ) -> float:
        """
        Compute overlap between attention map and true critical curve.
        
        Args:
            attention_map: Attention map [H, W]
            lens_model: Ground truth lens model
            pixel_scale: Pixel scale in arcsec/pixel
            
        Returns:
            Overlap score (0-1)
        """
        # Get true critical curve
        x_crit, y_crit = lens_model.critical_curve()
        
        # Convert to pixel coordinates
        H, W = attention_map.shape
        x_crit_pix = x_crit / pixel_scale
        y_crit_pix = y_crit / pixel_scale
        
        # Create binary mask for critical curve
        crit_mask = np.zeros((H, W), dtype=bool)
        
        # Interpolate critical curve onto pixel grid
        from scipy.interpolate import interp1d
        
        # Sort points by angle
        angles = np.arctan2(y_crit_pix - H/2, x_crit_pix - W/2)
        sort_idx = np.argsort(angles)
        
        x_sorted = x_crit_pix[sort_idx]
        y_sorted = y_crit_pix[sort_idx]
        
        # Create smooth curve
        t = np.linspace(0, 1, len(x_sorted))
        f_x = interp1d(t, x_sorted, kind='cubic', bounds_error=False, fill_value='extrapolate')
        f_y = interp1d(t, y_sorted, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        t_new = np.linspace(0, 1, 1000)
        x_smooth = f_x(t_new)
        y_smooth = f_y(t_new)
        
        # Create mask
        for i in range(len(x_smooth)):
            x, y = int(x_smooth[i]), int(y_smooth[i])
            if 0 <= x < W and 0 <= y < H:
                crit_mask[y, x] = True
        
        # Dilate mask to account for pixelization
        from scipy.ndimage import binary_dilation
        crit_mask = binary_dilation(crit_mask, structure=np.ones((3, 3)))
        
        # Compute overlap with attention map
        attention_threshold = 0.5 * attention_map.max()
        attention_mask = attention_map > attention_threshold
        
        # Compute intersection over union
        intersection = np.logical_and(crit_mask, attention_mask).sum()
        union = np.logical_or(crit_mask, attention_mask).sum()
        
        if union == 0:
            return 0.0
        
        overlap = intersection / union
        return overlap
    
    def _compute_caustic_overlap(
        self,
        attention_map: np.ndarray,
        lens_model: Any,
        pixel_scale: float
    ) -> float:
        """
        Compute overlap between attention map and true caustic.
        
        Args:
            attention_map: Attention map [H, W]
            lens_model: Ground truth lens model
            pixel_scale: Pixel scale in arcsec/pixel
            
        Returns:
            Caustic overlap score (0-1)
        """
        # Get true caustic
        x_caustic, y_caustic = lens_model.caustic()
        
        # Convert to pixel coordinates
        H, W = attention_map.shape
        x_caustic_pix = x_caustic / pixel_scale
        y_caustic_pix = y_caustic / pixel_scale
        
        # Create binary mask for caustic
        caustic_mask = np.zeros((H, W), dtype=bool)
        
        # Interpolate caustic onto pixel grid
        from scipy.interpolate import interp1d
        
        # Sort points by angle
        angles = np.arctan2(y_caustic_pix - H/2, x_caustic_pix - W/2)
        sort_idx = np.argsort(angles)
        
        x_sorted = x_caustic_pix[sort_idx]
        y_sorted = y_caustic_pix[sort_idx]
        
        # Create smooth curve
        t = np.linspace(0, 1, len(x_sorted))
        f_x = interp1d(t, x_sorted, kind='cubic', bounds_error=False, fill_value='extrapolate')
        f_y = interp1d(t, y_sorted, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        t_new = np.linspace(0, 1, 1000)
        x_smooth = f_x(t_new)
        y_smooth = f_y(t_new)
        
        # Create mask
        for i in range(len(x_smooth)):
            x, y = int(x_smooth[i]), int(y_smooth[i])
            if 0 <= x < W and 0 <= y < H:
                caustic_mask[y, x] = True
        
        # Dilate mask
        from scipy.ndimage import binary_dilation
        caustic_mask = binary_dilation(caustic_mask, structure=np.ones((3, 3)))
        
        # Compute overlap with attention map
        attention_threshold = 0.3 * attention_map.max()  # Lower threshold for caustic
        attention_mask = attention_map > attention_threshold
        
        # Compute intersection over union
        intersection = np.logical_and(caustic_mask, attention_mask).sum()
        union = np.logical_or(caustic_mask, attention_mask).sum()
        
        if union == 0:
            return 0.0
        
        overlap = intersection / union
        return overlap
    
    def validate_lensing_equation_realistic(
        self,
        attention_maps: torch.Tensor,
        ground_truth_lens_models: List[Any],
        source_positions: torch.Tensor,
        pixel_scale: float = 0.1
    ) -> Dict[str, float]:
        """
        Validate lensing equation using realistic lens models.
        
        Args:
            attention_maps: Attention maps [B, H, W]
            ground_truth_lens_models: List of ground truth lens models
            source_positions: Source positions [B, 2] in arcsec
            pixel_scale: Pixel scale in arcsec/pixel
            
        Returns:
            Dictionary with realistic lensing equation validation metrics
        """
        attn_np = attention_maps.detach().cpu().numpy()
        source_pos = source_positions.detach().cpu().numpy()
        
        metrics = {
            'realistic_lensing_residual_mae': [],
            'realistic_lensing_residual_rmse': [],
            'realistic_lensing_residual_max': [],
            'realistic_lensing_equation_satisfied': []
        }
        
        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]
            lens_model = ground_truth_lens_models[i]
            src_pos = source_pos[i]
            
            # Find image positions from attention map
            image_positions = self._find_image_positions_realistic(attn_map, pixel_scale)
            
            if len(image_positions) > 0:
                # Compute lensing equation residuals using realistic model
                residuals = self._compute_lensing_equation_residual_realistic(
                    image_positions, src_pos, lens_model
                )
                
                if residuals is not None and len(residuals) > 0:
                    mae = np.mean(np.abs(residuals))
                    rmse = np.sqrt(np.mean(residuals**2))
                    max_residual = np.max(np.abs(residuals))
                    equation_satisfied = 1.0 if mae < 0.1 else 0.0  # 0.1 arcsec tolerance
                    
                    metrics['realistic_lensing_residual_mae'].append(mae)
                    metrics['realistic_lensing_residual_rmse'].append(rmse)
                    metrics['realistic_lensing_residual_max'].append(max_residual)
                    metrics['realistic_lensing_equation_satisfied'].append(equation_satisfied)
        
        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0.0
        
        return metrics
    
    def _find_image_positions_realistic(
        self, 
        attention_map: np.ndarray, 
        pixel_scale: float
    ) -> List[np.ndarray]:
        """Find image positions from attention map using realistic approach."""
        from skimage.feature import peak_local_maxima
        
        # Find local maxima in attention map
        peaks = peak_local_maxima(attention_map, min_distance=5, threshold_abs=0.3)
        
        image_positions = []
        for peak in peaks:
            y, x = peak
            # Convert to arcsec
            pos_x = x * pixel_scale
            pos_y = y * pixel_scale
            image_positions.append(np.array([pos_x, pos_y]))
        
        return image_positions
    
    def _compute_lensing_equation_residual_realistic(
        self,
        image_positions: List[np.ndarray],
        source_position: np.ndarray,
        lens_model: Any
    ) -> Optional[np.ndarray]:
        """
        Compute lensing equation residuals using realistic lens model.
        
        Args:
            image_positions: List of image positions
            source_position: Source position
            lens_model: Realistic lens model
            
        Returns:
            Array of residuals
        """
        residuals = []
        
        for img_pos in image_positions:
            # Compute deflection angle using realistic model
            alpha_x, alpha_y = lens_model.deflection_angle(
                img_pos[0], img_pos[1]
            )
            
            # Lensing equation: β = θ - α(θ)
            predicted_source = img_pos - np.array([alpha_x, alpha_y])
            residual = np.linalg.norm(predicted_source - source_position)
            residuals.append(residual)
        
        return np.array(residuals) if residuals else None


def create_realistic_lens_models(
    einstein_radii: np.ndarray,
    ellipticities: np.ndarray = None,
    position_angles: np.ndarray = None,
    model_type: str = "SIE"
) -> List[Any]:
    """
    Create realistic lens models for validation.
    
    Args:
        einstein_radii: Array of Einstein radii
        ellipticities: Array of ellipticities (optional)
        position_angles: Array of position angles (optional)
        model_type: Type of lens model ("SIE", "NFW", "composite")
        
    Returns:
        List of lens models
    """
    models = []
    
    if ellipticities is None:
        ellipticities = np.zeros_like(einstein_radii)
    
    if position_angles is None:
        position_angles = np.zeros_like(einstein_radii)
    
    for i in range(len(einstein_radii)):
        if model_type == "SIE":
            model = SIELensModel(
                einstein_radius=einstein_radii[i],
                ellipticity=ellipticities[i],
                position_angle=position_angles[i]
            )
        elif model_type == "NFW":
            model = NFWLensModel(
                scale_radius=einstein_radii[i] / 2,  # Rough conversion
                concentration=5.0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        models.append(model)
    
    return models




