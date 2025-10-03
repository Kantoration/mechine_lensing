"""
Color Consistency Physics Prior for Gravitational Lensing

Implements the physics constraint that multiple images from the same source
should have matching intrinsic colors (achromatic lensing principle).

This module provides:
1. ColorAwarePhotometry: Enhanced photometry with color extraction
2. ColorConsistencyPrior: Physics-informed loss with robust handling
3. DataAwareColorPrior: Context-aware gating for different source types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ColorAwarePhotometry:
    """Enhanced photometry with color consistency validation."""
    
    def __init__(self, bands: List[str], target_fwhm: float = 1.0):
        """
        Initialize color-aware photometry.
        
        Args:
            bands: List of photometric bands (e.g., ['g', 'r', 'i', 'z', 'y'])
            target_fwhm: Target PSF FWHM for PSF matching
        """
        self.bands = bands
        self.target_fwhm = target_fwhm
        self.reddening_laws = {
            'Cardelli89_RV3.1': [3.1, 2.3, 1.6, 1.2, 0.8],  # g,r,i,z,y
            'Schlafly11': [3.0, 2.2, 1.5, 1.1, 0.7]
        }
    
    def extract_segment_colors(
        self, 
        images: Dict[str, np.ndarray], 
        segments: List[Dict],
        lens_light_model: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """
        Extract colors for each lensed segment with proper photometry.
        
        Args:
            images: Dict of {band: image_array}
            segments: List of segment dictionaries with masks
            lens_light_model: Optional lens light subtraction model
        
        Returns:
            Dict with color measurements per segment
        """
        results = {}
        
        for i, segment in enumerate(segments):
            segment_colors = {}
            segment_fluxes = {}
            segment_errors = {}
            
            for band in self.bands:
                if band not in images:
                    continue
                    
                img = images[band].copy()
                
                # Apply lens light subtraction if available
                if lens_light_model and band in lens_light_model:
                    img = img - lens_light_model[band]
                
                # Extract flux in segment aperture
                mask = segment['mask']
                flux, flux_err = self._aperture_photometry(img, mask)
                
                segment_fluxes[band] = flux
                segment_errors[band] = flux_err
            
            # Compute colors (magnitude differences)
            colors = self._compute_colors(segment_fluxes, segment_errors)
            
            results[f'segment_{i}'] = {
                'colors': colors,
                'fluxes': segment_fluxes,
                'errors': segment_errors,
                'band_mask': [band in images for band in self.bands],
                'segment_info': segment
            }
        
        return results
    
    def _aperture_photometry(
        self, 
        img: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[float, float]:
        """Perform aperture photometry with variance estimation."""
        try:
            from photutils.aperture import aperture_photometry
            from photutils.segmentation import SegmentationImage
            
            # Create aperture from mask
            seg_img = SegmentationImage(mask.astype(int))
            aperture = seg_img.make_cutout(img, mask)
            
            # Estimate background
            bg_mask = ~mask
            bg_median = np.median(img[bg_mask])
            bg_std = np.std(img[bg_mask])
            
            # Compute flux and error
            flux = np.sum(img[mask]) - bg_median * np.sum(mask)
            flux_err = np.sqrt(np.sum(mask) * bg_std**2)
            
            return flux, flux_err
            
        except ImportError:
            logger.warning("photutils not available, using simple photometry")
            # Fallback to simple photometry
            flux = np.sum(img[mask])
            flux_err = np.sqrt(np.sum(mask)) * np.std(img)
            return flux, flux_err
    
    def _compute_colors(
        self, 
        fluxes: Dict[str, float], 
        errors: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute colors as magnitude differences."""
        colors = {}
        
        # Use r-band as reference
        if 'r' not in fluxes:
            return colors
            
        ref_flux = fluxes['r']
        ref_mag = -2.5 * np.log10(ref_flux) if ref_flux > 0 else 99.0
        
        for band in self.bands:
            if band == 'r' or band not in fluxes:
                continue
                
            if fluxes[band] > 0:
                mag = -2.5 * np.log10(fluxes[band])
                colors[f'{band}-r'] = mag - ref_mag
            else:
                colors[f'{band}-r'] = np.nan
        
        return colors


class ColorConsistencyPrior:
    """
    Physics-informed color consistency loss with robust handling of real-world effects.
    
    Implements the color consistency constraint:
    L_color(G) = Σ_s ρ((c_s - c̄_G - E_s R)^T Σ_s^{-1} (c_s - c̄_G - E_s R)) + λ_E Σ_s E_s^2
    """
    
    def __init__(
        self, 
        reddening_law: str = "Cardelli89_RV3.1",
        lambda_E: float = 0.05,
        robust_delta: float = 0.1,
        color_consistency_weight: float = 0.1
    ):
        """
        Initialize color consistency prior.
        
        Args:
            reddening_law: Reddening law to use ('Cardelli89_RV3.1' or 'Schlafly11')
            lambda_E: Regularization weight for differential extinction
            robust_delta: Huber loss threshold for outlier handling
            color_consistency_weight: Overall weight for color consistency loss
        """
        self.reddening_vec = torch.tensor(self._get_reddening_law(reddening_law))
        self.lambda_E = lambda_E
        self.delta = robust_delta
        self.weight = color_consistency_weight
        
    def _get_reddening_law(self, law_name: str) -> List[float]:
        """Get reddening law vector for color bands."""
        laws = {
            'Cardelli89_RV3.1': [2.3, 1.6, 1.2, 0.8],  # g-r, r-i, i-z, z-y
            'Schlafly11': [2.2, 1.5, 1.1, 0.7]
        }
        return laws.get(law_name, laws['Cardelli89_RV3.1'])
    
    def huber_loss(self, r2: torch.Tensor) -> torch.Tensor:
        """Robust Huber loss for outlier handling."""
        d = self.delta
        return torch.where(
            r2 < d**2, 
            0.5 * r2, 
            d * (torch.sqrt(r2) - 0.5 * d)
        )
    
    @torch.no_grad()
    def solve_differential_extinction(
        self, 
        c_minus_cbar: torch.Tensor, 
        Sigma_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve for optimal differential extinction E_s in closed form.
        
        E* = argmin_E (c - c̄ - E R)^T Σ^{-1} (c - c̄ - E R) + λ_E E^2
        """
        # Ridge regression along reddening vector
        num = torch.einsum('bi,bij,bj->b', c_minus_cbar, Sigma_inv, self.reddening_vec)
        den = torch.einsum('i,bij,j->b', self.reddening_vec, Sigma_inv, self.reddening_vec) + self.lambda_E
        return num / (den + 1e-8)
    
    def __call__(
        self, 
        colors: List[torch.Tensor], 
        color_covs: List[torch.Tensor], 
        groups: List[List[int]],
        band_masks: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute color consistency loss for grouped lensed segments.
        
        Args:
            colors: List of color vectors per segment [B-1]
            color_covs: List of color covariance matrices [B-1, B-1]
            groups: List of lists defining lens systems
            band_masks: List of band availability masks
        
        Returns:
            Color consistency loss
        """
        if not groups or not colors:
            return torch.tensor(0.0, device=colors[0].device if colors else 'cpu')
        
        total_loss = torch.tensor(0.0, device=colors[0].device)
        valid_groups = 0
        
        for group in groups:
            if len(group) < 2:  # Need at least 2 segments for color comparison
                continue
                
            # Stack colors and covariances for this group
            group_colors = torch.stack([colors[i] for i in group])  # [N, B-1]
            group_covs = torch.stack([color_covs[i] for i in group])  # [N, B-1, B-1]
            group_masks = torch.stack([band_masks[i] for i in group])  # [N, B-1]
            
            # Apply band masks (set missing bands to zero)
            group_colors = group_colors * group_masks.float()
            
            # Compute robust mean (median) of colors in group
            cbar = torch.median(group_colors, dim=0).values  # [B-1]
            
            # Compute residuals
            c_minus_cbar = group_colors - cbar.unsqueeze(0)  # [N, B-1]
            
            # Solve for differential extinction
            E = self.solve_differential_extinction(c_minus_cbar, group_covs)  # [N]
            
            # Apply extinction correction
            extinction_correction = E.unsqueeze(1) * self.reddening_vec.unsqueeze(0)  # [N, B-1]
            corrected_residuals = c_minus_cbar - extinction_correction  # [N, B-1]
            
            # Compute Mahalanobis distance
            r2 = torch.einsum('ni,nij,nj->n', corrected_residuals, group_covs, corrected_residuals)
            
            # Apply robust loss
            group_loss = self.huber_loss(r2).mean()
            total_loss += group_loss
            valid_groups += 1
        
        return (total_loss / max(valid_groups, 1)) * self.weight
    
    def compute_color_distance(
        self, 
        colors_i: torch.Tensor, 
        colors_j: torch.Tensor,
        cov_i: torch.Tensor,
        cov_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute color distance between two segments for graph construction.
        
        d_color(s_i, s_j) = min_E |(c_i - c_j - E R)|_{Σ^{-1}}
        """
        # Solve for optimal extinction between pair
        c_diff = colors_i - colors_j
        cov_combined = cov_i + cov_j
        
        E_opt = self.solve_differential_extinction(
            c_diff.unsqueeze(0), 
            cov_combined.unsqueeze(0)
        )[0]
        
        # Apply extinction correction
        corrected_diff = c_diff - E_opt * self.reddening_vec
        
        # Compute Mahalanobis distance
        distance = torch.sqrt(
            torch.einsum('i,ij,j', corrected_diff, torch.inverse(cov_combined), corrected_diff)
        )
        
        return distance


class DataAwareColorPrior:
    """Color consistency prior with data-aware gating."""
    
    def __init__(self, base_prior: ColorConsistencyPrior):
        """
        Initialize data-aware color prior.
        
        Args:
            base_prior: Base color consistency prior
        """
        self.base_prior = base_prior
        self.quasar_detector = QuasarMorphologyDetector()
        self.microlensing_estimator = MicrolensingRiskEstimator()
    
    def compute_prior_weight(
        self, 
        images: torch.Tensor,
        metadata: Dict,
        groups: List[List[int]]
    ) -> torch.Tensor:
        """
        Compute per-system prior weight based on data characteristics.
        
        Args:
            images: Input images
            metadata: Metadata dictionary
            groups: List of lens system groups
        
        Returns:
            Weight tensor [num_groups] in [0, 1]
        """
        weights = []
        
        for group in groups:
            # Check if system is quasar-like
            is_quasar = self.quasar_detector.is_quasar_like(images[group])
            
            # Estimate microlensing risk
            microlensing_risk = self.microlensing_estimator.estimate_risk(
                metadata, group
            )
            
            # Check for strong time delays
            time_delay_risk = self._estimate_time_delay_risk(metadata, group)
            
            # Compute combined weight
            if is_quasar or microlensing_risk > 0.7 or time_delay_risk > 0.5:
                weight = 0.1  # Strongly downweight
            elif microlensing_risk > 0.3 or time_delay_risk > 0.2:
                weight = 0.5  # Moderate downweight
            else:
                weight = 1.0  # Full weight
            
            weights.append(weight)
        
        return torch.tensor(weights, device=images.device)
    
    def _estimate_time_delay_risk(self, metadata: Dict, group: List[int]) -> float:
        """Estimate time delay risk based on image separations and lens mass."""
        # Simplified time delay risk estimation
        # In practice, this would use more sophisticated models
        if 'image_separations' in metadata:
            max_sep = max(metadata['image_separations'])
            if max_sep > 5.0:  # Large separation suggests long time delays
                return 0.8
        return 0.1
    
    def __call__(self, *args, **kwargs):
        """Apply data-aware gating to color consistency loss."""
        base_loss = self.base_prior(*args, **kwargs)
        
        # Apply per-group weights
        if "groups" in kwargs and "images" in kwargs:
            weights = self.compute_prior_weight(
                kwargs["images"], 
                kwargs.get("metadata", {}),
                kwargs["groups"]
            )
            base_loss = base_loss * weights.mean()
        
        return base_loss


class QuasarMorphologyDetector:
    """Detect quasar-like morphology for color prior gating."""
    
    def is_quasar_like(self, images: torch.Tensor) -> bool:
        """
        Detect if images show quasar-like morphology.
        
        Args:
            images: Input images
            
        Returns:
            True if quasar-like morphology detected
        """
        # Simplified quasar detection based on point-source morphology
        # In practice, this would use more sophisticated analysis
        
        # Check for high central concentration
        center_flux = images[:, :, images.shape[2]//2-2:images.shape[2]//2+2, 
                            images.shape[3]//2-2:images.shape[3]//2+2].mean()
        total_flux = images.mean()
        
        concentration = center_flux / total_flux
        
        return concentration > 0.3  # High central concentration suggests point source


class MicrolensingRiskEstimator:
    """Estimate microlensing risk for color prior gating."""
    
    def estimate_risk(self, metadata: Dict, group: List[int]) -> float:
        """
        Estimate microlensing risk for a lens system.
        
        Args:
            metadata: Metadata dictionary
            group: List of segment indices
            
        Returns:
            Microlensing risk score [0, 1]
        """
        # Simplified microlensing risk estimation
        # In practice, this would use lens mass, stellar density, etc.
        
        risk = 0.1  # Base risk
        
        # Increase risk for high-mass lenses
        if 'lens_mass' in metadata:
            if metadata['lens_mass'] > 1e12:  # High mass
                risk += 0.3
        
        # Increase risk for dense stellar fields
        if 'stellar_density' in metadata:
            if metadata['stellar_density'] > 100:  # High stellar density
                risk += 0.4
        
        return min(risk, 1.0)
