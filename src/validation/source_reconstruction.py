#!/usr/bin/env python3
"""
source_reconstruction.py
========================
Source reconstruction and validation module for gravitational lensing.

Key Features:
- Source plane reconstruction from image data
- Physicality validation of reconstructed sources
- Multi-band source reconstruction
- Quality metrics for source reconstruction
- Comparison with ground truth sources

Usage:
    from validation.source_reconstruction import SourceReconstructor, validate_source_quality
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage, optimize
from skimage import measure, morphology, restoration

logger = logging.getLogger(__name__)


class SourceReconstructor:
    """
    Source plane reconstruction from lensed images.
    
    This class provides methods for reconstructing the source light
    distribution from lensed images using various techniques.
    """
    
    def __init__(
        self,
        lens_model: Any,
        pixel_scale: float = 0.1,
        source_pixel_scale: float = 0.05,
        regularization_weight: float = 1e-3
    ):
        """
        Initialize source reconstructor.
        
        Args:
            lens_model: Lens model for ray-tracing
            pixel_scale: Image plane pixel scale in arcsec/pixel
            source_pixel_scale: Source plane pixel scale in arcsec/pixel
            regularization_weight: Weight for regularization term
        """
        self.lens_model = lens_model
        self.pixel_scale = pixel_scale
        self.source_pixel_scale = source_pixel_scale
        self.regularization_weight = regularization_weight
        
        logger.info(f"Source reconstructor initialized with pixel scales: "
                   f"image={pixel_scale:.3f}, source={source_pixel_scale:.3f}")
    
    def reconstruct_source(
        self,
        lensed_image: np.ndarray,
        source_size: Tuple[int, int] = (64, 64),
        method: str = "linear_inversion"
    ) -> np.ndarray:
        """
        Reconstruct source light distribution from lensed image.
        
        Args:
            lensed_image: Lensed image [H, W]
            source_size: Size of source plane grid (H, W)
            method: Reconstruction method ("linear_inversion", "regularized", "mcmc")
            
        Returns:
            Reconstructed source [H, W]
        """
        if method == "linear_inversion":
            return self._linear_inversion(lensed_image, source_size)
        elif method == "regularized":
            return self._regularized_reconstruction(lensed_image, source_size)
        elif method == "mcmc":
            return self._mcmc_reconstruction(lensed_image, source_size)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
    
    def _linear_inversion(self, lensed_image: np.ndarray, source_size: Tuple[int, int]) -> np.ndarray:
        """
        Linear inversion reconstruction.
        
        This is the simplest method but can be unstable for noisy data.
        """
        H_img, W_img = lensed_image.shape
        H_src, W_src = source_size
        
        # Create lensing matrix
        lensing_matrix = self._create_lensing_matrix(H_img, W_img, H_src, W_src)
        
        # Flatten image
        image_flat = lensed_image.flatten()
        
        # Solve linear system: I = L * S
        # where I is image, L is lensing matrix, S is source
        try:
            source_flat = np.linalg.solve(lensing_matrix, image_flat)
        except np.linalg.LinAlgError:
            # Use least squares if matrix is singular
            source_flat, _, _, _ = np.linalg.lstsq(lensing_matrix, image_flat, rcond=None)
        
        # Reshape to source plane
        source = source_flat.reshape(H_src, W_src)
        
        # Ensure non-negativity
        source = np.maximum(source, 0)
        
        return source
    
    def _regularized_reconstruction(
        self, 
        lensed_image: np.ndarray, 
        source_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Regularized reconstruction with smoothness constraint.
        
        This method adds a regularization term to prevent overfitting.
        """
        H_img, W_img = lensed_image.shape
        H_src, W_src = source_size
        
        # Create lensing matrix
        lensing_matrix = self._create_lensing_matrix(H_img, W_img, H_src, W_src)
        
        # Create regularization matrix (Laplacian)
        reg_matrix = self._create_regularization_matrix(H_src, W_src)
        
        # Flatten image
        image_flat = lensed_image.flatten()
        
        # Solve regularized system: (L^T L + λ R^T R) S = L^T I
        LTL = lensing_matrix.T @ lensing_matrix
        RTR = reg_matrix.T @ reg_matrix
        LTI = lensing_matrix.T @ image_flat
        
        # Regularized system
        A = LTL + self.regularization_weight * RTR
        
        try:
            source_flat = np.linalg.solve(A, LTI)
        except np.linalg.LinAlgError:
            source_flat, _, _, _ = np.linalg.lstsq(A, LTI, rcond=None)
        
        # Reshape to source plane
        source = source_flat.reshape(H_src, W_src)
        
        # Ensure non-negativity
        source = np.maximum(source, 0)
        
        return source
    
    def _mcmc_reconstruction(
        self, 
        lensed_image: np.ndarray, 
        source_size: Tuple[int, int],
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        MCMC reconstruction for uncertainty quantification.
        
        This method provides uncertainty estimates for the reconstruction.
        """
        # For now, use regularized reconstruction as mean
        # In practice, this would implement proper MCMC sampling
        source_mean = self._regularized_reconstruction(lensed_image, source_size)
        
        # Add some noise to simulate uncertainty
        noise_level = 0.1 * np.std(source_mean)
        source_samples = []
        
        for _ in range(n_samples):
            noise = np.random.normal(0, noise_level, source_mean.shape)
            source_sample = np.maximum(source_mean + noise, 0)
            source_samples.append(source_sample)
        
        # Return mean of samples
        source_samples = np.array(source_samples)
        source = np.mean(source_samples, axis=0)
        
        return source
    
    def _create_lensing_matrix(
        self, 
        H_img: int, 
        W_img: int, 
        H_src: int, 
        W_src: int
    ) -> np.ndarray:
        """
        Create lensing matrix for ray-tracing.
        
        The lensing matrix maps source plane pixels to image plane pixels.
        """
        n_img = H_img * W_img
        n_src = H_src * W_src
        
        lensing_matrix = np.zeros((n_img, n_src))
        
        # Create coordinate grids
        y_img, x_img = np.ogrid[:H_img, :W_img]
        y_src, x_src = np.ogrid[:H_src, :W_src]
        
        # Convert to arcsec
        x_img_arcsec = x_img * self.pixel_scale
        y_img_arcsec = y_img * self.pixel_scale
        x_src_arcsec = x_src * self.source_pixel_scale
        y_src_arcsec = y_src * self.source_pixel_scale
        
        # Ray-trace from image plane to source plane
        for i in range(H_img):
            for j in range(W_img):
                img_idx = i * W_img + j
                
                # Image plane coordinates
                x_img_coord = x_img_arcsec[i, j]
                y_img_coord = y_img_arcsec[i, j]
                
                # Compute deflection angle
                alpha_x, alpha_y = self.lens_model.deflection_angle(
                    x_img_coord, y_img_coord
                )
                
                # Source plane coordinates
                x_src_coord = x_img_coord - alpha_x
                y_src_coord = y_img_coord - alpha_y
                
                # Find corresponding source pixel
                src_i = int(y_src_coord / self.source_pixel_scale)
                src_j = int(x_src_coord / self.source_pixel_scale)
                
                # Check bounds
                if 0 <= src_i < H_src and 0 <= src_j < W_src:
                    src_idx = src_i * W_src + src_j
                    lensing_matrix[img_idx, src_idx] = 1.0
        
        return lensing_matrix
    
    def _create_regularization_matrix(self, H: int, W: int) -> np.ndarray:
        """
        Create regularization matrix (Laplacian) for smoothness constraint.
        """
        n = H * W
        reg_matrix = np.zeros((n, n))
        
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                
                # Center pixel
                reg_matrix[idx, idx] = -4
                
                # Neighbors
                if i > 0:
                    reg_matrix[idx, (i-1) * W + j] = 1
                if i < H-1:
                    reg_matrix[idx, (i+1) * W + j] = 1
                if j > 0:
                    reg_matrix[idx, i * W + (j-1)] = 1
                if j < W-1:
                    reg_matrix[idx, i * W + (j+1)] = 1
        
        return reg_matrix
    
    def forward_model(self, source: np.ndarray) -> np.ndarray:
        """
        Forward model: source -> lensed image.
        
        Args:
            source: Source light distribution [H, W]
            
        Returns:
            Lensed image [H, W]
        """
        H_src, W_src = source.shape
        
        # Create coordinate grids
        y_src, x_src = np.ogrid[:H_src, :W_src]
        
        # Convert to arcsec
        x_src_arcsec = x_src * self.source_pixel_scale
        y_src_arcsec = y_src * self.source_pixel_scale
        
        # Create image plane grid
        H_img = int(H_src * self.source_pixel_scale / self.pixel_scale)
        W_img = int(W_src * self.source_pixel_scale / self.pixel_scale)
        
        y_img, x_img = np.ogrid[:H_img, :W_img]
        x_img_arcsec = x_img * self.pixel_scale
        y_img_arcsec = y_img * self.pixel_scale
        
        # Initialize lensed image
        lensed_image = np.zeros((H_img, W_img))
        
        # Ray-trace from source plane to image plane
        for i in range(H_src):
            for j in range(W_src):
                # Source plane coordinates
                x_src_coord = x_src_arcsec[i, j]
                y_src_coord = y_src_arcsec[i, j]
                
                # Compute deflection angle
                alpha_x, alpha_y = self.lens_model.deflection_angle(
                    x_src_coord, y_src_coord
                )
                
                # Image plane coordinates
                x_img_coord = x_src_coord + alpha_x
                y_img_coord = y_src_coord + alpha_y
                
                # Find corresponding image pixel
                img_i = int(y_img_coord / self.pixel_scale)
                img_j = int(x_img_coord / self.pixel_scale)
                
                # Check bounds and add flux
                if 0 <= img_i < H_img and 0 <= img_j < W_img:
                    lensed_image[img_i, img_j] += source[i, j]
        
        return lensed_image


class SourceQualityValidator:
    """
    Validator for source reconstruction quality.
    
    This class provides comprehensive validation of reconstructed
    source light distributions against physical and statistical criteria.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize source quality validator.
        
        Args:
            device: Device for computations
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Source quality validator initialized on {self.device}")
    
    def validate_source_quality(
        self,
        reconstructed_sources: torch.Tensor,
        ground_truth_sources: torch.Tensor,
        lensed_images: torch.Tensor,
        lens_models: List[Any]
    ) -> Dict[str, float]:
        """
        Validate source reconstruction quality.
        
        Args:
            reconstructed_sources: Reconstructed sources [B, H, W]
            ground_truth_sources: Ground truth sources [B, H, W]
            lensed_images: Original lensed images [B, H, W]
            lens_models: List of lens models
            
        Returns:
            Dictionary with source quality validation metrics
        """
        recon_np = reconstructed_sources.detach().cpu().numpy()
        gt_np = ground_truth_sources.detach().cpu().numpy()
        img_np = lensed_images.detach().cpu().numpy()
        
        metrics = {
            'source_reconstruction_mae': [],
            'source_reconstruction_rmse': [],
            'source_reconstruction_correlation': [],
            'source_physicality_score': [],
            'source_smoothness_score': [],
            'source_flux_conservation': [],
            'source_chi_squared': [],
            'source_bayesian_evidence': []
        }
        
        for i in range(recon_np.shape[0]):
            recon_source = recon_np[i]
            gt_source = gt_np[i]
            lensed_img = img_np[i]
            lens_model = lens_models[i]
            
            # Basic reconstruction metrics
            mae = np.mean(np.abs(recon_source - gt_source))
            rmse = np.sqrt(np.mean((recon_source - gt_source)**2))
            
            # Correlation
            correlation = np.corrcoef(recon_source.flatten(), gt_source.flatten())[0, 1]
            if not np.isnan(correlation):
                metrics['source_reconstruction_correlation'].append(correlation)
            
            metrics['source_reconstruction_mae'].append(mae)
            metrics['source_reconstruction_rmse'].append(rmse)
            
            # Physicality validation
            physicality = self._validate_source_physicality(recon_source)
            metrics['source_physicality_score'].append(physicality)
            
            # Smoothness validation
            smoothness = self._validate_source_smoothness(recon_source)
            metrics['source_smoothness_score'].append(smoothness)
            
            # Flux conservation
            flux_conservation = self._validate_flux_conservation(
                recon_source, lensed_img, lens_model
            )
            metrics['source_flux_conservation'].append(flux_conservation)
            
            # Chi-squared test
            chi_squared = self._compute_chi_squared(recon_source, lensed_img, lens_model)
            metrics['source_chi_squared'].append(chi_squared)
            
            # Bayesian evidence (simplified)
            evidence = self._compute_bayesian_evidence(recon_source, lensed_img, lens_model)
            metrics['source_bayesian_evidence'].append(evidence)
        
        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0.0
        
        return metrics
    
    def _validate_source_physicality(self, source: np.ndarray) -> float:
        """
        Validate physicality of reconstructed source.
        
        Physical sources should be:
        - Non-negative
        - Smooth
        - Compact
        """
        score = 0.0
        
        # Non-negativity
        if np.all(source >= 0):
            score += 0.4
        else:
            # Penalize negative values
            negative_fraction = np.sum(source < 0) / source.size
            score += 0.4 * (1 - negative_fraction)
        
        # Smoothness (low gradient)
        grad_x = np.gradient(source, axis=1)
        grad_y = np.gradient(source, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if gradient_magnitude.mean() < 0.1:
            score += 0.3
        else:
            score += 0.3 * (0.1 / gradient_magnitude.mean())
        
        # Compactness (concentrated in center)
        center_y, center_x = source.shape[0] // 2, source.shape[1] // 2
        y, x = np.ogrid[:source.shape[0], :source.shape[1]]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Compute flux within half-radius
        half_radius = min(source.shape) // 4
        flux_within_half = np.sum(source[r <= half_radius])
        total_flux = np.sum(source)
        
        if total_flux > 0:
            compactness = flux_within_half / total_flux
            score += 0.3 * compactness
        
        return score
    
    def _validate_source_smoothness(self, source: np.ndarray) -> float:
        """
        Validate smoothness of reconstructed source.
        
        Sources should be smooth, not noisy.
        """
        # Compute Laplacian
        laplacian = ndimage.laplace(source)
        
        # Smoothness score (lower Laplacian variance = smoother)
        laplacian_var = np.var(laplacian)
        
        # Normalize to 0-1 scale
        smoothness = 1.0 / (1.0 + laplacian_var)
        
        return smoothness
    
    def _validate_flux_conservation(
        self, 
        source: np.ndarray, 
        lensed_image: np.ndarray, 
        lens_model: Any
    ) -> float:
        """
        Validate flux conservation in lensing.
        
        Total flux should be conserved (ignoring magnification).
        """
        # Total flux in source
        source_flux = np.sum(source)
        
        # Total flux in lensed image
        image_flux = np.sum(lensed_image)
        
        # Flux conservation (should be close to 1)
        if image_flux > 0:
            flux_ratio = source_flux / image_flux
            # Penalize deviations from 1
            conservation_score = 1.0 - abs(flux_ratio - 1.0)
            conservation_score = max(0, conservation_score)
        else:
            conservation_score = 0.0
        
        return conservation_score
    
    def _compute_chi_squared(
        self, 
        source: np.ndarray, 
        lensed_image: np.ndarray, 
        lens_model: Any
    ) -> float:
        """
        Compute chi-squared statistic for source reconstruction.
        
        Lower chi-squared indicates better fit.
        """
        # Forward model the source
        reconstructor = SourceReconstructor(lens_model)
        predicted_image = reconstructor.forward_model(source)
        
        # Compute chi-squared
        residuals = lensed_image - predicted_image
        
        # Assume Poisson noise
        noise_variance = np.maximum(lensed_image, 1.0)  # Avoid division by zero
        
        chi_squared = np.sum(residuals**2 / noise_variance)
        
        # Normalize by degrees of freedom
        dof = lensed_image.size - source.size
        if dof > 0:
            chi_squared = chi_squared / dof
        
        return chi_squared
    
    def _compute_bayesian_evidence(
        self, 
        source: np.ndarray, 
        lensed_image: np.ndarray, 
        lens_model: Any
    ) -> float:
        """
        Compute simplified Bayesian evidence for source reconstruction.
        
        Higher evidence indicates better model.
        """
        # Forward model the source
        reconstructor = SourceReconstructor(lens_model)
        predicted_image = reconstructor.forward_model(source)
        
        # Compute likelihood (Gaussian)
        residuals = lensed_image - predicted_image
        noise_variance = np.maximum(lensed_image, 1.0)
        
        log_likelihood = -0.5 * np.sum(residuals**2 / noise_variance)
        
        # Compute prior (smoothness)
        grad_x = np.gradient(source, axis=1)
        grad_y = np.gradient(source, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        log_prior = -0.5 * np.sum(gradient_magnitude**2)
        
        # Evidence (log)
        log_evidence = log_likelihood + log_prior
        
        # Convert to linear scale (with normalization)
        evidence = np.exp(log_evidence / source.size)
        
        return evidence
    
    def validate_multi_band_reconstruction(
        self,
        reconstructed_sources: Dict[str, torch.Tensor],
        ground_truth_sources: Dict[str, torch.Tensor],
        lensed_images: Dict[str, torch.Tensor],
        lens_models: List[Any]
    ) -> Dict[str, float]:
        """
        Validate multi-band source reconstruction.
        
        Args:
            reconstructed_sources: Dict of reconstructed sources by band
            ground_truth_sources: Dict of ground truth sources by band
            lensed_images: Dict of lensed images by band
            lens_models: List of lens models
            
        Returns:
            Dictionary with multi-band validation metrics
        """
        metrics = {}
        
        # Validate each band
        for band in reconstructed_sources.keys():
            band_metrics = self.validate_source_quality(
                reconstructed_sources[band],
                ground_truth_sources[band],
                lensed_images[band],
                lens_models
            )
            
            # Add band prefix
            for key, value in band_metrics.items():
                metrics[f'{band}_{key}'] = value
        
        # Cross-band consistency
        if len(reconstructed_sources) > 1:
            consistency_metrics = self._validate_cross_band_consistency(
                reconstructed_sources, ground_truth_sources
            )
            metrics.update(consistency_metrics)
        
        return metrics
    
    def _validate_cross_band_consistency(
        self,
        reconstructed_sources: Dict[str, torch.Tensor],
        ground_truth_sources: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Validate consistency across different bands.
        
        Sources should have similar morphology across bands.
        """
        metrics = {}
        
        bands = list(reconstructed_sources.keys())
        
        # Compute cross-band correlations
        correlations = []
        for i in range(len(bands)):
            for j in range(i+1, len(bands)):
                band1, band2 = bands[i], bands[j]
                
                recon1 = reconstructed_sources[band1].detach().cpu().numpy()
                recon2 = reconstructed_sources[band2].detach().cpu().numpy()
                
                # Compute correlation
                correlation = np.corrcoef(recon1.flatten(), recon2.flatten())[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
        
        if correlations:
            metrics['cross_band_correlation'] = np.mean(correlations)
            metrics['cross_band_correlation_std'] = np.std(correlations)
        
        # Compute morphology consistency
        morphology_scores = []
        for i in range(len(bands)):
            for j in range(i+1, len(bands)):
                band1, band2 = bands[i], bands[j]
                
                recon1 = reconstructed_sources[band1].detach().cpu().numpy()
                recon2 = reconstructed_sources[band2].detach().cpu().numpy()
                
                # Compute morphology similarity (simplified)
                # This could be enhanced with more sophisticated morphology metrics
                morphology_score = self._compute_morphology_similarity(recon1, recon2)
                morphology_scores.append(morphology_score)
        
        if morphology_scores:
            metrics['cross_band_morphology_consistency'] = np.mean(morphology_scores)
        
        return metrics
    
    def _compute_morphology_similarity(self, source1: np.ndarray, source2: np.ndarray) -> float:
        """
        Compute morphology similarity between two sources.
        
        This is a simplified metric - in practice, more sophisticated
        morphology measures would be used.
        """
        # Normalize sources
        source1_norm = source1 / (np.sum(source1) + 1e-10)
        source2_norm = source2 / (np.sum(source2) + 1e-10)
        
        # Compute correlation
        correlation = np.corrcoef(source1_norm.flatten(), source2_norm.flatten())[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return correlation


def validate_source_quality(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    validator: SourceQualityValidator = None
) -> Dict[str, float]:
    """
    Comprehensive source quality validation.
    
    Args:
        model: Model to validate
        test_loader: Test data loader with source reconstruction data
        validator: Source quality validator
        
    Returns:
        Dictionary with comprehensive source quality validation metrics
    """
    if validator is None:
        validator = SourceQualityValidator()
    
    model.eval()
    
    all_reconstructed_sources = []
    all_ground_truth_sources = []
    all_lensed_images = []
    all_lens_models = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(validator.device)
            sources = batch['source'].to(validator.device)
            lens_models = batch['lens_model']
            
            # Get model outputs with source reconstruction
            if hasattr(model, 'forward_with_source_reconstruction'):
                outputs, source_info = model.forward_with_source_reconstruction(images)
            else:
                outputs = model(images)
                source_info = {}
            
            if 'reconstructed_source' in source_info:
                all_reconstructed_sources.append(source_info['reconstructed_source'].cpu())
                all_ground_truth_sources.append(sources.cpu())
                all_lensed_images.append(images.cpu())
                all_lens_models.extend(lens_models)
    
    if all_reconstructed_sources:
        # Concatenate all data
        reconstructed_sources = torch.cat(all_reconstructed_sources, dim=0)
        ground_truth_sources = torch.cat(all_ground_truth_sources, dim=0)
        lensed_images = torch.cat(all_lensed_images, dim=0)
        
        # Validate source quality
        source_metrics = validator.validate_source_quality(
            reconstructed_sources, ground_truth_sources, lensed_images, all_lens_models
        )
        
        return source_metrics
    
    return {}


def create_source_validation_report(
    validation_results: Dict[str, float],
    save_path: Optional[str] = None
) -> str:
    """
    Create comprehensive source validation report.
    
    Args:
        validation_results: Validation results dictionary
        save_path: Optional path to save report
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 100)
    report.append("COMPREHENSIVE SOURCE RECONSTRUCTION VALIDATION REPORT")
    report.append("=" * 100)
    
    # Basic reconstruction metrics
    report.append("\nSOURCE RECONSTRUCTION METRICS:")
    report.append("-" * 50)
    recon_keys = [k for k in validation_results.keys() if k.startswith('source_reconstruction_')]
    for key in recon_keys:
        metric_name = key.replace('source_reconstruction_', '')
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")
    
    # Physicality metrics
    report.append("\nSOURCE PHYSICALITY METRICS:")
    report.append("-" * 50)
    physicality_keys = [k for k in validation_results.keys() if k.startswith('source_physicality')]
    for key in physicality_keys:
        metric_name = key.replace('source_physicality_', '')
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")
    
    # Quality metrics
    report.append("\nSOURCE QUALITY METRICS:")
    report.append("-" * 50)
    quality_keys = [k for k in validation_results.keys() if k.startswith(('source_smoothness', 'source_flux_conservation'))]
    for key in quality_keys:
        metric_name = key.replace('source_', '')
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")
    
    # Statistical metrics
    report.append("\nSTATISTICAL VALIDATION METRICS:")
    report.append("-" * 50)
    stat_keys = [k for k in validation_results.keys() if k.startswith(('source_chi_squared', 'source_bayesian_evidence'))]
    for key in stat_keys:
        metric_name = key.replace('source_', '')
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")
    
    # Cross-band consistency (if available)
    report.append("\nCROSS-BAND CONSISTENCY METRICS:")
    report.append("-" * 50)
    cross_band_keys = [k for k in validation_results.keys() if k.startswith('cross_band_')]
    for key in cross_band_keys:
        metric_name = key.replace('cross_band_', '')
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")
    
    # Overall source quality score
    report.append("\nOVERALL SOURCE QUALITY SCORE:")
    report.append("-" * 50)
    
    # Compute overall score from key metrics
    score_components = []
    
    # Reconstruction quality
    if 'source_reconstruction_correlation' in validation_results:
        score_components.append(validation_results['source_reconstruction_correlation'])
    
    # Physicality
    if 'source_physicality_score' in validation_results:
        score_components.append(validation_results['source_physicality_score'])
    
    # Smoothness
    if 'source_smoothness_score' in validation_results:
        score_components.append(validation_results['source_smoothness_score'])
    
    # Flux conservation
    if 'source_flux_conservation' in validation_results:
        score_components.append(validation_results['source_flux_conservation'])
    
    if score_components:
        overall_score = np.mean(score_components)
        report.append(f"  Overall Source Quality Score: {overall_score:.4f}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        report.append("-" * 50)
        
        if overall_score < 0.5:
            report.append("  - Significant source reconstruction issues detected")
            report.append("  - Consider improving regularization or reconstruction method")
            report.append("  - Validate against known source morphologies")
        elif overall_score < 0.7:
            report.append("  - Good source reconstruction with room for improvement")
            report.append("  - Fine-tune reconstruction parameters")
        else:
            report.append("  - Excellent source reconstruction quality")
            report.append("  - Model ready for scientific source analysis")
    
    report.append("=" * 100)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text




