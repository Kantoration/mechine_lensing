#!/usr/bin/env python3
"""
lensing_metrics.py
==================
Advanced lensing-specific validation metrics for gravitational lensing analysis.

Key Features:
- Einstein radius estimation and validation
- Arc multiplicity and parity analysis
- Time delay distribution validation
- Lensing equation residual analysis
- Source reconstruction quality metrics

Usage:
    from validation.lensing_metrics import LensingMetricsValidator, validate_lensing_physics
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage import measure
from skimage.feature import peak_local_maxima

logger = logging.getLogger(__name__)


class LensingMetricsValidator:
    """
    Advanced validator for lensing-specific physics and metrics.

    This validator provides comprehensive analysis of gravitational lensing
    properties including Einstein radius, arc characteristics, and lensing
    equation residuals.
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize lensing metrics validator.

        Args:
            device: Device for computations
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Physical constants
        self.c = 299792458  # Speed of light (m/s)
        self.G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
        self.H0 = 70.0  # Hubble constant (km/s/Mpc)

        logger.info(f"Lensing metrics validator initialized on {self.device}")

    def validate_einstein_radius(
        self,
        attention_maps: torch.Tensor,
        ground_truth_einstein_radius: torch.Tensor,
        pixel_scale: float = 0.1,  # arcsec/pixel
        tolerance: float = 0.2,
    ) -> Dict[str, float]:
        """
        Validate Einstein radius estimation from attention maps.

        Args:
            attention_maps: Attention maps [B, H, W]
            ground_truth_einstein_radius: Ground truth Einstein radii [B] in arcsec
            pixel_scale: Pixel scale in arcsec/pixel
            tolerance: Relative tolerance for validation

        Returns:
            Dictionary with Einstein radius validation metrics
        """
        attn_np = attention_maps.detach().cpu().numpy()
        gt_einstein = ground_truth_einstein_radius.detach().cpu().numpy()

        metrics = {
            "einstein_radius_mae": [],
            "einstein_radius_rmse": [],
            "einstein_radius_relative_error": [],
            "einstein_radius_correlation": [],
            "einstein_radius_within_tolerance": [],
        }

        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]
            gt_radius = gt_einstein[i]

            # Estimate Einstein radius from attention map
            estimated_radius = self._estimate_einstein_radius(attn_map, pixel_scale)

            if estimated_radius is not None and gt_radius > 0:
                # Compute metrics
                mae = abs(estimated_radius - gt_radius)
                rmse = (estimated_radius - gt_radius) ** 2
                relative_error = mae / gt_radius
                within_tolerance = 1.0 if relative_error <= tolerance else 0.0

                metrics["einstein_radius_mae"].append(mae)
                metrics["einstein_radius_rmse"].append(rmse)
                metrics["einstein_radius_relative_error"].append(relative_error)
                metrics["einstein_radius_within_tolerance"].append(within_tolerance)

        # Compute correlations
        if len(metrics["einstein_radius_mae"]) > 1:
            estimated_radii = [
                self._estimate_einstein_radius(attn_np[i], pixel_scale)
                for i in range(len(metrics["einstein_radius_mae"]))
            ]
            valid_estimates = [
                (est, gt_einstein[i])
                for i, est in enumerate(estimated_radii)
                if est is not None and gt_einstein[i] > 0
            ]

            if len(valid_estimates) > 1:
                est_vals, gt_vals = zip(*valid_estimates)
                correlation = np.corrcoef(est_vals, gt_vals)[0, 1]
                if not np.isnan(correlation):
                    metrics["einstein_radius_correlation"].append(correlation)

        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0.0

        return metrics

    def _estimate_einstein_radius(
        self, attention_map: np.ndarray, pixel_scale: float
    ) -> Optional[float]:
        """
        Estimate Einstein radius from attention map.

        Args:
            attention_map: Attention map [H, W]
            pixel_scale: Pixel scale in arcsec/pixel

        Returns:
            Estimated Einstein radius in arcsec
        """
        # Find lens center (highest attention)
        center_y, center_x = np.unravel_index(
            np.argmax(attention_map), attention_map.shape
        )

        # Create radial profile
        H, W = attention_map.shape
        y, x = np.ogrid[:H, :W]
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Compute radial average
        r_max = min(center_x, center_y, W - center_x, H - center_y)
        r_bins = np.arange(0, r_max, 1)
        radial_profile = []

        for r_bin in r_bins:
            mask = (r >= r_bin) & (r < r_bin + 1)
            if mask.sum() > 0:
                radial_profile.append(attention_map[mask].mean())
            else:
                radial_profile.append(0.0)

        radial_profile = np.array(radial_profile)

        # Find Einstein radius (radius where attention drops to half maximum)
        max_attention = radial_profile.max()
        half_max = max_attention * 0.5

        # Find first radius where attention drops below half maximum
        einstein_idx = np.where(radial_profile < half_max)[0]
        if len(einstein_idx) > 0:
            einstein_radius_pixels = einstein_idx[0]
            einstein_radius_arcsec = einstein_radius_pixels * pixel_scale
            return einstein_radius_arcsec

        return None

    def validate_arc_multiplicity(
        self,
        attention_maps: torch.Tensor,
        ground_truth_multiplicity: torch.Tensor,
        min_arc_length: float = 5.0,  # pixels
    ) -> Dict[str, float]:
        """
        Validate arc multiplicity detection.

        Args:
            attention_maps: Attention maps [B, H, W]
            ground_truth_multiplicity: Ground truth arc multiplicities [B]
            min_arc_length: Minimum arc length in pixels

        Returns:
            Dictionary with multiplicity validation metrics
        """
        attn_np = attention_maps.detach().cpu().numpy()
        gt_multiplicity = ground_truth_multiplicity.detach().cpu().numpy()

        metrics = {
            "multiplicity_mae": [],
            "multiplicity_accuracy": [],
            "multiplicity_precision": [],
            "multiplicity_recall": [],
            "multiplicity_f1": [],
        }

        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]
            gt_mult = gt_multiplicity[i]

            # Estimate multiplicity from attention map
            estimated_multiplicity = self._estimate_arc_multiplicity(
                attn_map, min_arc_length
            )

            # Compute metrics
            mae = abs(estimated_multiplicity - gt_mult)
            accuracy = 1.0 if estimated_multiplicity == gt_mult else 0.0

            metrics["multiplicity_mae"].append(mae)
            metrics["multiplicity_accuracy"].append(accuracy)

        # Compute precision, recall, F1 for binary classification (single vs multiple)
        binary_gt = (gt_multiplicity > 1).astype(int)
        binary_pred = []

        for i in range(attn_np.shape[0]):
            estimated_mult = self._estimate_arc_multiplicity(attn_np[i], min_arc_length)
            binary_pred.append(1 if estimated_mult > 1 else 0)

        binary_pred = np.array(binary_pred)

        # Compute binary metrics
        tp = np.sum((binary_pred == 1) & (binary_gt == 1))
        fp = np.sum((binary_pred == 1) & (binary_gt == 0))
        fn = np.sum((binary_pred == 0) & (binary_gt == 1))
        tn = np.sum((binary_pred == 0) & (binary_gt == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics["multiplicity_precision"].append(precision)
        metrics["multiplicity_recall"].append(recall)
        metrics["multiplicity_f1"].append(f1)

        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0.0

        return metrics

    def _estimate_arc_multiplicity(
        self, attention_map: np.ndarray, min_arc_length: float
    ) -> int:
        """
        Estimate number of distinct arcs from attention map.

        Args:
            attention_map: Attention map [H, W]
            min_arc_length: Minimum arc length in pixels

        Returns:
            Estimated number of distinct arcs
        """
        # Threshold attention map
        threshold = 0.5
        binary_map = (attention_map > threshold).astype(np.uint8)

        # Find connected components
        labeled_map = measure.label(binary_map)

        # Count arcs that meet minimum length requirement
        arc_count = 0
        for region in measure.regionprops(labeled_map):
            # Check if region is elongated enough to be an arc
            if region.major_axis_length >= min_arc_length:
                # Check elongation (major/minor axis ratio)
                if region.minor_axis_length > 0:
                    elongation = region.major_axis_length / region.minor_axis_length
                    if elongation >= 2.0:  # Arc-like elongation
                        arc_count += 1

        return arc_count

    def validate_arc_parity(
        self, attention_maps: torch.Tensor, ground_truth_parity: torch.Tensor
    ) -> Dict[str, float]:
        """
        Validate arc parity (orientation) detection.

        Args:
            attention_maps: Attention maps [B, H, W]
            ground_truth_parity: Ground truth arc parities [B] (1 for tangential, -1 for radial)

        Returns:
            Dictionary with parity validation metrics
        """
        attn_np = attention_maps.detach().cpu().numpy()
        gt_parity = ground_truth_parity.detach().cpu().numpy()

        metrics = {
            "parity_accuracy": [],
            "parity_precision_tangential": [],
            "parity_recall_tangential": [],
            "parity_precision_radial": [],
            "parity_recall_radial": [],
        }

        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]
            gt_par = gt_parity[i]

            # Estimate parity from attention map
            estimated_parity = self._estimate_arc_parity(attn_map)

            # Compute accuracy
            accuracy = 1.0 if estimated_parity == gt_par else 0.0
            metrics["parity_accuracy"].append(accuracy)

        # Compute precision and recall for each parity type
        for parity_type in [1, -1]:  # tangential, radial
            tp = np.sum(
                (gt_parity == parity_type)
                & (
                    np.array(
                        [
                            self._estimate_arc_parity(attn_np[i])
                            for i in range(len(gt_parity))
                        ]
                    )
                    == parity_type
                )
            )
            fp = np.sum(
                (gt_parity != parity_type)
                & (
                    np.array(
                        [
                            self._estimate_arc_parity(attn_np[i])
                            for i in range(len(gt_parity))
                        ]
                    )
                    == parity_type
                )
            )
            fn = np.sum(
                (gt_parity == parity_type)
                & (
                    np.array(
                        [
                            self._estimate_arc_parity(attn_np[i])
                            for i in range(len(gt_parity))
                        ]
                    )
                    != parity_type
                )
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            parity_name = "tangential" if parity_type == 1 else "radial"
            metrics[f"parity_precision_{parity_name}"].append(precision)
            metrics[f"parity_recall_{parity_name}"].append(recall)

        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0.0

        return metrics

    def _estimate_arc_parity(self, attention_map: np.ndarray) -> int:
        """
        Estimate arc parity from attention map.

        Args:
            attention_map: Attention map [H, W]

        Returns:
            Estimated parity (1 for tangential, -1 for radial)
        """
        # Find lens center
        center_y, center_x = np.unravel_index(
            np.argmax(attention_map), attention_map.shape
        )

        # Create coordinate grids
        H, W = attention_map.shape
        y, x = np.ogrid[:H, :W]

        # Compute radial and tangential components
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero
        r[r == 0] = 1e-6

        # Radial and tangential unit vectors
        r_hat_x = dx / r
        r_hat_y = dy / r
        t_hat_x = -dy / r  # Perpendicular to radial
        t_hat_y = dx / r

        # Compute gradients
        grad_x = np.gradient(attention_map, axis=1)
        grad_y = np.gradient(attention_map, axis=0)

        # Project gradients onto radial and tangential directions
        radial_gradient = grad_x * r_hat_x + grad_y * r_hat_y
        tangential_gradient = grad_x * t_hat_x + grad_y * t_hat_y

        # Determine parity based on dominant gradient direction
        radial_strength = np.mean(np.abs(radial_gradient))
        tangential_strength = np.mean(np.abs(tangential_gradient))

        return 1 if tangential_strength > radial_strength else -1

    def validate_lensing_equation_residual(
        self,
        attention_maps: torch.Tensor,
        source_positions: torch.Tensor,
        lens_parameters: Dict[str, torch.Tensor],
        pixel_scale: float = 0.1,
    ) -> Dict[str, float]:
        """
        Validate lensing equation residuals.

        Args:
            attention_maps: Attention maps [B, H, W]
            source_positions: Source positions [B, 2] in arcsec
            lens_parameters: Dictionary with lens parameters
            pixel_scale: Pixel scale in arcsec/pixel

        Returns:
            Dictionary with lensing equation validation metrics
        """
        attn_np = attention_map.detach().cpu().numpy()
        source_pos = source_positions.detach().cpu().numpy()

        metrics = {
            "lensing_residual_mae": [],
            "lensing_residual_rmse": [],
            "lensing_residual_max": [],
            "lensing_equation_satisfied": [],
        }

        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]
            src_pos = source_pos[i]

            # Estimate lens center and Einstein radius from attention map
            lens_center = self._estimate_lens_center(attn_map, pixel_scale)
            einstein_radius = self._estimate_einstein_radius(attn_map, pixel_scale)

            if lens_center is not None and einstein_radius is not None:
                # Compute lensing equation residual
                residual = self._compute_lensing_equation_residual(
                    src_pos, lens_center, einstein_radius, attn_map, pixel_scale
                )

                if residual is not None:
                    mae = np.mean(np.abs(residual))
                    rmse = np.sqrt(np.mean(residual**2))
                    max_residual = np.max(np.abs(residual))
                    equation_satisfied = (
                        1.0 if mae < 0.1 else 0.0
                    )  # 0.1 arcsec tolerance

                    metrics["lensing_residual_mae"].append(mae)
                    metrics["lensing_residual_rmse"].append(rmse)
                    metrics["lensing_residual_max"].append(max_residual)
                    metrics["lensing_equation_satisfied"].append(equation_satisfied)

        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0.0

        return metrics

    def _estimate_lens_center(
        self, attention_map: np.ndarray, pixel_scale: float
    ) -> Optional[Tuple[float, float]]:
        """Estimate lens center from attention map."""
        # Find center of mass of high-attention regions
        threshold = 0.5
        high_attention = attention_map > threshold

        if high_attention.sum() == 0:
            return None

        # Compute center of mass
        y_coords, x_coords = np.where(high_attention)
        weights = attention_map[high_attention]

        center_x = np.average(x_coords, weights=weights) * pixel_scale
        center_y = np.average(y_coords, weights=weights) * pixel_scale

        return (center_x, center_y)

    def _compute_lensing_equation_residual(
        self,
        source_position: np.ndarray,
        lens_center: Tuple[float, float],
        einstein_radius: float,
        attention_map: np.ndarray,
        pixel_scale: float,
    ) -> Optional[np.ndarray]:
        """
        Compute lensing equation residual.

        For a point mass lens: β = θ - α(θ)
        where α(θ) = θ_E^2 / |θ| * θ_hat
        """
        # Find image positions from attention map
        image_positions = self._find_image_positions(attention_map, pixel_scale)

        if len(image_positions) == 0:
            return None

        residuals = []
        for img_pos in image_positions:
            # Compute deflection angle for point mass lens
            theta = img_pos - np.array(lens_center)
            theta_mag = np.linalg.norm(theta)

            if theta_mag > 0:
                theta_hat = theta / theta_mag
                alpha = (einstein_radius**2 / theta_mag) * theta_hat

                # Lensing equation: β = θ - α
                predicted_source = img_pos - alpha
                residual = np.linalg.norm(predicted_source - source_position)
                residuals.append(residual)

        return np.array(residuals) if residuals else None

    def _find_image_positions(
        self, attention_map: np.ndarray, pixel_scale: float
    ) -> List[np.ndarray]:
        """Find image positions from attention map."""
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

    def validate_time_delay_distribution(
        self,
        attention_maps: torch.Tensor,
        ground_truth_time_delays: torch.Tensor,
        lens_redshift: float = 0.5,
        source_redshift: float = 2.0,
    ) -> Dict[str, float]:
        """
        Validate time delay distribution for multiply-imaged sources.

        Args:
            attention_maps: Attention maps [B, H, W]
            ground_truth_time_delays: Ground truth time delays [B, N_images] in days
            lens_redshift: Lens redshift
            source_redshift: Source redshift

        Returns:
            Dictionary with time delay validation metrics
        """
        attn_np = attention_maps.detach().cpu().numpy()
        gt_delays = ground_truth_time_delays.detach().cpu().numpy()

        metrics = {
            "time_delay_mae": [],
            "time_delay_rmse": [],
            "time_delay_correlation": [],
            "time_delay_relative_error": [],
        }

        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]
            gt_delay = gt_delays[i]

            # Estimate time delays from attention map
            estimated_delays = self._estimate_time_delays(
                attn_map, lens_redshift, source_redshift
            )

            if estimated_delays is not None and len(estimated_delays) == len(gt_delay):
                # Compute metrics
                mae = np.mean(np.abs(estimated_delays - gt_delay))
                rmse = np.sqrt(np.mean((estimated_delays - gt_delay) ** 2))

                # Relative error
                valid_gt = gt_delay[gt_delay > 0]
                valid_est = estimated_delays[gt_delay > 0]
                if len(valid_gt) > 0:
                    relative_error = np.mean(np.abs(valid_est - valid_gt) / valid_gt)
                else:
                    relative_error = 0.0

                # Correlation
                if len(gt_delay) > 1:
                    correlation = np.corrcoef(estimated_delays, gt_delay)[0, 1]
                    if not np.isnan(correlation):
                        metrics["time_delay_correlation"].append(correlation)

                metrics["time_delay_mae"].append(mae)
                metrics["time_delay_rmse"].append(rmse)
                metrics["time_delay_relative_error"].append(relative_error)

        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0.0

        return metrics

    def _estimate_time_delays(
        self, attention_map: np.ndarray, lens_redshift: float, source_redshift: float
    ) -> Optional[np.ndarray]:
        """
        Estimate time delays from attention map.

        Time delay: Δt = (1 + z_l) * D_l * D_s / (D_ls * c) * (1/2 * |θ - β|² - ψ(θ))
        """
        # Find image positions
        image_positions = self._find_image_positions(
            attention_map, 0.1
        )  # 0.1 arcsec/pixel

        if len(image_positions) < 2:
            return None

        # Estimate lens parameters
        lens_center = self._estimate_lens_center(attention_map, 0.1)
        einstein_radius = self._estimate_einstein_radius(attention_map, 0.1)

        if lens_center is None or einstein_radius is None:
            return None

        # Compute cosmological distances (simplified)
        D_l = self._compute_angular_diameter_distance(lens_redshift)
        D_s = self._compute_angular_diameter_distance(source_redshift)
        D_ls = self._compute_angular_diameter_distance(lens_redshift, source_redshift)

        # Compute time delays
        time_delays = []
        for img_pos in image_positions:
            # Simplified time delay calculation
            # For point mass lens: ψ(θ) = θ_E² * ln(|θ|/θ_E)
            theta = img_pos - np.array(lens_center)
            theta_mag = np.linalg.norm(theta)

            if theta_mag > 0:
                # Potential at image position
                psi = einstein_radius**2 * np.log(theta_mag / einstein_radius)

                # Time delay (simplified)
                time_delay = (1 + lens_redshift) * D_l * D_s / (D_ls * self.c) * psi
                time_delays.append(time_delay)

        return np.array(time_delays) if time_delays else None

    def _compute_angular_diameter_distance(
        self, z1: float, z2: Optional[float] = None
    ) -> float:
        """
        Compute angular diameter distance (simplified).

        For flat universe with Ω_m = 0.3, Ω_Λ = 0.7
        """
        if z2 is None:
            # Distance to redshift z1
            # Simplified calculation - in practice use proper cosmology
            return 1000.0 * z1  # Mpc (very simplified)
        else:
            # Distance between z1 and z2
            return 1000.0 * abs(z2 - z1)  # Mpc (very simplified)


def validate_lensing_physics(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    validator: LensingMetricsValidator = None,
) -> Dict[str, float]:
    """
    Comprehensive lensing physics validation.

    Args:
        model: Model to validate
        test_loader: Test data loader with lensing metadata
        validator: Lensing metrics validator

    Returns:
        Dictionary with comprehensive lensing validation metrics
    """
    if validator is None:
        validator = LensingMetricsValidator()

    model.eval()

    all_metrics = {
        "einstein_radius": [],
        "arc_multiplicity": [],
        "arc_parity": [],
        "lensing_equation": [],
        "time_delays": [],
    }

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(validator.device)

            # Get model outputs with attention
            if hasattr(model, "forward_with_attention"):
                outputs, attention_info = model.forward_with_attention(images)
            else:
                outputs = model(images)
                attention_info = {}

            if "attention_maps" in attention_info:
                attention_maps = attention_info["attention_maps"]

                # Einstein radius validation
                if "einstein_radius" in batch:
                    einstein_radius = batch["einstein_radius"].to(validator.device)
                    einstein_metrics = validator.validate_einstein_radius(
                        attention_maps, einstein_radius
                    )
                    all_metrics["einstein_radius"].append(einstein_metrics)

                # Arc multiplicity validation
                if "arc_multiplicity" in batch:
                    multiplicity = batch["arc_multiplicity"].to(validator.device)
                    multiplicity_metrics = validator.validate_arc_multiplicity(
                        attention_maps, multiplicity
                    )
                    all_metrics["arc_multiplicity"].append(multiplicity_metrics)

                # Arc parity validation
                if "arc_parity" in batch:
                    parity = batch["arc_parity"].to(validator.device)
                    parity_metrics = validator.validate_arc_parity(
                        attention_maps, parity
                    )
                    all_metrics["arc_parity"].append(parity_metrics)

                # Lensing equation validation
                if "source_position" in batch and "lens_parameters" in batch:
                    source_pos = batch["source_position"].to(validator.device)
                    lens_params = batch["lens_parameters"]
                    lensing_metrics = validator.validate_lensing_equation_residual(
                        attention_maps, source_pos, lens_params
                    )
                    all_metrics["lensing_equation"].append(lensing_metrics)

                # Time delay validation
                if "time_delays" in batch:
                    time_delays = batch["time_delays"].to(validator.device)
                    delay_metrics = validator.validate_time_delay_distribution(
                        attention_maps, time_delays
                    )
                    all_metrics["time_delays"].append(delay_metrics)

    # Average all metrics
    final_metrics = {}
    for category, metrics_list in all_metrics.items():
        if metrics_list:
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    final_metrics[f"{category}_{key}"] = np.mean(values)

    return final_metrics


def create_lensing_validation_report(
    validation_results: Dict[str, float], save_path: Optional[str] = None
) -> str:
    """
    Create comprehensive lensing validation report.

    Args:
        validation_results: Validation results dictionary
        save_path: Optional path to save report

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 100)
    report.append("COMPREHENSIVE LENSING PHYSICS VALIDATION REPORT")
    report.append("=" * 100)

    # Einstein radius validation
    report.append("\nEINSTEIN RADIUS VALIDATION:")
    report.append("-" * 50)
    einstein_keys = [
        k for k in validation_results.keys() if k.startswith("einstein_radius_")
    ]
    for key in einstein_keys:
        metric_name = key.replace("einstein_radius_", "")
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")

    # Arc multiplicity validation
    report.append("\nARC MULTIPLICITY VALIDATION:")
    report.append("-" * 50)
    multiplicity_keys = [
        k for k in validation_results.keys() if k.startswith("arc_multiplicity_")
    ]
    for key in multiplicity_keys:
        metric_name = key.replace("arc_multiplicity_", "")
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")

    # Arc parity validation
    report.append("\nARC PARITY VALIDATION:")
    report.append("-" * 50)
    parity_keys = [k for k in validation_results.keys() if k.startswith("arc_parity_")]
    for key in parity_keys:
        metric_name = key.replace("arc_parity_", "")
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")

    # Lensing equation validation
    report.append("\nLENSING EQUATION VALIDATION:")
    report.append("-" * 50)
    lensing_keys = [
        k for k in validation_results.keys() if k.startswith("lensing_equation_")
    ]
    for key in lensing_keys:
        metric_name = key.replace("lensing_equation_", "")
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")

    # Time delay validation
    report.append("\nTIME DELAY VALIDATION:")
    report.append("-" * 50)
    delay_keys = [k for k in validation_results.keys() if k.startswith("time_delays_")]
    for key in delay_keys:
        metric_name = key.replace("time_delays_", "")
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")

    # Overall physics score
    report.append("\nOVERALL LENSING PHYSICS SCORE:")
    report.append("-" * 50)

    physics_components = []
    if "einstein_radius_within_tolerance" in validation_results:
        physics_components.append(
            validation_results["einstein_radius_within_tolerance"]
        )
    if "arc_multiplicity_accuracy" in validation_results:
        physics_components.append(validation_results["arc_multiplicity_accuracy"])
    if "arc_parity_accuracy" in validation_results:
        physics_components.append(validation_results["arc_parity_accuracy"])
    if "lensing_equation_satisfied" in validation_results:
        physics_components.append(validation_results["lensing_equation_satisfied"])

    if physics_components:
        overall_score = np.mean(physics_components)
        report.append(f"  Overall Lensing Physics Score: {overall_score:.4f}")

        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        report.append("-" * 50)

        if overall_score < 0.5:
            report.append("  - Significant physics validation issues detected")
            report.append("  - Consider retraining with physics-regularized loss")
            report.append("  - Validate attention mechanisms on known lens systems")
        elif overall_score < 0.7:
            report.append("  - Good physics alignment with room for improvement")
            report.append("  - Fine-tune attention mechanisms for better physics")
        else:
            report.append("  - Excellent physics alignment")
            report.append("  - Model ready for scientific deployment")

    report.append("=" * 100)

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report_text)

    return report_text
