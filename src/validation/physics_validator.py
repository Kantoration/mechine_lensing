#!/usr/bin/env python3
"""
physics_validator.py
====================
Physics validation framework for lensing attention mechanisms.

Key Features:
- Validation against known lens models
- Physics constraint verification
- Attention map analysis for scientific interpretation
- Benchmarking against classical methods

Usage:
    from validation.physics_validator import PhysicsValidator, validate_attention_physics
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure

logger = logging.getLogger(__name__)


class PhysicsValidator:
    """
    Physics validation framework for lensing attention mechanisms.

    This class provides comprehensive validation of attention mechanisms
    against known physics principles and classical detection methods.
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize physics validator.

        Args:
            device: Device for computations
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Physics constants
        self.c = 299792458  # Speed of light (m/s)
        self.G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)

        logger.info(f"Physics validator initialized on {self.device}")

    def validate_arc_detection(
        self,
        attention_maps: torch.Tensor,
        ground_truth_arcs: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Validate arc detection performance of attention maps.

        Args:
            attention_maps: Attention maps [B, H, W]
            ground_truth_arcs: Ground truth arc masks [B, H, W]
            threshold: Threshold for binary attention maps

        Returns:
            Dictionary with validation metrics
        """
        # Convert to numpy for classical analysis
        attn_np = attention_maps.detach().cpu().numpy()
        gt_np = ground_truth_arcs.detach().cpu().numpy()

        metrics = {
            "precision": [],
            "recall": [],
            "f1_score": [],
            "iou": [],
            "arc_completeness": [],
            "arc_elongation": [],
        }

        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]
            gt_map = gt_np[i]

            # Binary attention map
            attn_binary = (attn_map > threshold).astype(np.uint8)

            # Classical metrics
            precision, recall, f1, iou = self._compute_classical_metrics(
                attn_binary, gt_map
            )
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1_score"].append(f1)
            metrics["iou"].append(iou)

            # Physics-based metrics
            completeness = self._compute_arc_completeness(attn_binary, gt_map)
            elongation = self._compute_arc_elongation(attn_binary)
            metrics["arc_completeness"].append(completeness)
            metrics["arc_elongation"].append(elongation)

        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key])

        return metrics

    def _compute_classical_metrics(
        self, pred: np.ndarray, gt: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Compute classical detection metrics."""
        # Intersection and union
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        # Avoid division by zero
        if union == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Metrics
        precision = intersection / pred.sum() if pred.sum() > 0 else 0.0
        recall = intersection / gt.sum() if gt.sum() > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou = intersection / union

        return precision, recall, f1, iou

    def _compute_arc_completeness(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute arc completeness (fraction of true arc detected)."""
        if gt.sum() == 0:
            return 1.0 if pred.sum() == 0 else 0.0

        # Find connected components in ground truth
        gt_components = measure.label(gt)

        completeness_scores = []
        for region in measure.regionprops(gt_components):
            # Extract region
            minr, minc, maxr, maxc = region.bbox
            gt_region = gt[minr:maxr, minc:maxc]
            pred_region = pred[minr:maxr, minc:maxc]

            # Compute completeness for this region
            intersection = np.logical_and(pred_region, gt_region).sum()
            completeness = intersection / gt_region.sum()
            completeness_scores.append(completeness)

        return np.mean(completeness_scores) if completeness_scores else 0.0

    def _compute_arc_elongation(self, pred: np.ndarray) -> float:
        """Compute average elongation of detected arcs."""
        if pred.sum() == 0:
            return 0.0

        # Find connected components
        components = measure.label(pred)

        elongation_scores = []
        for region in measure.regionprops(components):
            # Compute elongation (major_axis_length / minor_axis_length)
            if region.minor_axis_length > 0:
                elongation = region.major_axis_length / region.minor_axis_length
                elongation_scores.append(elongation)

        return np.mean(elongation_scores) if elongation_scores else 0.0

    def validate_curvature_detection(
        self,
        attention_maps: torch.Tensor,
        ground_truth_curvature: torch.Tensor,
        curvature_threshold: float = 0.1,
    ) -> Dict[str, float]:
        """
        Validate curvature detection performance.

        Args:
            attention_maps: Attention maps [B, H, W]
            ground_truth_curvature: Ground truth curvature maps [B, H, W]
            curvature_threshold: Threshold for significant curvature

        Returns:
            Dictionary with curvature validation metrics
        """
        attn_np = attention_maps.detach().cpu().numpy()
        gt_curvature = ground_truth_curvature.detach().cpu().numpy()

        metrics = {
            "curvature_correlation": [],
            "curvature_precision": [],
            "curvature_recall": [],
            "curvature_f1": [],
        }

        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]
            gt_curv = gt_curvature[i]

            # Compute curvature correlation
            correlation = np.corrcoef(attn_map.flatten(), gt_curv.flatten())[0, 1]
            if not np.isnan(correlation):
                metrics["curvature_correlation"].append(correlation)

            # Binary curvature detection
            attn_curvature = (attn_map > curvature_threshold).astype(np.uint8)
            gt_curvature_binary = (gt_curv > curvature_threshold).astype(np.uint8)

            # Classical metrics
            precision, recall, f1, _ = self._compute_classical_metrics(
                attn_curvature, gt_curvature_binary
            )
            metrics["curvature_precision"].append(precision)
            metrics["curvature_recall"].append(recall)
            metrics["curvature_f1"].append(f1)

        # Average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if metrics[key] else 0.0

        return metrics

    def validate_physics_constraints(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        expected_physics: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Validate that the model respects physics constraints.

        Args:
            model: Model to validate
            test_images: Test images [B, C, H, W]
            expected_physics: Expected physics properties

        Returns:
            Dictionary with physics validation metrics
        """
        model.eval()

        with torch.no_grad():
            # Get model outputs
            if hasattr(model, "forward_with_attention"):
                outputs, attention_info = model.forward_with_attention(test_images)
            else:
                outputs = model(test_images)
                attention_info = {}

            # Validate attention maps
            physics_metrics = {}

            if "attention_maps" in attention_info:
                attention_maps = attention_info["attention_maps"]

                # Validate attention properties
                physics_metrics.update(
                    self._validate_attention_properties(attention_maps)
                )

                # Validate against expected physics
                if "expected_arcs" in expected_physics:
                    arc_metrics = self.validate_arc_detection(
                        attention_maps, expected_physics["expected_arcs"]
                    )
                    physics_metrics.update(arc_metrics)

                if "expected_curvature" in expected_physics:
                    curvature_metrics = self.validate_curvature_detection(
                        attention_maps, expected_physics["expected_curvature"]
                    )
                    physics_metrics.update(curvature_metrics)

            # Validate model predictions
            physics_metrics.update(
                self._validate_prediction_physics(outputs, expected_physics)
            )

        return physics_metrics

    def _validate_attention_properties(
        self, attention_maps: torch.Tensor
    ) -> Dict[str, float]:
        """Validate basic attention map properties."""
        metrics = {}

        # Attention map statistics
        metrics["attention_mean"] = attention_maps.mean().item()
        metrics["attention_std"] = attention_maps.std().item()
        metrics["attention_max"] = attention_maps.max().item()
        metrics["attention_min"] = attention_maps.min().item()

        # Attention sparsity (fraction of high-attention pixels)
        high_attention = (attention_maps > 0.5).float()
        metrics["attention_sparsity"] = high_attention.mean().item()

        # Attention smoothness (local variation)
        # Compute gradient magnitude
        grad_x = torch.abs(attention_maps[:, :, 1:] - attention_maps[:, :, :-1])
        grad_y = torch.abs(attention_maps[:, :, 1:, :] - attention_maps[:, :, :-1, :])
        smoothness = 1.0 / (1.0 + grad_x.mean() + grad_y.mean())
        metrics["attention_smoothness"] = smoothness.item()

        return metrics

    def _validate_prediction_physics(
        self, predictions: torch.Tensor, expected_physics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate that predictions respect physics constraints."""
        metrics = {}

        # Convert predictions to probabilities
        if predictions.dim() > 1 and predictions.shape[-1] > 1:
            probs = F.softmax(predictions, dim=-1)
        else:
            probs = torch.sigmoid(predictions)

        # Validate prediction statistics
        metrics["prediction_mean"] = probs.mean().item()
        metrics["prediction_std"] = probs.std().item()

        # Validate against expected lensing properties
        if "expected_lens_fraction" in expected_physics:
            expected_fraction = expected_physics["expected_lens_fraction"]
            actual_fraction = (probs > 0.5).float().mean().item()
            metrics["lens_fraction_error"] = abs(actual_fraction - expected_fraction)

        return metrics

    def benchmark_against_classical(
        self,
        attention_maps: torch.Tensor,
        ground_truth: torch.Tensor,
        classical_methods: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark attention-based detection against classical methods.

        Args:
            attention_maps: Attention maps [B, H, W]
            ground_truth: Ground truth masks [B, H, W]
            classical_methods: List of classical methods to benchmark

        Returns:
            Dictionary with benchmark results
        """
        if classical_methods is None:
            classical_methods = ["canny", "sobel", "laplacian", "gabor"]

        results = {}

        # Attention-based results
        attention_metrics = self.validate_arc_detection(attention_maps, ground_truth)
        results["attention"] = attention_metrics

        # Classical method results
        for method in classical_methods:
            classical_maps = self._apply_classical_method(attention_maps, method)
            classical_metrics = self.validate_arc_detection(
                classical_maps, ground_truth
            )
            results[method] = classical_metrics

        return results

    def _apply_classical_method(
        self, attention_maps: torch.Tensor, method: str
    ) -> torch.Tensor:
        """Apply classical edge/arc detection method."""
        attn_np = attention_maps.detach().cpu().numpy()
        classical_maps = []

        for i in range(attn_np.shape[0]):
            attn_map = attn_np[i]

            if method == "canny":
                from skimage import feature

                classical_map = feature.canny(attn_map, sigma=1.0)
            elif method == "sobel":
                from skimage import filters

                classical_map = filters.sobel(attn_map)
            elif method == "laplacian":
                from skimage import filters

                classical_map = filters.laplace(attn_map)
            elif method == "gabor":
                from skimage import filters

                classical_map = filters.gabor(attn_map, frequency=0.1)[0]
            else:
                classical_map = attn_map

            classical_maps.append(classical_map)

        return torch.tensor(np.stack(classical_maps), device=attention_maps.device)


def validate_attention_physics(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    validator: PhysicsValidator = None,
) -> Dict[str, float]:
    """
    Comprehensive physics validation of attention mechanisms.

    Args:
        model: Model to validate
        test_loader: Test data loader
        validator: Physics validator instance

    Returns:
        Dictionary with comprehensive validation metrics
    """
    if validator is None:
        validator = PhysicsValidator()

    model.eval()

    all_metrics = {
        "attention_properties": [],
        "arc_detection": [],
        "curvature_detection": [],
        "physics_constraints": [],
    }

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(validator.device)
            labels = batch["label"].to(validator.device)

            # Get model outputs with attention
            if hasattr(model, "forward_with_attention"):
                outputs, attention_info = model.forward_with_attention(images)
            else:
                outputs = model(images)
                attention_info = {}

            # Validate attention properties
            if "attention_maps" in attention_info:
                attention_maps = attention_info["attention_maps"]

                # Basic attention properties
                attn_props = validator._validate_attention_properties(attention_maps)
                all_metrics["attention_properties"].append(attn_props)

                # Arc detection (if ground truth available)
                if "arc_mask" in batch:
                    arc_masks = batch["arc_mask"].to(validator.device)
                    arc_metrics = validator.validate_arc_detection(
                        attention_maps, arc_masks
                    )
                    all_metrics["arc_detection"].append(arc_metrics)

                # Curvature detection (if ground truth available)
                if "curvature_map" in batch:
                    curvature_maps = batch["curvature_map"].to(validator.device)
                    curvature_metrics = validator.validate_curvature_detection(
                        attention_maps, curvature_maps
                    )
                    all_metrics["curvature_detection"].append(curvature_metrics)

            # Physics constraints validation
            expected_physics = {"expected_lens_fraction": labels.float().mean().item()}
            physics_metrics = validator.validate_physics_constraints(
                model, images, expected_physics
            )
            all_metrics["physics_constraints"].append(physics_metrics)

    # Average all metrics
    final_metrics = {}
    for category, metrics_list in all_metrics.items():
        if metrics_list:
            # Average across batches
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    final_metrics[f"{category}_{key}"] = np.mean(values)

    return final_metrics


def create_physics_validation_report(
    validation_results: Dict[str, float], save_path: Optional[str] = None
) -> str:
    """
    Create a comprehensive physics validation report.

    Args:
        validation_results: Validation results dictionary
        save_path: Optional path to save report

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("PHYSICS VALIDATION REPORT")
    report.append("=" * 80)

    # Attention properties
    report.append("\nATTENTION PROPERTIES:")
    report.append("-" * 40)
    attn_keys = [
        k for k in validation_results.keys() if k.startswith("attention_properties_")
    ]
    for key in attn_keys:
        metric_name = key.replace("attention_properties_", "")
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")

    # Arc detection
    report.append("\nARC DETECTION PERFORMANCE:")
    report.append("-" * 40)
    arc_keys = [k for k in validation_results.keys() if k.startswith("arc_detection_")]
    for key in arc_keys:
        metric_name = key.replace("arc_detection_", "")
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")

    # Curvature detection
    report.append("\nCURVATURE DETECTION PERFORMANCE:")
    report.append("-" * 40)
    curv_keys = [
        k for k in validation_results.keys() if k.startswith("curvature_detection_")
    ]
    for key in curv_keys:
        metric_name = key.replace("curvature_detection_", "")
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")

    # Physics constraints
    report.append("\nPHYSICS CONSTRAINTS:")
    report.append("-" * 40)
    physics_keys = [
        k for k in validation_results.keys() if k.startswith("physics_constraints_")
    ]
    for key in physics_keys:
        metric_name = key.replace("physics_constraints_", "")
        value = validation_results[key]
        report.append(f"  {metric_name}: {value:.4f}")

    # Summary
    report.append("\nSUMMARY:")
    report.append("-" * 40)

    # Overall physics score
    physics_score = 0.0
    if "arc_detection_f1_score" in validation_results:
        physics_score += validation_results["arc_detection_f1_score"]
    if "curvature_detection_curvature_correlation" in validation_results:
        physics_score += validation_results["curvature_detection_curvature_correlation"]
    if "physics_constraints_lens_fraction_error" in validation_results:
        physics_score += (
            1.0 - validation_results["physics_constraints_lens_fraction_error"]
        )

    physics_score = physics_score / 3.0  # Average of three components
    report.append(f"  Overall Physics Score: {physics_score:.4f}")

    # Recommendations
    report.append("\nRECOMMENDATIONS:")
    report.append("-" * 40)

    if physics_score < 0.5:
        report.append("  - Consider increasing physics regularization weight")
        report.append("  - Validate against more diverse lensing scenarios")
        report.append("  - Check attention map interpretability")
    elif physics_score < 0.7:
        report.append("  - Good physics alignment, minor improvements possible")
        report.append("  - Consider fine-tuning attention mechanisms")
    else:
        report.append("  - Excellent physics alignment")
        report.append("  - Model ready for scientific applications")

    report.append("=" * 80)

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report_text)

    return report_text
