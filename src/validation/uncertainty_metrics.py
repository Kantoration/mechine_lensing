#!/usr/bin/env python3
"""
uncertainty_metrics.py
======================
Uncertainty quantification metrics for scientific inference.

Key Features:
- Predictive uncertainty estimation
- Calibration assessment
- Confidence interval validation
- Epistemic vs aleatoric uncertainty separation
- Scientific inference reliability metrics

Usage:
    from validation.uncertainty_metrics import UncertaintyValidator, validate_predictive_uncertainty
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class UncertaintyValidator:
    """
    Validator for uncertainty quantification in scientific inference.
    
    This validator provides comprehensive assessment of predictive uncertainty,
    calibration, and reliability for scientific applications.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize uncertainty validator.
        
        Args:
            device: Device for computations
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Uncertainty validator initialized on {self.device}")
    
    def validate_predictive_uncertainty(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        ground_truth: torch.Tensor,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[str, float]:
        """
        Validate predictive uncertainty estimates.
        
        Args:
            predictions: Model predictions [B]
            uncertainties: Uncertainty estimates [B]
            ground_truth: Ground truth values [B]
            confidence_levels: List of confidence levels to validate
            
        Returns:
            Dictionary with uncertainty validation metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        unc_np = uncertainties.detach().cpu().numpy()
        gt_np = ground_truth.detach().cpu().numpy()
        
        metrics = {}
        
        # Coverage analysis
        for conf_level in confidence_levels:
            coverage = self._compute_coverage(pred_np, unc_np, gt_np, conf_level)
            metrics[f'coverage_{conf_level}'] = coverage
        
        # Calibration metrics
        metrics.update(self._compute_calibration_metrics(pred_np, unc_np, gt_np))
        
        # Sharpness metrics
        metrics.update(self._compute_sharpness_metrics(unc_np))
        
        # Reliability metrics
        metrics.update(self._compute_reliability_metrics(pred_np, unc_np, gt_np))
        
        return metrics
    
    def _compute_coverage(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        ground_truth: np.ndarray,
        confidence_level: float
    ) -> float:
        """Compute coverage for given confidence level."""
        # Compute confidence intervals
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = predictions - z_score * uncertainties
        upper_bound = predictions + z_score * uncertainties
        
        # Check coverage
        covered = (ground_truth >= lower_bound) & (ground_truth <= upper_bound)
        coverage = np.mean(covered)
        
        return coverage
    
    def _compute_calibration_metrics(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """Compute calibration metrics."""
        metrics = {}
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(predictions, uncertainties, ground_truth)
        metrics['ece'] = ece
        
        # Maximum Calibration Error (MCE)
        mce = self._compute_mce(predictions, uncertainties, ground_truth)
        metrics['mce'] = mce
        
        # Reliability diagram metrics
        reliability_metrics = self._compute_reliability_diagram_metrics(
            predictions, uncertainties, ground_truth
        )
        metrics.update(reliability_metrics)
        
        return metrics
    
    def _compute_ece(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        ground_truth: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        # Convert to probabilities for binary classification
        if predictions.min() >= 0 and predictions.max() <= 1:
            probs = predictions
        else:
            probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = ground_truth[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_mce(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        ground_truth: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Maximum Calibration Error."""
        # Convert to probabilities for binary classification
        if predictions.min() >= 0 and predictions.max() <= 1:
            probs = predictions
        else:
            probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = ground_truth[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _compute_reliability_diagram_metrics(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """Compute reliability diagram metrics."""
        metrics = {}
        
        # Convert to probabilities
        if predictions.min() >= 0 and predictions.max() <= 1:
            probs = predictions
        else:
            probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        
        # Compute reliability curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            ground_truth, probs, n_bins=10
        )
        
        # Reliability metrics
        metrics['reliability_mse'] = np.mean((fraction_of_positives - mean_predicted_value)**2)
        metrics['reliability_mae'] = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Brier score
        brier_score = np.mean((probs - ground_truth)**2)
        metrics['brier_score'] = brier_score
        
        return metrics
    
    def _compute_sharpness_metrics(self, uncertainties: np.ndarray) -> Dict[str, float]:
        """Compute sharpness metrics."""
        metrics = {}
        
        # Average uncertainty
        metrics['mean_uncertainty'] = np.mean(uncertainties)
        metrics['std_uncertainty'] = np.std(uncertainties)
        
        # Uncertainty distribution
        metrics['uncertainty_entropy'] = self._compute_uncertainty_entropy(uncertainties)
        
        # Sharpness (inverse of average uncertainty)
        metrics['sharpness'] = 1.0 / (1.0 + metrics['mean_uncertainty'])
        
        return metrics
    
    def _compute_uncertainty_entropy(self, uncertainties: np.ndarray) -> float:
        """Compute entropy of uncertainty distribution."""
        # Normalize uncertainties to probabilities
        unc_normalized = uncertainties / (uncertainties.sum() + 1e-8)
        
        # Compute entropy
        entropy = -np.sum(unc_normalized * np.log(unc_normalized + 1e-8))
        
        return entropy
    
    def _compute_reliability_metrics(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """Compute reliability metrics for scientific inference."""
        metrics = {}
        
        # Prediction accuracy
        if predictions.min() >= 0 and predictions.max() <= 1:
            probs = predictions
        else:
            probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        
        pred_labels = (probs > 0.5).astype(int)
        accuracy = np.mean(pred_labels == ground_truth)
        metrics['accuracy'] = accuracy
        
        # Confidence-weighted accuracy
        conf_weights = 1.0 / (uncertainties + 1e-8)
        conf_weighted_acc = np.average(pred_labels == ground_truth, weights=conf_weights)
        metrics['confidence_weighted_accuracy'] = conf_weighted_acc
        
        # Uncertainty correlation with error
        errors = np.abs(probs - ground_truth)
        uncertainty_error_corr = np.corrcoef(uncertainties, errors)[0, 1]
        if not np.isnan(uncertainty_error_corr):
            metrics['uncertainty_error_correlation'] = uncertainty_error_corr
        
        # High-confidence accuracy
        high_conf_mask = uncertainties < np.percentile(uncertainties, 25)
        if high_conf_mask.sum() > 0:
            high_conf_acc = np.mean(pred_labels[high_conf_mask] == ground_truth[high_conf_mask])
            metrics['high_confidence_accuracy'] = high_conf_acc
        
        # Low-confidence accuracy
        low_conf_mask = uncertainties > np.percentile(uncertainties, 75)
        if low_conf_mask.sum() > 0:
            low_conf_acc = np.mean(pred_labels[low_conf_mask] == ground_truth[low_conf_mask])
            metrics['low_confidence_accuracy'] = low_conf_acc
        
        return metrics
    
    def validate_epistemic_aleatoric_separation(
        self,
        epistemic_uncertainties: torch.Tensor,
        aleatoric_uncertainties: torch.Tensor,
        ground_truth: torch.Tensor,
        predictions: torch.Tensor
    ) -> Dict[str, float]:
        """
        Validate separation of epistemic and aleatoric uncertainties.
        
        Args:
            epistemic_uncertainties: Epistemic uncertainty estimates [B]
            aleatoric_uncertainties: Aleatoric uncertainty estimates [B]
            ground_truth: Ground truth values [B]
            predictions: Model predictions [B]
            
        Returns:
            Dictionary with epistemic/aleatoric validation metrics
        """
        epi_np = epistemic_uncertainties.detach().cpu().numpy()
        ale_np = aleatoric_uncertainties.detach().cpu().numpy()
        gt_np = ground_truth.detach().cpu().numpy()
        pred_np = predictions.detach().cpu().numpy()
        
        metrics = {}
        
        # Total uncertainty
        total_uncertainty = epi_np + ale_np
        
        # Epistemic uncertainty should correlate with model uncertainty
        # (higher when model is less certain about its predictions)
        model_uncertainty = np.abs(pred_np - 0.5)  # Distance from decision boundary
        epi_correlation = np.corrcoef(epi_np, model_uncertainty)[0, 1]
        if not np.isnan(epi_correlation):
            metrics['epistemic_model_correlation'] = epi_correlation
        
        # Aleatoric uncertainty should correlate with data noise
        # (higher for inherently noisy samples)
        data_noise = np.abs(pred_np - gt_np)  # Prediction error as proxy for data noise
        ale_correlation = np.corrcoef(ale_np, data_noise)[0, 1]
        if not np.isnan(ale_correlation):
            metrics['aleatoric_data_correlation'] = ale_correlation
        
        # Epistemic should be higher for out-of-distribution samples
        # (we can't test this without OOD data, but we can check variance)
        metrics['epistemic_variance'] = np.var(epi_np)
        metrics['aleatoric_variance'] = np.var(ale_np)
        
        # Ratio of epistemic to total uncertainty
        epi_ratio = np.mean(epi_np / (total_uncertainty + 1e-8))
        ale_ratio = np.mean(ale_np / (total_uncertainty + 1e-8))
        metrics['epistemic_ratio'] = epi_ratio
        metrics['aleatoric_ratio'] = ale_ratio
        
        # Mutual information between epistemic and aleatoric
        # (should be low if well separated)
        if len(epi_np) > 1 and len(ale_np) > 1:
            mi = self._compute_mutual_information(epi_np, ale_np)
            metrics['epistemic_aleatoric_mi'] = mi
        
        return metrics
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two variables."""
        # Discretize variables
        x_discrete = np.digitize(x, np.linspace(x.min(), x.max(), 10))
        y_discrete = np.digitize(y, np.linspace(y.min(), y.max(), 10))
        
        # Compute joint and marginal distributions
        joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=10)
        joint_prob = joint_hist / joint_hist.sum()
        
        x_prob = joint_prob.sum(axis=1)
        y_prob = joint_prob.sum(axis=0)
        
        # Compute mutual information
        mi = 0
        for i in range(joint_prob.shape[0]):
            for j in range(joint_prob.shape[1]):
                if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
        
        return mi
    
    def validate_confidence_intervals(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        ground_truth: torch.Tensor,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[str, float]:
        """
        Validate confidence interval construction.
        
        Args:
            predictions: Model predictions [B]
            uncertainties: Uncertainty estimates [B]
            ground_truth: Ground truth values [B]
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with confidence interval validation metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        unc_np = uncertainties.detach().cpu().numpy()
        gt_np = ground_truth.detach().cpu().numpy()
        
        metrics = {}
        
        for conf_level in confidence_levels:
            # Compute confidence intervals
            alpha = 1 - conf_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower_bound = pred_np - z_score * unc_np
            upper_bound = pred_np + z_score * unc_np
            
            # Coverage
            covered = (gt_np >= lower_bound) & (gt_np <= upper_bound)
            coverage = np.mean(covered)
            metrics[f'ci_coverage_{conf_level}'] = coverage
            
            # Interval width
            interval_width = np.mean(upper_bound - lower_bound)
            metrics[f'ci_width_{conf_level}'] = interval_width
            
            # Calibration error
            calibration_error = abs(coverage - conf_level)
            metrics[f'ci_calibration_error_{conf_level}'] = calibration_error
        
        return metrics
    
    def validate_uncertainty_calibration(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Validate uncertainty calibration for scientific inference.
        
        Args:
            predictions: Model predictions [B]
            uncertainties: Uncertainty estimates [B]
            ground_truth: Ground truth values [B]
            
        Returns:
            Dictionary with calibration validation metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        unc_np = uncertainties.detach().cpu().numpy()
        gt_np = ground_truth.detach().cpu().numpy()
        
        metrics = {}
        
        # Convert to probabilities if needed
        if pred_np.min() >= 0 and pred_np.max() <= 1:
            probs = pred_np
        else:
            probs = torch.sigmoid(torch.tensor(pred_np)).numpy()
        
        # Temperature scaling for calibration
        temperature = self._fit_temperature_scaling(probs, gt_np)
        metrics['temperature'] = temperature
        
        # Calibrated predictions
        calibrated_probs = self._apply_temperature_scaling(probs, temperature)
        
        # Calibration metrics on calibrated predictions
        calibrated_metrics = self._compute_calibration_metrics(
            calibrated_probs, unc_np, gt_np
        )
        for key, value in calibrated_metrics.items():
            metrics[f'calibrated_{key}'] = value
        
        # Reliability diagram for calibrated predictions
        reliability_metrics = self._compute_reliability_diagram_metrics(
            calibrated_probs, unc_np, gt_np
        )
        for key, value in reliability_metrics.items():
            metrics[f'calibrated_{key}'] = value
        
        return metrics
    
    def _fit_temperature_scaling(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """Fit temperature scaling for calibration."""
        # Convert to logits
        logits = np.log(predictions / (1 - predictions + 1e-8))
        
        # Fit temperature using cross-entropy loss
        def temperature_loss(temp):
            scaled_logits = logits / temp
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            loss = -np.mean(ground_truth * np.log(scaled_probs + 1e-8) + 
                           (1 - ground_truth) * np.log(1 - scaled_probs + 1e-8))
            return loss
        
        # Optimize temperature
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
        
        return result.x if result.success else 1.0
    
    def _apply_temperature_scaling(
        self,
        predictions: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling to predictions."""
        logits = np.log(predictions / (1 - predictions + 1e-8))
        scaled_logits = logits / temperature
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        
        return scaled_probs


def validate_predictive_uncertainty(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    validator: UncertaintyValidator = None
) -> Dict[str, float]:
    """
    Comprehensive uncertainty validation for scientific inference.
    
    Args:
        model: Model to validate
        test_loader: Test data loader
        validator: Uncertainty validator
        
    Returns:
        Dictionary with comprehensive uncertainty validation metrics
    """
    if validator is None:
        validator = UncertaintyValidator()
    
    model.eval()
    
    all_predictions = []
    all_uncertainties = []
    all_ground_truth = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(validator.device)
            labels = batch['label'].to(validator.device)
            
            # Get model outputs with uncertainty
            if hasattr(model, 'forward_with_uncertainty'):
                outputs, uncertainty_info = model.forward_with_uncertainty(images)
            else:
                outputs = model(images)
                uncertainty_info = {}
            
            # Collect predictions and uncertainties
            all_predictions.append(outputs.cpu())
            all_ground_truth.append(labels.cpu())
            
            if 'uncertainty' in uncertainty_info:
                all_uncertainties.append(uncertainty_info['uncertainty'].cpu())
            
            if 'epistemic_uncertainty' in uncertainty_info:
                all_epistemic.append(uncertainty_info['epistemic_uncertainty'].cpu())
            
            if 'aleatoric_uncertainty' in uncertainty_info:
                all_aleatoric.append(uncertainty_info['aleatoric_uncertainty'].cpu())
    
    # Concatenate all data
    predictions = torch.cat(all_predictions, dim=0)
    ground_truth = torch.cat(all_ground_truth, dim=0)
    
    all_metrics = {}
    
    # Basic uncertainty validation
    if all_uncertainties:
        uncertainties = torch.cat(all_uncertainties, dim=0)
        uncertainty_metrics = validator.validate_predictive_uncertainty(
            predictions, uncertainties, ground_truth
        )
        all_metrics.update(uncertainty_metrics)
    
    # Epistemic/aleatoric separation validation
    if all_epistemic and all_aleatoric:
        epistemic = torch.cat(all_epistemic, dim=0)
        aleatoric = torch.cat(all_aleatoric, dim=0)
        separation_metrics = validator.validate_epistemic_aleatoric_separation(
            epistemic, aleatoric, ground_truth, predictions
        )
        all_metrics.update(separation_metrics)
    
    # Confidence interval validation
    if all_uncertainties:
        ci_metrics = validator.validate_confidence_intervals(
            predictions, uncertainties, ground_truth
        )
        all_metrics.update(ci_metrics)
    
    # Uncertainty calibration validation
    if all_uncertainties:
        calibration_metrics = validator.validate_uncertainty_calibration(
            predictions, uncertainties, ground_truth
        )
        all_metrics.update(calibration_metrics)
    
    return all_metrics


def create_uncertainty_validation_report(
    validation_results: Dict[str, float],
    save_path: Optional[str] = None
) -> str:
    """
    Create comprehensive uncertainty validation report.
    
    Args:
        validation_results: Validation results dictionary
        save_path: Optional path to save report
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 100)
    report.append("COMPREHENSIVE UNCERTAINTY QUANTIFICATION VALIDATION REPORT")
    report.append("=" * 100)
    
    # Coverage analysis
    report.append("\nCOVERAGE ANALYSIS:")
    report.append("-" * 50)
    coverage_keys = [k for k in validation_results.keys() if k.startswith('coverage_')]
    for key in coverage_keys:
        conf_level = key.replace('coverage_', '')
        value = validation_results[key]
        report.append(f"  {conf_level}% confidence coverage: {value:.4f}")
    
    # Calibration metrics
    report.append("\nCALIBRATION METRICS:")
    report.append("-" * 50)
    calibration_keys = [k for k in validation_results.keys() if k.startswith(('ece', 'mce', 'brier_score'))]
    for key in calibration_keys:
        value = validation_results[key]
        report.append(f"  {key}: {value:.4f}")
    
    # Sharpness metrics
    report.append("\nSHARPNESS METRICS:")
    report.append("-" * 50)
    sharpness_keys = [k for k in validation_results.keys() if k.startswith(('mean_uncertainty', 'sharpness'))]
    for key in sharpness_keys:
        value = validation_results[key]
        report.append(f"  {key}: {value:.4f}")
    
    # Reliability metrics
    report.append("\nRELIABILITY METRICS:")
    report.append("-" * 50)
    reliability_keys = [k for k in validation_results.keys() if k.startswith(('accuracy', 'confidence_weighted'))]
    for key in reliability_keys:
        value = validation_results[key]
        report.append(f"  {key}: {value:.4f}")
    
    # Epistemic/Aleatoric separation
    report.append("\nEPISTEMIC/ALEATORIC SEPARATION:")
    report.append("-" * 50)
    separation_keys = [k for k in validation_results.keys() if k.startswith(('epistemic_', 'aleatoric_'))]
    for key in separation_keys:
        value = validation_results[key]
        report.append(f"  {key}: {value:.4f}")
    
    # Confidence intervals
    report.append("\nCONFIDENCE INTERVAL VALIDATION:")
    report.append("-" * 50)
    ci_keys = [k for k in validation_results.keys() if k.startswith('ci_')]
    for key in ci_keys:
        value = validation_results[key]
        report.append(f"  {key}: {value:.4f}")
    
    # Overall uncertainty score
    report.append("\nOVERALL UNCERTAINTY SCORE:")
    report.append("-" * 50)
    
    # Compute overall score from key metrics
    score_components = []
    
    # Coverage score (closer to expected is better)
    if 'coverage_0.95' in validation_results:
        coverage_score = 1.0 - abs(validation_results['coverage_0.95'] - 0.95)
        score_components.append(coverage_score)
    
    # Calibration score (lower ECE is better)
    if 'ece' in validation_results:
        calibration_score = max(0, 1.0 - validation_results['ece'])
        score_components.append(calibration_score)
    
    # Reliability score (higher accuracy is better)
    if 'accuracy' in validation_results:
        reliability_score = validation_results['accuracy']
        score_components.append(reliability_score)
    
    if score_components:
        overall_score = np.mean(score_components)
        report.append(f"  Overall Uncertainty Score: {overall_score:.4f}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        report.append("-" * 50)
        
        if overall_score < 0.5:
            report.append("  - Significant uncertainty calibration issues detected")
            report.append("  - Consider retraining with uncertainty-aware loss")
            report.append("  - Implement temperature scaling for better calibration")
        elif overall_score < 0.7:
            report.append("  - Good uncertainty calibration with room for improvement")
            report.append("  - Fine-tune uncertainty estimation methods")
        else:
            report.append("  - Excellent uncertainty calibration")
            report.append("  - Model ready for scientific inference")
    
    report.append("=" * 100)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text
