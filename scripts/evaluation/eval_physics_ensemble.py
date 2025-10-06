#!/usr/bin/env python3
"""
Evaluation script for Physics-Informed Ensemble
===============================================

This script evaluates physics-informed ensemble models with comprehensive
analysis including attention maps, physics consistency, and uncertainty estimation.

Key Features:
- Physics consistency validation
- Attention map visualization and analysis
- Uncertainty quantification
- Comparative analysis with traditional models

Usage:
    python scripts/eval_physics_ensemble.py --checkpoint checkpoints/best_physics_ensemble.pt
    python scripts/eval_physics_ensemble.py --checkpoint checkpoints/best_physics_ensemble.pt --visualize
"""

# Standard library imports
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from src.datasets.lens_dataset import LensDataset
from src.models.ensemble.physics_informed_ensemble import PhysicsInformedEnsemble
from src.metrics.classification import compute_classification_metrics
from src.validation.physics_validator import PhysicsValidator, validate_attention_physics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class PhysicsEnsembleEvaluator:
    """
    Comprehensive evaluator for physics-informed ensemble models.
    
    Provides detailed analysis including:
    - Standard classification metrics
    - Physics consistency validation
    - Attention map quality assessment
    - Uncertainty analysis
    - Comparative performance analysis
    """
    
    def __init__(
        self,
        model: PhysicsInformedEnsemble,
        device: torch.device,
        save_dir: Optional[Path] = None
    ):
        """Initialize physics ensemble evaluator."""
        self.model = model
        self.device = device
        self.save_dir = save_dir or Path('results')
        self.save_dir.mkdir(exist_ok=True)
        
        # Physics validator
        self.physics_validator = PhysicsValidator(device)
        
        # Results storage
        self.results = {}
        
        logger.info("Initialized physics ensemble evaluator")
    
    def evaluate(
        self,
        data_loader: DataLoader,
        visualize: bool = False,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of physics-informed ensemble.
        
        Args:
            data_loader: Data loader for evaluation
            visualize: Whether to create visualizations
            save_predictions: Whether to save detailed predictions
            
        Returns:
            Dictionary containing all evaluation results
        """
        self.model.eval()
        
        logger.info("Starting comprehensive evaluation...")
        
        # Collect predictions and analyses
        all_predictions = []
        all_labels = []
        all_member_predictions = []
        all_uncertainties = []
        all_physics_analyses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Prepare inputs
                inputs = self._prepare_inputs(batch)
                labels = batch['label'].float().to(self.device)
                
                # Forward pass with detailed analysis
                outputs = self.model(inputs)
                physics_analysis = self.model.get_physics_analysis(inputs)
                
                # Store results
                all_predictions.append(outputs['prediction'].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_member_predictions.append(outputs['member_predictions'].cpu().numpy())
                all_uncertainties.append(outputs['member_uncertainties'].cpu().numpy())
                all_physics_analyses.append(physics_analysis)
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(data_loader)}")
        
        # Concatenate results
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        member_predictions = np.concatenate(all_member_predictions, axis=0)
        uncertainties = np.concatenate(all_uncertainties, axis=0)
        
        # Standard classification metrics
        logger.info("Computing classification metrics...")
        classification_metrics = compute_classification_metrics(
            y_true=labels,
            y_pred=predictions,
            y_pred_binary=(predictions > 0.5).astype(int)
        )
        
        # Physics consistency analysis
        logger.info("Analyzing physics consistency...")
        physics_metrics = self._analyze_physics_consistency(all_physics_analyses)
        
        # Uncertainty analysis
        logger.info("Analyzing uncertainty estimates...")
        uncertainty_metrics = self._analyze_uncertainties(
            predictions, labels, uncertainties
        )
        
        # Member analysis
        logger.info("Analyzing ensemble members...")
        member_metrics = self._analyze_ensemble_members(
            member_predictions, labels, self.model.member_names
        )
        
        # Attention analysis (if available)
        attention_metrics = {}
        if all_physics_analyses and 'attention_maps' in all_physics_analyses[0]:
            logger.info("Analyzing attention maps...")
            attention_metrics = self._analyze_attention_maps(all_physics_analyses, labels)
        
        # Compile all results
        results = {
            'classification_metrics': classification_metrics,
            'physics_metrics': physics_metrics,
            'uncertainty_metrics': uncertainty_metrics,
            'member_metrics': member_metrics,
            'attention_metrics': attention_metrics,
            'summary': self._create_summary(
                classification_metrics, physics_metrics, uncertainty_metrics
            )
        }
        
        # Save detailed predictions if requested
        if save_predictions:
            self._save_detailed_predictions(
                predictions, labels, member_predictions, uncertainties
            )
        
        # Create visualizations if requested
        if visualize:
            logger.info("Creating visualizations...")
            self._create_visualizations(results, all_physics_analyses, labels)
        
        # Save results
        self._save_results(results)
        
        self.results = results
        return results
    
    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for different model architectures."""
        image = batch['image'].to(self.device)
        
        inputs = {}
        for name in self.model.member_names:
            target_size = self.model.member_input_sizes[name]
            
            if image.size(-1) != target_size:
                resized_image = torch.nn.functional.interpolate(
                    image, size=(target_size, target_size), 
                    mode='bilinear', align_corners=False
                )
                inputs[name] = resized_image
            else:
                inputs[name] = image
        
        return inputs
    
    def _analyze_physics_consistency(self, physics_analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze physics consistency across all samples."""
        prediction_variances = []
        physics_correlations = []
        physics_losses = []
        
        for analysis in physics_analyses:
            physics_consistency = analysis['physics_consistency']
            prediction_variances.append(physics_consistency['prediction_variance'])
            physics_correlations.append(physics_consistency['physics_traditional_correlation'])
            physics_losses.append(physics_consistency['physics_loss'])
        
        return {
            'mean_prediction_variance': np.mean(prediction_variances),
            'std_prediction_variance': np.std(prediction_variances),
            'mean_physics_correlation': np.mean(physics_correlations),
            'mean_physics_loss': np.mean(physics_losses),
            'physics_consistency_score': 1.0 / (1.0 + np.mean(prediction_variances))
        }
    
    def _analyze_uncertainties(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """Analyze uncertainty estimates and calibration."""
        # Average uncertainty across ensemble members
        mean_uncertainty = np.mean(uncertainties, axis=1)
        
        # Uncertainty vs prediction confidence
        prediction_confidence = np.abs(predictions - 0.5) * 2  # [0, 1]
        uncertainty_confidence_corr = np.corrcoef(mean_uncertainty, prediction_confidence)[0, 1]
        
        # Uncertainty for correct vs incorrect predictions
        correct_mask = (predictions > 0.5) == (labels > 0.5)
        uncertainty_correct = mean_uncertainty[correct_mask]
        uncertainty_incorrect = mean_uncertainty[~correct_mask]
        
        return {
            'mean_uncertainty': np.mean(mean_uncertainty),
            'uncertainty_confidence_correlation': uncertainty_confidence_corr,
            'uncertainty_correct_mean': np.mean(uncertainty_correct) if len(uncertainty_correct) > 0 else 0.0,
            'uncertainty_incorrect_mean': np.mean(uncertainty_incorrect) if len(uncertainty_incorrect) > 0 else 0.0,
            'uncertainty_discrimination': np.mean(uncertainty_incorrect) - np.mean(uncertainty_correct) if len(uncertainty_incorrect) > 0 and len(uncertainty_correct) > 0 else 0.0
        }
    
    def _analyze_ensemble_members(
        self,
        member_predictions: np.ndarray,
        labels: np.ndarray,
        member_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze individual ensemble member performance."""
        member_metrics = {}
        
        for i, name in enumerate(member_names):
            preds = member_predictions[:, i]
            metrics = compute_classification_metrics(
                y_true=labels,
                y_pred=preds,
                y_pred_binary=(preds > 0.5).astype(int)
            )
            member_metrics[name] = metrics
        
        return member_metrics
    
    def _analyze_attention_maps(
        self,
        physics_analyses: List[Dict[str, Any]],
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Analyze attention map quality and physics consistency."""
        # This is a simplified analysis - would need ground truth attention maps
        # for comprehensive evaluation
        
        attention_qualities = []
        physics_consistencies = []
        
        for i, analysis in enumerate(physics_analyses):
            if 'attention_maps' in analysis:
                # Simple quality metrics based on attention distribution
                for model_name, maps in analysis['attention_maps'].items():
                    if 'enhanced_light_transformer' in model_name:
                        for map_name, attention_map in maps.items():
                            if attention_map.size > 0:
                                # Entropy as a measure of attention diversity
                                entropy = -np.sum(attention_map * np.log(attention_map + 1e-8))
                                attention_qualities.append(entropy)
                                
                                # Physics consistency (placeholder)
                                physics_consistencies.append(1.0)
        
        return {
            'mean_attention_entropy': np.mean(attention_qualities) if attention_qualities else 0.0,
            'attention_physics_consistency': np.mean(physics_consistencies) if physics_consistencies else 0.0
        }
    
    def _create_summary(
        self,
        classification_metrics: Dict[str, float],
        physics_metrics: Dict[str, float],
        uncertainty_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Create summary of key metrics."""
        return {
            'accuracy': classification_metrics['accuracy'],
            'f1_score': classification_metrics['f1_score'],
            'roc_auc': classification_metrics['roc_auc'],
            'physics_consistency_score': physics_metrics['physics_consistency_score'],
            'uncertainty_discrimination': uncertainty_metrics['uncertainty_discrimination'],
            'mean_physics_correlation': physics_metrics['mean_physics_correlation']
        }
    
    def _save_detailed_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        member_predictions: np.ndarray,
        uncertainties: np.ndarray
    ) -> None:
        """Save detailed predictions and analysis."""
        detailed_results = {
            'ensemble_predictions': predictions,
            'labels': labels,
            'member_predictions': member_predictions,
            'member_uncertainties': uncertainties,
            'member_names': self.model.member_names
        }
        
        np.savez(
            self.save_dir / 'detailed_predictions.npz',
            **detailed_results
        )
        
        logger.info(f"Saved detailed predictions to {self.save_dir / 'detailed_predictions.npz'}")
    
    def _create_visualizations(
        self,
        results: Dict[str, Any],
        physics_analyses: List[Dict[str, Any]],
        labels: np.ndarray
    ) -> None:
        """Create comprehensive visualizations."""
        # Member performance comparison
        self._plot_member_performance(results['member_metrics'])
        
        # Physics consistency analysis
        self._plot_physics_consistency(results['physics_metrics'])
        
        # Uncertainty analysis
        self._plot_uncertainty_analysis(results['uncertainty_metrics'], physics_analyses, labels)
    
    def _plot_member_performance(self, member_metrics: Dict[str, Dict[str, float]]) -> None:
        """Plot ensemble member performance comparison."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(member_metrics))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [member_metrics[name][metric] for name in member_metrics.keys()]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Ensemble Members')
        ax.set_ylabel('Metric Value')
        ax.set_title('Ensemble Member Performance Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(list(member_metrics.keys()), rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'member_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_physics_consistency(self, physics_metrics: Dict[str, float]) -> None:
        """Plot physics consistency metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Physics consistency score
        ax1.bar(['Physics Consistency Score'], [physics_metrics['physics_consistency_score']])
        ax1.set_ylabel('Score')
        ax1.set_title('Physics Consistency Score')
        ax1.set_ylim(0, 1)
        
        # Physics-traditional correlation
        ax2.bar(['Physics-Traditional Correlation'], [physics_metrics['mean_physics_correlation']])
        ax2.set_ylabel('Correlation')
        ax2.set_title('Physics-Traditional Model Correlation')
        ax2.set_ylim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'physics_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_uncertainty_analysis(
        self,
        uncertainty_metrics: Dict[str, float],
        physics_analyses: List[Dict[str, Any]],
        labels: np.ndarray
    ) -> None:
        """Plot uncertainty analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Uncertainty discrimination
        correct_unc = uncertainty_metrics['uncertainty_correct_mean']
        incorrect_unc = uncertainty_metrics['uncertainty_incorrect_mean']
        
        ax1.bar(['Correct Predictions', 'Incorrect Predictions'], [correct_unc, incorrect_unc])
        ax1.set_ylabel('Mean Uncertainty')
        ax1.set_title('Uncertainty for Correct vs Incorrect Predictions')
        
        # Uncertainty distribution by class
        lens_uncertainties = []
        nonlens_uncertainties = []
        
        for i, analysis in enumerate(physics_analyses):
            if i < len(labels):
                uncertainty = np.mean(analysis['member_uncertainties'])
                if labels[i] > 0.5:
                    lens_uncertainties.append(uncertainty)
                else:
                    nonlens_uncertainties.append(uncertainty)
        
        ax2.boxplot([nonlens_uncertainties, lens_uncertainties], labels=['Non-Lens', 'Lens'])
        ax2.set_ylabel('Uncertainty')
        ax2.set_title('Uncertainty Distribution by Class')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results."""
        # Save as torch file for detailed analysis
        torch.save(results, self.save_dir / 'physics_ensemble_evaluation.pt')
        
        # Save summary as JSON for easy reading
        import json
        with open(self.save_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(results['summary'], f, indent=2)
        
        logger.info(f"Saved evaluation results to {self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Physics-Informed Ensemble')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-root', type=str, 
                        default='data/processed/data_realistic_test',
                        help='Root directory for test data')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # Create model
    ensemble = PhysicsInformedEnsemble(
        member_configs=config['members'],
        physics_weight=config['ensemble'].get('physics_weight', 0.1),
        uncertainty_estimation=config['ensemble'].get('uncertainty_estimation', True),
        attention_analysis=config['ensemble'].get('attention_analysis', True)
    )
    ensemble.load_state_dict(checkpoint['model_state_dict'])
    ensemble.to(device)
    
    # Create test dataset
    test_dataset = LensDataset(
        data_root=args.data_root,
        split="test",
        transform_config={"resize": 112}
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Test set size: {len(test_dataset)}")
    
    # Create evaluator
    evaluator = PhysicsEnsembleEvaluator(
        model=ensemble,
        device=device,
        save_dir=Path(args.save_dir)
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        data_loader=test_loader,
        visualize=args.visualize,
        save_predictions=True
    )
    
    # Print summary
    logger.info("Evaluation completed!")
    logger.info("Summary Results:")
    for metric, value in results['summary'].items():
        logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()


