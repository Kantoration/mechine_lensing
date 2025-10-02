#!/usr/bin/env python3
"""
Demo script showing the complete calibrated ensemble workflow.

This script demonstrates:
1. Loading multiple trained models
2. Creating an uncertainty-weighted ensemble
3. Fitting temperature scaling for calibration
4. Evaluating with comprehensive metrics
5. Generating calibration plots

Usage:
    python scripts/demo_calibrated_ensemble.py --config configs/baseline.yaml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

from models.ensemble.weighted import UncertaintyWeightedEnsemble
from models.ensemble.registry import make_model
from calibration.temperature import TemperatureScaler, compute_calibration_metrics
from metrics.calibration import reliability_diagram
from utils.config import load_config
from datasets.lens_dataset import LensDataset

def create_mock_ensemble() -> UncertaintyWeightedEnsemble:
    """Create a mock ensemble for demonstration."""
    print("🏗️ Creating mock ensemble...")
    
    # Create mock models (normally you'd load trained checkpoints)
    members = []
    member_names = []
    
    # Mock ResNet
    backbone_resnet, head_resnet, _ = make_model("resnet18")
    members.append((backbone_resnet, head_resnet))
    member_names.append("resnet18")
    
    # Mock ViT
    backbone_vit, head_vit, _ = make_model("vit_b16")
    members.append((backbone_vit, head_vit))
    member_names.append("vit_b16")
    
    # Create ensemble
    ensemble = UncertaintyWeightedEnsemble(
        members=members,
        member_names=member_names
    )
    
    print(f"✅ Created ensemble with {len(members)} members")
    return ensemble

def create_mock_data(batch_size: int = 100) -> tuple:
    """Create mock validation data for demonstration."""
    print(f"📊 Creating mock validation data ({batch_size} samples)...")
    
    # Create synthetic data that's somewhat realistic
    torch.manual_seed(42)
    
    # Mock inputs (different sizes for different models)
    inputs = {
        "resnet18": torch.randn(batch_size, 3, 64, 64),
        "vit_b16": torch.randn(batch_size, 3, 224, 224)
    }
    
    # Mock labels (balanced)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    print(f"✅ Created mock data: {batch_size} samples, {labels.mean():.1%} positive")
    return inputs, labels

def demonstrate_ensemble_prediction(ensemble: UncertaintyWeightedEnsemble, inputs: Dict[str, torch.Tensor]) -> tuple:
    """Demonstrate ensemble prediction with uncertainty."""
    print("🔮 Running ensemble prediction...")
    
    ensemble.eval()
    with torch.no_grad():
        # Get ensemble predictions with uncertainty
        pred_mean, pred_var, member_contributions = ensemble.predict_with_uncertainty(
            inputs, mc_samples=5
        )
        
        # Convert to probabilities
        probs = torch.sigmoid(pred_mean)
        
        print(f"✅ Predictions: mean={pred_mean.mean():.3f}, var={pred_var.mean():.3f}")
        print(f"   Probabilities: min={probs.min():.3f}, max={probs.max():.3f}")
        print(f"   Member contributions: {len(member_contributions)} members")
        
    return pred_mean, pred_var

def demonstrate_temperature_scaling(logits: torch.Tensor, labels: torch.Tensor) -> TemperatureScaler:
    """Demonstrate temperature scaling for calibration."""
    print("🌡️ Fitting temperature scaling...")
    
    # Create and fit temperature scaler
    scaler = TemperatureScaler()
    scaler.fit(logits, labels, max_iter=100, verbose=True)
    
    # Compute before/after metrics
    metrics_before = compute_calibration_metrics(logits, labels)
    metrics_after = compute_calibration_metrics(logits, labels, scaler)
    
    print("\n📊 Calibration Improvement:")
    print(f"   NLL: {metrics_before['nll']:.4f} → {metrics_after['nll']:.4f}")
    print(f"   ECE: {metrics_before['ece']:.4f} → {metrics_after['ece']:.4f}")
    print(f"   Brier: {metrics_before['brier']:.4f} → {metrics_after['brier']:.4f}")
    
    return scaler

def create_calibration_plots(logits: torch.Tensor, labels: torch.Tensor, scaler: TemperatureScaler):
    """Create and save calibration plots."""
    print("📈 Creating calibration plots...")
    
    # Before calibration
    probs_before = torch.sigmoid(logits)
    reliability_diagram(probs_before, labels, title="Before Temperature Scaling", save_path=Path("results/reliability_before.png"))
    
    # After calibration  
    probs_after = torch.sigmoid(scaler(logits))
    reliability_diagram(probs_after, labels, title="After Temperature Scaling", save_path=Path("results/reliability_after.png"))
    
    print("✅ Saved calibration plots to results/reliability_before.png and results/reliability_after.png")

def demonstrate_uncertainty_analysis(pred_mean: torch.Tensor, pred_var: torch.Tensor, labels: torch.Tensor):
    """Demonstrate uncertainty analysis."""
    print("🎯 Analyzing prediction uncertainty...")
    
    # Convert to probabilities
    probs = torch.sigmoid(pred_mean)
    uncertainty = torch.sqrt(pred_var)
    
    # Compute prediction correctness
    predictions = (probs > 0.5).float()
    correct = (predictions == labels).float()
    
    # Analyze uncertainty vs correctness
    correct_mask = correct == 1
    incorrect_mask = correct == 0
    
    if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
        uncertainty_correct = uncertainty[correct_mask].mean()
        uncertainty_incorrect = uncertainty[incorrect_mask].mean()
        
        print(f"📊 Uncertainty Analysis:")
        print(f"   Correct predictions: {uncertainty_correct:.4f} ± uncertainty")
        print(f"   Incorrect predictions: {uncertainty_incorrect:.4f} ± uncertainty")
        print(f"   Ratio (incorrect/correct): {uncertainty_incorrect/uncertainty_correct:.2f}x")
        
        if uncertainty_incorrect > uncertainty_correct:
            print("✅ Higher uncertainty on incorrect predictions (good!)")
        else:
            print("⚠️ Lower uncertainty on incorrect predictions (needs improvement)")

def main():
    """Main demonstration function."""
    print("🚀 CALIBRATED ENSEMBLE DEMONSTRATION")
    print("=" * 50)
    
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    try:
        # 1. Create mock ensemble
        ensemble = create_mock_ensemble()
        
        # 2. Create mock data
        inputs, labels = create_mock_data(batch_size=200)
        
        # 3. Get ensemble predictions
        pred_mean, pred_var = demonstrate_ensemble_prediction(ensemble, inputs)
        
        # 4. Fit temperature scaling
        scaler = demonstrate_temperature_scaling(pred_mean, labels)
        
        # 5. Create calibration plots
        create_calibration_plots(pred_mean, labels, scaler)
        
        # 6. Analyze uncertainty
        demonstrate_uncertainty_analysis(pred_mean, pred_var, labels)
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("✅ Ensemble fusion working")
        print("✅ Temperature scaling working") 
        print("✅ Calibration metrics working")
        print("✅ Uncertainty analysis working")
        print("📁 Results saved to results/")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
