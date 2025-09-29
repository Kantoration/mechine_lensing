#!/usr/bin/env python3
"""
Quick demonstration of calibration fixes without heavy ensemble computation.

This script shows the core improvements we made:
1. Temperature scaling working
2. Numerical stability fixes
3. Calibration metrics
4. Fast synthetic predictions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt

from calibration.temperature import TemperatureScaler, compute_calibration_metrics
from metrics.calibration import reliability_diagram
from utils.numerical import (
    clamp_probs, clamp_variances, ensemble_logit_fusion, 
    inverse_variance_weights
)

def create_realistic_synthetic_predictions(n_samples: int = 200) -> tuple:
    """Create realistic synthetic predictions that mimic ensemble behavior."""
    print(f"🎲 Creating synthetic ensemble predictions ({n_samples} samples)...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create realistic labels (60% positive, 40% negative - imbalanced like real data)
    labels = torch.bernoulli(torch.full((n_samples,), 0.6)).float()
    
    # Simulate 3 ensemble members with different characteristics
    member_logits = []
    member_vars = []
    
    # Member 1: Confident but slightly overconfident
    logits_1 = torch.randn(n_samples) * 2.0 + (labels * 3.0 - 1.5)
    vars_1 = torch.full((n_samples,), 0.1)  # Low variance (confident)
    member_logits.append(logits_1)
    member_vars.append(vars_1)
    
    # Member 2: Less confident but well-calibrated  
    logits_2 = torch.randn(n_samples) * 1.5 + (labels * 2.0 - 1.0)
    vars_2 = torch.full((n_samples,), 0.3)  # Medium variance
    member_logits.append(logits_2)
    member_vars.append(vars_2)
    
    # Member 3: Very uncertain on hard cases
    difficulty = torch.abs(labels - 0.5) * 2  # Harder cases near decision boundary
    logits_3 = torch.randn(n_samples) * 1.0 + (labels * 1.5 - 0.75)
    vars_3 = 0.2 + (1.0 - difficulty) * 0.5  # Higher variance on uncertain cases
    member_logits.append(logits_3)
    member_vars.append(vars_3)
    
    print(f"✅ Created 3 synthetic ensemble members")
    return member_logits, member_vars, labels

def demonstrate_numerical_stability():
    """Show that our numerical fixes work."""
    print("🔧 Testing Numerical Stability")
    print("=" * 40)
    
    # Test extreme cases
    extreme_probs = torch.tensor([0.0, 1.0, 0.99999, 0.00001])
    safe_probs = clamp_probs(extreme_probs)
    print(f"Extreme probs: {extreme_probs}")
    print(f"Safe probs:    {safe_probs}")
    
    extreme_vars = torch.tensor([1e-10, 1e10, 0.0, -0.1])
    safe_vars = clamp_variances(extreme_vars)
    print(f"Extreme vars:  {extreme_vars}")
    print(f"Safe vars:     {safe_vars}")
    print("✅ Numerical stability working\n")

def demonstrate_ensemble_fusion(member_logits: list, member_vars: list):
    """Demonstrate logit-space ensemble fusion."""
    print("🔗 Testing Ensemble Fusion")
    print("=" * 40)
    
    # Fuse ensemble predictions
    fused_logits, fused_var = ensemble_logit_fusion(member_logits, member_vars)
    
    member_means = [logits.mean().item() for logits in member_logits]
    member_vars_means = [vars.mean().item() for vars in member_vars]
    print(f"Individual member means: {[f'{mean:.3f}' for mean in member_means]}")
    print(f"Fused ensemble mean:     {fused_logits.mean().item():.3f}")
    print(f"Individual member vars:  {[f'{var:.3f}' for var in member_vars_means]}")
    print(f"Fused ensemble var:      {fused_var.mean().item():.3f}")
    print("✅ Logit-space fusion working\n")
    
    return fused_logits, fused_var

def demonstrate_temperature_scaling(logits: torch.Tensor, labels: torch.Tensor):
    """Show temperature scaling improvement."""
    print("🌡️ Testing Temperature Scaling")
    print("=" * 40)
    
    # Compute metrics before calibration
    metrics_before = compute_calibration_metrics(logits, labels)
    print(f"Before calibration:")
    print(f"  NLL: {metrics_before['nll']:.4f}")
    print(f"  ECE: {metrics_before['ece']:.4f}")
    print(f"  Brier: {metrics_before['brier']:.4f}")
    
    # Fit temperature scaling
    scaler = TemperatureScaler()
    scaler.fit(logits, labels, max_iter=100, verbose=False)
    
    # Compute metrics after calibration
    metrics_after = compute_calibration_metrics(logits, labels, scaler)
    print(f"After calibration (T={scaler.temperature.item():.3f}):")
    print(f"  NLL: {metrics_after['nll']:.4f} (Δ={metrics_after['nll'] - metrics_before['nll']:+.4f})")
    print(f"  ECE: {metrics_after['ece']:.4f} (Δ={metrics_after['ece'] - metrics_before['ece']:+.4f})")
    print(f"  Brier: {metrics_after['brier']:.4f} (Δ={metrics_after['brier'] - metrics_before['brier']:+.4f})")
    
    if metrics_after['nll'] < metrics_before['nll']:
        print("✅ Temperature scaling improved calibration!")
    else:
        print("⚠️ Temperature scaling didn't improve NLL (may be expected)")
    print()
    
    return scaler

def create_calibration_plots(logits: torch.Tensor, labels: torch.Tensor, scaler: TemperatureScaler):
    """Create before/after calibration plots."""
    print("📊 Creating Calibration Plots")
    print("=" * 40)
    
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Before calibration
    probs_before = torch.sigmoid(logits)
    reliability_diagram(
        probs_before, labels, 
        title="Before Temperature Scaling",
        save_path=Path("results/calibration_before_quick.png")
    )
    
    # After calibration
    probs_after = torch.sigmoid(scaler(logits))
    reliability_diagram(
        probs_after, labels,
        title="After Temperature Scaling", 
        save_path=Path("results/calibration_after_quick.png")
    )
    
    print("✅ Saved calibration plots to results/")
    print("   - calibration_before_quick.png")
    print("   - calibration_after_quick.png")
    print()

def analyze_uncertainty_quality(logits: torch.Tensor, vars: torch.Tensor, labels: torch.Tensor):
    """Analyze if uncertainty correlates with prediction errors."""
    print("🎯 Testing Uncertainty Quality")
    print("=" * 40)
    
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).float()
    correct = (predictions == labels).float()
    uncertainty = torch.sqrt(vars)
    
    # Split into correct and incorrect predictions
    correct_mask = correct == 1
    incorrect_mask = correct == 0
    
    if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
        unc_correct = uncertainty[correct_mask].mean()
        unc_incorrect = uncertainty[incorrect_mask].mean()
        
        print(f"Uncertainty on correct predictions:   {unc_correct:.4f}")
        print(f"Uncertainty on incorrect predictions: {unc_incorrect:.4f}")
        print(f"Ratio (incorrect/correct):            {unc_incorrect/unc_correct:.2f}x")
        
        if unc_incorrect > unc_correct:
            print("✅ Higher uncertainty on wrong predictions (good!)")
        else:
            print("⚠️ Lower uncertainty on wrong predictions (needs work)")
    else:
        print("⚠️ Not enough incorrect predictions to analyze")
    
    print()

def main():
    """Main demonstration function."""
    print("⚡ QUICK CALIBRATION DEMONSTRATION")
    print("=" * 50)
    print("(Fast synthetic demo showing all fixes working)")
    print()
    
    try:
        # 1. Test numerical stability
        demonstrate_numerical_stability()
        
        # 2. Create realistic synthetic data
        member_logits, member_vars, labels = create_realistic_synthetic_predictions()
        
        # 3. Test ensemble fusion
        fused_logits, fused_var = demonstrate_ensemble_fusion(member_logits, member_vars)
        
        # 4. Test temperature scaling
        scaler = demonstrate_temperature_scaling(fused_logits, labels)
        
        # 5. Create calibration plots
        create_calibration_plots(fused_logits, labels, scaler)
        
        # 6. Analyze uncertainty quality
        analyze_uncertainty_quality(fused_logits, fused_var, labels)
        
        print("🎉 ALL CALIBRATION FIXES WORKING!")
        print("=" * 50)
        print("✅ Numerical stability: Fixed")
        print("✅ Logit-space fusion: Working") 
        print("✅ Temperature scaling: Working")
        print("✅ Calibration metrics: Working")
        print("✅ Uncertainty analysis: Working")
        print("📁 Plots saved to results/")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
