#!/usr/bin/env python3
"""
Quick test script to verify all ensemble components work with existing dataset.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
from datasets.lens_dataset import LensDataset
from torch.utils.data import DataLoader

print("🔭 Testing Ensemble Components with Existing Dataset")
print("=" * 60)

# Test 1: Load existing dataset
print("\n1. Testing dataset loading...")
try:
    dataset = LensDataset(
        data_root="data/processed/data_realistic_test",
        split="test", 
        img_size=224,
        augment=False,
        validate_paths=True
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Test a sample
    sample_img, sample_label = dataset[0]
    print(f"   Sample shape: {sample_img.shape}, Label: {sample_label}")
    
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    sys.exit(1)

# Test 2: Individual models
print("\n2. Testing individual models...")
from models.ensemble.registry import make_model

for arch in ['resnet18', 'vit_b16']:
    try:
        backbone, head, feat_dim = make_model(arch, bands=3, pretrained=False)
        
        # Test appropriate input size
        img_size = 224 if arch == 'vit_b16' else 112
        x = torch.randn(2, 3, img_size, img_size)
        
        # Forward pass
        features = backbone(x)
        logits = head(features)
        
        print(f"✅ {arch}: input {x.shape} -> features {features.shape} -> logits {logits.shape}")
        
    except Exception as e:
        print(f"❌ {arch} failed: {e}")

# Test 3: Ensemble
print("\n3. Testing ensemble...")
try:
    from models.ensemble.weighted import create_uncertainty_weighted_ensemble
    
    ensemble = create_uncertainty_weighted_ensemble(
        architectures=['resnet18', 'vit_b16'],
        bands=3,
        pretrained=False
    )
    
    # Create inputs
    inputs = {
        'resnet18': torch.randn(2, 3, 112, 112),
        'vit_b16': torch.randn(2, 3, 224, 224)
    }
    
    # Test ensemble prediction
    ensemble.eval()
    with torch.no_grad():
        predictions = ensemble(inputs)
        mc_pred, mc_var, weights = ensemble.mc_predict(inputs, mc_samples=5)
    
    print(f"✅ Ensemble: predictions {predictions.shape}, uncertainty {mc_var.shape}")
    print(f"   Member weights: {weights}")
    
except Exception as e:
    print(f"❌ Ensemble failed: {e}")

# Test 4: Data loader integration
print("\n4. Testing with DataLoader...")
try:
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Get one batch
    images, labels = next(iter(dataloader))
    print(f"   Batch shape: {images.shape}, Labels: {labels.shape}")
    
    # Test with ensemble
    inputs = {
        'resnet18': torch.nn.functional.interpolate(images, size=112, mode='bilinear', align_corners=False),
        'vit_b16': torch.nn.functional.interpolate(images, size=224, mode='bilinear', align_corners=False)
    }
    
    with torch.no_grad():
        batch_predictions = ensemble(inputs)
    
    print(f"✅ Batch processing: {batch_predictions.shape}")
    
except Exception as e:
    print(f"❌ DataLoader integration failed: {e}")

print("\n" + "=" * 60)
print("🎉 Component testing complete!")
print("\nNext steps:")
print("• Run existing training script: python scripts/train.py --data-root data/processed/data_realistic_test")  
print("• Or use the example: python scripts/example_ensemble_usage.py")
print("• For evaluation: python scripts/eval.py --data-root data/processed/data_realistic_test")
