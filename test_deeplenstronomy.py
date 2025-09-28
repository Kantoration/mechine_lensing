#!/usr/bin/env python3
"""
Test script for DeepLenstronomy integration.
Generates a small sample of gravitational lens images.
"""

import deeplenstronomy
import os
from pathlib import Path

def test_deeplenstronomy():
    """Test DeepLenstronomy dataset generation."""
    print("🔭 Testing DeepLenstronomy...")
    
    # Create output directory
    output_dir = Path("data_deeplenstronomy_test")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Generate dataset using DeepLenstronomy
        print("📁 Generating dataset...")
        deeplenstronomy.make_dataset(
            config_path="configs/deeplenstronomy_config.yaml",
            output_path=str(output_dir)
        )
        
        print(f"✅ DeepLenstronomy test successful!")
        print(f"📂 Output saved to: {output_dir}")
        
        # Check what was generated
        if output_dir.exists():
            files = list(output_dir.rglob("*"))
            print(f"📊 Generated {len(files)} files")
            for f in files[:5]:  # Show first 5 files
                print(f"  - {f}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
        
        return True
        
    except Exception as e:
        print(f"❌ DeepLenstronomy test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_deeplenstronomy()
    if success:
        print("\n🎉 DeepLenstronomy is working correctly!")
        print("Next steps:")
        print("1. Generate larger dataset")
        print("2. Create non-lens comparison dataset")
        print("3. Train model on DeepLenstronomy data")
    else:
        print("\n💥 DeepLenstronomy test failed. Check configuration.")
