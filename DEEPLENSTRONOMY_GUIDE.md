# Complete Guide to deeplenstronomy Integration

## Overview

This guide provides practical, actionable guidance for using deeplenstronomy's SimAPI to generate synthetic astronomical datasets. It addresses the common issues and provides robust solutions.

## The Reality of deeplenstronomy's SimAPI

### What the Documentation Doesn't Tell You

1. **SimAPI Constructor Complexity**: The SimAPI requires specific parameters that aren't well documented:
   - `numpix`: Image size in pixels
   - `kwargs_single_band`: Dictionary of observational parameters
   - `kwargs_model`: Dictionary of physical model parameters

2. **No Direct YAML Support**: Despite examples suggesting otherwise, SimAPI doesn't directly accept YAML files.

3. **Parameter Format Requirements**: All parameters must be in specific nested dictionary formats that match lenstronomy's internal structure.

## Correct SimAPI Usage Pattern

### 1. Basic Instantiation

```python
import deeplenstronomy.image_generator as dls_ig

# Required parameters
numpix = 64  # Image size

kwargs_single_band = {
    'pixel_scale': 0.1,          # arcsec/pixel
    'exposure_time': 5400.0,     # seconds  
    'magnitude_zero_point': 25.0, # AB magnitude zero point
    'read_noise': 10.0,          # electrons
    'ccd_gain': 1.0,             # electrons/ADU
    'sky_brightness': 22.0,      # mag/arcsec^2
}

kwargs_model = {
    'lens_model_list': ['SIE'],           # Mass models
    'source_light_model_list': ['SERSIC'], # Source light models
    'lens_light_model_list': [],          # Lens light (optional)
    'point_source_model_list': [],        # Point sources (optional)
}

# Create SimAPI instance
sim_api = dls_ig.SimAPI(numpix, kwargs_single_band, kwargs_model)
```

### 2. Parameter Dictionary Structure

The `kwargs_model` dictionary must contain lists of model names that correspond to lenstronomy models:

**Lens Mass Models:**
- `'SIE'`: Singular Isothermal Ellipsoid
- `'NFW'`: Navarro-Frenk-White profile
- `'SHEAR'`: External shear
- `'CONVERGENCE'`: External convergence

**Source Light Models:**
- `'SERSIC'`: Sersic profile
- `'GAUSSIAN'`: Gaussian profile
- `'UNIFORM'`: Uniform disk

**Lens Light Models:**
- `'SERSIC'`: Sersic profile for lens galaxy
- `'HERNQUIST'`: Hernquist profile

### 3. Method Calling

SimAPI typically provides these methods (availability varies by version):

```python
# Try these in order:
methods_to_try = [
    'run',                # Most common
    'generate',          # Alternative
    'simulate',          # Some versions
    'generate_images',   # Explicit name
    'run_all',          # Batch processing
]

for method_name in methods_to_try:
    if hasattr(sim_api, method_name):
        method = getattr(sim_api, method_name)
        try:
            method()  # Try without arguments
            break
        except TypeError:
            try:
                method(output_dir)  # Try with output directory
                break
            except:
                continue
```

## YAML Configuration Structure

### Working Configuration Example

```yaml
General:
  n_train: 1800
  n_test: 200
  image_size: 64
  seed: 42
  balance: 0.5

# Observational parameters (maps to kwargs_single_band)
Observation:
  pixel_scale: 0.1
  exposure_time: 5400.0
  magnitude_zero_point: 25.0
  read_noise: 10.0
  sky_brightness: 22.0

# Physical model parameters (maps to kwargs_model)
Models:
  lens_mass_models: ["SIE"]
  source_light_models: ["SERSIC"]
  lens_light_models: []

# Parameter ranges for random generation
ParameterRanges:
  lens_mass:
    einstein_radius: [0.5, 2.0]  # arcsec
    ellipticity: [0.0, 0.3]
    center_x: [-0.1, 0.1]        # arcsec offset
    center_y: [-0.1, 0.1]
  
  source_light:
    magnitude: [20.0, 24.0]      # AB mag
    effective_radius: [0.1, 0.8] # arcsec
    sersic_index: [1.0, 4.0]
    ellipticity: [0.0, 0.7]
```

## Expected Directory Structure

### Input Structure
```
project/
├── configs/
│   ├── quick.yaml              # Basic config
│   ├── comprehensive.yaml      # Full config
│   └── deeplenstronomy.yaml    # DLS-specific config
├── src/
│   ├── make_dataset_robust.py  # Main script
│   └── utils/
└── data/                       # Output directory
```

### Output Structure (Created by Script)
```
data/
├── train/
│   ├── lens/
│   │   ├── lens_train_0000.png
│   │   ├── lens_train_0001.png
│   │   └── ...
│   └── nonlens/
│       ├── nonlens_train_0000.png
│       └── ...
├── test/
│   ├── lens/
│   └── nonlens/
├── train.csv                   # filepath,label
├── test.csv                    # filepath,label
└── metadata/                   # Optional: parameter logs
    ├── generation_log.txt
    └── parameters.json
```

## Error Handling and Debugging

### Common Error Messages and Solutions

#### 1. "SimAPI.__init__() missing required positional arguments"
**Problem**: Incorrect constructor parameters
**Solution**: Ensure you provide all three required parameters:
```python
sim_api = dls_ig.SimAPI(numpix, kwargs_single_band, kwargs_model)
```

#### 2. "No attribute 'run'" or similar method errors
**Problem**: Method name varies by version
**Solution**: Use the robust method detection pattern:
```python
candidate_methods = ['run', 'generate', 'simulate', 'generate_images']
for method_name in candidate_methods:
    if hasattr(sim_api, method_name) and callable(getattr(sim_api, method_name)):
        try:
            getattr(sim_api, method_name)()
            break
        except Exception as e:
            print(f"Method {method_name} failed: {e}")
            continue
```

#### 3. "Invalid parameter format" errors
**Problem**: Parameter dictionaries don't match expected structure
**Solution**: Use parameter validation:
```python
def validate_kwargs_model(kwargs_model):
    required_keys = ['lens_model_list', 'source_light_model_list']
    for key in required_keys:
        if key not in kwargs_model:
            raise ValueError(f"Missing required key: {key}")
        if not isinstance(kwargs_model[key], list):
            raise ValueError(f"{key} must be a list")
    return True
```

### Debugging Checklist

1. **Check imports**:
   ```python
   import deeplenstronomy as dls
   import deeplenstronomy.image_generator as dls_ig
   print("Available attributes:", dir(dls_ig))
   ```

2. **Validate parameters**:
   ```python
   print("numpix:", numpix, type(numpix))
   print("kwargs_single_band:", kwargs_single_band)
   print("kwargs_model:", kwargs_model)
   ```

3. **Test instantiation**:
   ```python
   try:
       sim_api = dls_ig.SimAPI(numpix, kwargs_single_band, kwargs_model)
       print("SimAPI created successfully")
       print("Available methods:", [m for m in dir(sim_api) if not m.startswith('_')])
   except Exception as e:
       print(f"SimAPI creation failed: {e}")
   ```

## Best Practices for Integration

### 1. Version-Agnostic Code

```python
class DeeplenstronomyWrapper:
    def __init__(self):
        self.version = self._detect_version()
        self.api_variant = self._detect_api_variant()
    
    def _detect_version(self):
        try:
            import deeplenstronomy
            return getattr(deeplenstronomy, '__version__', 'unknown')
        except:
            return None
    
    def _detect_api_variant(self):
        # Detect which API pattern is available
        try:
            import deeplenstronomy.image_generator as dls_ig
            if hasattr(dls_ig, 'SimAPI'):
                return 'simapi'
            elif hasattr(dls_ig, 'ImageGenerator'):
                return 'imagegen'
            else:
                return 'unknown'
        except:
            return None
```

### 2. Graceful Fallbacks

```python
def generate_dataset_with_fallback(config, output_dir):
    backends = ['deeplenstronomy', 'synthetic']
    
    for backend in backends:
        try:
            if backend == 'deeplenstronomy':
                return generate_with_deeplenstronomy(config, output_dir)
            else:
                return generate_synthetic(config, output_dir)
        except Exception as e:
            print(f"Backend {backend} failed: {e}")
            if backend == backends[-1]:  # Last backend
                raise
            print(f"Trying fallback backend...")
```

### 3. Parameter Validation

```python
def validate_deeplenstronomy_config(config):
    """Validate config before passing to deeplenstronomy."""
    errors = []
    
    # Check required sections
    required_sections = ['General', 'Observation', 'Models']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Check parameter ranges
    if 'ParameterRanges' in config:
        ranges = config['ParameterRanges']
        for param_type, params in ranges.items():
            for param_name, param_range in params.items():
                if not isinstance(param_range, list) or len(param_range) != 2:
                    errors.append(f"Invalid range for {param_type}.{param_name}")
    
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(errors))
    
    return True
```

## Production Usage Example

```python
#!/usr/bin/env python3
"""Production-ready deeplenstronomy dataset generator."""

import sys
from pathlib import Path
import yaml
import traceback

def main():
    config_file = Path("configs/comprehensive.yaml")
    output_dir = Path("data")
    
    try:
        # Load and validate config
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        validate_deeplenstronomy_config(config)
        
        # Try deeplenstronomy first
        try:
            generate_with_deeplenstronomy(config, output_dir)
            print("✅ deeplenstronomy generation successful")
        except Exception as e:
            print(f"⚠️  deeplenstronomy failed: {e}")
            print("🔄 Falling back to synthetic generation...")
            generate_synthetic(config, output_dir)
            print("✅ Synthetic generation successful")
        
        # Create CSV files
        create_csv_files(output_dir)
        
        # Validate output
        validate_dataset(output_dir)
        
        print("🎉 Dataset generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Testing Your Setup

Use this script to test your deeplenstronomy installation:

```python
#!/usr/bin/env python3
"""Test script for deeplenstronomy setup."""

def test_deeplenstronomy():
    """Comprehensive test of deeplenstronomy installation."""
    
    print("Testing deeplenstronomy installation...")
    
    # Test 1: Import
    try:
        import deeplenstronomy as dls
        import deeplenstronomy.image_generator as dls_ig
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Check available components
    print("Available components:")
    print(f"  deeplenstronomy: {dir(dls)}")
    print(f"  image_generator: {dir(dls_ig)}")
    
    # Test 3: Try SimAPI instantiation
    try:
        numpix = 64
        kwargs_single_band = {'pixel_scale': 0.1, 'exposure_time': 5400.0, 'magnitude_zero_point': 25.0}
        kwargs_model = {'lens_model_list': ['SIE'], 'source_light_model_list': ['SERSIC']}
        
        sim_api = dls_ig.SimAPI(numpix, kwargs_single_band, kwargs_model)
        print("✅ SimAPI instantiation successful")
        
        # Test 4: Check available methods
        methods = [m for m in dir(sim_api) if not m.startswith('_') and callable(getattr(sim_api, m))]
        print(f"Available methods: {methods}")
        
        return True
        
    except Exception as e:
        print(f"❌ SimAPI test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_deeplenstronomy()
    if success:
        print("\n🎉 deeplenstronomy appears to be working correctly!")
    else:
        print("\n⚠️  deeplenstronomy has issues. Consider using synthetic backend.")
```

This guide should give you everything you need to successfully integrate deeplenstronomy into your workflow, with robust error handling and fallback mechanisms.
