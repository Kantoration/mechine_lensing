# 🚨 Priority 0 Fixes Implementation Guide

## Quick Reference for Critical Scientific Corrections

This document provides a quick reference for the Priority 0 fixes implemented in response to comprehensive scientific review.

---

## ✅ **What Has Been Implemented**

### **1. Dataset Labeling & Provenance** 🏷️

**Problem**: GalaxiesML incorrectly implied to have lens labels; CASTLES is positive-only

**Solution Implemented**:
```python
# Label provenance tracking in metadata
metadata = ImageMetadataV2(
    label_source='pretrain:galaxiesml',  # or 'obs:castles', 'sim:bologna', 'weak:gzoo'
    label_confidence=0.0,  # 0.0 for pretrain, 1.0 for confirmed
    ...
)
```

**Usage**:
```bash
# GalaxiesML - PRETRAINING ONLY
python scripts/convert_real_datasets.py \
    --dataset galaxiesml \
    --input data/raw/GalaxiesML/train.h5 \
    --output data/processed/real \
    --split train

# CASTLES - POSITIVE ONLY (warns about hard negatives)
python scripts/convert_real_datasets.py \
    --dataset castles \
    --input data/raw/CASTLES/ \
    --output data/processed/real \
    --split train
```

**Documentation Updated**:
- ✅ README.md: Prominent warnings added
- ✅ Integration Plan: Dataset usage clarification section
- ✅ Scripts: Built-in warnings and validation

---

### **2. Image Format & Dynamic Range** 📸

**Problem**: PNG format clips to 8-bit, losing faint arc signals

**Solution Implemented**:
```python
# Save as 16-bit TIFF with LZW compression
img_pil = Image.fromarray((img * 65535).astype(np.uint16), mode='I;16')
img_pil.save(filepath, format='TIFF', compression='lzw')

# Preserve variance maps
if variance_map is not None:
    var_pil = Image.fromarray((variance_map * 65535).astype(np.uint16), mode='I;16')
    var_pil.save(variance_path, format='TIFF', compression='lzw')
```

**Benefits**:
- ✅ Full dynamic range preservation (16-bit vs 8-bit)
- ✅ Critical for low flux-ratio systems (<0.1)
- ✅ Variance maps for weighted loss
- ✅ LZW compression (lossless, ~2-3x smaller than uncompressed)

**File Sizes**:
- 8-bit PNG: ~50-100KB
- 16-bit TIFF (LZW): ~150-300KB
- Variance map: +150-300KB (optional)

---

### **3. PSF Handling** 🔭

**Problem**: Naive Gaussian blur insufficient for PSF-sensitive arcs

**Solution Implemented**:
```python
class PSFMatcher:
    @staticmethod
    def match_psf_fourier(
        img: np.ndarray, 
        source_fwhm: float, 
        target_fwhm: float,
        pixel_scale: float = 0.2
    ) -> Tuple[np.ndarray, float]:
        """Fourier-domain PSF matching."""
        # Compute kernel FWHM
        kernel_fwhm = np.sqrt(target_fwhm**2 - source_fwhm**2)
        kernel_sigma_pixels = (kernel_fwhm / 2.355) / pixel_scale
        
        # Fourier-domain convolution
        img_fft = fft.fft2(img)
        kernel_fft = np.exp(-2*np.pi**2 * kernel_sigma_pixels**2 * r2 / (nx*ny))
        img_convolved = np.real(fft.ifft2(img_fft * kernel_fft))
        
        return img_convolved, psf_residual
```

**Usage**:
```bash
# Specify target PSF for homogenization
python scripts/convert_real_datasets.py \
    --dataset castles \
    --target-psf 1.0 \
    ...
```

**Metadata Tracking**:
```python
metadata = ImageMetadataV2(
    psf_fwhm=0.6,  # Source PSF FWHM (arcsec)
    target_psf_fwhm=1.0,  # Target after matching
    psf_matched=True,
    ...
)
```

---

### **4. Metadata Schema V2.0** 📋

**Problem**: Inconsistent metadata across surveys; missing stratification keys

**Solution Implemented**:
```python
@dataclass
class ImageMetadataV2:
    # Label Provenance (CRITICAL)
    label_source: str  # 'sim:bologna' | 'obs:castles' | 'weak:gzoo' | 'pretrain:galaxiesml'
    label_confidence: float  # 0.0-1.0
    
    # Observational Parameters (for stratification + FiLM)
    seeing: float  # arcsec
    psf_fwhm: float  # arcsec
    pixel_scale: float  # arcsec/pixel
    survey: str  # 'hsc' | 'sdss' | 'hst' | 'des' | 'kids' | 'relics'
    
    # Quality flags
    variance_map_available: bool
    psf_matched: bool
    target_psf_fwhm: float
    
    # Schema versioning
    schema_version: str = "2.0"
```

**Stratification Usage**:
- ✅ Redshift bins (5 bins)
- ✅ Magnitude bins (5 bins)
- ✅ **Seeing bins (3 bins)** [NEW]
- ✅ **PSF FWHM bins (3 bins)** [NEW]
- ✅ **Pixel scale bins (3 bins)** [NEW]
- ✅ **Survey/instrument** [NEW]
- ✅ Label

---

## 🎯 **Quick Start: Convert Your First Dataset**

### **Step 1: Install Dependencies**

```bash
cd demo/lens-demo
pip install -r requirements.txt
```

New dependencies added:
- `astropy>=5.0.0` (FITS handling)
- `h5py>=3.7.0` (HDF5 handling)
- `photutils>=1.8.0` (PSF estimation)

### **Step 2: Download Sample Data**

```bash
# Example: GalaxiesML sample
wget https://zenodo.org/records/13878122/files/GalaxiesML_sample_1000.h5 \
    -O data/raw/GalaxiesML/sample.h5
```

### **Step 3: Convert with Priority 0 Fixes**

```bash
# GalaxiesML (pretraining only)
python scripts/convert_real_datasets.py \
    --dataset galaxiesml \
    --input data/raw/GalaxiesML/sample.h5 \
    --output data/processed/real \
    --split train \
    --image-size 224

# Check output
ls -lh data/processed/real/train/galaxiesml_pretrain/
```

**Expected Output**:
```
galaxiesml_train_000000.tif  (16-bit TIFF, ~200KB)
galaxiesml_train_000001.tif
...
train_galaxiesml_pretrain.csv  (metadata with schema v2.0)
```

### **Step 4: Verify Conversion**

```python
import pandas as pd
from PIL import Image
import numpy as np

# Load metadata
df = pd.read_csv('data/processed/real/train_galaxiesml_pretrain.csv')
print(df.head())
print(f"\nSchema version: {df['schema_version'].iloc[0]}")
print(f"Label source: {df['label_source'].iloc[0]}")

# Check image format
img = Image.open(df['filepath'].iloc[0])
print(f"Image mode: {img.mode}")  # Should be 'I;16'
print(f"Image size: {img.size}")
print(f"Bit depth: 16-bit" if img.mode == 'I;16' else "ERROR: Not 16-bit")

# Check dynamic range
img_array = np.array(img)
print(f"Value range: [{img_array.min()}, {img_array.max()}]")
print(f"Dynamic range preserved: {img_array.max() > 255}")
```

---

## 🔍 **Verification Checklist**

After running the converter, verify these critical fixes:

- [ ] **16-bit TIFF files** created (not PNG)
  - Check: `file *.tif` should show "TIFF image data, 16-bit"
  
- [ ] **Variance maps** saved when available
  - Check: `*_var.tif` files exist for CASTLES
  
- [ ] **Label provenance** tracked in metadata
  - Check: CSV has `label_source` column
  - Check: GalaxiesML shows `pretrain:galaxiesml`
  - Check: CASTLES shows `obs:castles`
  
- [ ] **Extended metadata** present
  - Check: CSV has `seeing`, `psf_fwhm`, `pixel_scale` columns
  - Check: `schema_version` is "2.0"
  
- [ ] **PSF matching** applied and logged
  - Check: `psf_matched` is True
  - Check: `target_psf_fwhm` matches specified value
  
- [ ] **Warnings displayed** for critical issues
  - Check: GalaxiesML warns "NO LENS LABELS"
  - Check: CASTLES warns "POSITIVE-ONLY"

---

## 📚 **Additional Resources**

### **Related Documentation**:
- [Integration Implementation Plan](INTEGRATION_IMPLEMENTATION_PLAN.md) - Full technical specs
- [README](../README.md) - Dataset usage warnings
- [Lightning Integration Guide](LIGHTNING_INTEGRATION_GUIDE.md) - Cloud training setup

### **Next Steps**:
1. ✅ **Done**: Priority 0 fixes implemented
2. **Priority 1**: Implement soft-gated physics loss
3. **Priority 1**: Extend stratified validation
4. **Priority 2**: Implement Bologna metrics
5. **Priority 3**: Add SMACS J0723 validation

### **Scientific References**:
- Bologna Challenge: Transformer superiority on TPR metrics
- GalaxiesML paper: [arXiv:2410.00271](https://arxiv.org/abs/2410.00271)
- CASTLES Database: [CfA Harvard](https://lweb.cfa.harvard.edu/castles/)
- PSF-sensitive arcs: Known issue in strong lensing detection

---

## 🆘 **Troubleshooting**

### **Issue**: "photutils not found"
```bash
pip install photutils>=1.8.0
```

### **Issue**: "FITS file cannot be opened"
```bash
# Check file integrity
python -c "from astropy.io import fits; fits.info('your_file.fits')"
```

### **Issue**: "Memory error with large HDF5"
```python
# Process in batches
with h5py.File('large_file.h5', 'r') as f:
    total = len(f['images'])
    batch_size = 1000
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = f['images'][start:end]
        # Process batch...
```

### **Issue**: "Conversion too slow"
```bash
# Use smaller image size for testing
python scripts/convert_real_datasets.py --image-size 128 ...

# Or process subset
# Modify script to process first N images only
```

---

*Last Updated: 2025-10-03*
*Status: Priority 0 Fixes Implemented* ✅

