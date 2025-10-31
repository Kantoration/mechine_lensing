# For Astronomers: Machine Learning Guide to Gravitational Lens Detection

## TL;DR

This tool uses AI to automatically find gravitational lenses in telescope images, processing thousands of images in hours (vs. months of manual work). Use it to run inference on your images, check performance on known lenses, and understand the physics behind the detection.

---

## What Is Gravitational Lensing? (5 Minutes Reading)

**Gravitational lensing** occurs when massive objects bend light from distant galaxies. Think of it like looking through a glass lens - the light path gets distorted, creating spectacular arc-like patterns.

### Why This Matters

- **Dark matter mapping**: Lens positions reveal where dark matter is located
- **Cosmology constraints**: Lensing helps measure the expansion rate of the universe
- **High-redshift galaxies**: Lensed images let us see very distant galaxies more clearly

### The Detection Challenge

Finding lenses is hard because:
1. **Rarity**: Only ~1 in 1,000 galaxy clusters produces visible lensing arcs
2. **Subtle signatures**: Arcs can be faint and blend with other features
3. **Human limitations**: Even expert astronomers miss lenses in crowded images
4. **Scale diversity**: Different telescopes see different scales (HST: 0.05 arcsec/pixel, SDSS: 0.4 arcsec/pixel)

**Current methods**: Manual inspection by astronomers (slow, subjective, inconsistent)

**This tool**: Automated detection with machine learning (fast, reproducible, scalable)

---

## How ML Finds Lenses (Intuitive)

### The Analogy: Training a Grad Student

Imagine training a graduate student to recognize lenses:
1. **Show examples**: "Here are 100 confirmed lenses, here are 100 non-lenses"
2. **Learn patterns**: Student notices arcs, color differences, radial positions
3. **Generalize**: Student can now identify new lenses

**Machine learning does the same**, but:
- Can process **millions** of examples
- Never gets tired
- Consistent across all images
- Learns from **all** pixels simultaneously (not just obvious features)

### Different "Expert Models"

Our system uses multiple models, like having different expert observers:

| Expert Model | What They're Good At |
|--------------|---------------------|
| **ResNet-18** | Local patterns, edges, arc segments |
| **ViT-B/16** | Global relationships, overall image structure |
| **Light Transformer** | Arc morphology, specialized lens features |

**Why multiple models?** Just like you'd want multiple astronomers to review a candidate lens, we combine their opinions for higher confidence.

---

## Understanding the Results

### Model Output

When you run inference, you get:
```csv
image_path, lens_probability, confidence, is_lens
image_001.fits, 0.923, high, True
image_002.fits, 0.156, low, False
image_003.fits, 0.847, medium, True
```

**Interpreting probabilities**:
- **> 0.9**: Very likely a lens (high confidence)
- **0.7 - 0.9**: Probable lens (medium confidence, review recommended)
- **< 0.7**: Unlikely to be a lens (low confidence)

### When to Trust the Model

✅ **Trust when**:
- Multiple models agree (ensemble confidence > 0.95)
- Arc-like features are visible to the eye
- Probability is consistent across different image bands

⚠️ **Review manually when**:
- Models disagree significantly (ensemble uncertainty is high)
- Probability is borderline (0.7 - 0.9)
- Image has unusual artifacts or data quality issues

---

## Dataset Preparation (Plain English)

### What Images Work?

Your FITS files need:
- **Multiple bands**: Typically 3 bands (e.g., g, r, i from SDSS)
- **Proper headers**: Pixel scale (`PIXSCALE` in arcsec/pixel) is **required**
- **Standard format**: FITS with image data in Primary HDU

### Required Metadata

**P1 Hardening Update**: The system now **requires** explicit metadata (no silent defaults):

- `pixel_scale_arcsec`: **REQUIRED** - Your telescope's pixel scale (e.g., SDSS: 0.396 arcsec/pixel)
- `sigma_crit`: **REQUIRED for physics pipelines** - Critical surface density (Msun/pc²)

**Why?** This prevents incorrect physics calculations from wrong pixel scales.

### FITS File Structure

```
YourImage.fits
├── HDU 0 (Primary)
│   ├── Data: (height × width × bands) image array
│   ├── Header keys (REQUIRED):
│   │   PIXSCALE = 0.187        ← arcsec/pixel
│   │   BANDS = "g,r,i"          ← band names
│   │   Z_LENS = 0.5             ← lens redshift
│   │   Z_SOURCE = 2.0           ← source redshift
│   │   SIGMA_CRIT = 1.234e15    ← Msun/pc² (if using physics)
```

### CSV Index File

Create a CSV file pointing to your FITS files:
```csv
filepath,label,pixel_scale_arcsec,sigma_crit
data/image_001.fits,1,0.187,1.5e8
data/image_002.fits,0,0.187,1.5e8
```

**Columns**:
- `filepath`: Path to FITS file (relative or absolute)
- `label`: 1 = lens, 0 = non-lens
- `pixel_scale_arcsec`: **REQUIRED** - arcsec/pixel
- `sigma_crit`: Required for physics pipelines

---

## Quick Start: 5 Minutes to Predictions

### You Have
- FITS images (with required headers)

### You Want
- Model predictions on your images

### Steps

**1. Install (1 minute)**
```bash
pip install -r requirements.txt
```

**2. Get Pre-trained Model (1 minute)**
```bash
# Download ensemble model (if available)
# Or use a trained checkpoint from your training runs
```

**3. Run Predictions (1 minute)**
```bash
python scripts/inference.py \
    --model checkpoint.ckpt \
    --data_dir ./my_images/ \
    --output predictions.csv \
    --confidence_threshold 0.9
```

**4. Check Results (1 minute)**
```bash
# View predictions
head predictions.csv

# Expected output:
# image_001.fits, 0.923, high, True
# image_002.fits, 0.156, low, False
```

### Next Steps

- [Understand the results](#understanding-the-results)
- [Use on your own data](#dataset-preparation-plain-english)
- [Troubleshooting](TROUBLESHOOTING.md)

---

## Interpreting Common Errors

### "Out of memory"

**What it means**: Your GPU/computer doesn't have enough memory.

**How to fix**:
1. Use smaller images (reduce `--image_size`)
2. Process fewer images at once (reduce `--batch_size`)
3. Use CPU instead of GPU (slower but always works)

### "NaN in loss"

**What it means**: Something went wrong with the physics calculations.

**How to fix**:
1. Check that `pixel_scale_arcsec` is correct in your CSV
2. Verify `sigma_crit` values are reasonable (typically 1e7 - 1e9 Msun/pc²)
3. Try disabling physics constraints temporarily

### "Poor performance"

**What it means**: Model accuracy is lower than expected.

**How to debug**:
1. Check class balance (should be ~50% lenses, ~50% non-lenses in training)
2. Verify image normalization is correct
3. Ensure training data quality is good

**See full troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Paper & Citation

### What Research Does This Implement?

This project implements state-of-the-art gravitational lens detection methods from:
- **Bologna Challenge** (2018): Benchmark for lens detection algorithms
- **DES Strong Lensing** (2019-2023): Real survey applications
- **LSST Science Pipeline**: Production-grade processing

### How to Cite

If you use this code in your research, please cite:
- The repository: [GitHub link]
- Key papers: See [docs/PHYSICS.md](PHYSICS.md) for complete references

---

## Getting More Help

- **Technical details**: [FOR_ML_DEVELOPERS.md](FOR_ML_DEVELOPERS.md)
- **Data format specifics**: [DATASETS.md](DATASETS.md)
- **Physics explanations**: [PHYSICS.md](PHYSICS.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Repository structure**: [REPO_MAP.md](REPO_MAP.md)

---

## Navigation

**← Back to**: [README.md](../README.md)  
**→ Next**: [Quick Start Guide](QUICKSTART_ASTRONOMERS.md) | [Troubleshooting](TROUBLESHOOTING.md)

