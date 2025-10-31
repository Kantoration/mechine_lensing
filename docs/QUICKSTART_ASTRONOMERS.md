# Quick Start: 5 Minutes to Predictions (For Astronomers)

## You Have
- FITS images with gravitational lens candidates

## You Want
- Model predictions on your images

---

## Step 1: Install (1 minute)

```bash
# Install dependencies
pip install -r requirements.txt
```

**Verify installation**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## Step 2: Get Pre-trained Model (1 minute)

**Option A: Download pre-trained ensemble**
```bash
# If model is available on releases
wget https://github.com/.../releases/download/v1.0/ensemble_final.ckpt -O models/
```

**Option B: Use your own trained model**
```bash
# Point to your checkpoint
CHECKPOINT="lightning_logs/version_0/checkpoints/best.ckpt"
```

---

## Step 3: Prepare Your Data

### Create CSV Index

Create a CSV file (`my_images.csv`) listing your FITS files:

```csv
filepath,pixel_scale_arcsec
data/image_001.fits,0.187
data/image_002.fits,0.187
data/image_003.fits,0.187
```

**Required columns**:
- `filepath`: Path to FITS file
- `pixel_scale_arcsec`: Your telescope's pixel scale (e.g., SDSS: 0.396, HSC: 0.168)

**Optional columns** (for physics pipelines):
- `sigma_crit`: Critical surface density (Msun/pc²)
- `z_lens`: Lens redshift
- `z_source`: Source redshift

### Verify FITS Headers

Your FITS files should have:
- `PIXSCALE` header key (or provide in CSV)
- `BANDS` header key (e.g., "g,r,i") or specify in code

**Check a FITS file**:
```bash
python -c "from astropy.io import fits; h=fits.open('your_image.fits'); print(h[0].header['PIXSCALE'])"
```

---

## Step 4: Run Predictions (1 minute)

```bash
python scripts/inference.py \
    --model models/ensemble_final.ckpt \
    --csv_path my_images.csv \
    --output predictions.csv \
    --confidence_threshold 0.9
```

**Arguments**:
- `--model`: Path to trained checkpoint
- `--csv_path`: Your CSV index file
- `--output`: Where to save predictions
- `--confidence_threshold`: Minimum probability to mark as "lens" (default: 0.9)

---

## Step 5: Check Results (1 minute)

```bash
# View predictions
head predictions.csv
```

**Expected output**:
```csv
image_path,lens_probability,confidence,is_lens
data/image_001.fits,0.923,high,True
data/image_002.fits,0.156,low,False
data/image_003.fits,0.847,medium,True
```

**Interpreting results**:
- **lens_probability > 0.9**: Very likely a lens
- **lens_probability 0.7-0.9**: Probable lens (review manually)
- **lens_probability < 0.7**: Unlikely to be a lens

---

## Troubleshooting

### "FileNotFoundError: FITS file not found"

**Fix**: Check that paths in CSV match actual file locations
```bash
# Verify paths
python scripts/validate_dataset.py --csv_path my_images.csv
```

### "ValueError: missing 'pixel_scale_arcsec'"

**Fix**: Add `pixel_scale_arcsec` column to your CSV (required)

### "CUDA out of memory"

**Fix**: Use CPU instead
```bash
python scripts/inference.py ... --device cpu
```

**See**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more help

---

## Next Steps

- [Understand the results](FOR_ASTRONOMERS.md#understanding-the-results)
- [Train on your own data](QUICKSTART_ML_DEVELOPERS.md)
- [Learn about the physics](PHYSICS.md)

---

**← Back to**: [README.md](../README.md) | [For Astronomers Guide](FOR_ASTRONOMERS.md)

