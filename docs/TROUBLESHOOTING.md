# Troubleshooting & FAQ

## Installation & Setup

### Error: "ModuleNotFoundError: No module named 'torch'"

**Cause**: PyTorch not installed

**Solution**:
```bash
pip install torch torchvision torchaudio
# For CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify**:
```bash
python -c "import torch; print(torch.__version__)"
```

---

### Error: "CUDA out of memory"

**Causes**:
1. Batch size too large
2. Image resolution too high
3. Not using mixed precision
4. Model too large for GPU

**Solutions** (in order of preference):

**Option 1: Reduce batch size (easiest)**
```bash
python src/lit_train.py --batch_size 16  # was 64
```

**Option 2: Use mixed precision (keeps quality)**
```bash
python src/lit_train.py --precision 16-mixed  # was 32
```

**Option 3: Reduce image size (trades accuracy)**
```bash
python src/lit_train.py --image_size 32  # was 64
```

**Option 4: CPU training (slowest but always works)**
```bash
python src/lit_train.py --device cpu
```

**Check your GPU memory**:
```bash
nvidia-smi  # See which processes use memory
```

---

## Data Issues

### Error: "FileNotFoundError: /path/to/image.fits not found"

**Cause**: FITS paths in CSV don't match actual files

**Solution**:
```bash
# Validate dataset paths
python scripts/validate_dataset.py \
    --csv data/train.csv \
    --fits_root data/fits/
```

This will tell you which files are missing.

---

### Error: "ValueError: FITS sample missing 'pixel_scale_arcsec'"

**Cause**: Missing required metadata (P1 hardening - no silent defaults)

**Solution**:
Add `pixel_scale_arcsec` column to your CSV:
```csv
filepath,label,pixel_scale_arcsec
data/image_001.fits,1,0.187
```

**Or** ensure FITS header has `PIXSCALE` key:
```python
from astropy.io import fits
h = fits.open('image.fits')
h[0].header['PIXSCALE'] = 0.187  # arcsec/pixel
h.writeto('image.fits', overwrite=True)
```

---

### Error: "ValueError: Cannot find normalization for band 'u'"

**Cause**: Band 'u' not in config normalization dictionary

**Solution**:

**Option A: Use auto-normalization**
```yaml
# In config.yaml
normalization: auto  # Compute from training data
```

**Option B: Provide band statistics**
```yaml
normalization:
  u: {mean: 0.015, std: 0.010}
  g: {mean: 0.018, std: 0.012}
  r: {mean: 0.020, std: 0.013}
```

---

## Training Issues

### Problem: "Loss is NaN after 2 epochs"

This indicates numerical instability. Check in order:

**1. Check learning rate (too high = instability)**
```yaml
optimizer:
  lr: 1e-4  # Try 1e-4 instead of 1e-3
```

**2. Check batch normalization settings**
```yaml
batch_norm_momentum: 0.01  # Use lower momentum for small datasets
```

**3. Check physics constraints (if enabled)**
```yaml
physics:
  loss_weight: 0.1  # Too high? Try 0.01
```

**Debug script**:
```bash
python scripts/debug_training.py \
    --config your_config.yaml \
    --num_batches 5 \
    --verbose
```

---

### Problem: "Model overfits (train loss→0, val loss→high)"

The model memorizes training data. Add regularization:

**Solution 1: More data augmentation**
```yaml
augmentation:
  rotation: 30
  shift: 0.1
  scale: 0.2
```

**Solution 2: More regularization**
```yaml
dropout: 0.5           # was 0.3
weight_decay: 1e-4     # was 1e-5
early_stopping: 20     # Stop if val doesn't improve
```

**Solution 3: More training data** (best solution if possible)

---

### Problem: "Ensemble members have low diversity"

Members should disagree on some predictions. If they always agree, they're not diverse.

**Cause**: All members trained identically

**Solution**: Train with different random seeds
```bash
for seed in 42 43 44; do
    python src/lit_train.py \
        --config configs/baseline.yaml \
        --seed $seed \
        --output checkpoints/model_seed_$seed.ckpt
done
```

**Check diversity**:
```bash
python scripts/analyze_ensemble_diversity.py \
    --models checkpoints/model_seed_*.ckpt \
    --test_data data/test/
```

---

## Inference Issues

### Problem: "Model predicts all zeros (or all ones)"

The model always predicts the same class.

**Cause 1**: Severe class imbalance in training  
**Cause 2**: Model didn't train properly  
**Cause 3**: Input preprocessing mismatch

**Debug**:
```bash
# Check class balance
python scripts/analyze_dataset.py data/train.csv
# Output should show ~50% lens, ~50% non-lens

# Check input preprocessing
python scripts/debug_inference.py \
    --model checkpoint.ckpt \
    --sample data/test/image.fits \
    --verbose
```

---

## Physics Issues

### Error: "TypeError: gradient2d() missing required argument 'dx'"

**Cause**: Physics operators require explicit spacing (P1 hardening)

**Solution**: Pass `dx` and `dy` explicitly:
```python
from mlensing.gnn.physics_ops import gradient2d

# Extract from metadata
dx = sample['meta']['dx']  # radians
dy = sample['meta']['dy']  # radians

gx, gy = gradient2d(field, dx=dx, dy=dy)
```

**Or** use backward-compatible `pixel_scale_rad`:
```python
pixel_scale_rad = 0.187 * (3.14159 / 180 / 3600)  # Convert arcsec to rad
gx, gy = gradient2d(field, pixel_scale_rad=pixel_scale_rad)
```

---

### Error: "ValueError: Graph builder requires explicit PhysicsScale"

**Cause**: Graph builder no longer defaults to `PhysicsScale(pixel_scale_arcsec=0.1)`

**Solution**: Provide explicit `PhysicsScale`:
```python
from mlensing.gnn.physics_ops import PhysicsScale
from mlensing.gnn.graph_builder import build_grid_graph

scale = PhysicsScale(pixel_scale_arcsec=0.187)
graph = build_grid_graph(images, physics_scale=scale)
```

---

## Performance Issues

### Problem: "Training is very slow"

**Check 1**: Using CPU instead of GPU?
```bash
python scripts/check_device.py
# Should print: Using CUDA device 0
```

**Check 2**: Data loading bottleneck?
```bash
# Profile data loading
python src/lit_train.py --profile_dataloader
```

**Check 3**: Wrong number of workers?
```yaml
data:
  num_workers: 4  # If available CPUs >> 4, increase this
```

---

## Model Issues

### Problem: "Cannot load checkpoint - version mismatch"

**Cause**: Checkpoint trained with different code version

**Solution**: Retrain or use model from same code version

```bash
# Check checkpoint info
python scripts/inspect_checkpoint.py checkpoint.ckpt
```

---

## Getting Help

- **Bugs**: [GitHub Issues](https://github.com/Kantoration/mechine_lensing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Kantoration/mechine_lensing/discussions)
- **Documentation**: [Complete Docs](.)

---

**← Back to**: [README.md](../README.md)

