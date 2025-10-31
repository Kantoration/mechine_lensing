# Quick Start: 30 Minutes to Training (For ML Developers)

## You Have
- Labeled FITS images (organized in directory)
- GPU or CPU available

## You Want
- Custom trained model on your data

---

## Step 1: Install (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools
```

**Verify GPU** (optional):
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Step 2: Prepare Data (5 minutes)

### Create CSV Index

```bash
python scripts/prepare_dataset.py \
    --fits_dir ./labeled_images/ \
    --output_dir ./data/processed/ \
    --train_split 0.8 \
    --test_split 0.1
```

**This creates**:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

**Required CSV format**:
```csv
filepath,label,pixel_scale_arcsec
data/image_001.fits,1,0.187
data/image_002.fits,0,0.187
```

**Columns**:
- `filepath`: Path to FITS file
- `label`: 1 = lens, 0 = non-lens
- `pixel_scale_arcsec`: **REQUIRED** - arcsec/pixel

**For physics pipelines**, also include:
- `sigma_crit`: Critical surface density (Msun/pc²)

---

## Step 3: Configure Training (5 minutes)

Edit `configs/baseline.yaml`:

```yaml
model:
  name: resnet18
  bands: 3
  pretrained: false

data:
  csv_path: data/processed/train.csv
  batch_size: 64
  num_workers: 4

training:
  max_epochs: 50
  learning_rate: 1e-3
  weight_decay: 1e-4
```

**Key parameters**:
- `batch_size`: Adjust based on GPU memory (64 default, reduce if OOM)
- `learning_rate`: Start with 1e-3, adjust if loss unstable
- `max_epochs`: 50-100 typical, use early stopping

---

## Step 4: Train (15 minutes)

```bash
python src/lit_train.py \
    --config configs/baseline.yaml \
    --data_dir ./data/processed/ \
    --trainer.max_epochs 50 \
    --trainer.devices 1
```

**Monitor training**:
```bash
# TensorBoard logs
tensorboard --logdir lightning_logs/
```

**Expected output**:
```
Epoch 0: train_loss=0.65, val_loss=0.58, val_acc=0.82
Epoch 1: train_loss=0.52, val_loss=0.51, val_acc=0.85
...
Epoch 49: train_loss=0.12, val_loss=0.15, val_acc=0.96
```

---

## Step 5: Evaluate (5 minutes)

```bash
python src/evaluate.py \
    --checkpoint lightning_logs/version_0/checkpoints/best.ckpt \
    --data_dir ./data/processed/test/
```

**Expected metrics**:
```
Accuracy: 0.963
Precision: 0.945
Recall: 0.912
F1 Score: 0.928
ROC-AUC: 0.989
```

---

## Advanced: Train Ensemble

```bash
# Train multiple models with different seeds
for seed in 42 43 44; do
    python src/lit_train.py \
        --config configs/baseline.yaml \
        --seed $seed \
        --output checkpoints/model_seed_$seed.ckpt
done

# Evaluate ensemble
python src/evaluate_ensemble.py \
    --checkpoints checkpoints/model_seed_*.ckpt \
    --data_dir ./data/processed/test/
```

---

## Understanding Config

**Model selection**:
```yaml
model:
  name: resnet18      # Options: resnet18, resnet50, vit_b_16
  bands: 3            # Number of bands (g, r, i = 3)
  pretrained: false   # Use ImageNet pretrained (not recommended for astronomy)
```

**Data augmentation**:
```yaml
augmentation:
  rotation: 10        # Random rotation ±10 degrees
  flip: true          # Horizontal/vertical flips
  jitter: false       # Color jitter (OFF by default for physics)
```

**Normalization**:
```yaml
normalization:
  g: {mean: 0.018, std: 0.012}
  r: {mean: 0.020, std: 0.013}
  i: {mean: 0.022, std: 0.014}
# Or use 'auto' to compute from training data
```

**See**: [CONFIGURATION.md](CONFIGURATION.md) for complete reference

---

## Troubleshooting

### "CUDA out of memory"

**Fix**: Reduce batch size
```yaml
data:
  batch_size: 32  # was 64
```

### "Loss is NaN"

**Fix**: Lower learning rate
```yaml
training:
  learning_rate: 1e-4  # was 1e-3
```

### "Model overfits"

**Fix**: Add regularization
```yaml
training:
  weight_decay: 1e-4
  dropout: 0.5
```

**See**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for complete guide

---

## Next Steps

- [Architecture details](FOR_ML_DEVELOPERS.md#section-2-architecture-overview)
- [Training best practices](FOR_ML_DEVELOPERS.md#section-4-training-details)
- [Reproduce paper results](FOR_ML_DEVELOPERS.md#section-9-reproducibility--testing)

---

**← Back to**: [README.md](../README.md) | [For ML Developers Guide](FOR_ML_DEVELOPERS.md)

