# For ML Developers: Technical Architecture & Implementation

## TL;DR

Binary classification of 64×64 astronomical images using ensemble of CNNs (ResNet-18/34) and Vision Transformers (ViT-B/16). **SOTA performance**: 96.3% accuracy, 98.9% ROC-AUC. Start here: [Architecture](#section-2-architecture-overview) → [Data](#section-3-data-pipeline) → [Training](#section-4-training-details) → [Evaluation](#section-5-evaluation-methodology)

---

## Section 1: Problem Framing

### Task: Binary Classification

- **Input**: Multi-band astronomical images (typically 3 bands: g, r, i), shape `[B, C, H, W]` where `C=3`, `H=W=64` (default)
- **Output**: `P(lens | image)`, probability of gravitational lens presence
- **Challenge**: Extreme class imbalance (1:1000 lens-to-non-lens ratio), small training sets, need for calibrated uncertainty

### Dataset Characteristics

| Property | Value | Notes |
|----------|-------|-------|
| **Image size** | 64×64 pixels (default) | Can be 32×32 to 224×224 |
| **Bands** | 3 (g, r, i) typical | SDSS, HSC, LSST compatible |
| **Pixel scale** | 0.1 - 0.4 arcsec/pixel | Survey-dependent |
| **Class ratio** | 1:1000 (lens:non-lens) | Positive-Unlabeled learning used |
| **Training samples** | 10K - 100K | Synthetic + real |

---

## Section 2: Architecture Overview

### Three Model Families

| Architecture | Strengths | Weaknesses | Use Case |
|--------------|-----------|------------|----------|
| **ResNet-18/34** | Fast, efficient, proven | Local features only | Production baseline |
| **ViT-B/16** | Global context, SOTA | Memory-hungry, slower | When accuracy critical |
| **Light Transformer** | Arc-aware attention | Domain-specific | Research/validation |

### Ensemble Strategy

**Uncertainty-weighted fusion**:
```python
# Each member produces: logits, uncertainty (from MC-dropout)
# Final prediction = weighted average with inverse-variance weights
final_logit = Σ(weight_i × logit_i) / Σ(weight_i)
where weight_i = 1 / (uncertainty_i + ε)
```

**Diversity requirements**:
- Different architectures (ResNet vs ViT)
- Different random seeds
- Different training data subsets (if available)

---

## Section 3: Data Pipeline

### FITS → CSV → PyTorch Dataset

**Step 1: FITS Files (Input)**
```
Image.fits
├── HDU 0: Primary (image data, shape [H, W, Bands])
├── HDU 1: Optional maps (kappa, psi, alpha)
└── Header: PIXSCALE, BANDS, Z_LENS, Z_SOURCE, SIGMA_CRIT
```

**Step 2: CSV Index (Intermediate)**
```csv
filepath,label,pixel_scale_arcsec,sigma_crit,z_lens,z_source
data/image_001.fits,1,0.187,1.5e8,0.5,2.0
```

**Step 3: PyTorch Dataset**
```python
from src.datasets.lens_fits_dataset import LensFITSDataset

dataset = LensFITSDataset(
    csv_path="train.csv",
    band_hdus={'g': 1, 'r': 2, 'i': 3},
    require_sigma_crit=True  # For physics pipelines
)
```

### Augmentation Strategy

**Flux-preserving transforms**:
- Random horizontal/vertical flips
- Small rotations (±10 degrees)
- **No color jitter by default** (preserves physics)

**Normalization**:
- **Per-band** zero-mean, unit-variance (default)
- Survey-specific stats from `ModelContract` (when available)
- **ImageNet normalization removed** (P1 hardening)

---

## Section 4: Training Details

### Loss Function

**Primary**: Weighted Binary Cross-Entropy
```python
loss = -[w_pos × y × log(p) + w_neg × (1-y) × log(1-p)]
where w_pos / w_neg = class_weight_ratio (typically 10-100)
```

**Optional**: Physics-informed loss (Poisson residual)
```python
physics_loss = ||∇²ψ - 2κ||²  # Enforces lensing equation
total_loss = classification_loss + λ × physics_loss
```

### Optimizer & Scheduler

- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: Cosine annealing with warmup (warmup=10% of total steps)
- **Batch size**: 64 (adjustable based on GPU memory)

### Hyperparameter Sensitivity

| Parameter | Sensitive Range | Impact |
|-----------|-----------------|--------|
| Learning rate | 1e-4 to 1e-3 | High (2-5% accuracy) |
| Weight decay | 1e-5 to 1e-4 | Medium (1-2% accuracy) |
| Dropout | 0.1 to 0.3 | Medium (1-2% accuracy) |
| Class weights | 10 to 100 | High (5-10% recall) |

---

## Section 5: Evaluation Methodology

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | When model says "lens", how often correct? |
| **Recall** | TP / (TP + FN) | What fraction of lenses did we find? |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balanced metric |
| **ROC-AUC** | Area under ROC curve | Discrimination ability |

### Calibration

**ECE (Expected Calibration Error)**:
```python
ECE = Σ |accuracy_bin_i - confidence_bin_i| × bin_size_i
```

**MCE (Maximum Calibration Error)**: Worst-case bin calibration

**Target**: ECE < 0.05 for reliable uncertainty estimates

---

## Section 6: Physics Constraints (Optional)

### When to Use

✅ **Use physics constraints when**:
- Training on synthetic data (ground truth κ, ψ available)
- Need physically consistent predictions
- Research on lensing physics

❌ **Skip physics constraints when**:
- Training on real data only (no κ, ψ maps)
- Maximum accuracy is the only goal
- Computational budget is tight

### Implementation

```python
from mlensing.gnn.physics_ops import poisson_residual

# Requires explicit dx, dy (P1 hardening)
dx = pixel_scale_arcsec * (π / 180 / 3600)  # radians
dy = pixel_scale_y_arcsec * (π / 180 / 3600)  # radians

# Compute Poisson residual: ∇²ψ - 2κ
residual = poisson_residual(psi, kappa, dx=dx, dy=dy)
physics_loss = residual.abs().mean()
```

**See**: [PHYSICS.md](PHYSICS.md) for complete details

---

## Section 7: Ensemble Methods

### Uncertainty-Weighted Fusion

**Input**: N ensemble members, each producing:
- `logits`: [B, num_classes]
- `uncertainty`: [B] (variance from MC-dropout)

**Output**: Weighted average with inverse-variance weights

```python
weights = 1.0 / (uncertainty + ε)  # ε = 1e-6
weights = weights / weights.sum()  # Normalize
final_logits = Σ(weights_i × logits_i)
```

### Physics-Informed Fusion

**Additional signal**: Per-sample physics loss
```python
# Members with lower physics loss get higher weight
physics_weight = exp(-physics_loss / temperature)
final_weight = uncertainty_weight × physics_weight
```

### Diversity Analysis

**Measure**: Disagreement between members
```python
disagreement = std(logits_members)  # Higher = more diverse
```

**Target**: Disagreement > 0.1 on test set (indicates real diversity)

---

## Section 8: Advanced Features

### Self-Supervised Pretraining

**MoCo v3 adaptation**:
- Contrastive learning on unlabeled FITS images
- Improves initialization for small labeled sets
- 10-15% accuracy boost on limited data

### Positive-Unlabeled Learning

**For extreme class imbalance** (π = 10⁻³):
```python
# Estimate labeling propensity: c = P(label=1 | lens=1)
# Then: P(lens=1 | x) ≈ P(label=1 | x) / c
```

**See**: `docs/CLUSTER_LENSING_SECTION.md` for PU learning implementation

---

## Section 9: Reproducibility & Testing

### Reproducing Paper Results

```bash
# 1. Generate synthetic dataset
python scripts/generate_dataset.py \
    --config configs/paper_synthetic.yaml \
    --num_samples 50000

# 2. Train ensemble members
for seed in 42 43 44; do
    python src/lit_train.py \
        --config configs/baseline.yaml \
        --seed $seed
done

# 3. Evaluate
python scripts/eval_paper.py --generate_figures
```

### Test Suite

**Critical tests** (run before any release):
```bash
pytest tests/test_operators_anisotropic.py \
       tests/test_fits_loader_meta.py \
       tests/test_ssl_schedule.py \
       tests/test_kappa_pooling_area.py \
       tests/test_tiled_inference_equiv.py \
       tests/test_sie_smoke.py
```

**Coverage**: 31 tests, all passing, 3× consecutive runs verified

---

## Section 10: Integration Paths

### Lightning AI Cloud Training

```python
from lightning import CloudCompute

trainer = Trainer(
    cloud_compute=CloudCompute("gpu-fast", wait=True)
)
```

### ONNX Export

```python
model_onnx = torch.onnx.export(model, dummy_input, "model.onnx")
```

### Hugging Face Model Hub

```python
# Future: Push trained models to HF hub
# Enables easy sharing and versioning
```

---

## Quick Start: 30 Minutes to Training

### Steps

**1. Prepare Data (5 minutes)**
```bash
python scripts/prepare_dataset.py \
    --fits_dir ./my_images/ \
    --output_dir ./data/processed/ \
    --train_split 0.8
```

**2. Train (15 minutes)**
```bash
python src/lit_train.py \
    --config configs/baseline.yaml \
    --data_dir ./data/processed/ \
    --max_epochs 50
```

**3. Evaluate**
```bash
python src/evaluate.py \
    --checkpoint lightning_logs/version_0/checkpoints/best.ckpt \
    --data_dir ./data/processed/test/
```

**See**: [QUICKSTART_ML_DEVELOPERS.md](QUICKSTART_ML_DEVELOPERS.md) for detailed walkthrough

---

## Navigation

**← Back to**: [README.md](../README.md)  
**→ Next**: [Quick Start](QUICKSTART_ML_DEVELOPERS.md) | [Architecture Deep Dive](#section-2-architecture-overview) | [Troubleshooting](TROUBLESHOOTING.md)

