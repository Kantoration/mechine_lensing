# Glossary: Key Terms Explained

## Astronomy Terms

### Gravitational Lensing
The bending of light by massive objects. When a massive foreground object (lens) sits between us and a distant background object (source), the light path gets distorted, creating arc-like patterns.

### Einstein Radius (θ_E)
The characteristic angular scale of lensing. For galaxy-scale lenses: θ_E = 1″–2″. For cluster-scale lenses: θ_E = 10″–30″. Larger masses produce larger Einstein radii.

### Pixel Scale (arcsec/pixel)
The angular size of one pixel in the image. Examples:
- SDSS: 0.396 arcsec/pixel
- HSC: 0.168 arcsec/pixel
- HST/ACS: 0.05 arcsec/pixel

### Critical Surface Density (σ_crit)
The surface mass density required to produce strong lensing: σ_crit = c²/(4πG) × D_s/(D_l × D_ls). Units: Msun/pc². Typical values: 1e7 - 1e9 Msun/pc².

### Convergence (κ)
The dimensionless surface mass density. κ = Σ/σ_crit, where Σ is the projected mass density. Related to lensing potential ψ via Poisson equation: ∇²ψ = 2κ.

### Lensing Potential (ψ)
The 2D gravitational potential related to convergence. Deflection angles α are derived from ψ: α = ∇ψ.

### Deflection Angle (α)
The angle by which light rays are bent. Components: α_x, α_y. Units: radians. Derived from potential: α = ∇ψ.

---

## Machine Learning Terms

### Ensemble Method
Combining multiple models' predictions for higher accuracy and reliability. Our ensemble uses uncertainty-weighted fusion: predictions from confident models get higher weight.

### MC-Dropout (Monte Carlo Dropout)
A technique for estimating model uncertainty. Run inference multiple times with dropout enabled, compute variance of predictions. Higher variance = higher uncertainty.

### ROC-AUC
Receiver Operating Characteristic - Area Under Curve. Measures how well the model discriminates between classes. Range: 0.5 (random) to 1.0 (perfect). Our models achieve 0.989.

### Calibration Error (ECE/MCE)
**ECE** (Expected Calibration Error): Average difference between predicted confidence and actual accuracy across confidence bins.  
**MCE** (Maximum Calibration Error): Worst-case bin calibration error.  
Target: ECE < 0.05 for reliable uncertainty estimates.

### Positive-Unlabeled (PU) Learning
Learning from data where only positive examples are labeled, negatives are unlabeled. Used for extreme class imbalance (π = 10⁻³ for cluster lensing).

---

## Cross-Domain Terms

### Signal-to-Noise Ratio (SNR)
**Astronomy**: Ratio of source flux to background noise. Higher SNR = easier detection.  
**ML**: Ratio of signal variance to noise variance. Higher SNR = easier learning.

### Normalization
**Astronomy**: Converting flux values to standard units (e.g., zero-point magnitudes).  
**ML**: Scaling pixel values to zero-mean, unit-variance for numerical stability. Our default: per-band zero-mean, unit-variance (preserves flux calibration).

### Anisotropy
**Physics**: Different properties in different directions. In lensing: different pixel scales in x and y (dx ≠ dy).  
**ML**: Models that handle directional biases. Our physics operators support explicit dx/dy for anisotropic grids.

---

## Project-Specific Terms

### ModelContract
A metadata structure defining model expectations:
- `bands`: Band names (e.g., ['g', 'r', 'i'])
- `normalization`: Per-band mean/std statistics
- `pixel_scale_arcsec`: Pixel scale in arcseconds
- `task_type`: 'classification' or 'regression'
- `input_type`: 'image' or 'graph'

### PhysicsScale
Data structure for managing pixel scales and physical constants:
- `pixel_scale_arcsec`: Pixel scale in arcseconds
- `dx`, `dy`: Explicit spacing in radians (supports anisotropy)
- Used by physics operators to ensure correct units.

### P1 Hardening
Code changes enforcing explicit metadata (no silent defaults):
- `pixel_scale_arcsec`: **REQUIRED** in datasets
- `sigma_crit`: **REQUIRED** for physics pipelines
- Explicit `dx`/`dy`: **REQUIRED** in physics operators
- Prevents incorrect physics calculations from missing metadata.

### Tiled Inference
Processing large images by splitting into overlapping tiles, then blending results. Used for cluster-scale images (128×128 or larger). Blending uses Hanning window to reduce edge artifacts.

---

## Quick Reference

| Term | Domain | Short Definition |
|------|--------|------------------|
| Einstein Radius | Astronomy | Characteristic lensing scale |
| Convergence (κ) | Physics | Dimensionless mass density |
| Ensemble | ML | Multiple models combined |
| MC-Dropout | ML | Uncertainty estimation technique |
| ROC-AUC | ML | Discrimination metric (0.5-1.0) |
| Pixel Scale | Astronomy | Arcseconds per pixel |
| ModelContract | Project | Model metadata structure |
| P1 Hardening | Project | Explicit metadata enforcement |

---

**← Back to**: [README.md](../README.md)

