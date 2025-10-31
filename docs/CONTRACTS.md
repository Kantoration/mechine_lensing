# Model Contracts

Contracts define physics-aware input/output expectations for each model.

## Fields

| Field | Type | Description |
|------|------|-------------|
| `name` | str | Architecture name (e.g., `resnet18`, `lens_gnn`) |
| `bands` | list[str] | Explicit band order (e.g., `[g, r, i]`) |
| `input_size` | int | Expected input resolution for image models |
| `normalization` | dict | Per-band mean/std used by transforms |
| `pixel_scale_arcsec` | float? | Pixel scale in arcsec/pixel (optional; use if not providing `dx/dy`) |
| `dx`, `dy` | float? | Grid spacing in radians; preferred for operators |
| `sigma_crit_policy` | str | `dimensionless` or `physical` (κ units policy) |
| `task_type` | str | `classification`, `regression_kappa`, `regression_psi`, `regression_alpha` |
| `input_type` | str | `image`, `image+kappa`, or `full_maps` |

## Examples

### Classification (ResNet-18)
```yaml
models:
  - name: resnet18
    bands: [g, r, i]
    input_size: 224
    normalization:
      g: {mean: 0.018, std: 0.012}
      r: {mean: 0.020, std: 0.013}
      i: {mean: 0.022, std: 0.014}
    task_type: classification
    sigma_crit_policy: dimensionless
    input_type: image
```

### Regression (LensGNN)
```yaml
models:
  - name: lens_gnn
    bands: [g, r, i]
    input_size: 224
    pixel_scale_arcsec: 0.1  # or provide dx/dy directly
    normalization:
      g: {mean: 0.018, std: 0.012}
      r: {mean: 0.020, std: 0.013}
      i: {mean: 0.022, std: 0.014}
    task_type: regression_kappa
    sigma_crit_policy: dimensionless
    input_type: full_maps
```

## Why Contracts?
- Ensemble integrity: consistent band order, units, and shapes across members
- Physics invariants: explicit `dx/dy` avoid hidden 1-pixel assumptions
- Reproducibility: declared normalization and task/input types enable repeatable runs

## See Also
- Source: `src/models/ensemble/registry.py` (ModelContract)
- Tests: `tests/test_fits_loader_meta.py`, `tests/test_operators_anisotropic.py`
- Data transforms: `src/datasets/transforms.py`
