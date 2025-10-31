# Datasets & Loaders

## LensFITSDataset (FITS, multi-HDU)
- Bands from HDUs: map band names → HDU indices (e.g., `{g:0, r:1, i:2}`)
- Optional maps: `kappa`, `psi`, `alpha`; shape alignment required
- Optional `mask` and `psf` from HDUs or separate files

### Required Metadata

**P1 Hardening**: Physics metadata is now **required** (no silent defaults):
- `pixel_scale_arcsec`: **REQUIRED** - Raises `ValueError` if missing (cannot derive dx/dy)
- `sigma_crit`: **REQUIRED** when `require_sigma_crit=True` (default: `False` for backward compat)

The dataset will raise `ValueError` during `__getitem__` if:
- `pixel_scale_arcsec` is missing from CSV and FITS header lacks `PIXSCALE`
- `sigma_crit` is missing when `require_sigma_crit=True` (for physics pipelines)

### Metadata Schema
```yaml
meta:
  bands: [g, r, i]
  pixel_scale_arcsec: 0.1      # REQUIRED (no default)
  pixel_scale_y_arcsec: 0.12   # Optional (defaults to pixel_scale_arcsec)
  dx: 4.848e-7                  # radians (auto-derived from pixel_scale_arcsec)
  dy: 5.818e-7                  # radians (auto-derived, supports anisotropy)
  z_l: 0.3
  z_s: 1.5
  sigma_crit: 1.5e8             # REQUIRED for physics pipelines (when require_sigma_crit=True)
```

### Usage Example
```python
from src.datasets.lens_fits_dataset import LensFITSDataset

# For physics pipelines, require explicit metadata
dataset = LensFITSDataset(
    csv_path="manifest.csv",
    band_hdus={'g': 1, 'r': 2, 'i': 3},
    require_sigma_crit=True  # Enforce sigma_crit for physics losses
)

# CSV must include:
# filepath, pixel_scale_arcsec, sigma_crit (when require_sigma_crit=True)
```

### Transforms
- Use `src/datasets/transforms.py` for per-survey normalization (not ImageNet)
- Default normalization: zero-mean, unit-variance per band (preserves flux calibration)
- Color jitter is opt-in and off by default

## ClusterLensingDataset (large FOV, tiling)
- Loads large cutouts; optional tiling with `tile` and `overlap`
- Provides `coords_grid` in arcsec and full meta with pixel scale
- Optional mask and PSF

### Required Metadata

**P1 Hardening**: Same requirements as `LensFITSDataset`:
- `pixel_scale_arcsec`: **REQUIRED** in CSV (checked at `__init__`)
- `sigma_crit`: **REQUIRED** when `require_sigma_crit=True` (for physics pipelines)

The dataset will raise `ValueError` during `__init__` if `pixel_scale_arcsec` column is missing.

## See Also
- Source: `src/datasets/lens_fits_dataset.py`, `src/datasets/cluster_lensing.py`, `src/datasets/transforms.py`
- Tests: `tests/test_fits_loader_meta.py`
- Tiled inference: `mlensing/gnn/inference_utils.py` and `docs/TILED_INFERENCE.md`
