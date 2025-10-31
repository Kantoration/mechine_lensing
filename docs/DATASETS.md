# Datasets & Loaders

## LensFITSDataset (FITS, multi-HDU)
- Bands from HDUs: map band names → HDU indices (e.g., `{g:0, r:1, i:2}`)
- Optional maps: `kappa`, `psi`, `alpha`; shape alignment required
- Optional `mask` and `psf` from HDUs or separate files

### Metadata Schema
```yaml
meta:
  bands: [g, r, i]
  pixel_scale_arcsec: 0.1
  dx: 4.848e-7   # radians
  dy: 4.848e-7   # radians
  z_l: 0.3
  z_s: 1.5
  sigma_crit: 1.0
```

### Transforms
- Use `src/datasets/transforms.py` for per-survey normalization (not ImageNet)
- Color jitter is opt-in and off by default

## ClusterLensingDataset (large FOV, tiling)
- Loads large cutouts; optional tiling with `tile` and `overlap`
- Provides `coords_grid` in arcsec and full meta with pixel scale
- Optional mask and PSF

## See Also
- Source: `src/datasets/lens_fits_dataset.py`, `src/datasets/cluster_lensing.py`, `src/datasets/transforms.py`
- Tests: `tests/test_fits_loader_meta.py`
- Tiled inference: `mlensing/gnn/inference_utils.py` and `docs/TILED_INFERENCE.md`
