# P1 Hardening Summary

This document summarizes the P1 hardening changes applied to enforce physics metadata and remove unsafe defaults.

## Overview

The P1 hardening effort ensures that:
1. Physics pipelines **fail-fast** on missing critical metadata
2. No silent defaults that could lead to incorrect physics calculations
3. Survey-aware normalization replaces ImageNet defaults
4. Anisotropic grids are properly handled throughout

---

## Changes by Component

### 1. Dataloaders

#### LensFITSDataset & ClusterLensingDataset

**Before**: Silent defaults for `pixel_scale_arcsec=0.1` and `sigma_crit=1.0`

**After**: 
- `pixel_scale_arcsec`: **REQUIRED** - Raises `ValueError` if missing
- `sigma_crit`: **REQUIRED** when `require_sigma_crit=True` (default: `False` for backward compat)
- `dx`, `dy`: Auto-derived from pixel scales (supports anisotropy via `pixel_scale_y_arcsec`)

**Breaking Change**: If your CSV/dataset lacks `pixel_scale_arcsec`, you must add it. The loader will no longer silently default to 0.1 arcsec/pixel.

**Migration**:
```python
# Old (would silently default)
dataset = LensFITSDataset(csv_path="data.csv", band_hdus={'g': 0})

# New (explicit metadata required)
# CSV must have: filepath, pixel_scale_arcsec
dataset = LensFITSDataset(
    csv_path="data.csv", 
    band_hdus={'g': 0},
    require_sigma_crit=True  # For physics pipelines
)
```

---

### 2. Physics Operators

#### Explicit Spacing Required

**Before**: Operators could use `pixel_scale_rad=1.0` as silent default

**After**: 
- Operators require either `(dx, dy)` OR `pixel_scale_rad` (backward compat)
- Raise `TypeError` if neither provided
- No averaging of `dx/dy` - preserves anisotropy

**Function Signatures**:
```python
# Preferred (explicit anisotropy)
gradient2d(field, dx=1e-5, dy=1e-5, bc=...)
laplacian2d(field, dx=1e-5, dy=1e-5, bc=...)
poisson_residual(psi, kappa, dx=1e-5, dy=1e-5, ...)

# Backward compat (isotropic)
gradient2d(field, pixel_scale_rad=1e-5, bc=...)
```

**Migration**: If calling physics ops directly, ensure you pass spacing explicitly.

---

### 3. Graph Builder

#### PhysicsScale Required

**Before**: `build_grid_graph(images, physics_scale=None)` would default to `PhysicsScale(pixel_scale_arcsec=0.1)`

**After**: Requires explicit `PhysicsScale` - raises `ValueError` if `None`

**Migration**:
```python
# Old (would default)
graph = build_grid_graph(images)

# New (explicit scale required)
from mlensing.gnn.physics_ops import PhysicsScale
scale = PhysicsScale(pixel_scale_arcsec=0.1)
graph = build_grid_graph(images, physics_scale=scale)
```

---

### 4. LensGNN

#### Physics Scale in Batch Meta

**Before**: Could fall back to `pixel_scale_rad=1.0` if `physics_scale` missing

**After**: Requires `physics_scale` in batch meta - raises `ValueError` if missing

**Migration**: Ensure your dataloader/graph builder provides `physics_scale` in batch meta.

---

### 5. Normalization

#### ImageNet Removed

**Before**: Default normalization used ImageNet stats `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

**After**: 
- Astronomy default: zero-mean, unit-variance per band
- Survey-specific stats from `ModelContract` when available
- Color jitter opt-in only (default OFF)

**Migration**: 
- No action needed - automatically uses astronomy defaults
- Provide survey stats in `ModelContract` for optimal performance
- Explicitly enable color jitter if desired: `use_color_jitter=True`

---

## Testing

All changes are covered by tests:

- `tests/test_loader_require_meta.py` - Metadata enforcement
- `tests/test_no_imagenet_norm.py` - ImageNet removal verification
- `tests/test_no_isotropic_defaults.py` - Explicit spacing requirements
- `tests/test_graph_requires_scale.py` - Graph builder validation
- `tests/test_lensgnn_anisotropic.py` - Anisotropy handling

Run verification:
```bash
py scripts/ci_gates.py  # CI gate checks
py -m pytest tests/test_loader_require_meta.py tests/test_no_imagenet_norm.py tests/test_no_isotropic_defaults.py tests/test_graph_requires_scale.py tests/test_lensgnn_anisotropic.py -v
```

---

## Verification

After applying P1 hardening, verify:

1. **No ImageNet normalization**: `py scripts/grep_regressions.py`
2. **No isotropic defaults**: CI gates check for `pixel_scale_rad=1.0` and `PhysicsScale(pixel_scale_arcsec=0.1)`
3. **Tests pass**: All 31 tests passing, 3× consecutive runs

---

## Backward Compatibility

- `require_sigma_crit=False` by default (allows existing non-physics pipelines)
- Physics ops accept `pixel_scale_rad` for legacy code paths
- No breaking changes for code that already provides explicit metadata

---

## References

- Full verification report: `REPORT_RELEASE.md`
- Original findings: `REPORT.md`
- CI gates: `scripts/ci_gates.py`
- Regression checks: `scripts/grep_regressions.py`

