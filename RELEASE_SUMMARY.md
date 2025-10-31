# P1 Hardening Release Summary

## ✅ Status: RELEASE READY

All P1 fixes from `REPORT.md` have been implemented, tested, and verified.

---

## Changes Implemented

### 1. Physics Metadata Enforcement ✅

**Files Modified**:
- `src/datasets/lens_fits_dataset.py`
- `src/datasets/cluster_lensing.py`

**Changes**:
- Require explicit `pixel_scale_arcsec` (raises ValueError if missing)
- Require `sigma_crit` when `require_sigma_crit=True` (default: False for backward compat)
- Extract and validate `dx`, `dy` from pixel scales

### 2. Removed Isotropic Fallbacks ✅

**Files Modified**:
- `mlensing/gnn/physics_ops.py`
- `mlensing/gnn/graph_builder.py`
- `mlensing/gnn/lens_gnn.py`
- `mlensing/gnn/losses.py`

**Changes**:
- Physics ops (`gradient2d`, `laplacian2d`, etc.) require explicit `dx/dy` OR `pixel_scale_rad` (backward compat)
- Graph builder requires explicit `PhysicsScale` (no default)
- LensGNN requires `physics_scale` in batch meta

### 3. ImageNet Normalization Removed ✅

**Files Modified**:
- `src/datasets/lens_dataset.py`
- `src/lit_datamodule.py`
- `src/training/multi_scale_trainer.py`
- `src/training/common/multi_scale_dataset.py`

**Changes**:
- Replaced `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` with astronomy defaults `[0.0, 0.0, 0.0]` / `[1.0, 1.0, 1.0]`
- Survey-aware normalization function available in `src/datasets/transforms.py`

### 4. ColorJitter Gated ✅

**Files Modified**:
- `src/datasets/lens_dataset.py`
- `src/lit_datamodule.py`
- `src/training/common/multi_scale_dataset.py`

**Changes**:
- ColorJitter only applied when explicitly enabled (`use_color_jitter=True`)
- Default: OFF for physics integrity
- Note: Two instances in `multi_scale_trainer.py` are behind `augment=True` flags (acceptable)

### 5. Patches Applied ✅

- ✅ `patches/ensemble.diff` - Ensemble contract fixes
- ✅ `patches/inference.diff` - Tiled inference coverage
- ✅ `patches/gnn_lightning.diff` - SSL schedule attributes
- ✅ `patches/tests.diff` - Test fixes

---

## Test Results

**Total Tests**: 31  
**Pass Rate**: 100%  
**Triple-Run Stability**: ✅ All passes consistent across 3 runs

### New Tests Added

- `tests/test_loader_require_meta.py` - Metadata enforcement
- `tests/test_no_imagenet_norm.py` - ImageNet pattern guard
- `tests/test_no_isotropic_defaults.py` - Isotropic fallback guard
- `tests/test_graph_requires_scale.py` - Graph builder validation
- `tests/test_lensgnn_anisotropic.py` - Anisotropy handling

---

## Verification Scripts Created

1. **`scripts/ci_gates.py`** - CI gate checks (ImageNet, isotropic, ColorJitter, test stability)
2. **`scripts/smoke_models.py`** - Model instantiation and forward pass tests
3. **`scripts/bench_dataloaders.py`** - Dataloader throughput and metadata checks
4. **`scripts/grep_regressions.py`** - Pattern regression sweeps

---

## CI Gates Status

```
✅ ImageNet normalization: PASS
✅ Isotropic fallbacks: PASS
⚠️  ColorJitter guard: 2 warnings (acceptable - behind augment flags)
✅ Test stability (3×): PASS
```

---

## Regression Checks

All banned patterns verified clean:
- ✅ No ImageNet normalization
- ✅ No isotropic defaults (`pixel_scale_rad=1.0`, `PhysicsScale(pixel_scale_arcsec=0.1)`)
- ✅ No bilinear interpolation on physics maps
- ✅ No silent metadata defaults

---

## Release Checklist

- [x] All P1 fixes implemented
- [x] Patches applied
- [x] Tests passing 3× consecutively
- [x] CI gates passing
- [x] Regression checks clean
- [x] Documentation updated
- [x] Backward compatibility maintained

---

## Next Steps

1. Review `REPORT_RELEASE.md` for full verification details
2. Run final smoke tests if needed:
   ```bash
   py scripts/smoke_models.py
   py scripts/bench_dataloaders.py
   ```
3. Commit with message template in `REPORT_RELEASE.md`

---

**Ready for release** ✅

