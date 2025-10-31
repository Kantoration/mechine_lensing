# Tiled Inference

Large FOVs can be processed via tiled inference with Hanning blending.

## API
- `predict_full(model_fn, image)` → dict (e.g., `{"kappa": [1,1,H,W]}`)
- `predict_tiled(model_fn, image, tile=128, overlap=32, key="kappa")` → stitched map

## Blending
- Uses a 2D Hanning window per tile for smooth stitching
- `overlap` should be ~1/4 tile to reduce seams

## Equivalence
- Expected \( \mathrm{MAE} < 10^{-3} \) between full and tiled κ for stable models

## See Also
- Source: `mlensing/gnn/inference_utils.py`
- Test: `tests/test_tiled_inference_equiv.py`
- Cluster dataset: `src/datasets/cluster_lensing.py`
