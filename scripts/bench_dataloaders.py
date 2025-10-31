#!/usr/bin/env python3
"""
Benchmark dataloaders for throughput, memory, and metadata enforcement.
"""

import sys
import time
import psutil
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from astropy.io import fits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.lens_fits_dataset import LensFITSDataset
from src.datasets.cluster_lensing import ClusterLensingDataset
from torch.utils.data import DataLoader


def create_dummy_fits_data(tmp_path, num_samples=100, image_shape=(64, 64), num_bands=3):
    """Create dummy FITS files for testing."""
    csv_path = tmp_path / "manifest.csv"
    
    records = []
    for i in range(num_samples):
        fits_path = tmp_path / f"sample_{i:04d}.fits"
        
        # Create multi-HDU FITS file
        hdul = fits.HDUList()
        
        # Primary HDU with header metadata
        primary = fits.PrimaryHDU()
        primary.header['PIXSCALE'] = 0.1  # arcsec/pixel
        hdul.append(primary)
        
        # Band HDUs
        for band_idx in range(num_bands):
            data = np.random.randn(*image_shape).astype(np.float32)
            hdu = fits.ImageHDU(data=data, name=f'BAND_{band_idx}')
            hdul.append(hdu)
        
        hdul.writeto(fits_path, overwrite=True)
        
        records.append({
            'filepath': str(fits_path),
            'pixel_scale_arcsec': 0.1,
            'pixel_scale_y_arcsec': 0.12,  # Anisotropic
            'sigma_crit': 1.5e8,
            'z_l': 0.3,
            'z_s': 1.5,
        })
    
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    
    return csv_path


def benchmark_fits_loader(tmp_path, num_workers=0):
    """Benchmark FITS loader throughput and metadata."""
    logger.info(f"\nBenchmarking FITS loader (num_workers={num_workers})...")
    
    csv_path = create_dummy_fits_data(tmp_path, num_samples=200)
    
    dataset = LensFITSDataset(
        csv_path=csv_path,
        band_hdus={'g': 1, 'r': 2, 'i': 3},
        require_sigma_crit=False  # For benchmark, don't require
    )
    
    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Warmup
    for i, batch in enumerate(loader):
        if i >= 2:
            break
    
    # Benchmark
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    start = time.time()
    samples_processed = 0
    metadata_issues = []
    
    for batch in loader:
        samples_processed += batch['image'].shape[0]
        
        # Check metadata integrity
        if 'meta' in batch:
            for meta in batch.get('meta', []):
                if meta is None:
                    metadata_issues.append("None meta")
                elif isinstance(meta, dict):
                    if 'dx' not in meta or meta['dx'] is None:
                        metadata_issues.append("Missing dx")
                    if 'dy' not in meta or meta['dy'] is None:
                        metadata_issues.append("Missing dy")
    
    elapsed = time.time() - start
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    samples_per_sec = samples_processed / elapsed
    mem_peak = mem_after - mem_before
    
    logger.info(f"  Samples/sec: {samples_per_sec:.2f}")
    logger.info(f"  Memory delta: {mem_peak:.2f} MB")
    logger.info(f"  Metadata issues: {len(metadata_issues)}")
    
    if metadata_issues:
        logger.warning(f"  Issues: {set(metadata_issues)}")
    
    return {
        'samples_per_sec': samples_per_sec,
        'memory_mb': mem_peak,
        'metadata_issues': len(metadata_issues),
        'num_workers': num_workers,
    }


def test_metadata_enforcement(tmp_path):
    """Test that metadata enforcement works."""
    logger.info("\nTesting metadata enforcement...")
    
    csv_path = tmp_path / "manifest.csv"
    
    # Test: missing pixel_scale_arcsec
    df = pd.DataFrame({'filepath': ['test.fits']})
    df.to_csv(csv_path, index=False)
    
    fits_path = tmp_path / "test.fits"
    hdu = fits.PrimaryHDU(data=np.random.randn(64, 64).astype(np.float32))
    hdu.header.pop('PIXSCALE', None)
    hdu.writeto(fits_path, overwrite=True)
    
    dataset = LensFITSDataset(csv_path=csv_path, band_hdus={'g': 0}, require_sigma_crit=False)
    
    try:
        _ = dataset[0]
        logger.error("  ✗ Should raise on missing pixel_scale_arcsec")
        return False
    except ValueError:
        logger.info("  ✓ Correctly raises on missing pixel_scale_arcsec")
    
    # Test: missing sigma_crit when required
    df['pixel_scale_arcsec'] = 0.1
    df.to_csv(csv_path, index=False)
    
    dataset_required = LensFITSDataset(csv_path=csv_path, band_hdus={'g': 0}, require_sigma_crit=True)
    
    try:
        _ = dataset_required[0]
        logger.error("  ✗ Should raise on missing sigma_crit when required")
        return False
    except ValueError:
        logger.info("  ✓ Correctly raises on missing sigma_crit when required")
    
    return True


def main():
    """Run dataloader benchmarks."""
    import tempfile
    
    logger.info("Dataloader Benchmarks")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Metadata enforcement test
        enforcement_ok = test_metadata_enforcement(tmp_path)
        
        # Throughput benchmarks
        results = []
        for num_workers in [0, 4, 8]:
            try:
                result = benchmark_fits_loader(tmp_path, num_workers=num_workers)
                results.append(result)
            except Exception as e:
                logger.error(f"  ✗ num_workers={num_workers} failed: {e}")
                results.append({'num_workers': num_workers, 'error': str(e)})
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        
        logger.info(f"Metadata enforcement: {'✓ PASS' if enforcement_ok else '✗ FAIL'}")
        
        for result in results:
            if 'error' in result:
                logger.info(f"num_workers={result['num_workers']}: ✗ {result['error']}")
            else:
                logger.info(f"num_workers={result['num_workers']}: "
                          f"{result['samples_per_sec']:.2f} samples/sec, "
                          f"{result['memory_mb']:.2f} MB, "
                          f"{result['metadata_issues']} metadata issues")
        
        return 0 if enforcement_ok and all('error' not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

