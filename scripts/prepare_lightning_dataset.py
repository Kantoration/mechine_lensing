#!/usr/bin/env python3
"""
Dataset preparation script for Lightning AI integration.

This script converts local datasets to WebDataset format for cloud streaming
and uploads them to cloud storage (S3, GCS, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import fsspec
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lit_datamodule import create_webdataset_shards, upload_shards_to_cloud

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare dataset for Lightning AI")
    
    # Input/Output arguments
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of local dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for WebDataset shards")
    parser.add_argument("--cloud-url", type=str, default=None,
                        help="Cloud storage URL for upload (e.g., s3://bucket/path/)")
    
    # Sharding arguments
    parser.add_argument("--shard-size", type=int, default=1000,
                        help="Number of samples per shard")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Image size for compression")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality (1-100)")
    
    # Cloud storage arguments
    parser.add_argument("--storage-options", type=str, default=None,
                        help="Storage options as JSON string")
    parser.add_argument("--upload-only", action="store_true",
                        help="Only upload existing shards (skip creation)")
    
    # Validation arguments
    parser.add_argument("--validate", action="store_true",
                        help="Validate dataset after creation")
    
    args = parser.parse_args()
    
    # Validate inputs
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create WebDataset shards
        if not args.upload_only:
            logger.info("Creating WebDataset shards...")
            create_webdataset_shards(
                data_root=data_root,
                output_dir=output_dir,
                shard_size=args.shard_size,
                image_size=args.image_size,
                quality=args.quality
            )
            logger.info(f"Created shards in {output_dir}")
        
        # Upload to cloud storage
        if args.cloud_url:
            logger.info(f"Uploading shards to {args.cloud_url}...")
            
            # Parse storage options
            storage_options = {}
            if args.storage_options:
                import json
                storage_options = json.loads(args.storage_options)
            
            upload_shards_to_cloud(
                local_dir=output_dir,
                cloud_url=args.cloud_url,
                storage_options=storage_options
            )
            logger.info("Upload completed")
        
        # Validate dataset
        if args.validate:
            logger.info("Validating dataset...")
            validate_webdataset(output_dir)
            logger.info("Validation completed")
        
        logger.info("Dataset preparation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise


def validate_webdataset(shard_dir: Path) -> None:
    """Validate WebDataset shards."""
    import tarfile
    import io
    from PIL import Image
    
    shard_files = list(shard_dir.glob("*.tar"))
    if not shard_files:
        raise ValueError("No shard files found")
    
    logger.info(f"Validating {len(shard_files)} shard files...")
    
    total_samples = 0
    for shard_file in tqdm(shard_files, desc="Validating shards"):
        with tarfile.open(shard_file, "r") as tar:
            members = tar.getmembers()
            
            # Count samples (each sample has .jpg and .cls files)
            jpg_files = [m for m in members if m.name.endswith(".jpg")]
            cls_files = [m for m in members if m.name.endswith(".cls")]
            
            if len(jpg_files) != len(cls_files):
                raise ValueError(f"Mismatched files in {shard_file}")
            
            # Validate a few samples
            for i, jpg_member in enumerate(jpg_files[:5]):  # Check first 5 samples
                # Extract and validate image
                jpg_data = tar.extractfile(jpg_member).read()
                img = Image.open(io.BytesIO(jpg_data))
                
                if img.mode != "RGB":
                    raise ValueError(f"Invalid image mode in {shard_file}: {img.mode}")
                
                # Extract and validate label
                cls_member = cls_files[i]
                cls_data = tar.extractfile(cls_member).read().decode()
                label = int(cls_data)
                
                if label not in [0, 1]:
                    raise ValueError(f"Invalid label in {shard_file}: {label}")
            
            total_samples += len(jpg_files)
    
    logger.info(f"Validation passed: {total_samples} samples in {len(shard_files)} shards")


if __name__ == "__main__":
    exit(main())

