#!/usr/bin/env python3
r"""
generate_dataset.py
===================
Production-grade astronomical dataset generator following scientific computing best practices.

This refactored version addresses critical issues in scientific software development:
- Proper logging instead of print statements (Real Python logging guide)
- Type-safe configuration with validation (Effective Python Item 90)
- Atomic file operations to prevent corruption (Python Cookbook Recipe 5.18)
- Comprehensive metadata tracking for reproducibility (Ten Simple Rules for Reproducible Research)
- Unit testable architecture with dependency injection
- Structured error handling with context preservation

Author: Scientific Computing Team
License: MIT
Version: 2.0.0

References:
- Real Python: Python Logging Guide
- Effective Python (2nd Ed): Items 89-91 on Configuration and Validation
- Python Cookbook (3rd Ed): Recipe 5.18 on Atomic File Operations
- Ten Simple Rules for Reproducible Computational Research (PLOS Comp Bio)
- PEP 484: Type Hints
- PEP 526: Variable Annotations

Usage:
    python scripts/generate_dataset.py --config configs/comprehensive.yaml --out data --backend auto --log-level INFO
"""

from __future__ import annotations

# Standard library imports
import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Protocol

# Third-party imports
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

# Configure warnings to be logged instead of printed to stderr
# Best Practice: Centralized warning management through logging system
import warnings
warnings.filterwarnings('default')
logging.captureWarnings(True)

# ============================================================================
# CONFIGURATION SCHEMA AND VALIDATION
# ============================================================================
# Best Practice: Use @dataclass for type-safe, self-documenting configuration
# Why: Catches typos at runtime, provides IDE support, makes schema explicit
# Reference: Effective Python Item 37 (Use dataclasses for simple data containers)

@dataclass(frozen=True)  # frozen=True makes config immutable after creation
class GeneralConfig:
    """General dataset generation parameters with validation."""
    n_train: int = 1800
    n_test: int = 200
    image_size: int = 64
    seed: int = 42
    balance: float = 0.5
    backend: str = "synthetic"  # Backend to use: "synthetic", "deeplenstronomy", or "auto"
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.
        
        Why: Fail fast with clear error messages instead of silent corruption.
        Reference: Effective Python Item 90 (Consider static analysis via mypy)
        """
        if self.n_train < 1:
            raise ValueError(f"n_train must be positive, got {self.n_train}")
        if self.n_test < 0:
            raise ValueError(f"n_test must be non-negative, got {self.n_test}")
        if self.image_size < 8:
            raise ValueError(f"image_size too small for meaningful features, got {self.image_size}")
        if not (0.0 <= self.balance <= 1.0):
            raise ValueError(f"balance must be in [0,1], got {self.balance}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.backend not in ["synthetic", "deeplenstronomy", "auto"]:
            raise ValueError(f"backend must be 'synthetic', 'deeplenstronomy', or 'auto', got '{self.backend}'")


@dataclass(frozen=True)
class NoiseConfig:
    """Noise model parameters for realistic image simulation."""
    gaussian_sigma: float = 0.02
    poisson_strength: float = 0.0
    background_level: float = 0.01
    readout_noise: float = 5.0
    
    def __post_init__(self) -> None:
        if self.gaussian_sigma < 0:
            raise ValueError(f"gaussian_sigma must be non-negative, got {self.gaussian_sigma}")
        if self.poisson_strength < 0:
            raise ValueError(f"poisson_strength must be non-negative, got {self.poisson_strength}")


@dataclass(frozen=True)
class LensArcConfig:
    """Gravitational lensing arc simulation parameters."""
    # Arc parameters
    min_radius: float = 8.0
    max_radius: float = 20.0
    arc_width_min: float = 2.0
    arc_width_max: float = 4.0
    min_arcs: int = 1
    max_arcs: int = 3
    blur_sigma: float = 1.0
    brightness_min: float = 0.7
    brightness_max: float = 1.0
    asymmetry: float = 0.2
    
    # Background galaxy parameters (lens images contain galaxies + arcs)
    galaxy_sigma_min: float = 4.0
    galaxy_sigma_max: float = 8.0
    galaxy_brightness_min: float = 0.4
    galaxy_brightness_max: float = 0.7
    galaxy_ellipticity_min: float = 0.0
    galaxy_ellipticity_max: float = 0.4
    
    def __post_init__(self) -> None:
        if self.min_radius >= self.max_radius:
            raise ValueError("min_radius must be < max_radius")
        if self.arc_width_min >= self.arc_width_max:
            raise ValueError("arc_width_min must be < arc_width_max")
        if self.min_arcs > self.max_arcs:
            raise ValueError("min_arcs must be <= max_arcs")
        if self.galaxy_sigma_min >= self.galaxy_sigma_max:
            raise ValueError("galaxy_sigma_min must be < galaxy_sigma_max")
        if self.galaxy_brightness_min >= self.galaxy_brightness_max:
            raise ValueError("galaxy_brightness_min must be < galaxy_brightness_max")
        if self.galaxy_ellipticity_min >= self.galaxy_ellipticity_max:
            raise ValueError("galaxy_ellipticity_min must be < galaxy_ellipticity_max")


@dataclass(frozen=True)
class GalaxyBlobConfig:
    """Non-lens galaxy simulation parameters."""
    sigma_min: float = 2.0
    sigma_max: float = 6.0
    ellipticity_min: float = 0.0
    ellipticity_max: float = 0.6
    blur_sigma: float = 0.6
    brightness_min: float = 0.6
    brightness_max: float = 1.0
    sersic_index_min: float = 1.0
    sersic_index_max: float = 4.0
    
    # Multi-component galaxy parameters
    n_components_min: int = 1
    n_components_max: int = 2
    
    def __post_init__(self) -> None:
        if self.sigma_min >= self.sigma_max:
            raise ValueError("sigma_min must be < sigma_max")
        if not (0.0 <= self.ellipticity_min <= self.ellipticity_max <= 1.0):
            raise ValueError("ellipticity values must be in [0,1] with min <= max")
        if self.n_components_min < 1:
            raise ValueError("n_components_min must be >= 1")
        if self.n_components_min > self.n_components_max:
            raise ValueError("n_components_min must be <= n_components_max")


@dataclass(frozen=True)
class OutputConfig:
    """Output formatting and metadata options."""
    create_class_subdirs: bool = True
    create_split_subdirs: bool = True
    lens_prefix: str = "lens"
    nonlens_prefix: str = "nonlens"
    image_format: str = "PNG"
    image_quality: int = 95
    include_metadata: bool = False
    relative_paths: bool = True
    
    def __post_init__(self) -> None:
        valid_formats = {"PNG", "JPEG", "FITS"}
        if self.image_format not in valid_formats:
            raise ValueError(f"image_format must be one of {valid_formats}")
        if not (1 <= self.image_quality <= 100):
            raise ValueError("image_quality must be in [1,100]")


@dataclass(frozen=True)
class ValidationConfig:
    """Quality control and validation parameters."""
    check_image_integrity: bool = True  # Validate generated images
    sample_fraction: float = 0.1        # Fraction of images to validate
    min_brightness: float = 0.01        # Minimum acceptable brightness
    max_brightness: float = 1.0         # Maximum acceptable brightness


@dataclass(frozen=True)
class DebugConfig:
    """Debug and logging configuration."""
    save_sample_images: bool = False     # Save sample images for inspection
    log_generation_stats: bool = True    # Log detailed generation statistics
    verbose_validation: bool = False     # Detailed validation logging


@dataclass(frozen=True)
class DatasetConfig:
    """Complete type-safe configuration for dataset generation.
    
    Why dataclass: Self-documenting, IDE-friendly, validation at construction.
    Why frozen: Immutable configuration prevents accidental modification.
    """
    general: GeneralConfig = field(default_factory=GeneralConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    lens_arcs: LensArcConfig = field(default_factory=LensArcConfig)
    galaxy_blob: GalaxyBlobConfig = field(default_factory=GalaxyBlobConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


# ============================================================================
# LOGGING SETUP
# ============================================================================
# Best Practice: Structured logging with appropriate levels
# Why: Enables filtering, redirection, integration with monitoring systems
# Reference: Real Python - Python Logging: A Starters Guide

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure structured logging for scientific reproducibility.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger instance
        
    Why this approach:
    - Structured logs are parseable and filterable
    - Multiple handlers allow console + file output
    - Timestamps enable performance analysis
    - Process info helps with parallel execution debugging
    """
    # Clear any existing handlers to avoid duplication
    logger = logging.getLogger('dataset_generator')
    logger.handlers.clear()
    
    # Set level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logger.setLevel(numeric_level)
    
    # Create formatter with scientific metadata
    # Include timestamp, level, function name, and message
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler for persistent logs
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


# Global logger instance - initialized in main()
logger: logging.Logger = logging.getLogger('dataset_generator')


# ============================================================================
# ATOMIC FILE OPERATIONS
# ============================================================================
# Best Practice: Atomic writes prevent corruption from interruptions
# Why: Scientific data integrity requires all-or-nothing file operations
# Reference: Python Cookbook Recipe 5.18 - Making a Directory of Files

@contextmanager
def atomic_write(target_path: Path, mode: str = 'w', **kwargs) -> Iterator[Any]:
    """Context manager for atomic file writes.
    
    Writes to temporary file first, then renames to target atomically.
    Prevents partial/corrupt files if process is interrupted.
    
    Args:
        target_path: Final destination path
        mode: File open mode
        **kwargs: Additional arguments for open()
        
    Yields:
        File handle for writing
        
    Example:
        with atomic_write(Path("data.csv")) as f:
            df.to_csv(f, index=False)
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in same directory as target
    # This ensures atomic rename works (same filesystem)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f'.tmp_{target_path.name}_',
        suffix='.tmp'
    )
    
    try:
        with open(temp_fd, mode, **kwargs) as temp_file:
            yield temp_file
            temp_file.flush()  # Ensure data is written
            os.fsync(temp_fd)  # Force OS to write to disk
        
        # Atomic rename - either succeeds completely or fails completely
        Path(temp_path).rename(target_path)
        logger.debug(f"Atomically wrote {target_path}")
        
    except Exception:
        # Clean up temporary file on any error
        try:
            Path(temp_path).unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_save_image(image: Image.Image, path: Path, **save_kwargs) -> None:
    """Atomically save PIL Image to prevent corruption.
    
    Args:
        image: PIL Image to save
        path: Destination path
        **save_kwargs: Additional arguments for Image.save()
    """
    with atomic_write(path, mode='wb') as f:
        image.save(f, **save_kwargs)


# ============================================================================
# METADATA AND TRACEABILITY
# ============================================================================
# Best Practice: Track all parameters for reproducibility
# Why: Scientific reproducibility requires complete parameter logging
# Reference: Ten Simple Rules for Reproducible Computational Research

@dataclass
class ImageMetadata:
    """Comprehensive metadata for generated images.
    
    Tracks all parameters used in image generation for full reproducibility.
    This enables post-hoc analysis and debugging of dataset quality issues.
    """
    filename: str
    label: int  # 0=non-lens, 1=lens
    split: str  # 'train' or 'test'
    generation_time: float
    random_seed: int
    image_size: int
    
    # Physics/simulation parameters
    brightness: float
    noise_level: float
    
    # Lens-specific parameters (None for non-lens images)
    n_arcs: Optional[int] = None
    arc_radii: Optional[List[float]] = None
    arc_widths: Optional[List[float]] = None
    arc_angles: Optional[List[float]] = None
    
    # Galaxy-specific parameters (None for lens images)
    galaxy_sigma: Optional[float] = None
    galaxy_ellipticity: Optional[float] = None
    galaxy_angle: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return asdict(self)


class MetadataTracker:
    """Centralized metadata collection for reproducibility.
    
    Collects all generation parameters and provides structured export.
    Essential for scientific reproducibility and dataset analysis.
    """
    
    def __init__(self):
        self.metadata: List[ImageMetadata] = []
        self.generation_start_time = time.time()
        self.config_snapshot: Optional[Dict[str, Any]] = None
    
    def set_config_snapshot(self, config: DatasetConfig) -> None:
        """Store complete configuration for reproducibility."""
        self.config_snapshot = asdict(config)
    
    def add_image_metadata(self, metadata: ImageMetadata) -> None:
        """Add metadata for a single generated image."""
        self.metadata.append(metadata)
    
    def export_to_csv(self, path: Path) -> None:
        """Export metadata to CSV with atomic write."""
        if not self.metadata:
            logger.warning("No metadata to export")
            return
            
        df = pd.DataFrame([m.to_dict() for m in self.metadata])
        
        with atomic_write(path, mode='w') as f:
            df.to_csv(f, index=False)
        
        logger.info(f"Exported metadata for {len(self.metadata)} images to {path}")
    
    def export_config_snapshot(self, path: Path) -> None:
        """Export complete configuration as JSON."""
        if self.config_snapshot is None:
            logger.warning("No configuration snapshot to export")
            return
            
        export_data = {
            'config': self.config_snapshot,
            'generation_metadata': {
                'start_time': self.generation_start_time,
                'total_images': len(self.metadata),
                'python_version': sys.version,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__,
            }
        }
        
        with atomic_write(path, mode='w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported configuration snapshot to {path}")


# ============================================================================
# CONFIGURATION LOADING AND VALIDATION
# ============================================================================

def load_and_validate_config(config_path: Path) -> DatasetConfig:
    """Load YAML configuration with comprehensive validation.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated DatasetConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
        ValueError: If configuration values are invalid
        
    Why this approach:
    - Explicit validation catches errors early
    - Type-safe configuration prevents runtime errors
    - Clear error messages aid debugging
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")
    
    # Extract and validate each section
    try:
        config = DatasetConfig(
            general=GeneralConfig(**raw_config.get('General', {})),
            noise=NoiseConfig(**raw_config.get('Noise', {})),
            lens_arcs=LensArcConfig(**raw_config.get('LensArcs', {})),
            galaxy_blob=GalaxyBlobConfig(**raw_config.get('GalaxyBlob', {})),
            output=OutputConfig(**raw_config.get('Output', {})),
            validation=ValidationConfig(**raw_config.get('Validation', {})),
            debug=DebugConfig(**raw_config.get('Debug', {}))
        )
        
        logger.info("Configuration validation successful")
        logger.debug(f"Config: {config}")
        
        return config
        
    except TypeError as e:
        raise ValueError(f"Configuration validation failed: {e}")


# ============================================================================
# SCIENTIFIC IMAGE GENERATION
# ============================================================================
# Best Practice: Separate concerns with single-responsibility classes
# Why: Testable, maintainable, and extensible architecture

class SyntheticImageGenerator:
    """Physics-based synthetic astronomical image generator.
    
    Generates scientifically plausible gravitational lens and galaxy images
    with full parameter tracking for reproducibility.
    """
    
    def __init__(self, config: DatasetConfig, rng: np.random.Generator, metadata_tracker: MetadataTracker):
        """Initialize generator with validated configuration.
        
        Args:
            config: Validated configuration
            rng: Seeded random number generator for reproducibility
            metadata_tracker: Centralized metadata collection
        """
        self.config = config
        self.rng = rng
        self.metadata_tracker = metadata_tracker
        self.image_size = config.general.image_size
        
        logger.info(f"Initialized synthetic generator (image_size={self.image_size})")
    
    def create_lens_arc_image(self, image_id: str, split: str) -> Tuple[np.ndarray, ImageMetadata]:
        """Generate realistic gravitational lensing image: galaxy + subtle arcs.
        
        Key improvement: Lens images now contain a background galaxy PLUS faint arcs,
        making them much more similar to non-lens images and realistic.
        
        Returns:
            Tuple of (image_array, metadata)
        """
        start_time = time.time()
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        center = self.image_size // 2
        
        arc_config = self.config.lens_arcs
        
        # STEP 1: Create background galaxy (similar to non-lens images)
        # This makes lens images more realistic - they contain galaxies too!
        galaxy_sigma = self.rng.uniform(
            arc_config.galaxy_sigma_min, 
            arc_config.galaxy_sigma_max
        )
        galaxy_ellipticity = self.rng.uniform(
            arc_config.galaxy_ellipticity_min,
            arc_config.galaxy_ellipticity_max
        )
        galaxy_angle = self.rng.uniform(0, np.pi)
        galaxy_brightness = self.rng.uniform(
            arc_config.galaxy_brightness_min,
            arc_config.galaxy_brightness_max
        )
        
        # Create galaxy using same method as non-lens images
        y, x = np.ogrid[:self.image_size, :self.image_size]
        x = x - center
        y = y - center
        
        cos_a, sin_a = np.cos(galaxy_angle), np.sin(galaxy_angle)
        x_rot = cos_a * x - sin_a * y
        y_rot = sin_a * x + cos_a * y
        
        a = galaxy_sigma
        b = galaxy_sigma * (1 - galaxy_ellipticity)
        
        galaxy = np.exp(-0.5 * ((x_rot/a)**2 + (y_rot/b)**2))
        img += galaxy * galaxy_brightness
        
        # STEP 2: Add subtle lensing arcs (the key difference)
        n_arcs = self.rng.integers(arc_config.min_arcs, arc_config.max_arcs + 1)
        
        # Track parameters for reproducibility
        arc_radii = []
        arc_widths = []
        arc_angles = []
        
        for _ in range(n_arcs):
            radius = self.rng.uniform(arc_config.min_radius, arc_config.max_radius)
            width = self.rng.uniform(arc_config.arc_width_min, arc_config.arc_width_max)
            start_angle = self.rng.uniform(0, 2 * np.pi)
            arc_length = self.rng.uniform(np.pi/4, np.pi/2)  # Shorter arcs (more realistic)
            
            arc_radii.append(radius)
            arc_widths.append(width)
            arc_angles.append(start_angle)
            
            # Create arc as a subtle addition to the galaxy
            brightness = self.rng.uniform(arc_config.brightness_min, arc_config.brightness_max)
            
            # Generate arc points
            n_segments = max(8, int(arc_length * radius / 3))
            for i in range(n_segments):
                angle = start_angle + (i / n_segments) * arc_length
                
                # Arc center line
                arc_x = center + radius * np.cos(angle)
                arc_y = center + radius * np.sin(angle)
                
                # Add arc as small Gaussian blobs (more realistic than lines)
                y_arc, x_arc = np.ogrid[:self.image_size, :self.image_size]
                arc_gaussian = np.exp(-0.5 * (((x_arc - arc_x)/width)**2 + ((y_arc - arc_y)/width)**2))
                img += arc_gaussian * brightness * 0.3  # Subtle addition
        
        # Apply realistic blur (PSF)
        if arc_config.blur_sigma > 0:
            img = gaussian_filter(img, sigma=arc_config.blur_sigma)
        
        # Add realistic noise
        img, noise_level = self._add_noise(img)
        
        # Normalize to prevent oversaturation
        img = np.clip(img, 0, 1)
        
        # Create comprehensive metadata
        metadata = ImageMetadata(
            filename=image_id,
            label=1,  # Lens class
            split=split,
            generation_time=time.time() - start_time,
            random_seed=self.config.general.seed,
            image_size=self.image_size,
            brightness=galaxy_brightness,
            noise_level=noise_level,
            n_arcs=n_arcs,
            arc_radii=arc_radii,
            arc_widths=arc_widths,
            arc_angles=arc_angles,
            galaxy_sigma=galaxy_sigma,
            galaxy_ellipticity=galaxy_ellipticity,
            galaxy_angle=galaxy_angle
        )
        
        return img, metadata
    
    def create_galaxy_blob_image(self, image_id: str, split: str) -> Tuple[np.ndarray, ImageMetadata]:
        """Generate realistic non-lens galaxy image with complexity.
        
        Key improvement: Add multiple components and realistic features to make
        non-lens galaxies more complex and similar to lens galaxy backgrounds.
        """
        start_time = time.time()
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        center = self.image_size // 2
        
        blob_config = self.config.galaxy_blob
        
        # Generate multiple galaxy components (realistic galaxies are complex)
        n_components = self.rng.integers(
            blob_config.n_components_min,
            blob_config.n_components_max + 1
        )
        
        total_brightness = 0
        component_params = []
        
        for comp_i in range(n_components):
            # Each component can have different properties
            sigma = self.rng.uniform(blob_config.sigma_min, blob_config.sigma_max)
            ellipticity = self.rng.uniform(blob_config.ellipticity_min, blob_config.ellipticity_max)
            angle = self.rng.uniform(0, np.pi)
            
            # Distribute brightness among components
            if comp_i == 0:
                brightness = self.rng.uniform(blob_config.brightness_min, blob_config.brightness_max)
            else:
                # Secondary components are fainter
                brightness = self.rng.uniform(0.2, 0.5) * brightness
            
            # Small offset for secondary components (realistic galaxy structure)
            offset_x = self.rng.uniform(-2, 2) if comp_i > 0 else 0
            offset_y = self.rng.uniform(-2, 2) if comp_i > 0 else 0
            
            # Create coordinate grids with offset
            y, x = np.ogrid[:self.image_size, :self.image_size]
            x = x - (center + offset_x)
            y = y - (center + offset_y)
            
            # Apply rotation and ellipticity
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = cos_a * x - sin_a * y
            y_rot = sin_a * x + cos_a * y
            
            a = sigma
            b = sigma * (1 - ellipticity)
            
            gaussian = np.exp(-0.5 * ((x_rot/a)**2 + (y_rot/b)**2))
            img += gaussian * brightness
            
            total_brightness += brightness
            component_params.append({
                'sigma': sigma, 'ellipticity': ellipticity, 'angle': angle,
                'brightness': brightness, 'offset_x': offset_x, 'offset_y': offset_y
            })
        
        # Apply realistic blur (PSF) - same as lens images
        if blob_config.blur_sigma > 0:
            img = gaussian_filter(img, sigma=blob_config.blur_sigma)
        
        # Add realistic noise - same as lens images
        img, noise_level = self._add_noise(img)
        
        # Normalize to prevent oversaturation
        img = np.clip(img, 0, 1)
        
        # Create metadata with component information
        metadata = ImageMetadata(
            filename=image_id,
            label=0,  # Non-lens class
            split=split,
            generation_time=time.time() - start_time,
            random_seed=self.config.general.seed,
            image_size=self.image_size,
            brightness=total_brightness,
            noise_level=noise_level,
            galaxy_sigma=component_params[0]['sigma'],  # Primary component
            galaxy_ellipticity=component_params[0]['ellipticity'],
            galaxy_angle=component_params[0]['angle']
        )
        
        return img, metadata
    
    def _add_noise(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Add realistic noise with level tracking."""
        noise_config = self.config.noise
        total_noise = 0.0
        
        # Gaussian noise
        if noise_config.gaussian_sigma > 0:
            noise = self.rng.normal(0, noise_config.gaussian_sigma, img.shape)
            img = img + noise
            total_noise += noise_config.gaussian_sigma
        
        # Poisson noise
        if noise_config.poisson_strength > 0:
            scaled = img * noise_config.poisson_strength * 1000
            scaled = np.maximum(scaled, 0)
            noisy_scaled = self.rng.poisson(scaled).astype(np.float32)
            img = noisy_scaled / (noise_config.poisson_strength * 1000)
            total_noise += noise_config.poisson_strength
        
        return np.clip(img, 0, 1), total_noise
    
    def generate_dataset(self, output_dir: Path) -> None:
        """Generate complete dataset with atomic operations and metadata tracking."""
        logger.info("Starting synthetic dataset generation")
        
        general = self.config.general
        output = self.config.output
        
        # Create directory structure
        if output.create_split_subdirs and output.create_class_subdirs:
            dirs_to_create = [
                output_dir / "train" / "lens",
                output_dir / "train" / "nonlens",
                output_dir / "test" / "lens",
                output_dir / "test" / "nonlens"
            ]
        else:
            dirs_to_create = [output_dir]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Generate training set
        self._generate_split(
            n_images=general.n_train,
            split="train",
            output_dir=output_dir,
            balance=general.balance
        )
        
        # Generate test set
        self._generate_split(
            n_images=general.n_test,
            split="test", 
            output_dir=output_dir,
            balance=general.balance
        )
        
        logger.info(f"Generated {general.n_train + general.n_test} total images")
    
    def _generate_split(self, n_images: int, split: str, output_dir: Path, balance: float) -> None:
        """Generate train or test split with progress logging."""
        n_lens = int(n_images * balance)
        n_nonlens = n_images - n_lens
        
        logger.info(f"Generating {split} split: {n_lens} lens + {n_nonlens} non-lens images")
        
        output = self.config.output
        
        # Generate lens images
        for i in range(n_lens):
            image_id = f"{output.lens_prefix}_{split}_{i:04d}"
            
            # Generate image and metadata
            img_array, metadata = self.create_lens_arc_image(image_id, split)
            
            # Determine output path
            if output.create_split_subdirs and output.create_class_subdirs:
                img_path = output_dir / split / "lens" / f"{image_id}.{output.image_format.lower()}"
            else:
                img_path = output_dir / f"{image_id}.{output.image_format.lower()}"
            
            # Save image atomically
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            atomic_save_image(img_pil, img_path, format=output.image_format, quality=output.image_quality)
            
            # Track metadata
            metadata.filename = str(img_path.relative_to(output_dir)) if output.relative_paths else str(img_path)
            self.metadata_tracker.add_image_metadata(metadata)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Generated {i + 1}/{n_lens} lens images for {split}")
        
        # Generate non-lens images
        for i in range(n_nonlens):
            image_id = f"{output.nonlens_prefix}_{split}_{i:04d}"
            
            img_array, metadata = self.create_galaxy_blob_image(image_id, split)
            
            if output.create_split_subdirs and output.create_class_subdirs:
                img_path = output_dir / split / "nonlens" / f"{image_id}.{output.image_format.lower()}"
            else:
                img_path = output_dir / f"{image_id}.{output.image_format.lower()}"
            
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            atomic_save_image(img_pil, img_path, format=output.image_format, quality=output.image_quality)
            
            metadata.filename = str(img_path.relative_to(output_dir)) if output.relative_paths else str(img_path)
            self.metadata_tracker.add_image_metadata(metadata)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Generated {i + 1}/{n_nonlens} non-lens images for {split}")


# ============================================================================
# DATASET PIPELINE AND VALIDATION
# ============================================================================

def infer_label_from_path(path: Path) -> int:
    """Infer class label from file path structure.
    
    Returns:
        1 for lens images, 0 for non-lens images
    """
    path_str = str(path).lower()
    
    # Check for explicit non-lens indicators first
    if any(keyword in path_str for keyword in ["nonlens", "non-lens", "negative"]):
        return 0
    
    # Check for lens indicators
    if any(keyword in path_str for keyword in ["lens", "lensed", "einstein"]):
        return 1
    
    # Default to non-lens if ambiguous
    logger.warning(f"Ambiguous path for labeling: {path}, defaulting to non-lens")
    return 0


def create_csv_files(output_dir: Path, config: DatasetConfig, metadata_tracker: MetadataTracker) -> None:
    """Create train/test CSV files with optional metadata.
    
    Uses atomic writes to prevent corruption during CSV creation.
    """
    logger.info("Creating CSV label files")
    
    # Find all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.fits'}
    all_images = []
    for ext in image_extensions:
        all_images.extend(output_dir.rglob(f"*{ext}"))
    
    if not all_images:
        raise RuntimeError(f"No images found in {output_dir}")
    
    # Split by directory structure
    train_images = [p for p in all_images if '/train/' in str(p) or '\\train\\' in str(p)]
    test_images = [p for p in all_images if '/test/' in str(p) or '\\test\\' in str(p)]
    
    # If no clear split, put everything in train
    if not train_images and not test_images:
        logger.warning("No train/test split detected, putting all images in train.csv")
        train_images = all_images
        test_images = []
    
    # Create base CSV data
    def create_csv_data(images: List[Path]) -> List[Dict[str, Any]]:
        records = []
        for img_path in images:
            relative_path = img_path.relative_to(output_dir) if config.output.relative_paths else img_path
            record = {
                'filepath': str(relative_path).replace('\\', '/'),  # Consistent path separators
                'label': infer_label_from_path(img_path)
            }
            records.append(record)
        return records
    
    # Create train.csv
    train_records = create_csv_data(train_images)
    train_df = pd.DataFrame(train_records)
    
    with atomic_write(output_dir / "train.csv") as f:
        train_df.to_csv(f, index=False)
    
    logger.info(f"Created train.csv with {len(train_records)} images")
    
    # Create test.csv
    test_records = create_csv_data(test_images)
    test_df = pd.DataFrame(test_records)
    
    with atomic_write(output_dir / "test.csv") as f:
        test_df.to_csv(f, index=False)
    
    logger.info(f"Created test.csv with {len(test_records)} images")
    
    # Export metadata if requested
    if config.output.include_metadata and metadata_tracker.metadata:
        metadata_tracker.export_to_csv(output_dir / "metadata.csv")
        metadata_tracker.export_config_snapshot(output_dir / "config_snapshot.json")


def validate_dataset(output_dir: Path, config: DatasetConfig) -> None:
    """Comprehensive dataset validation with quality checks."""
    if not config.validation.enable_checks:
        logger.info("Validation disabled, skipping checks")
        return
    
    logger.info("Validating generated dataset")
    
    # Check required files exist
    required_files = [output_dir / "train.csv", output_dir / "test.csv"]
    for file_path in required_files:
        if not file_path.exists():
            raise RuntimeError(f"Required file missing: {file_path}")
    
    # Load and validate CSV files
    train_df = pd.read_csv(output_dir / "train.csv")
    test_df = pd.read_csv(output_dir / "test.csv")
    
    if len(train_df) == 0:
        raise RuntimeError("train.csv is empty")
    
    # Validate image accessibility and quality
    sample_images = list(train_df['filepath'].head(5)) + list(test_df['filepath'].head(2))
    
    for relative_path in sample_images:
        img_path = output_dir / relative_path
        
        if not img_path.exists():
            raise RuntimeError(f"Image file missing: {img_path}")
        
        try:
            with Image.open(img_path) as img:
                if config.validation.check_image_stats:
                    img_array = np.array(img)
                    brightness = np.mean(img_array) / 255.0
                    
                    if not (config.validation.min_brightness <= brightness <= config.validation.max_brightness):
                        logger.warning(f"Image {img_path} brightness {brightness:.3f} outside expected range")
                        
        except Exception as e:
            raise RuntimeError(f"Cannot open image {img_path}: {e}")
    
    # Print summary statistics
    train_lens = len(train_df[train_df['label'] == 1])
    train_nonlens = len(train_df[train_df['label'] == 0])
    test_lens = len(test_df[test_df['label'] == 1]) if len(test_df) > 0 else 0
    test_nonlens = len(test_df[test_df['label'] == 0]) if len(test_df) > 0 else 0
    
    logger.info(f"Dataset validation successful:")
    logger.info(f"  Training: {train_lens} lens, {train_nonlens} non-lens")
    logger.info(f"  Test: {test_lens} lens, {test_nonlens} non-lens")
    logger.info(f"  Total: {len(train_df) + len(test_df)} images")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """Main entry point with comprehensive error handling and logging."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Scientific astronomical dataset generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with INFO logging
  python scripts/generate_dataset.py --config configs/comprehensive.yaml
  
  # Debug mode with file logging
  python scripts/generate_dataset.py --log-level DEBUG --log-file logs/generation.log
  
  # Production run with validation
  python scripts/generate_dataset.py --validate --backend synthetic
        """
    )
    
    parser.add_argument("--config", type=Path, default="configs/comprehensive.yaml",
                        help="YAML configuration file")
    parser.add_argument("--out", type=Path, default="data",
                        help="Output directory")
    parser.add_argument("--backend", choices=["synthetic"], default="synthetic",
                        help="Generation backend (only synthetic implemented)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=Path, help="Optional log file")
    parser.add_argument("--validate", action="store_true",
                        help="Run comprehensive validation")
    
    args = parser.parse_args()
    
    # Initialize logging first
    global logger
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("=" * 60)
    logger.info("SCIENTIFIC ASTRONOMICAL DATASET GENERATOR")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.out}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Validation: {args.validate}")
    
    try:
        # Load and validate configuration
        config = load_and_validate_config(args.config)
        
        # Initialize metadata tracking
        metadata_tracker = MetadataTracker()
        metadata_tracker.set_config_snapshot(config)
        
        # Initialize random number generator with explicit seed
        rng = np.random.Generator(np.random.PCG64(config.general.seed))
        logger.info(f"Initialized RNG with seed {config.general.seed}")
        
        # Create output directory
        args.out.mkdir(parents=True, exist_ok=True)
        
        # Generate dataset
        if args.backend == "synthetic":
            generator = SyntheticImageGenerator(config, rng, metadata_tracker)
            generator.generate_dataset(args.out)
        else:
            raise ValueError(f"Backend {args.backend} not implemented")
        
        # Create CSV files
        create_csv_files(args.out, config, metadata_tracker)
        
        # Optional validation
        if args.validate:
            validate_dataset(args.out, config)
        
        logger.info("Dataset generation completed successfully!")
        logger.info(f"Output: {args.out}")
        logger.info(f"Files: train.csv, test.csv" + 
                   (", metadata.csv, config_snapshot.json" if config.output.include_metadata else ""))
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
