#!/usr/bin/env python3
"""
Unit tests for scientific dataset generator.

Demonstrates testing best practices for scientific computing:
- Isolated, reproducible tests
- Property-based testing for numerical code
- Mocking external dependencies
- Testing error conditions explicitly

Reference: Python Testing with pytest (Brian Okken)
"""

import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import yaml
import pandas as pd

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from make_dataset_scientific import (
    DatasetConfig, GeneralConfig, NoiseConfig, LensArcConfig,
    load_and_validate_config, SyntheticImageGenerator, MetadataTracker,
    atomic_write, setup_logging
)


class TestConfigValidation:
    """Test configuration loading and validation.
    
    Why test config validation:
    - Prevents silent failures from typos
    - Documents expected parameter ranges
    - Ensures error messages are helpful
    """
    
    def test_valid_config_loads_successfully(self):
        """Test that a valid configuration loads without errors."""
        config = DatasetConfig()  # Use defaults
        assert config.general.n_train == 1800
        assert config.general.image_size == 64
        assert 0.0 <= config.general.balance <= 1.0
    
    def test_invalid_image_size_raises_error(self):
        """Test that invalid image size is caught early."""
        with pytest.raises(ValueError, match="image_size too small"):
            GeneralConfig(image_size=4)  # Too small for meaningful features
    
    def test_invalid_balance_raises_error(self):
        """Test that balance parameter is properly validated."""
        with pytest.raises(ValueError, match="balance must be in"):
            GeneralConfig(balance=1.5)  # > 1.0
        
        with pytest.raises(ValueError, match="balance must be in"):
            GeneralConfig(balance=-0.1)  # < 0.0
    
    def test_negative_training_samples_raises_error(self):
        """Test that negative sample counts are rejected."""
        with pytest.raises(ValueError, match="n_train must be positive"):
            GeneralConfig(n_train=-10)
    
    def test_invalid_arc_radius_range_raises_error(self):
        """Test that arc radius validation works."""
        with pytest.raises(ValueError, match="min_radius must be < max_radius"):
            LensArcConfig(min_radius=20, max_radius=10)  # Inverted range
    
    def test_config_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        # Create temporary YAML file
        config_data = {
            'General': {
                'n_train': 100,
                'n_test': 20,
                'image_size': 32,
                'seed': 123
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = Path(f.name)
        
        try:
            config = load_and_validate_config(yaml_path)
            assert config.general.n_train == 100
            assert config.general.n_test == 20
            assert config.general.image_size == 32
            assert config.general.seed == 123
        finally:
            yaml_path.unlink()
    
    def test_malformed_yaml_raises_error(self):
        """Test that malformed YAML files are handled gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            yaml_path = Path(f.name)
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_and_validate_config(yaml_path)
        finally:
            yaml_path.unlink()


class TestAtomicOperations:
    """Test atomic file operations for data integrity.
    
    Why test atomic operations:
    - Critical for preventing data corruption
    - Complex error handling needs verification
    - File system edge cases must be handled
    """
    
    def test_atomic_write_success(self):
        """Test successful atomic write operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "test_file.txt"
            test_content = "test data"
            
            with atomic_write(target_path) as f:
                f.write(test_content)
            
            # File should exist and contain correct data
            assert target_path.exists()
            assert target_path.read_text() == test_content
    
    def test_atomic_write_failure_cleanup(self):
        """Test that temporary files are cleaned up on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "test_file.txt"
            
            try:
                with atomic_write(target_path) as f:
                    f.write("partial data")
                    raise RuntimeError("Simulated failure")
            except RuntimeError:
                pass
            
            # Target file should not exist
            assert not target_path.exists()
            
            # No temporary files should remain
            temp_files = list(Path(tmpdir).glob("*.tmp"))
            assert len(temp_files) == 0
    
    def test_atomic_write_creates_parent_directories(self):
        """Test that parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "subdir" / "nested" / "file.txt"
            
            with atomic_write(target_path) as f:
                f.write("test")
            
            assert target_path.exists()
            assert target_path.read_text() == "test"


class TestSyntheticImageGeneration:
    """Test synthetic image generation with scientific validation.
    
    Why test image generation:
    - Numerical algorithms need property-based testing
    - Output quality affects downstream ML performance
    - Parameter ranges must be validated
    """
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return DatasetConfig(
            general=GeneralConfig(n_train=10, n_test=5, image_size=32, seed=42)
        )
    
    @pytest.fixture
    def rng(self):
        """Provide seeded random number generator."""
        return np.random.Generator(np.random.PCG64(42))
    
    @pytest.fixture
    def metadata_tracker(self):
        """Provide metadata tracker."""
        return MetadataTracker()
    
    def test_lens_image_generation_reproducible(self, config, rng, metadata_tracker):
        """Test that lens image generation is reproducible."""
        generator = SyntheticImageGenerator(config, rng, metadata_tracker)
        
        # Generate same image twice
        img1, meta1 = generator.create_lens_arc_image("test_001", "train")
        
        # Reset RNG to same state
        rng = np.random.Generator(np.random.PCG64(42))
        generator.rng = rng
        
        img2, meta2 = generator.create_lens_arc_image("test_001", "train")
        
        # Images should be identical (within floating point precision)
        np.testing.assert_allclose(img1, img2, rtol=1e-10)
        
        # Metadata should be consistent
        assert meta1.n_arcs == meta2.n_arcs
        assert len(meta1.arc_radii) == len(meta2.arc_radii)
    
    def test_lens_image_properties(self, config, rng, metadata_tracker):
        """Test that generated lens images have expected properties."""
        generator = SyntheticImageGenerator(config, rng, metadata_tracker)
        img, metadata = generator.create_lens_arc_image("test_lens", "train")
        
        # Image should be correct size
        assert img.shape == (32, 32)
        
        # Image values should be in valid range
        assert np.all(img >= 0.0)
        assert np.all(img <= 1.0)
        
        # Image should not be completely empty or saturated
        assert np.mean(img) > 0.01  # Some signal present
        assert np.mean(img) < 0.99  # Not saturated
        
        # Metadata should be complete
        assert metadata.label == 1  # Lens class
        assert metadata.n_arcs is not None
        assert metadata.arc_radii is not None
        assert len(metadata.arc_radii) == metadata.n_arcs
    
    def test_galaxy_image_properties(self, config, rng, metadata_tracker):
        """Test that generated galaxy images have expected properties."""
        generator = SyntheticImageGenerator(config, rng, metadata_tracker)
        img, metadata = generator.create_galaxy_blob_image("test_galaxy", "train")
        
        # Basic image properties
        assert img.shape == (32, 32)
        assert np.all(img >= 0.0)
        assert np.all(img <= 1.0)
        
        # Galaxy-specific metadata
        assert metadata.label == 0  # Non-lens class
        assert metadata.galaxy_sigma is not None
        assert metadata.galaxy_ellipticity is not None
        assert 0.0 <= metadata.galaxy_ellipticity <= 1.0
    
    def test_noise_addition_affects_images(self, config, rng, metadata_tracker):
        """Test that noise addition actually changes images."""
        # Create config with significant noise
        noisy_config = DatasetConfig(
            general=config.general,
            noise=NoiseConfig(gaussian_sigma=0.1)  # High noise
        )
        
        generator = SyntheticImageGenerator(noisy_config, rng, metadata_tracker)
        
        # Generate image with noise
        img_noisy, _ = generator.create_lens_arc_image("test_noisy", "train")
        
        # Generate same image without noise
        clean_config = DatasetConfig(
            general=config.general,
            noise=NoiseConfig(gaussian_sigma=0.0)  # No noise
        )
        
        # Reset RNG
        rng = np.random.Generator(np.random.PCG64(42))
        clean_generator = SyntheticImageGenerator(clean_config, rng, metadata_tracker)
        img_clean, _ = clean_generator.create_lens_arc_image("test_clean", "train")
        
        # Images should be different due to noise
        assert not np.allclose(img_noisy, img_clean, atol=1e-3)
        
        # But structure should be similar (correlation should be high)
        correlation = np.corrcoef(img_noisy.flatten(), img_clean.flatten())[0, 1]
        assert correlation > 0.8  # Strong correlation despite noise
    
    def test_parameter_ranges_respected(self, config, rng, metadata_tracker):
        """Test that generation parameters stay within configured ranges."""
        generator = SyntheticImageGenerator(config, rng, metadata_tracker)
        
        # Generate multiple images to test parameter distributions
        for _ in range(10):
            _, metadata = generator.create_lens_arc_image(f"test_{_}", "train")
            
            # Check arc parameters are in range
            arc_config = config.lens_arcs
            assert arc_config.min_arcs <= metadata.n_arcs <= arc_config.max_arcs
            
            for radius in metadata.arc_radii:
                assert arc_config.min_radius <= radius <= arc_config.max_radius
            
            for width in metadata.arc_widths:
                assert arc_config.arc_width_min <= width <= arc_config.arc_width_max


class TestMetadataTracking:
    """Test metadata collection for reproducibility.
    
    Why test metadata:
    - Essential for scientific reproducibility
    - Complex data structures need validation
    - Export formats must be correct
    """
    
    def test_metadata_collection(self):
        """Test that metadata is collected correctly."""
        tracker = MetadataTracker()
        
        # Mock metadata
        from make_dataset_scientific import ImageMetadata
        metadata = ImageMetadata(
            filename="test_001.png",
            label=1,
            split="train",
            generation_time=0.1,
            random_seed=42,
            image_size=64,
            brightness=0.8,
            noise_level=0.02,
            n_arcs=2,
            arc_radii=[10.0, 15.0],
            arc_widths=[2.5, 3.0],
            arc_angles=[0.5, 1.2]
        )
        
        tracker.add_image_metadata(metadata)
        
        assert len(tracker.metadata) == 1
        assert tracker.metadata[0].filename == "test_001.png"
        assert tracker.metadata[0].n_arcs == 2
    
    def test_metadata_csv_export(self):
        """Test CSV export functionality."""
        tracker = MetadataTracker()
        
        # Add sample metadata
        from make_dataset_scientific import ImageMetadata
        metadata = ImageMetadata(
            filename="test.png",
            label=1,
            split="train",
            generation_time=0.1,
            random_seed=42,
            image_size=64,
            brightness=0.8,
            noise_level=0.02
        )
        
        tracker.add_image_metadata(metadata)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "metadata.csv"
            tracker.export_to_csv(csv_path)
            
            # Verify CSV was created and has correct structure
            assert csv_path.exists()
            df = pd.read_csv(csv_path)
            
            assert len(df) == 1
            assert "filename" in df.columns
            assert "label" in df.columns
            assert "brightness" in df.columns
            assert df.iloc[0]["filename"] == "test.png"
            assert df.iloc[0]["label"] == 1


class TestLogging:
    """Test logging setup and functionality.
    
    Why test logging:
    - Critical for debugging production issues
    - Configuration must be correct
    - Log levels must work as expected
    """
    
    def test_logger_setup_with_console_handler(self):
        """Test basic logger setup."""
        logger = setup_logging("INFO")
        
        assert logger.name == "dataset_generator"
        assert logger.level == 20  # INFO level
        assert len(logger.handlers) >= 1  # At least console handler
    
    def test_logger_setup_with_file_handler(self):
        """Test logger setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging("DEBUG", log_file)
            
            # Log a test message
            logger.info("Test message")
            
            # Verify log file was created and contains message
            assert log_file.exists()
            log_content = log_file.read_text()
            assert "Test message" in log_content
            assert "INFO" in log_content
    
    def test_invalid_log_level_raises_error(self):
        """Test that invalid log levels are rejected."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging("INVALID_LEVEL")


class TestIntegration:
    """Integration tests for complete pipeline.
    
    Why integration tests:
    - Verify components work together
    - Catch interface mismatches
    - Test realistic usage patterns
    """
    
    def test_complete_small_dataset_generation(self):
        """Test generating a complete small dataset."""
        config = DatasetConfig(
            general=GeneralConfig(n_train=4, n_test=2, image_size=16, seed=42)
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Initialize components
            metadata_tracker = MetadataTracker()
            metadata_tracker.set_config_snapshot(config)
            rng = np.random.Generator(np.random.PCG64(42))
            
            # Generate dataset
            generator = SyntheticImageGenerator(config, rng, metadata_tracker)
            generator.generate_dataset(output_dir)
            
            # Verify directory structure
            assert (output_dir / "train" / "lens").exists()
            assert (output_dir / "train" / "nonlens").exists()
            assert (output_dir / "test" / "lens").exists()
            assert (output_dir / "test" / "nonlens").exists()
            
            # Count generated images
            train_lens_images = list((output_dir / "train" / "lens").glob("*.png"))
            train_nonlens_images = list((output_dir / "train" / "nonlens").glob("*.png"))
            test_lens_images = list((output_dir / "test" / "lens").glob("*.png"))
            test_nonlens_images = list((output_dir / "test" / "nonlens").glob("*.png"))
            
            # With balance=0.5, should have equal numbers
            assert len(train_lens_images) == 2  # 4 * 0.5
            assert len(train_nonlens_images) == 2  # 4 * 0.5
            assert len(test_lens_images) == 1   # 2 * 0.5
            assert len(test_nonlens_images) == 1  # 2 * 0.5
            
            # Verify metadata was collected
            assert len(metadata_tracker.metadata) == 6  # Total images
    
    @pytest.mark.parametrize("image_size", [16, 32, 64])
    def test_different_image_sizes(self, image_size):
        """Test generation with different image sizes."""
        config = DatasetConfig(
            general=GeneralConfig(n_train=2, n_test=1, image_size=image_size, seed=42)
        )
        
        metadata_tracker = MetadataTracker()
        rng = np.random.Generator(np.random.PCG64(42))
        generator = SyntheticImageGenerator(config, rng, metadata_tracker)
        
        img, metadata = generator.create_lens_arc_image("test", "train")
        
        assert img.shape == (image_size, image_size)
        assert metadata.image_size == image_size


# ============================================================================
# PYTEST CONFIGURATION AND FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Run tests with: python -m pytest tests/test_dataset_generator.py -v
