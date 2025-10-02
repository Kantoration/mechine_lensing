#!/usr/bin/env python3
"""
Test suite for the datasets package.

This module tests the core functionality of the datasets package to ensure
that all imports work correctly and the basic interfaces are functional.
"""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test imports
try:
    from src.datasets.lens_dataset import LensDataset, LensDatasetError
    from src.datasets.optimized_dataloader import create_dataloaders
    from src.datasets import LensDataset as LensDatasetImport
    from src.datasets import create_dataloaders as create_dataloaders_import
    IMPORTS_WORKING = True
except ImportError as e:
    IMPORTS_WORKING = False
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None


class TestDatasetImports(unittest.TestCase):
    """Test that all dataset imports work correctly."""
    
    def test_datasets_imports(self):
        """Test that the datasets package can be imported."""
        self.assertTrue(IMPORTS_WORKING, f"Dataset imports failed: {IMPORT_ERROR}")
    
    def test_lens_dataset_import(self):
        """Test that LensDataset can be imported."""
        self.assertIsNotNone(LensDatasetImport)
        self.assertEqual(LensDatasetImport, LensDataset)
    
    def test_create_dataloaders_import(self):
        """Test that create_dataloaders can be imported."""
        self.assertIsNotNone(create_dataloaders_import)
        self.assertEqual(create_dataloaders_import, create_dataloaders)


class TestLensDataset(unittest.TestCase):
    """Test the LensDataset class functionality."""
    
    def setUp(self):
        """Set up test data."""
        if not IMPORTS_WORKING:
            self.skipTest("Dataset imports not working")
        
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = Path(self.temp_dir)
        
        # Create directory structure
        (self.data_root / "train" / "lens").mkdir(parents=True)
        (self.data_root / "train" / "nonlens").mkdir(parents=True)
        (self.data_root / "test" / "lens").mkdir(parents=True)
        (self.data_root / "test" / "nonlens").mkdir(parents=True)
        
        # Create sample images
        self.create_sample_images()
    
    def tearDown(self):
        """Clean up test data."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_images(self):
        """Create sample images and CSV files for testing."""
        import pandas as pd
        
        # Create a few sample images and collect paths
        train_data = []
        test_data = []
        
        for i in range(3):
            # Train lens images
            img_path = self.data_root / "train" / "lens" / f"lens_train_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='red')
            img.save(img_path)
            train_data.append({"filepath": str(img_path), "label": 1})
            
            # Train non-lens images
            img_path = self.data_root / "train" / "nonlens" / f"nonlens_train_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='blue')
            img.save(img_path)
            train_data.append({"filepath": str(img_path), "label": 0})
            
            # Test lens images
            img_path = self.data_root / "test" / "lens" / f"lens_test_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='green')
            img.save(img_path)
            test_data.append({"filepath": str(img_path), "label": 1})
            
            # Test non-lens images
            img_path = self.data_root / "test" / "nonlens" / f"nonlens_test_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='yellow')
            img.save(img_path)
            test_data.append({"filepath": str(img_path), "label": 0})
        
        # Create CSV files
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        train_df.to_csv(self.data_root / "train.csv", index=False)
        test_df.to_csv(self.data_root / "test.csv", index=False)
    
    def test_lens_dataset_creation(self):
        """Test that LensDataset can be created."""
        dataset = LensDataset(
            data_root=str(self.data_root),
            split="train",
            img_size=64,
            augment=False,
            validate_paths=True
        )
        self.assertIsInstance(dataset, LensDataset)
        self.assertEqual(len(dataset), 6)  # 3 lens + 3 nonlens
    
    def test_lens_dataset_getitem(self):
        """Test that LensDataset returns correct (image, label) pairs."""
        dataset = LensDataset(
            data_root=str(self.data_root),
            split="train",
            img_size=64,
            augment=False,
            validate_paths=True
        )
        
        image, label = dataset[0]
        
        # Check that we get a tensor and a label
        import torch
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, (int, np.integer))
        self.assertIn(label, [0, 1])  # Binary classification
    
    def test_lens_dataset_error_handling(self):
        """Test that LensDataset handles errors gracefully."""
        # Test with non-existent directory
        with self.assertRaises(LensDatasetError):
            LensDataset(
                data_root="/non/existent/path",
                split="train",
                img_size=64,
                augment=False,
                validate_paths=True
            )


class TestOptimizedDataloader(unittest.TestCase):
    """Test the optimized dataloader functionality."""
    
    def setUp(self):
        """Set up test data."""
        if not IMPORTS_WORKING:
            self.skipTest("Dataset imports not working")
        
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = Path(self.temp_dir)
        
        # Create directory structure
        (self.data_root / "train" / "lens").mkdir(parents=True)
        (self.data_root / "train" / "nonlens").mkdir(parents=True)
        (self.data_root / "test" / "lens").mkdir(parents=True)
        (self.data_root / "test" / "nonlens").mkdir(parents=True)
        
        # Create sample images
        self.create_sample_images()
    
    def tearDown(self):
        """Clean up test data."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_images(self):
        """Create sample images and CSV files for testing."""
        import pandas as pd
        
        # Create a few sample images and collect paths
        train_data = []
        test_data = []
        
        for i in range(10):
            # Train lens images
            img_path = self.data_root / "train" / "lens" / f"lens_train_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='red')
            img.save(img_path)
            train_data.append({"filepath": str(img_path), "label": 1})
            
            # Train non-lens images
            img_path = self.data_root / "train" / "nonlens" / f"nonlens_train_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='blue')
            img.save(img_path)
            train_data.append({"filepath": str(img_path), "label": 0})
            
            # Test lens images
            img_path = self.data_root / "test" / "lens" / f"lens_test_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='green')
            img.save(img_path)
            test_data.append({"filepath": str(img_path), "label": 1})
            
            # Test non-lens images
            img_path = self.data_root / "test" / "nonlens" / f"nonlens_test_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='yellow')
            img.save(img_path)
            test_data.append({"filepath": str(img_path), "label": 0})
        
        # Create CSV files
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        train_df.to_csv(self.data_root / "train.csv", index=False)
        test_df.to_csv(self.data_root / "test.csv", index=False)
    
    def test_create_dataloaders(self):
        """Test that create_dataloaders works correctly."""
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=str(self.data_root),
            batch_size=4,
            img_size=64,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            val_split=0.2
        )
        
        # Check that we get DataLoader objects
        from torch.utils.data import DataLoader
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)
        
        # Test that we can iterate through the loaders
        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 2)  # (images, labels)
        
        images, labels = batch
        self.assertEqual(images.shape[0], 4)  # Batch size
        self.assertEqual(labels.shape[0], 4)  # Batch size


if __name__ == '__main__':
    print("Running dataset tests...")
    
    # Run tests
    unittest.main(verbosity=2)
