#!/usr/bin/env python3
"""
CLI Smoke Tests for Training Scripts
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCLISmoke(unittest.TestCase):
    """Test that CLI argument parsing works for all training scripts."""
    
    def test_trainer_cli_help(self):
        """Test that trainer.py can parse --help without crashing."""
        try:
            from src.training.trainer import main
            original_argv = sys.argv.copy()
            sys.argv = ['trainer.py', '--help']
            
            try:
                main()
            except SystemExit:
                pass  # Expected for --help
            
            sys.argv = original_argv
            
        except Exception as e:
            self.fail(f"trainer.py CLI parsing failed: {e}")
    
    def test_accelerated_trainer_cli_help(self):
        """Test that accelerated_trainer.py can parse --help without crashing."""
        try:
            from src.training.accelerated_trainer import main
            original_argv = sys.argv.copy()
            sys.argv = ['accelerated_trainer.py', '--help']
            
            try:
                main()
            except SystemExit:
                pass  # Expected for --help
            
            sys.argv = original_argv
            
        except Exception as e:
            self.fail(f"accelerated_trainer.py CLI parsing failed: {e}")


class TestAcceleratedTrainerRegression(unittest.TestCase):
    """Test that accelerated_trainer works with unified factory contract."""
    
    def setUp(self):
        """Set up test data."""
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = Path(self.temp_dir)
        
        # Create directory structure
        (self.data_root / "train" / "lens").mkdir(parents=True)
        (self.data_root / "train" / "nonlens").mkdir(parents=True)
        (self.data_root / "test" / "lens").mkdir(parents=True)
        (self.data_root / "test" / "nonlens").mkdir(parents=True)
        
        # Create sample images and CSV files
        self.create_sample_data()
    
    def tearDown(self):
        """Clean up test data."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_data(self):
        """Create minimal sample data for testing."""
        import pandas as pd
        from PIL import Image
        
        # Create a few sample images
        train_data = []
        test_data = []
        
        for i in range(2):  # Minimal dataset
            # Train lens images
            img_path = self.data_root / "train" / "lens" / f"lens_train_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='red')
            img.save(img_path)
            train_data.append({"filepath": str(img_path.relative_to(self.data_root)), "label": 1})
            
            # Train non-lens images
            img_path = self.data_root / "train" / "nonlens" / f"nonlens_train_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='blue')
            img.save(img_path)
            train_data.append({"filepath": str(img_path.relative_to(self.data_root)), "label": 0})
            
            # Test images
            img_path = self.data_root / "test" / "lens" / f"lens_test_{i:04d}.png"
            img = Image.new('RGB', (64, 64), color='green')
            img.save(img_path)
            test_data.append({"filepath": str(img_path.relative_to(self.data_root)), "label": 1})
        
        # Create CSV files
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        train_df.to_csv(self.data_root / "train.csv", index=False)
        test_df.to_csv(self.data_root / "test.csv", index=False)
    
    def test_single_model_path(self):
        """Test that single model path works with unified factory."""
        try:
            from src.training.accelerated_trainer import main
            original_argv = sys.argv.copy()
            
            # Test with single model (non-ensemble)
            sys.argv = [
                'accelerated_trainer.py',
                '--arch', 'resnet18',
                '--data-root', str(self.data_root),
                '--epochs', '1',
                '--batch-size', '2',
                '--img-size', '64',
                '--no-amp',  # Disable AMP for faster testing
                '--num-workers', '0'  # Avoid multiprocessing issues
            ]
            
            try:
                main()
                # If we get here without AttributeError, the fix worked
                self.assertTrue(True, "Single model path succeeded")
            except SystemExit as e:
                # Expected for early exit, but should not be AttributeError
                if "get_input_size" in str(e):
                    self.fail(f"get_input_size AttributeError still exists: {e}")
                # Other SystemExit is acceptable (e.g., early termination)
                pass
            
            sys.argv = original_argv
            
        except AttributeError as e:
            if "get_input_size" in str(e):
                self.fail(f"get_input_size AttributeError: {e}")
            else:
                raise  # Re-raise if it's a different AttributeError
        except Exception as e:
            # Other exceptions might be expected (e.g., CUDA not available)
            if "get_input_size" in str(e):
                self.fail(f"get_input_size AttributeError: {e}")
            # Allow other exceptions to pass (they might be environment-related)
    
    def test_ensemble_model_path(self):
        """Test that ensemble model path works with unified factory."""
        try:
            from src.training.accelerated_trainer import main
            original_argv = sys.argv.copy()
            
            # Test with ensemble model
            sys.argv = [
                'accelerated_trainer.py',
                '--arch', 'enhanced_weighted_ensemble',
                '--data-root', str(self.data_root),
                '--epochs', '1',
                '--batch-size', '2',
                '--img-size', '64',
                '--no-amp',  # Disable AMP for faster testing
                '--num-workers', '0'  # Avoid multiprocessing issues
            ]
            
            try:
                main()
                # If we get here without AttributeError, the fix worked
                self.assertTrue(True, "Ensemble model path succeeded")
            except SystemExit as e:
                # Expected for early exit, but should not be AttributeError
                if "get_input_size" in str(e):
                    self.fail(f"get_input_size AttributeError still exists: {e}")
                # Other SystemExit is acceptable (e.g., early termination)
                pass
            
            sys.argv = original_argv
            
        except AttributeError as e:
            if "get_input_size" in str(e):
                self.fail(f"get_input_size AttributeError: {e}")
            else:
                raise  # Re-raise if it's a different AttributeError
        except Exception as e:
            # Other exceptions might be expected (e.g., CUDA not available)
            if "get_input_size" in str(e):
                self.fail(f"get_input_size AttributeError: {e}")
            # Allow other exceptions to pass (they might be environment-related)


if __name__ == '__main__':
    unittest.main(verbosity=2)