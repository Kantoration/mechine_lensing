#!/usr/bin/env python3
"""
Test GPUtil fallback behavior in benchmark utilities.

This module tests that the benchmark utilities can be imported and used
gracefully when GPUtil is not available.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestGPUtilFallback(unittest.TestCase):
    """Test that benchmark utilities work without GPUtil."""
    
    def test_import_without_gputil(self):
        """Test that benchmark utilities can be imported when GPUtil is missing."""
        with patch.dict('sys.modules', {'GPUtil': None}):
            # Remove GPUtil from modules if it exists
            if 'GPUtil' in sys.modules:
                del sys.modules['GPUtil']
            
            # This should not raise an ImportError or NameError
            try:
                from src.utils.benchmark import BenchmarkSuite, PerformanceMetrics, GPUTIL_AVAILABLE
                self.assertFalse(GPUTIL_AVAILABLE, "GPUTIL_AVAILABLE should be False when GPUtil is missing")
                self.assertTrue(True, "Import succeeded without GPUtil")
            except (ImportError, NameError) as e:
                self.fail(f"Import failed without GPUtil: {e}")
    
    def test_gputil_availability_flag(self):
        """Test that GPUTIL_AVAILABLE flag is set correctly."""
        # Test with GPUtil available (normal case)
        try:
            import GPUtil
            from src.utils.benchmark import GPUTIL_AVAILABLE
            # If GPUtil is actually available, the flag should be True
            # If not, it should be False
            self.assertIsInstance(GPUTIL_AVAILABLE, bool)
        except ImportError:
            # GPUtil not available, should be False
            from src.utils.benchmark import GPUTIL_AVAILABLE
            self.assertFalse(GPUTIL_AVAILABLE)
    
    def test_benchmark_suite_creation_without_gputil(self):
        """Test that BenchmarkSuite can be created without GPUtil."""
        with patch.dict('sys.modules', {'GPUtil': None}):
            # Remove GPUtil from modules if it exists
            if 'GPUtil' in sys.modules:
                del sys.modules['GPUtil']
            
            try:
                from src.utils.benchmark import BenchmarkSuite
                suite = BenchmarkSuite()
                self.assertIsNotNone(suite)
            except Exception as e:
                self.fail(f"BenchmarkSuite creation failed without GPUtil: {e}")
    
    def test_profiler_without_gputil(self):
        """Test that profiler works without GPUtil."""
        with patch.dict('sys.modules', {'GPUtil': None}):
            # Remove GPUtil from modules if it exists
            if 'GPUtil' in sys.modules:
                del sys.modules['GPUtil']
            
            try:
                from src.utils.benchmark import BenchmarkSuite
                import torch
                
                suite = BenchmarkSuite()
                
                # Create a dummy model for testing
                model = torch.nn.Linear(10, 1)
                dummy_input = torch.randn(1, 10)
                
                # This should work without GPUtil
                suite.start_profiling()
                _ = model(dummy_input)
                
                # End profiling - should not crash
                metrics = suite.end_profiling()
                
                # Should have basic metrics but no GPU utilization
                self.assertIn('total_time', metrics)
                self.assertNotIn('gpu_utilization', metrics)
                self.assertNotIn('gpu_temperature', metrics)
                
            except Exception as e:
                self.fail(f"Profiler failed without GPUtil: {e}")
    
    def test_gputil_mock_behavior(self):
        """Test behavior when GPUtil is mocked."""
        # Mock GPUtil to simulate it being available but failing
        mock_gputil = MagicMock()
        mock_gputil.getGPUs.return_value = []
        
        with patch.dict('sys.modules', {'GPUtil': mock_gputil}):
            try:
                from src.utils.benchmark import BenchmarkSuite
                import torch
                
                suite = BenchmarkSuite()
                
                # Create a dummy model for testing
                model = torch.nn.Linear(10, 1)
                dummy_input = torch.randn(1, 10)
                
                # This should work with mocked GPUtil
                suite.start_profiling()
                _ = model(dummy_input)
                
                # End profiling
                metrics = suite.end_profiling()
                
                # Should have basic metrics
                self.assertIn('total_time', metrics)
                
            except Exception as e:
                self.fail(f"Profiler failed with mocked GPUtil: {e}")


class TestBenchmarkImportStability(unittest.TestCase):
    """Test that benchmark utilities are stable across different import scenarios."""
    
    def test_multiple_imports(self):
        """Test that multiple imports work correctly."""
        try:
            # Import multiple times to ensure stability
            from src.utils.benchmark import BenchmarkSuite, PerformanceMetrics
            from src.utils.benchmark import GPUTIL_AVAILABLE
            from src.utils.benchmark import BenchmarkSuite as BS2
            
            self.assertTrue(True, "Multiple imports succeeded")
            
        except Exception as e:
            self.fail(f"Multiple imports failed: {e}")
    
    def test_import_with_logging_warning(self):
        """Test that import warnings are handled correctly."""
        import logging
        
        # Test that import works regardless of GPUtil availability
        try:
            from src.utils.benchmark import BenchmarkSuite
            # If we get here, the import worked
            self.assertTrue(True, "Import succeeded")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
        
        # Test that GPUTIL_AVAILABLE flag is properly set
        from src.utils.benchmark import GPUTIL_AVAILABLE
        self.assertIsInstance(GPUTIL_AVAILABLE, bool)
    
    def test_logger_initialization_order(self):
        """Test that logger is initialized before GPUtil import handling."""
        # This test specifically verifies the fix for the logger reference before assignment issue
        # The key test is that no NameError occurs when importing without GPUtil
        
        # Test that import works without crashing due to logger reference before assignment
        try:
            # Simulate GPUtil not being available by mocking it
            with patch.dict('sys.modules', {'GPUtil': None}):
                # Remove GPUtil from modules if it exists
                if 'GPUtil' in sys.modules:
                    del sys.modules['GPUtil']
                
                # This should not raise a NameError about logger
                import importlib
                if 'src.utils.benchmark' in sys.modules:
                    # Force reload to test the import logic
                    importlib.reload(sys.modules['src.utils.benchmark'])
                else:
                    from src.utils.benchmark import BenchmarkSuite
                
                # If we get here without NameError, the fix worked
                self.assertTrue(True, "Import succeeded without logger NameError")
                
        except NameError as e:
            if 'logger' in str(e):
                self.fail(f"Logger NameError still exists: {e}")
            else:
                raise  # Re-raise if it's a different NameError


if __name__ == '__main__':
    print("Running GPUtil fallback tests...")
    unittest.main(verbosity=2)
