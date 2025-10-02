#!/usr/bin/env python3
"""
Unit tests for PerformanceMonitor throughput metrics.

This module tests that the PerformanceMonitor correctly calculates
samples_per_second and batches_per_second based on actual sample counts
rather than epochs per second.
"""

import unittest
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor throughput calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.training.accelerated_trainer import PerformanceMonitor
        self.monitor = PerformanceMonitor()
    
    def test_initial_state(self):
        """Test that monitor starts with correct initial state."""
        self.assertEqual(self.monitor.total_samples_processed, 0)
        self.assertEqual(self.monitor.total_batches_processed, 0)
        self.assertEqual(len(self.monitor.epoch_times), 0)
        self.assertEqual(len(self.monitor.samples_per_epoch), 0)
        self.assertEqual(len(self.monitor.batches_per_epoch), 0)
    
    def test_record_batch(self):
        """Test that record_batch correctly tracks samples and batches."""
        # Record some batches
        self.monitor.record_batch(32)  # 32 samples
        self.monitor.record_batch(16)  # 16 samples
        self.monitor.record_batch(32)  # 32 samples
        
        self.assertEqual(self.monitor.total_samples_processed, 80)
        self.assertEqual(self.monitor.total_batches_processed, 3)
    
    def test_epoch_with_sample_tracking(self):
        """Test that end_epoch correctly tracks samples and batches."""
        # Start an epoch
        self.monitor.start_epoch()
        
        # Simulate some training time
        time.sleep(0.01)  # 10ms
        
        # End epoch with sample and batch counts
        samples_processed = 1000
        batches_processed = 32
        epoch_time = self.monitor.end_epoch(samples_processed, batches_processed)
        
        # Check that epoch time is recorded
        self.assertGreater(epoch_time, 0)
        self.assertEqual(len(self.monitor.epoch_times), 1)
        
        # Check that samples and batches are recorded
        self.assertEqual(self.monitor.total_samples_processed, samples_processed)
        self.assertEqual(self.monitor.total_batches_processed, batches_processed)
        self.assertEqual(self.monitor.samples_per_epoch[0], samples_processed)
        self.assertEqual(self.monitor.batches_per_epoch[0], batches_processed)
    
    def test_throughput_calculation_with_samples(self):
        """Test that samples_per_second is calculated correctly with sample tracking."""
        # Simulate 2 epochs with known sample counts
        for epoch in range(2):
            self.monitor.start_epoch()
            time.sleep(0.01)  # 10ms per epoch
            samples = 1000 * (epoch + 1)  # 1000, 2000 samples
            batches = 32 * (epoch + 1)    # 32, 64 batches
            self.monitor.end_epoch(samples, batches)
        
        stats = self.monitor.get_stats()
        
        # Check that we have the right total counts
        self.assertEqual(stats['total_samples_processed'], 3000)  # 1000 + 2000
        self.assertEqual(stats['total_batches_processed'], 96)    # 32 + 64
        
        # Check that samples_per_second is calculated correctly
        # Total time should be ~20ms, so samples_per_second should be ~3000/0.02 = 150,000
        self.assertGreater(stats['samples_per_second'], 100000)  # Should be very high for this test
        self.assertLess(stats['samples_per_second'], 500000)     # But not unreasonably high
        
        # Check that batches_per_second is calculated correctly
        self.assertGreater(stats['batches_per_second'], 4000)    # 96 batches / 0.02s = 4800
        self.assertLess(stats['batches_per_second'], 10000)
        
        # Check that we have the additional metrics
        self.assertIn('avg_samples_per_epoch', stats)
        self.assertIn('avg_batches_per_epoch', stats)
        self.assertEqual(stats['avg_samples_per_epoch'], 1500)  # (1000 + 2000) / 2
        self.assertEqual(stats['avg_batches_per_epoch'], 48)    # (32 + 64) / 2
    
    def test_throughput_calculation_without_samples(self):
        """Test fallback to epochs_per_second when no sample tracking."""
        # Create a new monitor and don't provide sample counts
        from src.training.accelerated_trainer import PerformanceMonitor
        monitor = PerformanceMonitor()
        
        # Simulate epochs without sample tracking
        for epoch in range(3):
            monitor.start_epoch()
            time.sleep(0.01)  # 10ms per epoch
            monitor.end_epoch()  # No sample counts provided
        
        stats = monitor.get_stats()
        
        # Should fall back to epochs per second calculation
        self.assertIn('samples_per_second', stats)
        self.assertIn('epochs_per_second', stats)
        
        # samples_per_second should equal epochs_per_second when no sample tracking
        self.assertEqual(stats['samples_per_second'], stats['epochs_per_second'])
        
        # Should be ~3 epochs / 0.03s = 100 epochs/second
        self.assertGreater(stats['epochs_per_second'], 80)
        self.assertLess(stats['epochs_per_second'], 150)
    
    def test_multiple_epochs_cumulative_tracking(self):
        """Test that tracking is cumulative across multiple epochs."""
        # Simulate multiple epochs with different sample counts
        epoch_data = [
            (1000, 32),  # 1000 samples, 32 batches
            (1500, 48),  # 1500 samples, 48 batches
            (800, 25),   # 800 samples, 25 batches
        ]
        
        for samples, batches in epoch_data:
            self.monitor.start_epoch()
            time.sleep(0.005)  # 5ms per epoch
            self.monitor.end_epoch(samples, batches)
        
        stats = self.monitor.get_stats()
        
        # Check cumulative totals
        expected_total_samples = sum(s for s, b in epoch_data)
        expected_total_batches = sum(b for s, b in epoch_data)
        
        self.assertEqual(stats['total_samples_processed'], expected_total_samples)
        self.assertEqual(stats['total_batches_processed'], expected_total_batches)
        
        # Check per-epoch tracking
        self.assertEqual(len(self.monitor.samples_per_epoch), 3)
        self.assertEqual(len(self.monitor.batches_per_epoch), 3)
        
        for i, (expected_samples, expected_batches) in enumerate(epoch_data):
            self.assertEqual(self.monitor.samples_per_epoch[i], expected_samples)
            self.assertEqual(self.monitor.batches_per_epoch[i], expected_batches)
    
    def test_stats_structure(self):
        """Test that get_stats returns all expected metrics."""
        # Add some data
        self.monitor.start_epoch()
        time.sleep(0.01)
        self.monitor.end_epoch(1000, 32)
        
        stats = self.monitor.get_stats()
        
        # Check that all expected keys are present
        expected_keys = [
            'avg_epoch_time',
            'total_training_time',
            'samples_per_second',
            'batches_per_second',
            'avg_samples_per_epoch',
            'avg_batches_per_epoch',
            'total_samples_processed',
            'total_batches_processed'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing key: {key}")
            self.assertIsInstance(stats[key], (int, float), f"Key {key} should be numeric")
    
    def test_performance_monitor_integration(self):
        """Test PerformanceMonitor with realistic training scenario."""
        # Simulate a realistic training scenario
        batch_size = 32
        num_batches = 100
        total_samples_per_epoch = batch_size * num_batches
        
        # Simulate 5 epochs
        for epoch in range(5):
            self.monitor.start_epoch()
            
            # Simulate realistic training time (0.1 seconds per epoch)
            time.sleep(0.01)  # Shortened for test speed
            
            self.monitor.end_epoch(total_samples_per_epoch, num_batches)
        
        stats = self.monitor.get_stats()
        
        # Verify realistic throughput calculations
        total_samples = 5 * total_samples_per_epoch  # 5 epochs
        total_batches = 5 * num_batches
        
        self.assertEqual(stats['total_samples_processed'], total_samples)
        self.assertEqual(stats['total_batches_processed'], total_batches)
        
        # samples_per_second should be total_samples / total_time
        # With ~0.05s total time and 16000 samples, should be ~320,000 samples/sec
        expected_samples_per_second = total_samples / stats['total_training_time']
        self.assertAlmostEqual(stats['samples_per_second'], expected_samples_per_second, delta=1000)
        
        # batches_per_second should be total_batches / total_time
        expected_batches_per_second = total_batches / stats['total_training_time']
        self.assertAlmostEqual(stats['batches_per_second'], expected_batches_per_second, delta=100)


if __name__ == '__main__':
    print("Running PerformanceMonitor throughput tests...")
    unittest.main(verbosity=2)
