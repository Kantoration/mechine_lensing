#!/usr/bin/env python3
"""
performance_test.py
==================
Comprehensive performance testing script for the lens ML pipeline.

Key Features:
- Benchmark all model architectures
- Compare training vs inference performance
- Test mixed precision acceleration
- Memory usage profiling
- Cloud deployment readiness testing

Usage:
    python scripts/performance_test.py --models resnet18,vit_b_16 --test-training
    python scripts/performance_test.py --full-suite --amp --output-dir benchmarks/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.datasets.lens_dataset import LensDataset
from src.models import build_model, list_available_models
from src.models.ensemble.registry import make_model as make_ensemble_model
from src.utils.benchmark import BenchmarkSuite, PerformanceMetrics
from src.utils.numerical import clamp_probs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceTester:
    """Comprehensive performance testing suite."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        """
        Initialize performance tester.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_suite = BenchmarkSuite(str(self.output_dir))
        self.results = {}
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Get available architectures
        models_dict = list_available_models()
        self.available_archs = models_dict.get('single_models', []) + models_dict.get('physics_models', [])
        try:
            from src.models.ensemble.registry import list_available_models as list_ensemble_models
            ensemble_archs = list_ensemble_models()
            self.available_archs.extend(ensemble_archs)
            self.available_archs = list(dict.fromkeys(self.available_archs))
        except ImportError:
            pass
        
        logger.info(f"Available architectures: {self.available_archs}")
    
    def create_test_dataset(self, data_root: str, img_size: int = 64, num_samples: int = 1000) -> DataLoader:
        """
        Create a test dataset for benchmarking.
        
        Args:
            data_root: Root directory of dataset
            img_size: Image size
            num_samples: Number of samples to use
            
        Returns:
            DataLoader for testing
        """
        try:
            # Try to use real dataset
            dataset = LensDataset(
                data_root=data_root, split="test", img_size=img_size,
                augment=False, validate_paths=True
            )
            
            # Limit dataset size for faster benchmarking
            if len(dataset) > num_samples:
                indices = torch.randperm(len(dataset))[:num_samples]
                dataset = torch.utils.data.Subset(dataset, indices)
            
            logger.info(f"Using real dataset: {len(dataset)} samples")
            
        except Exception as e:
            logger.warning(f"Could not load real dataset: {e}")
            logger.info("Creating synthetic dataset for benchmarking...")
            
            # Create synthetic dataset
            from torch.utils.data import TensorDataset
            
            X = torch.randn(num_samples, 3, img_size, img_size)
            y = torch.randint(0, 2, (num_samples,))
            dataset = TensorDataset(X, y)
            
            logger.info(f"Created synthetic dataset: {len(dataset)} samples")
        
        return DataLoader(
            dataset, batch_size=32, shuffle=False,
            num_workers=2, pin_memory=torch.cuda.is_available()
        )
    
    def create_model(self, arch: str, img_size: int) -> nn.Module:
        """
        Create a model for the given architecture.
        
        Args:
            arch: Architecture name
            img_size: Image size
            
        Returns:
            Model instance
        """
        try:
            if arch in ['trans_enc_s', 'light_transformer']:
                # Use ensemble registry for advanced models
                backbone, head, feature_dim = make_ensemble_model(
                    name=arch, bands=3, pretrained=True, dropout_p=0.5
                )
                model = nn.Sequential(backbone, head)
            else:
                # Use legacy factory for standard models
                model = build_model(arch=arch, pretrained=True, dropout_rate=0.5)
            
            # Auto-detect image size if not specified
            if hasattr(model, 'get_input_size'):
                detected_size = model.get_input_size()
                if img_size != detected_size:
                    logger.info(f"Auto-detected image size for {arch}: {detected_size}")
                    img_size = detected_size
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {arch}: {e}")
            return None
    
    def benchmark_inference(
        self,
        arch: str,
        dataloader: DataLoader,
        use_amp: bool = False,
        img_size: int = 64
    ) -> Optional[PerformanceMetrics]:
        """
        Benchmark inference performance for a single architecture.
        
        Args:
            arch: Architecture name
            dataloader: Data loader
            use_amp: Use automatic mixed precision
            img_size: Image size
            
        Returns:
            Performance metrics or None if failed
        """
        logger.info(f"Benchmarking inference: {arch} (AMP: {use_amp})")
        
        # Create model
        model = self.create_model(arch, img_size)
        if model is None:
            return None
        
        try:
            # Benchmark inference
            metrics = self.benchmark_suite.benchmark_inference(
                model=model,
                dataloader=dataloader,
                use_amp=use_amp,
                device=self.device
            )
            
            # Add architecture info
            metrics.architecture = arch
            metrics.img_size = img_size
            
            logger.info(f"{arch} inference: {metrics.samples_per_second:.1f} samples/sec")
            return metrics
            
        except Exception as e:
            logger.error(f"Inference benchmark failed for {arch}: {e}")
            return None
    
    def benchmark_training(
        self,
        arch: str,
        dataloader: DataLoader,
        use_amp: bool = False,
        img_size: int = 64,
        num_epochs: int = 1
    ) -> Optional[PerformanceMetrics]:
        """
        Benchmark training performance for a single architecture.
        
        Args:
            arch: Architecture name
            dataloader: Data loader
            use_amp: Use automatic mixed precision
            img_size: Image size
            num_epochs: Number of epochs to run
            
        Returns:
            Performance metrics or None if failed
        """
        logger.info(f"Benchmarking training: {arch} (AMP: {use_amp})")
        
        # Create model
        model = self.create_model(arch, img_size)
        if model is None:
            return None
        
        try:
            # Setup training
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
            # Benchmark training
            metrics = self.benchmark_suite.benchmark_training(
                model=model,
                train_loader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                use_amp=use_amp,
                device=self.device
            )
            
            # Add architecture info
            metrics.architecture = arch
            metrics.img_size = img_size
            
            logger.info(f"{arch} training: {metrics.samples_per_second:.1f} samples/sec")
            return metrics
            
        except Exception as e:
            logger.error(f"Training benchmark failed for {arch}: {e}")
            return None
    
    def run_comprehensive_benchmark(
        self,
        architectures: List[str],
        test_training: bool = False,
        test_inference: bool = True,
        use_amp: bool = False,
        data_root: str = "data_scientific_test",
        img_size: int = 64
    ) -> Dict[str, Dict[str, PerformanceMetrics]]:
        """
        Run comprehensive benchmark across multiple architectures.
        
        Args:
            architectures: List of architectures to test
            test_training: Whether to test training performance
            test_inference: Whether to test inference performance
            use_amp: Use automatic mixed precision
            data_root: Dataset root directory
            img_size: Image size
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        # Create test dataset
        dataloader = self.create_test_dataset(data_root, img_size)
        
        for arch in architectures:
            if arch not in self.available_archs:
                logger.warning(f"Architecture {arch} not available, skipping")
                continue
            
            results[arch] = {}
            
            # Test inference
            if test_inference:
                inference_metrics = self.benchmark_inference(
                    arch, dataloader, use_amp, img_size
                )
                if inference_metrics:
                    results[arch]['inference'] = inference_metrics
            
            # Test training
            if test_training:
                training_metrics = self.benchmark_training(
                    arch, dataloader, use_amp, img_size, num_epochs=1
                )
                if training_metrics:
                    results[arch]['training'] = training_metrics
        
        self.results = results
        return results
    
    def compare_amp_performance(
        self,
        architectures: List[str],
        data_root: str = "data_scientific_test",
        img_size: int = 64
    ) -> Dict[str, Dict[str, PerformanceMetrics]]:
        """
        Compare performance with and without automatic mixed precision.
        
        Args:
            architectures: List of architectures to test
            data_root: Dataset root directory
            img_size: Image size
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        for arch in architectures:
            if arch not in self.available_archs:
                continue
            
            logger.info(f"Comparing AMP performance for {arch}")
            results[arch] = {}
            
            # Create test dataset
            dataloader = self.create_test_dataset(data_root, img_size)
            
            # Test without AMP
            metrics_fp32 = self.benchmark_inference(arch, dataloader, use_amp=False, img_size=img_size)
            if metrics_fp32:
                results[arch]['fp32'] = metrics_fp32
            
            # Test with AMP (if GPU available)
            if self.device.type == 'cuda':
                metrics_amp = self.benchmark_inference(arch, dataloader, use_amp=True, img_size=img_size)
                if metrics_amp:
                    results[arch]['amp'] = metrics_amp
                    
                    # Calculate speedup
                    if metrics_fp32:
                        speedup = metrics_amp.samples_per_second / metrics_fp32.samples_per_second
                        logger.info(f"{arch} AMP speedup: {speedup:.2f}x")
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE PERFORMANCE REPORT")
        report.append("=" * 100)
        
        # Summary table
        report.append(f"\n{'Architecture':<20} {'Task':<12} {'Throughput':<15} {'Memory':<10} {'GPU Mem':<10} {'AMP':<5}")
        report.append("-" * 100)
        
        for arch, tasks in self.results.items():
            for task, metrics in tasks.items():
                gpu_mem = f"{metrics.gpu_memory_gb:.1f}GB" if metrics.gpu_memory_gb else "N/A"
                report.append(
                    f"{arch:<20} {task:<12} {metrics.samples_per_second:<15.1f} "
                    f"{metrics.peak_memory_gb:<10.1f} {gpu_mem:<10} {metrics.use_amp!s:<5}"
                )
        
        # Detailed results
        report.append("\n" + "=" * 100)
        report.append("DETAILED RESULTS")
        report.append("=" * 100)
        
        for arch, tasks in self.results.items():
            report.append(f"\n{arch.upper()}:")
            
            for task, metrics in tasks.items():
                report.append(f"  {task.upper()}:")
                report.append(f"    Throughput: {metrics.samples_per_second:.1f} samples/sec")
                report.append(f"    Batch Time: {metrics.avg_batch_time:.3f}s")
                report.append(f"    Memory: {metrics.peak_memory_gb:.1f} GB")
                report.append(f"    Model Size: {metrics.model_size_mb:.1f} MB")
                report.append(f"    Parameters: {metrics.num_parameters:,}")
                
                if metrics.gpu_memory_gb:
                    report.append(f"    GPU Memory: {metrics.gpu_memory_gb:.1f} GB")
                
                if metrics.gpu_utilization:
                    report.append(f"    GPU Utilization: {metrics.gpu_utilization:.1f}%")
        
        # Performance recommendations
        report.append("\n" + "=" * 100)
        report.append("PERFORMANCE RECOMMENDATIONS")
        report.append("=" * 100)
        
        # Find best performers
        best_inference = None
        best_training = None
        best_throughput = 0
        best_training_throughput = 0
        
        for arch, tasks in self.results.items():
            if 'inference' in tasks:
                throughput = tasks['inference'].samples_per_second
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_inference = arch
            
            if 'training' in tasks:
                throughput = tasks['training'].samples_per_second
                if throughput > best_training_throughput:
                    best_training_throughput = throughput
                    best_training = arch
        
        if best_inference:
            report.append(f"Best Inference: {best_inference} ({best_throughput:.1f} samples/sec)")
        
        if best_training:
            report.append(f"Best Training: {best_training} ({best_training_throughput:.1f} samples/sec)")
        
        # AMP recommendations
        amp_available = any(
            any(metrics.use_amp for metrics in tasks.values())
            for tasks in self.results.values()
        )
        
        if amp_available and self.device.type == 'cuda':
            report.append("\nAMP Recommendations:")
            report.append("- Use AMP for 2-3x speedup on GPU")
            report.append("- AMP is most beneficial for large models (ViT, ResNet34+)")
            report.append("- Monitor memory usage with AMP enabled")
        
        # Memory recommendations
        max_memory = max(
            max(metrics.peak_memory_gb for metrics in tasks.values())
            for tasks in self.results.values()
        )
        
        report.append(f"\nMemory Recommendations:")
        report.append(f"- Peak memory usage: {max_memory:.1f} GB")
        if max_memory > 8:
            report.append("- Consider using gradient checkpointing for large models")
            report.append("- Reduce batch size if memory is limited")
        
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save benchmark results to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = {}
        for arch, tasks in self.results.items():
            serializable_results[arch] = {}
            for task, metrics in tasks.items():
                # Convert PerformanceMetrics to dict
                metrics_dict = {
                    'total_time': metrics.total_time,
                    'avg_batch_time': metrics.avg_batch_time,
                    'samples_per_second': metrics.samples_per_second,
                    'batches_per_second': metrics.batches_per_second,
                    'peak_memory_gb': metrics.peak_memory_gb,
                    'avg_memory_gb': metrics.avg_memory_gb,
                    'gpu_memory_gb': metrics.gpu_memory_gb,
                    'gpu_utilization': metrics.gpu_utilization,
                    'gpu_temperature': metrics.gpu_temperature,
                    'model_size_mb': metrics.model_size_mb,
                    'num_parameters': metrics.num_parameters,
                    'batch_size': metrics.batch_size,
                    'num_workers': metrics.num_workers,
                    'use_amp': metrics.use_amp,
                    'device': metrics.device,
                    'architecture': getattr(metrics, 'architecture', None),
                    'img_size': getattr(metrics, 'img_size', None)
                }
                serializable_results[arch][task] = metrics_dict
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Performance results saved to {output_path}")
        return output_path


def main():
    """Main performance testing function."""
    parser = argparse.ArgumentParser(description="Comprehensive performance testing")
    
    # Test configuration
    parser.add_argument("--models", type=str, default="resnet18,vit_b_16",
                        help="Comma-separated list of models to test")
    parser.add_argument("--full-suite", action="store_true",
                        help="Test all available architectures")
    parser.add_argument("--test-training", action="store_true",
                        help="Test training performance")
    parser.add_argument("--test-inference", action="store_true", default=True,
                        help="Test inference performance")
    parser.add_argument("--compare-amp", action="store_true",
                        help="Compare AMP vs FP32 performance")
    
    # Performance options
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--img-size", type=int, default=64,
                        help="Image size for testing")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples for benchmarking")
    
    # Data options
    parser.add_argument("--data-root", type=str, default="data_scientific_test",
                        help="Dataset root directory")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="benchmarks",
                        help="Output directory for results")
    parser.add_argument("--save-results", action="store_true", default=True,
                        help="Save results to file")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    tester = PerformanceTester(args.output_dir)
    
    # Determine architectures to test
    if args.full_suite:
        architectures = tester.available_archs
        logger.info(f"Testing full suite: {architectures}")
    else:
        architectures = [arch.strip() for arch in args.models.split(",")]
        logger.info(f"Testing specified models: {architectures}")
    
    # Run benchmarks
    if args.compare_amp:
        logger.info("Running AMP comparison benchmark...")
        results = tester.compare_amp_performance(
            architectures, args.data_root, args.img_size
        )
    else:
        logger.info("Running comprehensive benchmark...")
        results = tester.run_comprehensive_benchmark(
            architectures=architectures,
            test_training=args.test_training,
            test_inference=args.test_inference,
            use_amp=args.amp,
            data_root=args.data_root,
            img_size=args.img_size
        )
    
    # Generate and display report
    report = tester.generate_report()
    print(report)
    
    # Save results
    if args.save_results:
        output_path = tester.save_results()
        report_path = output_path.with_suffix('.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
    
    logger.info("Performance testing completed!")


if __name__ == "__main__":
    main()




