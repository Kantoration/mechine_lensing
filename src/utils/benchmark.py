#!/usr/bin/env python3
"""
benchmark.py
============
Performance benchmarking utilities for training and inference.

Key Features:
- Comprehensive performance profiling
- Memory usage monitoring
- GPU utilization tracking (requires GPUtil package)
- Throughput and latency measurements
- Comparative analysis across models and configurations

Dependencies:
- GPUtil: Optional dependency for GPU utilization monitoring
  Install with: pip install GPUtil
  Or install dev requirements: pip install -r requirements-dev.txt

Usage:
    from utils.benchmark import BenchmarkSuite, profile_training, profile_inference
"""

import time
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Initialize logger first so it's available for GPUtil import handling
logger = logging.getLogger(__name__)

# Optional dependencies for GPU and system monitoring
try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPUTIL_AVAILABLE = False
    logger.warning("GPUtil not available. GPU utilization monitoring will be disabled.")

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. System monitoring will be disabled.")


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""

    # Timing metrics
    total_time: float
    avg_batch_time: float
    samples_per_second: float
    batches_per_second: float

    # Memory metrics
    peak_memory_gb: float
    avg_memory_gb: float

    # Model metrics
    model_size_mb: float
    num_parameters: int

    # Configuration
    batch_size: int
    num_workers: int
    use_amp: bool
    device: str

    # Optional GPU metrics
    gpu_memory_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_temperature: Optional[float] = None


class BenchmarkSuite:
    """Comprehensive benchmarking suite."""

    def __init__(self, output_dir: str = "benchmarks"):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.start_time = None
        self.start_memory = None

        logger.info(f"Initialized benchmark suite: {self.output_dir}")

    def start_profiling(self):
        """Start profiling session."""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1e9

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.debug("Started profiling session")

    def end_profiling(self) -> Dict[str, float]:
        """
        End profiling session and return metrics.

        Returns:
            Dictionary of profiling metrics
        """
        if self.start_time is None:
            raise RuntimeError("Profiling session not started")

        end_time = time.time()
        end_memory = psutil.virtual_memory().used / 1e9

        metrics = {
            "total_time": end_time - self.start_time,
            "memory_delta_gb": end_memory - self.start_memory,
            "peak_memory_gb": end_memory,
        }

        if torch.cuda.is_available():
            metrics["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            metrics["gpu_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9

        # GPU utilization (if available)
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    metrics["gpu_utilization"] = gpus[0].load * 100
                    metrics["gpu_temperature"] = gpus[0].temperature
            except Exception as e:
                logger.debug(f"Failed to get GPU utilization: {e}")
        else:
            logger.debug("GPU utilization monitoring disabled (GPUtil not available)")

        self.start_time = None
        self.start_memory = None

        logger.debug(f"Ended profiling session: {metrics}")
        return metrics

    def benchmark_training(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1,
        use_amp: bool = False,
        device: Optional[torch.device] = None,
    ) -> PerformanceMetrics:
        """
        Benchmark training performance.

        Args:
            model: Model to benchmark
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            num_epochs: Number of epochs to run
            use_amp: Use automatic mixed precision
            device: Device to run on

        Returns:
            Performance metrics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.train()

        # Setup mixed precision
        scaler = (
            torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None
        )

        # Start profiling
        self.start_profiling()

        total_samples = 0
        batch_times = []

        for epoch in range(num_epochs):
            epoch_start = time.time()

            for batch_idx, (images, labels) in enumerate(train_loader):
                batch_start = time.time()

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(images).squeeze(1)
                        loss = criterion(logits, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(images).squeeze(1)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                batch_size = images.size(0)
                total_samples += batch_size

                # Log progress
                if batch_idx % 10 == 0:
                    logger.debug(
                        f"Epoch {epoch + 1}, Batch {batch_idx}: {batch_time:.3f}s"
                    )

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

        # End profiling
        profiling_metrics = self.end_profiling()

        # Calculate metrics
        total_time = profiling_metrics["total_time"]
        avg_batch_time = np.mean(batch_times)
        samples_per_second = total_samples / total_time
        batches_per_second = len(batch_times) / total_time

        # Model metrics
        model_size_mb = (
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        )
        num_parameters = sum(p.numel() for p in model.parameters())

        metrics = PerformanceMetrics(
            total_time=total_time,
            avg_batch_time=avg_batch_time,
            samples_per_second=samples_per_second,
            batches_per_second=batches_per_second,
            peak_memory_gb=profiling_metrics["peak_memory_gb"],
            avg_memory_gb=profiling_metrics["memory_delta_gb"],
            gpu_memory_gb=profiling_metrics.get("gpu_memory_gb"),
            gpu_utilization=profiling_metrics.get("gpu_utilization"),
            gpu_temperature=profiling_metrics.get("gpu_temperature"),
            model_size_mb=model_size_mb,
            num_parameters=num_parameters,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
            use_amp=use_amp,
            device=str(device),
        )

        self.results.append(metrics)
        logger.info(
            f"Training benchmark completed: {samples_per_second:.1f} samples/sec"
        )

        return metrics

    def benchmark_inference(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        use_amp: bool = False,
        device: Optional[torch.device] = None,
        warmup_batches: int = 5,
    ) -> PerformanceMetrics:
        """
        Benchmark inference performance.

        Args:
            model: Model to benchmark
            dataloader: Data loader
            use_amp: Use automatic mixed precision
            device: Device to run on
            warmup_batches: Number of warmup batches

        Returns:
            Performance metrics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        # Warmup
        logger.info(f"Warming up with {warmup_batches} batches...")
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= warmup_batches:
                    break

                images = images.to(device, non_blocking=True)

                if use_amp and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        _ = model(images)
                else:
                    _ = model(images)

        # Start profiling
        self.start_profiling()

        total_samples = 0
        batch_times = []

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                batch_start = time.time()

                images = images.to(device, non_blocking=True)

                if use_amp and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        _ = model(images)
                else:
                    _ = model(images)

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                batch_size = images.size(0)
                total_samples += batch_size

                # Log progress
                if batch_idx % 20 == 0:
                    logger.debug(f"Batch {batch_idx}: {batch_time:.3f}s")

        # End profiling
        profiling_metrics = self.end_profiling()

        # Calculate metrics
        total_time = profiling_metrics["total_time"]
        avg_batch_time = np.mean(batch_times)
        samples_per_second = total_samples / total_time
        batches_per_second = len(batch_times) / total_time

        # Model metrics
        model_size_mb = (
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        )
        num_parameters = sum(p.numel() for p in model.parameters())

        metrics = PerformanceMetrics(
            total_time=total_time,
            avg_batch_time=avg_batch_time,
            samples_per_second=samples_per_second,
            batches_per_second=batches_per_second,
            peak_memory_gb=profiling_metrics["peak_memory_gb"],
            avg_memory_gb=profiling_metrics["memory_delta_gb"],
            gpu_memory_gb=profiling_metrics.get("gpu_memory_gb"),
            gpu_utilization=profiling_metrics.get("gpu_utilization"),
            gpu_temperature=profiling_metrics.get("gpu_temperature"),
            model_size_mb=model_size_mb,
            num_parameters=num_parameters,
            batch_size=dataloader.batch_size,
            num_workers=dataloader.num_workers,
            use_amp=use_amp,
            device=str(device),
        )

        self.results.append(metrics)
        logger.info(
            f"Inference benchmark completed: {samples_per_second:.1f} samples/sec"
        )

        return metrics

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader: DataLoader,
        use_amp: bool = False,
        device: Optional[torch.device] = None,
    ) -> Dict[str, PerformanceMetrics]:
        """
        Compare performance across multiple models.

        Args:
            models: Dictionary of model_name -> model
            dataloader: Data loader
            use_amp: Use automatic mixed precision
            device: Device to run on

        Returns:
            Dictionary of model_name -> performance metrics
        """
        results = {}

        for name, model in models.items():
            logger.info(f"Benchmarking {name}...")
            metrics = self.benchmark_inference(model, dataloader, use_amp, device)
            results[name] = metrics

        return results

    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save benchmark results to file.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        output_path = self.output_dir / filename

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append(asdict(result))

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Benchmark results saved to {output_path}")
        return output_path

    def generate_report(self) -> str:
        """
        Generate a human-readable benchmark report.

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No benchmark results available."

        report = []
        report.append("=" * 80)
        report.append("BENCHMARK REPORT")
        report.append("=" * 80)

        for i, result in enumerate(self.results):
            report.append(f"\nBenchmark {i + 1}:")
            report.append(f"  Device: {result.device}")
            report.append(f"  Batch Size: {result.batch_size}")
            report.append(f"  AMP: {result.use_amp}")
            report.append(f"  Model Size: {result.model_size_mb:.1f} MB")
            report.append(f"  Parameters: {result.num_parameters:,}")
            report.append(f"  Throughput: {result.samples_per_second:.1f} samples/sec")
            report.append(f"  Avg Batch Time: {result.avg_batch_time:.3f}s")
            report.append(f"  Peak Memory: {result.peak_memory_gb:.1f} GB")

            if result.gpu_memory_gb:
                report.append(f"  GPU Memory: {result.gpu_memory_gb:.1f} GB")

            if result.gpu_utilization:
                report.append(f"  GPU Utilization: {result.gpu_utilization:.1f}%")

        # Summary statistics
        if len(self.results) > 1:
            report.append("\nSUMMARY:")
            report.append(
                f"  Best Throughput: {max(r.samples_per_second for r in self.results):.1f} samples/sec"
            )
            report.append(
                f"  Average Throughput: {np.mean([r.samples_per_second for r in self.results]):.1f} samples/sec"
            )
            report.append(
                f"  Memory Range: {min(r.peak_memory_gb for r in self.results):.1f} - {max(r.peak_memory_gb for r in self.results):.1f} GB"
            )

        report.append("=" * 80)

        return "\n".join(report)


def profile_training(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 1,
    use_amp: bool = False,
    device: Optional[torch.device] = None,
) -> PerformanceMetrics:
    """
    Quick training profiling function.

    Args:
        model: Model to profile
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to run
        use_amp: Use automatic mixed precision
        device: Device to run on

    Returns:
        Performance metrics
    """
    suite = BenchmarkSuite()
    return suite.benchmark_training(
        model, train_loader, criterion, optimizer, num_epochs, use_amp, device
    )


def profile_inference(
    model: nn.Module,
    dataloader: DataLoader,
    use_amp: bool = False,
    device: Optional[torch.device] = None,
) -> PerformanceMetrics:
    """
    Quick inference profiling function.

    Args:
        model: Model to profile
        dataloader: Data loader
        use_amp: Use automatic mixed precision
        device: Device to run on

    Returns:
        Performance metrics
    """
    suite = BenchmarkSuite()
    return suite.benchmark_inference(model, dataloader, use_amp, device)


def benchmark_ensemble(
    ensemble_models: Dict[str, nn.Module],
    dataloader: DataLoader,
    use_amp: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, PerformanceMetrics]:
    """
    Benchmark ensemble inference performance.

    Args:
        ensemble_models: Dictionary of model_name -> model
        dataloader: Data loader
        use_amp: Use automatic mixed precision
        device: Device to run on

    Returns:
        Dictionary of model_name -> performance metrics
    """
    suite = BenchmarkSuite()
    return suite.compare_models(ensemble_models, dataloader, use_amp, device)


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Create dummy data
    batch_size = 32
    num_samples = 1000
    input_size = (3, 64, 64)

    X = torch.randn(num_samples, *input_size)
    y = torch.randint(0, 2, (num_samples,))

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 1),
    )

    # Benchmark
    suite = BenchmarkSuite()
    metrics = suite.benchmark_inference(model, dataloader)

    print(suite.generate_report())
    suite.save_results()
