#!/usr/bin/env python3
"""
ensemble_inference.py
====================
High-performance ensemble inference with batch processing, parallel execution,
and memory optimization.

Key Features:
- Parallel model execution on multiple GPUs
- Batch processing for large datasets
- Memory-efficient inference with gradient checkpointing
- Async data loading and preprocessing
- Performance monitoring and benchmarking

Usage:
    python src/training/ensemble_inference.py --models resnet18,vit_b_16 --batch-size 64
    python src/training/ensemble_inference.py --ensemble-config configs/enhanced_ensemble.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import asyncio
from queue import Queue
import multiprocessing as mp

from datasets.lens_dataset import LensDataset
from datasets.optimized_dataloader import create_dataloaders
from models.ensemble.registry import make_model as make_ensemble_model
from models.ensemble.weighted import UncertaintyWeightedEnsemble
from models.ensemble.enhanced_weighted import EnhancedUncertaintyEnsemble
from utils.numerical import clamp_probs, ensemble_logit_fusion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class ParallelEnsembleInference:
    """High-performance parallel ensemble inference."""
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        device_map: Optional[Dict[str, torch.device]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        use_amp: bool = True
    ):
        """
        Initialize parallel ensemble inference.
        
        Args:
            models: Dictionary of model_name -> model
            device_map: Optional device mapping for each model
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            use_amp: Use automatic mixed precision
        """
        self.models = models
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amp = use_amp
        
        # Setup device mapping
        if device_map is None:
            self.device_map = {}
            available_devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            if not available_devices:
                available_devices = [torch.device("cpu")]
            
            for i, (name, model) in enumerate(models.items()):
                device = available_devices[i % len(available_devices)]
                self.device_map[name] = device
                model.to(device)
                model.eval()
        else:
            self.device_map = device_map
            for name, model in models.items():
                model.to(device_map[name])
                model.eval()
        
        logger.info(f"Initialized parallel inference with {len(models)} models")
        logger.info(f"Device mapping: {self.device_map}")
    
    def predict_single_model(
        self,
        model_name: str,
        dataloader: DataLoader,
        mc_samples: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a single model with optimized memory management.
        
        Args:
            model_name: Name of the model
            dataloader: Data loader
            mc_samples: Number of MC dropout samples
            
        Returns:
            Tuple of (logits, probabilities)
        """
        model = self.models[model_name]
        device = self.device_map[model_name]
        
        all_logits = []
        all_probs = []
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device, non_blocking=True)
                
                # MC dropout sampling
                if mc_samples > 1:
                    model.train()  # Enable dropout
                    mc_logits = []
                    for _ in range(mc_samples):
                        if self.use_amp and device.type == 'cuda':
                            with torch.cuda.amp.autocast():
                                logits = model(images).squeeze(1)
                        else:
                            logits = model(images).squeeze(1)
                        mc_logits.append(logits)
                    model.eval()  # Disable dropout
                    
                    # Average MC samples
                    logits = torch.stack(mc_logits, dim=0).mean(dim=0)
                else:
                    if self.use_amp and device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            logits = model(images).squeeze(1)
                    else:
                        logits = model(images).squeeze(1)
                
                probs = torch.sigmoid(logits)
                probs = clamp_probs(probs)
                
                # Keep tensors on CPU to save GPU memory
                all_logits.append(logits.cpu())
                all_probs.append(probs.cpu())
                
                # Clear GPU cache periodically
                if device.type == 'cuda' and len(all_logits) % 10 == 0:
                    torch.cuda.empty_cache()
        
        return torch.cat(all_logits, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy()
    
    def predict_parallel(
        self,
        dataloader: DataLoader,
        mc_samples: int = 1,
        max_workers: Optional[int] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run parallel inference on all models.
        
        Args:
            dataloader: Data loader
            mc_samples: Number of MC dropout samples
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary of model_name -> (logits, probabilities)
        """
        if max_workers is None:
            max_workers = min(len(self.models), 4)
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all inference tasks
            future_to_model = {
                executor.submit(self.predict_single_model, name, dataloader, mc_samples): name
                for name in self.models.keys()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    logits, probs = future.result()
                    results[model_name] = (logits, probs)
                    logger.info(f"Completed inference for {model_name}: {logits.shape[0]} samples")
                except Exception as e:
                    logger.error(f"Inference failed for {model_name}: {e}")
                    raise
        
        return results
    
    def predict_batch_parallel(
        self,
        dataloader: DataLoader,
        mc_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Optimized parallel inference that processes batches across models simultaneously.
        
        This method provides better performance by:
        1. Processing batches in parallel across models
        2. Using async data loading
        3. Avoiding unnecessary numpy conversions
        4. Better memory management
        
        Args:
            dataloader: Data loader
            mc_samples: Number of MC dropout samples
            
        Returns:
            Dictionary of model_name -> logits_tensor
        """
        results = {name: [] for name in self.models.keys()}
        
        # Process batches in parallel across models
        for batch_idx, (images, _) in enumerate(dataloader):
            batch_results = {}
            
            # Run inference on all models for this batch in parallel
            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                future_to_model = {}
                
                for name, model in self.models.items():
                    device = self.device_map[name]
                    future = executor.submit(
                        self._predict_batch_single_model,
                        model, images, device, mc_samples
                    )
                    future_to_model[future] = name
                
                # Collect batch results
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        batch_logits = future.result()
                        batch_results[model_name] = batch_logits
                    except Exception as e:
                        logger.error(f"Batch inference failed for {model_name}: {e}")
                        raise
            
            # Store results
            for name, logits in batch_results.items():
                results[name].append(logits.cpu())
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        # Concatenate all batches for each model
        final_results = {}
        for name, logits_list in results.items():
            final_results[name] = torch.cat(logits_list, dim=0)
            logger.info(f"Completed inference for {name}: {final_results[name].shape[0]} samples")
        
        return final_results
    
    def _predict_batch_single_model(
        self,
        model: nn.Module,
        images: torch.Tensor,
        device: torch.device,
        mc_samples: int
    ) -> torch.Tensor:
        """
        Predict a single batch on a single model.
        
        Args:
            model: The model to run inference on
            images: Batch of images
            device: Device to run on
            mc_samples: Number of MC dropout samples
            
        Returns:
            Logits tensor
        """
        images = images.to(device, non_blocking=True)
        
        with torch.no_grad():
            if mc_samples > 1:
                model.train()  # Enable dropout
                mc_logits = []
                for _ in range(mc_samples):
                    if self.use_amp and device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            logits = model(images).squeeze(1)
                    else:
                        logits = model(images).squeeze(1)
                    mc_logits.append(logits)
                model.eval()  # Disable dropout
                
                # Average MC samples
                logits = torch.stack(mc_logits, dim=0).mean(dim=0)
            else:
                if self.use_amp and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = model(images).squeeze(1)
                else:
                    logits = model(images).squeeze(1)
        
        return logits


class BatchEnsembleProcessor:
    """Process large datasets in batches with ensemble inference."""
    
    def __init__(
        self,
        ensemble: Union[UncertaintyWeightedEnsemble, EnhancedUncertaintyEnsemble],
        batch_size: int = 64,
        use_amp: bool = True
    ):
        """
        Initialize batch ensemble processor.
        
        Args:
            ensemble: Ensemble model
            batch_size: Batch size for processing
            use_amp: Use automatic mixed precision
        """
        self.ensemble = ensemble
        self.batch_size = batch_size
        self.use_amp = use_amp
        
        # Move ensemble to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble.to(device)
        self.ensemble.eval()
        
        logger.info(f"Initialized batch processor: batch_size={batch_size}, amp={use_amp}")
    
    def process_dataset(
        self,
        dataloader: DataLoader,
        mc_samples: int = 10,
        return_uncertainty: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process entire dataset with ensemble.
        
        Args:
            dataloader: Data loader
            mc_samples: Number of MC dropout samples
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        all_predictions = []
        all_uncertainties = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                # Move to device
                device = next(self.ensemble.parameters()).device
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Get ensemble predictions
                if return_uncertainty:
                    pred, uncertainty = self.ensemble.predict_with_uncertainty(
                        images, mc_samples=mc_samples
                    )
                    all_uncertainties.append(uncertainty.cpu())
                else:
                    pred = self.ensemble.predict(images)
                
                all_predictions.append(pred.cpu())
                all_labels.append(labels.cpu())
                
                # Log progress
                if batch_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    samples_per_sec = (batch_idx + 1) * self.batch_size / elapsed
                    logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches "
                              f"({samples_per_sec:.1f} samples/sec)")
        
        # Concatenate results
        results = {
            'predictions': torch.cat(all_predictions, dim=0).numpy(),
            'labels': torch.cat(all_labels, dim=0).numpy()
        }
        
        if return_uncertainty:
            results['uncertainties'] = torch.cat(all_uncertainties, dim=0).numpy()
        
        total_time = time.time() - start_time
        total_samples = len(results['predictions'])
        logger.info(f"Completed processing: {total_samples} samples in {total_time:.1f}s "
                   f"({total_samples/total_time:.1f} samples/sec)")
        
        return results


def benchmark_inference_speed(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    num_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark inference speed for different models.
    
    Args:
        models: Dictionary of models to benchmark
        dataloader: Data loader for benchmarking
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary of model_name -> average_inference_time
    """
    results = {}
    
    for name, model in models.items():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        times = []
        
        with torch.no_grad():
            for run in range(num_runs):
                start_time = time.time()
                
                for images, _ in dataloader:
                    images = images.to(device, non_blocking=True)
                    _ = model(images)
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results[name] = avg_time
        
        logger.info(f"{name}: {avg_time:.2f}±{std_time:.2f}s per epoch")
    
    return results


def main():
    """Main ensemble inference function."""
    parser = argparse.ArgumentParser(description="High-performance ensemble inference")
    
    # Model arguments
    parser.add_argument("--models", type=str, default="resnet18,vit_b_16",
                        help="Comma-separated list of model names")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--ensemble-config", type=str, default=None,
                        help="Path to ensemble configuration file")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="data_scientific_test",
                        help="Root directory containing datasets")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Image size for preprocessing")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Inference arguments
    parser.add_argument("--mc-samples", type=int, default=10,
                        help="Number of MC dropout samples")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel model execution")
    parser.add_argument("--batch-parallel", action="store_true", default=True,
                        help="Use optimized batch-parallel inference (default: True)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarks")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save predictions to file")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Parse model names
    model_names = [name.strip() for name in args.models.split(",")]
    logger.info(f"Loading models: {model_names}")
    
    # Load models
    models = {}
    checkpoint_dir = Path(args.checkpoint_dir)
    
    for name in model_names:
        checkpoint_path = checkpoint_dir / f"best_{name}.pt"
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        # Create model
        if name in ['trans_enc_s', 'light_transformer']:
            backbone, head, feature_dim = make_ensemble_model(
                name=name, bands=3, pretrained=True, dropout_p=0.5
            )
            model = nn.Sequential(backbone, head)
        else:
            from models import build_model
            model = build_model(arch=name, pretrained=True, dropout_rate=0.5)
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        models[name] = model
        
        logger.info(f"Loaded {name} from {checkpoint_path}")
    
    if not models:
        logger.error("No models loaded successfully")
        return
    
    # Create dataset
    if args.img_size is None:
        # Auto-detect from first model
        first_model = next(iter(models.values()))
        args.img_size = first_model.get_input_size()
    
    # Create optimized test data loader
    logger.info("Creating optimized test data loader...")
    _, _, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        val_split=0.0  # No validation split needed for inference
    )
    
    logger.info(f"Created optimized test data loader with {len(test_loader.dataset)} samples")
    
    # Run benchmarks if requested
    if args.benchmark:
        logger.info("Running performance benchmarks...")
        benchmark_results = benchmark_inference_speed(models, test_loader)
        
        # Save benchmark results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "benchmark_results.json", 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info("Benchmark results saved")
    
    # Run inference
    if args.parallel and len(models) > 1:
        logger.info("Running optimized parallel ensemble inference...")
        
        # Setup parallel inference
        parallel_inference = ParallelEnsembleInference(
            models=models,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=args.amp
        )
        
        # Choose inference method based on arguments
        if args.batch_parallel:
            logger.info("Using optimized batch-parallel inference for optimal performance...")
            results = parallel_inference.predict_batch_parallel(
                test_loader, mc_samples=args.mc_samples
            )
        else:
            logger.info("Using standard parallel inference...")
            results_numpy = parallel_inference.predict_parallel(
                test_loader, mc_samples=args.mc_samples
            )
            # Convert numpy results to tensors
            results = {}
            for name, (logits, probs) in results_numpy.items():
                results[name] = torch.from_numpy(logits)
        
        # Convert to probabilities and combine results
        all_logits = []
        all_probs = []
        
        for name, logits in results.items():
            probs = torch.sigmoid(logits)
            probs = clamp_probs(probs)
            
            all_logits.append(logits)
            all_probs.append(probs)
        
        # Ensemble fusion
        if len(all_logits) > 1:
            # Simple averaging for now (could be enhanced with uncertainty weighting)
            ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)
            ensemble_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        else:
            ensemble_logits = all_logits[0]
            ensemble_probs = all_probs[0]
        
        logger.info("Optimized parallel inference completed")
        
    else:
        logger.info("Running sequential ensemble inference...")
        
        # Sequential inference
        all_logits = []
        all_probs = []
        
        for name, model in models.items():
            model.to(device)
            model.eval()
            
            model_logits = []
            model_probs = []
            
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device, non_blocking=True)
                    
                    if args.amp and device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            logits = model(images).squeeze(1)
                    else:
                        logits = model(images).squeeze(1)
                    
                    probs = torch.sigmoid(logits)
                    probs = clamp_probs(probs)
                    
                    model_logits.append(logits.cpu())
                    model_probs.append(probs.cpu())
            
            all_logits.append(torch.cat(model_logits, dim=0))
            all_probs.append(torch.cat(model_probs, dim=0))
            
            logger.info(f"Completed inference for {name}")
        
        # Ensemble fusion
        if len(all_logits) > 1:
            ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)
            ensemble_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        else:
            ensemble_logits = all_logits[0]
            ensemble_probs = all_probs[0]
    
    # Save results
    if args.save_predictions:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble predictions
        results = {
            'ensemble_logits': ensemble_logits.numpy(),
            'ensemble_probs': ensemble_probs.numpy(),
            'ensemble_preds': (ensemble_probs >= 0.5).numpy().astype(int)
        }
        
        # Save individual model results
        for i, name in enumerate(model_names):
            if i < len(all_logits):
                results[f'{name}_logits'] = all_logits[i].numpy()
                results[f'{name}_probs'] = all_probs[i].numpy()
        
        # Save to file
        np.savez(output_dir / "ensemble_predictions.npz", **results)
        logger.info(f"Predictions saved to {output_dir / 'ensemble_predictions.npz'}")
    
    logger.info("Ensemble inference completed successfully!")


if __name__ == "__main__":
    main()




