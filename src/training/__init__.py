"""
Training utilities and trainers for gravitational lens classification.
"""

# Import from common module (new architecture)
try:
    from .common import (
        BaseTrainer,
        PerformanceMixin,
        PerformanceMonitor,
        create_optimized_dataloaders,
    )
    from .accelerated_trainer_refactored import AcceleratedTrainer
    from .multi_scale_trainer_refactored import (
        MultiScaleTrainer,
        ProgressiveMultiScaleTrainer,
    )

    __all__ = [
        "BaseTrainer",
        "PerformanceMixin",
        "PerformanceMonitor",
        "create_optimized_dataloaders",
        "AcceleratedTrainer",
        "MultiScaleTrainer",
        "ProgressiveMultiScaleTrainer",
    ]
except ImportError:
    # Fallback to old imports if PyTorch not available
    __all__ = []
