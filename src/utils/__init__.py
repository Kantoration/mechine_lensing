"""
Utility functions for gravitational lens classification.
"""

from .config import load_config, validate_config
from .numerical import (
    clamp_probs, clamp_variances, stable_log_sigmoid,
    inverse_variance_weights, ensemble_logit_fusion
)
from .benchmark import (
    BenchmarkSuite, PerformanceMetrics, profile_training,
    profile_inference, benchmark_ensemble
)
from .device_utils import (
    get_device, move_model_to_device, move_batch_to_device,
    get_device_info, clear_device_cache, batch_cpu_transfer,
    batch_numpy_conversion, memory_efficient_visualization
)
from .seed_utils import (
    set_seed, get_current_seed, is_deterministic, reset_seeds,
    SeedContext, with_seed
)
from .logging_utils import (
    setup_logging, get_logger, set_log_level, enable_file_logging,
    disable_file_logging
)

__all__ = [
    # Config utilities
    'load_config',
    'validate_config',

    # Numerical utilities
    'clamp_probs',
    'clamp_variances',
    'stable_log_sigmoid',
    'inverse_variance_weights',
    'ensemble_logit_fusion',

    # Benchmark utilities
    'BenchmarkSuite',
    'PerformanceMetrics',
    'profile_training',
    'profile_inference',
    'benchmark_ensemble',

    # Device utilities
    'get_device',
    'move_model_to_device',
    'move_batch_to_device',
    'get_device_info',
    'clear_device_cache',
    'batch_cpu_transfer',
    'batch_numpy_conversion',
    'memory_efficient_visualization',

    # Seed utilities
    'set_seed',
    'get_current_seed',
    'is_deterministic',
    'reset_seeds',
    'SeedContext',
    'with_seed',

    # Logging utilities
    'setup_logging',
    'get_logger',
    'set_log_level',
    'enable_file_logging',
    'disable_file_logging'
]
