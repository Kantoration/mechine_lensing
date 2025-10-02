#!/usr/bin/env python3
"""
Unified Configuration System
===========================

Single configuration system for the entire project, replacing scattered
configuration files and hardcoded parameters.

Key Features:
- YAML-based configuration
- Environment variable overrides
- Validation and type checking
- Auto-tuning based on system capabilities
- Backward compatibility with existing configs
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import torch

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data loading configuration."""
    root: str = "data/processed/data_realistic_test"
    batch_size: int = 64
    img_size: int = 224
    val_split: float = 0.1
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    use_cache: bool = True
    cache_dir: Optional[str] = None
    validate_paths: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    type: str = "single"  # "single", "ensemble", "physics_informed"
    architecture: str = "resnet18"
    architectures: Optional[list] = None
    bands: int = 3
    pretrained: bool = True
    dropout_p: float = 0.2
    
    # Ensemble specific
    ensemble_strategy: str = "uncertainty_weighted"
    physics_weight: float = 0.1
    uncertainty_estimation: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Performance optimizations
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    gradient_clip_val: float = 1.0
    
    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = "checkpoints"
    best_model_name: str = "best_model.pt"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    save_predictions: bool = True
    plot_results: bool = True
    output_dir: str = "results"
    detailed_metrics: bool = True
    
    # Physics analysis
    physics_analysis: bool = True
    attention_analysis: bool = True


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "auto"
    seed: int = 42
    deterministic: bool = False
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(message)s"
    file: Optional[str] = None
    console: bool = True


@dataclass
class MonitoringConfig:
    """Performance monitoring configuration."""
    benchmark_dataloader: bool = True
    benchmark_model_creation: bool = True
    profile_training: bool = False
    memory_monitoring: bool = True


@dataclass
class UnifiedConfig:
    """Unified configuration for the entire project."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def __post_init__(self):
        """Post-initialization processing and validation."""
        self._apply_environment_overrides()
        self._auto_tune_system_parameters()
        self._validate_configuration()
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Data overrides
        if os.getenv('DATA_ROOT'):
            self.data.root = os.getenv('DATA_ROOT')
        if os.getenv('BATCH_SIZE'):
            self.data.batch_size = int(os.getenv('BATCH_SIZE'))
        if os.getenv('IMG_SIZE'):
            self.data.img_size = int(os.getenv('IMG_SIZE'))
        
        # Model overrides
        if os.getenv('MODEL_ARCH'):
            self.model.architecture = os.getenv('MODEL_ARCH')
        if os.getenv('PRETRAINED'):
            self.model.pretrained = os.getenv('PRETRAINED').lower() == 'true'
        
        # Training overrides
        if os.getenv('EPOCHS'):
            self.training.epochs = int(os.getenv('EPOCHS'))
        if os.getenv('LEARNING_RATE'):
            self.training.learning_rate = float(os.getenv('LEARNING_RATE'))
        
        # System overrides
        if os.getenv('DEVICE'):
            self.system.device = os.getenv('DEVICE')
        if os.getenv('SEED'):
            self.system.seed = int(os.getenv('SEED'))
    
    def _auto_tune_system_parameters(self):
        """Auto-tune system parameters based on hardware."""
        import os
        
        # Auto-tune number of workers
        if self.data.num_workers is None:
            cpu_count = os.cpu_count() or 1
            self.data.num_workers = min(8, max(1, int(cpu_count * 0.75)))
            logger.info(f"Auto-tuned num_workers: {self.data.num_workers}")
        
        if self.system.num_workers is None:
            self.system.num_workers = self.data.num_workers
        
        # Auto-tune pin_memory
        if self.data.pin_memory is None:
            self.data.pin_memory = torch.cuda.is_available()
            logger.info(f"Auto-tuned pin_memory: {self.data.pin_memory}")
        
        if self.system.pin_memory is None:
            self.system.pin_memory = self.data.pin_memory
        
        # Auto-tune device
        if self.system.device == "auto":
            self.system.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-tuned device: {self.system.device}")
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        # Validate data config
        if self.data.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.data.img_size <= 0:
            raise ValueError("img_size must be positive")
        if not 0 < self.data.val_split < 1:
            raise ValueError("val_split must be between 0 and 1")
        
        # Validate model config
        if self.model.type not in ["single", "ensemble", "physics_informed"]:
            raise ValueError(f"Unknown model type: {self.model.type}")
        
        if self.model.type in ["ensemble", "physics_informed"] and not self.model.architectures:
            self.model.architectures = [self.model.architecture]
        
        # Validate training config
        if self.training.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'system': self.system.__dict__,
            'logging': self.logging.__dict__,
            'monitoring': self.monitoring.__dict__
        }
    
    def save(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {path}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> UnifiedConfig:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        UnifiedConfig instance
    """
    if config_path is None:
        # Try to find default config
        default_paths = [
            "configs/unified.yaml",
            "config.yaml",
            "config.yml"
        ]
        
        for path in default_paths:
            if Path(path).exists():
                config_path = path
                break
        
        if config_path is None:
            logger.info("No configuration file found, using defaults")
            return UnifiedConfig()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configuration objects
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
    system_config = SystemConfig(**config_dict.get('system', {}))
    logging_config = LoggingConfig(**config_dict.get('logging', {}))
    monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
    
    return UnifiedConfig(
        data=data_config,
        model=model_config,
        training=training_config,
        evaluation=evaluation_config,
        system=system_config,
        logging=logging_config,
        monitoring=monitoring_config
    )


def create_config_from_args(args) -> UnifiedConfig:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        UnifiedConfig instance
    """
    # Map command line arguments to configuration
    data_config = DataConfig(
        root=getattr(args, 'data_root', 'data/processed/data_realistic_test'),
        batch_size=getattr(args, 'batch_size', 64),
        img_size=getattr(args, 'img_size', 224),
        val_split=getattr(args, 'val_split', 0.1),
        num_workers=getattr(args, 'num_workers', None),
        use_cache=getattr(args, 'use_cache', True)
    )
    
    model_config = ModelConfig(
        type=getattr(args, 'model_type', 'single'),
        architecture=getattr(args, 'arch', 'resnet18'),
        pretrained=getattr(args, 'pretrained', True),
        dropout_p=getattr(args, 'dropout_rate', 0.2)
    )
    
    training_config = TrainingConfig(
        epochs=getattr(args, 'epochs', 20),
        learning_rate=getattr(args, 'lr', 1e-3),
        weight_decay=getattr(args, 'weight_decay', 1e-4),
        checkpoint_dir=getattr(args, 'checkpoint_dir', 'checkpoints')
    )
    
    system_config = SystemConfig(
        device=getattr(args, 'device', 'auto'),
        seed=getattr(args, 'seed', 42)
    )
    
    return UnifiedConfig(
        data=data_config,
        model=model_config,
        training=training_config,
        system=system_config
    )


# Convenience functions for backward compatibility
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file (backward compatibility)."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file (backward compatibility)."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


