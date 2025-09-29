#!/usr/bin/env python3
"""
Configuration utilities for gravitational lens classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['General', 'LensArcs', 'GalaxyBlob', 'Output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate General section
    general = config['General']
    if general.get('n_train', 0) <= 0:
        raise ValueError("n_train must be positive")
    if general.get('n_test', 0) <= 0:
        raise ValueError("n_test must be positive")
    if not (0.0 <= general.get('balance', 0.5) <= 1.0):
        raise ValueError("balance must be between 0 and 1")
    
    logger.info("Configuration validation passed")


def save_config(config: Dict[str, Any], output_path: str | Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved configuration to: {output_path}")
