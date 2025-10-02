#!/usr/bin/env python3
"""
Path utilities for consistent project path resolution.

This module provides standardized path resolution across the entire project,
eliminating the need for scattered sys.path.insert() calls.
"""

import sys
from pathlib import Path
from typing import Optional


def setup_project_paths(script_path: Optional[Path] = None) -> Path:
    """
    Set up project paths for consistent imports.
    
    Args:
        script_path: Path to the script calling this function. If None, tries to use caller's __file__
        
    Returns:
        Path to the project root directory
        
    Example:
        # In any script file:
        from src.utils.path_utils import setup_project_paths
        project_root = setup_project_paths()
        
        # Now you can import from src modules:
        from models.unified_factory import create_model
    """
    if script_path is None:
        # Try to get the caller's file path
        try:
            import inspect
            frame = inspect.currentframe().f_back
            if '__file__' in frame.f_globals:
                script_path = Path(frame.f_globals['__file__'])
            else:
                # Fallback: use current working directory
                script_path = Path.cwd()
        except (AttributeError, KeyError):
            # Fallback: use current working directory
            script_path = Path.cwd()
    
    # Find project root (directory containing src/)
    current_path = script_path.resolve()
    
    # Walk up the directory tree to find project root
    while current_path.parent != current_path:  # Not at filesystem root
        src_dir = current_path / "src"
        if src_dir.exists() and src_dir.is_dir():
            # Found project root
            project_root = current_path
            src_path = src_dir
            
            # Add src to Python path if not already there
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            return project_root
        
        current_path = current_path.parent
    
    raise RuntimeError(f"Could not find project root (directory containing 'src/') starting from {script_path}")


def get_project_root() -> Path:
    """
    Get the project root directory without modifying sys.path.
    
    Returns:
        Path to the project root directory
    """
    # Find project root from current file location
    current_path = Path(__file__).resolve()
    
    # Walk up to find project root
    while current_path.parent != current_path:
        src_dir = current_path / "src"
        if src_dir.exists() and src_dir.is_dir():
            return current_path
        current_path = current_path.parent
    
    raise RuntimeError("Could not find project root")


def get_data_dir() -> Path:
    """
    Get the data directory path.
    
    Returns:
        Path to the data directory
    """
    return get_project_root() / "data"


def get_config_dir() -> Path:
    """
    Get the config directory path.
    
    Returns:
        Path to the config directory
    """
    return get_project_root() / "configs"


def get_results_dir() -> Path:
    """
    Get the results directory path.
    
    Returns:
        Path to the results directory
    """
    return get_project_root() / "results"


def get_checkpoints_dir() -> Path:
    """
    Get the checkpoints directory path.
    
    Returns:
        Path to the checkpoints directory
    """
    return get_project_root() / "checkpoints"


# Convenience function for backward compatibility
def add_src_to_path():
    """Add src directory to Python path (legacy function)."""
    setup_project_paths()
