#!/usr/bin/env python3
"""
CLI Smoke Tests for Training Scripts
"""

import unittest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCLISmoke(unittest.TestCase):
    """Test that CLI argument parsing works for all training scripts."""
    
    def test_trainer_cli_help(self):
        """Test that trainer.py can parse --help without crashing."""
        try:
            from src.training.trainer import main
            original_argv = sys.argv.copy()
            sys.argv = ['trainer.py', '--help']
            
            try:
                main()
            except SystemExit:
                pass  # Expected for --help
            
            sys.argv = original_argv
            
        except Exception as e:
            self.fail(f"trainer.py CLI parsing failed: {e}")
    
    def test_accelerated_trainer_cli_help(self):
        """Test that accelerated_trainer.py can parse --help without crashing."""
        try:
            from src.training.accelerated_trainer import main
            original_argv = sys.argv.copy()
            sys.argv = ['accelerated_trainer.py', '--help']
            
            try:
                main()
            except SystemExit:
                pass  # Expected for --help
            
            sys.argv = original_argv
            
        except Exception as e:
            self.fail(f"accelerated_trainer.py CLI parsing failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)