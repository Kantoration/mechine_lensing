#!/usr/bin/env python3
"""
Test runner for unified model factory tests.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py -v                 # Verbose output
    python run_tests.py test_unified_factory.py  # Run specific test file
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run tests with pytest."""
    # Setup project paths using centralized utility
    from src.utils.path_utils import setup_project_paths
    setup_project_paths()
    
    # Run pytest with the provided arguments
    cmd = ["python", "-m", "pytest"] + sys.argv[1:] + ["tests/"]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest")
        return 1


if __name__ == "__main__":
    sys.exit(main())


