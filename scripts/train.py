#!/usr/bin/env python3
"""
Main training script entry point.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from training.trainer import main

if __name__ == "__main__":
    main()
