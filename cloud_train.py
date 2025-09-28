#!/usr/bin/env python3
"""
cloud_train.py
==============
Automated cloud training setup for ViT models.

This script helps you easily deploy your training to cloud platforms
with minimal setup and cost optimization.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

def create_colab_notebook():
    """Create a Google Colab notebook for ViT training."""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Gravitational Lens Classification with ViT\n",
                    "## Automated training setup for Vision Transformer"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check GPU availability\n",
                    "import torch\n",
                    "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                    "    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install dependencies\n",
                    "!pip install torch torchvision scikit-learn pillow pandas numpy tqdm"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Upload your dataset\n",
                    "from google.colab import files\n",
                    "import zipfile\n",
                    "\n",
                    "print('Please upload your data_realistic_test.zip file:')\n",
                    "uploaded = files.upload()\n",
                    "\n",
                    "# Extract dataset\n",
                    "for filename in uploaded.keys():\n",
                    "    if filename.endswith('.zip'):\n",
                    "        with zipfile.ZipFile(filename, 'r') as zip_ref:\n",
                    "            zip_ref.extractall('.')\n",
                    "        print(f'Extracted {filename}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create simplified training script\n",
                    "training_code = '''\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.optim as optim\n",
                    "from torch.utils.data import DataLoader, Dataset\n",
                    "import torchvision.transforms as T\n",
                    "import torchvision.models as models\n",
                    "from torchvision.models import ViT_B_16_Weights\n",
                    "from PIL import Image\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from pathlib import Path\n",
                    "from tqdm import tqdm\n",
                    "\n",
                    "# Your training code here...\n",
                    "# [This would include your models.py and training logic]\n",
                    "'''\n",
                    "\n",
                    "with open('train_vit_colab.py', 'w') as f:\n",
                    "    f.write(training_code)\n",
                    "\n",
                    "print('Training script created!')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run ViT training with GPU acceleration\n",
                    "!python train_vit_colab.py --epochs 10 --batch-size 32"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Download trained model\n",
                    "from google.colab import files\n",
                    "\n",
                    "# Download model weights\n",
                    "files.download('best_vit_b_16.pt')\n",
                    "\n",
                    "# Download training history\n",
                    "files.download('training_history_vit_b_16.json')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Save notebook
    notebook_path = Path("ViT_Training_Colab.ipynb")
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"✅ Created Google Colab notebook: {notebook_path}")
    print("📝 Next steps:")
    print("1. Upload this notebook to Google Colab")
    print("2. Zip your data_realistic_test folder")
    print("3. Upload the zip file when prompted")
    print("4. Run all cells to train ViT with free GPU!")

def create_aws_setup_script():
    """Create AWS deployment script."""
    
    aws_script = '''#!/bin/bash
# AWS EC2 Setup Script for ViT Training

echo "🚀 Setting up AWS EC2 for ViT training..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3-pip python3-venv git

# Create virtual environment
python3 -m venv vit_env
source vit_env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas numpy pillow tqdm

# Install AWS CLI for data transfer
sudo apt install -y awscli

echo "✅ Environment setup complete!"
echo "📁 Now upload your code and data:"
echo "   scp -i your-key.pem -r lens-demo/ ubuntu@your-instance:/home/ubuntu/"
echo "🏃 Then run training:"
echo "   python src/train.py --arch vit_b_16 --epochs 10 --batch-size 32"
'''
    
    script_path = Path("aws_setup.sh")
    with open(script_path, 'w') as f:
        f.write(aws_script)
    
    os.chmod(script_path, 0o755)  # Make executable
    
    print(f"✅ Created AWS setup script: {script_path}")
    print("📝 Next steps:")
    print("1. Launch AWS EC2 instance (g4dn.xlarge recommended)")
    print("2. Copy this script to your instance")
    print("3. Run: bash aws_setup.sh")
    print("4. Upload your code and data")
    print("5. Start training!")

def estimate_costs():
    """Provide cost estimates for different cloud options."""
    
    print("\n💰 CLOUD TRAINING COST ESTIMATES")
    print("=" * 50)
    
    costs = {
        "Google Colab (Free)": {
            "cost": "$0",
            "time": "2-4 hours",
            "gpu": "Tesla T4/K80",
            "limits": "12h sessions, may disconnect"
        },
        "Google Colab Pro": {
            "cost": "$10/month",
            "time": "1-2 hours", 
            "gpu": "Tesla V100/A100",
            "limits": "Priority access, longer sessions"
        },
        "AWS g4dn.xlarge": {
            "cost": "$0.526/hour",
            "time": "1-2 hours",
            "gpu": "NVIDIA T4",
            "limits": "Pay per use, full control"
        },
        "AWS Spot Instance": {
            "cost": "$0.15-0.20/hour",
            "time": "1-2 hours",
            "gpu": "NVIDIA T4",
            "limits": "70% cheaper, may be interrupted"
        }
    }
    
    for service, details in costs.items():
        print(f"\n{service}:")
        for key, value in details.items():
            print(f"  {key.title()}: {value}")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("• Start with Google Colab (FREE) for experimentation")
    print("• Use AWS Spot Instances for production training")
    print("• Budget: $1-2 for complete ViT training")

def create_data_package():
    """Create a zip file of the dataset for cloud upload."""
    
    import zipfile
    
    data_dir = Path("data_realistic_test")
    if not data_dir.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return
    
    zip_path = Path("data_realistic_test.zip")
    
    print(f"📦 Creating data package: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(data_dir.parent)
                zipf.write(file_path, arcname)
                
    file_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Created {zip_path} ({file_size:.1f} MB)")
    print("📤 Ready for cloud upload!")

def main():
    """Main function to handle cloud training setup."""
    
    parser = argparse.ArgumentParser(description="Cloud training setup for ViT")
    parser.add_argument("--platform", choices=["colab", "aws", "estimate", "package"], 
                       default="colab", help="Cloud platform to setup")
    
    args = parser.parse_args()
    
    print("☁️  CLOUD TRAINING SETUP FOR VIT")
    print("=" * 40)
    
    if args.platform == "colab":
        create_colab_notebook()
    elif args.platform == "aws":
        create_aws_setup_script()
    elif args.platform == "estimate":
        estimate_costs()
    elif args.platform == "package":
        create_data_package()
    
    print(f"\n🎉 Setup complete for {args.platform}!")

if __name__ == "__main__":
    main()
