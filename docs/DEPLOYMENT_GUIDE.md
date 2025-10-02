# 🚀 Cloud Deployment Guide

This guide provides detailed instructions for deploying the gravitational lens classification system to various cloud platforms.

## Table of Contents

- [Overview](#overview)
- [Google Colab Deployment](#google-colab-deployment)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [Google Cloud Platform](#google-cloud-platform)
- [Azure Machine Learning](#azure-machine-learning)
- [Cost Analysis](#cost-analysis)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)

## Overview

Cloud deployment is recommended for:
- **ViT training**: Requires significant computational resources
- **Large-scale experiments**: Multiple model variants and hyperparameter sweeps
- **Production inference**: Serving models at scale
- **Collaborative research**: Shared access to computational resources

### Deployment Options Comparison

| Platform | Cost | Setup Complexity | GPU Access | Best For |
|----------|------|------------------|------------|----------|
| **Google Colab** | Free/Low | Very Easy | Limited | Prototyping, ViT training |
| **AWS EC2** | Medium | Medium | Full | Production, scalability |
| **Google Cloud** | Medium | Medium | Full | Integration with GCP services |
| **Azure ML** | Medium | Easy | Full | Enterprise, MLOps |

## Google Colab Deployment

### Setup Instructions

1. **Generate Colab Notebook**
```bash
# From your local machine
python cloud_train.py --platform colab --data-root data_realistic_test
```

2. **Upload Data to Google Drive**
```bash
# Package your dataset
python cloud_train.py --platform package --data-root data_realistic_test

# This creates data_realistic_test.zip
# Upload this file to your Google Drive
```

3. **Open Generated Notebook**
- Open `train_ensemble_colab.ipynb` in Google Colab
- Ensure GPU runtime: Runtime → Change runtime type → GPU

### Complete Colab Notebook Template

```python
# =====================================================
# Gravitational Lens Classification - Google Colab
# =====================================================

# 1. Setup Runtime
!nvidia-smi  # Check GPU availability

# 2. Install Dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install scikit-learn pandas numpy matplotlib pillow tqdm

# 3. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 4. Extract Dataset
!unzip "/content/drive/MyDrive/data_realistic_test.zip" -d /content/

# 5. Clone Repository or Upload Code
!git clone https://github.com/Kantoration/mechine_lensing.git
%cd mechine_lensing

# Alternative: Upload src folder manually
# from google.colab import files
# uploaded = files.upload()  # Upload src.zip

# 6. Verify Setup
!ls -la src/
!python src/models.py  # Should show available architectures

# 7. Quick Test
!python src/train.py --arch resnet18 --data-root /content/data_realistic_test --epochs 2 --batch-size 32

# 8. Train ResNet-18 (Fast)
print("🚀 Training ResNet-18...")
!python src/train.py \
  --arch resnet18 \
  --data-root /content/data_realistic_test \
  --epochs 10 \
  --batch-size 32 \
  --pretrained

# 9. Train ViT-B/16 (GPU Required)
print("🚀 Training ViT-B/16...")
!python src/train.py \
  --arch vit_b_16 \
  --data-root /content/data_realistic_test \
  --epochs 10 \
  --batch-size 16 \
  --pretrained

# 10. Evaluate Individual Models
print("📊 Evaluating ResNet-18...")
!python src/eval.py \
  --arch resnet18 \
  --weights checkpoints/best_resnet18.pt \
  --data-root /content/data_realistic_test

print("📊 Evaluating ViT-B/16...")
!python src/eval.py \
  --arch vit_b_16 \
  --weights checkpoints/best_vit_b_16.pt \
  --data-root /content/data_realistic_test

# 11. Ensemble Evaluation
print("🤝 Evaluating Ensemble...")
!python src/eval_ensemble.py \
  --cnn-weights checkpoints/best_resnet18.pt \
  --vit-weights checkpoints/best_vit_b_16.pt \
  --data-root /content/data_realistic_test \
  --save-predictions

# 12. Download Results
from google.colab import files
import shutil

# Create results archive
!zip -r results_complete.zip checkpoints/ results/

# Download
files.download('results_complete.zip')

print("✅ Training and evaluation complete!")
print("📥 Results downloaded to your local machine")
```

### Colab Pro Benefits

- **Faster GPUs**: V100, A100 access
- **Longer sessions**: Up to 24 hours
- **Priority access**: Less queueing
- **More memory**: Up to 25GB RAM

## AWS EC2 Deployment

### Instance Selection

| Instance Type | vCPUs | Memory | GPU | Storage | Cost/Hour | Best For |
|---------------|-------|--------|-----|---------|-----------|----------|
| **t3.large** | 2 | 8 GB | None | EBS | $0.083 | ResNet training |
| **g4dn.xlarge** | 4 | 16 GB | T4 | 125 GB SSD | $0.526 | ViT training |
| **p3.2xlarge** | 8 | 61 GB | V100 | EBS | $3.06 | Large-scale experiments |

### Setup Instructions

1. **Launch Instance**
```bash
# Create key pair
aws ec2 create-key-pair --key-name lens-classification --query 'KeyMaterial' --output text > lens-classification.pem
chmod 400 lens-classification.pem

# Launch instance (Ubuntu 20.04 LTS)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type g4dn.xlarge \
  --key-name lens-classification \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx
```

2. **Connect and Setup**
```bash
# SSH into instance
ssh -i lens-classification.pem ubuntu@<instance-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv git -y

# Install NVIDIA drivers (for GPU instances)
sudo apt install nvidia-driver-470 -y
sudo reboot

# After reboot, verify GPU
nvidia-smi
```

3. **Install CUDA and PyTorch**
```bash
# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA
nvcc --version
```

4. **Setup Project**
```bash
# Clone repository
git clone https://github.com/Kantoration/mechine_lensing.git
cd mechine_lensing

# Create virtual environment
python3 -m venv lens_env
source lens_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Generate dataset
python src/make_dataset_scientific.py --config configs/realistic.yaml --out data_realistic

# Train models
python src/train.py --arch resnet18 --data-root data_realistic --epochs 10
python src/train.py --arch vit_b_16 --data-root data_realistic --epochs 10

# Evaluate ensemble
python src/eval_ensemble.py \
  --cnn-weights checkpoints/best_resnet18.pt \
  --vit-weights checkpoints/best_vit_b_16.pt \
  --data-root data_realistic
```

5. **Automated Setup Script**
```bash
#!/bin/bash
# setup_aws.sh - Automated AWS setup script

set -e

echo "🚀 Setting up Gravitational Lens Classification on AWS..."

# System updates
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git wget -y

# NVIDIA drivers (for GPU instances)
if lspci | grep -i nvidia > /dev/null; then
    echo "📦 Installing NVIDIA drivers..."
    sudo apt install nvidia-driver-470 -y
    
    # Install CUDA
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
    
    # Update PATH
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Clone project
git clone https://github.com/Kantoration/mechine_lensing.git
cd mechine_lensing

# Setup Python environment
python3 -m venv lens_env
source lens_env/bin/activate

# Install PyTorch (GPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install scikit-learn pandas numpy matplotlib pillow tqdm pyyaml

echo "✅ Setup complete! Reboot if GPU instance, then activate environment:"
echo "source mechine_lensing/lens_env/bin/activate"
```

### Cost Optimization

1. **Use Spot Instances**
```bash
# Request spot instance (up to 70% cheaper)
aws ec2 request-spot-instances \
  --spot-price "0.15" \
  --instance-count 1 \
  --launch-specification '{
    "ImageId": "ami-0c02fb55956c7d316",
    "InstanceType": "g4dn.xlarge",
    "KeyName": "lens-classification",
    "SecurityGroupIds": ["sg-xxxxxxxx"]
  }'
```

2. **Auto-Shutdown Script**
```bash
#!/bin/bash
# auto_shutdown.sh - Prevent runaway costs

# Train models with timeout
timeout 2h python src/train.py --arch vit_b_16 --data-root data_realistic --epochs 20

# Sync results to S3
aws s3 sync checkpoints/ s3://your-bucket/checkpoints/
aws s3 sync results/ s3://your-bucket/results/

# Shutdown instance
sudo shutdown -h now
```

## Google Cloud Platform

### Setup with AI Platform

1. **Create Project and Enable APIs**
```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable ml.googleapis.com
```

2. **Create VM Instance**
```bash
# Create GPU instance
gcloud compute instances create lens-classifier \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --maintenance-policy=TERMINATE
```

3. **SSH and Setup**
```bash
# SSH into instance
gcloud compute ssh lens-classifier --zone=us-central1-a

# Clone and setup (similar to AWS)
git clone https://github.com/Kantoration/mechine_lensing.git
cd mechine_lensing
# ... rest of setup
```

## Azure Machine Learning

### Setup with Azure ML Studio

1. **Create Workspace**
```python
# setup_azure.py
from azureml.core import Workspace, Environment, ScriptRunConfig, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute

# Create workspace
ws = Workspace.create(
    name='lens-classification',
    subscription_id='your-subscription-id',
    resource_group='lens-rg',
    location='eastus'
)

# Create compute cluster
compute_config = AmlCompute.provisioning_configuration(
    vm_size='Standard_NC6',  # GPU instance
    max_nodes=1
)

compute_target = ComputeTarget.create(ws, 'gpu-cluster', compute_config)
```

2. **Submit Training Job**
```python
# Create environment
env = Environment.from_pip_requirements('lens-env', 'requirements.txt')

# Configure run
config = ScriptRunConfig(
    source_directory='src',
    script='train.py',
    arguments=['--arch', 'vit_b_16', '--epochs', '10'],
    compute_target=compute_target,
    environment=env
)

# Submit experiment
experiment = Experiment(ws, 'lens-classification')
run = experiment.submit(config)
```

## Cost Analysis

### Estimated Training Costs (USD)

| Platform | Instance Type | ResNet-18 (5 min) | ViT-B/16 (30 min) | Full Pipeline (45 min) |
|----------|---------------|--------------------|--------------------|------------------------|
| **Google Colab** | Free Tier | $0.00 | $0.00 | $0.00 |
| **Google Colab Pro** | Premium | $0.00* | $0.00* | $0.00* |
| **AWS EC2** | g4dn.xlarge | $0.04 | $0.26 | $0.39 |
| **AWS EC2 Spot** | g4dn.xlarge | $0.01 | $0.08 | $0.12 |
| **GCP** | n1-standard-4 + T4 | $0.05 | $0.30 | $0.45 |
| **Azure** | Standard_NC6 | $0.06 | $0.36 | $0.54 |

*Monthly subscription: $10/month

### Cost Optimization Strategies

1. **Use Free Tiers First**
   - Google Colab: Free GPU access with limitations
   - AWS: Free tier includes 750 hours of t2.micro
   - GCP: $300 credit for new users

2. **Spot/Preemptible Instances**
   - AWS Spot: Up to 70% discount
   - GCP Preemptible: Up to 80% discount
   - Risk: Can be terminated anytime

3. **Right-Size Instances**
   - ResNet-18: CPU instances sufficient
   - ViT-B/16: GPU required
   - Ensemble: Train separately, combine locally

4. **Data Transfer Optimization**
   - Use cloud storage in same region
   - Compress datasets before upload
   - Stream data instead of downloading

## Performance Benchmarks

### Training Time Comparison

| Model | Local CPU (Laptop) | Colab GPU (T4) | AWS GPU (T4) | AWS GPU (V100) |
|-------|-------------------|----------------|--------------|----------------|
| **ResNet-18** | 4 min | 1 min | 1 min | 30 sec |
| **ViT-B/16** | 45 min | 8 min | 8 min | 3 min |
| **Ensemble** | 49 min | 9 min | 9 min | 3.5 min |

### Memory Usage

| Model | Peak Memory | Recommended RAM |
|-------|-------------|-----------------|
| **ResNet-18** | 2 GB | 4 GB |
| **ViT-B/16** | 6 GB | 8 GB |
| **Ensemble** | 8 GB | 12 GB |

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
```python
# Reduce batch size
python src/train.py --arch vit_b_16 --batch-size 8  # Instead of 16

# Enable gradient checkpointing
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)
```

2. **Slow Data Loading**
```python
# Increase number of workers
python src/train.py --num-workers 4

# Use faster storage (SSD vs HDD)
# Store data on instance storage, not network storage
```

3. **Instance Termination**
```bash
# Save checkpoints frequently
python src/train.py --save-every 5  # Save every 5 epochs

# Use screen/tmux for persistent sessions
screen -S training
python src/train.py --arch vit_b_16 --epochs 20
# Ctrl+A, D to detach
# screen -r training to reattach
```

4. **Network Issues**
```bash
# Download models locally first
python -c "import torchvision.models as models; models.vit_b_16(pretrained=True)"

# Use local pip cache
pip install --cache-dir ./pip_cache torch torchvision
```

### Monitoring and Alerts

1. **Cost Monitoring**
```bash
# AWS CloudWatch billing alert
aws cloudwatch put-metric-alarm \
  --alarm-name "High-Billing" \
  --alarm-description "Alert when billing exceeds $10" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold 10.0 \
  --comparison-operator GreaterThanThreshold
```

2. **Training Monitoring**
```python
# Simple progress tracking
import time
import psutil

def log_system_stats():
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"Memory: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
```

This deployment guide provides comprehensive instructions for running the gravitational lens classification system on various cloud platforms, with cost optimization and troubleshooting guidance.





