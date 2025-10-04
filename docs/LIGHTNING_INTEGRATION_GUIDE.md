# ⚡ Lightning AI Integration Guide

This guide explains how to use Lightning AI with the gravitational lens classification project for cloud GPU training and dataset streaming.

## 🚀 Quick Start

### 1. Install Lightning Dependencies

```bash
# Install Lightning AI and related packages
pip install "pytorch-lightning>=2.4" lightning "torchmetrics>=1.4" "fsspec[s3,gcs]" webdataset

# Or install from requirements
pip install -r requirements.txt
```

### 2. Local Training with Lightning

```bash
# Train ResNet-18 locally
python src/lit_train.py --data-root data/processed --arch resnet18 --epochs 20

# Train with multiple GPUs
python src/lit_train.py --data-root data/processed --arch vit_b_16 --devices 2 --strategy ddp

# Train ensemble model
python src/lit_train.py --data-root data/processed --model-type ensemble --devices 2
```

### 3. Cloud Training with Lightning

```bash
# Prepare dataset for cloud streaming
python scripts/prepare_lightning_dataset.py --data-root data/processed --output-dir shards --cloud-url s3://your-bucket/lens-data

# Train on cloud with WebDataset
python src/lit_train.py --use-webdataset --train-urls "s3://your-bucket/lens-data/train-{0000..0099}.tar" --val-urls "s3://your-bucket/lens-data/val-{0000..0009}.tar" --devices 4 --accelerator gpu
```

## 📁 Project Structure

```
demo/lens-demo/
├── src/
│   ├── lit_system.py          # LightningModule wrappers
│   ├── lit_datamodule.py      # LightningDataModule wrappers
│   ├── lit_train.py           # Lightning training script
│   └── ...
├── configs/
│   ├── lightning_train.yaml   # Local training config
│   ├── lightning_cloud.yaml   # Cloud training config
│   └── lightning_ensemble.yaml # Ensemble training config
├── scripts/
│   └── prepare_lightning_dataset.py # Dataset preparation
└── docs/
    └── LIGHTNING_INTEGRATION_GUIDE.md # This guide
```

## 🔧 Lightning Components

### 1. LightningModule (`LitLensSystem`)

The `LitLensSystem` class wraps your existing models in a Lightning-compatible interface:

```python
from src.lit_system import LitLensSystem

# Create model
model = LitLensSystem(
    arch="resnet18",
    lr=3e-4,
    weight_decay=1e-4,
    dropout_rate=0.5,
    pretrained=True
)

# Lightning handles training, validation, and testing automatically
```

**Key Features:**
- Automatic GPU/CPU handling
- Built-in metrics (accuracy, precision, recall, F1, AUROC, AP)
- Learning rate scheduling
- Model compilation support
- Uncertainty quantification

### 2. LightningDataModule (`LensDataModule`)

The `LensDataModule` handles data loading for both local and cloud datasets:

```python
from src.lit_datamodule import LensDataModule

# Local dataset
datamodule = LensDataModule(
    data_root="data/processed",
    batch_size=64,
    num_workers=8,
    image_size=224
)

# WebDataset (cloud streaming)
datamodule = LensWebDatasetDataModule(
    train_urls="s3://bucket/train-{0000..0099}.tar",
    val_urls="s3://bucket/val-{0000..0009}.tar",
    batch_size=64,
    num_workers=16
)
```

**Key Features:**
- Local and cloud dataset support
- WebDataset streaming for large datasets
- Automatic data augmentation
- Optimized data loading for cloud instances

### 3. Lightning Trainer

The Lightning Trainer provides automatic scaling and optimization:

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    max_epochs=30,
    devices=4,                    # Use 4 GPUs
    accelerator="gpu",            # GPU training
    precision="bf16-mixed",       # Mixed precision
    strategy="ddp",               # Distributed training
    enable_checkpointing=True,    # Automatic checkpointing
    logger=True                   # Automatic logging
)
```

## ☁️ Cloud Training Setup

### 1. Lightning Cloud (Recommended)

Lightning Cloud provides managed GPU instances with automatic scaling:

```bash
# Install Lightning CLI
pip install lightning

# Login to Lightning Cloud
lightning login

# Create a workspace
lightning create workspace lens-training

# Run training job
lightning run app src/lit_train.py --use-webdataset --train-urls "s3://bucket/train-{0000..0099}.tar" --devices 4
```

### 2. AWS EC2 Setup

For custom AWS instances:

```bash
# Launch EC2 instance with GPU
# Install dependencies
pip install -r requirements.txt

# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Run training
python src/lit_train.py --use-webdataset --train-urls "s3://bucket/train-{0000..0099}.tar" --devices 1
```

### 3. Google Colab

For free GPU training:

```python
# Install dependencies
!pip install pytorch-lightning torchmetrics webdataset

# Upload your dataset or use cloud storage
# Run training
!python src/lit_train.py --data-root /content/data --arch resnet18 --epochs 10
```

## 📊 Dataset Preparation

### 1. Create WebDataset Shards

Convert your local dataset to WebDataset format for cloud streaming:

```bash
# Create shards from local dataset
python scripts/prepare_lightning_dataset.py \
    --data-root data/processed \
    --output-dir shards \
    --shard-size 1000 \
    --image-size 224 \
    --quality 95

# Upload to cloud storage
python scripts/prepare_lightning_dataset.py \
    --data-root data/processed \
    --output-dir shards \
    --cloud-url s3://your-bucket/lens-data \
    --upload-only
```

### 2. WebDataset Format

WebDataset shards contain compressed images and labels:

```
shard-0000.tar
├── 000000.jpg    # Compressed image
├── 000000.cls    # Label (0 or 1)
├── 000001.jpg
├── 000001.cls
└── ...
```

### 3. Cloud Storage Setup

Configure cloud storage for your datasets:

**AWS S3:**
```bash
# Set credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Upload dataset
aws s3 sync shards/ s3://your-bucket/lens-data/
```

**Google Cloud Storage:**
```bash
# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Upload dataset
gsutil -m cp -r shards/* gs://your-bucket/lens-data/
```

## 🎯 Training Configurations

### 1. Local Training

```yaml
# configs/lightning_train.yaml
model:
  arch: "resnet18"
  model_type: "single"
  pretrained: true
  dropout_rate: 0.5

training:
  epochs: 30
  batch_size: 64
  learning_rate: 3e-4

hardware:
  devices: 1
  accelerator: "auto"
  precision: "16-mixed"
```

### 2. Cloud Training

```yaml
# configs/lightning_cloud.yaml
model:
  arch: "vit_b_16"
  compile_model: true

training:
  epochs: 50
  batch_size: 128

hardware:
  devices: 4
  accelerator: "gpu"
  precision: "bf16-mixed"
  strategy: "ddp"

data:
  use_webdataset: true
  train_urls: "s3://bucket/train-{0000..0099}.tar"
  num_workers: 16
```

### 3. Ensemble Training

```yaml
# configs/lightning_ensemble.yaml
model:
  model_type: "ensemble"
  architectures: ["resnet18", "resnet34", "vit_b_16"]
  ensemble_strategy: "uncertainty_weighted"

training:
  epochs: 40
  batch_size: 64

hardware:
  devices: 2
  strategy: "ddp"
```

## 📈 Monitoring and Logging

### 1. Built-in Loggers

Lightning provides multiple logging options:

```python
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

# CSV logging (always enabled)
csv_logger = CSVLogger("logs", name="csv")

# TensorBoard logging
tb_logger = TensorBoardLogger("logs", name="tensorboard")

# Weights & Biases logging
wandb_logger = WandbLogger(project="lens-classification")

# Use multiple loggers
trainer = Trainer(logger=[csv_logger, tb_logger, wandb_logger])
```

### 2. Metrics Tracking

Automatic metrics tracking:

- **Training**: loss, accuracy, precision, recall, F1
- **Validation**: loss, accuracy, precision, recall, F1, AUROC, AP
- **Test**: loss, accuracy, precision, recall, F1, AUROC, AP

### 3. Checkpointing

Automatic model checkpointing:

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="{epoch:02d}-{val/auroc:.4f}",
    monitor="val/auroc",
    mode="max",
    save_top_k=3
)
```

## 🔄 Ensemble Training

### 1. Individual Model Training

Train multiple models separately:

```bash
# Train ResNet-18
python src/lit_train.py --arch resnet18 --checkpoint-dir checkpoints/resnet18

# Train ResNet-34
python src/lit_train.py --arch resnet34 --checkpoint-dir checkpoints/resnet34

# Train ViT
python src/lit_train.py --arch vit_b_16 --checkpoint-dir checkpoints/vit
```

### 2. Ensemble Training

Train ensemble models together:

```bash
# Train ensemble
python src/lit_train.py --model-type ensemble --archs resnet18,resnet34,vit_b_16 --devices 2
```

### 3. Ensemble Inference

Use trained ensemble for inference:

```python
from src.lit_system import LitEnsembleSystem

# Load ensemble model
ensemble = LitEnsembleSystem.load_from_checkpoint("checkpoints/ensemble/best.ckpt")

# Make predictions
predictions = ensemble.predict(test_dataloader)
```

## 🚀 Performance Optimization

### 1. Mixed Precision Training

Use mixed precision for faster training:

```python
trainer = Trainer(
    precision="bf16-mixed",  # For A100/H100
    # or
    precision="16-mixed",    # For other GPUs
)
```

### 2. Model Compilation

Compile models for faster inference:

```python
model = LitLensSystem(
    arch="resnet18",
    compile_model=True  # Enable torch.compile
)
```

### 3. Data Loading Optimization

Optimize data loading for cloud instances:

```python
datamodule = LensDataModule(
    num_workers=16,           # More workers for cloud
    pin_memory=True,          # Faster GPU transfer
    persistent_workers=True   # Keep workers alive
)
```

### 4. Distributed Training

Scale to multiple GPUs:

```python
trainer = Trainer(
    devices=4,
    strategy="ddp",           # Distributed data parallel
    accelerator="gpu"
)
```

## 🔧 Troubleshooting

### 1. Common Issues

**CUDA Out of Memory:**
```python
# Reduce batch size
trainer = Trainer(
    devices=1,
    precision="16-mixed"  # Use mixed precision
)

# Or use gradient accumulation
trainer = Trainer(
    accumulate_grad_batches=4  # Effective batch size = batch_size * 4
)
```

**WebDataset Connection Issues:**
```python
# Check credentials
import fsspec
fs = fsspec.filesystem("s3")
fs.ls("your-bucket")  # Should work

# Use local cache
datamodule = LensWebDatasetDataModule(
    cache_dir="/tmp/wds_cache"
)
```

**Slow Data Loading:**
```python
# Increase workers
datamodule = LensDataModule(
    num_workers=16,  # More workers
    pin_memory=True,
    persistent_workers=True
)
```

### 2. Debug Mode

Enable debug mode for troubleshooting:

```python
trainer = Trainer(
    fast_dev_run=True,        # Run 1 batch for testing
    limit_train_batches=10,   # Limit training batches
    limit_val_batches=5       # Limit validation batches
)
```

## 📚 Advanced Features

### 1. Custom Callbacks

Create custom callbacks for specific needs:

```python
from pytorch_lightning.callbacks import Callback

class CustomCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Custom validation logic
        pass

trainer = Trainer(callbacks=[CustomCallback()])
```

### 2. Custom Metrics

Add custom metrics:

```python
from torchmetrics import Metric

class CustomMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds, target):
        # Update metric state
        pass
    
    def compute(self):
        # Compute final metric
        return self.correct.float() / self.total
```

### 3. Hyperparameter Tuning

Use Lightning with hyperparameter tuning:

```python
import optuna
from pytorch_lightning.tuner import Tuner

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    model = LitLensSystem(lr=lr)
    datamodule = LensDataModule(batch_size=batch_size)
    
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, datamodule)
    
    return trainer.callback_metrics["val/auroc"].item()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

## 🎯 Best Practices

### 1. Reproducibility

Always set seeds for reproducible results:

```python
import pytorch_lightning as pl

# Set seed in Lightning
pl.seed_everything(42)

# Or in your script
set_seed(42)
```

### 2. Model Checkpointing

Save best models automatically:

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor="val/auroc",
    mode="max",
    save_top_k=3,
    save_last=True
)
```

### 3. Early Stopping

Prevent overfitting:

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val/auroc",
    mode="max",
    patience=10,
    min_delta=1e-4
)
```

### 4. Resource Management

Optimize resource usage:

```python
# Use appropriate precision
trainer = Trainer(
    precision="bf16-mixed" if torch.cuda.get_device_capability()[0] >= 8 else "16-mixed"
)

# Use appropriate batch size
batch_size = 128 if torch.cuda.get_device_properties(0).total_memory > 16e9 else 64
```

## 📞 Support

For Lightning-specific issues:

- **Lightning Documentation**: https://lightning.ai/docs/
- **Lightning Community**: https://lightning.ai/community/
- **GitHub Issues**: https://github.com/Lightning-AI/lightning/issues

For project-specific issues:

- **Project Issues**: https://github.com/Kantoration/mechine_lensing/issues
- **Documentation**: See other docs in this repository

---

**Happy Training with Lightning AI! ⚡**

