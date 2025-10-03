# 🚀 Comprehensive Integration Implementation Plan

## Gravitational Lens Classification System - Dataset & Model Integration

*This document provides the complete implementation plan for integrating real astronomical datasets and advanced model architectures with the existing Lightning AI infrastructure.*

---

## 📊 **Dataset Integration Specifications**

### **1. Supported Dataset Formats**

| Dataset | Format | Size | Resolution | Metadata |
|---------|--------|------|------------|----------|
| **GalaxiesML** | HDF5 | 15-50GB | 64×64, 127×127 | Redshift, Sérsic params, photometry |
| **Galaxy Zoo** | FITS/CSV | 100-200GB | Variable | Morphology labels, citizen votes |
| **CASTLES** | FITS | 1-10MB/file | Variable | Lens parameters, observations |

### **2. Data Pipeline Architecture**

```
Raw Data (FITS/HDF5)
    ↓
Conversion Pipeline (fits2hdf)
    ↓
Preprocessing (calibration, normalization)
    ↓
WebDataset Shards (cloud storage)
    ↓
Lightning StreamingDataset
    ↓
Model Training
```

### **3. Metadata Schema**

```python
metadata_schema = {
    # Observational Parameters
    'redshift': float,
    'seeing': float,  # arcsec
    'instrument': str,
    'bands': List[str],  # ['g', 'r', 'i', 'z', 'y']
    
    # Astrometric
    'ra': float,  # degrees
    'dec': float,  # degrees
    
    # Photometric
    'magnitude': Dict[str, float],  # per band
    'snr': float,
    
    # Physical Properties
    'sersic_index': float,
    'half_light_radius': float,
    'ellipticity': float,
    
    # Quality Metrics
    'completeness': float,
    'source_catalog': str
}
```

---

## 🧠 **Model Integration Architecture**

### **1. Unified Model Registry**

Extend `src/models/ensemble/registry.py`:

```python
# Advanced Models Registry Extension
ADVANCED_MODEL_REGISTRY = {
    'enhanced_vit': {
        'backbone_class': EnhancedViTBackbone,
        'backbone_kwargs': {
            'img_size': 224,
            'patch_size': 16,
            'attention_type': 'lensing_aware',
            'positional_encoding': 'astronomical'  # RA/Dec aware
        },
        'feature_dim': 768,
        'input_size': 224,
        'supports_physics': True,
        'supports_metadata': True,
        'description': 'Enhanced ViT with astronomical coordinate encoding'
    },
    
    'robust_resnet': {
        'backbone_class': RobustResNetBackbone,
        'backbone_kwargs': {
            'arch': 'resnet50',
            'adversarial_training': True,
            'noise_augmentation': True
        },
        'feature_dim': 2048,
        'input_size': 224,
        'supports_physics': False,
        'supports_metadata': False,
        'description': 'Adversarially trained ResNet for robustness'
    },
    
    'pinn_lens': {
        'backbone_class': PhysicsInformedBackbone,
        'backbone_kwargs': {
            'physics_constraints': ['lensing_equation', 'mass_conservation'],
            'differentiable_simulator': 'lenstronomy'
        },
        'feature_dim': 512,
        'input_size': 224,
        'supports_physics': True,
        'supports_metadata': True,
        'description': 'Physics-Informed Neural Network with lensing constraints'
    },
    
    'film_conditioned': {
        'backbone_class': FiLMConditionedBackbone,
        'backbone_kwargs': {
            'base_arch': 'resnet34',
            'metadata_dim': 10,
            'film_layers': [2, 3, 4]  # Which ResNet blocks to condition
        },
        'feature_dim': 512,
        'input_size': 224,
        'supports_physics': False,
        'supports_metadata': True,
        'description': 'FiLM-conditioned network for metadata integration'
    },
    
    'gat_lens': {
        'backbone_class': GraphAttentionBackbone,
        'backbone_kwargs': {
            'node_features': 128,
            'num_heads': 8,
            'num_layers': 4,
            'spatial_radius': 5.0  # arcsec
        },
        'feature_dim': 512,
        'input_size': 224,
        'supports_physics': True,
        'supports_metadata': True,
        'description': 'Graph Attention Network for multi-object lens systems'
    },
    
    'bayesian_ensemble': {
        'backbone_class': BayesianEnsembleBackbone,
        'backbone_kwargs': {
            'base_models': ['resnet18', 'vit_b16'],
            'num_mc_samples': 20,
            'prior_type': 'gaussian'
        },
        'feature_dim': 640,  # Combined
        'input_size': 224,
        'supports_physics': False,
        'supports_metadata': False,
        'description': 'Bayesian ensemble with uncertainty quantification'
    }
}
```

### **2. Enhanced Lightning Module**

Create `src/lit_advanced_system.py`:

```python
class LitAdvancedLensSystem(pl.LightningModule):
    """Advanced Lightning module with metadata conditioning and physics constraints."""
    
    def __init__(
        self,
        arch: str,
        model_type: str = "single",
        use_metadata: bool = False,
        use_physics: bool = False,
        physics_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model with advanced features
        self.model = self._create_advanced_model()
        
        # Setup physics constraints
        if use_physics:
            self.physics_validator = PhysicsValidator()
            self.differentiable_simulator = DifferentiableLensingSimulator()
        
        # Setup metrics
        self._setup_advanced_metrics()
    
    def _create_advanced_model(self):
        """Create model with advanced features."""
        if self.hparams.arch in ADVANCED_MODEL_REGISTRY:
            config = ADVANCED_MODEL_REGISTRY[self.hparams.arch]
            backbone_class = config['backbone_class']
            backbone = backbone_class(**config['backbone_kwargs'])
            head = BinaryHead(in_dim=config['feature_dim'], p=self.hparams.dropout_rate)
            return nn.Sequential(backbone, head)
        else:
            # Fall back to standard models
            return self._create_standard_model()
    
    def training_step(self, batch, batch_idx):
        """Training step with optional physics constraints."""
        x, y = batch["image"], batch["label"].float()
        metadata = batch.get("metadata", None)
        
        # Forward pass
        if self.hparams.use_metadata and metadata is not None:
            logits = self.model(x, metadata).squeeze(1)
        else:
            logits = self.model(x).squeeze(1)
        
        # Standard loss
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Add physics-informed loss
        if self.hparams.use_physics:
            physics_loss = self._compute_physics_loss(x, logits)
            loss = loss + self.hparams.physics_weight * physics_loss
            self.log("train/physics_loss", physics_loss)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def _compute_physics_loss(self, images, predictions):
        """Compute physics-informed loss."""
        # Generate synthetic lensing using differentiable simulator
        synthetic_lenses = self.differentiable_simulator(images)
        
        # Consistency loss
        consistency_loss = F.mse_loss(images, synthetic_lenses)
        
        # Physical plausibility score
        plausibility = self.physics_validator.validate(images, predictions)
        
        return consistency_loss + (1.0 - plausibility.mean())
```

---

## 🔄 **Data Pipeline Implementation**

### **1. Dataset Conversion Pipeline**

Create `scripts/convert_real_datasets.py`:

```python
#!/usr/bin/env python3
"""
Convert real astronomical datasets to project format.
Supports GalaxiesML (HDF5), Galaxy Zoo (FITS), and CASTLES (FITS).
"""

import h5py
import numpy as np
from astropy.io import fits
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DatasetConverter:
    """Universal converter for astronomical datasets."""
    
    def __init__(self, output_dir: Path, image_size: int = 224):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_galaxiesml(self, hdf5_path: Path, split: str = "train"):
        """
        Convert GalaxiesML HDF5 dataset to project format.
        
        Args:
            hdf5_path: Path to GalaxiesML HDF5 file
            split: Dataset split (train/val/test)
        """
        logger.info(f"Converting GalaxiesML dataset: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            images = f['images'][:]  # Shape: (N, H, W, C)
            labels = f['labels'][:]   # 0 or 1
            redshifts = f['redshift'][:]
            
            # Optional: Sérsic parameters
            if 'sersic_n' in f:
                sersic_n = f['sersic_n'][:]
                half_light_r = f['half_light_radius'][:]
                ellipticity = f['ellipticity'][:]
            
        # Create output directories
        lens_dir = self.output_dir / split / "lens"
        nonlens_dir = self.output_dir / split / "nonlens"
        lens_dir.mkdir(parents=True, exist_ok=True)
        nonlens_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and save images
        metadata_rows = []
        for idx, (img, label, z) in enumerate(tqdm(zip(images, labels, redshifts), 
                                                    total=len(images))):
            # Normalize and resize
            img = self._preprocess_image(img)
            
            # Save image
            if label == 1:
                filepath = lens_dir / f"lens_{split}_{idx:06d}.png"
            else:
                filepath = nonlens_dir / f"nonlens_{split}_{idx:06d}.png"
            
            Image.fromarray(img).save(filepath)
            
            # Build metadata
            metadata_row = {
                'filepath': str(filepath.relative_to(self.output_dir)),
                'label': int(label),
                'redshift': float(z),
                'source_catalog': 'GalaxiesML',
                'instrument': 'HSC',
                'bands': 'grizy'
            }
            
            # Add optional parameters
            if 'sersic_n' in locals():
                metadata_row.update({
                    'sersic_index': float(sersic_n[idx]),
                    'half_light_radius': float(half_light_r[idx]),
                    'ellipticity': float(ellipticity[idx])
                })
            
            metadata_rows.append(metadata_row)
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.to_csv(self.output_dir / f"{split}.csv", index=False)
        
        logger.info(f"Converted {len(images)} images from GalaxiesML")
    
    def convert_galaxy_zoo(self, fits_dir: Path, labels_csv: Path, split: str = "train"):
        """
        Convert Galaxy Zoo FITS images to project format.
        
        Args:
            fits_dir: Directory containing FITS files
            labels_csv: CSV with labels and metadata
            split: Dataset split
        """
        logger.info(f"Converting Galaxy Zoo dataset from: {fits_dir}")
        
        # Load labels
        labels_df = pd.read_csv(labels_csv)
        
        # Create output directories
        lens_dir = self.output_dir / split / "lens"
        nonlens_dir = self.output_dir / split / "nonlens"
        lens_dir.mkdir(parents=True, exist_ok=True)
        nonlens_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_rows = []
        for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
            fits_file = fits_dir / row['filename']
            
            if not fits_file.exists():
                continue
            
            # Load FITS image
            with fits.open(fits_file) as hdul:
                img = hdul[0].data
                header = hdul[0].header
            
            # Preprocess
            img = self._preprocess_fits_image(img)
            
            # Determine label (lens detection from morphology)
            label = self._determine_lens_label(row)
            
            # Save image
            if label == 1:
                filepath = lens_dir / f"lens_{split}_{idx:06d}.png"
            else:
                filepath = nonlens_dir / f"nonlens_{split}_{idx:06d}.png"
            
            Image.fromarray(img).save(filepath)
            
            # Build metadata
            metadata_row = {
                'filepath': str(filepath.relative_to(self.output_dir)),
                'label': int(label),
                'ra': float(header.get('RA', 0.0)),
                'dec': float(header.get('DEC', 0.0)),
                'source_catalog': 'Galaxy Zoo',
                'instrument': header.get('TELESCOP', 'SDSS')
            }
            metadata_rows.append(metadata_row)
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.to_csv(self.output_dir / f"{split}.csv", index=False)
        
        logger.info(f"Converted {len(metadata_rows)} images from Galaxy Zoo")
    
    def convert_castles(self, fits_dir: Path, split: str = "train"):
        """
        Convert CASTLES lens systems to project format.
        
        Args:
            fits_dir: Directory containing CASTLES FITS files
            split: Dataset split
        """
        logger.info(f"Converting CASTLES dataset from: {fits_dir}")
        
        # CASTLES contains confirmed lenses, so all labels are 1
        lens_dir = self.output_dir / split / "lens"
        lens_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_rows = []
        fits_files = list(fits_dir.glob("*.fits"))
        
        for idx, fits_file in enumerate(tqdm(fits_files)):
            # Load FITS image
            with fits.open(fits_file) as hdul:
                img = hdul[0].data
                header = hdul[0].header
            
            # Preprocess
            img = self._preprocess_fits_image(img)
            
            # Save image
            filepath = lens_dir / f"lens_{split}_{idx:06d}.png"
            Image.fromarray(img).save(filepath)
            
            # Build metadata
            metadata_row = {
                'filepath': str(filepath.relative_to(self.output_dir)),
                'label': 1,  # All CASTLES are confirmed lenses
                'ra': float(header.get('RA', 0.0)),
                'dec': float(header.get('DEC', 0.0)),
                'source_catalog': 'CASTLES',
                'instrument': header.get('TELESCOP', 'HST'),
                'lens_system': fits_file.stem
            }
            metadata_rows.append(metadata_row)
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.to_csv(self.output_dir / f"{split}.csv", index=False)
        
        logger.info(f"Converted {len(metadata_rows)} lens systems from CASTLES")
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess astronomical image."""
        # Normalize to 0-255
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        
        # Resize
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        return np.array(img_pil)
    
    def _preprocess_fits_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess FITS image with astronomical calibration."""
        # Handle NaN values
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip outliers (3-sigma)
        mean, std = img.mean(), img.std()
        img = np.clip(img, mean - 3*std, mean + 3*std)
        
        # Normalize and convert
        return self._preprocess_image(img)
    
    def _determine_lens_label(self, row: pd.Series) -> int:
        """Determine if Galaxy Zoo object is a lens based on morphology."""
        # Example heuristic: look for ring/arc features
        # This should be customized based on available Galaxy Zoo features
        if 'has_ring' in row and row['has_ring'] > 0.5:
            return 1
        if 'smooth' in row and row['smooth'] < 0.3:  # Not smooth = potential structure
            return 1
        return 0


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert astronomical datasets")
    parser.add_argument("--dataset", required=True, 
                       choices=['galaxiesml', 'galaxy_zoo', 'castles'])
    parser.add_argument("--input", required=True, help="Input directory or file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--split", default="train", choices=['train', 'val', 'test'])
    parser.add_argument("--image-size", type=int, default=224)
    
    args = parser.parse_args()
    
    converter = DatasetConverter(Path(args.output), args.image_size)
    
    if args.dataset == 'galaxiesml':
        converter.convert_galaxiesml(Path(args.input), args.split)
    elif args.dataset == 'galaxy_zoo':
        # Assumes --input is directory with FITS and labels.csv
        fits_dir = Path(args.input) / "images"
        labels_csv = Path(args.input) / "labels.csv"
        converter.convert_galaxy_zoo(fits_dir, labels_csv, args.split)
    elif args.dataset == 'castles':
        converter.convert_castles(Path(args.input), args.split)
```

### **2. Enhanced DataModule**

Extend `src/lit_datamodule.py`:

```python
class EnhancedLensDataModule(pl.LightningDataModule):
    """Enhanced DataModule with metadata support and cloud streaming."""
    
    def __init__(
        self,
        data_root: str = None,
        use_webdataset: bool = False,
        train_urls: List[str] = None,
        val_urls: List[str] = None,
        test_urls: List[str] = None,
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
        use_metadata: bool = False,
        metadata_columns: List[str] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.use_metadata = use_metadata
        self.metadata_columns = metadata_columns or [
            'redshift', 'seeing', 'magnitude', 'ra', 'dec'
        ]
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets with metadata support."""
        if self.hparams.use_webdataset:
            self._setup_webdataset(stage)
        else:
            self._setup_local_dataset(stage)
    
    def _setup_local_dataset(self, stage):
        """Setup local dataset with enhanced metadata."""
        if stage == "fit" or stage is None:
            # Load training dataset with metadata
            self.train_dataset = EnhancedLensDataset(
                data_root=self.hparams.data_root,
                split="train",
                img_size=self.hparams.image_size,
                augment=True,
                use_metadata=self.use_metadata,
                metadata_columns=self.metadata_columns
            )
            
            self.val_dataset = EnhancedLensDataset(
                data_root=self.hparams.data_root,
                split="val",
                img_size=self.hparams.image_size,
                augment=False,
                use_metadata=self.use_metadata,
                metadata_columns=self.metadata_columns
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = EnhancedLensDataset(
                data_root=self.hparams.data_root,
                split="test",
                img_size=self.hparams.image_size,
                augment=False,
                use_metadata=self.use_metadata,
                metadata_columns=self.metadata_columns
            )
```

---

## 🎯 **Implementation Priority Matrix**

| Priority | Component | Timeline | Dependencies |
|----------|-----------|----------|--------------|
| **P0** | Dataset conversion pipeline | Week 1 | None |
| **P0** | Enhanced DataModule with metadata | Week 1 | Dataset converter |
| **P1** | Model registry extension | Week 2 | None |
| **P1** | Enhanced Lightning module | Week 2 | Model registry |
| **P2** | Physics-informed components | Week 3 | Lightning module |
| **P2** | FiLM conditioning | Week 3 | Enhanced DataModule |
| **P3** | Bayesian uncertainty | Week 4 | All models trained |
| **P3** | Graph Attention Network | Week 4 | Spatial preprocessing |

---

## 🔧 **Configuration Templates**

### **Enhanced ViT Configuration**

Create `configs/enhanced_vit.yaml`:

```yaml
model:
  arch: "enhanced_vit"
  model_type: "single"
  pretrained: true
  dropout_rate: 0.3
  bands: 5  # g,r,i,z,y
  use_metadata: true
  use_physics: false

training:
  epochs: 50
  batch_size: 32  # ViT is memory-intensive
  learning_rate: 1e-4
  weight_decay: 1e-4
  scheduler_type: "cosine"

hardware:
  devices: 2
  accelerator: "gpu"
  precision: "bf16-mixed"  # Use bfloat16 on A100
  strategy: "ddp"

data:
  data_root: "data/processed/galaxiesml"
  val_split: 0.15
  num_workers: 16
  image_size: 224
  augment: true
  use_metadata: true
  metadata_columns: ['redshift', 'seeing', 'sersic_index']
```

### **Physics-Informed Configuration**

Create `configs/pinn_lens.yaml`:

```yaml
model:
  arch: "pinn_lens"
  model_type: "physics_informed"
  pretrained: false
  dropout_rate: 0.2
  bands: 5
  use_metadata: true
  use_physics: true
  physics_weight: 0.2
  
  physics_config:
    constraints:
      - "lensing_equation"
      - "mass_conservation"
      - "shear_consistency"
    simulator: "lenstronomy"
    differentiable: true

training:
  epochs: 60
  batch_size: 48
  learning_rate: 5e-5
  weight_decay: 1e-5
  scheduler_type: "plateau"

hardware:
  devices: 4
  accelerator: "gpu"
  precision: "32"  # Physics requires higher precision
  strategy: "ddp"
```

---

## 🧪 **Testing & Validation Plan**

### **Unit Tests**

Create `tests/test_advanced_models.py`:

```python
def test_enhanced_vit_creation():
    """Test Enhanced ViT model creation."""
    config = ModelConfig(
        model_type="single",
        architecture="enhanced_vit",
        bands=5,
        pretrained=False
    )
    model = create_model(config)
    assert model is not None
    
    # Test forward pass
    x = torch.randn(2, 5, 224, 224)
    output = model(x)
    assert output.shape == (2, 1)

def test_physics_informed_loss():
    """Test physics-informed loss computation."""
    validator = PhysicsValidator()
    simulator = DifferentiableLensingSimulator()
    
    # Test with synthetic data
    images = torch.randn(4, 5, 224, 224)
    predictions = torch.randn(4, 1)
    
    loss = compute_physics_loss(images, predictions, simulator, validator)
    assert loss.item() >= 0.0

def test_metadata_conditioning():
    """Test FiLM conditioning with metadata."""
    model = create_model(ModelConfig(
        architecture="film_conditioned",
        bands=5
    ))
    
    images = torch.randn(2, 5, 224, 224)
    metadata = torch.randn(2, 10)  # 10 metadata features
    
    output = model(images, metadata)
    assert output.shape == (2, 1)
```

### **Integration Tests**

Create `tests/test_dataset_integration.py`:

```python
def test_galaxiesml_conversion():
    """Test GalaxiesML dataset conversion."""
    converter = DatasetConverter(output_dir="data/test_output")
    converter.convert_galaxiesml(
        hdf5_path="data/raw/GalaxiesML/test.h5",
        split="train"
    )
    
    # Verify output
    assert (Path("data/test_output/train.csv")).exists()
    df = pd.read_csv("data/test_output/train.csv")
    assert 'redshift' in df.columns
    assert 'label' in df.columns

def test_enhanced_datamodule():
    """Test Enhanced DataModule with metadata."""
    dm = EnhancedLensDataModule(
        data_root="data/processed/galaxiesml",
        batch_size=16,
        use_metadata=True
    )
    dm.setup("fit")
    
    # Test batch loading
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    assert "image" in batch
    assert "label" in batch
    assert "metadata" in batch
    assert batch["metadata"].shape[1] > 0  # Has metadata features
```

---

## 📈 **Success Metrics**

### **Phase 1: Dataset Integration**
- ✅ Successfully load and process GalaxiesML dataset
- ✅ Convert 100K+ images without data loss
- ✅ Metadata extracted and validated
- ✅ WebDataset shards created for cloud streaming

### **Phase 2: Model Integration**
- ✅ All 6 new model architectures registered
- ✅ Models train without errors
- ✅ Metadata conditioning works correctly
- ✅ Physics constraints reduce false positives by >10%

### **Phase 3: Performance**
- ✅ Achieve >92% accuracy on GalaxiesML test set
- ✅ Ensemble achieves >95% accuracy
- ✅ Uncertainty calibration error < 5%
- ✅ Training scales to 4+ GPUs with linear speedup

### **Phase 4: Production**
- ✅ Deploy on Lightning AI Cloud
- ✅ Process 1000+ images/second
- ✅ Model serving API available
- ✅ Comprehensive logging and monitoring

---

## 🚀 **Quick Start Implementation**

### **Step 1: Convert Your First Dataset**

```bash
# Convert GalaxiesML dataset
python scripts/convert_real_datasets.py \
    --dataset galaxiesml \
    --input data/raw/GalaxiesML/train.h5 \
    --output data/processed/galaxiesml \
    --split train \
    --image-size 224

# Convert validation set
python scripts/convert_real_datasets.py \
    --dataset galaxiesml \
    --input data/raw/GalaxiesML/val.h5 \
    --output data/processed/galaxiesml \
    --split val \
    --image-size 224
```

### **Step 2: Train Enhanced ViT**

```bash
# Train with Lightning AI
python src/lit_train.py \
    --config configs/enhanced_vit.yaml \
    --trainer.accelerator=gpu \
    --trainer.devices=2 \
    --trainer.max_epochs=50
```

### **Step 3: Train Physics-Informed Model**

```bash
# Train PINN model
python src/lit_train.py \
    --config configs/pinn_lens.yaml \
    --trainer.accelerator=gpu \
    --trainer.devices=4 \
    --trainer.max_epochs=60
```

### **Step 4: Create Advanced Ensemble**

```bash
# Train ensemble with all models
make lit-train-advanced-ensemble \
    --models="enhanced_vit,robust_resnet,pinn_lens" \
    --devices=4
```

---

## 📚 **References & Resources**

- **GalaxiesML Paper**: https://arxiv.org/html/2410.00271v1
- **Lightning AI Docs**: https://lightning.ai/docs
- **Lenstronomy**: https://lenstronomy.readthedocs.io
- **FITS2HDF**: https://fits2hdf.readthedocs.io
- **Astropy**: https://docs.astropy.org

---

*Last Updated: 2025-10-03*
*Maintainer: Gravitational Lensing ML Team*

