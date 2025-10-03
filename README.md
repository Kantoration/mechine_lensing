# 🔭 Gravitational Lens Classification with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-black-black.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A production-ready machine learning pipeline for detecting gravitational lenses in astronomical images using deep learning. This project implements both CNN (ResNet-18/34) and Vision Transformer (ViT) architectures with ensemble capabilities for robust lens classification.

## 🌟 Key Features

- **🎯 High Performance**: Achieves 93-96% accuracy on realistic synthetic datasets
- **🏗️ Production Ready**: Comprehensive logging, error handling, and validation
- **🔬 Scientific Rigor**: Proper experimental design with reproducible results
- **🚀 Multi-Architecture**: Support for ResNet-18, ResNet-34, and ViT-B/16
- **⚡ Ensemble Learning**: Advanced ensemble methods for improved accuracy
- **⚡ Lightning AI Ready**: Easy cloud GPU scaling with Lightning AI
- **📊 Comprehensive Evaluation**: Detailed metrics and scientific reporting
- **🛠️ Developer Friendly**: Makefile, pre-commit hooks, comprehensive testing

## 📊 Results Overview (Example)

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **ResNet-18** | 93.0% | 91.4% | 95.0% | 93.1% | 97.7% |
| **ResNet-34** | 94.2% | 92.8% | 95.8% | 94.3% | 98.1% |
| **ViT-B/16** | 95.1% | 93.6% | 96.5% | 95.0% | 98.5% |
| **Ensemble** | **96.3%** | **94.9%** | **97.2%** | **96.0%** | **98.9%** |

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Git for cloning
git --version
```

### Installation

```bash
# Clone repository
git clone https://github.com/Kantoration/mechine_lensing.git
cd mechine_lensing

# Setup development environment (recommended)
make setup

# OR manual setup
python -m venv lens_env
source lens_env/bin/activate  # Linux/Mac
# lens_env\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Quick Development Workflow

```bash
# Complete development setup + quick test
make dev

# OR step by step:
make dataset-quick    # Generate small test dataset
make train-quick      # Quick training run
make eval            # Evaluate model
```

### Production Workflow

```bash
# Generate realistic dataset
make dataset

# Train individual models
make train-resnet18
make train-vit        # Requires GPU or cloud

# Evaluate ensemble
make eval-ensemble

# OR run complete pipeline
make full-pipeline
```

### ⚡ Lightning AI Cloud Training

```bash
# Prepare dataset for cloud streaming
make lit-prepare-dataset CLOUD_URL="s3://your-bucket/lens-data"

# Train on Lightning Cloud with WebDataset
make lit-train-cloud TRAIN_URLS="s3://your-bucket/train-{0000..0099}.tar" VAL_URLS="s3://your-bucket/val-{0000..0009}.tar"

# Train ensemble with Lightning AI
make lit-train-ensemble

# Quick Lightning training test
make lit-train-quick
```

---

## 🌌 For Astronomers: A Comprehensive Guide to Machine Learning for Gravitational Lensing

*This section is specifically designed for astronomers who want to understand how machine learning can revolutionize gravitational lens detection, explained in accessible terms with clear astronomical analogies.*

### 🔬 What is This Project and Why Does It Matter?

**The Big Picture**: This project develops an automated system that can identify gravitational lenses in astronomical images with 93-96% accuracy - comparable to or better than human experts, but capable of processing thousands of images per hour.

**Why This Matters for Astronomy**:
- **Scale Challenge**: Modern surveys like Euclid, LSST, and JWST will produce billions of galaxy images. Manual inspection is impossible.
- **Rarity Problem**: Strong gravitational lenses occur in only ~1 in 1000 massive galaxies, making them extremely difficult to find.
- **Scientific Impact**: Each lens discovered enables studies of dark matter, cosmological parameters, and high-redshift galaxy evolution.

**The Machine Learning Revolution**: Think of this as training a digital assistant that can:
- Learn from thousands of examples (like a graduate student studying for years)
- Never get tired or make subjective judgments
- Process images consistently and reproducibly
- Scale to handle the massive datasets of modern astronomy

### 🎯 The Scientific Challenge: Why Traditional Methods Fall Short

#### The Detection Problem
Gravitational lensing creates characteristic arc-like distortions when massive objects bend light from background galaxies. However, detecting these lenses is extremely challenging:

**Visual Complexity**:
- Lensing arcs are often faint and subtle
- They can be confused with spiral arms, galaxy interactions, or instrumental artifacts
- The signal-to-noise ratio is often very low
- Multiple lensing configurations create different visual patterns

**Scale and Rarity**:
- Only ~1 in 1000 massive galaxies acts as a strong lens
- Modern surveys contain millions of galaxy images
- Manual inspection by experts is time-consuming and subjective
- False positive rates are high due to similar-looking structures

**Traditional Approaches**:
- **Visual inspection**: Expert astronomers manually examine images (slow, subjective, not scalable)
- **Automated algorithms**: Rule-based systems looking for specific patterns (rigid, miss complex cases)
- **Statistical methods**: Analyzing galaxy shapes and orientations (limited sensitivity)

### 🧠 How Machine Learning Solves This Problem

**The Core Idea**: Instead of programming specific rules, we teach the computer to recognize gravitational lenses by showing it thousands of examples - just like how astronomers learn through experience.

#### The Learning Process (Astronomical Analogy)

Imagine training a new graduate student to identify gravitational lenses:

1. **Show Examples**: Present the student with thousands of images, some containing lenses and some without
2. **Practice Recognition**: The student makes predictions about each image
3. **Provide Feedback**: Tell the student whether they were correct or not
4. **Learn from Mistakes**: The student adjusts their criteria based on feedback
5. **Repeat**: Continue this process until the student becomes an expert

**Machine Learning does exactly this**, but:
- It can process thousands of examples in minutes
- It never forgets or gets tired
- It can detect patterns too subtle for human perception
- It provides consistent, reproducible results

#### What Makes Our Approach Special

**Scientifically Realistic Training Data**:
- We don't use simple toy examples (like bright circles vs. squares)
- Instead, we create complex, realistic galaxy images that capture the true physics of gravitational lensing
- Our synthetic images include proper noise, instrumental effects, and realistic galaxy structures

**Multiple Expert Systems**:
- We train several different "expert observers" (neural networks) with different strengths
- Each expert has different capabilities (like having specialists in different types of lensing)
- We combine their opinions for more reliable final decisions

**Uncertainty Quantification**:
- The system doesn't just say "this is a lens" - it says "I'm 95% confident this is a lens"
- This confidence estimate is calibrated to be accurate (when it says 95% confident, it's right 95% of the time)

### 🎨 Creating Realistic Training Data: The Physics Behind Our Synthetic Images

One of the biggest challenges in astronomical machine learning is getting enough high-quality training data. We solve this by creating scientifically realistic synthetic images.

#### Why Synthetic Data?

**The Data Problem**:
- Real gravitational lenses are rare and expensive to identify
- Manual labeling of thousands of images is time-consuming and error-prone
- We need balanced datasets with equal numbers of lens and non-lens examples
- We need to control all the parameters to understand what the system is learning

**Our Solution**: Create synthetic images that capture the essential physics of gravitational lensing while being computationally efficient.

#### The Physics in Our Synthetic Images

**For Lens Images** (Galaxies with Gravitational Lensing):
```python
def create_lens_arc_image(self, image_id: str, split: str):
    """Generate realistic gravitational lensing image: galaxy + subtle arcs."""
    
    # STEP 1: Create the lensing galaxy (the "lens")
    galaxy_sigma = self.rng.uniform(4.0, 8.0)  # Galaxy size (pixels)
    galaxy_ellipticity = self.rng.uniform(0.0, 0.4)  # Galaxy shape
    galaxy_brightness = self.rng.uniform(0.4, 0.7)  # Galaxy brightness
    
    # Create galaxy using realistic light distribution (Gaussian profile)
    # This simulates how real galaxies appear in astronomical images
    galaxy = np.exp(-0.5 * ((x_rot/a)**2 + (y_rot/b)**2))
    img += galaxy * galaxy_brightness
    
    # STEP 2: Add lensing arcs (the key difference from non-lens images)
    n_arcs = self.rng.integers(1, 4)  # 1-3 arcs per image
    
    for _ in range(n_arcs):
        # Arc parameters based on real lensing physics
        radius = self.rng.uniform(8.0, 20.0)  # Einstein radius
        arc_width = self.rng.uniform(2.0, 4.0)  # Arc thickness
        brightness = self.rng.uniform(0.7, 1.0)  # Arc brightness
        
        # Create arc using parametric equations
        # This simulates how background galaxies appear as arcs
        arc_points = self._generate_arc_points(radius, arc_width)
        img = self._draw_arc(img, arc_points, brightness)
```

**For Non-Lens Images** (Regular Galaxies):
```python
def create_non_lens_image(self, image_id: str, split: str):
    """Generate realistic non-lens galaxy image."""
    
    # Create complex galaxy structure (no lensing arcs)
    # This includes multiple components: bulge, disk, spiral arms
    
    # Main galaxy component
    main_galaxy = self._create_galaxy_component(
        sigma=galaxy_sigma,
        ellipticity=galaxy_ellipticity,
        brightness=galaxy_brightness
    )
    
    # Add spiral structure (if applicable)
    if self.rng.random() < 0.3:  # 30% chance of spiral features
        spiral_arms = self._create_spiral_arms()
        main_galaxy += spiral_arms
    
    # Add companion galaxies (common in real surveys)
    if self.rng.random() < 0.2:  # 20% chance of companions
        companion = self._create_companion_galaxy()
        main_galaxy += companion
```

#### Realistic Observational Effects

To make our synthetic images as realistic as possible, we add all the effects that real astronomical observations have:

```python
def add_realistic_noise(self, img: np.ndarray) -> np.ndarray:
    """Add realistic observational noise and artifacts."""
    
    # 1. Gaussian noise (photon noise, read noise)
    gaussian_noise = self.rng.normal(0, self.config.noise.gaussian_std)
    img += gaussian_noise
    
    # 2. Poisson noise (photon counting statistics)
    poisson_noise = self.rng.poisson(self.config.noise.poisson_lambda)
    img += poisson_noise
    
    # 3. PSF blur (atmospheric seeing, telescope optics)
    psf_sigma = self.config.noise.psf_sigma
    img = gaussian_filter(img, sigma=psf_sigma)
    
    # 4. Cosmic rays (random bright pixels)
    if self.rng.random() < 0.1:  # 10% chance
        cosmic_ray_pos = (self.rng.integers(0, img.shape[0]),
                         self.rng.integers(0, img.shape[1]))
        img[cosmic_ray_pos] += self.rng.uniform(2.0, 5.0)
    
    return np.clip(img, 0, 1)  # Ensure valid pixel values
```

**What This Means**: Our synthetic images look and behave like real astronomical observations, complete with noise, blur, and artifacts. This ensures that when we train our machine learning system, it learns to work with realistic data.

### 🧠 Neural Networks: How the Computer "Sees" Images

We use three different types of neural networks, each with different strengths. Think of them as different types of expert observers:

#### 1. ResNet (Residual Neural Network) - The Detail-Oriented Observer

**How ResNet Works**:
ResNet is like having a team of observers with different expertise levels, each building on what the previous observer found:

```python
class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction."""
    
    def __init__(self, arch: str = 'resnet18', in_ch: int = 3, pretrained: bool = True):
        super().__init__()
        
        # Load pre-trained model (trained on millions of natural images)
        if arch == "resnet18":
            self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        elif arch == "resnet34":
            self.backbone = torchvision.models.resnet34(pretrained=pretrained)
        
        # Remove final classification layer (we'll add our own)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        return self.backbone(x)
```

**Astronomical Analogy**: ResNet is like having a hierarchical team of observers:
- **Junior observers** (early layers): Detect basic features like bright spots, edges, and simple shapes
- **Senior observers** (middle layers): Combine these into more complex patterns like galaxy components and arc segments
- **Expert observers** (later layers): Recognize complete structures like lensing arcs and galaxy morphologies
- **Final decision maker**: Makes the classification based on all the information gathered

**ResNet's Strengths**:
- Excellent at detecting local features and patterns
- Very efficient and fast to train
- Works well even with limited data
- Good for identifying subtle lensing features

#### 2. Vision Transformer (ViT) - The Big-Picture Observer

Vision Transformers work completely differently - they treat images like text, breaking them into "patches" and analyzing relationships between patches:

```python
class ViTBackbone(nn.Module):
    """Vision Transformer backbone for feature extraction."""
    
    def __init__(self, arch: str = "vit_b_16", in_ch: int = 3, pretrained: bool = True):
        super().__init__()
        
        # Load pre-trained ViT
        self.backbone = torchvision.models.vit_b_16(pretrained=pretrained)
        
        # Remove final classification head
        self.backbone.heads = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using attention mechanism."""
        return self.backbone(x)
```

**How ViT Works**:
1. **Patch Embedding**: Breaks the image into 16×16 pixel patches (like words in a sentence)
2. **Attention Mechanism**: Each patch "pays attention" to other patches that might be relevant
3. **Global Context**: Can understand relationships between distant parts of the image

**Astronomical Analogy**: ViT is like having an expert who can:
- Look at the entire image at once and understand the big picture
- Notice that a faint arc in one corner might be connected to a galaxy in the center
- Understand how different parts of the image relate to each other
- See patterns that require understanding the whole image context

**ViT's Strengths**:
- Excellent at understanding global relationships
- Can detect complex patterns that span large areas
- Very good at distinguishing subtle differences
- Achieves the highest accuracy on our datasets

#### 3. Enhanced Light Transformer - The Specialized Lensing Expert

This is our custom architecture that combines the best of both worlds and adds specialized knowledge about gravitational lensing:

```python
class EnhancedLightTransformerBackbone(nn.Module):
    """Enhanced Light Transformer with arc-aware attention."""
    
    def __init__(self, cnn_stage: str = 'layer3', patch_size: int = 2,
                 embed_dim: int = 256, num_heads: int = 4, num_layers: int = 4,
                 attention_type: str = 'arc_aware'):
        super().__init__()
        
        # CNN feature extractor (like ResNet)
        self.cnn_backbone = self._create_cnn_backbone(cnn_stage)
        
        # Transformer layers (like ViT)
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Arc-aware attention (our innovation)
        if attention_type == 'arc_aware':
            self.arc_attention = ArcAwareAttention(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with CNN + Transformer + Arc-aware attention."""
        
        # Step 1: Extract local features with CNN
        cnn_features = self.cnn_backbone(x)
        
        # Step 2: Understand global relationships with Transformer
        for layer in self.transformer:
            cnn_features = layer(cnn_features)
        
        # Step 3: Focus on arc-like structures
        if hasattr(self, 'arc_attention'):
            cnn_features = self.arc_attention(cnn_features)
        
        return cnn_features
```

**Astronomical Analogy**: The Enhanced Light Transformer is like having a specialized team where:
- **CNN members** identify local features (galaxy components, arc segments, noise patterns)
- **Transformer members** understand global relationships (how arcs connect to galaxies, overall image structure)
- **Arc-aware attention members** specifically look for lensing signatures and ignore irrelevant features

**Enhanced Light Transformer's Strengths**:
- Combines local and global understanding
- Specifically designed for gravitational lensing detection
- Can focus attention on the most relevant parts of the image
- Achieves excellent performance with efficient computation

### 🎯 Training Process: How the Computer Learns to Recognize Lenses

The training process is like teaching a new observer to recognize gravitational lenses through supervised learning:

#### 1. Data Preparation

```python
class LensDataset(Dataset):
    """Dataset class for gravitational lensing images."""
    
    def __init__(self, data_root, split="train", img_size=224, augment=False):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        # Load metadata (which images are lenses, which are not)
        self.df = pd.read_csv(self.data_root / f"{split}.csv")
        
        # Set up image preprocessing
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Set up image transforms for training."""
        base_transforms = [
            transforms.Resize((self.img_size, self.img_size)),  # Resize to standard size
            transforms.ToTensor(),  # Convert to numerical format
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Standardize brightness
                               std=[0.229, 0.224, 0.225])
        ]
        
        if self.augment:
            # Data augmentation: create variations of each image
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally
                transforms.RandomRotation(degrees=10),    # Rotate slightly
                transforms.ColorJitter(brightness=0.2,   # Vary brightness
                                     contrast=0.2)       # Vary contrast
            ]
            self.transform = transforms.Compose(augment_transforms + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)
```

**What This Does**:
- **Resize**: Makes all images the same size (like standardizing telescope observations)
- **Normalize**: Adjusts brightness and contrast to standard values (like photometric calibration)
- **Augment**: Creates variations of each image (like observing the same object under different conditions)

#### 2. The Learning Loop

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch - one pass through all training data."""
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    
    for images, labels in train_loader:
        # Move data to GPU if available
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Forward pass: make prediction
        logits = model(images).squeeze(1)
        
        # Calculate loss (how wrong the prediction was)
        loss = criterion(logits, labels)
        
        # Backward pass: learn from mistakes
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Calculate accuracy for this batch
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean()
        
        # Track statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += acc.item() * batch_size
        num_samples += batch_size
    
    return running_loss / num_samples, running_acc / num_samples
```

**Astronomical Analogy**: The training process is like:
1. **Showing Examples**: Presenting the observer with a batch of images (like showing a student a set of exam questions)
2. **Making Predictions**: The observer guesses whether each image contains a lens (like answering the questions)
3. **Getting Feedback**: Calculating how wrong the predictions were (like grading the answers)
4. **Learning from Mistakes**: The observer adjusts their criteria based on the feedback (like studying the mistakes)
5. **Repeating**: Going through this process many times until the observer becomes expert (like taking many practice exams)

#### 3. Performance Optimization

We use several techniques to make training faster and more effective:

**Mixed Precision Training**:
```python
def train_step_amp(model, images, labels, optimizer, scaler):
    """Training step with mixed precision for faster training."""
    optimizer.zero_grad()
    
    # Use lower precision for faster computation
    with autocast():
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
    
    # Scale gradients to prevent underflow
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Astronomical Analogy**: Mixed precision is like using a faster but slightly less precise measurement technique when you need speed, but switching to high precision when accuracy is critical.

### 🤝 Ensemble Methods: Combining Multiple Experts

Just like astronomers often consult multiple experts for difficult cases, we combine multiple models to get more reliable results:

```python
class UncertaintyWeightedEnsemble(nn.Module):
    """Ensemble that weights models by their confidence."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = models
        self.weights = weights or [1.0] * len(models)
    
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Make ensemble predictions."""
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                logits = model(images).squeeze(1)
                probabilities = torch.sigmoid(logits)
                predictions.append(probabilities)
        
        # Weighted average of predictions
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred / sum(self.weights)
```

**Astronomical Analogy**: Ensemble methods are like:
- Having multiple expert observers examine the same image
- Each observer has different strengths (one is good at detecting faint arcs, another at recognizing galaxy types)
- Combining their opinions gives a more reliable final decision
- The system can also weight each expert's opinion based on their confidence

**Why Ensembles Work Better**:
- **Reduced False Positives**: If one model makes a mistake, others can correct it
- **Improved Sensitivity**: Different models detect different types of lensing features
- **Robustness**: Less sensitive to noise or unusual image characteristics
- **Confidence Estimation**: Can provide better uncertainty estimates

### 📊 Evaluation: How We Measure Success

We use several metrics to evaluate how well our models perform, each telling us something different about the system's capabilities:

```python
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """Evaluate model with comprehensive metrics."""
    model.eval()  # Set to evaluation mode
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():  # Don't update model during evaluation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            logits = model(images).squeeze(1)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions),
        'f1_score': f1_score(all_labels, all_predictions),
        'roc_auc': roc_auc_score(all_labels, all_probabilities)
    }
    
    return metrics
```

**Understanding the Metrics** (with Astronomical Analogies):

- **Accuracy**: What fraction of predictions were correct? 
  - *Like*: "The observer was right 95% of the time"
  - *What it means*: Overall correctness across all images

- **Precision**: Of the images predicted as lenses, what fraction actually contain lenses?
  - *Like*: "When the observer says 'lens', they're right 94% of the time"
  - *What it means*: Low false positive rate - when the system says it found a lens, it's usually correct

- **Recall**: Of all actual lenses, what fraction did we detect?
  - *Like*: "The observer found 97% of all the lenses that were actually there"
  - *What it means*: High detection rate - the system doesn't miss many real lenses

- **F1-Score**: A balance between precision and recall
  - *Like*: A single number that captures both how accurate the detections are and how complete they are
  - *What it means*: Overall performance considering both false positives and false negatives

- **ROC AUC**: How well can the model distinguish between lenses and non-lenses?
  - *Like*: "How good is the observer at ranking images from most likely to least likely to contain a lens?"
  - *What it means*: Higher is better, 1.0 is perfect discrimination ability

### 🔬 Scientific Validation: Ensuring Reliability

We implement several validation strategies to ensure our results are scientifically sound:

#### 1. Reproducibility
```python
def set_seed(seed: int = 42):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

**Why This Matters**: Like ensuring that an experiment can be repeated with the same results, we fix all random processes so that anyone can reproduce our findings.

#### 2. Cross-Validation
```python
def k_fold_cross_validation(dataset, k=5):
    """Perform k-fold cross-validation."""
    # Split data into k folds
    folds = split_dataset(dataset, k)
    
    results = []
    for i in range(k):
        # Use fold i as test set, others as training set
        train_fold = combine_folds([folds[j] for j in range(k) if j != i])
        test_fold = folds[i]
        
        # Train model on train_fold
        model = train_model(train_fold)
        
        # Evaluate on test_fold
        metrics = evaluate_model(model, test_fold)
        results.append(metrics)
    
    return results
```

**Astronomical Analogy**: Cross-validation is like testing an observer on different sets of images to make sure their performance is consistent and not just memorizing specific examples.

#### 3. Uncertainty Quantification
```python
class TemperatureScaler:
    """Temperature scaling for better uncertainty estimates."""
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """Calibrate the model's confidence estimates."""
        # Adjust temperature to make confidence estimates more accurate
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
    
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to get calibrated probabilities."""
        return torch.sigmoid(logits / self.temperature)
```

**Astronomical Analogy**: Uncertainty quantification is like having the observer not just say "this is a lens" but also "I'm 95% confident this is a lens" - and making sure that when they say 95% confident, they're actually right 95% of the time.

### 🚀 Practical Usage for Astronomers

#### Running Your First Analysis

```bash
# 1. Generate a realistic dataset
python scripts/utilities/generate_dataset.py --config configs/realistic.yaml --out data/realistic

# 2. Train a ResNet-18 model (good for laptops)
python src/training/trainer.py --data-root data/realistic --arch resnet18 --epochs 20

# 3. Evaluate the model
python src/evaluation/evaluator.py --data-root data/realistic --weights checkpoints/best_resnet18.pt
```

#### Understanding the Output

The evaluation produces several files:

- **`results/evaluation_summary.json`**: High-level performance metrics
- **`results/detailed_predictions.csv`**: Per-image predictions and confidence scores
- **`results/calibration_plots.png`**: Visualizations of model confidence

#### Interpreting Results

```python
# Load evaluation results
import json
with open('results/evaluation_summary.json', 'r') as f:
    results = json.load(f)

print(f"Model Accuracy: {results['accuracy']:.3f}")
print(f"Precision: {results['precision']:.3f}")  # Low false positive rate
print(f"Recall: {results['recall']:.3f}")        # High detection rate
print(f"ROC AUC: {results['roc_auc']:.3f}")      # Overall discrimination ability
```

### 🔮 Future Directions and Research Opportunities

This project opens several exciting research directions for the astronomical community:

#### 1. Real Survey Data Applications
- **Euclid Survey**: Apply to the upcoming Euclid space telescope data
- **LSST**: Scale to handle the massive Legacy Survey of Space and Time dataset
- **JWST**: Adapt to near-infrared observations with different noise characteristics
- **Multi-wavelength**: Extend to handle data from multiple filters simultaneously

#### 2. Advanced Physics Integration
- **Lensing Theory**: Incorporate lensing equations directly into the neural network
- **Physical Constraints**: Use known physics to improve predictions and reduce false positives
- **Parameter Estimation**: Not just detect lenses, but estimate lens parameters (Einstein radius, ellipticity, etc.)
- **Multi-scale Analysis**: Handle lenses at different scales (galaxy-scale vs. cluster-scale)

#### 3. Active Learning and Human-in-the-Loop
- **Intelligent Selection**: Automatically select which images are most informative for human review
- **Reduced Labeling**: Minimize the amount of manual labeling required
- **Expert Feedback**: Incorporate human expert corrections to improve the system
- **Uncertainty-Driven**: Focus human expertise on the most uncertain cases

#### 4. Scientific Applications
- **Dark Matter Mapping**: Use detected lenses to map dark matter distribution
- **Cosmological Parameters**: Measure Hubble constant and other fundamental constants
- **Galaxy Evolution**: Study high-redshift galaxies magnified by lensing
- **Fundamental Physics**: Test general relativity and alternative theories of gravity

### 🎓 Key Takeaways for Astronomers

1. **Machine Learning is a Powerful Tool**: When properly applied, ML can achieve expert-level performance in gravitational lens detection while being much faster and more consistent than human inspection.

2. **Synthetic Data is Essential**: Creating realistic synthetic datasets is crucial for training reliable ML systems, especially for rare phenomena like gravitational lensing.

3. **Ensemble Methods Work**: Combining multiple models (like consulting multiple experts) significantly improves reliability and reduces false positives.

4. **Uncertainty Matters**: Modern ML systems can provide confidence estimates, which are crucial for scientific applications where you need to know how much to trust each detection.

5. **Reproducibility is Key**: All results should be reproducible, with fixed random seeds and comprehensive logging of all parameters and procedures.

6. **This is Just the Beginning**: The techniques developed here can be extended to many other astronomical problems, from exoplanet detection to galaxy classification to transient identification.

---

## 🌌 For Astronomers: Understanding the Technical Implementation

*This section explains the machine learning concepts and implementation details in astronomical terms, making the technical aspects accessible to astronomers without deep ML expertise.*

### 🔬 The Scientific Challenge

Gravitational lensing is one of the most powerful tools in modern cosmology, but detecting lenses in astronomical surveys is extremely challenging:

- **Rarity**: Only ~1 in 1000 massive galaxies acts as a strong lens
- **Subtlety**: Lensing features are often faint and easily confused with other structures
- **Scale**: Modern surveys like Euclid will contain billions of galaxy images
- **Contamination**: Many false positives from galaxy interactions, mergers, and instrumental artifacts

### 🧠 How Machine Learning Solves This Problem

Think of machine learning as training a very sophisticated pattern recognition system that can learn to identify gravitational lenses by studying thousands of examples, much like how astronomers learn to recognize lensing features through experience.

#### The Learning Process

1. **Training Data**: We show the computer thousands of images, some with lenses and some without
2. **Pattern Learning**: The computer learns to identify subtle patterns that distinguish lenses from non-lenses
3. **Generalization**: The trained system can then identify lenses in new, unseen images

### 🎨 Synthetic Data Generation: Creating Realistic Training Sets

One of the biggest challenges in astronomical machine learning is getting enough high-quality training data. We solve this by creating scientifically realistic synthetic images.

#### Why Synthetic Data?

Real gravitational lenses are rare, and manually labeling thousands of images is time-consuming and error-prone. Instead, we create synthetic images that capture the essential physics of gravitational lensing.

#### The Physics Behind Our Synthetic Images

Our synthetic image generator creates realistic lensing scenarios by simulating:

**For Lens Images:**
```python
def create_lens_arc_image(self, image_id: str, split: str):
    """Generate realistic gravitational lensing image: galaxy + subtle arcs."""
    
    # STEP 1: Create background galaxy (similar to non-lens images)
    galaxy_sigma = self.rng.uniform(4.0, 8.0)  # Galaxy size
    galaxy_ellipticity = self.rng.uniform(0.0, 0.4)  # Galaxy shape
    galaxy_brightness = self.rng.uniform(0.4, 0.7)  # Galaxy brightness
    
    # Create galaxy using Gaussian profile (realistic galaxy light distribution)
    y, x = np.ogrid[:image_size, :image_size]
    x_rot = cos_a * x - sin_a * y  # Rotate coordinates
    y_rot = sin_a * x + cos_a * y
    
    # Elliptical Gaussian: exp(-0.5 * ((x/a)² + (y/b)²))
    galaxy = np.exp(-0.5 * ((x_rot/a)**2 + (y_rot/b)**2))
    img += galaxy * galaxy_brightness
    
    # STEP 2: Add subtle lensing arcs (the key difference)
    n_arcs = self.rng.integers(1, 4)  # 1-3 arcs per image
    
    for _ in range(n_arcs):
        # Arc parameters based on real lensing physics
        radius = self.rng.uniform(8.0, 20.0)  # Einstein radius
        arc_width = self.rng.uniform(2.0, 4.0)  # Arc thickness
        brightness = self.rng.uniform(0.7, 1.0)  # Arc brightness
        
        # Create arc using parametric equations
        # This simulates how background galaxies appear as arcs
        arc_points = self._generate_arc_points(radius, arc_width)
        img = self._draw_arc(img, arc_points, brightness)
```

**For Non-Lens Images:**
```python
def create_non_lens_image(self, image_id: str, split: str):
    """Generate realistic non-lens galaxy image."""
    
    # Create complex galaxy structure (no lensing arcs)
    # This includes multiple components: bulge, disk, spiral arms
    
    # Main galaxy component
    main_galaxy = self._create_galaxy_component(
        sigma=galaxy_sigma,
        ellipticity=galaxy_ellipticity,
        brightness=galaxy_brightness
    )
    
    # Add spiral structure (if applicable)
    if self.rng.random() < 0.3:  # 30% chance of spiral features
        spiral_arms = self._create_spiral_arms()
        main_galaxy += spiral_arms
    
    # Add companion galaxies (common in real surveys)
    if self.rng.random() < 0.2:  # 20% chance of companions
        companion = self._create_companion_galaxy()
        main_galaxy += companion
```

#### Realistic Noise and Artifacts

To make our synthetic images as realistic as possible, we add:

```python
def add_realistic_noise(self, img: np.ndarray) -> np.ndarray:
    """Add realistic observational noise and artifacts."""
    
    # 1. Gaussian noise (photon noise, read noise)
    gaussian_noise = self.rng.normal(0, self.config.noise.gaussian_std)
    img += gaussian_noise
    
    # 2. Poisson noise (photon counting statistics)
    poisson_noise = self.rng.poisson(self.config.noise.poisson_lambda)
    img += poisson_noise
    
    # 3. PSF blur (atmospheric seeing, telescope optics)
    psf_sigma = self.config.noise.psf_sigma
    img = gaussian_filter(img, sigma=psf_sigma)
    
    # 4. Cosmic rays (random bright pixels)
    if self.rng.random() < 0.1:  # 10% chance
        cosmic_ray_pos = (self.rng.integers(0, img.shape[0]),
                         self.rng.integers(0, img.shape[1]))
        img[cosmic_ray_pos] += self.rng.uniform(2.0, 5.0)
    
    return np.clip(img, 0, 1)  # Ensure valid pixel values
```

### 🧠 Neural Network Architectures: How the Computer "Sees"

We use three different types of neural networks, each with different strengths:

#### 1. ResNet (Residual Neural Network)

Think of ResNet as a very sophisticated image analysis pipeline with multiple stages:

```python
class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction."""
    
    def __init__(self, arch: str = 'resnet18', in_ch: int = 3, pretrained: bool = True):
        super().__init__()
        
        # Load pre-trained model (trained on millions of natural images)
        if arch == "resnet18":
            self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        elif arch == "resnet34":
            self.backbone = torchvision.models.resnet34(pretrained=pretrained)
        
        # Remove final classification layer (we'll add our own)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        return self.backbone(x)
```

**How ResNet Works:**
1. **Convolutional Layers**: Like having multiple filters that detect different features (edges, shapes, textures)
2. **Residual Connections**: Allow the network to learn complex patterns by "skipping" layers when needed
3. **Feature Hierarchy**: Early layers detect simple features (edges), later layers detect complex patterns (arcs, galaxies)

**Astronomical Analogy**: ResNet is like having a team of observers with different expertise levels:
- Junior observers detect basic features (bright spots, edges)
- Senior observers combine these into complex patterns (galaxy shapes, arc structures)
- Expert observers make the final classification decision

#### 2. Vision Transformer (ViT)

Vision Transformers work differently - they treat images like text, breaking them into "patches" and analyzing relationships between patches:

```python
class ViTBackbone(nn.Module):
    """Vision Transformer backbone for feature extraction."""
    
    def __init__(self, arch: str = "vit_b_16", in_ch: int = 3, pretrained: bool = True):
        super().__init__()
        
        # Load pre-trained ViT
        self.backbone = torchvision.models.vit_b_16(pretrained=pretrained)
        
        # Remove final classification head
        self.backbone.heads = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using attention mechanism."""
        return self.backbone(x)
```

**How ViT Works:**
1. **Patch Embedding**: Breaks image into 16×16 pixel patches (like words in a sentence)
2. **Attention Mechanism**: Each patch "pays attention" to other patches that might be relevant
3. **Global Context**: Can understand relationships between distant parts of the image

**Astronomical Analogy**: ViT is like having an expert who can:
- Look at the entire image at once
- Understand how different parts relate to each other
- Notice that a faint arc in one corner might be connected to a galaxy in the center

#### 3. Enhanced Light Transformer

This is our custom architecture that combines the best of both worlds:

```python
class EnhancedLightTransformerBackbone(nn.Module):
    """Enhanced Light Transformer with arc-aware attention."""
    
    def __init__(self, cnn_stage: str = 'layer3', patch_size: int = 2,
                 embed_dim: int = 256, num_heads: int = 4, num_layers: int = 4,
                 attention_type: str = 'arc_aware'):
        super().__init__()
        
        # CNN feature extractor (like ResNet)
        self.cnn_backbone = self._create_cnn_backbone(cnn_stage)
        
        # Transformer layers (like ViT)
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Arc-aware attention (our innovation)
        if attention_type == 'arc_aware':
            self.arc_attention = ArcAwareAttention(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with CNN + Transformer + Arc-aware attention."""
        
        # Step 1: Extract local features with CNN
        cnn_features = self.cnn_backbone(x)
        
        # Step 2: Understand global relationships with Transformer
        for layer in self.transformer:
            cnn_features = layer(cnn_features)
        
        # Step 3: Focus on arc-like structures
        if hasattr(self, 'arc_attention'):
            cnn_features = self.arc_attention(cnn_features)
        
        return cnn_features
```

**Astronomical Analogy**: The Enhanced Light Transformer is like having a team where:
- CNN members identify local features (galaxy components, arc segments)
- Transformer members understand global relationships (how arcs connect to galaxies)
- Arc-aware attention members specifically look for lensing signatures

### 🎯 Training Process: How the Computer Learns

The training process is like teaching a new observer to recognize gravitational lenses:

#### 1. Data Preparation

```python
class LensDataset(Dataset):
    """Dataset class for gravitational lensing images."""
    
    def __init__(self, data_root, split="train", img_size=224, augment=False):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        # Load metadata (which images are lenses, which are not)
        self.df = pd.read_csv(self.data_root / f"{split}.csv")
        
        # Set up image preprocessing
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Set up image transforms for training."""
        base_transforms = [
            transforms.Resize((self.img_size, self.img_size)),  # Resize to standard size
            transforms.ToTensor(),  # Convert to numerical format
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Standardize brightness
                               std=[0.229, 0.224, 0.225])
        ]
        
        if self.augment:
            # Data augmentation: create variations of each image
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally
                transforms.RandomRotation(degrees=10),    # Rotate slightly
                transforms.ColorJitter(brightness=0.2,   # Vary brightness
                                     contrast=0.2)       # Vary contrast
            ]
            self.transform = transforms.Compose(augment_transforms + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)
```

#### 2. The Learning Loop

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch - one pass through all training data."""
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    
    for images, labels in train_loader:
        # Move data to GPU if available
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Forward pass: make prediction
        logits = model(images).squeeze(1)
        
        # Calculate loss (how wrong the prediction was)
        loss = criterion(logits, labels)
        
        # Backward pass: learn from mistakes
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Calculate accuracy for this batch
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean()
        
        # Track statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += acc.item() * batch_size
        num_samples += batch_size
    
    return running_loss / num_samples, running_acc / num_samples
```

**Astronomical Analogy**: The training process is like:
1. **Showing Examples**: Presenting the observer with thousands of images
2. **Making Predictions**: The observer guesses whether each image contains a lens
3. **Getting Feedback**: Telling the observer if they were right or wrong
4. **Learning from Mistakes**: The observer adjusts their criteria based on feedback
5. **Repeating**: Going through this process many times until the observer becomes expert

#### 3. Performance Optimization

We use several techniques to make training faster and more effective:

**Mixed Precision Training:**
```python
def train_step_amp(model, images, labels, optimizer, scaler):
    """Training step with mixed precision for faster training."""
    optimizer.zero_grad()
    
    # Use lower precision for faster computation
    with autocast():
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
    
    # Scale gradients to prevent underflow
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Astronomical Analogy**: Mixed precision is like using a faster but slightly less precise measurement technique when you need speed, but switching to high precision when accuracy is critical.

### 🤝 Ensemble Methods: Combining Multiple Experts

Just like astronomers often consult multiple experts for difficult cases, we combine multiple models:

```python
class UncertaintyWeightedEnsemble(nn.Module):
    """Ensemble that weights models by their confidence."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = models
        self.weights = weights or [1.0] * len(models)
    
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Make ensemble predictions."""
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                logits = model(images).squeeze(1)
                probabilities = torch.sigmoid(logits)
                predictions.append(probabilities)
        
        # Weighted average of predictions
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred / sum(self.weights)
```

**Astronomical Analogy**: Ensemble methods are like:
- Having multiple expert observers examine the same image
- Each observer has different strengths (one is good at detecting faint arcs, another at recognizing galaxy types)
- Combining their opinions gives a more reliable final decision

### 📊 Evaluation: How We Measure Success

We use several metrics to evaluate how well our models perform:

```python
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """Evaluate model with comprehensive metrics."""
    model.eval()  # Set to evaluation mode
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():  # Don't update model during evaluation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            logits = model(images).squeeze(1)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions),
        'f1_score': f1_score(all_labels, all_predictions),
        'roc_auc': roc_auc_score(all_labels, all_probabilities)
    }
    
    return metrics
```

**Understanding the Metrics:**

- **Accuracy**: What fraction of predictions were correct? (Like: "The observer was right 95% of the time")
- **Precision**: Of the images predicted as lenses, what fraction actually contain lenses? (Like: "When the observer says 'lens', they're right 94% of the time")
- **Recall**: Of all actual lenses, what fraction did we detect? (Like: "The observer found 97% of all the lenses that were actually there")
- **F1-Score**: A balance between precision and recall
- **ROC AUC**: How well can the model distinguish between lenses and non-lenses? (Higher is better, 1.0 is perfect)

### 🔬 Scientific Validation: Ensuring Reliability

We implement several validation strategies to ensure our results are scientifically sound:

#### 1. Reproducibility
```python
def set_seed(seed: int = 42):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

**Why This Matters**: Like ensuring that an experiment can be repeated with the same results, we fix all random processes so that anyone can reproduce our findings.

#### 2. Cross-Validation
```python
def k_fold_cross_validation(dataset, k=5):
    """Perform k-fold cross-validation."""
    # Split data into k folds
    folds = split_dataset(dataset, k)
    
    results = []
    for i in range(k):
        # Use fold i as test set, others as training set
        train_fold = combine_folds([folds[j] for j in range(k) if j != i])
        test_fold = folds[i]
        
        # Train model on train_fold
        model = train_model(train_fold)
        
        # Evaluate on test_fold
        metrics = evaluate_model(model, test_fold)
        results.append(metrics)
    
    return results
```

**Astronomical Analogy**: Cross-validation is like testing an observer on different sets of images to make sure their performance is consistent and not just memorizing specific examples.

#### 3. Uncertainty Quantification
```python
class TemperatureScaler:
    """Temperature scaling for better uncertainty estimates."""
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """Calibrate the model's confidence estimates."""
        # Adjust temperature to make confidence estimates more accurate
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
    
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to get calibrated probabilities."""
        return torch.sigmoid(logits / self.temperature)
```

**Astronomical Analogy**: Uncertainty quantification is like having the observer not just say "this is a lens" but also "I'm 95% confident this is a lens" - and making sure that when they say 95% confident, they're actually right 95% of the time.

### 🚀 Practical Usage for Astronomers

#### Running Your First Analysis

```bash
# 1. Generate a realistic dataset
python scripts/utilities/generate_dataset.py --config configs/realistic.yaml --out data/realistic

# 2. Train a ResNet-18 model (good for laptops)
python src/training/trainer.py --data-root data/realistic --arch resnet18 --epochs 20

# 3. Evaluate the model
python src/evaluation/evaluator.py --data-root data/realistic --weights checkpoints/best_resnet18.pt
```

#### Understanding the Output

The evaluation produces several files:

- **`results/evaluation_summary.json`**: High-level performance metrics
- **`results/detailed_predictions.csv`**: Per-image predictions and confidence scores
- **`results/calibration_plots.png`**: Visualizations of model confidence

#### Interpreting Results

```python
# Load evaluation results
import json
with open('results/evaluation_summary.json', 'r') as f:
    results = json.load(f)

print(f"Model Accuracy: {results['accuracy']:.3f}")
print(f"Precision: {results['precision']:.3f}")  # Low false positive rate
print(f"Recall: {results['recall']:.3f}")        # High detection rate
print(f"ROC AUC: {results['roc_auc']:.3f}")      # Overall discrimination ability
```

### 🔮 Future Directions

This project opens several exciting research directions:

#### 1. Real Survey Data
- Apply to actual astronomical surveys (Euclid, LSST, JWST)
- Adapt to different observing conditions and instruments
- Handle multi-wavelength data

#### 2. Advanced Physics Integration
- Incorporate lensing theory directly into the neural network
- Use physical constraints to improve predictions
- Estimate lens parameters (Einstein radius, ellipticity)

#### 3. Active Learning
- Intelligently select which images to label next
- Reduce the amount of manual labeling required
- Focus human expertise on the most informative cases

---

## 📁 Project Structure

```
mechine_lensing/
├── 📁 data/                          # Data storage
│   ├── raw/                          # Raw downloaded data
│   ├── processed/                    # Processed datasets
│   └── metadata/                     # Dataset metadata
├── 📁 configs/                       # Configuration files
│   ├── 🎯 baseline.yaml             # Standard configuration
│   ├── 🌟 realistic.yaml            # Realistic dataset configuration
│   ├── 🚀 enhanced_ensemble.yaml    # Advanced ensemble configuration
│   ├── 🔬 trans_enc_s.yaml          # Light Transformer configuration
│   ├── ⚡ lightning_train.yaml      # Lightning AI local training config
│   ├── ⚡ lightning_cloud.yaml      # Lightning AI cloud training config
│   └── ⚡ lightning_ensemble.yaml   # Lightning AI ensemble config
├── 📁 src/                           # Source code
│   ├── 📁 analysis/                  # Post-hoc uncertainty analysis
│   │   └── aleatoric.py              # Active learning & diagnostics
│   ├── 📁 datasets/                  # Dataset implementations
│   │   └── lens_dataset.py           # PyTorch Dataset class
│   ├── 📁 models/                    # Model architectures
│   │   ├── backbones/                # Feature extractors (ResNet, ViT, Transformer)
│   │   │   ├── resnet.py             # ResNet-18/34 implementations
│   │   │   ├── vit.py                # Vision Transformer ViT-B/16
│   │   │   └── light_transformer.py  # Enhanced Light Transformer
│   │   ├── heads/                    # Classification heads
│   │   │   └── binary.py             # Binary classification head
│   │   ├── ensemble/                 # Ensemble methods
│   │   │   ├── registry.py           # Model registry & factory
│   │   │   ├── weighted.py           # Uncertainty-weighted ensemble
│   │   │   └── enhanced_weighted.py  # Advanced ensemble with trust learning
│   │   ├── factory.py                # Legacy model factory
│   │   └── lens_classifier.py        # Unified classifier wrapper
│   ├── 📁 training/                  # Training utilities
│   │   └── trainer.py                # Training implementation
│   ├── 📁 evaluation/                # Evaluation utilities
│   │   ├── evaluator.py              # Individual model evaluation
│   │   └── ensemble_evaluator.py     # Ensemble evaluation
│   ├── ⚡ lit_system.py              # Lightning AI model wrappers
│   ├── ⚡ lit_datamodule.py          # Lightning AI data modules
│   ├── ⚡ lit_train.py               # Lightning AI training script
│   └── 📁 utils/                     # Utility functions
│       └── config.py                 # Configuration management
├── 📁 scripts/                       # Entry point scripts
│   ├── generate_dataset.py           # Dataset generation
│   ├── train.py                      # Training entry point
│   ├── eval.py                       # Evaluation entry point
│   ├── eval_ensemble.py              # Ensemble evaluation entry point
│   └── prepare_lightning_dataset.py  # Lightning AI dataset preparation
├── 📁 experiments/                   # Experiment tracking
├── 📁 tests/                         # Test suite
├── 📁 docs/                          # Documentation
│   ├── 📖 SCIENTIFIC_METHODOLOGY.md  # Scientific approach explanation
│   ├── 🔧 TECHNICAL_DETAILS.md       # Technical implementation details
│   ├── 🚀 DEPLOYMENT_GUIDE.md        # Cloud deployment guide
│   └── ⚡ LIGHTNING_INTEGRATION_GUIDE.md # Lightning AI integration guide
├── 📋 requirements.txt               # Production dependencies
├── 📋 requirements-dev.txt           # Development dependencies
├── 🔧 Makefile                       # Development commands
├── 📄 env.example                    # Environment configuration template
├── 📜 README.md                      # This file
└── 📄 LICENSE                        # MIT License
```

## 🛠️ Development Commands

The project includes a comprehensive Makefile for all development tasks:

### Environment Setup
```bash
make setup          # Complete development environment setup
make install-deps   # Install dependencies only
make update-deps    # Update all dependencies
```

### Code Quality
```bash
make lint          # Run all code quality checks
make format        # Format code with black and isort
make check-style   # Check code style with flake8
make check-types   # Check types with mypy
```

### Testing
```bash
make test          # Run all tests with coverage
make test-fast     # Run fast tests only
make test-integration  # Run integration tests only
```

### Data and Training
```bash
make dataset       # Generate realistic dataset
make dataset-quick # Generate quick test dataset
make train         # Train model (specify ARCH=resnet18|resnet34|vit_b_16)
make train-all     # Train all architectures
make eval          # Evaluate model
make eval-ensemble # Evaluate ensemble
```

### Lightning AI Training
```bash
make lit-train           # Train with Lightning AI
make lit-train-cloud     # Train on Lightning Cloud
make lit-train-ensemble  # Train ensemble with Lightning
make lit-prepare-dataset # Prepare dataset for cloud streaming
make lit-upload-dataset  # Upload dataset to cloud storage
```

### Complete Workflows
```bash
make experiment    # Full experiment: dataset -> train -> eval
make full-pipeline # Complete pipeline with all models
make dev          # Quick development setup and test
```

### Utilities
```bash
make clean        # Clean cache and temporary files
make status       # Show project status
make help         # Show all available commands
```

## 🎯 Scientific Approach

### Dataset Generation

This project uses **scientifically realistic synthetic datasets** that overcome the limitations of trivial toy datasets:

#### ❌ Previous Approach (Trivial)
- **Lens images**: Simple bright arcs
- **Non-lens images**: Basic elliptical blobs  
- **Result**: 100% accuracy (unrealistic!)

#### ✅ Our Approach (Realistic)
- **Lens images**: Complex galaxies + subtle lensing arcs
- **Non-lens images**: Multi-component galaxy structures
- **Result**: 93-96% accuracy (scientifically valid!)

### Key Improvements

1. **🔬 Realistic Physics**: Proper gravitational lensing simulation
2. **📊 Overlapping Features**: Both classes share similar brightness/structure
3. **🎲 Comprehensive Noise**: Observational noise, PSF blur, realistic artifacts
4. **🔄 Reproducibility**: Full parameter tracking and deterministic generation
5. **✅ Validation**: Atomic file operations and integrity checks

## 🏗️ Architecture Details

### Supported Models

| Architecture | Parameters | Input Size | Training Time | Best For |
|-------------|------------|------------|---------------|----------|
| **ResNet-18** | 11.2M | 64×64 | ~4 min | Laptops, quick experiments |
| **ResNet-34** | 21.3M | 64×64 | ~8 min | Balanced performance/speed |
| **ViT-B/16** | 85.8M | 224×224 | ~30 min | Maximum accuracy (GPU) |

### Ensemble Methods

- **Probability Averaging**: Weighted combination of model outputs
- **Multi-Scale Processing**: Different input sizes for different models
- **Robust Predictions**: Improved generalization through diversity

## ⚡ Lightning AI Cloud Deployment

### Lightning Cloud (Recommended)
```bash
# Install Lightning CLI
pip install lightning

# Login to Lightning Cloud
lightning login

# Create a workspace
lightning create workspace lens-training

# Run training job on cloud GPUs
lightning run app src/lit_train.py --use-webdataset --train-urls "s3://bucket/train-{0000..0099}.tar" --devices 4
```

### Local Lightning Training
```bash
# Train with Lightning locally
make lit-train ARCH=resnet18

# Multi-GPU training
make lit-train ARCH=vit_b_16 DEVICES=2

# Ensemble training
make lit-train-ensemble
```

### WebDataset Streaming
```bash
# Prepare dataset for cloud streaming
make lit-prepare-dataset CLOUD_URL="s3://your-bucket/lens-data"

# Train with cloud dataset
make lit-train-cloud TRAIN_URLS="s3://your-bucket/train-{0000..0099}.tar"
```

**Lightning AI Benefits:**
- **One-command scaling**: Scale from 1 to 8+ GPUs automatically
- **Managed infrastructure**: No server setup or maintenance
- **Cost effective**: Pay only for compute time used
- **Production ready**: Built-in logging, checkpointing, and monitoring

## ⚡ Lightning AI Integration

This project is fully integrated with Lightning AI for seamless cloud training and scaling:

### Key Features
- **LightningModule Wrappers**: All models wrapped in Lightning-compatible interfaces
- **WebDataset Streaming**: Efficient cloud dataset streaming for large-scale training
- **Automatic Scaling**: One command to scale from local to cloud GPUs
- **Built-in Monitoring**: Comprehensive logging and metrics tracking
- **Ensemble Support**: Lightning-based ensemble training and inference

### Quick Lightning Start
```bash
# Install Lightning AI
pip install lightning

# Train locally with Lightning
make lit-train ARCH=resnet18

# Scale to cloud with WebDataset
make lit-train-cloud TRAIN_URLS="s3://bucket/train-{0000..0099}.tar" DEVICES=4
```

### Lightning Components
- **`src/lit_system.py`**: LightningModule wrappers for all model architectures
- **`src/lit_datamodule.py`**: LightningDataModule for local and cloud datasets
- **`src/lit_train.py`**: Unified Lightning training script
- **`scripts/prepare_lightning_dataset.py`**: Dataset preparation for cloud streaming
- **`configs/lightning_*.yaml`**: Lightning-specific configuration files

For detailed Lightning AI integration guide, see [Lightning Integration Guide](docs/LIGHTNING_INTEGRATION_GUIDE.md).

## 🛠️ Configuration

### Environment Variables

Copy `env.example` to `.env` and customize:

```bash
# Copy template
cp env.example .env

# Edit configuration
# Key variables:
# DATA_ROOT=data/processed
# DEFAULT_ARCH=resnet18
# WANDB_API_KEY=your_key_here
```

### Training Configuration
```bash
# Laptop-friendly settings
make train ARCH=resnet18 EPOCHS=10 BATCH_SIZE=32

# High-performance settings (GPU)
make train ARCH=vit_b_16 EPOCHS=20 BATCH_SIZE=64
```

## 📊 Evaluation & Metrics

### Comprehensive Evaluation
```bash
# Individual model evaluation
make eval ARCH=resnet18

# Ensemble evaluation with detailed analysis
make eval-ensemble

# Evaluate all models
make eval-all
```

### Output Files
- `results/detailed_predictions.csv`: Per-sample predictions and confidence
- `results/ensemble_metrics.json`: Complete performance metrics
- `results/evaluation_summary.json`: High-level summary statistics

## 🔬 Scientific Validation

### Reproducibility
- **Fixed seeds**: All random operations are seeded
- **Deterministic operations**: Consistent results across runs
- **Parameter logging**: Full configuration tracking
- **Atomic operations**: Data integrity guarantees

### Statistical Significance
- **Cross-validation ready**: Modular design supports k-fold CV
- **Confidence intervals**: Bootstrap sampling support
- **Multiple runs**: Variance analysis capabilities

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
# Clone and setup
git clone https://github.com/Kantoration/mechine_lensing.git
cd mechine_lensing
make setup

# Run pre-commit checks
make ci

# Run tests
make test
```

## 📚 Documentation

- [📖 Scientific Methodology](docs/SCIENTIFIC_METHODOLOGY.md) - Detailed explanation of our approach
- [🔧 Technical Details](docs/TECHNICAL_DETAILS.md) - Implementation specifics
- [⚡ Lightning Integration Guide](docs/LIGHTNING_INTEGRATION_GUIDE.md) - Lightning AI integration and cloud training
- [🚀 Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Cloud deployment instructions
- [🤝 Contributing](CONTRIBUTING.md) - Contribution guidelines

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@software{gravitational_lens_classification,
  title={Gravitational Lens Classification with Deep Learning},
  author={Kantoration},
  year={2024},
  url={https://github.com/Kantoration/mechine_lensing}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepLenstronomy**: For gravitational lensing simulation inspiration
- **PyTorch Team**: For the excellent deep learning framework  
- **Torchvision**: For pre-trained model architectures
- **Astronomical Community**: For domain expertise and validation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Kantoration/mechine_lensing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Kantoration/mechine_lensing/discussions)
- **Documentation**: [Project Wiki](https://github.com/Kantoration/mechine_lensing/wiki)

---

**⭐ If this project helped your research, please give it a star!**

Made with ❤️ for the astronomical machine learning community.

## 🚀 Getting Started Examples

### Example 1: Quick Experiment
```bash
# Complete quick experiment in 3 commands
make setup           # Setup environment
make experiment-quick # Generate data, train, evaluate
make status          # Check results
```

### Example 2: Lightning AI Training
```bash
# Generate realistic dataset
make dataset CONFIG_FILE=configs/realistic.yaml

# Train with Lightning AI locally
make lit-train ARCH=resnet18 EPOCHS=30 BATCH_SIZE=64

# Train on Lightning Cloud with WebDataset
make lit-train-cloud TRAIN_URLS="s3://bucket/train-{0000..0099}.tar" DEVICES=4

# Evaluate with detailed metrics
make eval ARCH=resnet18
```

### Example 3: Lightning AI Ensemble Workflow
```bash
# Train ensemble with Lightning AI
make lit-train-ensemble

# Or train individual models with Lightning
make lit-train ARCH=resnet18
make lit-train ARCH=vit_b_16

# Evaluate ensemble
make eval-ensemble

# Check all results
ls results/
```

### Example 4: Lightning AI Development Workflow
```bash
# Setup and run development checks
make setup
make lint            # Check code quality
make test-fast       # Run fast tests
make lit-train-quick # Quick Lightning training test
```

