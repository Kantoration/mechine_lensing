# Phase 2 Implementation Guide: Critical Enhancements

## 🎯 **Status: Ready for Implementation**

This document outlines the **immediate focus areas** for completing Phase 2 of the gravitational lensing system, building on the completed Priority 0 fixes.

---

## ✅ **Completed in Phase 1**

- ✅ Label provenance tracking with metadata schema v2.0
- ✅ 16-bit TIFF format with variance maps
- ✅ Fourier-domain PSF matching
- ✅ Dataset converter (`scripts/convert_real_datasets.py`)
- ✅ Lightning AI infrastructure and testing
- ✅ **Bologna metrics implementation** (`src/metrics/bologna_metrics.py`)

---

## 🔧 **Phase 2: Areas for Immediate Focus**

### **1. ✅ Bologna Challenge Metrics (COMPLETED)**

**Status**: ✅ **Fully Implemented**

**Implementation**: `src/metrics/bologna_metrics.py`

**Key Features**:
- `compute_tpr_at_fpr()`: TPR@FPR=0 and TPR@FPR=0.1 metrics
- `compute_flux_ratio_stratified_metrics()`: FNR tracking for low flux-ratio lenses
- `compute_bologna_metrics()`: Complete evaluation suite
- `format_bologna_metrics()`: Readable output formatting
- PyTorch-friendly wrappers

**Usage**:
```python
from src.metrics.bologna_metrics import compute_bologna_metrics, format_bologna_metrics

# Compute all Bologna metrics
metrics = compute_bologna_metrics(y_true, y_probs, flux_ratios)

# Print formatted results
print(format_bologna_metrics(metrics))

# Key metrics:
print(f"TPR@FPR=0: {metrics['tpr_at_fpr_0']:.3f}")
print(f"TPR@FPR=0.1: {metrics['tpr_at_fpr_0.1']:.3f}")
print(f"Low flux-ratio FNR: {metrics['low_flux_fnr']:.3f}")
```

**Next Steps**:
- ✅ Implementation complete
- 📋 Integration with evaluation pipeline
- 📋 Add to training logging

---

### **2. 🔄 Physics-Informed Loss Enhancement (In Progress)**

**Current Status**: Basic physics-informed architecture exists

**What Exists**:
- ✅ `PhysicsInformedModule` interface (`src/models/interfaces/physics_capable.py`)
- ✅ `PhysicsRegularizedAttention` (`src/models/attention/physics_regularized_attention.py`)
- ✅ `PhysicsInformedEnsemble` (`src/models/ensemble/physics_informed_ensemble.py`)

**What Needs Enhancement**:
- 📋 Soft-gated physics loss (replace hard thresholds)
- 📋 Batched simulator calls with caching
- 📋 Curriculum weighting

**Recommended Implementation**:
```python
def _compute_physics_loss(self, images, logits, metadata=None):
    """
    Enhanced physics loss with soft gating and batched simulation.
    """
    # Soft gate: continuous weighting instead of hard threshold
    probs = torch.sigmoid(logits)
    gate_weights = probs  # Weight by confidence
    
    # Batched simulator call with caching
    lens_params_batch = self.prediction_to_params_batch(logits, metadata)
    synthetic_images = self.differentiable_simulator.render_batch(
        lens_params_batch,
        cache_invariants=True  # Cache PSFs, source grids
    )
    
    # Per-sample consistency loss
    consistency_loss = F.mse_loss(
        images, synthetic_images, reduction='none'
    ).mean(dim=(1, 2, 3))
    
    # Curriculum weighting (anneal from 0.1 to 1.0)
    curriculum_weight = min(1.0, self.current_epoch / self.hparams.physics_warmup_epochs)
    
    # Combine with soft gating
    weighted_loss = (gate_weights * consistency_loss * curriculum_weight).mean()
    
    return weighted_loss
```

**Action Items**:
1. Add `_compute_physics_loss()` to `LitAdvancedLensSystem`
2. Implement `DifferentiableLensingSimulator.render_batch()`
3. Add curriculum scheduler to trainer config
4. Test on validation set

---

### **3. 📋 Extended Stratification (Planned)**

**Current Status**: Basic stratification in dataset converter

**What Needs Enhancement**:
- Extended stratification function with 7 factors
- Seeing, PSF FWHM, pixel scale bins
- Survey-aware splitting

**Recommended Implementation**:
```python
def create_stratified_splits_v2(
    metadata_df: pd.DataFrame,
    factors: List[str] = ['redshift', 'magnitude', 'seeing', 'psf_fwhm', 
                          'pixel_scale', 'survey', 'label']
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create 7-factor stratified splits for robust validation.
    """
    # Create bins for continuous variables
    strat_key = ""
    for factor in factors:
        if factor in ['redshift', 'magnitude']:
            bins = pd.qcut(metadata_df[factor].fillna(0), q=5, labels=False, duplicates='drop')
        elif factor in ['seeing', 'psf_fwhm', 'pixel_scale']:
            bins = pd.qcut(metadata_df[factor].fillna(1.0), q=3, labels=False, duplicates='drop')
        else:
            bins = metadata_df[factor].astype(str)
        
        strat_key = strat_key + bins.astype(str) + '_'
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        metadata_df, test_size=0.3, stratify=strat_key, random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=strat_key[temp_df.index], random_state=42
    )
    
    return train_df, val_df, test_df
```

**Action Items**:
1. Add `create_stratified_splits_v2()` to dataset converter
2. Test on GalaxiesML and CASTLES
3. Verify label balance across all factors
4. Add to documentation

---

### **4. 📋 Memory-Efficient Ensemble Training (Planned)**

**Current Status**: Conceptual design ready

**Implementation Needed**:
- Sequential model training with state cycling
- Adaptive batch size callback
- GPU memory profiling

**Recommended Approach**:
Use existing `PhysicsInformedEnsemble` as base and add:
1. Model state checkpointing between cycles
2. Automatic GPU memory clearing
3. Round-robin training schedule

**Action Items**:
1. Extend `PhysicsInformedEnsemble` with sequential mode
2. Add `AdaptiveBatchSizeCallback`
3. Profile memory usage on A100 GPUs
4. Document sequential vs. parallel trade-offs

---

### **5. 📋 Cross-Survey Validation (Planned)**

**Current Status**: PSF normalization implemented

**What's Needed**:
- Test harness for HSC/SDSS/HST validation
- Cross-survey performance metrics
- Domain adaptation evaluation

**Recommended Validation Protocol**:
```python
def validate_cross_survey(model, datasets):
    """
    Validate model performance across multiple surveys.
    """
    results = {}
    for survey_name, (test_loader, survey_config) in datasets.items():
        # Compute Bologna metrics per survey
        metrics = compute_bologna_metrics(...)
        results[survey_name] = metrics
        
        # Check for domain shift
        if results[survey_name]['auroc'] < 0.85:
            logger.warning(f"Possible domain shift detected for {survey_name}")
    
    return results
```

**Action Items**:
1. Create cross-survey test datasets
2. Implement validation harness
3. Measure performance degradation
4. Document findings

---

## 📊 **Implementation Priority Matrix**

| **Task** | **Priority** | **Status** | **Est. Time** | **Dependencies** |
|----------|-------------|------------|---------------|------------------|
| Bologna Metrics | P0 | ✅ Complete | - | None |
| Physics Loss Enhancement | P1 | 🔄 Design Ready | 2-3 days | Differentiable simulator |
| Extended Stratification | P1 | 📋 Planned | 1-2 days | Dataset converter |
| Memory-Efficient Ensemble | P2 | 📋 Planned | 3-4 days | Physics ensemble |
| Cross-Survey Validation | P2 | 📋 Planned | 2-3 days | Test datasets |

---

## 🚀 **Quick Start Commands**

### Test Bologna Metrics
```bash
# Run Bologna metrics on test data
cd demo/lens-demo
python -c "
from src.metrics.bologna_metrics import compute_bologna_metrics
import numpy as np

# Load your predictions
y_true = np.load('results/y_true.npy')
y_probs = np.load('results/y_probs.npy')
flux_ratios = np.load('results/flux_ratios.npy')  # Optional

# Compute metrics
metrics = compute_bologna_metrics(y_true, y_probs, flux_ratios)

# Print results
from src.metrics.bologna_metrics import format_bologna_metrics
print(format_bologna_metrics(metrics))
"
```

### Enhanced Physics Training (When Implemented)
```bash
# Train with enhanced physics loss
python src/lit_train.py \
    --config configs/pinn_lens.yaml \
    --model.use_physics=true \
    --model.physics_weight=0.2 \
    --model.physics_warmup_epochs=10 \
    --trainer.devices=2
```

### Extended Stratification (When Implemented)
```bash
# Convert with extended stratification
python scripts/convert_real_datasets.py \
    --dataset galaxiesml \
    --input data/raw/GalaxiesML/train.h5 \
    --output data/processed/real \
    --split train \
    --stratify-extended
```

---

## 📚 **Additional Resources**

- **Bologna Metrics**: [src/metrics/bologna_metrics.py](../src/metrics/bologna_metrics.py)
- **Physics Interfaces**: [src/models/interfaces/physics_capable.py](../src/models/interfaces/physics_capable.py)
- **Priority 0 Guide**: [PRIORITY_0_FIXES_GUIDE.md](PRIORITY_0_FIXES_GUIDE.md)
- **Integration Plan**: [INTEGRATION_IMPLEMENTATION_PLAN.md](INTEGRATION_IMPLEMENTATION_PLAN.md)

---

## ✅ **Verification Checklist**

**Phase 2 Readiness**:
- [x] Bologna metrics implemented and tested
- [ ] Physics loss enhancement designed
- [ ] Extended stratification specified
- [ ] Memory-efficient ensemble planned
- [ ] Cross-survey validation protocol defined

**Next Milestone**: Complete physics loss enhancement and test on validation set

---

*Last Updated: 2025-10-03*
*Status: Bologna Metrics Complete | Physics Enhancement Ready | Phase 2 In Progress*

