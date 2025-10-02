#!/usr/bin/env python3
"""
validation/__init__.py
======================
Comprehensive physics validation framework for gravitational lensing models.

Key Features:
- Realistic lens models (SIE, NFW, composite)
- Source reconstruction validation
- Uncertainty quantification
- Enhanced reporting with visualizations
- Machine-readable output
- Lensing-specific metrics (Einstein radius, arc multiplicity, parity)
- Time delay distribution validation
- Physics constraint verification
- Attention map analysis for scientific interpretation
- Benchmarking against classical methods

Usage:
    from validation import (
        PhysicsValidator, LensingMetricsValidator, UncertaintyValidator,
        SourceQualityValidator, RealisticLensValidator, EnhancedReporter,
        validate_attention_physics, validate_lensing_physics, 
        validate_predictive_uncertainty, validate_source_quality
    )
"""

from .physics_validator import PhysicsValidator, validate_attention_physics, create_physics_validation_report
from .lensing_metrics import LensingMetricsValidator, validate_lensing_physics, create_lensing_validation_report
from .uncertainty_metrics import UncertaintyValidator, validate_predictive_uncertainty, create_uncertainty_validation_report
from .source_reconstruction import SourceQualityValidator, validate_source_quality, create_source_validation_report
from .realistic_lens_models import (
    SIELensModel, NFWLensModel, CompositeLensModel, 
    RealisticLensValidator, create_realistic_lens_models
)
from .enhanced_reporting import EnhancedReporter, create_comprehensive_report
from .visualization import AttentionVisualizer, create_physics_plots, create_attention_analysis_report

__all__ = [
    # Core validators
    'PhysicsValidator',
    'LensingMetricsValidator', 
    'UncertaintyValidator',
    'SourceQualityValidator',
    'RealisticLensValidator',
    'EnhancedReporter',
    'AttentionVisualizer',
    
    # Lens models
    'SIELensModel',
    'NFWLensModel', 
    'CompositeLensModel',
    'create_realistic_lens_models',
    
    # Validation functions
    'validate_attention_physics',
    'validate_lensing_physics',
    'validate_predictive_uncertainty', 
    'validate_source_quality',
    
    # Report generation
    'create_physics_validation_report',
    'create_lensing_validation_report',
    'create_uncertainty_validation_report',
    'create_source_validation_report',
    'create_comprehensive_report',
    'create_physics_plots',
    'create_attention_analysis_report'
]




