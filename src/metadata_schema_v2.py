#!/usr/bin/env python3
"""
metadata_schema_v2.py
====================
Metadata Schema V2.0 for Gravitational Lensing Datasets

PRIORITY 0 FIXES IMPLEMENTED:
- Label provenance tracking (sim:bologna | obs:castles | weak:gzoo | pretrain:galaxiesml)
- Extended stratification parameters (z, mag, seeing, PSF FWHM, pixel scale, survey)
- Variance map support for uncertainty-weighted training
- PSF matching metadata for cross-survey homogenization

Author: Gravitational Lensing ML Team
Version: 2.0.0 (Post-Scientific-Review)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class ImageMetadataV2:
    """
    Metadata schema v2.0 with label provenance and extended observational parameters.
    
    CRITICAL FIELDS FOR PRIORITY 0 FIXES:
    - label_source: Track data provenance for proper usage
    - variance_map_available: Flag for variance-weighted loss
    - psf_fwhm, seeing, pixel_scale: For stratification and FiLM conditioning
    - schema_version: Version tracking for compatibility
    """
    
    # ============================================================================
    # REQUIRED FIELDS (no defaults)
    # ============================================================================
    
    # File paths
    filepath: str
    
    # Label Provenance (CRITICAL for proper dataset usage)
    label: int  # 0=non-lens, 1=lens, -1=unlabeled
    label_source: str  # 'sim:bologna' | 'obs:castles' | 'weak:gzoo' | 'pretrain:galaxiesml'
    label_confidence: float  # 0.0-1.0 (1.0 for Bologna/CASTLES, <0.5 for weak)
    
    # ============================================================================
    # OPTIONAL FIELDS (with defaults)
    # ============================================================================
    
    # File paths
    variance_map_path: Optional[str] = None
    
    # Redshift information
    z_phot: float = -1.0  # photometric redshift (-1 if missing)
    z_spec: float = -1.0  # spectroscopic redshift (-1 if missing)
    z_err: float = -1.0   # redshift error
    
    # Observational Parameters (CRITICAL for stratification)
    seeing: float = 1.0      # arcsec (atmospheric seeing)
    psf_fwhm: float = 0.8    # arcsec (PSF FWHM - CRITICAL for PSF-sensitive arcs)
    pixel_scale: float = 0.2 # arcsec/pixel
    instrument: str = "unknown"
    survey: str = "unknown"  # 'hsc' | 'sdss' | 'hst' | 'des' | 'kids' | 'relics'
    
    # Photometry
    magnitude: float = 20.0  # apparent magnitude
    snr: float = 10.0        # signal-to-noise ratio
    
    # Physical properties (for auxiliary tasks)
    sersic_index: float = 2.0        # Sérsic index
    half_light_radius: float = 1.0   # arcsec
    axis_ratio: float = 0.7          # b/a (minor/major axis ratio)
    
    # Quality flags
    variance_map_available: bool = False
    psf_matched: bool = False
    target_psf_fwhm: float = -1.0    # Target PSF for homogenization
    
    # Schema versioning
    schema_version: str = "2.0"


# ============================================================================
# LABEL SOURCE CONSTANTS
# ============================================================================

class LabelSources:
    """Constants for label source tracking."""
    
    # Simulation datasets
    SIM_BOLOGNA = "sim:bologna"           # Bologna Challenge (primary training)
    SIM_DEEPLENSTRONOMY = "sim:deeplens"  # deeplenstronomy simulations
    
    # Observational datasets
    OBS_CASTLES = "obs:castles"           # CASTLES confirmed lenses (positive-only)
    OBS_RELICS = "obs:relics"             # RELICS survey (hard negatives)
    
    # Weak supervision
    WEAK_GALAXY_ZOO = "weak:gzoo"         # Galaxy Zoo weak labels
    
    # Pretraining
    PRETRAIN_GALAXIESML = "pretrain:galaxiesml"  # GalaxiesML (NO lens labels)
    
    # All valid sources
    VALID_SOURCES = {
        SIM_BOLOGNA, SIM_DEEPLENSTRONOMY,
        OBS_CASTLES, OBS_RELICS,
        WEAK_GALAXY_ZOO, PRETRAIN_GALAXIESML
    }


# ============================================================================
# SURVEY CONSTANTS
# ============================================================================

class Surveys:
    """Constants for survey identification."""
    
    HSC = "hsc"           # Hyper Suprime-Cam
    SDSS = "sdss"         # Sloan Digital Sky Survey
    HST = "hst"           # Hubble Space Telescope
    DES = "des"           # Dark Energy Survey
    KIDS = "kids"         # Kilo-Degree Survey
    RELICS = "relics"     # RELICS survey
    CASTLES = "castles"   # CASTLES survey
    UNKNOWN = "unknown"   # Unknown/unspecified survey
    
    # All valid surveys
    VALID_SURVEYS = {HSC, SDSS, HST, DES, KIDS, RELICS, CASTLES, UNKNOWN}


# ============================================================================
# USAGE GUIDANCE
# ============================================================================

class DatasetUsage:
    """
    Critical usage guidance for different label sources.
    
    This prevents common mistakes in dataset usage.
    """
    
    USAGE_GUIDANCE = {
        LabelSources.SIM_BOLOGNA: {
            "usage": "PRIMARY TRAINING",
            "description": "Full labels, use for main training",
            "confidence": 1.0,
            "warnings": []
        },
        
        LabelSources.OBS_CASTLES: {
            "usage": "POSITIVE-ONLY",
            "description": "Confirmed lenses only - MUST pair with hard negatives",
            "confidence": 1.0,
            "warnings": [
                "⚠️  CASTLES is POSITIVE-ONLY",
                "   → Build hard negatives from RELICS non-lensed cores",
                "   → Or use matched galaxies from same survey"
            ]
        },
        
        LabelSources.PRETRAIN_GALAXIESML: {
            "usage": "PRETRAINING ONLY",
            "description": "NO lens labels - use for pretraining only",
            "confidence": 0.0,
            "warnings": [
                "⚠️  GalaxiesML has NO LENS LABELS",
                "   → Use for pretraining/self-supervised learning only",
                "   → DO NOT use for lens classification training"
            ]
        },
        
        LabelSources.WEAK_GALAXY_ZOO: {
            "usage": "WEAK SUPERVISION",
            "description": "Weak labels from citizen science",
            "confidence": 0.3,
            "warnings": [
                "⚠️  Galaxy Zoo labels are WEAK",
                "   → Use with uncertainty weighting",
                "   → Validate against confirmed lenses"
            ]
        }
    }
    
    @classmethod
    def get_usage_guidance(cls, label_source: str) -> Dict[str, Any]:
        """Get usage guidance for a label source."""
        return cls.USAGE_GUIDANCE.get(label_source, {
            "usage": "UNKNOWN",
            "description": "Unknown label source",
            "confidence": 0.0,
            "warnings": ["⚠️  Unknown label source - verify usage"]
        })


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_metadata(metadata: ImageMetadataV2) -> bool:
    """
    Validate metadata against schema v2.0.
    
    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    if not metadata.filepath:
        return False
    
    if metadata.label not in [-1, 0, 1]:
        return False
    
    if metadata.label_source not in LabelSources.VALID_SOURCES:
        return False
    
    if not (0.0 <= metadata.label_confidence <= 1.0):
        return False
    
    # Check survey
    if metadata.survey not in Surveys.VALID_SURVEYS:
        return False
    
    # Check redshift values
    if metadata.z_phot != -1.0 and not (0.0 <= metadata.z_phot <= 10.0):
        return False
    
    if metadata.z_spec != -1.0 and not (0.0 <= metadata.z_spec <= 10.0):
        return False
    
    return True


def create_metadata_dataframe(metadata_list: list[ImageMetadataV2]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a list of metadata objects.
    
    Args:
        metadata_list: List of ImageMetadataV2 objects
        
    Returns:
        pandas DataFrame with metadata
    """
    data = [vars(meta) for meta in metadata_list]
    df = pd.DataFrame(data)
    
    # Validate all metadata
    for idx, meta in enumerate(metadata_list):
        if not validate_metadata(meta):
            raise ValueError(f"Invalid metadata at index {idx}")
    
    return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the metadata schema."""
    
    # Create metadata for a Bologna Challenge simulation
    bologna_meta = ImageMetadataV2(
        filepath="train/sim_bologna_000001.tif",
        label=1,
        label_source=LabelSources.SIM_BOLOGNA,
        label_confidence=1.0,
        z_spec=0.5,
        seeing=0.6,
        psf_fwhm=0.6,
        pixel_scale=0.168,
        survey=Surveys.HSC,
        sersic_index=2.5,
        half_light_radius=1.2,
        schema_version="2.0"
    )
    
    # Create metadata for a CASTLES lens
    castles_meta = ImageMetadataV2(
        filepath="train/lens_castles_000001.tif",
        label=1,
        label_source=LabelSources.OBS_CASTLES,
        label_confidence=1.0,
        z_spec=0.8,
        seeing=0.1,
        psf_fwhm=0.1,
        pixel_scale=0.05,
        survey=Surveys.HST,
        variance_map_available=True,
        variance_map_path="train/lens_castles_000001_var.tif",
        schema_version="2.0"
    )
    
    # Create metadata for GalaxiesML (pretraining)
    galaxiesml_meta = ImageMetadataV2(
        filepath="train/galaxiesml_pretrain_000001.tif",
        label=-1,  # No label
        label_source=LabelSources.PRETRAIN_GALAXIESML,
        label_confidence=0.0,  # No lens labels
        z_spec=1.2,
        seeing=0.6,
        psf_fwhm=0.6,
        pixel_scale=0.168,
        survey=Surveys.HSC,
        schema_version="2.0"
    )
    
    # Get usage guidance
    bologna_guidance = DatasetUsage.get_usage_guidance(LabelSources.SIM_BOLOGNA)
    castles_guidance = DatasetUsage.get_usage_guidance(LabelSources.OBS_CASTLES)
    galaxiesml_guidance = DatasetUsage.get_usage_guidance(LabelSources.PRETRAIN_GALAXIESML)
    
    print("Bologna Challenge usage:", bologna_guidance["usage"])
    print("CASTLES usage:", castles_guidance["usage"])
    print("GalaxiesML usage:", galaxiesml_guidance["usage"])
    
    # Create DataFrame
    df = create_metadata_dataframe([bologna_meta, castles_meta, galaxiesml_meta])
    print("\nMetadata DataFrame:")
    print(df[['filepath', 'label', 'label_source', 'label_confidence', 'survey']].to_string())


if __name__ == "__main__":
    example_usage()
