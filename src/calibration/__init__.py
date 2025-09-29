"""
Model calibration utilities for improving prediction confidence.
"""

from .temperature import TemperatureScaler, fit_temperature_scaling

__all__ = [
    'TemperatureScaler',
    'fit_temperature_scaling'
]
