"""
Model Faker System for Elelem Testing

A comprehensive fake model system for testing Elelem without real API calls,
allowing precise control over responses and detailed analysis of requests.
"""

from .server import ModelFaker
from .config import FakerConfig
from .scenarios import ScenarioManager

__all__ = ['ModelFaker', 'FakerConfig', 'ScenarioManager']