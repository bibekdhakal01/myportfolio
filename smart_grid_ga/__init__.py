"""
Smart Grid Genetic Algorithm Optimization System

A parallelized Genetic Algorithm-based optimization system for smart grid energy distribution.
"""

from .core.genetic_algorithm import SmartGridGA
from .models.grid_model import SmartGrid
from .models.chromosome import GridChromosome

__version__ = "1.0.0"
__author__ = "Bibek Dhakal"

__all__ = [
    'SmartGridGA',
    'SmartGrid', 
    'GridChromosome'
]