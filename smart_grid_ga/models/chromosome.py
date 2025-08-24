"""
Chromosome representation for smart grid energy distribution optimization.

This module defines the chromosome encoding scheme for representing grid configurations,
including generators, loads, and transmission lines in the genetic algorithm.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import random


class GridChromosome:
    """
    Represents a chromosome encoding for smart grid configuration.
    
    The chromosome encodes:
    - Generator dispatch levels (power output for each generator)
    - Load distribution (demand allocation across nodes)
    - Transmission line status (on/off state)
    - Voltage levels at each node
    """
    
    def __init__(self, grid_size: Tuple[int, int], num_generators: int, num_loads: int):
        """
        Initialize a grid chromosome.
        
        Args:
            grid_size: Tuple of (rows, cols) for the grid dimensions
            num_generators: Number of generators in the grid
            num_loads: Number of load points in the grid
        """
        self.grid_size = grid_size
        self.num_generators = num_generators
        self.num_loads = num_loads
        self.num_nodes = grid_size[0] * grid_size[1]
        
        # Initialize chromosome components
        self.generator_dispatch = np.zeros(num_generators)  # MW output levels
        self.load_allocation = np.zeros(num_loads)  # Load distribution factors
        self.line_status = np.ones(self._calculate_num_lines())  # Line on/off (1/0)
        self.voltage_levels = np.ones(self.num_nodes)  # Per-unit voltage
        
        self.fitness = 0.0
        self.is_feasible = False
        
    def _calculate_num_lines(self) -> int:
        """Calculate maximum number of transmission lines in the grid."""
        rows, cols = self.grid_size
        # Horizontal lines + vertical lines
        horizontal_lines = rows * (cols - 1)
        vertical_lines = (rows - 1) * cols
        return horizontal_lines + vertical_lines
    
    def randomize(self, generator_limits: List[Tuple[float, float]], 
                  load_limits: List[Tuple[float, float]],
                  voltage_limits: Tuple[float, float] = (0.95, 1.05)):
        """
        Randomly initialize the chromosome within feasible bounds.
        
        Args:
            generator_limits: List of (min_mw, max_mw) for each generator
            load_limits: List of (min_load, max_load) for each load
            voltage_limits: (min_voltage, max_voltage) in per-unit
        """
        # Randomize generator dispatch within limits
        for i, (min_mw, max_mw) in enumerate(generator_limits):
            self.generator_dispatch[i] = random.uniform(min_mw, max_mw)
        
        # Randomize load allocation (sum should equal 1.0 for load distribution)
        random_values = np.random.random(self.num_loads)
        self.load_allocation = random_values / np.sum(random_values)
        
        # Randomize line status (80% probability of being on)
        self.line_status = np.random.choice([0, 1], size=len(self.line_status), p=[0.2, 0.8])
        
        # Randomize voltage levels within limits
        min_v, max_v = voltage_limits
        self.voltage_levels = np.random.uniform(min_v, max_v, self.num_nodes)
    
    def copy(self) -> 'GridChromosome':
        """Create a deep copy of the chromosome."""
        new_chromosome = GridChromosome(self.grid_size, self.num_generators, self.num_loads)
        new_chromosome.generator_dispatch = self.generator_dispatch.copy()
        new_chromosome.load_allocation = self.load_allocation.copy()
        new_chromosome.line_status = self.line_status.copy()
        new_chromosome.voltage_levels = self.voltage_levels.copy()
        new_chromosome.fitness = self.fitness
        new_chromosome.is_feasible = self.is_feasible
        return new_chromosome
    
    def get_total_generation(self) -> float:
        """Get total power generation in MW."""
        return np.sum(self.generator_dispatch)
    
    def get_active_lines(self) -> List[int]:
        """Get indices of active transmission lines."""
        return [i for i, status in enumerate(self.line_status) if status == 1]
    
    def get_network_connectivity(self) -> float:
        """Calculate network connectivity ratio (active lines / total possible lines)."""
        return np.sum(self.line_status) / len(self.line_status)
    
    def validate_power_balance(self, total_load: float, tolerance: float = 1e-6) -> bool:
        """
        Validate power balance constraint: generation = load + losses.
        
        Args:
            total_load: Total system load in MW
            tolerance: Acceptable tolerance for power balance
            
        Returns:
            True if power balance is satisfied
        """
        total_gen = self.get_total_generation()
        # Simplified loss calculation (2% of generation)
        estimated_losses = 0.02 * total_gen
        power_mismatch = abs(total_gen - total_load - estimated_losses)
        return power_mismatch <= tolerance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chromosome to dictionary representation."""
        return {
            'generator_dispatch': self.generator_dispatch.tolist(),
            'load_allocation': self.load_allocation.tolist(),
            'line_status': self.line_status.tolist(),
            'voltage_levels': self.voltage_levels.tolist(),
            'fitness': self.fitness,
            'is_feasible': self.is_feasible,
            'total_generation': self.get_total_generation(),
            'network_connectivity': self.get_network_connectivity()
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load chromosome from dictionary representation."""
        self.generator_dispatch = np.array(data['generator_dispatch'])
        self.load_allocation = np.array(data['load_allocation'])
        self.line_status = np.array(data['line_status'])
        self.voltage_levels = np.array(data['voltage_levels'])
        self.fitness = data.get('fitness', 0.0)
        self.is_feasible = data.get('is_feasible', False)
    
    def __str__(self) -> str:
        """String representation of the chromosome."""
        return (f"GridChromosome(fitness={self.fitness:.4f}, "
                f"total_gen={self.get_total_generation():.2f}MW, "
                f"connectivity={self.get_network_connectivity():.2f}, "
                f"feasible={self.is_feasible})")
    
    def __lt__(self, other: 'GridChromosome') -> bool:
        """Less than comparison based on fitness (for sorting)."""
        return self.fitness < other.fitness