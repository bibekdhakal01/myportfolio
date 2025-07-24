"""
Crossover operations for the genetic algorithm.

This module implements various crossover strategies for smart grid chromosomes,
including uniform crossover, arithmetic crossover, and problem-specific operators.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
from ..models.chromosome import GridChromosome
from ..models.grid_model import SmartGrid


class CrossoverStrategy(ABC):
    """Abstract base class for crossover strategies."""
    
    @abstractmethod
    def crossover(self, parent1: GridChromosome, parent2: GridChromosome) -> Tuple[GridChromosome, GridChromosome]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two offspring chromosomes
        """
        pass


class UniformCrossover(CrossoverStrategy):
    """
    Uniform crossover strategy.
    
    Each gene is inherited from either parent with equal probability.
    Good for maintaining diversity and exploring the search space.
    """
    
    def __init__(self, crossover_rate: float = 0.5):
        """
        Initialize uniform crossover.
        
        Args:
            crossover_rate: Probability of inheriting gene from parent1 vs parent2
        """
        self.crossover_rate = crossover_rate
    
    def crossover(self, parent1: GridChromosome, parent2: GridChromosome) -> Tuple[GridChromosome, GridChromosome]:
        """Perform uniform crossover."""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Crossover generator dispatch
        for i in range(len(offspring1.generator_dispatch)):
            if random.random() < self.crossover_rate:
                offspring1.generator_dispatch[i] = parent2.generator_dispatch[i]
                offspring2.generator_dispatch[i] = parent1.generator_dispatch[i]
        
        # Crossover load allocation
        for i in range(len(offspring1.load_allocation)):
            if random.random() < self.crossover_rate:
                offspring1.load_allocation[i] = parent2.load_allocation[i]
                offspring2.load_allocation[i] = parent1.load_allocation[i]
        
        # Crossover line status
        for i in range(len(offspring1.line_status)):
            if random.random() < self.crossover_rate:
                offspring1.line_status[i] = parent2.line_status[i]
                offspring2.line_status[i] = parent1.line_status[i]
        
        # Crossover voltage levels
        for i in range(len(offspring1.voltage_levels)):
            if random.random() < self.crossover_rate:
                offspring1.voltage_levels[i] = parent2.voltage_levels[i]
                offspring2.voltage_levels[i] = parent1.voltage_levels[i]
        
        # Normalize load allocation
        offspring1.load_allocation = offspring1.load_allocation / np.sum(offspring1.load_allocation)
        offspring2.load_allocation = offspring2.load_allocation / np.sum(offspring2.load_allocation)
        
        # Reset fitness (will be recalculated)
        offspring1.fitness = 0.0
        offspring2.fitness = 0.0
        
        return offspring1, offspring2


class ArithmeticCrossover(CrossoverStrategy):
    """
    Arithmetic crossover strategy.
    
    Creates offspring as weighted combinations of parents.
    Good for continuous optimization problems.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize arithmetic crossover.
        
        Args:
            alpha: Mixing parameter (0.0 to 1.0)
        """
        self.alpha = alpha
    
    def crossover(self, parent1: GridChromosome, parent2: GridChromosome) -> Tuple[GridChromosome, GridChromosome]:
        """Perform arithmetic crossover."""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Arithmetic crossover for continuous variables
        # Offspring1 = alpha * parent1 + (1-alpha) * parent2
        # Offspring2 = alpha * parent2 + (1-alpha) * parent1
        
        # Generator dispatch
        offspring1.generator_dispatch = (self.alpha * parent1.generator_dispatch + 
                                       (1 - self.alpha) * parent2.generator_dispatch)
        offspring2.generator_dispatch = (self.alpha * parent2.generator_dispatch + 
                                       (1 - self.alpha) * parent1.generator_dispatch)
        
        # Load allocation
        temp1 = self.alpha * parent1.load_allocation + (1 - self.alpha) * parent2.load_allocation
        temp2 = self.alpha * parent2.load_allocation + (1 - self.alpha) * parent1.load_allocation
        
        offspring1.load_allocation = temp1 / np.sum(temp1)  # Normalize
        offspring2.load_allocation = temp2 / np.sum(temp2)  # Normalize
        
        # Voltage levels
        offspring1.voltage_levels = (self.alpha * parent1.voltage_levels + 
                                   (1 - self.alpha) * parent2.voltage_levels)
        offspring2.voltage_levels = (self.alpha * parent2.voltage_levels + 
                                   (1 - self.alpha) * parent1.voltage_levels)
        
        # Binary line status - use uniform crossover
        for i in range(len(offspring1.line_status)):
            if random.random() < 0.5:
                offspring1.line_status[i] = parent2.line_status[i]
                offspring2.line_status[i] = parent1.line_status[i]
        
        # Reset fitness
        offspring1.fitness = 0.0
        offspring2.fitness = 0.0
        
        return offspring1, offspring2


class SinglePointCrossover(CrossoverStrategy):
    """
    Single-point crossover strategy.
    
    Swaps genes after a randomly chosen crossover point.
    Simple and effective for many problems.
    """
    
    def __init__(self):
        """Initialize single-point crossover."""
        pass
    
    def crossover(self, parent1: GridChromosome, parent2: GridChromosome) -> Tuple[GridChromosome, GridChromosome]:
        """Perform single-point crossover."""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Crossover generator dispatch
        if len(parent1.generator_dispatch) > 1:
            crossover_point = random.randint(1, len(parent1.generator_dispatch) - 1)
            offspring1.generator_dispatch[crossover_point:] = parent2.generator_dispatch[crossover_point:]
            offspring2.generator_dispatch[crossover_point:] = parent1.generator_dispatch[crossover_point:]
        
        # Crossover load allocation
        if len(parent1.load_allocation) > 1:
            crossover_point = random.randint(1, len(parent1.load_allocation) - 1)
            offspring1.load_allocation[crossover_point:] = parent2.load_allocation[crossover_point:]
            offspring2.load_allocation[crossover_point:] = parent1.load_allocation[crossover_point:]
        
        # Crossover line status
        if len(parent1.line_status) > 1:
            crossover_point = random.randint(1, len(parent1.line_status) - 1)
            offspring1.line_status[crossover_point:] = parent2.line_status[crossover_point:]
            offspring2.line_status[crossover_point:] = parent1.line_status[crossover_point:]
        
        # Crossover voltage levels
        if len(parent1.voltage_levels) > 1:
            crossover_point = random.randint(1, len(parent1.voltage_levels) - 1)
            offspring1.voltage_levels[crossover_point:] = parent2.voltage_levels[crossover_point:]
            offspring2.voltage_levels[crossover_point:] = parent1.voltage_levels[crossover_point:]
        
        # Normalize load allocation
        offspring1.load_allocation = offspring1.load_allocation / np.sum(offspring1.load_allocation)
        offspring2.load_allocation = offspring2.load_allocation / np.sum(offspring2.load_allocation)
        
        # Reset fitness
        offspring1.fitness = 0.0
        offspring2.fitness = 0.0
        
        return offspring1, offspring2


class TwoPointCrossover(CrossoverStrategy):
    """
    Two-point crossover strategy.
    
    Swaps genes between two randomly chosen crossover points.
    """
    
    def __init__(self):
        """Initialize two-point crossover."""
        pass
    
    def crossover(self, parent1: GridChromosome, parent2: GridChromosome) -> Tuple[GridChromosome, GridChromosome]:
        """Perform two-point crossover."""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Crossover generator dispatch
        if len(parent1.generator_dispatch) > 2:
            point1, point2 = sorted(random.sample(range(1, len(parent1.generator_dispatch)), 2))
            offspring1.generator_dispatch[point1:point2] = parent2.generator_dispatch[point1:point2]
            offspring2.generator_dispatch[point1:point2] = parent1.generator_dispatch[point1:point2]
        
        # Crossover load allocation
        if len(parent1.load_allocation) > 2:
            point1, point2 = sorted(random.sample(range(1, len(parent1.load_allocation)), 2))
            offspring1.load_allocation[point1:point2] = parent2.load_allocation[point1:point2]
            offspring2.load_allocation[point1:point2] = parent1.load_allocation[point1:point2]
        
        # Crossover line status
        if len(parent1.line_status) > 2:
            point1, point2 = sorted(random.sample(range(1, len(parent1.line_status)), 2))
            offspring1.line_status[point1:point2] = parent2.line_status[point1:point2]
            offspring2.line_status[point1:point2] = parent1.line_status[point1:point2]
        
        # Crossover voltage levels
        if len(parent1.voltage_levels) > 2:
            point1, point2 = sorted(random.sample(range(1, len(parent1.voltage_levels)), 2))
            offspring1.voltage_levels[point1:point2] = parent2.voltage_levels[point1:point2]
            offspring2.voltage_levels[point1:point2] = parent1.voltage_levels[point1:point2]
        
        # Normalize load allocation
        offspring1.load_allocation = offspring1.load_allocation / np.sum(offspring1.load_allocation)
        offspring2.load_allocation = offspring2.load_allocation / np.sum(offspring2.load_allocation)
        
        # Reset fitness
        offspring1.fitness = 0.0
        offspring2.fitness = 0.0
        
        return offspring1, offspring2


class SmartGridCrossover(CrossoverStrategy):
    """
    Smart grid specific crossover strategy.
    
    Incorporates domain knowledge about power systems to create
    meaningful offspring while maintaining system constraints.
    """
    
    def __init__(self, grid: SmartGrid, preserve_connectivity: bool = True):
        """
        Initialize smart grid crossover.
        
        Args:
            grid: Smart grid model for constraint checking
            preserve_connectivity: Whether to ensure network connectivity
        """
        self.grid = grid
        self.preserve_connectivity = preserve_connectivity
    
    def crossover(self, parent1: GridChromosome, parent2: GridChromosome) -> Tuple[GridChromosome, GridChromosome]:
        """Perform smart grid aware crossover."""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # 1. Economic dispatch crossover - preserve generation-load balance
        self._economic_dispatch_crossover(parent1, parent2, offspring1, offspring2)
        
        # 2. Network topology crossover - maintain connectivity
        if self.preserve_connectivity:
            self._connectivity_preserving_crossover(parent1, parent2, offspring1, offspring2)
        else:
            self._uniform_topology_crossover(parent1, parent2, offspring1, offspring2)
        
        # 3. Voltage profile crossover - maintain voltage quality
        self._voltage_profile_crossover(parent1, parent2, offspring1, offspring2)
        
        # 4. Load allocation crossover
        self._load_allocation_crossover(parent1, parent2, offspring1, offspring2)
        
        # Reset fitness
        offspring1.fitness = 0.0
        offspring2.fitness = 0.0
        
        return offspring1, offspring2
    
    def _economic_dispatch_crossover(self, parent1: GridChromosome, parent2: GridChromosome,
                                   offspring1: GridChromosome, offspring2: GridChromosome) -> None:
        """Crossover generator dispatch while maintaining power balance."""
        total_load = self.grid.get_total_load()
        
        # Choose crossover method based on parents' fitness
        if parent1.fitness > parent2.fitness:
            # Better parent contributes more to base dispatch
            alpha = 0.7
        else:
            alpha = 0.3
        
        # Arithmetic crossover for generator dispatch
        offspring1.generator_dispatch = (alpha * parent1.generator_dispatch + 
                                       (1 - alpha) * parent2.generator_dispatch)
        offspring2.generator_dispatch = (alpha * parent2.generator_dispatch + 
                                       (1 - alpha) * parent1.generator_dispatch)
        
        # Adjust dispatch to maintain power balance
        self._adjust_generation_balance(offspring1, total_load)
        self._adjust_generation_balance(offspring2, total_load)
    
    def _adjust_generation_balance(self, chromosome: GridChromosome, target_load: float) -> None:
        """Adjust generator dispatch to meet power balance."""
        current_generation = np.sum(chromosome.generator_dispatch)
        required_adjustment = target_load - current_generation
        
        if abs(required_adjustment) < 1e-6:
            return  # Already balanced
        
        # Adjust generators proportionally within their limits
        adjustable_capacity = 0.0
        for i, gen in enumerate(self.grid.generators):
            if required_adjustment > 0:  # Need more generation
                adjustable_capacity += max(0, gen.max_power - chromosome.generator_dispatch[i])
            else:  # Need less generation
                adjustable_capacity += max(0, chromosome.generator_dispatch[i] - gen.min_power)
        
        if adjustable_capacity > 0:
            adjustment_factor = required_adjustment / adjustable_capacity
            
            for i, gen in enumerate(self.grid.generators):
                if required_adjustment > 0:  # Increase generation
                    available_capacity = gen.max_power - chromosome.generator_dispatch[i]
                    adjustment = available_capacity * adjustment_factor
                else:  # Decrease generation
                    available_capacity = chromosome.generator_dispatch[i] - gen.min_power
                    adjustment = -available_capacity * adjustment_factor
                
                chromosome.generator_dispatch[i] += adjustment
                chromosome.generator_dispatch[i] = np.clip(chromosome.generator_dispatch[i],
                                                         gen.min_power, gen.max_power)
    
    def _connectivity_preserving_crossover(self, parent1: GridChromosome, parent2: GridChromosome,
                                         offspring1: GridChromosome, offspring2: GridChromosome) -> None:
        """Crossover line status while preserving network connectivity."""
        # Start with parents' line status
        offspring1.line_status = parent1.line_status.copy()
        offspring2.line_status = parent2.line_status.copy()
        
        # Randomly swap some lines between parents
        num_swaps = random.randint(1, len(parent1.line_status) // 4)
        swap_indices = random.sample(range(len(parent1.line_status)), num_swaps)
        
        for idx in swap_indices:
            # Temporarily swap lines
            temp1 = offspring1.line_status[idx]
            temp2 = offspring2.line_status[idx]
            
            offspring1.line_status[idx] = temp2
            offspring2.line_status[idx] = temp1
            
            # Check connectivity after swap
            stats1 = self.grid.get_network_statistics(offspring1.line_status)
            stats2 = self.grid.get_network_statistics(offspring2.line_status)
            
            # Revert swap if connectivity is lost
            if not stats1['is_connected']:
                offspring1.line_status[idx] = temp1
            if not stats2['is_connected']:
                offspring2.line_status[idx] = temp2
    
    def _uniform_topology_crossover(self, parent1: GridChromosome, parent2: GridChromosome,
                                  offspring1: GridChromosome, offspring2: GridChromosome) -> None:
        """Simple uniform crossover for line status."""
        for i in range(len(offspring1.line_status)):
            if random.random() < 0.5:
                offspring1.line_status[i] = parent2.line_status[i]
                offspring2.line_status[i] = parent1.line_status[i]
    
    def _voltage_profile_crossover(self, parent1: GridChromosome, parent2: GridChromosome,
                                 offspring1: GridChromosome, offspring2: GridChromosome) -> None:
        """Crossover voltage levels maintaining voltage quality."""
        # Use arithmetic crossover with random alpha for each node
        for i in range(len(offspring1.voltage_levels)):
            alpha = random.random()
            
            v1 = alpha * parent1.voltage_levels[i] + (1 - alpha) * parent2.voltage_levels[i]
            v2 = alpha * parent2.voltage_levels[i] + (1 - alpha) * parent1.voltage_levels[i]
            
            # Ensure voltage limits
            min_v, max_v = self.grid.voltage_limits
            offspring1.voltage_levels[i] = np.clip(v1, min_v, max_v)
            offspring2.voltage_levels[i] = np.clip(v2, min_v, max_v)
    
    def _load_allocation_crossover(self, parent1: GridChromosome, parent2: GridChromosome,
                                 offspring1: GridChromosome, offspring2: GridChromosome) -> None:
        """Crossover load allocation while maintaining normalization."""
        # Arithmetic crossover
        alpha = random.random()
        
        temp1 = alpha * parent1.load_allocation + (1 - alpha) * parent2.load_allocation
        temp2 = alpha * parent2.load_allocation + (1 - alpha) * parent1.load_allocation
        
        # Normalize to sum to 1.0
        offspring1.load_allocation = temp1 / np.sum(temp1)
        offspring2.load_allocation = temp2 / np.sum(temp2)


class AdaptiveCrossover(CrossoverStrategy):
    """
    Adaptive crossover strategy that selects crossover method based on
    population diversity and parent characteristics.
    """
    
    def __init__(self, grid: SmartGrid, strategies: Dict[str, CrossoverStrategy] = None):
        """
        Initialize adaptive crossover.
        
        Args:
            grid: Smart grid model
            strategies: Dictionary of crossover strategies to choose from
        """
        self.grid = grid
        self.strategies = strategies or {
            'uniform': UniformCrossover(),
            'arithmetic': ArithmeticCrossover(),
            'single_point': SinglePointCrossover(),
            'smart_grid': SmartGridCrossover(grid)
        }
        
        self.strategy_usage = {name: 0 for name in self.strategies.keys()}
        self.strategy_success = {name: 0 for name in self.strategies.keys()}
    
    def crossover(self, parent1: GridChromosome, parent2: GridChromosome) -> Tuple[GridChromosome, GridChromosome]:
        """Perform adaptive crossover."""
        # Select strategy based on parent characteristics
        strategy_name = self._select_strategy(parent1, parent2)
        
        # Perform crossover
        offspring1, offspring2 = self.strategies[strategy_name].crossover(parent1, parent2)
        
        # Track usage
        self.strategy_usage[strategy_name] += 1
        
        return offspring1, offspring2
    
    def _select_strategy(self, parent1: GridChromosome, parent2: GridChromosome) -> str:
        """Select appropriate crossover strategy based on parent characteristics."""
        # Calculate similarity between parents
        similarity = self._calculate_similarity(parent1, parent2)
        
        if similarity > 0.8:
            # High similarity - use explorative crossover
            return 'uniform'
        elif similarity < 0.3:
            # Low similarity - use conservative crossover
            return 'arithmetic'
        elif parent1.is_feasible and parent2.is_feasible:
            # Both feasible - use domain-specific crossover
            return 'smart_grid'
        else:
            # Default case
            return 'single_point'
    
    def _calculate_similarity(self, parent1: GridChromosome, parent2: GridChromosome) -> float:
        """Calculate similarity between two chromosomes."""
        similarities = []
        
        # Generator dispatch similarity
        if len(parent1.generator_dispatch) > 0:
            max_dispatch = max(np.max(parent1.generator_dispatch), np.max(parent2.generator_dispatch))
            if max_dispatch > 0:
                gen_diff = np.mean(np.abs(parent1.generator_dispatch - parent2.generator_dispatch))
                similarities.append(1.0 - gen_diff / max_dispatch)
        
        # Line status similarity (Hamming distance)
        if len(parent1.line_status) > 0:
            line_similarity = 1.0 - np.sum(parent1.line_status != parent2.line_status) / len(parent1.line_status)
            similarities.append(line_similarity)
        
        # Voltage similarity
        if len(parent1.voltage_levels) > 0:
            volt_diff = np.mean(np.abs(parent1.voltage_levels - parent2.voltage_levels))
            similarities.append(1.0 - volt_diff / 0.1)  # Normalize by typical voltage range
        
        return np.mean(similarities) if similarities else 0.0
    
    def update_strategy_success(self, strategy_name: str, success: bool) -> None:
        """Update success rate for a strategy."""
        if success:
            self.strategy_success[strategy_name] += 1
    
    def get_strategy_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about strategy usage and success."""
        stats = {}
        for name in self.strategies.keys():
            usage = self.strategy_usage[name]
            success = self.strategy_success[name]
            success_rate = success / usage if usage > 0 else 0.0
            
            stats[name] = {
                'usage_count': usage,
                'success_count': success,
                'success_rate': success_rate
            }
        
        return stats


def create_crossover_operator(strategy_name: str, grid: SmartGrid = None, **kwargs) -> CrossoverStrategy:
    """
    Factory function to create crossover operators.
    
    Args:
        strategy_name: Name of the crossover strategy
        grid: Smart grid model (required for some strategies)
        **kwargs: Strategy-specific parameters
        
    Returns:
        CrossoverStrategy instance
    """
    strategy_map = {
        'uniform': UniformCrossover,
        'arithmetic': ArithmeticCrossover,
        'single_point': SinglePointCrossover,
        'two_point': TwoPointCrossover,
        'smart_grid': lambda **kw: SmartGridCrossover(grid, **kw),
        'adaptive': lambda **kw: AdaptiveCrossover(grid, **kw)
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown crossover strategy: {strategy_name}")
    
    if strategy_name in ['smart_grid', 'adaptive'] and grid is None:
        raise ValueError(f"Grid model required for {strategy_name} crossover")
    
    return strategy_map[strategy_name](**kwargs)