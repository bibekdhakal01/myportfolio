"""
Mutation operations for the genetic algorithm.

This module implements various mutation strategies for smart grid chromosomes,
including Gaussian mutation, uniform mutation, and problem-specific operators.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
from ..models.chromosome import GridChromosome
from ..models.grid_model import SmartGrid


class MutationStrategy(ABC):
    """Abstract base class for mutation strategies."""
    
    @abstractmethod
    def mutate(self, chromosome: GridChromosome) -> GridChromosome:
        """
        Perform mutation on a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        pass


class GaussianMutation(MutationStrategy):
    """
    Gaussian mutation strategy.
    
    Adds Gaussian noise to continuous variables and flips binary variables
    with specified probabilities.
    """
    
    def __init__(self, mutation_rate: float = 0.1, 
                 gaussian_std: float = 0.1,
                 binary_flip_rate: float = 0.05):
        """
        Initialize Gaussian mutation.
        
        Args:
            mutation_rate: Probability of mutating each gene
            gaussian_std: Standard deviation for Gaussian noise
            binary_flip_rate: Probability of flipping binary variables
        """
        self.mutation_rate = mutation_rate
        self.gaussian_std = gaussian_std
        self.binary_flip_rate = binary_flip_rate
    
    def mutate(self, chromosome: GridChromosome) -> GridChromosome:
        """Perform Gaussian mutation."""
        mutated = chromosome.copy()
        
        # Mutate generator dispatch
        for i in range(len(mutated.generator_dispatch)):
            if random.random() < self.mutation_rate:
                noise = np.random.normal(0, self.gaussian_std)
                mutated.generator_dispatch[i] += noise * mutated.generator_dispatch[i]
        
        # Mutate load allocation
        for i in range(len(mutated.load_allocation)):
            if random.random() < self.mutation_rate:
                noise = np.random.normal(0, self.gaussian_std)
                mutated.load_allocation[i] += noise * mutated.load_allocation[i]
                mutated.load_allocation[i] = max(0, mutated.load_allocation[i])  # Keep positive
        
        # Normalize load allocation
        if np.sum(mutated.load_allocation) > 0:
            mutated.load_allocation = mutated.load_allocation / np.sum(mutated.load_allocation)
        
        # Mutate voltage levels
        for i in range(len(mutated.voltage_levels)):
            if random.random() < self.mutation_rate:
                noise = np.random.normal(0, self.gaussian_std)
                mutated.voltage_levels[i] += noise
        
        # Mutate line status (binary flip)
        for i in range(len(mutated.line_status)):
            if random.random() < self.binary_flip_rate:
                mutated.line_status[i] = 1 - mutated.line_status[i]
        
        # Reset fitness
        mutated.fitness = 0.0
        
        return mutated


class UniformMutation(MutationStrategy):
    """
    Uniform mutation strategy.
    
    Replaces genes with random values within valid ranges.
    """
    
    def __init__(self, mutation_rate: float = 0.1, 
                 intensity: float = 0.1):
        """
        Initialize uniform mutation.
        
        Args:
            mutation_rate: Probability of mutating each gene
            intensity: Intensity of mutation (fraction of valid range)
        """
        self.mutation_rate = mutation_rate
        self.intensity = intensity
    
    def mutate(self, chromosome: GridChromosome) -> GridChromosome:
        """Perform uniform mutation."""
        mutated = chromosome.copy()
        
        # Mutate generator dispatch
        for i in range(len(mutated.generator_dispatch)):
            if random.random() < self.mutation_rate:
                current_value = mutated.generator_dispatch[i]
                mutation_range = current_value * self.intensity
                mutated.generator_dispatch[i] += random.uniform(-mutation_range, mutation_range)
        
        # Mutate load allocation
        for i in range(len(mutated.load_allocation)):
            if random.random() < self.mutation_rate:
                current_value = mutated.load_allocation[i]
                mutation_range = current_value * self.intensity
                mutated.load_allocation[i] += random.uniform(-mutation_range, mutation_range)
                mutated.load_allocation[i] = max(0, mutated.load_allocation[i])
        
        # Normalize load allocation
        if np.sum(mutated.load_allocation) > 0:
            mutated.load_allocation = mutated.load_allocation / np.sum(mutated.load_allocation)
        
        # Mutate voltage levels
        for i in range(len(mutated.voltage_levels)):
            if random.random() < self.mutation_rate:
                mutation_range = 0.1 * self.intensity  # 10% of typical voltage range
                mutated.voltage_levels[i] += random.uniform(-mutation_range, mutation_range)
        
        # Mutate line status
        for i in range(len(mutated.line_status)):
            if random.random() < self.mutation_rate:
                mutated.line_status[i] = random.choice([0, 1])
        
        # Reset fitness
        mutated.fitness = 0.0
        
        return mutated


class PolynomialMutation(MutationStrategy):
    """
    Polynomial mutation strategy.
    
    Uses polynomial distribution for mutation, providing better control
    over the mutation distribution.
    """
    
    def __init__(self, mutation_rate: float = 0.1,
                 eta: float = 20.0):
        """
        Initialize polynomial mutation.
        
        Args:
            mutation_rate: Probability of mutating each gene
            eta: Distribution index (higher values = more concentrated around original)
        """
        self.mutation_rate = mutation_rate
        self.eta = eta
    
    def mutate(self, chromosome: GridChromosome) -> GridChromosome:
        """Perform polynomial mutation."""
        mutated = chromosome.copy()
        
        # Mutate generator dispatch
        for i in range(len(mutated.generator_dispatch)):
            if random.random() < self.mutation_rate:
                delta = self._polynomial_delta()
                mutated.generator_dispatch[i] *= (1 + delta)
        
        # Mutate load allocation
        for i in range(len(mutated.load_allocation)):
            if random.random() < self.mutation_rate:
                delta = self._polynomial_delta()
                mutated.load_allocation[i] *= (1 + delta)
                mutated.load_allocation[i] = max(0, mutated.load_allocation[i])
        
        # Normalize load allocation
        if np.sum(mutated.load_allocation) > 0:
            mutated.load_allocation = mutated.load_allocation / np.sum(mutated.load_allocation)
        
        # Mutate voltage levels
        for i in range(len(mutated.voltage_levels)):
            if random.random() < self.mutation_rate:
                delta = self._polynomial_delta() * 0.1  # Scale for voltage range
                mutated.voltage_levels[i] += delta
        
        # Binary mutation for line status
        for i in range(len(mutated.line_status)):
            if random.random() < self.mutation_rate:
                mutated.line_status[i] = 1 - mutated.line_status[i]
        
        # Reset fitness
        mutated.fitness = 0.0
        
        return mutated
    
    def _polynomial_delta(self) -> float:
        """Calculate polynomial mutation delta."""
        r = random.random()
        
        if r < 0.5:
            delta = (2 * r) ** (1.0 / (self.eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - r)) ** (1.0 / (self.eta + 1))
        
        return delta


class SmartGridMutation(MutationStrategy):
    """
    Smart grid specific mutation strategy.
    
    Incorporates domain knowledge to perform meaningful mutations
    while maintaining system constraints and feasibility.
    """
    
    def __init__(self, grid: SmartGrid, 
                 mutation_rate: float = 0.1,
                 preserve_constraints: bool = True):
        """
        Initialize smart grid mutation.
        
        Args:
            grid: Smart grid model
            mutation_rate: Base mutation rate
            preserve_constraints: Whether to maintain system constraints
        """
        self.grid = grid
        self.mutation_rate = mutation_rate
        self.preserve_constraints = preserve_constraints
    
    def mutate(self, chromosome: GridChromosome) -> GridChromosome:
        """Perform smart grid aware mutation."""
        mutated = chromosome.copy()
        
        # 1. Economic dispatch mutation
        self._economic_dispatch_mutation(mutated)
        
        # 2. Network topology mutation
        self._network_topology_mutation(mutated)
        
        # 3. Voltage control mutation
        self._voltage_control_mutation(mutated)
        
        # 4. Load management mutation
        self._load_management_mutation(mutated)
        
        # 5. Constraint repair if needed
        if self.preserve_constraints:
            self._repair_constraints(mutated)
        
        # Reset fitness
        mutated.fitness = 0.0
        
        return mutated
    
    def _economic_dispatch_mutation(self, chromosome: GridChromosome) -> None:
        """Mutate generator dispatch considering economic factors."""
        if random.random() > self.mutation_rate:
            return
        
        # Sort generators by cost efficiency
        generator_costs = []
        for i, gen in enumerate(self.grid.generators):
            current_power = chromosome.generator_dispatch[i]
            marginal_cost = 2 * gen.cost_coeff[0] * current_power + gen.cost_coeff[1]
            generator_costs.append((i, marginal_cost))
        
        generator_costs.sort(key=lambda x: x[1])  # Sort by marginal cost
        
        # Redistribute power from expensive to cheap generators
        expensive_gen_idx = generator_costs[-1][0]  # Most expensive
        cheap_gen_idx = generator_costs[0][0]      # Cheapest
        
        expensive_gen = self.grid.generators[expensive_gen_idx]
        cheap_gen = self.grid.generators[cheap_gen_idx]
        
        # Transfer power if possible
        power_transfer = min(
            chromosome.generator_dispatch[expensive_gen_idx] - expensive_gen.min_power,
            cheap_gen.max_power - chromosome.generator_dispatch[cheap_gen_idx],
            expensive_gen.max_power * 0.1  # Limit transfer to 10% of capacity
        )
        
        if power_transfer > 0:
            chromosome.generator_dispatch[expensive_gen_idx] -= power_transfer
            chromosome.generator_dispatch[cheap_gen_idx] += power_transfer
    
    def _network_topology_mutation(self, chromosome: GridChromosome) -> None:
        """Mutate network topology maintaining connectivity."""
        if random.random() > self.mutation_rate:
            return
        
        # Choose mutation type
        mutation_type = random.choice(['add_line', 'remove_line', 'swap_lines'])
        
        if mutation_type == 'add_line':
            # Add a disconnected line
            inactive_lines = [i for i, status in enumerate(chromosome.line_status) if status == 0]
            if inactive_lines:
                line_to_activate = random.choice(inactive_lines)
                chromosome.line_status[line_to_activate] = 1
        
        elif mutation_type == 'remove_line':
            # Remove a line while maintaining connectivity
            active_lines = [i for i, status in enumerate(chromosome.line_status) if status == 1]
            if len(active_lines) > self.grid.num_nodes - 1:  # Keep minimum spanning tree
                line_to_remove = random.choice(active_lines)
                
                # Temporarily remove line and check connectivity
                chromosome.line_status[line_to_remove] = 0
                stats = self.grid.get_network_statistics(chromosome.line_status)
                
                if not stats['is_connected']:
                    # Restore line if connectivity is lost
                    chromosome.line_status[line_to_remove] = 1
        
        elif mutation_type == 'swap_lines':
            # Swap status of two random lines
            line1, line2 = random.sample(range(len(chromosome.line_status)), 2)
            
            # Try swap and check connectivity
            original_status = chromosome.line_status.copy()
            chromosome.line_status[line1], chromosome.line_status[line2] = \
                chromosome.line_status[line2], chromosome.line_status[line1]
            
            stats = self.grid.get_network_statistics(chromosome.line_status)
            if not stats['is_connected']:
                # Revert swap if connectivity is lost
                chromosome.line_status = original_status
    
    def _voltage_control_mutation(self, chromosome: GridChromosome) -> None:
        """Mutate voltage levels considering voltage stability."""
        if random.random() > self.mutation_rate:
            return
        
        # Select a voltage zone to mutate (group of nearby nodes)
        rows, cols = self.grid.grid_size
        zone_size = min(3, rows, cols)  # 3x3 zone
        
        start_row = random.randint(0, max(0, rows - zone_size))
        start_col = random.randint(0, max(0, cols - zone_size))
        
        # Apply coordinated voltage mutation to the zone
        voltage_adjustment = random.uniform(-0.02, 0.02)  # ±2% voltage change
        
        for r in range(start_row, min(start_row + zone_size, rows)):
            for c in range(start_col, min(start_col + zone_size, cols)):
                node_id = r * cols + c
                if node_id < len(chromosome.voltage_levels):
                    chromosome.voltage_levels[node_id] += voltage_adjustment
    
    def _load_management_mutation(self, chromosome: GridChromosome) -> None:
        """Mutate load allocation considering demand response."""
        if random.random() > self.mutation_rate:
            return
        
        # Implement demand response mutation
        high_priority_loads = []
        low_priority_loads = []
        
        for i, load in enumerate(self.grid.loads):
            if load.priority <= 2:  # High priority
                high_priority_loads.append(i)
            else:  # Low priority
                low_priority_loads.append(i)
        
        # Transfer load from low to high priority if possible
        if high_priority_loads and low_priority_loads:
            high_idx = random.choice(high_priority_loads)
            low_idx = random.choice(low_priority_loads)
            
            transfer_amount = chromosome.load_allocation[low_idx] * 0.1  # 10% transfer
            
            chromosome.load_allocation[low_idx] -= transfer_amount
            chromosome.load_allocation[high_idx] += transfer_amount
            
            # Ensure non-negative values
            chromosome.load_allocation = np.maximum(chromosome.load_allocation, 0)
            
            # Renormalize
            chromosome.load_allocation = chromosome.load_allocation / np.sum(chromosome.load_allocation)
    
    def _repair_constraints(self, chromosome: GridChromosome) -> None:
        """Repair constraint violations."""
        # 1. Repair generator limits
        for i, gen in enumerate(self.grid.generators):
            if i < len(chromosome.generator_dispatch):
                chromosome.generator_dispatch[i] = np.clip(
                    chromosome.generator_dispatch[i], 
                    gen.min_power, 
                    gen.max_power
                )
        
        # 2. Repair voltage limits
        min_v, max_v = self.grid.voltage_limits
        chromosome.voltage_levels = np.clip(chromosome.voltage_levels, min_v, max_v)
        
        # 3. Repair power balance
        total_load = self.grid.get_total_load()
        current_gen = np.sum(chromosome.generator_dispatch)
        
        if abs(current_gen - total_load) > total_load * 0.05:  # 5% tolerance
            # Scale generation to match load
            scale_factor = total_load / current_gen if current_gen > 0 else 1.0
            
            for i, gen in enumerate(self.grid.generators):
                scaled_power = chromosome.generator_dispatch[i] * scale_factor
                chromosome.generator_dispatch[i] = np.clip(scaled_power, gen.min_power, gen.max_power)
        
        # 4. Ensure load allocation is normalized
        if np.sum(chromosome.load_allocation) > 0:
            chromosome.load_allocation = chromosome.load_allocation / np.sum(chromosome.load_allocation)


class AdaptiveMutation(MutationStrategy):
    """
    Adaptive mutation strategy that adjusts mutation parameters
    based on population diversity and convergence state.
    """
    
    def __init__(self, grid: SmartGrid,
                 base_mutation_rate: float = 0.1,
                 strategies: Dict[str, MutationStrategy] = None):
        """
        Initialize adaptive mutation.
        
        Args:
            grid: Smart grid model
            base_mutation_rate: Base mutation rate
            strategies: Dictionary of mutation strategies
        """
        self.grid = grid
        self.base_mutation_rate = base_mutation_rate
        self.current_mutation_rate = base_mutation_rate
        
        self.strategies = strategies or {
            'gaussian': GaussianMutation(base_mutation_rate),
            'uniform': UniformMutation(base_mutation_rate),
            'polynomial': PolynomialMutation(base_mutation_rate),
            'smart_grid': SmartGridMutation(grid, base_mutation_rate)
        }
        
        self.strategy_performance = {name: [] for name in self.strategies.keys()}
        self.current_strategy = 'gaussian'
        self.adaptation_counter = 0
    
    def mutate(self, chromosome: GridChromosome) -> GridChromosome:
        """Perform adaptive mutation."""
        # Adapt strategy and parameters periodically
        self.adaptation_counter += 1
        if self.adaptation_counter % 50 == 0:  # Adapt every 50 mutations
            self._adapt_strategy(chromosome)
        
        # Update mutation rates in strategies
        self._update_strategy_parameters()
        
        # Use current strategy
        return self.strategies[self.current_strategy].mutate(chromosome)
    
    def _adapt_strategy(self, chromosome: GridChromosome) -> None:
        """Adapt mutation strategy based on current state."""
        # Simple adaptation based on feasibility
        if chromosome.is_feasible:
            # Use fine-tuning mutations for feasible solutions
            self.current_strategy = 'polynomial'
            self.current_mutation_rate = self.base_mutation_rate * 0.5
        else:
            # Use more aggressive mutations for infeasible solutions
            self.current_strategy = 'smart_grid'
            self.current_mutation_rate = self.base_mutation_rate * 1.5
    
    def _update_strategy_parameters(self) -> None:
        """Update mutation parameters in all strategies."""
        for strategy in self.strategies.values():
            if hasattr(strategy, 'mutation_rate'):
                strategy.mutation_rate = self.current_mutation_rate
    
    def update_diversity_feedback(self, population_diversity: float) -> None:
        """Update mutation based on population diversity feedback."""
        if population_diversity < 0.1:  # Low diversity
            self.current_mutation_rate = self.base_mutation_rate * 2.0
            self.current_strategy = 'uniform'  # More disruptive
        elif population_diversity > 0.5:  # High diversity
            self.current_mutation_rate = self.base_mutation_rate * 0.5
            self.current_strategy = 'polynomial'  # More conservative
        else:  # Medium diversity
            self.current_mutation_rate = self.base_mutation_rate
            self.current_strategy = 'gaussian'


class MultiOperatorMutation(MutationStrategy):
    """
    Multi-operator mutation that applies multiple mutation strategies
    to different parts of the chromosome.
    """
    
    def __init__(self, grid: SmartGrid,
                 operator_probabilities: Dict[str, float] = None):
        """
        Initialize multi-operator mutation.
        
        Args:
            grid: Smart grid model
            operator_probabilities: Probabilities for each operator
        """
        self.grid = grid
        
        self.operators = {
            'gaussian': GaussianMutation(0.1),
            'uniform': UniformMutation(0.05),
            'smart_grid': SmartGridMutation(grid, 0.08)
        }
        
        self.probabilities = operator_probabilities or {
            'gaussian': 0.4,
            'uniform': 0.3,
            'smart_grid': 0.3
        }
    
    def mutate(self, chromosome: GridChromosome) -> GridChromosome:
        """Apply multiple mutation operators."""
        mutated = chromosome.copy()
        
        # Apply each operator with its probability
        for operator_name, probability in self.probabilities.items():
            if random.random() < probability:
                mutated = self.operators[operator_name].mutate(mutated)
        
        return mutated


def create_mutation_operator(strategy_name: str, grid: SmartGrid = None, **kwargs) -> MutationStrategy:
    """
    Factory function to create mutation operators.
    
    Args:
        strategy_name: Name of the mutation strategy
        grid: Smart grid model (required for some strategies)
        **kwargs: Strategy-specific parameters
        
    Returns:
        MutationStrategy instance
    """
    if strategy_name == 'gaussian':
        return GaussianMutation(**kwargs)
    elif strategy_name == 'uniform':
        return UniformMutation(**kwargs)
    elif strategy_name == 'polynomial':
        return PolynomialMutation(**kwargs)
    elif strategy_name == 'smart_grid':
        if grid is None:
            raise ValueError(f"Grid model required for {strategy_name} mutation")
        # Filter kwargs for SmartGridMutation
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ['preserve_constraints']}
        if 'mutation_rate' in kwargs:
            valid_kwargs['mutation_rate'] = kwargs['mutation_rate']
        return SmartGridMutation(grid, **valid_kwargs)
    elif strategy_name == 'adaptive':
        if grid is None:
            raise ValueError(f"Grid model required for {strategy_name} mutation")
        # Filter kwargs for AdaptiveMutation
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ['base_mutation_rate', 'strategies']}
        if 'mutation_rate' in kwargs:
            valid_kwargs['base_mutation_rate'] = kwargs['mutation_rate']
        return AdaptiveMutation(grid, **valid_kwargs)
    elif strategy_name == 'multi_operator':
        if grid is None:
            raise ValueError(f"Grid model required for {strategy_name} mutation")
        # Filter kwargs for MultiOperatorMutation
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ['operator_probabilities']}
        return MultiOperatorMutation(grid, **valid_kwargs)
    else:
        raise ValueError(f"Unknown mutation strategy: {strategy_name}")