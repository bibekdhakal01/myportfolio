"""
Selection strategies for the genetic algorithm.

This module implements various selection mechanisms including tournament selection,
roulette wheel selection, rank-based selection, and multi-objective selection.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
from ..models.chromosome import GridChromosome


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    
    @abstractmethod
    def select(self, population: List[GridChromosome], num_parents: int) -> List[GridChromosome]:
        """
        Select parents from population.
        
        Args:
            population: List of chromosomes to select from
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent chromosomes
        """
        pass


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection strategy.
    
    Selects individuals by running tournaments between random subsets
    of the population and choosing the best individual from each tournament.
    """
    
    def __init__(self, tournament_size: int = 3, selection_pressure: float = 1.0):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament
            selection_pressure: Probability that best individual wins (0.5 to 1.0)
        """
        self.tournament_size = tournament_size
        self.selection_pressure = selection_pressure
    
    def select(self, population: List[GridChromosome], num_parents: int) -> List[GridChromosome]:
        """Select parents using tournament selection."""
        if not population:
            return []
        
        selected_parents = []
        
        for _ in range(num_parents):
            # Run tournament
            tournament_participants = random.sample(population, 
                                                   min(self.tournament_size, len(population)))
            
            # Sort participants by fitness (best first)
            tournament_participants.sort(key=lambda x: x.fitness, reverse=True)
            
            # Select winner based on selection pressure
            if random.random() < self.selection_pressure:
                winner = tournament_participants[0]  # Best individual
            else:
                # Select randomly from tournament with bias toward better individuals
                weights = [1.0 / (i + 1) for i in range(len(tournament_participants))]
                winner = random.choices(tournament_participants, weights=weights)[0]
            
            selected_parents.append(winner.copy())
        
        return selected_parents


class RouletteWheelSelection(SelectionStrategy):
    """
    Roulette wheel (fitness proportionate) selection strategy.
    
    Selects individuals with probability proportional to their fitness.
    Handles negative fitness values by shifting all values to positive range.
    """
    
    def __init__(self, scaling_factor: float = 1.0):
        """
        Initialize roulette wheel selection.
        
        Args:
            scaling_factor: Factor to scale selection pressure
        """
        self.scaling_factor = scaling_factor
    
    def select(self, population: List[GridChromosome], num_parents: int) -> List[GridChromosome]:
        """Select parents using roulette wheel selection."""
        if not population:
            return []
        
        # Get fitness values
        fitness_values = np.array([chr.fitness for chr in population])
        
        # Handle negative fitness values by shifting
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values - min_fitness + 1e-8
        
        # Apply scaling
        fitness_values = fitness_values ** self.scaling_factor
        
        # Calculate selection probabilities
        total_fitness = np.sum(fitness_values)
        if total_fitness == 0:
            # Uniform selection if all fitness values are zero
            probabilities = np.ones(len(population)) / len(population)
        else:
            probabilities = fitness_values / total_fitness
        
        # Select parents
        selected_indices = np.random.choice(len(population), size=num_parents, 
                                          p=probabilities, replace=True)
        
        selected_parents = [population[i].copy() for i in selected_indices]
        return selected_parents


class RankBasedSelection(SelectionStrategy):
    """
    Rank-based selection strategy.
    
    Selects individuals based on their rank in the population rather than
    raw fitness values. This provides more consistent selection pressure.
    """
    
    def __init__(self, selection_pressure: float = 2.0):
        """
        Initialize rank-based selection.
        
        Args:
            selection_pressure: Linear ranking parameter (1.0 to 2.0)
        """
        self.selection_pressure = max(1.0, min(2.0, selection_pressure))
    
    def select(self, population: List[GridChromosome], num_parents: int) -> List[GridChromosome]:
        """Select parents using rank-based selection."""
        if not population:
            return []
        
        # Sort population by fitness (worst to best for ranking)
        sorted_population = sorted(population, key=lambda x: x.fitness)
        n = len(sorted_population)
        
        # Calculate rank-based probabilities
        probabilities = []
        for rank in range(n):  # rank 0 = worst, rank n-1 = best
            prob = (2 - self.selection_pressure + 
                   2 * (self.selection_pressure - 1) * rank / (n - 1)) / n
            probabilities.append(prob)
        
        # Select parents
        selected_parents = []
        for _ in range(num_parents):
            selected_idx = np.random.choice(n, p=probabilities)
            selected_parents.append(sorted_population[selected_idx].copy())
        
        return selected_parents


class StochasticUniversalSampling(SelectionStrategy):
    """
    Stochastic Universal Sampling (SUS) selection strategy.
    
    Provides lower variance than roulette wheel selection by using
    evenly spaced selection points.
    """
    
    def __init__(self):
        """Initialize SUS selection."""
        pass
    
    def select(self, population: List[GridChromosome], num_parents: int) -> List[GridChromosome]:
        """Select parents using stochastic universal sampling."""
        if not population or num_parents == 0:
            return []
        
        # Get fitness values and handle negatives
        fitness_values = np.array([chr.fitness for chr in population])
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values - min_fitness + 1e-8
        
        total_fitness = np.sum(fitness_values)
        if total_fitness == 0:
            # Random selection if all fitness values are zero
            return random.choices(population, k=num_parents)
        
        # Calculate cumulative fitness
        cumulative_fitness = np.cumsum(fitness_values)
        
        # Generate evenly spaced selection points
        step_size = total_fitness / num_parents
        start_point = random.uniform(0, step_size)
        selection_points = [start_point + i * step_size for i in range(num_parents)]
        
        # Select individuals
        selected_parents = []
        for point in selection_points:
            # Find first individual with cumulative fitness >= point
            selected_idx = np.searchsorted(cumulative_fitness, point)
            selected_idx = min(selected_idx, len(population) - 1)
            selected_parents.append(population[selected_idx].copy())
        
        return selected_parents


class MultiObjectiveSelection(SelectionStrategy):
    """
    Multi-objective selection strategy using NSGA-II concepts.
    
    Selects individuals based on Pareto dominance and crowding distance
    for multi-objective optimization problems.
    """
    
    def __init__(self, objectives: List[str] = None):
        """
        Initialize multi-objective selection.
        
        Args:
            objectives: List of objective names to consider
        """
        self.objectives = objectives or ['generation_cost', 'power_losses', 
                                       'voltage_deviation', 'reliability', 'emissions']
    
    def select(self, population: List[GridChromosome], num_parents: int) -> List[GridChromosome]:
        """Select parents using multi-objective criteria."""
        if not population:
            return []
        
        # Perform non-dominated sorting
        fronts = self._non_dominated_sort(population)
        
        # Calculate crowding distance for each front
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # Select parents from fronts
        selected_parents = []
        front_idx = 0
        
        while len(selected_parents) < num_parents and front_idx < len(fronts):
            current_front = fronts[front_idx]
            
            if len(selected_parents) + len(current_front) <= num_parents:
                # Add entire front
                selected_parents.extend([chr.copy() for chr in current_front])
            else:
                # Partial front selection based on crowding distance
                remaining_slots = num_parents - len(selected_parents)
                # Sort by crowding distance (descending)
                current_front.sort(key=lambda x: getattr(x, 'crowding_distance', 0), reverse=True)
                selected_parents.extend([chr.copy() for chr in current_front[:remaining_slots]])
            
            front_idx += 1
        
        return selected_parents
    
    def _non_dominated_sort(self, population: List[GridChromosome]) -> List[List[GridChromosome]]:
        """Perform non-dominated sorting."""
        fronts = []
        dominated_solutions = {i: [] for i in range(len(population))}
        domination_count = [0] * len(population)
        
        # Find domination relationships
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    if self._dominates(population[i], population[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(population[j], population[i]):
                        domination_count[i] += 1
        
        # First front (non-dominated solutions)
        first_front = []
        for i in range(len(population)):
            if domination_count[i] == 0:
                first_front.append(population[i])
        
        fronts.append(first_front)
        
        # Build subsequent fronts
        current_front_idx = 0
        while current_front_idx < len(fronts) and fronts[current_front_idx]:
            next_front = []
            
            for solution in fronts[current_front_idx]:
                solution_idx = population.index(solution)
                for dominated_idx in dominated_solutions[solution_idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        next_front.append(population[dominated_idx])
            
            if next_front:
                fronts.append(next_front)
            current_front_idx += 1
        
        return fronts
    
    def _dominates(self, solution1: GridChromosome, solution2: GridChromosome) -> bool:
        """Check if solution1 dominates solution2."""
        if not (hasattr(solution1, 'objectives') and hasattr(solution2, 'objectives')):
            return solution1.fitness > solution2.fitness
        
        obj1 = solution1.objectives
        obj2 = solution2.objectives
        
        better_in_any = False
        
        for obj_name in self.objectives:
            if obj_name in obj1 and obj_name in obj2:
                if obj1[obj_name] > obj2[obj_name]:  # Worse in this objective (minimization)
                    return False
                elif obj1[obj_name] < obj2[obj_name]:  # Better in this objective
                    better_in_any = True
        
        return better_in_any
    
    def _calculate_crowding_distance(self, front: List[GridChromosome]) -> None:
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for solution in front:
            solution.crowding_distance = 0.0
        
        # Calculate distance for each objective
        for obj_name in self.objectives:
            # Get objective values
            obj_values = []
            for solution in front:
                if hasattr(solution, 'objectives') and obj_name in solution.objectives:
                    obj_values.append(solution.objectives[obj_name])
                else:
                    obj_values.append(0.0)
            
            if len(set(obj_values)) <= 1:  # All values are the same
                continue
            
            # Sort by objective value
            sorted_indices = sorted(range(len(front)), key=lambda i: obj_values[i])
            
            # Boundary solutions get infinite distance
            front[sorted_indices[0]].crowding_distance = float('inf')
            front[sorted_indices[-1]].crowding_distance = float('inf')
            
            # Calculate distance for intermediate solutions
            obj_range = max(obj_values) - min(obj_values)
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    curr_idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    
                    distance = (obj_values[next_idx] - obj_values[prev_idx]) / obj_range
                    front[curr_idx].crowding_distance += distance


class AdaptiveSelection(SelectionStrategy):
    """
    Adaptive selection strategy that switches between different selection methods
    based on population diversity and convergence state.
    """
    
    def __init__(self, strategies: Dict[str, SelectionStrategy] = None):
        """
        Initialize adaptive selection.
        
        Args:
            strategies: Dictionary of selection strategies to choose from
        """
        self.strategies = strategies or {
            'tournament': TournamentSelection(tournament_size=3),
            'roulette': RouletteWheelSelection(),
            'rank': RankBasedSelection(),
            'sus': StochasticUniversalSampling()
        }
        
        self.strategy_performance = {name: [] for name in self.strategies.keys()}
        self.current_strategy = 'tournament'
        self.adaptation_interval = 10  # Generations between strategy evaluation
        self.generation_count = 0
    
    def select(self, population: List[GridChromosome], num_parents: int) -> List[GridChromosome]:
        """Select parents using adaptive strategy selection."""
        # Adapt strategy periodically
        if self.generation_count % self.adaptation_interval == 0:
            self._adapt_strategy(population)
        
        self.generation_count += 1
        
        # Use current strategy
        return self.strategies[self.current_strategy].select(population, num_parents)
    
    def _adapt_strategy(self, population: List[GridChromosome]) -> None:
        """Adapt selection strategy based on population state."""
        if not population:
            return
        
        # Calculate population diversity
        fitness_values = [chr.fitness for chr in population]
        diversity = np.std(fitness_values) / (np.mean(fitness_values) + 1e-8)
        
        # Choose strategy based on diversity
        if diversity < 0.1:  # Low diversity - need exploration
            self.current_strategy = 'roulette'  # More exploratory
        elif diversity > 0.5:  # High diversity - can be more selective
            self.current_strategy = 'tournament'  # More exploitative
        else:  # Medium diversity - balanced approach
            self.current_strategy = 'rank'
        
        # Optionally track performance (simplified)
        if self.current_strategy in self.strategy_performance:
            self.strategy_performance[self.current_strategy].append(diversity)


def create_selector(strategy_name: str, **kwargs) -> SelectionStrategy:
    """
    Factory function to create selection strategies.
    
    Args:
        strategy_name: Name of the selection strategy
        **kwargs: Strategy-specific parameters
        
    Returns:
        SelectionStrategy instance
    """
    strategy_map = {
        'tournament': TournamentSelection,
        'roulette': RouletteWheelSelection,
        'rank': RankBasedSelection,
        'sus': StochasticUniversalSampling,
        'multi_objective': MultiObjectiveSelection,
        'adaptive': AdaptiveSelection
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown selection strategy: {strategy_name}")
    
    return strategy_map[strategy_name](**kwargs)