"""
Main genetic algorithm implementation for smart grid optimization.

This module implements the core genetic algorithm that coordinates all components
(population, selection, crossover, mutation) to solve the smart grid optimization problem.
"""

import numpy as np
import time
import random
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass

from .population import Population
from .selection import SelectionStrategy, create_selector
from .crossover import CrossoverStrategy, create_crossover_operator
from .mutation import MutationStrategy, create_mutation_operator
from ..models.grid_model import SmartGrid
from ..models.chromosome import GridChromosome
from ..models.fitness import FitnessEvaluator


@dataclass
class GAParameters:
    """Configuration parameters for the genetic algorithm."""
    population_size: int = 100
    generations: int = 500
    elite_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    
    # Selection parameters
    selection_strategy: str = 'tournament'
    tournament_size: int = 3
    
    # Crossover parameters
    crossover_strategy: str = 'smart_grid'
    
    # Mutation parameters
    mutation_strategy: str = 'adaptive'
    
    # Convergence criteria
    convergence_threshold: float = 1e-6
    stagnation_generations: int = 50
    
    # Multi-objective parameters
    multi_objective: bool = True
    objective_weights: Dict[str, float] = None
    
    # Parallel processing
    parallel_evaluation: bool = False
    num_processes: int = 4


class SmartGridGA:
    """
    Main genetic algorithm class for smart grid optimization.
    
    This class coordinates all GA components to solve the multi-objective
    smart grid energy distribution optimization problem.
    """
    
    def __init__(self, grid: SmartGrid, parameters: GAParameters = None):
        """
        Initialize the genetic algorithm.
        
        Args:
            grid: Smart grid model to optimize
            parameters: GA configuration parameters
        """
        self.grid = grid
        self.parameters = parameters or GAParameters()
        
        # Initialize components
        self.fitness_evaluator = FitnessEvaluator(
            grid, 
            weights=self.parameters.objective_weights
        )
        
        self.population = Population(
            size=self.parameters.population_size,
            grid=grid,
            evaluator=self.fitness_evaluator
        )
        
        # Initialize operators
        self.selector = create_selector(
            self.parameters.selection_strategy,
            tournament_size=self.parameters.tournament_size
        )
        
        self.crossover_operator = create_crossover_operator(
            self.parameters.crossover_strategy,
            grid=grid
        )
        
        self.mutation_operator = create_mutation_operator(
            self.parameters.mutation_strategy,
            grid=grid,
            mutation_rate=self.parameters.mutation_rate
        )
        
        # Algorithm state
        self.current_generation = 0
        self.best_solution = None
        self.best_fitness = -np.inf
        self.convergence_history = []
        self.runtime_statistics = {}
        self.is_converged = False
        
        # Callbacks for monitoring
        self.generation_callbacks: List[Callable] = []
        self.convergence_callbacks: List[Callable] = []
        
        # Multi-objective results
        self.pareto_front = []
        self.pareto_history = []
    
    def optimize(self, verbose: bool = True) -> GridChromosome:
        """
        Run the genetic algorithm optimization.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Best solution found
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting Smart Grid GA Optimization")
            print(f"Grid size: {self.grid.grid_size}")
            print(f"Generators: {len(self.grid.generators)}")
            print(f"Loads: {len(self.grid.loads)}")
            print(f"Population size: {self.parameters.population_size}")
            print(f"Generations: {self.parameters.generations}")
            print("-" * 50)
        
        # Initialize population
        self._initialize_population()
        
        # Main evolution loop
        for generation in range(self.parameters.generations):
            self.current_generation = generation
            
            # Evolve population
            self._evolve_generation()
            
            # Update statistics
            self._update_statistics()
            
            # Check convergence
            if self._check_convergence():
                if verbose:
                    print(f"Converged at generation {generation}")
                break
            
            # Execute callbacks
            self._execute_generation_callbacks()
            
            # Print progress
            if verbose and (generation % 10 == 0 or generation == self.parameters.generations - 1):
                self._print_progress()
        
        # Finalize optimization
        self._finalize_optimization()
        
        end_time = time.time()
        self.runtime_statistics['total_time'] = end_time - start_time
        
        if verbose:
            print("-" * 50)
            print(f"Optimization completed in {self.runtime_statistics['total_time']:.2f} seconds")
            print(f"Best fitness: {self.best_fitness:.6f}")
            print(f"Final generation: {self.current_generation}")
        
        return self.best_solution
    
    def _initialize_population(self) -> None:
        """Initialize the population using appropriate strategies."""
        # Use heuristic initialization for better starting point
        initialization_strategies = ['random', 'economic', 'reliable', 'balanced']
        self.population.initialize_heuristic(initialization_strategies)
        
        # Update normalization factors for fitness evaluation
        self.fitness_evaluator.update_normalization_factors(self.population.chromosomes)
        
        # Initialize best solution
        self.population.update_statistics()
        self.best_solution = self.population.best_chromosome.copy()
        self.best_fitness = self.population.best_fitness
    
    def _evolve_generation(self) -> None:
        """Evolve the population for one generation."""
        # Apply elitism - preserve best individuals
        elite_chromosomes = self.population.apply_elitism(self.parameters.elite_size)
        
        # Generate offspring
        offspring = self._generate_offspring()
        
        # Create new population (elites + offspring)
        new_population = elite_chromosomes + offspring
        
        # Trim to population size (select best individuals)
        new_population.sort(key=lambda x: x.fitness, reverse=True)
        self.population.chromosomes = new_population[:self.parameters.population_size]
        
        # Advance generation
        self.population.advance_generation()
    
    def _generate_offspring(self) -> List[GridChromosome]:
        """Generate offspring through selection, crossover, and mutation."""
        offspring = []
        num_offspring = self.parameters.population_size - self.parameters.elite_size
        
        while len(offspring) < num_offspring:
            # Selection
            parents = self.selector.select(self.population.chromosomes, 2)
            
            if len(parents) < 2:
                break
            
            # Crossover
            if random.random() < self.parameters.crossover_rate:
                child1, child2 = self.crossover_operator.crossover(parents[0], parents[1])
            else:
                child1, child2 = parents[0].copy(), parents[1].copy()
            
            # Mutation
            child1 = self.mutation_operator.mutate(child1)
            child2 = self.mutation_operator.mutate(child2)
            
            # Evaluate offspring
            self.fitness_evaluator.evaluate(child1)
            self.fitness_evaluator.evaluate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:num_offspring]
    
    def _update_statistics(self) -> None:
        """Update algorithm statistics and best solution."""
        self.population.update_statistics()
        
        # Update best solution
        if self.population.best_fitness > self.best_fitness:
            self.best_solution = self.population.best_chromosome.copy()
            self.best_fitness = self.population.best_fitness
        
        # Update convergence history
        self.convergence_history.append({
            'generation': self.current_generation,
            'best_fitness': self.best_fitness,
            'average_fitness': self.population.average_fitness,
            'diversity': self.population.population_diversity,
            'feasible_rate': sum(1 for c in self.population.chromosomes if c.is_feasible) / len(self.population.chromosomes)
        })
        
        # Update Pareto front for multi-objective optimization
        if self.parameters.multi_objective:
            current_pareto = self.fitness_evaluator.get_pareto_front(self.population.chromosomes)
            self.pareto_front = current_pareto
            self.pareto_history.append(len(current_pareto))
        
        # Update fitness normalization factors
        if self.current_generation % 20 == 0:  # Update every 20 generations
            self.fitness_evaluator.update_normalization_factors(self.population.chromosomes)
    
    def _check_convergence(self) -> bool:
        """Check if the algorithm has converged."""
        if len(self.convergence_history) < self.parameters.stagnation_generations:
            return False
        
        # Check fitness improvement stagnation
        recent_best = [entry['best_fitness'] for entry in self.convergence_history[-self.parameters.stagnation_generations:]]
        improvement = max(recent_best) - min(recent_best)
        
        if improvement < self.parameters.convergence_threshold:
            self.is_converged = True
            return True
        
        # Check diversity collapse
        recent_diversity = [entry['diversity'] for entry in self.convergence_history[-10:]]
        avg_diversity = np.mean(recent_diversity)
        
        if avg_diversity < 0.001:  # Very low diversity
            self.is_converged = True
            return True
        
        return False
    
    def _execute_generation_callbacks(self) -> None:
        """Execute generation callbacks for monitoring."""
        for callback in self.generation_callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Warning: Generation callback failed: {e}")
    
    def _print_progress(self) -> None:
        """Print optimization progress."""
        stats = self.population.get_population_summary()
        
        print(f"Gen {self.current_generation:3d}: "
              f"Best={self.best_fitness:.6f}, "
              f"Avg={stats['average_fitness']:.6f}, "
              f"Div={stats['diversity_metrics']['fitness_diversity']:.4f}, "
              f"Feasible={stats['feasibility_rate']:.2%}")
    
    def _finalize_optimization(self) -> None:
        """Finalize optimization and clean up."""
        # Final statistics
        self.runtime_statistics.update({
            'generations_run': self.current_generation + 1,
            'evaluations': (self.current_generation + 1) * self.parameters.population_size,
            'convergence_achieved': self.is_converged,
            'final_diversity': self.population.population_diversity
        })
        
        # Execute convergence callbacks
        for callback in self.convergence_callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Warning: Convergence callback failed: {e}")
    
    def add_generation_callback(self, callback: Callable) -> None:
        """Add a callback function to be called each generation."""
        self.generation_callbacks.append(callback)
    
    def add_convergence_callback(self, callback: Callable) -> None:
        """Add a callback function to be called when optimization finishes."""
        self.convergence_callbacks.append(callback)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        final_population_stats = self.population.get_population_summary()
        
        summary = {
            'parameters': {
                'population_size': self.parameters.population_size,
                'generations': self.parameters.generations,
                'selection_strategy': self.parameters.selection_strategy,
                'crossover_strategy': self.parameters.crossover_strategy,
                'mutation_strategy': self.parameters.mutation_strategy
            },
            'results': {
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.to_dict() if self.best_solution else None,
                'convergence_achieved': self.is_converged,
                'generations_run': self.current_generation + 1
            },
            'performance': self.runtime_statistics,
            'population_stats': final_population_stats,
            'convergence_history': self.convergence_history
        }
        
        if self.parameters.multi_objective:
            summary['pareto_front'] = [sol.to_dict() for sol in self.pareto_front]
            summary['pareto_front_size_history'] = self.pareto_history
        
        return summary
    
    def get_best_solution_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of the best solution."""
        if not self.best_solution:
            return {}
        
        # Validate solution
        is_feasible, violations = self.grid.validate_solution_feasibility(self.best_solution)
        
        # Get fitness breakdown
        fitness_summary = self.fitness_evaluator.get_evaluation_summary(self.best_solution)
        
        # Get network statistics
        network_stats = self.grid.get_network_statistics(self.best_solution.line_status)
        
        analysis = {
            'feasibility': {
                'is_feasible': is_feasible,
                'violations': violations
            },
            'fitness_breakdown': fitness_summary,
            'network_statistics': network_stats,
            'economic_metrics': {
                'total_generation_cost': self.grid.get_total_generation_cost(self.best_solution.generator_dispatch),
                'total_generation': self.best_solution.get_total_generation(),
                'total_load': self.grid.get_total_load()
            },
            'technical_metrics': {
                'voltage_range': (np.min(self.best_solution.voltage_levels), 
                                np.max(self.best_solution.voltage_levels)),
                'voltage_deviation': np.std(self.best_solution.voltage_levels),
                'network_connectivity': self.best_solution.get_network_connectivity()
            }
        }
        
        return analysis
    
    def restart_optimization(self, new_parameters: GAParameters = None) -> None:
        """Restart optimization with new parameters."""
        if new_parameters:
            self.parameters = new_parameters
        
        # Reset state
        self.current_generation = 0
        self.best_solution = None
        self.best_fitness = -np.inf
        self.convergence_history = []
        self.is_converged = False
        self.pareto_front = []
        self.pareto_history = []
        
        # Reinitialize components if needed
        if new_parameters:
            self._reinitialize_components()
    
    def _reinitialize_components(self) -> None:
        """Reinitialize GA components with new parameters."""
        # Update fitness evaluator
        if self.parameters.objective_weights:
            self.fitness_evaluator.weights = self.parameters.objective_weights
        
        # Update population size
        self.population.size = self.parameters.population_size
        
        # Recreate operators
        self.selector = create_selector(
            self.parameters.selection_strategy,
            tournament_size=self.parameters.tournament_size
        )
        
        self.crossover_operator = create_crossover_operator(
            self.parameters.crossover_strategy,
            grid=self.grid
        )
        
        self.mutation_operator = create_mutation_operator(
            self.parameters.mutation_strategy,
            grid=self.grid,
            mutation_rate=self.parameters.mutation_rate
        )
    
    def export_results(self, filename: str) -> None:
        """Export optimization results to file."""
        import json
        
        results = self.get_optimization_summary()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results = convert_numpy(results)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {filename}")


def create_smart_grid_ga(grid_size: Tuple[int, int] = (5, 5),
                        num_generators: int = 3,
                        num_loads: int = 5,
                        **ga_params) -> SmartGridGA:
    """
    Factory function to create a complete smart grid GA system.
    
    Args:
        grid_size: Grid dimensions (rows, cols)
        num_generators: Number of generators
        num_loads: Number of loads
        **ga_params: Additional GA parameters
        
    Returns:
        Configured SmartGridGA instance
    """
    # Create grid
    grid = SmartGrid(grid_size)
    
    # Add generators
    for i in range(num_generators):
        node = random.randint(0, grid.num_nodes - 1)
        min_power = random.uniform(10, 50)
        max_power = random.uniform(min_power + 50, min_power + 200)
        cost_coeff = (
            random.uniform(0.005, 0.02),  # Quadratic coefficient
            random.uniform(10, 30),       # Linear coefficient
            random.uniform(50, 150)       # Constant coefficient
        )
        grid.add_generator(node, min_power, max_power, cost_coeff)
    
    # Add loads
    for i in range(num_loads):
        node = random.randint(0, grid.num_nodes - 1)
        demand = random.uniform(20, 100)
        priority = random.randint(1, 5)
        grid.add_load(node, demand, priority)
    
    # Create GA parameters
    parameters = GAParameters(**ga_params)
    
    # Create GA
    ga = SmartGridGA(grid, parameters)
    
    return ga