"""
OpenMP parallelization for genetic algorithm operations.

This module provides parallel implementations of GA operations using
multiprocessing to simulate OpenMP-style parallelization.
"""

import multiprocessing as mp
import numpy as np
import time
from typing import List, Callable, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from ..models.chromosome import GridChromosome
from ..models.grid_model import SmartGrid
from ..models.fitness import FitnessEvaluator


class ParallelEvaluator:
    """
    Parallel fitness evaluator for population evaluation.
    
    Distributes fitness evaluation across multiple processes
    to accelerate population evaluation.
    """
    
    def __init__(self, grid: SmartGrid, evaluator: FitnessEvaluator, 
                 num_processes: int = None):
        """
        Initialize parallel evaluator.
        
        Args:
            grid: Smart grid model
            evaluator: Fitness evaluator
            num_processes: Number of parallel processes (default: CPU count)
        """
        self.grid = grid
        self.evaluator = evaluator
        self.num_processes = num_processes or mp.cpu_count()
        
        print(f"Parallel evaluator initialized with {self.num_processes} processes")
    
    def evaluate_population_parallel(self, population: List[GridChromosome]) -> List[float]:
        """
        Evaluate population fitness in parallel.
        
        Args:
            population: List of chromosomes to evaluate
            
        Returns:
            List of fitness values
        """
        if len(population) <= self.num_processes:
            # For small populations, sequential evaluation might be faster
            return [self.evaluator.evaluate(chr) for chr in population]
        
        # Split population into chunks for parallel processing
        chunk_size = max(1, len(population) // self.num_processes)
        chunks = [population[i:i + chunk_size] for i in range(0, len(population), chunk_size)]
        
        fitness_values = []
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit evaluation tasks
            future_to_chunk = {
                executor.submit(self._evaluate_chunk, chunk): chunk 
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_fitness = future.result()
                fitness_values.extend(chunk_fitness)
        
        return fitness_values
    
    def _evaluate_chunk(self, chunk: List[GridChromosome]) -> List[float]:
        """Evaluate a chunk of chromosomes (executed in subprocess)."""
        # Create evaluator instance in subprocess
        evaluator = FitnessEvaluator(self.grid, weights=self.evaluator.weights)
        
        fitness_values = []
        for chromosome in chunk:
            fitness = evaluator.evaluate(chromosome)
            fitness_values.append(fitness)
        
        return fitness_values


class ParallelGeneticOperations:
    """
    Parallel implementations of genetic operations.
    
    Provides parallel crossover and mutation operations
    for improved performance on large populations.
    """
    
    def __init__(self, num_processes: int = None):
        """
        Initialize parallel genetic operations.
        
        Args:
            num_processes: Number of parallel processes
        """
        self.num_processes = num_processes or mp.cpu_count()
    
    def parallel_crossover(self, parents: List[GridChromosome], 
                          crossover_operator, crossover_rate: float) -> List[GridChromosome]:
        """
        Perform crossover operations in parallel.
        
        Args:
            parents: List of parent chromosomes
            crossover_operator: Crossover operator to use
            crossover_rate: Probability of crossover
            
        Returns:
            List of offspring chromosomes
        """
        if len(parents) < 2:
            return parents.copy()
        
        # Pair parents for crossover
        parent_pairs = []
        for i in range(0, len(parents) - 1, 2):
            parent_pairs.append((parents[i], parents[i + 1]))
        
        if len(parent_pairs) <= 2:
            # Sequential for small numbers
            offspring = []
            for parent1, parent2 in parent_pairs:
                if np.random.random() < crossover_rate:
                    child1, child2 = crossover_operator.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                offspring.extend([child1, child2])
            return offspring
        
        # Parallel crossover for larger populations
        offspring = []
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit crossover tasks
            futures = []
            for parent1, parent2 in parent_pairs:
                future = executor.submit(
                    self._crossover_pair, 
                    parent1, parent2, crossover_operator, crossover_rate
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                child1, child2 = future.result()
                offspring.extend([child1, child2])
        
        return offspring
    
    def _crossover_pair(self, parent1: GridChromosome, parent2: GridChromosome,
                       crossover_operator, crossover_rate: float) -> tuple:
        """Perform crossover on a pair of parents (executed in subprocess)."""
        if np.random.random() < crossover_rate:
            return crossover_operator.crossover(parent1, parent2)
        else:
            return parent1.copy(), parent2.copy()
    
    def parallel_mutation(self, population: List[GridChromosome], 
                         mutation_operator) -> List[GridChromosome]:
        """
        Perform mutation operations in parallel.
        
        Args:
            population: Population to mutate
            mutation_operator: Mutation operator to use
            
        Returns:
            List of mutated chromosomes
        """
        if len(population) <= self.num_processes:
            # Sequential for small populations
            return [mutation_operator.mutate(chr) for chr in population]
        
        # Parallel mutation
        mutated_population = []
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit mutation tasks
            futures = [
                executor.submit(mutation_operator.mutate, chromosome) 
                for chromosome in population
            ]
            
            # Collect results
            for future in as_completed(futures):
                mutated_chromosome = future.result()
                mutated_population.append(mutated_chromosome)
        
        return mutated_population


class ParallelGA:
    """
    Parallel genetic algorithm implementation.
    
    Coordinates parallel evaluation and genetic operations
    for improved performance on multi-core systems.
    """
    
    def __init__(self, ga_instance, num_processes: int = None):
        """
        Initialize parallel GA wrapper.
        
        Args:
            ga_instance: Base GA instance to parallelize
            num_processes: Number of parallel processes
        """
        self.ga = ga_instance
        self.num_processes = num_processes or mp.cpu_count()
        
        # Initialize parallel components
        self.parallel_evaluator = ParallelEvaluator(
            self.ga.grid, 
            self.ga.fitness_evaluator, 
            self.num_processes
        )
        
        self.parallel_operations = ParallelGeneticOperations(self.num_processes)
        
        print(f"Parallel GA initialized with {self.num_processes} processes")
    
    def optimize_parallel(self, verbose: bool = True) -> GridChromosome:
        """
        Run parallel genetic algorithm optimization.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Best solution found
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting Parallel Smart Grid GA Optimization")
            print(f"Processes: {self.num_processes}")
            print(f"Population size: {self.ga.parameters.population_size}")
            print("-" * 50)
        
        # Initialize population
        self.ga._initialize_population()
        
        # Main evolution loop with parallel operations
        for generation in range(self.ga.parameters.generations):
            self.ga.current_generation = generation
            
            # Parallel evolution
            self._evolve_generation_parallel()
            
            # Update statistics
            self.ga._update_statistics()
            
            # Check convergence
            if self.ga._check_convergence():
                if verbose:
                    print(f"Converged at generation {generation}")
                break
            
            # Print progress
            if verbose and (generation % 10 == 0 or generation == self.ga.parameters.generations - 1):
                self.ga._print_progress()
        
        # Finalize
        self.ga._finalize_optimization()
        
        end_time = time.time()
        self.ga.runtime_statistics['total_time'] = end_time - start_time
        
        if verbose:
            print("-" * 50)
            print(f"Parallel optimization completed in {self.ga.runtime_statistics['total_time']:.2f} seconds")
            print(f"Speedup potential: {self.num_processes}x theoretical")
        
        return self.ga.best_solution
    
    def _evolve_generation_parallel(self) -> None:
        """Evolve population using parallel operations."""
        # Apply elitism
        elite_chromosomes = self.ga.population.apply_elitism(self.ga.parameters.elite_size)
        
        # Parallel selection
        parents = self.ga.selector.select(
            self.ga.population.chromosomes, 
            self.ga.parameters.population_size - self.ga.parameters.elite_size
        )
        
        # Parallel crossover
        offspring = self.parallel_operations.parallel_crossover(
            parents, 
            self.ga.crossover_operator, 
            self.ga.parameters.crossover_rate
        )
        
        # Parallel mutation
        offspring = self.parallel_operations.parallel_mutation(
            offspring, 
            self.ga.mutation_operator
        )
        
        # Parallel fitness evaluation
        fitness_values = self.parallel_evaluator.evaluate_population_parallel(offspring)
        
        # Update fitness values
        for chromosome, fitness in zip(offspring, fitness_values):
            chromosome.fitness = fitness
        
        # Create new population
        new_population = elite_chromosomes + offspring[:self.ga.parameters.population_size - self.ga.parameters.elite_size]
        
        # Update population
        self.ga.population.chromosomes = new_population
        self.ga.population.advance_generation()


def benchmark_parallel_performance(grid_size: tuple = (6, 6), 
                                 num_generators: int = 4,
                                 num_loads: int = 6,
                                 population_size: int = 100,
                                 generations: int = 50) -> dict:
    """
    Benchmark parallel vs sequential performance.
    
    Args:
        grid_size: Grid dimensions
        num_generators: Number of generators
        num_loads: Number of loads
        population_size: Population size
        generations: Number of generations
        
    Returns:
        Performance comparison results
    """
    from ..core.genetic_algorithm import create_smart_grid_ga, GAParameters
    
    print("Benchmarking Parallel vs Sequential Performance")
    print("=" * 50)
    
    # Create test system
    parameters = GAParameters(
        population_size=population_size,
        generations=generations,
        elite_size=5
    )
    
    # Sequential benchmark
    print("Running sequential GA...")
    ga_sequential = create_smart_grid_ga(
        grid_size=grid_size,
        num_generators=num_generators,
        num_loads=num_loads,
        **parameters.__dict__
    )
    
    start_time = time.time()
    best_sequential = ga_sequential.optimize(verbose=False)
    sequential_time = time.time() - start_time
    
    # Parallel benchmark
    print("Running parallel GA...")
    ga_parallel_wrapper = create_smart_grid_ga(
        grid_size=grid_size,
        num_generators=num_generators,
        num_loads=num_loads,
        **parameters.__dict__
    )
    
    parallel_ga = ParallelGA(ga_parallel_wrapper)
    
    start_time = time.time()
    best_parallel = parallel_ga.optimize_parallel(verbose=False)
    parallel_time = time.time() - start_time
    
    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    
    results = {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'processes': parallel_ga.num_processes,
        'sequential_fitness': best_sequential.fitness,
        'parallel_fitness': best_parallel.fitness,
        'population_size': population_size,
        'generations': generations
    }
    
    print(f"\nBenchmark Results:")
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Parallel time: {parallel_time:.2f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Processes used: {parallel_ga.num_processes}")
    print(f"Sequential fitness: {best_sequential.fitness:.6f}")
    print(f"Parallel fitness: {best_parallel.fitness:.6f}")
    
    return results


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_parallel_performance()
    print(f"\nParallel processing test completed successfully!")
    print(f"Achieved {results['speedup']:.1f}x speedup with {results['processes']} processes")