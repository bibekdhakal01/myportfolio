"""
Population management for the genetic algorithm.

This module handles population initialization, management, and basic operations
for the smart grid optimization genetic algorithm.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from ..models.chromosome import GridChromosome
from ..models.grid_model import SmartGrid
from ..models.fitness import FitnessEvaluator


class Population:
    """
    Manages a population of chromosomes for the genetic algorithm.
    
    Handles population initialization, fitness evaluation, statistics tracking,
    and population-level operations like sorting and selection.
    """
    
    def __init__(self, size: int, grid: SmartGrid, evaluator: FitnessEvaluator):
        """
        Initialize population.
        
        Args:
            size: Population size
            grid: Smart grid model
            evaluator: Fitness evaluator
        """
        self.size = size
        self.grid = grid
        self.evaluator = evaluator
        self.chromosomes: List[GridChromosome] = []
        self.generation = 0
        
        # Population statistics
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.diversity_history = []
        
        # Current population statistics
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.average_fitness = 0.0
        self.population_diversity = 0.0
    
    def initialize_random(self) -> None:
        """Initialize population with random chromosomes."""
        self.chromosomes = []
        
        # Extract generator and load limits from grid
        generator_limits = [(gen.min_power, gen.max_power) for gen in self.grid.generators]
        load_limits = [(0.5 * load.demand, 1.5 * load.demand) for load in self.grid.loads]
        
        for _ in range(self.size):
            chromosome = GridChromosome(
                grid_size=self.grid.grid_size,
                num_generators=len(self.grid.generators),
                num_loads=len(self.grid.loads)
            )
            
            # Randomize chromosome
            chromosome.randomize(generator_limits, load_limits)
            
            self.chromosomes.append(chromosome)
        
        # Evaluate initial population
        self.evaluate_all()
        self.update_statistics()
    
    def initialize_heuristic(self, strategies: List[str] = None) -> None:
        """
        Initialize population using heuristic strategies.
        
        Args:
            strategies: List of initialization strategies to use
        """
        strategies = strategies or ['random', 'economic', 'reliable', 'balanced']
        
        self.chromosomes = []
        chromosomes_per_strategy = self.size // len(strategies)
        
        for strategy in strategies:
            for _ in range(chromosomes_per_strategy):
                chromosome = self._create_heuristic_chromosome(strategy)
                self.chromosomes.append(chromosome)
        
        # Fill remaining slots with random chromosomes
        while len(self.chromosomes) < self.size:
            chromosome = self._create_heuristic_chromosome('random')
            self.chromosomes.append(chromosome)
        
        self.evaluate_all()
        self.update_statistics()
    
    def _create_heuristic_chromosome(self, strategy: str) -> GridChromosome:
        """Create a chromosome using a specific heuristic strategy."""
        chromosome = GridChromosome(
            grid_size=self.grid.grid_size,
            num_generators=len(self.grid.generators),
            num_loads=len(self.grid.loads)
        )
        
        if strategy == 'economic':
            # Economic dispatch: cheaper generators at higher capacity
            total_load = self.grid.get_total_load()
            remaining_load = total_load
            
            # Sort generators by cost (cheapest first)
            sorted_gens = sorted(enumerate(self.grid.generators), 
                               key=lambda x: x[1].cost_coeff[1])  # Linear cost coefficient
            
            for i, (gen_idx, gen) in enumerate(sorted_gens):
                if remaining_load > 0:
                    # Allocate as much as possible to cheaper generators
                    allocation = min(gen.max_power, remaining_load * 1.1)  # 10% margin
                    chromosome.generator_dispatch[gen_idx] = max(gen.min_power, allocation)
                    remaining_load -= allocation
                else:
                    chromosome.generator_dispatch[gen_idx] = gen.min_power
            
        elif strategy == 'reliable':
            # Reliable configuration: maximize network connectivity
            chromosome.line_status = np.ones(len(chromosome.line_status))  # All lines active
            
            # Balanced generation
            total_load = self.grid.get_total_load()
            avg_generation = total_load / len(self.grid.generators)
            
            for i, gen in enumerate(self.grid.generators):
                chromosome.generator_dispatch[i] = np.clip(avg_generation, gen.min_power, gen.max_power)
        
        elif strategy == 'balanced':
            # Balanced approach
            total_load = self.grid.get_total_load()
            
            # Proportional generation based on capacity
            total_capacity = sum(gen.max_power for gen in self.grid.generators)
            
            for i, gen in enumerate(self.grid.generators):
                proportion = gen.max_power / total_capacity
                target_gen = total_load * proportion
                chromosome.generator_dispatch[i] = np.clip(target_gen, gen.min_power, gen.max_power)
            
            # Moderate line connectivity (70% of lines active)
            num_active_lines = int(0.7 * len(chromosome.line_status))
            active_indices = random.sample(range(len(chromosome.line_status)), num_active_lines)
            chromosome.line_status = np.zeros(len(chromosome.line_status))
            chromosome.line_status[active_indices] = 1
        
        else:  # 'random'
            generator_limits = [(gen.min_power, gen.max_power) for gen in self.grid.generators]
            load_limits = [(0.5 * load.demand, 1.5 * load.demand) for load in self.grid.loads]
            chromosome.randomize(generator_limits, load_limits)
        
        # Always randomize voltage levels within limits
        min_v, max_v = self.grid.voltage_limits
        chromosome.voltage_levels = np.random.uniform(min_v, max_v, chromosome.voltage_levels.shape)
        
        # Normalize load allocation
        chromosome.load_allocation = chromosome.load_allocation / np.sum(chromosome.load_allocation)
        
        return chromosome
    
    def evaluate_all(self) -> None:
        """Evaluate fitness for all chromosomes in the population."""
        fitness_values = self.evaluator.evaluate_population(self.chromosomes)
        
        for chromosome, fitness in zip(self.chromosomes, fitness_values):
            chromosome.fitness = fitness
    
    def update_statistics(self) -> None:
        """Update population statistics."""
        if not self.chromosomes:
            return
        
        fitness_values = [chr.fitness for chr in self.chromosomes]
        
        # Best chromosome
        best_idx = np.argmax(fitness_values)
        self.best_chromosome = self.chromosomes[best_idx].copy()
        self.best_fitness = self.best_chromosome.fitness
        
        # Average fitness
        self.average_fitness = np.mean(fitness_values)
        
        # Population diversity (standard deviation of fitness)
        self.population_diversity = np.std(fitness_values)
        
        # Update history
        self.best_fitness_history.append(self.best_fitness)
        self.average_fitness_history.append(self.average_fitness)
        self.diversity_history.append(self.population_diversity)
    
    def sort_by_fitness(self, descending: bool = True) -> None:
        """Sort chromosomes by fitness."""
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=descending)
    
    def get_top_individuals(self, n: int) -> List[GridChromosome]:
        """Get top n individuals by fitness."""
        self.sort_by_fitness()
        return self.chromosomes[:n]
    
    def get_bottom_individuals(self, n: int) -> List[GridChromosome]:
        """Get bottom n individuals by fitness."""
        self.sort_by_fitness()
        return self.chromosomes[-n:]
    
    def replace_worst(self, new_chromosomes: List[GridChromosome]) -> None:
        """
        Replace worst chromosomes with new ones.
        
        Args:
            new_chromosomes: New chromosomes to add
        """
        self.sort_by_fitness()
        n_replace = min(len(new_chromosomes), len(self.chromosomes))
        
        # Replace worst chromosomes
        self.chromosomes[-n_replace:] = new_chromosomes[:n_replace]
        
        # Re-evaluate if needed
        for chromosome in new_chromosomes[:n_replace]:
            if chromosome.fitness == 0.0:  # Not evaluated yet
                self.evaluator.evaluate(chromosome)
        
        self.update_statistics()
    
    def apply_elitism(self, elite_size: int) -> List[GridChromosome]:
        """
        Extract elite individuals for preservation.
        
        Args:
            elite_size: Number of elite individuals to preserve
            
        Returns:
            List of elite chromosomes
        """
        elite_chromosomes = self.get_top_individuals(elite_size)
        return [chr.copy() for chr in elite_chromosomes]
    
    def calculate_diversity_metrics(self) -> Dict[str, float]:
        """
        Calculate detailed diversity metrics for the population.
        
        Returns:
            Dictionary of diversity metrics
        """
        if len(self.chromosomes) < 2:
            return {'fitness_diversity': 0.0, 'genetic_diversity': 0.0, 'phenotypic_diversity': 0.0}
        
        # Fitness diversity
        fitness_values = [chr.fitness for chr in self.chromosomes]
        fitness_diversity = np.std(fitness_values) / (np.mean(fitness_values) + 1e-8)
        
        # Genetic diversity (average pairwise distance)
        genetic_diversity = self._calculate_genetic_diversity()
        
        # Phenotypic diversity (diversity in objective values)
        phenotypic_diversity = self._calculate_phenotypic_diversity()
        
        return {
            'fitness_diversity': fitness_diversity,
            'genetic_diversity': genetic_diversity,
            'phenotypic_diversity': phenotypic_diversity
        }
    
    def _calculate_genetic_diversity(self) -> float:
        """Calculate genetic diversity as average Hamming distance."""
        if len(self.chromosomes) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.chromosomes)):
            for j in range(i + 1, len(self.chromosomes)):
                distance = self._hamming_distance(self.chromosomes[i], self.chromosomes[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _calculate_phenotypic_diversity(self) -> float:
        """Calculate phenotypic diversity based on objective values."""
        if len(self.chromosomes) < 2:
            return 0.0
        
        # Collect objective values
        objective_matrix = []
        for chromosome in self.chromosomes:
            if hasattr(chromosome, 'objectives'):
                obj_values = []
                for obj_name in ['generation_cost', 'power_losses', 'voltage_deviation', 
                               'reliability', 'emissions']:
                    if obj_name in chromosome.objectives:
                        obj_values.append(chromosome.objectives[obj_name])
                if obj_values:
                    objective_matrix.append(obj_values)
        
        if len(objective_matrix) < 2:
            return 0.0
        
        # Calculate average standard deviation across objectives
        objective_matrix = np.array(objective_matrix)
        diversities = []
        
        for i in range(objective_matrix.shape[1]):
            column_std = np.std(objective_matrix[:, i])
            column_mean = np.mean(objective_matrix[:, i])
            if column_mean != 0:
                diversities.append(column_std / abs(column_mean))
        
        return np.mean(diversities) if diversities else 0.0
    
    def _hamming_distance(self, chr1: GridChromosome, chr2: GridChromosome) -> float:
        """Calculate normalized Hamming distance between two chromosomes."""
        total_distance = 0.0
        total_elements = 0
        
        # Compare line status (binary)
        line_diff = np.sum(chr1.line_status != chr2.line_status)
        total_distance += line_diff
        total_elements += len(chr1.line_status)
        
        # Compare generator dispatch (continuous, normalized)
        gen_diff = np.mean(np.abs(chr1.generator_dispatch - chr2.generator_dispatch))
        gen_range = np.max([np.max(chr1.generator_dispatch), np.max(chr2.generator_dispatch)]) + 1e-8
        total_distance += gen_diff / gen_range
        total_elements += 1
        
        # Compare voltage levels (continuous, normalized)
        volt_diff = np.mean(np.abs(chr1.voltage_levels - chr2.voltage_levels))
        total_distance += volt_diff / 0.1  # Normalize by typical voltage range
        total_elements += 1
        
        return total_distance / total_elements
    
    def get_population_summary(self) -> Dict[str, Any]:
        """Get comprehensive population summary."""
        diversity_metrics = self.calculate_diversity_metrics()
        
        feasible_count = sum(1 for chr in self.chromosomes if chr.is_feasible)
        
        summary = {
            'generation': self.generation,
            'size': len(self.chromosomes),
            'best_fitness': self.best_fitness,
            'average_fitness': self.average_fitness,
            'worst_fitness': min(chr.fitness for chr in self.chromosomes) if self.chromosomes else 0,
            'feasible_solutions': feasible_count,
            'feasibility_rate': feasible_count / len(self.chromosomes) if self.chromosomes else 0,
            'diversity_metrics': diversity_metrics,
            'convergence_rate': self._calculate_convergence_rate()
        }
        
        return summary
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate based on fitness history."""
        if len(self.best_fitness_history) < 10:
            return 0.0
        
        recent_improvements = 0
        for i in range(len(self.best_fitness_history) - 10, len(self.best_fitness_history) - 1):
            if self.best_fitness_history[i + 1] > self.best_fitness_history[i]:
                recent_improvements += 1
        
        return recent_improvements / 9.0  # 9 comparisons in 10 generations
    
    def advance_generation(self) -> None:
        """Advance to next generation."""
        self.generation += 1
        self.update_statistics()