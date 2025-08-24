"""
Fitness evaluation functions for smart grid optimization.

This module implements multi-objective fitness functions for evaluating
grid configurations based on cost, reliability, efficiency, and constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import math
from ..models.grid_model import SmartGrid
from ..models.chromosome import GridChromosome


class FitnessEvaluator:
    """
    Multi-objective fitness evaluator for smart grid optimization.
    
    Evaluates solutions based on:
    - Economic objectives (generation cost, operation cost)
    - Technical objectives (power losses, voltage stability)
    - Reliability objectives (network connectivity, contingency analysis)
    - Environmental objectives (emissions, renewable integration)
    """
    
    def __init__(self, grid: SmartGrid, weights: Dict[str, float] = None):
        """
        Initialize fitness evaluator.
        
        Args:
            grid: Smart grid model
            weights: Objective weights for multi-objective optimization
        """
        self.grid = grid
        
        # Default objective weights
        self.weights = weights or {
            'generation_cost': 0.4,
            'power_losses': 0.2,
            'voltage_deviation': 0.15,
            'reliability': 0.15,
            'emissions': 0.1
        }
        
        # Normalization factors (will be updated during optimization)
        self.normalization = {
            'generation_cost': 1000.0,  # $/hour
            'power_losses': 10.0,       # MW
            'voltage_deviation': 0.1,    # per-unit
            'reliability': 1.0,          # normalized
            'emissions': 100.0           # tons/hour
        }
        
        # Penalty factors for constraint violations
        self.penalty_factors = {
            'power_balance': 1000.0,
            'voltage_limits': 500.0,
            'thermal_limits': 800.0,
            'generation_limits': 600.0,
            'connectivity': 2000.0
        }
    
    def evaluate(self, chromosome: GridChromosome) -> float:
        """
        Evaluate the fitness of a chromosome.
        
        Args:
            chromosome: Grid chromosome to evaluate
            
        Returns:
            Fitness value (higher is better)
        """
        # Calculate individual objectives
        objectives = self._calculate_objectives(chromosome)
        
        # Calculate constraint violations
        penalties = self._calculate_penalties(chromosome)
        
        # Combine objectives with weights
        weighted_fitness = 0.0
        for obj_name, value in objectives.items():
            if obj_name in self.weights:
                normalized_value = value / self.normalization[obj_name]
                weighted_fitness += self.weights[obj_name] * (1.0 - normalized_value)
        
        # Apply penalties for constraint violations
        total_penalty = sum(penalties.values())
        final_fitness = max(0.0, weighted_fitness - total_penalty)
        
        # Store detailed evaluation results
        chromosome.fitness = final_fitness
        chromosome.is_feasible = total_penalty == 0.0
        
        # Store objective breakdown for analysis
        if not hasattr(chromosome, 'objectives'):
            chromosome.objectives = {}
        chromosome.objectives.update(objectives)
        chromosome.objectives['penalties'] = penalties
        
        return final_fitness
    
    def _calculate_objectives(self, chromosome: GridChromosome) -> Dict[str, float]:
        """Calculate all objective function values."""
        objectives = {}
        
        # 1. Generation Cost Objective
        objectives['generation_cost'] = self._calculate_generation_cost(chromosome)
        
        # 2. Power Losses Objective
        objectives['power_losses'] = self._calculate_power_losses(chromosome)
        
        # 3. Voltage Deviation Objective
        objectives['voltage_deviation'] = self._calculate_voltage_deviation(chromosome)
        
        # 4. Reliability Objective
        objectives['reliability'] = self._calculate_reliability(chromosome)
        
        # 5. Emissions Objective
        objectives['emissions'] = self._calculate_emissions(chromosome)
        
        return objectives
    
    def _calculate_generation_cost(self, chromosome: GridChromosome) -> float:
        """Calculate total generation cost."""
        return self.grid.get_total_generation_cost(chromosome.generator_dispatch)
    
    def _calculate_power_losses(self, chromosome: GridChromosome) -> float:
        """Calculate transmission power losses."""
        # Build admittance matrix
        Y = self.grid.build_admittance_matrix(chromosome.line_status)
        
        # Simplified loss calculation based on voltage levels and line flows
        total_losses = 0.0
        
        for i, line in enumerate(self.grid.lines):
            if i < len(chromosome.line_status) and chromosome.line_status[i] == 1:
                v_from = chromosome.voltage_levels[line.from_node]
                v_to = chromosome.voltage_levels[line.to_node]
                
                # Simplified loss calculation
                voltage_diff = abs(v_from - v_to)
                line_loss = line.reactance * voltage_diff**2 / line.capacity
                total_losses += line_loss
        
        return total_losses
    
    def _calculate_voltage_deviation(self, chromosome: GridChromosome) -> float:
        """Calculate voltage deviation from nominal (1.0 pu)."""
        nominal_voltage = 1.0
        deviations = np.abs(chromosome.voltage_levels - nominal_voltage)
        return np.sum(deviations)
    
    def _calculate_reliability(self, chromosome: GridChromosome) -> float:
        """Calculate system reliability metrics."""
        stats = self.grid.get_network_statistics(chromosome.line_status)
        
        # Reliability based on connectivity and redundancy
        connectivity_score = stats['connectivity']
        
        # Penalize disconnected networks
        connection_penalty = 0.0 if stats['is_connected'] else 1.0
        
        # Reward shorter average path lengths (lower diameter)
        diameter_score = 1.0 / (1.0 + stats['diameter']) if stats['diameter'] != np.inf else 0.0
        
        # Clustering coefficient (network robustness)
        clustering_score = stats['average_clustering']
        
        # Combined reliability score
        reliability_score = (0.4 * connectivity_score + 
                           0.3 * diameter_score + 
                           0.2 * clustering_score + 
                           0.1 * (1.0 - connection_penalty))
        
        return 1.0 - reliability_score  # Return as minimization objective
    
    def _calculate_emissions(self, chromosome: GridChromosome) -> float:
        """Calculate CO2 emissions from generation."""
        # Emission factors (tons CO2/MWh) for different generator types
        # This is simplified - in practice, would depend on fuel type
        base_emission_factor = 0.5  # tons CO2/MWh
        
        total_emissions = 0.0
        for i, gen in enumerate(self.grid.generators):
            if i < len(chromosome.generator_dispatch):
                power = chromosome.generator_dispatch[i]
                # Assume quadratic relationship between power and emissions
                emission_factor = base_emission_factor * (1.0 + 0.1 * power / gen.max_power)
                total_emissions += emission_factor * power
        
        return total_emissions
    
    def _calculate_penalties(self, chromosome: GridChromosome) -> Dict[str, float]:
        """Calculate penalty values for constraint violations."""
        penalties = {}
        
        # 1. Power Balance Penalty
        total_gen = chromosome.get_total_generation()
        total_load = self.grid.get_total_load()
        power_imbalance = abs(total_gen - total_load)
        penalties['power_balance'] = self.penalty_factors['power_balance'] * power_imbalance / total_load
        
        # 2. Voltage Limits Penalty
        min_v, max_v = self.grid.voltage_limits
        voltage_violations = 0.0
        for voltage in chromosome.voltage_levels:
            if voltage < min_v:
                voltage_violations += (min_v - voltage)**2
            elif voltage > max_v:
                voltage_violations += (voltage - max_v)**2
        penalties['voltage_limits'] = self.penalty_factors['voltage_limits'] * voltage_violations
        
        # 3. Thermal Limits Penalty
        line_constraints = self.grid.check_line_constraints(
            chromosome.voltage_levels, 
            np.zeros(self.grid.num_nodes),  # Simplified: zero voltage angles
            chromosome.line_status
        )
        thermal_violations = sum(1.0 for satisfied in line_constraints if not satisfied)
        penalties['thermal_limits'] = self.penalty_factors['thermal_limits'] * thermal_violations
        
        # 4. Generation Limits Penalty
        gen_violations = 0.0
        for i, gen in enumerate(self.grid.generators):
            if i < len(chromosome.generator_dispatch):
                power = chromosome.generator_dispatch[i]
                if power < gen.min_power:
                    gen_violations += (gen.min_power - power)**2
                elif power > gen.max_power:
                    gen_violations += (power - gen.max_power)**2
        penalties['generation_limits'] = self.penalty_factors['generation_limits'] * gen_violations
        
        # 5. Network Connectivity Penalty
        stats = self.grid.get_network_statistics(chromosome.line_status)
        if not stats['is_connected']:
            penalties['connectivity'] = self.penalty_factors['connectivity']
        else:
            penalties['connectivity'] = 0.0
        
        return penalties
    
    def evaluate_population(self, population: List[GridChromosome]) -> List[float]:
        """
        Evaluate fitness for an entire population.
        
        Args:
            population: List of chromosomes to evaluate
            
        Returns:
            List of fitness values
        """
        fitness_values = []
        for chromosome in population:
            fitness = self.evaluate(chromosome)
            fitness_values.append(fitness)
        
        return fitness_values
    
    def get_pareto_front(self, population: List[GridChromosome]) -> List[GridChromosome]:
        """
        Identify Pareto-optimal solutions in the population.
        
        Args:
            population: Population of chromosomes
            
        Returns:
            List of Pareto-optimal chromosomes
        """
        pareto_front = []
        
        for i, candidate in enumerate(population):
            is_dominated = False
            
            for j, other in enumerate(population):
                if i != j and self._dominates(other, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def _dominates(self, solution1: GridChromosome, solution2: GridChromosome) -> bool:
        """
        Check if solution1 dominates solution2 (Pareto dominance).
        
        Args:
            solution1: First solution
            solution2: Second solution
            
        Returns:
            True if solution1 dominates solution2
        """
        if not hasattr(solution1, 'objectives') or not hasattr(solution2, 'objectives'):
            return solution1.fitness > solution2.fitness
        
        obj1 = solution1.objectives
        obj2 = solution2.objectives
        
        # Check if solution1 is better or equal in all objectives
        better_in_any = False
        for obj_name in ['generation_cost', 'power_losses', 'voltage_deviation', 'reliability', 'emissions']:
            if obj_name in obj1 and obj_name in obj2:
                if obj1[obj_name] > obj2[obj_name]:  # Worse in this objective
                    return False
                elif obj1[obj_name] < obj2[obj_name]:  # Better in this objective
                    better_in_any = True
        
        return better_in_any
    
    def update_normalization_factors(self, population: List[GridChromosome]) -> None:
        """
        Update normalization factors based on population statistics.
        
        Args:
            population: Current population for statistics
        """
        if not population:
            return
        
        # Calculate statistics for each objective
        for obj_name in self.normalization.keys():
            values = []
            for chromosome in population:
                if hasattr(chromosome, 'objectives') and obj_name in chromosome.objectives:
                    values.append(chromosome.objectives[obj_name])
            
            if values:
                # Use 90th percentile as normalization factor
                self.normalization[obj_name] = np.percentile(values, 90)
    
    def get_evaluation_summary(self, chromosome: GridChromosome) -> Dict[str, Any]:
        """
        Get detailed evaluation summary for a chromosome.
        
        Args:
            chromosome: Chromosome to analyze
            
        Returns:
            Dictionary with detailed evaluation results
        """
        if not hasattr(chromosome, 'objectives'):
            self.evaluate(chromosome)
        
        summary = {
            'fitness': chromosome.fitness,
            'is_feasible': chromosome.is_feasible,
            'objectives': chromosome.objectives.copy() if hasattr(chromosome, 'objectives') else {},
            'total_generation': chromosome.get_total_generation(),
            'network_connectivity': chromosome.get_network_connectivity(),
            'power_balance': abs(chromosome.get_total_generation() - self.grid.get_total_load()),
            'voltage_range': (np.min(chromosome.voltage_levels), np.max(chromosome.voltage_levels))
        }
        
        return summary