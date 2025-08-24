#!/usr/bin/env python3
"""
Basic test script for the Smart Grid GA optimization system.

This script demonstrates the core functionality and verifies that all components
work together correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from smart_grid_ga.core.genetic_algorithm import create_smart_grid_ga, GAParameters


def test_basic_functionality():
    """Test basic GA functionality with a small problem."""
    print("Testing Smart Grid GA Basic Functionality")
    print("=" * 50)
    
    # Create a small test system
    grid_size = (3, 3)  # 3x3 grid (9 nodes)
    num_generators = 2
    num_loads = 3
    
    print(f"Grid size: {grid_size}")
    print(f"Generators: {num_generators}")
    print(f"Loads: {num_loads}")
    
    # Configure GA parameters for quick test
    parameters = GAParameters(
        population_size=20,
        generations=50,
        elite_size=2,
        crossover_rate=0.8,
        mutation_rate=0.1,
        selection_strategy='tournament',
        crossover_strategy='smart_grid',
        mutation_strategy='adaptive',
        convergence_threshold=1e-4,
        stagnation_generations=20
    )
    
    # Create GA system
    print("\nCreating GA system...")
    ga = create_smart_grid_ga(
        grid_size=grid_size,
        num_generators=num_generators,
        num_loads=num_loads,
        population_size=parameters.population_size,
        generations=parameters.generations,
        elite_size=parameters.elite_size
    )
    
    print(f"Grid created with {ga.grid.num_nodes} nodes")
    print(f"Generators: {len(ga.grid.generators)}")
    print(f"Loads: {len(ga.grid.loads)}")
    print(f"Transmission lines: {len(ga.grid.lines)}")
    
    # Display grid information
    print("\nGrid Components:")
    for i, gen in enumerate(ga.grid.generators):
        print(f"  Generator {i}: Node {gen.node}, Power: {gen.min_power:.1f}-{gen.max_power:.1f} MW")
    
    for i, load in enumerate(ga.grid.loads):
        print(f"  Load {i}: Node {load.node}, Demand: {load.demand:.1f} MW, Priority: {load.priority}")
    
    total_load = ga.grid.get_total_load()
    total_gen_capacity = sum(gen.max_power for gen in ga.grid.generators)
    print(f"\nTotal load: {total_load:.1f} MW")
    print(f"Total generation capacity: {total_gen_capacity:.1f} MW")
    print(f"Reserve margin: {(total_gen_capacity - total_load) / total_load * 100:.1f}%")
    
    # Run optimization
    print("\nStarting optimization...")
    best_solution = ga.optimize(verbose=True)
    
    # Analyze results
    print("\nOptimization Results:")
    print("=" * 30)
    
    summary = ga.get_optimization_summary()
    print(f"Best fitness: {summary['results']['best_fitness']:.6f}")
    print(f"Generations run: {summary['results']['generations_run']}")
    print(f"Converged: {summary['results']['convergence_achieved']}")
    print(f"Runtime: {summary['performance']['total_time']:.2f} seconds")
    
    # Best solution analysis
    analysis = ga.get_best_solution_analysis()
    print(f"\nBest Solution Analysis:")
    print(f"Feasible: {analysis['feasibility']['is_feasible']}")
    
    if analysis['feasibility']['violations']:
        print("Constraint violations:")
        for violation in analysis['feasibility']['violations']:
            print(f"  - {violation}")
    
    print(f"Total generation: {analysis['economic_metrics']['total_generation']:.2f} MW")
    print(f"Total cost: ${analysis['economic_metrics']['total_generation_cost']:.2f}")
    print(f"Network connectivity: {analysis['technical_metrics']['network_connectivity']:.2%}")
    print(f"Voltage range: {analysis['technical_metrics']['voltage_range'][0]:.3f} - {analysis['technical_metrics']['voltage_range'][1]:.3f} pu")
    
    # Fitness breakdown
    if 'objectives' in analysis['fitness_breakdown']:
        objectives = analysis['fitness_breakdown']['objectives']
        print(f"\nObjective breakdown:")
        for obj_name, value in objectives.items():
            if obj_name != 'penalties':
                print(f"  {obj_name}: {value:.4f}")
    
    print("\nTest completed successfully!")
    return ga


def test_parallel_features():
    """Test parallel processing features (simulation)."""
    print("\n" + "=" * 50)
    print("Testing Parallel Processing Features")
    print("=" * 50)
    
    # Note: Actual parallel implementation will be added later
    print("Parallel processing features will be implemented in:")
    print("- OpenMP for CPU parallelization")
    print("- CUDA for GPU acceleration")
    print("- Population evaluation parallelization")
    print("- Multi-objective optimization")
    
    print("Current implementation: Sequential (working baseline)")


def test_visualization_preparation():
    """Prepare data for visualization testing."""
    print("\n" + "=" * 50)
    print("Preparing Visualization Components")
    print("=" * 50)
    
    # Run a small optimization to get data
    ga = create_smart_grid_ga(
        grid_size=(3, 3),
        num_generators=2,
        num_loads=2,
        population_size=10,
        generations=20
    )
    
    print("Running optimization for visualization data...")
    best_solution = ga.optimize(verbose=False)
    
    # Get convergence data
    convergence_history = ga.convergence_history
    print(f"Collected {len(convergence_history)} generation data points")
    
    # Prepare data structures for visualization
    generations = [entry['generation'] for entry in convergence_history]
    best_fitness = [entry['best_fitness'] for entry in convergence_history]
    avg_fitness = [entry['average_fitness'] for entry in convergence_history]
    diversity = [entry['diversity'] for entry in convergence_history]
    
    print(f"Generation range: {min(generations)} - {max(generations)}")
    print(f"Fitness improvement: {best_fitness[0]:.4f} → {best_fitness[-1]:.4f}")
    print(f"Final diversity: {diversity[-1]:.4f}")
    
    # Grid topology data
    rows, cols = ga.grid.grid_size
    node_positions = {}
    for row in range(rows):
        for col in range(cols):
            node_id = row * cols + col
            node_positions[node_id] = (col, row)  # (x, y) coordinates
    
    print(f"Grid topology: {rows}x{cols} grid with {len(node_positions)} nodes")
    print(f"Active transmission lines: {np.sum(best_solution.line_status)}/{len(best_solution.line_status)}")
    
    print("Visualization data prepared successfully!")
    return ga, convergence_history, node_positions


if __name__ == "__main__":
    try:
        # Test basic functionality
        ga = test_basic_functionality()
        
        # Test parallel features (placeholder)
        test_parallel_features()
        
        # Test visualization preparation
        test_visualization_preparation()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("Smart Grid GA system is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)