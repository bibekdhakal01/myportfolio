"""
Comprehensive example demonstrating the Smart Grid GA system with parallel processing.

This example showcases:
1. Basic optimization
2. Parallel processing capabilities  
3. Multi-objective optimization
4. Visualization and analysis
5. Performance benchmarking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time

from smart_grid_ga.core.genetic_algorithm import create_smart_grid_ga, GAParameters
from smart_grid_ga.parallel.openmp_ga import ParallelGA, benchmark_parallel_performance
from smart_grid_ga.visualization.convergence_plots import ConvergencePlotter
from smart_grid_ga.visualization.grid_visualization import GridVisualizer


def comprehensive_demo():
    """Run comprehensive demonstration of the Smart Grid GA system."""
    print("Smart Grid Genetic Algorithm - Comprehensive Demo")
    print("=" * 60)
    
    # 1. Basic Sequential Optimization
    print("\n1. Basic Sequential Optimization")
    print("-" * 40)
    
    parameters = GAParameters(
        population_size=40,
        generations=30,
        elite_size=4,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    ga = create_smart_grid_ga(
        grid_size=(4, 4),
        num_generators=3,
        num_loads=4,
        **parameters.__dict__
    )
    
    print(f"System: {ga.grid.grid_size[0]}x{ga.grid.grid_size[1]} grid")
    print(f"Components: {len(ga.grid.generators)} generators, {len(ga.grid.loads)} loads")
    print(f"Total capacity: {sum(g.max_power for g in ga.grid.generators):.1f} MW")
    print(f"Total load: {ga.grid.get_total_load():.1f} MW")
    
    start_time = time.time()
    best_solution = ga.optimize(verbose=False)
    sequential_time = time.time() - start_time
    
    print(f"Sequential optimization: {sequential_time:.2f} seconds")
    print(f"Best fitness: {ga.best_fitness:.6f}")
    
    # 2. Parallel Processing Demo
    print("\n2. Parallel Processing Capabilities")
    print("-" * 40)
    
    # Create parallel wrapper
    ga_parallel = create_smart_grid_ga(
        grid_size=(4, 4),
        num_generators=3,
        num_loads=4,
        **parameters.__dict__
    )
    
    parallel_wrapper = ParallelGA(ga_parallel)
    
    start_time = time.time()
    best_parallel = parallel_wrapper.optimize_parallel(verbose=False)
    parallel_time = time.time() - start_time
    
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    print(f"Parallel optimization: {parallel_time:.2f} seconds")
    print(f"Speedup achieved: {speedup:.2f}x")
    print(f"Processes used: {parallel_wrapper.num_processes}")
    
    # 3. Multi-Objective Optimization
    print("\n3. Multi-Objective Optimization")
    print("-" * 40)
    
    multi_obj_params = GAParameters(
        population_size=60,
        generations=40,
        multi_objective=True,
        objective_weights={
            'generation_cost': 0.3,
            'power_losses': 0.2,
            'voltage_deviation': 0.2,
            'reliability': 0.2,
            'emissions': 0.1
        },
        selection_strategy='multi_objective'
    )
    
    ga_multi = create_smart_grid_ga(
        grid_size=(5, 5),
        num_generators=4,
        num_loads=6,
        **multi_obj_params.__dict__
    )
    
    best_multi = ga_multi.optimize(verbose=False)
    
    pareto_front = ga_multi.pareto_front
    print(f"Pareto front size: {len(pareto_front)}")
    print(f"Multi-objective fitness: {ga_multi.best_fitness:.6f}")
    
    # Show objective breakdown
    if hasattr(best_multi, 'objectives'):
        print("Objective values:")
        for obj_name, value in best_multi.objectives.items():
            if obj_name != 'penalties':
                print(f"  {obj_name}: {value:.4f}")
    
    # 4. Visualization and Analysis
    print("\n4. Visualization and Analysis")
    print("-" * 40)
    
    plotter = ConvergencePlotter()
    visualizer = GridVisualizer()
    
    # Create comprehensive plots
    print("Generating convergence analysis...")
    plotter.plot_fitness_convergence(ga.convergence_history, 'demo_convergence.png')
    plotter.plot_diversity_evolution(ga.convergence_history, 'demo_diversity.png')
    
    print("Generating grid topology visualization...")
    visualizer.plot_grid_topology(ga.grid, best_solution, 'demo_grid.png')
    visualizer.plot_voltage_profile(best_solution, 'demo_voltage.png')
    
    # Performance comparison plot
    print("Creating performance comparison...")
    plt.figure(figsize=(10, 6))
    
    methods = ['Sequential', 'Parallel']
    times = [sequential_time, parallel_time]
    colors = ['skyblue', 'lightgreen']
    
    plt.bar(methods, times, color=colors, alpha=0.7)
    plt.ylabel('Time (seconds)')
    plt.title('Sequential vs Parallel Performance')
    plt.grid(True, alpha=0.3)
    
    # Add speedup annotation
    plt.text(1, parallel_time + 0.1, f'{speedup:.1f}x speedup', 
             ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('demo_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed Solution Analysis
    print("\n5. Best Solution Analysis")
    print("-" * 40)
    
    analysis = ga.get_best_solution_analysis()
    
    print(f"Solution feasibility: {analysis['feasibility']['is_feasible']}")
    print(f"Total generation cost: ${analysis['economic_metrics']['total_generation_cost']:.2f}")
    print(f"Power generation: {analysis['economic_metrics']['total_generation']:.1f} MW")
    print(f"Network connectivity: {analysis['technical_metrics']['network_connectivity']:.1%}")
    
    voltage_min, voltage_max = analysis['technical_metrics']['voltage_range']
    print(f"Voltage range: {voltage_min:.3f} - {voltage_max:.3f} pu")
    print(f"Voltage deviation: {analysis['technical_metrics']['voltage_deviation']:.4f}")
    
    if analysis['feasibility']['violations']:
        print("Constraint violations:")
        for violation in analysis['feasibility']['violations'][:3]:  # Show first 3
            print(f"  - {violation}")
    
    # Generator dispatch summary
    print("\nGenerator dispatch:")
    for i, gen in enumerate(ga.grid.generators):
        if i < len(best_solution.generator_dispatch):
            dispatch = best_solution.generator_dispatch[i]
            utilization = dispatch / gen.max_power * 100
            print(f"  Gen {i}: {dispatch:.1f} MW ({utilization:.1f}% capacity)")
    
    # 6. Performance Summary
    print("\n6. Performance Summary")
    print("-" * 40)
    
    total_evaluations = ga.current_generation * ga.parameters.population_size
    evaluations_per_second = total_evaluations / sequential_time
    
    print(f"Total generations: {ga.current_generation}")
    print(f"Total evaluations: {total_evaluations:,}")
    print(f"Evaluations/second: {evaluations_per_second:.0f}")
    print(f"Convergence: {'Yes' if ga.is_converged else 'No'}")
    
    # Memory usage estimation
    pop_size = ga.parameters.population_size
    chromosome_size = (len(best_solution.generator_dispatch) + 
                      len(best_solution.load_allocation) + 
                      len(best_solution.line_status) + 
                      len(best_solution.voltage_levels))
    
    estimated_memory = pop_size * chromosome_size * 8 / 1024  # KB (assuming 8 bytes per float)
    print(f"Estimated memory usage: {estimated_memory:.1f} KB")
    
    print("\nDemo visualization files created:")
    print("  - demo_convergence.png: Fitness convergence")
    print("  - demo_diversity.png: Population diversity")
    print("  - demo_grid.png: Grid topology")
    print("  - demo_voltage.png: Voltage profile")
    print("  - demo_performance.png: Performance comparison")
    
    return {
        'sequential_ga': ga,
        'parallel_ga': ga_parallel,
        'multi_objective_ga': ga_multi,
        'performance': {
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup
        }
    }


def performance_scaling_test():
    """Test performance scaling with different problem sizes."""
    print("\n" + "=" * 60)
    print("Performance Scaling Test")
    print("=" * 60)
    
    test_configs = [
        {'grid_size': (3, 3), 'generators': 2, 'loads': 3, 'pop_size': 30},
        {'grid_size': (4, 4), 'generators': 3, 'loads': 4, 'pop_size': 40},
        {'grid_size': (5, 5), 'generators': 4, 'loads': 6, 'pop_size': 50},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\nTest {i}: {config['grid_size'][0]}x{config['grid_size'][1]} grid")
        print(f"  Components: {config['generators']} generators, {config['loads']} loads")
        print(f"  Population: {config['pop_size']}")
        
        # Quick benchmark
        start_time = time.time()
        ga = create_smart_grid_ga(
            grid_size=config['grid_size'],
            num_generators=config['generators'],
            num_loads=config['loads'],
            population_size=config['pop_size'],
            generations=20  # Reduced for testing
        )
        
        best = ga.optimize(verbose=False)
        test_time = time.time() - start_time
        
        print(f"  Time: {test_time:.2f} seconds")
        print(f"  Best fitness: {ga.best_fitness:.6f}")
        
        results.append({
            'config': config,
            'time': test_time,
            'fitness': ga.best_fitness,
            'nodes': config['grid_size'][0] * config['grid_size'][1]
        })
    
    # Plot scaling results
    plt.figure(figsize=(10, 6))
    
    nodes = [r['nodes'] for r in results]
    times = [r['time'] for r in results]
    
    plt.plot(nodes, times, 'o-', linewidth=2, markersize=8, color='navy')
    plt.xlabel('Number of Grid Nodes')
    plt.ylabel('Optimization Time (seconds)')
    plt.title('Performance Scaling with Problem Size')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (n, t) in enumerate(zip(nodes, times)):
        plt.annotate(f'{t:.1f}s', (n, t), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('scaling_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nScaling test completed. Results saved to scaling_test.png")
    return results


if __name__ == "__main__":
    try:
        print("Starting comprehensive Smart Grid GA demonstration...")
        
        # Run main demo
        demo_results = comprehensive_demo()
        
        # Run scaling test
        scaling_results = performance_scaling_test()
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nSummary:")
        perf = demo_results['performance']
        print(f"✓ Sequential optimization: {perf['sequential_time']:.2f} seconds")
        print(f"✓ Parallel optimization: {perf['parallel_time']:.2f} seconds") 
        print(f"✓ Speedup achieved: {perf['speedup']:.2f}x")
        print(f"✓ Multi-objective optimization completed")
        print(f"✓ Comprehensive visualizations generated")
        print(f"✓ Performance scaling analysis completed")
        
        print("\nThe Smart Grid GA system is fully functional with:")
        print("- Advanced genetic algorithm implementation")
        print("- Multi-objective optimization capabilities")
        print("- Parallel processing support")
        print("- Comprehensive visualization tools")
        print("- Performance analysis and benchmarking")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)