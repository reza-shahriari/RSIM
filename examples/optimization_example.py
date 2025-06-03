#!/usr/bin/env python3
"""
Example usage of optimization algorithms in simlib.

This script demonstrates how to use the four optimization algorithms:
- Simulated Annealing
- Genetic Algorithm  
- Particle Swarm Optimization
- Response Surface Methodology

All algorithms are tested on common benchmark functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add simlib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simlib.optimization import (SimulatedAnnealing, GeneticAlgorithm, 
                                ParticleSwarmOptimization, ResponseSurfaceMethodology)

def rosenbrock(x):
    """Rosenbrock function - classic optimization test function
    Global minimum at (1, 1) with value 0
    """
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def sphere(x):
    """Sphere function - simple convex optimization test
    Global minimum at origin with value 0
    """
    return np.sum(x**2)

def rastrigin(x):
    """Rastrigin function - multimodal test function
    Global minimum at origin with value 0
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    """Ackley function - multimodal test function
    Global minimum at origin with value 0
    """
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

def run_optimization_comparison():
    """Compare all optimization algorithms on benchmark functions"""
    
    # Test functions with their bounds and known optima
    test_functions = {
        'Rosenbrock': {
            'func': rosenbrock,
            'bounds': [(-2, 2), (-1, 3)],
            'optimum': [1, 1],
            'min_value': 0
        },
        'Sphere': {
            'func': sphere,
            'bounds': [(-5, 5), (-5, 5)],
            'optimum': [0, 0],
            'min_value': 0
        },
        'Rastrigin': {
            'func': rastrigin,
            'bounds': [(-5.12, 5.12), (-5.12, 5.12)],
            'optimum': [0, 0],
            'min_value': 0
        },
        'Ackley': {
            'func': ackley,
            'bounds': [(-5, 5), (-5, 5)],
            'optimum': [0, 0],
            'min_value': 0
        }
    }
    
    # Choose function to optimize
    func_name = 'Rosenbrock'  # Change this to test different functions
    test_func = test_functions[func_name]
    
    print(f"Optimizing {func_name} function")
    print(f"Known optimum: {test_func['optimum']} with value {test_func['min_value']}")
    print("=" * 60)
    
    # Common parameters
    bounds = test_func['bounds']
    objective = test_func['func']
    random_seed = 42
    
    results = {}
    
    # 1. Simulated Annealing
    print("\n1. Simulated Annealing")
    print("-" * 30)
    initial_solution = np.array([0.5, 0.5])  # Starting point
    sa = SimulatedAnnealing(
        objective_function=objective,
        initial_solution=initial_solution,
        bounds=bounds,
        initial_temperature=100.0,
        cooling_rate=0.95,
        max_iterations=5000,
        step_size=0.5,
        random_seed=random_seed
    )
    
    sa_result = sa.run()
    results['SA'] = sa_result
    
    print(f"Best solution: {sa_result.results['best_solution']}")
    print(f"Best cost: {sa_result.results['best_cost']:.6f}")
    print(f"Iterations: {sa_result.results['iterations']}")
    print(f"Execution time: {sa_result.execution_time:.3f}s")
    
    # 2. Genetic Algorithm
    print("\n2. Genetic Algorithm")
    print("-" * 30)
    ga = GeneticAlgorithm(
        objective_function=objective,
        bounds=bounds,
        population_size=50,
        max_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        random_seed=random_seed
    )
    
    ga_result = ga.run()
    results['GA'] = ga_result
    
    print(f"Best solution: {ga_result.results['best_solution']}")
    print(f"Best fitness: {ga_result.results['best_fitness']:.6f}")
    print(f"Generations: {ga_result.results['generations']}")
    print(f"Execution time: {ga_result.execution_time:.3f}s")
    
    # 3. Particle Swarm Optimization
    print("\n3. Particle Swarm Optimization")
    print("-" * 30)
    pso = ParticleSwarmOptimization(
        objective_function=objective,
        bounds=bounds,
        n_particles=30,
        max_iterations=100,
        w=0.7,
        c1=2.0,
        c2=2.0,
        random_seed=random_seed
    )
    
    pso_result = pso.run()
    results['PSO'] = pso_result
    
    print(f"Best solution: {pso_result.results['best_solution']}")
    print(f"Best fitness: {pso_result.results['best_fitness']:.6f}")
    print(f"Iterations: {pso_result.results['iterations']}")
    print(f"Execution time: {pso_result.execution_time:.3f}s")
    
    # 4. Response Surface Methodology
    print("\n4. Response Surface Methodology")
    print("-" * 30)
    rsm = ResponseSurfaceMethodology(
        objective_function=objective,
        bounds=bounds,
        polynomial_degree=2,
        max_iterations=20,
        step_size=0.3,
        random_seed=random_seed
    )
    
    rsm_result = rsm.run()
    results['RSM'] = rsm_result
    
    print(f"Best solution: {rsm_result.results['best_solution']}")
    print(f"Best fitness: {rsm_result.results['best_fitness']:.6f}")
    print(f"Iterations: {rsm_result.results['iterations']}")
    print(f"Converged: {rsm_result.results['converged']}")
    print(f"Execution time: {rsm_result.execution_time:.3f}s")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"{'Algorithm':<25} {'Best Value':<15} {'Error':<15} {'Time (s)':<10}")
    print("-" * 65)
    
    true_optimum = test_func['min_value']
    
    for name, result in results.items():
        if name == 'SA':
            best_value = result.results['best_cost']
        else:
            best_value = result.results['best_fitness']
        
        error = abs(best_value - true_optimum)
        time_taken = result.execution_time
        
        print(f"{name:<25} {best_value:<15.6f} {error:<15.6e} {time_taken:<10.3f}")
    
    # Create visualization
    create_comparison_plots(results, func_name)
    
    return results

def create_comparison_plots(results, func_name):
    """Create comparison plots for all algorithms"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Convergence comparison
    for name, result in results.items():
        if name == 'SA':
            costs = result.raw_data['best_costs']
            ax1.plot(costs, label=f'{name}', linewidth=2)
        elif name == 'GA':
            fitness = result.raw_data['best_fitness_history']
            ax1.plot(fitness, label=f'{name}', linewidth=2)
        elif name == 'PSO':
            fitness = result.raw_data['global_best_history']
            ax1.plot(fitness, label=f'{name}', linewidth=2)
        elif name == 'RSM':
            fitness = result.raw_data['best_fitness_history']
            ax1.plot(fitness, label=f'{name}', linewidth=2, marker='o')
    
    ax1.set_xlabel('Iteration/Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title(f'Convergence Comparison - {func_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Final solutions
    true_opt = [1, 1] if func_name == 'Rosenbrock' else [0, 0]
    
    for i, (name, result) in enumerate(results.items()):
        if name == 'SA':
            solution = result.results['best_solution']
        else:
            solution = result.results['best_solution']
        
        ax2.scatter(solution[0], solution[1], s=100, label=f'{name}', alpha=0.7)
    
    ax2.scatter(true_opt[0], true_opt[1], s=200, color='red', marker='*', 
               label='True Optimum', zorder=5)
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_title('Final Solutions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Execution times
    names = list(results.keys())
    times = [result.execution_time for result in results.values()]
    
    bars = ax3.bar(names, times, alpha=0.7, color=['blue', 'orange', 'green', 'red'])
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Execution Time Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.3f}s', ha='center', va='bottom')
    
    # Plot 4: Error from true optimum
    true_min = 0  # All test functions have minimum value of 0
    errors = []
    
    for name, result in results.items():
        if name == 'SA':
            best_value = result.results['best_cost']
        else:
            best_value = result.results['best_fitness']
        errors.append(abs(best_value - true_min))
    
    bars = ax4.bar(names, errors, alpha=0.7, color=['blue', 'orange', 'green', 'red'])
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Error from True Optimum')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{error:.2e}', ha='center', va='bottom', rotation=45)
    
    plt.tight_layout()
    plt.show()

def demonstrate_individual_algorithms():
    """Demonstrate each algorithm individually with visualizations"""
    
    # Use Rosenbrock function for demonstration
    bounds = [(-2, 2), (-1, 3)]
    
    print("Demonstrating individual algorithms with visualizations...")
    print("=" * 60)
    
    # Simulated Annealing
    print("\nSimulated Annealing Demonstration:")
    sa = SimulatedAnnealing(
        objective_function=rosenbrock,
        initial_solution=np.array([-1.5, 2.5]),
        bounds=bounds,
        initial_temperature=50.0,
        cooling_rate=0.95,
        max_iterations=3000,
        random_seed=42
    )
    sa_result = sa.run()
    sa.visualize()
    
    # Genetic Algorithm
    print("\nGenetic Algorithm Demonstration:")
    ga = GeneticAlgorithm(
        objective_function=rosenbrock,
        bounds=bounds,
        population_size=40,
        max_generations=80,
        random_seed=42
    )
    ga_result = ga.run()
    ga.visualize()
    
    # Particle Swarm Optimization
    print("\nParticle Swarm Optimization Demonstration:")
    pso = ParticleSwarmOptimization(
        objective_function=rosenbrock,
        bounds=bounds,
        n_particles=25,
        max_iterations=80,
        random_seed=42
    )
    pso_result = pso.run()
    pso.visualize()
    
    # Response Surface Methodology
    print("\nResponse Surface Methodology Demonstration:")
    rsm = ResponseSurfaceMethodology(
        objective_function=rosenbrock,
        bounds=bounds,
        max_iterations=15,
        random_seed=42
    )
    rsm_result = rsm.run()
    rsm.visualize()

if __name__ == "__main__":
    print("Optimization Algorithms Demonstration")
    print("=" * 60)
    
    # Run comparison
    results = run_optimization_comparison()
    
    # Ask user if they want to see individual demonstrations
    response = input("\nWould you like to see individual algorithm demonstrations? (y/n): ")
    if response.lower() in ['y', 'yes']:
        demonstrate_individual_algorithms()
    
    print("\nDemonstration complete!")
