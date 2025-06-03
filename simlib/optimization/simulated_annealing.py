import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Callable, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult

class SimulatedAnnealing(BaseSimulation):
    """
    Simulated Annealing optimization algorithm implementation.
    
    Simulated Annealing is a probabilistic optimization technique inspired by the 
    annealing process in metallurgy. It explores the solution space by accepting 
    both improving and worsening moves with a probability that decreases over time,
    allowing escape from local optima.
    
    Mathematical Background:
    -----------------------
    - Acceptance probability: P(accept) = exp(-ΔE / T) for ΔE > 0
    - Temperature schedule: T(t) = T₀ * α^t (geometric cooling)
    - Metropolis criterion for move acceptance
    - Convergence depends on cooling schedule and neighborhood structure
    
    Algorithm Steps:
    ---------------
    1. Initialize with random solution and high temperature
    2. Generate neighbor solution by perturbation
    3. Accept if better, or with probability exp(-ΔE/T) if worse
    4. Reduce temperature according to cooling schedule
    5. Repeat until stopping criteria met
    
    Applications:
    ------------
    - Combinatorial optimization (TSP, scheduling)
    - Continuous function optimization
    - Neural network training
    - Image processing and computer vision
    - VLSI design and circuit layout
    - Protein folding prediction
    
    Parameters:
    -----------
    objective_function : callable
        Function to minimize f(x) -> float
    initial_solution : array-like
        Starting point for optimization
    bounds : list of tuples, optional
        [(min, max)] bounds for each dimension
    initial_temperature : float, default=100.0
        Starting temperature T₀
    cooling_rate : float, default=0.95
        Temperature reduction factor α (0 < α < 1)
    min_temperature : float, default=1e-8
        Minimum temperature (stopping criterion)
    max_iterations : int, default=10000
        Maximum number of iterations
    step_size : float, default=1.0
        Standard deviation for Gaussian perturbations
    random_seed : int, optional
        Seed for reproducible results
    
    Methods:
    --------
    configure(...) : bool
        Configure algorithm parameters
    run(**kwargs) : SimulationResult
        Execute the optimization
    visualize(result=None, **kwargs) : None
        Plot optimization progress and results
    """
    
    def __init__(self, objective_function: Callable, initial_solution: np.ndarray,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 initial_temperature: float = 100.0, cooling_rate: float = 0.95,
                 min_temperature: float = 1e-8, max_iterations: int = 10000,
                 step_size: float = 1.0, random_seed: Optional[int] = None):
        super().__init__("Simulated Annealing")
        
        self.objective_function = objective_function
        self.initial_solution = np.array(initial_solution)
        self.bounds = bounds
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.step_size = step_size
        
        self.parameters.update({
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations': max_iterations,
            'step_size': step_size,
            'random_seed': random_seed,
            'dimension': len(initial_solution)
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.is_configured = True
    
    def configure(self, initial_temperature: float = 100.0, cooling_rate: float = 0.95,
                 min_temperature: float = 1e-8, max_iterations: int = 10000,
                 step_size: float = 1.0) -> bool:
        """Configure simulated annealing parameters"""
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.step_size = step_size
        
        self.parameters.update({
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations': max_iterations,
            'step_size': step_size
        })
        
        self.is_configured = True
        return True
    
    def _generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate neighbor solution by Gaussian perturbation"""
        neighbor = solution + np.random.normal(0, self.step_size, len(solution))
        
        # Apply bounds if specified
        if self.bounds is not None:
            for i, (min_val, max_val) in enumerate(self.bounds):
                neighbor[i] = np.clip(neighbor[i], min_val, max_val)
        
        return neighbor
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute simulated annealing optimization"""
        if not self.is_configured:
            raise RuntimeError("Algorithm not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Initialize
        current_solution = self.initial_solution.copy()
        current_cost = self.objective_function(current_solution)
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        # Tracking arrays
        costs = [current_cost]
        best_costs = [best_cost]
        temperatures = [self.initial_temperature]
        solutions = [current_solution.copy()]
        accepted_moves = []
        
        temperature = self.initial_temperature
        iteration = 0
        
        while temperature > self.min_temperature and iteration < self.max_iterations:
            # Generate neighbor
            neighbor = self._generate_neighbor(current_solution)
            neighbor_cost = self.objective_function(neighbor)
            
            # Calculate cost difference
            delta_cost = neighbor_cost - current_cost
            
            # Accept or reject move
            if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / temperature):
                current_solution = neighbor
                current_cost = neighbor_cost
                accepted_moves.append(True)
                
                # Update best solution
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
            else:
                accepted_moves.append(False)
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Record progress
            costs.append(current_cost)
            best_costs.append(best_cost)
            temperatures.append(temperature)
            solutions.append(current_solution.copy())
            
            iteration += 1
        
        execution_time = time.time() - start_time
        
        # Calculate statistics
        acceptance_rate = np.mean(accepted_moves)
        improvement = self.objective_function(self.initial_solution) - best_cost
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'best_solution': best_solution.tolist(),
                'best_cost': best_cost,
                'initial_cost': self.objective_function(self.initial_solution),
                'improvement': improvement,
                'iterations': iteration,
                'final_temperature': temperature,
                'acceptance_rate': acceptance_rate
            },
            statistics={
                'convergence_iteration': np.argmin(best_costs),
                'cost_reduction_percentage': (improvement / abs(self.objective_function(self.initial_solution))) * 100,
                'average_cost': np.mean(costs),
                'cost_std': np.std(costs)
            },
            raw_data={
                'costs': costs,
                'best_costs': best_costs,
                'temperatures': temperatures,
                'solutions': solutions,
                'accepted_moves': accepted_moves
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize simulated annealing optimization progress"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No optimization results available. Run the optimization first.")
            return
        
        costs = result.raw_data['costs']
        best_costs = result.raw_data['best_costs']
        temperatures = result.raw_data['temperatures']
        solutions = result.raw_data['solutions']
        accepted_moves = result.raw_data['accepted_moves']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Cost evolution
        iterations = range(len(costs))
        ax1.plot(iterations, costs, 'b-', alpha=0.7, linewidth=1, label='Current Cost')
        ax1.plot(iterations, best_costs, 'r-', linewidth=2, label='Best Cost')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.set_title('Cost Evolution During Optimization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Temperature schedule
        ax2.plot(iterations, temperatures, 'orange', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Temperature Cooling Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Acceptance rate over time (moving average)
        window_size = max(1, len(accepted_moves) // 50)
        acceptance_moving_avg = []
        for i in range(len(accepted_moves)):
            start_idx = max(0, i - window_size)
            acceptance_moving_avg.append(np.mean(accepted_moves[start_idx:i+1]))
        
        ax3.plot(range(len(acceptance_moving_avg)), acceptance_moving_avg, 'green', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Acceptance Rate')
        ax3.set_title(f'Acceptance Rate (Moving Average, window={window_size})')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Solution space exploration (for 2D problems)
        if len(self.initial_solution) == 2:
            solutions_array = np.array(solutions)
            x_coords = solutions_array[:, 0]
            y_coords = solutions_array[:, 1]
            
            # Color by iteration (time progression)
            scatter = ax4.scatter(x_coords, y_coords, c=iterations, cmap='viridis', 
                                alpha=0.6, s=20)
            ax4.scatter(self.initial_solution[0], self.initial_solution[1], 
                       color='red', s=100, marker='s', label='Start', zorder=5)
            ax4.scatter(result.results['best_solution'][0], result.results['best_solution'][1], 
                       color='gold', s=100, marker='*', label='Best', zorder=5)
            
            ax4.set_xlabel('X₁')
            ax4.set_ylabel('X₂')
            ax4.set_title('Solution Space Exploration')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Iteration')
        else:
            # For higher dimensions, show parameter evolution
            solutions_array = np.array(solutions)
            for dim in range(min(3, len(self.initial_solution))):
                ax4.plot(iterations, solutions_array[:, dim], 
                        label=f'x_{dim+1}', alpha=0.7)
            
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Parameter Value')
            ax4.set_title('Parameter Evolution Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nOptimization Summary:")
        print(f"Initial cost: {result.results['initial_cost']:.6f}")
        print(f"Best cost: {result.results['best_cost']:.6f}")
        print(f"Improvement: {result.results['improvement']:.6f}")
        print(f"Iterations: {result.results['iterations']}")
        print(f"Acceptance rate: {result.results['acceptance_rate']:.3f}")
        print(f"Best solution: {result.results['best_solution']}")
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'initial_temperature': {
                'type': 'float',
                'default': 100.0,
                'min': 0.1,
                'max': 1000.0,
                'description': 'Starting temperature'
            },
            'cooling_rate': {
                'type': 'float',
                'default': 0.95,
                'min': 0.8,
                'max': 0.999,
                'description': 'Temperature reduction factor'
            },
            'min_temperature': {
                'type': 'float',
                'default': 1e-8,
                'min': 1e-12,
                'max': 1e-3,
                'description': 'Minimum temperature (stopping criterion)'
            },
            'max_iterations': {
                'type': 'int',
                'default': 10000,
                'min': 100,
                'max': 100000,
                'description': 'Maximum number of iterations'
            },
            'step_size': {
                'type': 'float',
                'default': 1.0,
                'min': 0.01,
                'max': 10.0,
                'description': 'Standard deviation for perturbations'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate optimization parameters"""
        errors = []
        if self.initial_temperature <= 0:
            errors.append("initial_temperature must be positive")
        if not (0 < self.cooling_rate < 1):
            errors.append("cooling_rate must be between 0 and 1")
        if self.min_temperature <= 0:
            errors.append("min_temperature must be positive")
        if self.min_temperature >= self.initial_temperature:
            errors.append("min_temperature must be less than initial_temperature")
        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")
        if self.step_size <= 0:
            errors.append("step_size must be positive")
        return errors
