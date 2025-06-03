import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Callable, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult

class ParticleSwarmOptimization(BaseSimulation):
    """
    Particle Swarm Optimization (PSO) algorithm implementation.
    
    PSO is a population-based optimization technique inspired by the social behavior
    of bird flocking or fish schooling. Particles move through the search space
    influenced by their own best position and the global best position found by
    the swarm.
    
    Mathematical Background:
    -----------------------
    - Velocity update: v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
    - Position update: x(t+1) = x(t) + v(t+1)
    - Inertia weight w controls exploration vs exploitation
    - Cognitive coefficient c1 influences personal best attraction
    - Social coefficient c2 influences global best attraction
    
    Algorithm Steps:
    ---------------
    1. Initialize particles with random positions and velocities
    2. Evaluate fitness of each particle
    3. Update personal best and global best positions
    4. Update velocities based on PSO equation
    5. Update positions and apply boundary constraints
    6. Repeat until convergence or max iterations
    
    Applications:
    ------------
    - Continuous function optimization
    - Neural network training
    - Feature selection
    - Parameter tuning
    - Engineering design optimization
    - Portfolio optimization
    
    Parameters:
    -----------
    objective_function : callable
        Function to minimize f(x) -> float
    bounds : list of tuples
        [(min, max)] bounds for each dimension
    n_particles : int, default=30
        Number of particles in swarm
    max_iterations : int, default=100
        Maximum number of iterations
    w : float, default=0.7
        Inertia weight
    c1 : float, default=2.0
        Cognitive coefficient (personal best attraction)
    c2 : float, default=2.0
        Social coefficient (global best attraction)
    v_max : float, optional
        Maximum velocity (fraction of search space)
    random_seed : int, optional
        Seed for reproducible results
    """
    
    def __init__(self, objective_function: Callable, bounds: List[Tuple[float, float]],
                 n_particles: int = 30, max_iterations: int = 100,
                 w: float = 0.7, c1: float = 2.0, c2: float = 2.0,
                 v_max: Optional[float] = None, random_seed: Optional[int] = None):
        super().__init__("Particle Swarm Optimization")
        
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dimension = len(bounds)
        
        # Set maximum velocity as fraction of search space
        if v_max is None:
            self.v_max = 0.2 * np.array([max_val - min_val for min_val, max_val in bounds])
        else:
            self.v_max = v_max * np.array([max_val - min_val for min_val, max_val in bounds])
        
        self.parameters.update({
            'n_particles': n_particles,
            'max_iterations': max_iterations,
            'w': w,
            'c1': c1,
            'c2': c2,
            'v_max_fraction': v_max if v_max is not None else 0.2,
            'dimension': self.dimension,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.is_configured = True
    
    def configure(self, n_particles: int = 30, max_iterations: int = 100,
                 w: float = 0.7, c1: float = 2.0, c2: float = 2.0,
                 v_max: Optional[float] = None) -> bool:
        """Configure PSO parameters"""
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Update maximum velocity
        if v_max is None:
            self.v_max = 0.2 * np.array([max_val - min_val for min_val, max_val in self.bounds])
        else:
            self.v_max = v_max * np.array([max_val - min_val for min_val, max_val in self.bounds])
        
        self.parameters.update({
            'n_particles': n_particles,
            'max_iterations': max_iterations,
            'w': w,
            'c1': c1,
            'c2': c2,
            'v_max_fraction': v_max if v_max is not None else 0.2
        })
        
        self.is_configured = True
        return True
    
    def _initialize_swarm(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particle positions and velocities"""
        positions = np.zeros((self.n_particles, self.dimension))
        velocities = np.zeros((self.n_particles, self.dimension))
        
        for i in range(self.dimension):
            min_val, max_val = self.bounds[i]
            positions[:, i] = np.random.uniform(min_val, max_val, self.n_particles)
            velocities[:, i] = np.random.uniform(-self.v_max[i], self.v_max[i], self.n_particles)
        
        return positions, velocities
    
    def _apply_bounds(self, positions: np.ndarray) -> np.ndarray:
        """Apply boundary constraints to positions"""
        bounded_positions = positions.copy()
        for i in range(self.dimension):
            min_val, max_val = self.bounds[i]
            bounded_positions[:, i] = np.clip(bounded_positions[:, i], min_val, max_val)
        return bounded_positions
    
    def _apply_velocity_bounds(self, velocities: np.ndarray) -> np.ndarray:
        """Apply velocity constraints"""
        bounded_velocities = velocities.copy()
        for i in range(self.dimension):
            bounded_velocities[:, i] = np.clip(bounded_velocities[:, i], 
                                             -self.v_max[i], self.v_max[i])
        return bounded_velocities
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute particle swarm optimization"""
        if not self.is_configured:
            raise RuntimeError("Algorithm not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Initialize swarm
        positions, velocities = self._initialize_swarm()
        
        # Evaluate initial fitness
        fitness = np.array([self.objective_function(pos) for pos in positions])
        
        # Initialize personal best
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()
        
        # Initialize global best
        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = fitness[global_best_idx]
        
        # Tracking arrays
        global_best_history = [global_best_fitness]
        mean_fitness_history = [np.mean(fitness)]
        std_fitness_history = [np.std(fitness)]
        position_history = [positions.copy()]
        velocity_history = [velocities.copy()]
        
        # PSO main loop
        for iteration in range(self.max_iterations):
            # Update velocities
            r1 = np.random.random((self.n_particles, self.dimension))
            r2 = np.random.random((self.n_particles, self.dimension))
            
            cognitive_component = self.c1 * r1 * (personal_best_positions - positions)
            social_component = self.c2 * r2 * (global_best_position - positions)
            
            velocities = (self.w * velocities + 
                         cognitive_component + 
                         social_component)
            
            # Apply velocity bounds
            velocities = self._apply_velocity_bounds(velocities)
            
            # Update positions
            positions = positions + velocities
            
            # Apply position bounds
            positions = self._apply_bounds(positions)
            
            # Evaluate fitness
            fitness = np.array([self.objective_function(pos) for pos in positions])
            
            # Update personal best
            improved_mask = fitness < personal_best_fitness
            personal_best_positions[improved_mask] = positions[improved_mask]
            personal_best_fitness[improved_mask] = fitness[improved_mask]
            
            # Update global best
            best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[best_idx] < global_best_fitness:
                global_best_position = personal_best_positions[best_idx].copy()
                global_best_fitness = personal_best_fitness[best_idx]
            
            # Record progress
            global_best_history.append(global_best_fitness)
            mean_fitness_history.append(np.mean(fitness))
            std_fitness_history.append(np.std(fitness))
            position_history.append(positions.copy())
            velocity_history.append(velocities.copy())
        
        execution_time = time.time() - start_time
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'best_solution': global_best_position.tolist(),
                'best_fitness': global_best_fitness,
                'final_positions': positions.tolist(),
                'final_fitness': fitness.tolist(),
                'iterations': self.max_iterations,
                'convergence_iteration': np.argmin(global_best_history)
            },
            statistics={
                'improvement': global_best_history[0] - global_best_fitness,
                'convergence_rate': (global_best_history[0] - global_best_fitness) / self.max_iterations,
                'final_diversity': np.std(fitness),
                'mean_velocity': np.mean(np.linalg.norm(velocities, axis=1))
            },
            raw_data={
                'global_best_history': global_best_history,
                'mean_fitness_history': mean_fitness_history,
                'std_fitness_history': std_fitness_history,
                'position_history': position_history,
                'velocity_history': velocity_history
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize PSO optimization progress"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No optimization results available. Run the optimization first.")
            return
        
        global_best = result.raw_data['global_best_history']
        mean_fitness = result.raw_data['mean_fitness_history']
        std_fitness = result.raw_data['std_fitness_history']
        position_history = result.raw_data['position_history']
        velocity_history = result.raw_data['velocity_history']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Fitness evolution
        iterations = range(len(global_best))
        ax1.plot(iterations, global_best, 'r-', linewidth=2, label='Global Best')
        ax1.plot(iterations, mean_fitness, 'b-', linewidth=1, label='Mean Fitness')
        ax1.fill_between(iterations,
                        np.array(mean_fitness) - np.array(std_fitness),
                        np.array(mean_fitness) + np.array(std_fitness),
                        alpha=0.3, color='blue', label='±1 Std Dev')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution Over Iterations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Swarm diversity
        ax2.plot(iterations, std_fitness, 'green', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness Standard Deviation')
        ax2.set_title('Swarm Diversity Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Particle trajectories (2D case) or parameter evolution
        if self.dimension == 2:
            # Show particle trajectories for 2D problems
            for i in range(min(10, self.n_particles)):  # Show first 10 particles
                trajectory = np.array([pos[i] for pos in position_history])
                ax3.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.6, linewidth=1)
                ax3.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=30, alpha=0.7)
                ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=30, alpha=0.7)
            
            # Mark global best
            ax3.scatter(result.results['best_solution'][0], 
                       result.results['best_solution'][1],
                       color='gold', s=100, marker='*', label='Global Best', zorder=5)
            ax3.set_xlabel('X₁')
            ax3.set_ylabel('X₂')
            ax3.set_title('Particle Trajectories')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            # For higher dimensions, show global best evolution
            global_best_positions = []
            for iteration in range(len(position_history)):
                positions = position_history[iteration]
                fitness = [self.objective_function(pos) for pos in positions]
                best_idx = np.argmin(fitness)
                global_best_positions.append(positions[best_idx])
            
            global_best_array = np.array(global_best_positions)
            for dim in range(min(3, self.dimension)):
                ax3.plot(iterations[:-1], global_best_array[:, dim], 
                        label=f'x_{dim+1}', alpha=0.7)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Parameter Value')
            ax3.set_title('Global Best Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Velocity analysis
        velocity_magnitudes = []
        for velocities in velocity_history:
            magnitudes = np.linalg.norm(velocities, axis=1)
            velocity_magnitudes.append([np.mean(magnitudes), np.std(magnitudes)])
        
        velocity_magnitudes = np.array(velocity_magnitudes)
        ax4.plot(iterations, velocity_magnitudes[:, 0], 'purple', linewidth=2, label='Mean Velocity')
        ax4.fill_between(iterations,
                        velocity_magnitudes[:, 0] - velocity_magnitudes[:, 1],
                        velocity_magnitudes[:, 0] + velocity_magnitudes[:, 1],
                        alpha=0.3, color='purple', label='±1 Std Dev')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Velocity Magnitude')
        ax4.set_title('Swarm Velocity Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\nOptimization Summary:")
        print(f"Best fitness: {result.results['best_fitness']:.6f}")
        print(f"Best solution: {result.results['best_solution']}")
        print(f"Iterations: {result.results['iterations']}")
        print(f"Convergence iteration: {result.results['convergence_iteration']}")
        print(f"Total improvement: {result.statistics['improvement']:.6f}")
        print(f"Final swarm diversity: {result.statistics['final_diversity']:.6f}")
        print(f"Mean final velocity: {result.statistics['mean_velocity']:.6f}")
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'n_particles': {
                'type': 'int',
                'default': 30,
                'min': 5,
                'max': 200,
                'description': 'Number of particles in swarm'
            },
            'max_iterations': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 1000,
                'description': 'Maximum number of iterations'
            },
            'w': {
                'type': 'float',
                'default': 0.7,
                'min': 0.1,
                'max': 1.0,
                'description': 'Inertia weight'
            },
            'c1': {
                'type': 'float',
                'default': 2.0,
                'min': 0.0,
                'max': 4.0,
                'description': 'Cognitive coefficient (personal best attraction)'
            },
            'c2': {
                'type': 'float',
                'default': 2.0,
                'min': 0.0,
                'max': 4.0,
                'description': 'Social coefficient (global best attraction)'
            },
            'v_max_fraction': {
                'type': 'float',
                'default': 0.2,
                'min': 0.05,
                'max': 1.0,
                'description': 'Maximum velocity as fraction of search space'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate optimization parameters"""
        errors = []
        if self.n_particles < 2:
            errors.append("n_particles must be at least 2")
        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")
        if self.w < 0:
            errors.append("w (inertia weight) must be non-negative")
        if self.c1 < 0:
            errors.append("c1 (cognitive coefficient) must be non-negative")
        if self.c2 < 0:
            errors.append("c2 (social coefficient) must be non-negative")
        if self.c1 + self.c2 == 0:
            errors.append("At least one of c1 or c2 must be positive")
        return errors
