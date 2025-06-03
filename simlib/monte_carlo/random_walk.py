import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult

class RandomWalk1D(BaseSimulation):
    """
    One-dimensional random walk simulation using Monte Carlo methods.
    
    A random walk is a mathematical formalization of a path that consists of a succession 
    of random steps. In one dimension, at each time step, the walker moves either left 
    or right with specified probabilities and step sizes.
    
    Mathematical Background:
    -----------------------
    - At each step i, the walker moves by ±step_size with probability p and (1-p)
    - Position after n steps: X_n = X_0 + Σ(i=1 to n) S_i
    - Expected displacement: E[X_n] = n * step_size * (2p - 1)
    - Variance: Var[X_n] = n * step_size² * 4p(1-p)
    - For symmetric walk (p=0.5): E[X_n] = 0, Var[X_n] = n * step_size²
    
    Applications:
    ------------
    - Brownian motion modeling
    - Stock price movements (simplified)
    - Diffusion processes
    - Physics: particle motion, polymer chains
    - Biology: animal foraging patterns
    - Economics: market fluctuations
    
    Simulation Features:
    -------------------
    - Configurable step probability (bias)
    - Multiple independent walks for statistical analysis
    - Real-time visualization of paths and statistics
    - Theoretical vs empirical comparison
    - Convergence analysis
    
    Parameters:
    -----------
    n_steps : int, default=1000
        Number of steps in each random walk
    step_size : float, default=1.0
        Size of each step (positive value)
    n_walks : int, default=1
        Number of independent walks to simulate
    step_probability : float, default=0.5
        Probability of taking a positive step (vs negative)
        0.5 = symmetric walk, >0.5 = rightward bias, <0.5 = leftward bias
    random_seed : int, optional
        Seed for random number generator for reproducible results
    
    Attributes:
    -----------
    walks : list of numpy.ndarray
        Position data for each walk (including starting position at 0)
    result : SimulationResult
        Complete simulation results including statistics and raw data
    
    Methods:
    --------
    configure(n_steps, step_size, n_walks, step_probability) : bool
        Configure simulation parameters
    run(**kwargs) : SimulationResult
        Execute the random walk simulation
    visualize(result=None, **kwargs) : None
        Create comprehensive visualizations of the walk(s)
    validate_parameters() : List[str]
        Validate current parameters and return any errors
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Simple symmetric random walk
    >>> rw = RandomWalk1D(n_steps=1000, random_seed=42)
    >>> result = rw.run()
    >>> print(f"Final position: {result.results['final_positions'][0]}")
    
    >>> # Biased random walk (tendency to move right)
    >>> rw_biased = RandomWalk1D(n_steps=1000, step_probability=0.7)
    >>> result = rw_biased.run()
    >>> rw_biased.visualize()
    
    >>> # Multiple walks for statistical analysis
    >>> rw_multi = RandomWalk1D(n_steps=500, n_walks=100)
    >>> result = rw_multi.run()
    >>> print(f"Mean final position: {result.results['mean_final_position']}")
    
    Notes:
    ------
    - The walk always starts at position 0
    - Positions array includes the starting position, so length is n_steps + 1
    - For large n_walks, visualization automatically adjusts transparency
    - Theoretical statistics assume independent, identically distributed steps
    - Performance scales as O(n_steps * n_walks)
    
    References:
    -----------
    - Weiss, G. H. (1994). Aspects and Applications of the Random Walk
    - Spitzer, F. (2001). Principles of Random Walk
    - Hughes, B. D. (1995). Random Walks and Random Environments
    """

    
    def __init__(self, n_steps: int = 1000, step_size: float = 1.0, n_walks: int = 1, 
                 step_probability: float = 0.5, random_seed: Optional[int] = None):
        super().__init__("1D Random Walk")
        
        # Initialize parameters
        self.n_steps = n_steps
        self.step_size = step_size
        self.n_walks = n_walks
        self.step_probability = step_probability  # Probability of +1 step (vs -1)
        
        # Store in parameters dict for base class
        self.parameters.update({
            'n_steps': n_steps,
            'step_size': step_size,
            'n_walks': n_walks,
            'step_probability': step_probability,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state for storing walk data
        self.walks = None
        self.is_configured = True
    
    def configure(self, n_steps: int = 1000, step_size: float = 1.0, 
                 n_walks: int = 1, step_probability: float = 0.5) -> bool:
        """Configure random walk parameters"""
        self.n_steps = n_steps
        self.step_size = step_size
        self.n_walks = n_walks
        self.step_probability = step_probability
        
        # Update parameters dict
        self.parameters.update({
            'n_steps': n_steps,
            'step_size': step_size,
            'n_walks': n_walks,
            'step_probability': step_probability
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute random walk simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        walks = []
        final_positions = []
        max_distances = []
        mean_positions = []
        
        for walk_idx in range(self.n_walks):
            # Generate random steps
            random_numbers = np.random.random(self.n_steps)
            steps = np.where(random_numbers < self.step_probability, 
                           self.step_size, -self.step_size)
            
            # Calculate cumulative position (starting at 0)
            positions = np.zeros(self.n_steps + 1)
            positions[1:] = np.cumsum(steps)
            
            walks.append(positions)
            final_positions.append(positions[-1])
            max_distances.append(np.max(np.abs(positions)))
            mean_positions.append(np.mean(positions))
        
        self.walks = walks
        execution_time = time.time() - start_time
        
        # Calculate statistics
        final_positions = np.array(final_positions)
        max_distances = np.array(max_distances)
        mean_positions = np.array(mean_positions)
        
        # Theoretical calculations
        expected_step = (2 * self.step_probability - 1) * self.step_size  # E[X_i]
        theoretical_mean = self.n_steps * expected_step
        theoretical_variance = self.n_steps * (self.step_size ** 2) * (1 - (2 * self.step_probability - 1) ** 2)
        theoretical_std = np.sqrt(theoretical_variance)
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'final_positions': final_positions.tolist(),
                'mean_final_position': np.mean(final_positions),
                'std_final_position': np.std(final_positions),
                'max_distance_reached': np.max(max_distances),
                'mean_max_distance': np.mean(max_distances),
                'min_final_position': np.min(final_positions),
                'max_final_position': np.max(final_positions)
            },
            statistics={
                'theoretical_mean': theoretical_mean,
                'empirical_mean': np.mean(final_positions),
                'theoretical_variance': theoretical_variance,
                'empirical_variance': np.var(final_positions),
                'theoretical_std': theoretical_std,
                'empirical_std': np.std(final_positions),
                'mean_of_means': np.mean(mean_positions)
            },
            raw_data=np.array(walks),
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize random walk simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        walks = result.raw_data
        
        if self.n_walks == 1:
            # Single walk visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot the walk
            steps = range(len(walks[0]))
            ax1.plot(steps, walks[0], 'b-', linewidth=1.5, alpha=0.8)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax1.set_xlabel('Step Number')
            ax1.set_ylabel('Position')
            ax1.set_title(f'1D Random Walk ({self.n_steps} steps, step_size={self.step_size})')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics text
            final_pos = result.results['final_positions'][0]
            max_dist = result.results['max_distance_reached']
            ax1.text(0.02, 0.98, f'Final Position: {final_pos:.2f}\nMax Distance: {max_dist:.2f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Plot step distribution
            steps_taken = np.diff(walks[0])
            unique_steps, counts = np.unique(steps_taken, return_counts=True)
            
            ax2.bar(unique_steps, counts, alpha=0.7, edgecolor='black', width=0.5)
            ax2.set_xlabel('Step Size')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Steps Taken')
            ax2.grid(True, alpha=0.3)
            
            # Add step probability info
            positive_steps = np.sum(steps_taken > 0)
            total_steps = len(steps_taken)
            empirical_prob = positive_steps / total_steps
            ax2.text(0.02, 0.98, f'Positive steps: {positive_steps}/{total_steps}\n'
                                f'Empirical p: {empirical_prob:.3f}\n'
                                f'Theoretical p: {self.step_probability:.3f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            
        else:
            # Multiple walks visualization
            fig = plt.figure(figsize=(16, 12))
            
            # Create 2x3 subplot layout
            ax1 = plt.subplot(2, 3, 1)
            ax2 = plt.subplot(2, 3, 2)
            ax3 = plt.subplot(2, 3, 3)
            ax4 = plt.subplot(2, 3, 4)
            ax5 = plt.subplot(2, 3, 5)
            ax6 = plt.subplot(2, 3, 6)
            
            # Plot 1: All walks
            for i, walk in enumerate(walks):
                steps = range(len(walk))
                alpha = min(0.8, 20.0 / self.n_walks)  # Fade out for many walks
                ax1.plot(steps, walk, linewidth=1, alpha=alpha)
            
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax1.set_xlabel('Step Number')
            ax1.set_ylabel('Position')
            ax1.set_title(f'{self.n_walks} Random Walks ({self.n_steps} steps each)')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Final position histogram
            final_positions = result.results['final_positions']
            bins = min(30, max(10, self.n_walks//5))
            ax2.hist(final_positions, bins=bins, alpha=0.7, edgecolor='black', density=True)
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Origin')
            ax2.axvline(x=np.mean(final_positions), color='green', linestyle='-', linewidth=2,
                       label=f'Empirical Mean: {np.mean(final_positions):.2f}')
            ax2.axvline(x=result.statistics['theoretical_mean'], color='orange', linestyle='-', linewidth=2,
                       label=f'Theoretical Mean: {result.statistics["theoretical_mean"]:.2f}')
            ax2.set_xlabel('Final Position')
            ax2.set_ylabel('Density')
            ax2.set_title('Distribution of Final Positions')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Maximum distance histogram
            max_distances = [np.max(np.abs(walk)) for walk in walks]
            ax3.hist(max_distances, bins=bins, alpha=0.7, edgecolor='black', density=True)
            ax3.axvline(x=np.mean(max_distances), color='green', linestyle='-', linewidth=2,
                       label=f'Mean: {np.mean(max_distances):.2f}')
            ax3.set_xlabel('Maximum Distance from Origin')
            ax3.set_ylabel('Density')
            ax3.set_title('Distribution of Maximum Distances')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Statistics comparison
            theoretical_values = [result.statistics['theoretical_mean'], result.statistics['theoretical_std']]
            empirical_values = [result.statistics['empirical_mean'], result.statistics['empirical_std']]
            
            x = np.arange(2)
            width = 0.35
            
            ax4.bar(x - width/2, theoretical_values, width, label='Theoretical', alpha=0.7, color='blue')
            ax4.bar(x + width/2, empirical_values, width, label='Empirical', alpha=0.7, color='orange')
            
            ax4.set_ylabel('Value')
            ax4.set_title('Theoretical vs Empirical Statistics')
            ax4.set_xticks(x)
            ax4.set_xticklabels(['Mean', 'Std Dev'])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (theo, emp) in enumerate(zip(theoretical_values, empirical_values)):
                ax4.text(i - width/2, theo + abs(theo)*0.01, f'{theo:.2f}', 
                        ha='center', va='bottom', fontsize=9)
                ax4.text(i + width/2, emp + abs(emp)*0.01, f'{emp:.2f}', 
                        ha='center', va='bottom', fontsize=9)
            
            # Plot 5: Mean position over time (ensemble average)
            if self.n_walks > 1:
                ensemble_mean = np.mean(walks, axis=0)
                steps = range(len(ensemble_mean))
                ax5.plot(steps, ensemble_mean, 'g-', linewidth=2, label='Ensemble Mean')
                ax5.axhline(y=result.statistics['theoretical_mean'], color='red', 
                           linestyle='--', label=f'Theoretical Final Mean: {result.statistics["theoretical_mean"]:.2f}')
                ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax5.set_xlabel('Step Number')
                ax5.set_ylabel('Mean Position')
                ax5.set_title('Ensemble Average Position Over Time')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            
            # Plot 6: Variance over time
            if self.n_walks > 1:
                ensemble_var = np.var(walks, axis=0)
                steps = range(len(ensemble_var))
                ax6.plot(steps, ensemble_var, 'purple', linewidth=2, label='Empirical Variance')
                
                # Theoretical variance grows linearly with time
                theoretical_var_over_time = np.arange(len(ensemble_var)) * (self.step_size ** 2) * \
                                          (1 - (2 * self.step_probability - 1) ** 2)
                ax6.plot(steps, theoretical_var_over_time, 'r--', linewidth=2, label='Theoretical Variance')
                
                ax6.set_xlabel('Step Number')
                ax6.set_ylabel('Variance')
                ax6.set_title('Variance Growth Over Time')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'n_steps': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 100000,
                'description': 'Number of steps in the walk'
            },
            'step_size': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Size of each step'
            },
            'n_walks': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 1000,
                'description': 'Number of independent walks'
            },
            'step_probability': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probability of positive step (vs negative)'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        if self.n_steps < 1:
            errors.append("n_steps must be positive")
        if self.n_steps > 100000:
            errors.append("n_steps should not exceed 100,000 for performance reasons")
        if self.step_size <= 0:
            errors.append("step_size must be positive")
        if self.n_walks < 1:
            errors.append("n_walks must be positive")
        if self.n_walks > 1000:
            errors.append("n_walks should not exceed 1,000 for performance reasons")
        if not (0 <= self.step_probability <= 1):
            errors.append("step_probability must be between 0 and 1")
        return errors


class RandomWalk2D(BaseSimulation):
    """
    Two-dimensional random walk simulation on a square lattice.
    
    A 2D random walk extends the concept to two dimensions where at each step, 
    the walker moves in one of four cardinal directions (up, down, left, right) 
    with equal probability. This models diffusion processes, Brownian motion, 
    and various physical and biological phenomena in 2D space.
    
    Mathematical Background:
    -----------------------
    - At each step, walker moves by ±step_size in x OR y direction
    - Four possible moves: (±step_size, 0) or (0, ±step_size)
    - Each direction has probability 1/4
    - Position after n steps: (X_n, Y_n) = (X_0, Y_0) + Σ(i=1 to n) (S_x_i, S_y_i)
    - Expected displacement: E[X_n] = E[Y_n] = 0 (symmetric)
    - Variance per dimension: Var[X_n] = Var[Y_n] = n * step_size²
    - Expected distance²: E[R²_n] = E[X²_n + Y²_n] = 2n * step_size²
    - Expected distance: E[R_n] ≈ step_size * √(πn/2) for large n
    
    Physical Interpretation:
    -----------------------
    - Diffusion coefficient: D = step_size²/(2Δt) where Δt is time step
    - Mean squared displacement grows linearly: <R²> = 4Dt
    - Random walk is recurrent in 2D (will return to origin infinitely often)
    - Characteristic distance scales as √t
    
    Applications:
    ------------
    - Brownian motion of particles in 2D
    - Heat diffusion in thin materials
    - Animal foraging and migration patterns
    - Epidemiological spread models
    - Financial modeling (currency pairs)
    - Polymer chain configurations
    - Search algorithms and optimization
    - Percolation theory
    
    Simulation Features:
    -------------------
    - Equal probability movement in 4 cardinal directions
    - Multiple independent walks for ensemble statistics
    - Distance tracking from origin over time
    - Real-time path visualization with start/end markers
    - Statistical analysis of final positions and distances
    - Theoretical vs empirical comparisons
    - Ensemble averaging capabilities
    
    Parameters:
    -----------
    n_steps : int, default=1000
        Number of steps in each random walk
    step_size : float, default=1.0
        Size of each step in lattice units
    n_walks : int, default=1
        Number of independent walks to simulate
    random_seed : int, optional
        Seed for random number generator for reproducible results
    
    Attributes:
    -----------
    walks_x : list of numpy.ndarray
        X-coordinate data for each walk (including starting position at 0)
    walks_y : list of numpy.ndarray  
        Y-coordinate data for each walk (including starting position at 0)
    result : SimulationResult
        Complete simulation results including statistics and raw data
    
    Methods:
    --------
    configure(n_steps, step_size, n_walks) : bool
        Configure simulation parameters
    run(**kwargs) : SimulationResult
        Execute the 2D random walk simulation
    visualize(result=None, **kwargs) : None
        Create comprehensive visualizations including paths, distributions, and statistics
    validate_parameters() : List[str]
        Validate current parameters and return any errors
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Single 2D random walk
    >>> rw2d = RandomWalk2D(n_steps=2000, random_seed=42)
    >>> result = rw2d.run()
    >>> final_x, final_y = result.results['final_x_positions'][0], result.results['final_y_positions'][0]
    >>> print(f"Final position: ({final_x}, {final_y})")
    >>> print(f"Distance from origin: {result.results['final_distances'][0]:.2f}")
    
    >>> # Multiple walks for statistical analysis
    >>> rw2d_multi = RandomWalk2D(n_steps=1000, n_walks=100)
    >>> result = rw2d_multi.run()
    >>> print(f"Mean distance from origin: {result.results['mean_final_distance']:.2f}")
    >>> rw2d_multi.visualize()
    
    >>> # Large step size simulation
    >>> rw2d_large = RandomWalk2D(n_steps=500, step_size=2.0, n_walks=50)
    >>> result = rw2d_large.run()
    
    Visualization Outputs:
    ---------------------
    Single Walk:
    - 2D path plot with start (green) and end (red) markers
    - Distance from origin over time
    - X and Y coordinates vs time
    
    Multiple Walks:
    - Overlay of all walk paths
    - Final position scatter plot
    - Distance distribution histogram
    - X and Y coordinate distributions
    - Theoretical vs empirical statistics comparison
    - Mean squared displacement growth over time
    
    Performance Notes:
    -----------------
    - Memory usage scales as O(n_steps * n_walks)
    - Computation time scales as O(n_steps * n_walks)
    - Visualization automatically adjusts for large numbers of walks
    - Recommended limits: n_steps ≤ 50,000, n_walks ≤ 500
    
    Statistical Properties:
    ----------------------
    - Returns to origin infinitely often (recurrent in 2D)
    - Final positions follow 2D Gaussian distribution for large n
    - Distance distribution approaches Rayleigh distribution
    - Displacement variance grows linearly with time
    - No preferred direction (isotropic)
    
    References:
    -----------
    - Berg, H. C. (1993). Random Walks in Biology
    - Codling, E. A., et al. (2008). Random walk models in biology. J. R. Soc. Interface
    - Redner, S. (2001). A Guide to First-Passage Processes
    - Lawler, G. F. (1991). Intersections of Random Walks
    """

    
    def __init__(self, n_steps: int = 1000, step_size: float = 1.0, n_walks: int = 1, 
                 random_seed: Optional[int] = None):
        super().__init__("2D Random Walk")
        
        # Initialize parameters
        self.n_steps = n_steps
        self.step_size = step_size
        self.n_walks = n_walks
        
        # Store in parameters dict for base class
        self.parameters.update({
            'n_steps': n_steps,
            'step_size': step_size,
            'n_walks': n_walks,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state for storing walk data
        self.walks_x = None
        self.walks_y = None
        self.is_configured = True
    
    def configure(self, n_steps: int = 1000, step_size: float = 1.0, 
                 n_walks: int = 1) -> bool:
        """Configure 2D random walk parameters"""
        self.n_steps = n_steps
        self.step_size = step_size
        self.n_walks = n_walks
        
        # Update parameters dict
        self.parameters.update({
            'n_steps': n_steps,
            'step_size': step_size,
            'n_walks': n_walks
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute 2D random walk simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        walks_x = []
        walks_y = []
        final_distances = []
        max_distances = []
        final_positions = []
        
        for walk_idx in range(self.n_walks):
            # Generate random directions (0, 1, 2, 3 for right, up, left, down)
            directions = np.random.randint(0, 4, self.n_steps)
            
            # Convert to step vectors
            dx = np.zeros(self.n_steps)
            dy = np.zeros(self.n_steps)
            
            dx[directions == 0] = self.step_size   # right
            dx[directions == 2] = -self.step_size  # left
            dy[directions == 1] = self.step_size   # up
            dy[directions == 3] = -self.step_size  # down
            
            # Calculate cumulative positions (starting at origin)
            x_positions = np.zeros(self.n_steps + 1)
            y_positions = np.zeros(self.n_steps + 1)
            x_positions[1:] = np.cumsum(dx)
            y_positions[1:] = np.cumsum(dy)
            
            # Calculate distances from origin
            distances = np.sqrt(x_positions**2 + y_positions**2)
            
            walks_x.append(x_positions)
            walks_y.append(y_positions)
            final_distances.append(distances[-1])
            max_distances.append(np.max(distances))
            final_positions.append((x_positions[-1], y_positions[-1]))
        
        self.walks_x = walks_x
        self.walks_y = walks_y
        execution_time = time.time() - start_time
        
        # Calculate statistics
        final_distances = np.array(final_distances)
        max_distances = np.array(max_distances)
        final_x = [pos[0] for pos in final_positions]
        final_y = [pos[1] for pos in final_positions]
        
        # Theoretical calculations for 2D random walk
        theoretical_mean_distance = 0  # Expected displacement is 0
        theoretical_variance = self.n_steps * (self.step_size ** 2)  # For each dimension
        theoretical_std = np.sqrt(theoretical_variance)
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'final_distances': final_distances.tolist(),
                'mean_final_distance': np.mean(final_distances),
                'std_final_distance': np.std(final_distances),
                'max_distance_reached': np.max(max_distances),
                'mean_max_distance': np.mean(max_distances),
                'final_x_positions': final_x,
                'final_y_positions': final_y,
                'mean_final_x': np.mean(final_x),
                'mean_final_y': np.mean(final_y)
            },
            statistics={
                'theoretical_mean_x': 0,
                'theoretical_mean_y': 0,
                'empirical_mean_x': np.mean(final_x),
                'empirical_mean_y': np.mean(final_y),
                'theoretical_std_per_dimension': theoretical_std,
                'empirical_std_x': np.std(final_x),
                'empirical_std_y': np.std(final_y),
                'theoretical_mean_distance_squared': theoretical_variance * 2,  # Sum of variances for x and y
                'empirical_mean_distance_squared': np.mean(final_distances**2)
            },
            raw_data={'x_walks': np.array(walks_x), 'y_walks': np.array(walks_y)},
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize 2D random walk simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        walks_x = result.raw_data['x_walks']
        walks_y = result.raw_data['y_walks']
        
        if self.n_walks == 1:
            # Single walk visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: The 2D walk path
            x_path = walks_x[0]
            y_path = walks_y[0]
            
            ax1.plot(x_path, y_path, 'b-', linewidth=1, alpha=0.7, label='Path')
            ax1.scatter(x_path[0], y_path[0], color='green', s=100, marker='o', label='Start', zorder=5)
            ax1.scatter(x_path[-1], y_path[-1], color='red', s=100, marker='s', label='End', zorder=5)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title(f'2D Random Walk Path ({self.n_steps} steps)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Plot 2: Distance from origin over time
            distances = np.sqrt(x_path**2 + y_path**2)
            steps = range(len(distances))
            ax2.plot(steps, distances, 'purple', linewidth=2)
            ax2.set_xlabel('Step Number')
            ax2.set_ylabel('Distance from Origin')
            ax2.set_title('Distance from Origin Over Time')
            ax2.grid(True, alpha=0.3)
            
            # Add final distance text
            final_dist = distances[-1]
            max_dist = np.max(distances)
            ax2.text(0.02, 0.98, f'Final Distance: {final_dist:.2f}\nMax Distance: {max_dist:.2f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Plot 3: X position over time
            ax3.plot(steps, x_path, 'r-', linewidth=1.5, label='X position')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Step Number')
            ax3.set_ylabel('X Position')
            ax3.set_title('X Coordinate Over Time')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Y position over time
            ax4.plot(steps, y_path, 'g-', linewidth=1.5, label='Y position')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Step Number')
            ax4.set_ylabel('Y Position')
            ax4.set_title('Y Coordinate Over Time')
            ax4.grid(True, alpha=0.3)
            
        else:
            # Multiple walks visualization
            fig = plt.figure(figsize=(18, 12))
            
            # Create 2x3 subplot layout
            ax1 = plt.subplot(2, 3, 1)
            ax2 = plt.subplot(2, 3, 2)
            ax3 = plt.subplot(2, 3, 3)
            ax4 = plt.subplot(2, 3, 4)
            ax5 = plt.subplot(2, 3, 5)
            ax6 = plt.subplot(2, 3, 6)
            
            # Plot 1: All walk paths
            for i in range(self.n_walks):
                alpha = min(0.6, 50.0 / self.n_walks)  # Fade out for many walks
                ax1.plot(walks_x[i], walks_y[i], linewidth=1, alpha=alpha)
                # Mark start and end points for first few walks
                if i < 5:
                    ax1.scatter(walks_x[i][0], walks_y[i][0], color='green', s=20, alpha=0.7)
                    ax1.scatter(walks_x[i][-1], walks_y[i][-1], color='red', s=20, alpha=0.7)
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title(f'{self.n_walks} 2D Random Walk Paths')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Plot 2: Final position scatter
            final_x = result.results['final_x_positions']
            final_y = result.results['final_y_positions']
            ax2.scatter(final_x, final_y, alpha=0.6, s=30)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Final X Position')
            ax2.set_ylabel('Final Y Position')
            ax2.set_title('Distribution of Final Positions')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            
            # Plot 3: Final distance histogram
            final_distances = result.results['final_distances']
            bins = min(30, max(10, self.n_walks//5))
            ax3.hist(final_distances, bins=bins, alpha=0.7, edgecolor='black', density=True)
            ax3.axvline(x=np.mean(final_distances), color='green', linestyle='-', linewidth=2,
                       label=f'Mean: {np.mean(final_distances):.2f}')
            ax3.set_xlabel('Final Distance from Origin')
            ax3.set_ylabel('Density')
            ax3.set_title('Distribution of Final Distances')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: X and Y coordinate distributions
            ax4.hist(final_x, bins=bins//2, alpha=0.5, label='X coordinates', density=True)
            ax4.hist(final_y, bins=bins//2, alpha=0.5, label='Y coordinates', density=True)
            ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Final Coordinate Value')
            ax4.set_ylabel('Density')
            ax4.set_title('Distribution of Final X and Y Coordinates')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Statistics comparison
            theo_std = result.statistics['theoretical_std_per_dimension']
            emp_std_x = result.statistics['empirical_std_x']
            emp_std_y = result.statistics['empirical_std_y']
            
            categories = ['Theoretical\n(per dim)', 'Empirical X', 'Empirical Y']
            values = [theo_std, emp_std_x, emp_std_y]
            colors = ['blue', 'orange', 'green']
            
            bars = ax5.bar(categories, values, alpha=0.7, color=colors)
            ax5.set_ylabel('Standard Deviation')
            ax5.set_title('Theoretical vs Empirical Standard Deviations')
            ax5.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # Plot 6: Mean distance squared over time (ensemble average)
            if self.n_walks > 1:
                mean_dist_sq_over_time = []
                for step in range(self.n_steps + 1):
                    distances_at_step = []
                    for walk_idx in range(self.n_walks):
                        x = walks_x[walk_idx][step]
                        y = walks_y[walk_idx][step]
                        dist_sq = x**2 + y**2
                        distances_at_step.append(dist_sq)
                    mean_dist_sq_over_time.append(np.mean(distances_at_step))
                
                steps = range(len(mean_dist_sq_over_time))
                ax6.plot(steps, mean_dist_sq_over_time, 'purple', linewidth=2, label='Empirical')
                
                # Theoretical: E[R²] = 2σ²t for 2D random walk
                theoretical_dist_sq = np.arange(len(mean_dist_sq_over_time)) * 2 * (self.step_size ** 2)
                ax6.plot(steps, theoretical_dist_sq, 'r--', linewidth=2, label='Theoretical')
                
                ax6.set_xlabel('Step Number')
                ax6.set_ylabel('Mean Distance² from Origin')
                ax6.set_title('Mean Squared Distance Growth Over Time')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'n_steps': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 50000,
                'description': 'Number of steps in the walk'
            },
            'step_size': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Size of each step'
            },
            'n_walks': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 500,
                'description': 'Number of independent walks'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        if self.n_steps < 1:
            errors.append("n_steps must be positive")
        if self.n_steps > 50000:
            errors.append("n_steps should not exceed 50,000 for performance reasons")
        if self.step_size <= 0:
            errors.append("step_size must be positive")
        if self.n_walks < 1:
            errors.append("n_walks must be positive")
        if self.n_walks > 500:
            errors.append("n_walks should not exceed 500 for performance reasons")
        return errors
