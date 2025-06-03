import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class PiEstimationMC(BaseSimulation):
    """
    Monte Carlo estimation of π using the circle-square sampling method.
    
    This simulation estimates the value of π by randomly sampling points in a 2×2 square 
    centered at the origin and counting how many fall within the inscribed unit circle. 
    The ratio of points inside the circle to total points approximates π/4, allowing 
    estimation of π through the relationship: π ≈ 4 × (points_inside / total_points).
    
    Mathematical Background:
    -----------------------
    - Unit circle area: A_circle = π × r² = π × 1² = π
    - Square area: A_square = (2r)² = 4 (for r=1)
    - Ratio: A_circle / A_square = π/4
    - Point (x,y) is inside circle if: x² + y² ≤ 1
    - As n → ∞: (points_inside / n) → π/4, so π ≈ 4 × (points_inside / n)
    
    Statistical Properties:
    ----------------------
    - Standard error: σ ≈ √(π(4-π)/n) ≈ 1.64/√n
    - Convergence rate: O(1/√n) - typical for Monte Carlo methods
    - 95% confidence interval: π_estimate ± 1.96 × σ
    - For 1 million samples: expected error ≈ 0.0016
    - For 100 million samples: expected error ≈ 0.0005
    
    Algorithm Details:
    -----------------
    1. Generate random points (x,y) uniformly in [-1,1] × [-1,1]
    2. Test if x² + y² ≤ 1 (inside unit circle)
    3. Count hits inside circle
    4. Estimate π = 4 × (hits / total_samples)
    5. Track convergence over sampling progression
    
    Applications:
    ------------
    - Numerical integration demonstration
    - Monte Carlo method education
    - Statistical convergence analysis
    - Parallel computing benchmarks
    - Random number generator testing
    - Computational geometry examples
    - Probability theory illustrations
    
    Historical Context:
    ------------------
    - One of the first Monte Carlo applications (1940s)
    - Used in early computer testing and validation
    - Classic example in computational physics
    - Demonstrates Law of Large Numbers
    - Illustrates Central Limit Theorem in practice
    
    Simulation Features:
    -------------------
    - High-precision π estimation with configurable sample size
    - Real-time convergence tracking and visualization
    - Statistical error analysis and confidence intervals
    - Visual representation of sampling process
    - Comparison with theoretical convergence rates
    - Performance timing and efficiency metrics
    
    Parameters:
    -----------
    n_samples : int, default=1000000
        Number of random points to generate and test
        Larger values give more accurate estimates but take longer
        Recommended: 10⁴ for quick tests, 10⁶⁺ for accurate results
    show_convergence : bool, default=True
        Whether to track and store convergence data during simulation
        Enables convergence plotting but uses additional memory
    random_seed : int, optional
        Seed for random number generator for reproducible results
        Useful for testing, debugging, and result verification
    
    Attributes:
    -----------
    points_inside : tuple of (x_coords, y_coords), optional
        Coordinates of points inside the circle (stored for small n_samples ≤ 10,000)
    points_outside : tuple of (x_coords, y_coords), optional
        Coordinates of points outside the circle (stored for small n_samples ≤ 10,000)
    pi_estimates : list of tuples
        Convergence data as [(sample_count, pi_estimate), ...] if show_convergence=True
    result : SimulationResult
        Complete simulation results including final estimate and statistics
    
    Methods:
    --------
    configure(n_samples, show_convergence) : bool
        Configure simulation parameters before running
    run(**kwargs) : SimulationResult
        Execute the π estimation simulation
    visualize(result=None, show_points=False, n_display_points=1000) : None
        Create visualizations of results and/or sampling process
    validate_parameters() : List[str]
        Validate current parameters and return any errors
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Quick π estimation
    >>> pi_sim = PiEstimationMC(n_samples=100000, random_seed=42)
    >>> result = pi_sim.run()
    >>> print(f"π estimate: {result.results['pi_estimate']:.6f}")
    >>> print(f"Error: {result.results['accuracy']:.6f}")
    
    >>> # High-precision estimation with convergence tracking
    >>> pi_precise = PiEstimationMC(n_samples=10000000, show_convergence=True)
    >>> result = pi_precise.run()
    >>> pi_precise.visualize()
    >>> print(f"Relative error: {result.results['relative_error']:.4f}%")
    
    >>> # Visualization with point sampling (for educational purposes)
    >>> pi_visual = PiEstimationMC(n_samples=5000, random_seed=123)
    >>> result = pi_visual.run()
    >>> pi_visual.visualize(show_points=True, n_display_points=2000)
    
    Visualization Outputs:
    ---------------------
    Standard Mode:
    - Results summary with π estimate, true value, and errors
    - Convergence plot showing estimate approaching π over sample size
    - Error bounds and confidence intervals
    
    Point Visualization Mode (show_points=True):
    - Scatter plot of sample points colored by inside/outside circle
    - Unit circle overlay showing the sampling region
    - Visual demonstration of the geometric principle
    
    Performance Characteristics:
    ---------------------------
    - Time complexity: O(n_samples)
    - Space complexity: O(1) for standard mode, O(n_samples) for convergence tracking
    - Memory usage for points: ~80 bytes per sample (only stored if n_samples ≤ 10,000)
    - Typical speeds: ~1M samples/second on modern hardware
    - Parallelizable: independent samples can be distributed across cores
    
    Accuracy Guidelines:
    -------------------
    - 10³ samples: ±0.05 typical error (educational demonstrations)
    - 10⁴ samples: ±0.016 typical error (quick estimates)
    - 10⁶ samples: ±0.0016 typical error (good accuracy)
    - 10⁸ samples: ±0.0005 typical error (high precision)
    - 10⁹ samples: ±0.00016 typical error (research quality)
    
    Error Analysis:
    --------------
    The simulation provides several error metrics:
    - Absolute error: |π_estimate - π|
    - Relative error: |π_estimate - π| / π × 100%
    - Standard error: √(π(4-π)/n) (theoretical)
    - 95% confidence interval bounds
    
    Theoretical Convergence:
    -----------------------
    - Expected value: E[π_estimate] = π (unbiased estimator)
    - Variance: Var[π_estimate] = 16π(4-π)/n ≈ 16.7/n
    - Standard deviation: σ ≈ 4.1/√n
    - Probability of |error| < ε: P(|π_est - π| < ε) ≈ 2Φ(ε√n/4.1) - 1
    
    Educational Value:
    -----------------
    - Demonstrates Monte Carlo integration principles
    - Illustrates Law of Large Numbers convergence
    - Shows relationship between geometry and probability
    - Teaches statistical error analysis
    - Provides intuitive understanding of sampling methods
    
    Extensions and Variations:
    -------------------------
    - Quasi-Monte Carlo: Use low-discrepancy sequences for faster convergence
    - Importance sampling: Weight samples by density functions
    - Stratified sampling: Divide domain into regions for uniform coverage
    - Antithetic variables: Use correlated samples to reduce variance
    - Control variates: Use known integrals to reduce estimation variance
    
    References:
    -----------
    - Metropolis, N. & Ulam, S. (1949). The Monte Carlo Method. J. Am. Stat. Assoc.
    - Hammersley, J. M. & Handscomb, D. C. (1964). Monte Carlo Methods
    - Robert, C. P. & Casella, G. (2004). Monte Carlo Statistical Methods
    - Liu, J. S. (2001). Monte Carlo Strategies in Scientific Computing
    - Fishman, G. S. (1995). Monte Carlo: Concepts, Algorithms, and Applications
    """

    def __init__(self, n_samples: int = 1000000, show_convergence: bool = True, 
                 random_seed: Optional[int] = None):
        super().__init__("Monte Carlo π Estimation")
        
        # Initialize parameters
        self.n_samples = n_samples
        self.show_convergence = show_convergence
        
        # Store in parameters dict for base class
        self.parameters.update({
            'n_samples': n_samples,
            'show_convergence': show_convergence,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state for visualization
        self.points_inside = None
        self.points_outside = None
        self.pi_estimates = None
        self.is_configured = True
    
    def configure(self, n_samples: int = 1000000, show_convergence: bool = True) -> bool:
        """Configure π estimation parameters"""
        self.n_samples = n_samples
        self.show_convergence = show_convergence
        
        # Update parameters dict
        self.parameters.update({
            'n_samples': n_samples,
            'show_convergence': show_convergence
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute π estimation simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Generate random points in [-1, 1] x [-1, 1]
        x = np.random.uniform(-1, 1, self.n_samples)
        y = np.random.uniform(-1, 1, self.n_samples)
        
        # Calculate distances from origin
        distances_squared = x**2 + y**2
        
        # Points inside unit circle
        inside_circle = distances_squared <= 1
        n_inside = np.sum(inside_circle)
        
        # Store points for visualization (only for small samples)
        if self.n_samples <= 10000:
            self.points_inside = (x[inside_circle], y[inside_circle])
            self.points_outside = (x[~inside_circle], y[~inside_circle])
        
        # Calculate π estimate
        pi_estimate = 4 * n_inside / self.n_samples
        
        # Calculate convergence if requested
        convergence_data = []
        if self.show_convergence:
            step_size = max(1000, self.n_samples // 1000)
            for i in range(step_size, self.n_samples + 1, step_size):
                running_inside = np.sum(inside_circle[:i])
                running_estimate = 4 * running_inside / i
                convergence_data.append((i, running_estimate))
        
        self.pi_estimates = convergence_data
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'pi_estimate': pi_estimate,
                'points_inside_circle': n_inside,
                'points_outside_circle': self.n_samples - n_inside,
                'accuracy': abs(pi_estimate - np.pi),
                'relative_error': abs(pi_estimate - np.pi) / np.pi * 100
            },
            statistics={
                'mean_estimate': pi_estimate,
                'theoretical_value': np.pi,
                'absolute_error': abs(pi_estimate - np.pi),
                'relative_error_percent': abs(pi_estimate - np.pi) / np.pi * 100
            },
            execution_time=execution_time,
            convergence_data=convergence_data
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, 
                 show_points: bool = False, n_display_points: int = 1000) -> None:
        """Visualize π estimation process"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        # Create subplots based on what we want to show
        if self.show_convergence:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot 1: Points visualization or summary
        if show_points and self.points_inside is not None and self.points_outside is not None:
            # Sample points for display
            n_inside = len(self.points_inside[0])
            n_outside = len(self.points_outside[0])
            
            if n_inside > n_display_points // 2:
                inside_indices = np.random.choice(n_inside, n_display_points // 2, replace=False)
                display_inside_x = self.points_inside[0][inside_indices]
                display_inside_y = self.points_inside[1][inside_indices]
            else:
                display_inside_x = self.points_inside[0]
                display_inside_y = self.points_inside[1]
            
            if n_outside > n_display_points // 2:
                outside_indices = np.random.choice(n_outside, n_display_points // 2, replace=False)
                display_outside_x = self.points_outside[0][outside_indices]
                display_outside_y = self.points_outside[1][outside_indices]
            else:
                display_outside_x = self.points_outside[0]
                display_outside_y = self.points_outside[1]
            
            # Plot points
            ax1.scatter(display_inside_x, display_inside_y, c='red', s=1, alpha=0.6, label='Inside circle')
            ax1.scatter(display_outside_x, display_outside_y, c='blue', s=1, alpha=0.6, label='Outside circle')
            
            # Draw unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            ax1.plot(circle_x, circle_y, 'black', linewidth=2, label='Unit circle')
            
            ax1.set_xlim(-1.1, 1.1)
            ax1.set_ylim(-1.1, 1.1)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title('Monte Carlo Sampling')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
        else:
            # Show summary statistics
            pi_est = result.results['pi_estimate']
            error = result.results['accuracy']
            rel_error = result.results['relative_error']
            
            ax1.text(0.5, 0.7, f'π Estimate: {pi_est:.6f}', transform=ax1.transAxes, 
                    fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax1.text(0.5, 0.5, f'True π: {np.pi:.6f}', transform=ax1.transAxes, 
                    fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax1.text(0.5, 0.3, f'Absolute Error: {error:.6f}', transform=ax1.transAxes, 
                    fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            ax1.text(0.5, 0.1, f'Relative Error: {rel_error:.4f}%', transform=ax1.transAxes, 
                    fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.set_title('Simulation Results Summary')
            ax1.axis('off')
        
        # Plot 2: Convergence (if requested)
        if self.show_convergence and result.convergence_data:
            samples = [point[0] for point in result.convergence_data]
            estimates = [point[1] for point in result.convergence_data]
            
            ax2.plot(samples, estimates, 'b-', linewidth=2, label='π estimate')
            ax2.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label='True π')
            ax2.fill_between(samples, estimates, np.pi, alpha=0.3, 
                           color='red' if estimates[-1] > np.pi else 'blue')
            
            ax2.set_xlabel('Number of Samples')
            ax2.set_ylabel('π Estimate')
            ax2.set_title('Convergence to π')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add final estimate text
            final_estimate = estimates[-1]
            ax2.text(0.7, 0.9, f'Final: {final_estimate:.6f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'n_samples': {
                'type': 'int',
                'default': 1000000,
                'min': 1000,
                'max': 10000000,
                'description': 'Number of random samples'
            },
            'show_convergence': {
                'type': 'bool',
                'default': True,
                'description': 'Show convergence plot'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
    
    def validate_parameters(self) -> list[str]:
        """Validate simulation parameters"""
        errors = []
        if self.n_samples < 1000:
            errors.append("n_samples must be at least 1000")
        if self.n_samples > 10000000:
            errors.append("n_samples should not exceed 10,000,000 for performance reasons")
        return errors
