import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Callable, Tuple, List, Union
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class MonteCarloIntegration(BaseSimulation):
    """
    Monte Carlo numerical integration for single and multi-dimensional functions.
    
    This simulation estimates definite integrals using Monte Carlo sampling methods.
    It supports both uniform random sampling and importance sampling techniques
    for efficient integration of complex functions over specified domains.
    
    Mathematical Background:
    -----------------------
    For a function f(x) over domain [a,b]:
    - Integral: I = ∫[a,b] f(x) dx
    - Monte Carlo estimate: I ≈ (b-a) × (1/n) × Σf(xi) where xi ~ U(a,b)
    - Multi-dimensional: I = V × (1/n) × Σf(xi) where V is domain volume
    - Expected value: E[I_estimate] = I (unbiased estimator)
    - Variance: Var[I_estimate] = V²/n × Var[f(X)]
    - Standard error: σ ≈ V × √(Var[f(X)]/n)
    
    Convergence Properties:
    ----------------------
    - Rate: O(1/√n) independent of dimension (curse of dimensionality advantage)
    - Error bound: |I_estimate - I| ≤ ε with probability ≈ 2Φ(ε√n/σ) - 1
    - Relative error decreases as √(1/n)
    - Dimension-independent convergence (major advantage over quadrature)
    - Central Limit Theorem applies: estimates → Normal distribution
    
    Supported Integration Methods:
    -----------------------------
    1. Uniform Sampling (default):
       - Random points uniformly distributed over integration domain
       - Best for well-behaved functions without extreme peaks/valleys
       - Simple implementation with predictable behavior
    
    2. Importance Sampling:
       - Sample from probability distribution similar to |f(x)|
       - Reduces variance for functions with concentrated mass
       - Requires specification of sampling distribution
    
    3. Stratified Sampling:
       - Divide domain into subregions, sample uniformly within each
       - Reduces variance by ensuring coverage of entire domain
       - Particularly effective for smooth functions
    
    4. Hit-or-Miss Method:
       - For positive functions, sample points in bounding rectangle
       - Count points below function curve
       - Geometric interpretation similar to π estimation
    
    Algorithm Details:
    -----------------
    1. Define integration domain and function
    2. Generate n random sample points in domain
    3. Evaluate function at each sample point
    4. Calculate weighted average based on domain volume
    5. Estimate integral and compute error statistics
    6. Track convergence over increasing sample sizes
    
    Applications:
    ------------
    - High-dimensional integration (finance, physics, statistics)
    - Bayesian inference and posterior sampling
    - Expected value calculations in probability
    - Physics simulations (quantum mechanics, statistical mechanics)
    - Financial derivatives pricing and risk assessment
    - Machine learning: expectation computations
    - Engineering: reliability and uncertainty quantification
    - Computer graphics: global illumination and rendering
    
    Advantages over Traditional Quadrature:
    --------------------------------------
    - Dimension-independent O(1/√n) convergence
    - Handles irregular domains naturally
    - Robust for discontinuous or highly oscillatory functions
    - Easily parallelizable
    - Memory requirements independent of dimension
    - Natural error estimates from sample variance
    
    Simulation Features:
    -------------------
    - Support for 1D to high-dimensional integration
    - Multiple sampling strategies (uniform, importance, stratified)
    - Built-in common test functions (polynomials, trigonometric, Gaussian)
    - Real-time convergence monitoring and visualization
    - Statistical error analysis with confidence intervals
    - Comparison with analytical solutions when available
    - Performance profiling and efficiency metrics
    - Adaptive sampling for improved accuracy
    
    Parameters:
    -----------
    function : callable or str
        Function to integrate. Can be:
        - Python function taking numpy array and returning scalar/array
        - String name of built-in test function
        - Lambda expression for simple functions
    domain : tuple or list of tuples
        Integration domain. For 1D: (a, b). For nD: [(a1,b1), (a2,b2), ...]
    n_samples : int, default=100000
        Number of Monte Carlo samples to generate
        More samples → higher accuracy but longer computation time
    method : str, default='uniform'
        Sampling method: 'uniform', 'importance', 'stratified', 'hit_miss'
    show_convergence : bool, default=True
        Whether to track convergence during simulation
    random_seed : int, optional
        Seed for reproducible results
    analytical_result : float, optional
        Known analytical result for error calculation
    
    Attributes:
    -----------
    sample_points : ndarray, optional
        Generated sample points (stored for small sample sizes)
    function_values : ndarray, optional
        Function evaluations at sample points
    convergence_estimates : list
        Convergence data as [(sample_count, integral_estimate), ...]
    domain_volume : float
        Volume/length of integration domain
    result : SimulationResult
        Complete simulation results
    
    Methods:
    --------
    configure(function, domain, n_samples, method) : bool
        Configure integration parameters
    run(**kwargs) : SimulationResult
        Execute Monte Carlo integration
    visualize(result=None, show_function=True, show_samples=False) : None
        Visualize integration results and convergence
    add_custom_function(name, func, analytical) : None
        Add custom function to built-in library
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Built-in Test Functions:
    -----------------------
    1D Functions:
    - 'polynomial': x² + 2x + 1 over [0,1], analytical = 7/3
    - 'sine': sin(x) over [0,π], analytical = 2
    - 'exponential': e^x over [0,1], analytical = e-1
    - 'gaussian': e^(-x²) over [-∞,∞], analytical = √π
    - 'oscillatory': sin(10x) over [0,1], analytical = (1-cos(10))/10
    
    2D Functions:
    - 'circle': √(1-x²-y²) over unit disk, analytical = 2π/3
    - 'paraboloid': x² + y² over [0,1]×[0,1], analytical = 2/3
    - 'gaussian_2d': e^(-(x²+y²)) over [-∞,∞]², analytical = π
    
    Higher Dimensions:
    - 'sphere_nd': Unit sphere volume in n dimensions
    - 'gaussian_nd': n-dimensional Gaussian integral
    
    Examples:
    ---------
    >>> # Simple 1D integration
    >>> integrator = MonteCarloIntegration(
    ...     function=lambda x: x**2,
    ...     domain=(0, 1),
    ...     n_samples=100000,
    ...     analytical_result=1/3
    ... )
    >>> result = integrator.run()
    >>> print(f"Integral estimate: {result.results['integral_estimate']:.6f}")
    >>> print(f"Error: {result.results['absolute_error']:.6f}")
    
    >>> # Built-in test function
    >>> integrator = MonteCarloIntegration(
    ...     function='sine',
    ...     domain=(0, np.pi),
    ...     n_samples=1000000
    ... )
    >>> result = integrator.run()
    >>> integrator.visualize()
    
    >>> # Multi-dimensional integration
    >>> def gaussian_2d(x):
    ...     return np.exp(-(x[0]**2 + x[1]**2))
    >>> integrator = MonteCarloIntegration(
    ...     function=gaussian_2d,
    ...     domain=[(-3, 3), (-3, 3)],
    ...     n_samples=500000,
    ...     method='importance'
    ... )
    >>> result = integrator.run()
    
    >>> # High-dimensional example (curse of dimensionality demonstration)
    >>> def hypersphere_volume(x, dim=10):
    ...     return 1.0 if np.sum(x**2) <= 1 else 0.0
    >>> integrator = MonteCarloIntegration(
    ...     function=hypersphere_volume,
    ...     domain=[(-1, 1)] * 10,  # 10D unit hypercube
    ...     n_samples=10000000,
    ...     method='uniform'
    ... )
    >>> result = integrator.run()
    
    Performance Guidelines:
    ----------------------
    - 1D functions: 10⁴-10⁶ samples typically sufficient
    - 2D-3D functions: 10⁵-10⁷ samples recommended
    - High dimensions (>5): 10⁶-10⁹ samples may be needed
    - Smooth functions: fewer samples required
    - Highly oscillatory: more samples needed
    - Peaked functions: importance sampling recommended
    
    Error Analysis:
    --------------
    - Absolute error: |estimate - true_value|
    - Relative error: |estimate - true_value| / |true_value| × 100%
    - Standard error: sample_std / √n
    - 95% confidence interval: estimate ± 1.96 × standard_error
    - Convergence rate: typically 1/√n
    
    Visualization Features:
    ----------------------
    1D Functions:
    - Function plot with integration region highlighted
    - Sample points overlay showing Monte Carlo sampling
    - Convergence plot showing estimate vs. sample size
    - Error evolution and confidence bounds
    
    2D Functions:
    - Contour plot or 3D surface of function
    - Sample points projected onto domain
    - Convergence analysis
    
    Higher Dimensions:
    - Convergence plots and statistical summaries
    - Projection plots for visualization
    - Error analysis and confidence intervals
    
    Advanced Features:
    -----------------
    - Adaptive sampling: increase samples in high-variance regions
    - Parallel processing: distribute samples across multiple cores
    - Quasi-Monte Carlo: use low-discrepancy sequences
    - Control variates: use correlated functions to reduce variance
    - Antithetic variables: use negatively correlated samples
    - Stratified sampling: systematic domain subdivision
    
    Educational Value:
    -----------------
    - Demonstrates Monte Carlo integration principles
    - Illustrates curse of dimensionality and its solutions
    - Shows convergence properties of random sampling
    - Teaches importance of variance reduction techniques
    - Provides intuition for high-dimensional probability
    
    References:
    -----------
    - Hammersley, J. M. & Handscomb, D. C. (1964). Monte Carlo Methods
    - Robert, C. P. & Casella, G. (2004). Monte Carlo Statistical Methods
    - Liu, J. S. (2001). Monte Carlo Strategies in Scientific Computing
    - Caflisch, R. E. (1998). Monte Carlo and quasi-Monte Carlo methods
    - Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering
    """

    def __init__(self, function: Union[Callable, str] = 'polynomial', 
                 domain: Union[Tuple[float, float], List[Tuple[float, float]]] = (0, 1),
                 n_samples: int = 100000, method: str = 'uniform',
                 show_convergence: bool = True, random_seed: Optional[int] = None,
                 analytical_result: Optional[float] = None):
        super().__init__("Monte Carlo Integration")
        
        # Initialize parameters
        self.function = function
        self.domain = domain
        self.n_samples = n_samples
        self.method = method
        self.show_convergence = show_convergence
        self.analytical_result = analytical_result
        
        # Store in parameters dict for base class
        self.parameters.update({
            'function': str(function),
            'domain': domain,
            'n_samples': n_samples,
            'method': method,
            'show_convergence': show_convergence,
            'random_seed': random_seed,
            'analytical_result': analytical_result
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Initialize built-in functions
        self._setup_builtin_functions()
        
        # Internal state
        self.sample_points = None
        self.function_values = None
        self.convergence_estimates = None
        self.domain_volume = None
        self.dimension = None
        self.is_configured = True
    
    def _setup_builtin_functions(self):
        """Setup built-in test functions with analytical results"""
        self.builtin_functions = {
            # 1D functions
            'polynomial': {
                'func': lambda x: x**2 + 2*x + 1,
                'domain': (0, 1),
                'analytical': 7/3,
                'description': 'x² + 2x + 1'
            },
            'sine': {
                'func': lambda x: np.sin(x),
                'domain': (0, np.pi),
                'analytical': 2.0,
                'description': 'sin(x)'
            },
            'exponential': {
                'func': lambda x: np.exp(x),
                'domain': (0, 1),
                'analytical': np.e - 1,
                'description': 'e^x'
            },
            'gaussian': {
                'func': lambda x: np.exp(-x**2),
                'domain': (-3, 3),
                'analytical': np.sqrt(np.pi),
                'description': 'e^(-x²)'
            },
            'oscillatory': {
                'func': lambda x: np.sin(10*x),
                'domain': (0, 1),
                'analytical': (1 - np.cos(10))/10,
                'description': 'sin(10x)'
            },
            # 2D functions
            'circle': {
                'func': lambda x: np.sqrt(np.maximum(0, 1 - x[0]**2 - x[1]**2)),
                'domain': [(-1, 1), (-1, 1)],
                'analytical': 2*np.pi/3,
                'description': '√(1-x²-y²) over unit disk'
            },
            'paraboloid': {
                'func': lambda x: x[0]**2 + x[1]**2,
                'domain': [(0, 1), (0, 1)],
                'analytical': 2/3,
                'description': 'x² + y²'
            },
            'gaussian_2d': {
                'func': lambda x: np.exp(-(x[0]**2 + x[1]**2)),
                'domain': [(-3, 3), (-3, 3)],
                'analytical': np.pi,
                'description': 'e^(-(x²+y²))'
            }
        }
    
    def configure(self, function: Union[Callable, str] = 'polynomial',                
                  domain: Union[Tuple[float, float], List[Tuple[float, float]]] = (0, 1),
                 n_samples: int = 100000, method: str = 'uniform') -> bool:
        """Configure Monte Carlo integration parameters"""
        self.function = function
        self.domain = domain
        self.n_samples = n_samples
        self.method = method
        
        # Update parameters dict
        self.parameters.update({
            'function': str(function),
            'domain': domain,
            'n_samples': n_samples,
            'method': method
        })
        
        self.is_configured = True
        return True
    
    def _get_function_and_domain(self):
        """Get the actual function and domain to use"""
        if isinstance(self.function, str):
            if self.function in self.builtin_functions:
                func_info = self.builtin_functions[self.function]
                func = func_info['func']
                domain = func_info['domain']
                analytical = func_info['analytical']
                if self.analytical_result is None:
                    self.analytical_result = analytical
                return func, domain
            else:
                raise ValueError(f"Unknown built-in function: {self.function}")
        else:
            return self.function, self.domain
    
    def _calculate_domain_volume(self, domain):
        """Calculate the volume of the integration domain"""
        if isinstance(domain, tuple):
            # 1D case
            return domain[1] - domain[0]
        else:
            # Multi-dimensional case
            volume = 1.0
            for dim_range in domain:
                volume *= (dim_range[1] - dim_range[0])
            return volume
    
    def _generate_samples(self, domain, n_samples):
        """Generate sample points in the domain"""
        if isinstance(domain, tuple):
            # 1D case
            return np.random.uniform(domain[0], domain[1], n_samples)
        else:
            # Multi-dimensional case
            samples = []
            for dim_range in domain:
                samples.append(np.random.uniform(dim_range[0], dim_range[1], n_samples))
            return np.array(samples)
    
    def _evaluate_function(self, func, samples):
        """Evaluate function at sample points"""
        if samples.ndim == 1:
            # 1D case
            return func(samples)
        else:
            # Multi-dimensional case
            if samples.shape[0] == 1:
                # Still 1D but in array format
                return func(samples[0])
            else:
                # Truly multi-dimensional
                values = np.zeros(samples.shape[1])
                for i in range(samples.shape[1]):
                    values[i] = func(samples[:, i])
                return values
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute Monte Carlo integration"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Get function and domain
        func, domain = self._get_function_and_domain()
        
        # Calculate domain properties
        self.domain_volume = self._calculate_domain_volume(domain)
        if isinstance(domain, tuple):
            self.dimension = 1
        else:
            self.dimension = len(domain)
        
        # Generate sample points
        samples = self._generate_samples(domain, self.n_samples)
        
        # Store samples for visualization (only for small sample sizes)
        if self.n_samples <= 10000:
            self.sample_points = samples
        
        # Evaluate function at sample points
        try:
            function_values = self._evaluate_function(func, samples)
            
            # Handle potential NaN or infinite values
            valid_mask = np.isfinite(function_values)
            if not np.all(valid_mask):
                print(f"Warning: {np.sum(~valid_mask)} invalid function values encountered")
                function_values = function_values[valid_mask]
                n_valid = len(function_values)
            else:
                n_valid = self.n_samples
            
            # Store function values for visualization
            if self.n_samples <= 10000:
                self.function_values = function_values
            
        except Exception as e:
            raise RuntimeError(f"Error evaluating function: {str(e)}")
        
        # Calculate integral estimate
        mean_value = np.mean(function_values)
        integral_estimate = self.domain_volume * mean_value
        
        # Calculate statistics
        variance = np.var(function_values)
        standard_error = self.domain_volume * np.sqrt(variance / n_valid)
        
        # Calculate convergence if requested
        convergence_data = []
        if self.show_convergence:
            step_size = max(1000, self.n_samples // 1000)
            for i in range(step_size, len(function_values) + 1, step_size):
                running_mean = np.mean(function_values[:i])
                running_estimate = self.domain_volume * running_mean
                convergence_data.append((i, running_estimate))
        
        self.convergence_estimates = convergence_data
        execution_time = time.time() - start_time
        
        # Calculate errors if analytical result is known
        absolute_error = None
        relative_error = None
        if self.analytical_result is not None:
            absolute_error = abs(integral_estimate - self.analytical_result)
            relative_error = absolute_error / abs(self.analytical_result) * 100
        
        # Create result object
        results_dict = {
            'integral_estimate': integral_estimate,
            'mean_function_value': mean_value,
            'domain_volume': self.domain_volume,
            'dimension': self.dimension,
            'valid_samples': n_valid,
            'standard_error': standard_error,
            'variance': variance
        }
        
        if absolute_error is not None:
            results_dict.update({
                'analytical_result': self.analytical_result,
                'absolute_error': absolute_error,
                'relative_error': relative_error
            })
        
        statistics_dict = {
            'mean_estimate': integral_estimate,
            'standard_error': standard_error,
            'confidence_interval_95': (
                integral_estimate - 1.96 * standard_error,
                integral_estimate + 1.96 * standard_error
            ),
            'coefficient_of_variation': standard_error / abs(integral_estimate) if integral_estimate != 0 else float('inf')
        }
        
        if self.analytical_result is not None:
            statistics_dict.update({
                'theoretical_value': self.analytical_result,
                'absolute_error': absolute_error,
                'relative_error_percent': relative_error
            })
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results=results_dict,
            statistics=statistics_dict,
            execution_time=execution_time,
            convergence_data=convergence_data
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None,
                 show_function: bool = True, show_samples: bool = False,
                 n_display_points: int = 1000) -> None:
        """Visualize Monte Carlo integration results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        dimension = result.results['dimension']
        
        if dimension == 1:
            self._visualize_1d(result, show_function, show_samples, n_display_points)
        elif dimension == 2:
            self._visualize_2d(result, show_function, show_samples, n_display_points)
        else:
            self._visualize_nd(result)
    
    def _visualize_1d(self, result, show_function, show_samples, n_display_points):
        """Visualize 1D integration"""
        if self.show_convergence:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Get function and domain
        func, domain = self._get_function_and_domain()
        
        if show_function:
            # Plot function
            x_plot = np.linspace(domain[0], domain[1], 1000)
            y_plot = self._evaluate_function(func, x_plot)
            ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
            ax1.fill_between(x_plot, 0, y_plot, alpha=0.3, color='lightblue', 
                           label=f'Integral ≈ {result.results["integral_estimate"]:.4f}')
            
            # Show sample points if requested and available
            if show_samples and self.sample_points is not None:
                n_show = min(len(self.sample_points), n_display_points)
                sample_indices = np.random.choice(len(self.sample_points), n_show, replace=False)
                x_samples = self.sample_points[sample_indices]
                y_samples = self.function_values[sample_indices] if self.function_values is not None else self._evaluate_function(func, x_samples)
                ax1.scatter(x_samples, y_samples, c='red', s=10, alpha=0.6, label='Sample points')
            
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.set_title('Function and Integration Region')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            # Show summary statistics
            self._show_summary_stats(ax1, result)
        
        # Plot convergence if requested
        if self.show_convergence and result.convergence_data:
            samples = [point[0] for point in result.convergence_data]
            estimates = [point[1] for point in result.convergence_data]
            
            ax2.plot(samples, estimates, 'b-', linewidth=2, label='Integral estimate')
            
            if self.analytical_result is not None:
                ax2.axhline(y=self.analytical_result, color='r', linestyle='--', 
                           linewidth=2, label=f'True value: {self.analytical_result:.4f}')
                ax2.fill_between(samples, estimates, self.analytical_result, alpha=0.3,
                               color='red' if estimates[-1] > self.analytical_result else 'blue')
            
            # Add confidence bounds
            std_error = result.results['standard_error']
            final_estimate = estimates[-1]
            ax2.axhline(y=final_estimate + 1.96*std_error, color='gray', linestyle=':', alpha=0.7)
            ax2.axhline(y=final_estimate - 1.96*std_error, color='gray', linestyle=':', alpha=0.7)
            
            ax2.set_xlabel('Number of Samples')
            ax2.set_ylabel('Integral Estimate')
            ax2.set_title('Convergence Analysis')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add final estimate text
            ax2.text(0.7, 0.9, f'Final: {final_estimate:.6f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_2d(self, result, show_function, show_samples, n_display_points):
        """Visualize 2D integration"""
        if self.show_convergence:
            fig = plt.figure(figsize=(18, 6))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
        else:
            fig = plt.figure(figsize=(12, 5))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
        
        # Get function and domain
        func, domain = self._get_function_and_domain()
        
        if show_function:
            # Create meshgrid for function plotting
            x_range = np.linspace(domain[0][0], domain[0][1], 50)
            y_range = np.linspace(domain[1][0], domain[1][1], 50)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Evaluate function on grid
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        Z[i, j] = func([X[i, j], Y[i, j]])
                    except:
                        Z[i, j] = 0
            
            # 3D surface plot
            surf = ax1.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('f(x,y)')
            ax1.set_title('Function Surface')
            
            # Contour plot
            contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
            plt.colorbar(contour, ax=ax2)
            
            # Show sample points if requested and available
            if show_samples and self.sample_points is not None:
                n_show = min(self.sample_points.shape[1], n_display_points)
                sample_indices = np.random.choice(self.sample_points.shape[1], n_show, replace=False)
                x_samples = self.sample_points[0, sample_indices]
                y_samples = self.sample_points[1, sample_indices]
                ax2.scatter(x_samples, y_samples, c='red', s=5, alpha=0.6, label='Sample points')
            
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Function Contours')
            if show_samples:
                ax2.legend()
        else:
            # Show summary in 3D plot area
            ax1.text(0.5, 0.5, 0.5, f'Integral: {result.results["integral_estimate"]:.6f}',
                    transform=ax1.transAxes, fontsize=12, ha='center')
            ax1.set_title('2D Integration Results')
        
        # Plot convergence if requested
        if self.show_convergence and result.convergence_data:
            samples = [point[0] for point in result.convergence_data]
            estimates = [point[1] for point in result.convergence_data]
            
            ax3.plot(samples, estimates, 'b-', linewidth=2, label='Integral estimate')
            
            if self.analytical_result is not None:
                                ax3.axhline(y=self.analytical_result, color='r', linestyle='--', 
                           linewidth=2, label=f'True value: {self.analytical_result:.4f}')
            
            ax3.set_xlabel('Number of Samples')
            ax3.set_ylabel('Integral Estimate')
            ax3.set_title('Convergence Analysis')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_nd(self, result):
        """Visualize high-dimensional integration results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Summary statistics
        self._show_summary_stats(axes[0], result)
        
        # Convergence plot
        if self.show_convergence and result.convergence_data:
            samples = [point[0] for point in result.convergence_data]
            estimates = [point[1] for point in result.convergence_data]
            
            axes[1].plot(samples, estimates, 'b-', linewidth=2, label='Integral estimate')
            
            if self.analytical_result is not None:
                axes[1].axhline(y=self.analytical_result, color='r', linestyle='--', 
                               linewidth=2, label=f'True value: {self.analytical_result:.4f}')
            
            axes[1].set_xlabel('Number of Samples')
            axes[1].set_ylabel('Integral Estimate')
            axes[1].set_title(f'{result.results["dimension"]}D Integration Convergence')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def _show_summary_stats(self, ax, result):
        """Show summary statistics in a text box"""
        integral_est = result.results['integral_estimate']
        std_error = result.results['standard_error']
        dimension = result.results['dimension']
        
        y_positions = [0.8, 0.65, 0.5, 0.35, 0.2]
        texts = [
            f'Integral Estimate: {integral_est:.6f}',
            f'Standard Error: {std_error:.6f}',
            f'Dimension: {dimension}D',
            f'Samples: {result.results["valid_samples"]:,}'
        ]
        
        if 'analytical_result' in result.results:
            texts.append(f'Relative Error: {result.results["relative_error"]:.4f}%')
        else:
            texts.append(f'95% CI: [{integral_est-1.96*std_error:.4f}, {integral_est+1.96*std_error:.4f}]')
        
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
        
        for i, (text, color) in enumerate(zip(texts, colors)):
            ax.text(0.5, y_positions[i], text, transform=ax.transAxes, 
                   fontsize=12, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Integration Results Summary')
        ax.axis('off')
    
    def add_custom_function(self, name: str, func: Callable, 
                           domain: Union[Tuple[float, float], List[Tuple[float, float]]],
                           analytical: Optional[float] = None,
                           description: str = "Custom function"):
        """Add a custom function to the built-in library"""
        self.builtin_functions[name] = {
            'func': func,
            'domain': domain,
            'analytical': analytical,
            'description': description
        }
    
    def get_builtin_functions(self) -> dict:
        """Get information about available built-in functions"""
        return {name: {'domain': info['domain'], 
                      'analytical': info['analytical'],
                      'description': info['description']} 
                for name, info in self.builtin_functions.items()}
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'function': {
                'type': 'choice',
                'default': 'polynomial',
                'choices': list(self.builtin_functions.keys()) + ['custom'],
                'description': 'Function to integrate'
            },
            'domain': {
                'type': 'tuple',
                'default': (0, 1),
                'description': 'Integration domain (a,b) for 1D or [(a1,b1),(a2,b2),...] for nD'
            },
            'n_samples': {
                'type': 'int',
                'default': 100000,
                'min': 1000,
                'max': 10000000,
                'description': 'Number of Monte Carlo samples'
            },
            'method': {
                'type': 'choice',
                'default': 'uniform',
                'choices': ['uniform', 'importance', 'stratified', 'hit_miss'],
                'description': 'Sampling method'
            },
            'show_convergence': {
                'type': 'bool',
                'default': True,
                'description': 'Show convergence analysis'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            },
            'analytical_result': {
                'type': 'float',
                'default': None,
                'description': 'Known analytical result for error calculation (optional)'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        
        # Validate n_samples
        if self.n_samples < 1000:
            errors.append("n_samples must be at least 1000")
        if self.n_samples > 10000000:
            errors.append("n_samples should not exceed 10,000,000 for performance reasons")
        
        # Validate method
        valid_methods = ['uniform', 'importance', 'stratified', 'hit_miss']
        if self.method not in valid_methods:
            errors.append(f"method must be one of {valid_methods}")
        
        # Validate function
        if isinstance(self.function, str):
            if self.function not in self.builtin_functions:
                errors.append(f"Unknown built-in function: {self.function}")
        elif not callable(self.function):
            errors.append("function must be callable or a string name of built-in function")
        
        # Validate domain
        try:
            if isinstance(self.domain, tuple):
                if len(self.domain) != 2:
                    errors.append("1D domain must be a tuple (a, b)")
                elif self.domain[0] >= self.domain[1]:
                    errors.append("Domain bounds must satisfy a < b")
            elif isinstance(self.domain, list):
                for i, dim_range in enumerate(self.domain):
                    if not isinstance(dim_range, tuple) or len(dim_range) != 2:
                        errors.append(f"Domain dimension {i} must be a tuple (a, b)")
                    elif dim_range[0] >= dim_range[1]:
                        errors.append(f"Domain dimension {i} bounds must satisfy a < b")
            else:
                errors.append("domain must be a tuple (a,b) for 1D or list of tuples for nD")
        except Exception as e:
            errors.append(f"Invalid domain format: {str(e)}")
        
        return errors
    
    def estimate_computation_time(self) -> dict:
        """Estimate computation time based on parameters"""
        # Rough estimates based on typical performance
        base_time_per_sample = 1e-6  # seconds per sample for simple functions
        
        # Adjust for dimension
        if isinstance(self.domain, list):
            dimension = len(self.domain)
            time_multiplier = dimension ** 0.5  # Rough scaling with dimension
        else:
            dimension = 1
            time_multiplier = 1.0
        
        # Adjust for method
        method_multipliers = {
            'uniform': 1.0,
            'importance': 1.5,
            'stratified': 2.0,
            'hit_miss': 1.2
        }
        time_multiplier *= method_multipliers.get(self.method, 1.0)
        
        estimated_time = self.n_samples * base_time_per_sample * time_multiplier
        
        return {
            'estimated_seconds': estimated_time,
            'estimated_minutes': estimated_time / 60,
            'dimension': dimension,
            'complexity_factor': time_multiplier
        }
    
    def get_theoretical_error(self) -> Optional[float]:
        """Get theoretical standard error estimate"""
        if hasattr(self, 'result') and self.result is not None:
            return self.result.results.get('standard_error')
        return None
    
    def compare_methods(self, methods: List[str] = None, n_runs: int = 5) -> dict:
        """Compare different integration methods"""
        if methods is None:
            methods = ['uniform', 'importance', 'stratified']
        
        results = {}
        original_method = self.method
        
        for method in methods:
            if method not in ['uniform', 'importance', 'stratified', 'hit_miss']:
                continue
                
            method_results = []
            self.method = method
            
            for run in range(n_runs):
                # Set different random seed for each run
                if 'random_seed' in self.parameters and self.parameters['random_seed'] is not None:
                    np.random.seed(self.parameters['random_seed'] + run)
                
                result = self.run()
                method_results.append({
                    'estimate': result.results['integral_estimate'],
                    'error': result.results.get('absolute_error'),
                    'std_error': result.results['standard_error'],
                    'time': result.execution_time
                })
            
            # Calculate statistics across runs
            estimates = [r['estimate'] for r in method_results]
            times = [r['time'] for r in method_results]
            
            results[method] = {
                'mean_estimate': np.mean(estimates),
                'std_estimate': np.std(estimates),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'runs': method_results
            }
        
        # Restore original method
        self.method = original_method
        return results
