import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Callable, Tuple, Union
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class ImportanceSampling(BaseSimulation):
    """
    Importance Sampling variance reduction technique for Monte Carlo integration.
    
    Importance sampling is a Monte Carlo variance reduction technique that samples from
    a proposal distribution rather than the original distribution to reduce estimation
    variance. It's particularly effective when the integrand has regions of high importance
    that are poorly sampled by standard Monte Carlo methods.
    
    Mathematical Background:
    -----------------------
    Standard Monte Carlo estimates integrals of the form:
        I = ∫ f(x) p(x) dx = E_p[f(X)]
    
    Importance sampling rewrites this as:
        I = ∫ f(x) p(x)/q(x) q(x) dx = E_q[f(X) w(X)]
    
    Where:
    - p(x): original probability density function
    - q(x): importance (proposal) density function  
    - w(x) = p(x)/q(x): importance weight
    - The estimator becomes: Î = (1/n) Σ f(X_i) w(X_i), X_i ~ q(x)
    
    Variance Reduction Principle:
    ----------------------------
    - Optimal q*(x) ∝ |f(x)| p(x) minimizes variance
    - Good q(x) should have heavy tails where |f(x)| p(x) is large
    - Variance reduction factor can be substantial (10x-1000x common)
    - Poor choice of q(x) can increase variance dramatically
    
    Key Properties:
    --------------
    - Unbiased estimator: E_q[Î] = I for any valid q(x)
    - Variance: Var_q[Î] = (1/n) Var_q[f(X)w(X)]
    - Efficiency gain: Var_standard / Var_importance
    - Requires q(x) > 0 wherever f(x)p(x) ≠ 0
    - Most effective for rare event simulation and tail estimation
    
    Algorithm Details:
    -----------------
    1. Choose importance density q(x) based on problem structure
    2. Generate samples X_1, ..., X_n from q(x)
    3. Compute importance weights w_i = p(X_i) / q(X_i)
    4. Estimate integral: Î = (1/n) Σ f(X_i) w_i
    5. Track convergence and compare with standard Monte Carlo
    
    Built-in Example Functions:
    --------------------------
    The class includes several demonstration problems:
    
    1. Exponential Tail Integration:
       - Integrand: f(x) = exp(-x²/2) for x > a (a large)
       - Standard sampling misses rare tail events
       - Importance sampling with exponential proposal
    
    2. Rare Event Probability:
       - P(X > threshold) where X ~ N(0,1), threshold large
       - Standard MC requires enormous sample sizes
       - Exponential tilting provides massive variance reduction
    
    3. Option Pricing (Financial):
       - European call option valuation
       - Importance sampling for out-of-the-money options
       - Variance reduction in tail probability estimation
    
    4. Reliability Analysis:
       - P(failure) = P(g(X) < 0) where failure is rare
       - Standard MC inefficient for small failure probabilities
       - Importance sampling concentrates samples near failure region
    
    Applications:
    ------------
    - Rare event simulation (reliability, finance, queueing)
    - Tail probability estimation
    - High-dimensional integration
    - Bayesian posterior sampling
    - Risk assessment and extreme value analysis
    - Physics simulations (particle transport, quantum Monte Carlo)
    - Machine learning (variational inference, reinforcement learning)
    
    Theoretical Foundations:
    -----------------------
    - Radon-Nikodym theorem provides mathematical foundation
    - Central Limit Theorem still applies with modified variance
    - Effective sample size: n_eff = (Σw_i)² / Σw_i²
    - Coefficient of variation of weights indicates efficiency
    - Asymptotic normality: √n(Î - I) → N(0, σ²_IS)
    
    Simulation Features:
    -------------------
    - Multiple built-in example problems with known solutions
    - Custom function integration with user-defined densities
    - Automatic proposal distribution selection for common cases
    - Real-time convergence comparison with standard Monte Carlo
    - Importance weight diagnostics and efficiency metrics
    - Visual analysis of sampling distributions and convergence
    - Statistical testing of variance reduction effectiveness
    
    Parameters:
    -----------
    problem_type : str, default='exponential_tail'
        Type of integration problem to solve
        Options: 'exponential_tail', 'rare_event', 'option_pricing', 'custom'
    n_samples : int, default=10000
        Number of importance samples to generate
    comparison_samples : int, default=None
        Number of standard MC samples for comparison (default: same as n_samples)
    integrand : callable, optional
        Custom function f(x) to integrate (required for 'custom' problem_type)
    original_pdf : callable, optional
        Original probability density p(x) (required for 'custom' problem_type)
    importance_pdf : callable, optional
        Importance density q(x) (required for 'custom' problem_type)
    importance_sampler : callable, optional
        Function to generate samples from q(x) (required for 'custom' problem_type)
    problem_params : dict, default={}
        Parameters specific to the chosen problem type
    show_convergence : bool, default=True
        Whether to track convergence during simulation
    random_seed : int, optional
        Seed for reproducible results
    
    Attributes:
    -----------
    samples : ndarray
        Generated importance samples
    weights : ndarray
        Computed importance weights
    weighted_values : ndarray
        f(X_i) * w_i for each sample
    convergence_data : list
        Convergence tracking data
    comparison_data : list
        Standard Monte Carlo comparison data
    efficiency_metrics : dict
        Variance reduction and efficiency statistics
    result : SimulationResult
        Complete simulation results
    
    Methods:
    --------
    configure(problem_type, n_samples, **kwargs) : bool
        Configure importance sampling parameters
    run(**kwargs) : SimulationResult
        Execute importance sampling simulation
    visualize(result=None, show_weights=True, show_samples=True) : None
        Create comprehensive visualization of results
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI generation
    set_custom_problem(integrand, original_pdf, importance_pdf, sampler) : None
        Define custom integration problem
    
    Examples:
    ---------
    >>> # Exponential tail integration
    >>> is_sim = ImportanceSampling(
    ...     problem_type='exponential_tail',
    ...     n_samples=10000,
    ...     problem_params={'threshold': 3.0}
    ... )
    >>> result = is_sim.run()
    >>> print(f"Variance reduction factor: {result.results['variance_reduction_factor']:.2f}")
    
    >>> # Rare event probability estimation
    >>> rare_sim = ImportanceSampling(
    ...     problem_type='rare_event',
    ...     n_samples=5000,
    ...     problem_params={'threshold': 4.0}
    ... )
    >>> result = rare_sim.run()
    >>> rare_sim.visualize()
    
    >>> # Custom integration problem
    >>> def my_integrand(x):
    ...     return np.exp(-x**2) * np.sin(x)
    >>> def standard_normal_pdf(x):
    ...     return np.exp(-x**2/2) / np.sqrt(2*np.pi)
    >>> def importance_pdf(x):
    ...     return np.exp(-np.abs(x)) / 2  # Laplace distribution
    >>> def importance_sampler(n):
    ...     return np.random.laplace(0, 1, n)
    >>> 
    >>> custom_sim = ImportanceSampling(problem_type='custom')
    >>> custom_sim.set_custom_problem(my_integrand, standard_normal_pdf, 
    ...                              importance_pdf, importance_sampler)
    >>> result = custom_sim.run(n_samples=20000)
    
    Visualization Outputs:
    ---------------------
    Standard Visualization:
    - Convergence comparison between importance sampling and standard MC
    - Importance weight distribution and diagnostics
    - Sample distribution comparison
    - Efficiency metrics and variance reduction summary
    
    Advanced Visualization:
    - Effective sample size evolution
    - Weight coefficient of variation over time
    - Integrand evaluation across sample space
    - Proposal vs target distribution overlay
    
    Performance Characteristics:
    ---------------------------
    - Time complexity: O(n_samples) for sampling + O(n_samples) for weight computation
    - Space complexity: O(n_samples) for storing samples and weights
    - Variance reduction: Problem-dependent, can range from 2x to 1000x+
    - Convergence rate: Still O(1/√n) but with reduced constant
    - Computational overhead: Typically 10-50% more than standard MC
    
    Efficiency Guidelines:
    ---------------------
    - Effective sample size should be > 10% of actual sample size
    - Weight coefficient of variation should be < 5 for good efficiency
    - Variance reduction factor > 2 indicates successful implementation
    - Monitor for weight degeneracy (few samples with very high weights)
    
    Common Pitfalls and Solutions:
    -----------------------------
    1. Poor Proposal Choice:
       - Symptom: High weight variance, low effective sample size
       - Solution: Choose q(x) closer to optimal |f(x)|p(x)
    
    2. Weight Degeneracy:
       - Symptom: Few samples dominate the estimate
       - Solution: Improve proposal or use adaptive importance sampling
    
    3. Infinite Variance:
       - Symptom: Weights have infinite variance theoretically
       - Solution: Ensure q(x) has heavier tails than f(x)p(x)
    
    4. Negative Weights:
       - Symptom: Biased estimates
       - Solution: Ensure proper density ratio computation
    
    Advanced Techniques:
    -------------------
    - Adaptive importance sampling: Update proposal during simulation
    - Multiple importance sampling: Combine several proposals
    - Sequential importance sampling: For time series and filtering
    - Cross-entropy method: Optimize proposal parameters
    - Mixture importance sampling: Use mixture of proposals
    
    Statistical Diagnostics:
    -----------------------
    - Effective sample size: n_eff = (Σw_i)²/(Σw_i²)
    - Weight coefficient of variation: CV = std(weights)/mean(weights)
    - Variance reduction factor: Var_MC / Var_IS
    - Relative efficiency: (Time_MC × Var_MC) / (Time_IS × Var_IS)
    - Weight entropy: H = -Σ(w_i/Σw_j) log(w_i/Σw_j)
    
    References:
    -----------
    - Rubinstein, R. Y. & Kroese, D. P. (2016). Simulation and the Monte Carlo Method
    - Owen, A. B. (2013). Monte Carlo theory, methods and examples
    - Robert, C. P. & Casella, G. (2004). Monte Carlo Statistical Methods
    - Glynn, P. W. & Iglehart, D. L. (1989). Importance sampling for stochastic simulations
    - Bucklew, J. A. (2004). Introduction to Rare Event Simulation
    """

    def __init__(self, problem_type: str = 'exponential_tail', n_samples: int = 10000,
                 comparison_samples: Optional[int] = None, integrand: Optional[Callable] = None,
                 original_pdf: Optional[Callable] = None, importance_pdf: Optional[Callable] = None,
                 importance_sampler: Optional[Callable] = None, problem_params: dict = {},
                 show_convergence: bool = True, random_seed: Optional[int] = None):
        super().__init__("Importance Sampling")
        
        # Initialize parameters
        self.problem_type = problem_type
        self.n_samples = n_samples
        self.comparison_samples = comparison_samples or n_samples
        self.integrand = integrand
        self.original_pdf = original_pdf
        self.importance_pdf = importance_pdf
        self.importance_sampler = importance_sampler
        self.problem_params = problem_params.copy()
        self.show_convergence = show_convergence
        
        # Store in parameters dict for base class
        self.parameters.update({
            'problem_type': problem_type,
            'n_samples': n_samples,
            'comparison_samples': self.comparison_samples,
            'problem_params': problem_params,
            'show_convergence': show_convergence,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state
        self.samples = None
        self.weights = None
        self.weighted_values = None
        self.convergence_data = None
        self.comparison_data = None
        self.efficiency_metrics = {}
        self.true_value = None
        
        self.is_configured = True
    
    def configure(self, problem_type: str = 'exponential_tail', n_samples: int = 10000,
                 comparison_samples: Optional[int] = None, problem_params: dict = {},
                 show_convergence: bool = True) -> bool:
        """Configure importance sampling parameters"""
        self.problem_type = problem_type
        self.n_samples = n_samples
        self.comparison_samples = comparison_samples or n_samples
        self.problem_params = problem_params.copy()
        self.show_convergence = show_convergence
        
        # Update parameters dict
        self.parameters.update({
            'problem_type': problem_type,
            'n_samples': n_samples,
            'comparison_samples': self.comparison_samples,
            'problem_params': problem_params,
            'show_convergence': show_convergence
        })
        
        self.is_configured = True
        return True
    
    def set_custom_problem(self, integrand: Callable, original_pdf: Callable,
                          importance_pdf: Callable, importance_sampler: Callable) -> None:
        """Set custom integration problem"""
        self.integrand = integrand
        self.original_pdf = original_pdf
        self.importance_pdf = importance_pdf
        self.importance_sampler = importance_sampler
        self.problem_type = 'custom'
        self.parameters['problem_type'] = 'custom'
    
    def _setup_exponential_tail_problem(self) -> Tuple[Callable, Callable, Callable, Callable, float]:
        """Setup exponential tail integration problem"""
        threshold = self.problem_params.get('threshold', 3.0)
        
        def integrand(x):
            return np.exp(-x**2/2) * (x > threshold)
        
        def original_pdf(x):
            return np.exp(-x**2/2) / np.sqrt(2*np.pi)
        def importance_pdf(x):
            # Exponential distribution shifted to start at threshold
            return np.exp(-(x - threshold)) * (x >= threshold)
        
        def importance_sampler(n):
            return threshold + np.random.exponential(1.0, n)
        
        # True value (analytical)
        true_value = np.exp(-threshold**2/2) / np.sqrt(2*np.pi)
        
        return integrand, original_pdf, importance_pdf, importance_sampler, true_value
    
    def _setup_rare_event_problem(self) -> Tuple[Callable, Callable, Callable, Callable, float]:
        """Setup rare event probability estimation"""
        threshold = self.problem_params.get('threshold', 4.0)
        
        def integrand(x):
            return (x > threshold).astype(float)
        
        def original_pdf(x):
            return np.exp(-x**2/2) / np.sqrt(2*np.pi)
        
        def importance_pdf(x):
            # Shifted normal distribution
            mu = threshold + 1.0
            return np.exp(-(x - mu)**2/2) / np.sqrt(2*np.pi)
        
        def importance_sampler(n):
            mu = threshold + 1.0
            return np.random.normal(mu, 1.0, n)
        
        # True value using complementary error function
        from scipy.stats import norm
        true_value = 1 - norm.cdf(threshold)
        
        return integrand, original_pdf, importance_pdf, importance_sampler, true_value
    
    def _setup_option_pricing_problem(self) -> Tuple[Callable, Callable, Callable, Callable, float]:
        """Setup option pricing problem"""
        strike = self.problem_params.get('strike', 110.0)
        spot = self.problem_params.get('spot', 100.0)
        rate = self.problem_params.get('rate', 0.05)
        volatility = self.problem_params.get('volatility', 0.2)
        maturity = self.problem_params.get('maturity', 1.0)
        
        def integrand(z):
            # Black-Scholes stock price at maturity
            stock_price = spot * np.exp((rate - 0.5*volatility**2)*maturity + volatility*np.sqrt(maturity)*z)
            payoff = np.maximum(stock_price - strike, 0)
            return np.exp(-rate * maturity) * payoff
        
        def original_pdf(z):
            return np.exp(-z**2/2) / np.sqrt(2*np.pi)
        
        def importance_pdf(z):
            # Shift distribution towards in-the-money region
            d1 = (np.log(spot/strike) + (rate + 0.5*volatility**2)*maturity) / (volatility*np.sqrt(maturity))
            mu = d1 + 1.0  # Shift further into the money
            return np.exp(-(z - mu)**2/2) / np.sqrt(2*np.pi)
        
        def importance_sampler(n):
            d1 = (np.log(spot/strike) + (rate + 0.5*volatility**2)*maturity) / (volatility*np.sqrt(maturity))
            mu = d1 + 1.0
            return np.random.normal(mu, 1.0, n)
        
        # True value using Black-Scholes formula
        from scipy.stats import norm
        d1 = (np.log(spot/strike) + (rate + 0.5*volatility**2)*maturity) / (volatility*np.sqrt(maturity))
        d2 = d1 - volatility*np.sqrt(maturity)
        true_value = spot*norm.cdf(d1) - strike*np.exp(-rate*maturity)*norm.cdf(d2)
        
        return integrand, original_pdf, importance_pdf, importance_sampler, true_value
    
    def _run_standard_monte_carlo(self, integrand: Callable, n_samples: int) -> Tuple[float, np.ndarray]:
        """Run standard Monte Carlo for comparison"""
        if self.problem_type == 'exponential_tail':
            # Sample from standard normal, but only keep samples > threshold
            threshold = self.problem_params.get('threshold', 3.0)
            samples_needed = 0
            estimates = []
            
            while samples_needed < n_samples:
                batch_size = min(10000, n_samples - samples_needed)
                x = np.random.normal(0, 1, batch_size * 10)  # Generate extra samples
                valid_samples = x[x > threshold]
                
                if len(valid_samples) > 0:
                    n_valid = min(len(valid_samples), batch_size)
                    values = integrand(valid_samples[:n_valid])
                    estimates.extend(values)
                    samples_needed += n_valid
                
                if len(valid_samples) == 0:  # Avoid infinite loop
                    estimates.extend([0] * batch_size)
                    samples_needed += batch_size
            
            estimates = np.array(estimates[:n_samples])
            
        elif self.problem_type == 'rare_event':
            x = np.random.normal(0, 1, n_samples)
            estimates = integrand(x)
            
        elif self.problem_type == 'option_pricing':
            z = np.random.normal(0, 1, n_samples)
            estimates = integrand(z)
            
        else:  # custom
            # For custom problems, assume sampling from standard normal
            x = np.random.normal(0, 1, n_samples)
            estimates = integrand(x)
        
        return np.mean(estimates), estimates
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute importance sampling simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Setup problem-specific functions
        if self.problem_type == 'exponential_tail':
            integrand, original_pdf, importance_pdf, importance_sampler, true_value = self._setup_exponential_tail_problem()
        elif self.problem_type == 'rare_event':
            integrand, original_pdf, importance_pdf, importance_sampler, true_value = self._setup_rare_event_problem()
        elif self.problem_type == 'option_pricing':
            integrand, original_pdf, importance_pdf, importance_sampler, true_value = self._setup_option_pricing_problem()
        elif self.problem_type == 'custom':
            if not all([self.integrand, self.original_pdf, self.importance_pdf, self.importance_sampler]):
                raise ValueError("Custom problem requires all functions to be defined")
            integrand = self.integrand
            original_pdf = self.original_pdf
            importance_pdf = self.importance_pdf
            importance_sampler = self.importance_sampler
            true_value = self.problem_params.get('true_value', None)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")
        
        self.true_value = true_value
        
        # Generate importance samples
        self.samples = importance_sampler(self.n_samples)
        
        # Compute importance weights
        original_densities = original_pdf(self.samples)
        importance_densities = importance_pdf(self.samples)
        
        # Avoid division by zero
        importance_densities = np.maximum(importance_densities, 1e-10)
        self.weights = original_densities / importance_densities
        
        # Compute weighted function values
        function_values = integrand(self.samples)
        self.weighted_values = function_values * self.weights
        
        # Importance sampling estimate
        is_estimate = np.mean(self.weighted_values)
        
        # Run standard Monte Carlo for comparison
        mc_estimate, mc_values = self._run_standard_monte_carlo(integrand, self.comparison_samples)
        
        # Compute convergence data
        convergence_data = []
        comparison_data = []
        
        if self.show_convergence:
            step_size = max(100, self.n_samples // 100)
            for i in range(step_size, self.n_samples + 1, step_size):
                # Importance sampling convergence
                running_is_estimate = np.mean(self.weighted_values[:i])
                convergence_data.append((i, running_is_estimate))
                
                # Standard MC convergence
                if i <= len(mc_values):
                    running_mc_estimate = np.mean(mc_values[:i])
                    comparison_data.append((i, running_mc_estimate))
        
        self.convergence_data = convergence_data
        self.comparison_data = comparison_data
        
        # Compute efficiency metrics
        is_variance = np.var(self.weighted_values)
        mc_variance = np.var(mc_values) if len(mc_values) > 1 else float('inf')
        
        # Effective sample size
        sum_weights = np.sum(self.weights)
        sum_weights_squared = np.sum(self.weights**2)
        effective_sample_size = sum_weights**2 / sum_weights_squared if sum_weights_squared > 0 else 0
        
        # Weight coefficient of variation
        weight_cv = np.std(self.weights) / np.mean(self.weights) if np.mean(self.weights) > 0 else float('inf')
        
        # Variance reduction factor
        variance_reduction_factor = mc_variance / is_variance if is_variance > 0 else float('inf')
        
        self.efficiency_metrics = {
            'effective_sample_size': effective_sample_size,
            'weight_coefficient_variation': weight_cv,
            'variance_reduction_factor': variance_reduction_factor,
            'is_variance': is_variance,
            'mc_variance': mc_variance
        }
        
        execution_time = time.time() - start_time
        
        # Create result object
        results_dict = {
            'importance_sampling_estimate': is_estimate,
            'standard_monte_carlo_estimate': mc_estimate,
            'variance_reduction_factor': variance_reduction_factor,
            'effective_sample_size': effective_sample_size,
            'weight_coefficient_variation': weight_cv,
            'is_standard_error': np.sqrt(is_variance / self.n_samples),
            'mc_standard_error': np.sqrt(mc_variance / self.comparison_samples)
        }
        
        if true_value is not None:
            results_dict.update({
                'true_value': true_value,
                'is_absolute_error': abs(is_estimate - true_value),
                'mc_absolute_error': abs(mc_estimate - true_value),
                'is_relative_error': abs(is_estimate - true_value) / abs(true_value) * 100 if true_value != 0 else float('inf'),
                'mc_relative_error': abs(mc_estimate - true_value) / abs(true_value) * 100 if true_value != 0 else float('inf')
            })
        
        statistics_dict = {
            'mean_importance_weight': np.mean(self.weights),
            'max_importance_weight': np.max(self.weights),
            'min_importance_weight': np.min(self.weights),
            'weight_entropy': -np.sum((self.weights / np.sum(self.weights)) * 
                                    np.log(self.weights / np.sum(self.weights) + 1e-10))
        }
        
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
                 show_weights: bool = True, show_samples: bool = True) -> None:
        """Visualize importance sampling results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        # Create subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Convergence comparison
        ax1 = plt.subplot(2, 3, 1)
        if self.convergence_data and self.comparison_data:
            is_samples = [point[0] for point in self.convergence_data]
            is_estimates = [point[1] for point in self.convergence_data]
            mc_samples = [point[0] for point in self.comparison_data]
            mc_estimates = [point[1] for point in self.comparison_data]
            
            ax1.plot(is_samples, is_estimates, 'b-', linewidth=2, label='Importance Sampling')
            ax1.plot(mc_samples, mc_estimates, 'r-', linewidth=2, label='Standard Monte Carlo')
            
            if self.true_value is not None:
                ax1.axhline(y=self.true_value, color='g', linestyle='--', linewidth=2, label='True Value')
            
            ax1.set_xlabel('Number of Samples')
            ax1.set_ylabel('Estimate')
            ax1.set_title('Convergence Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Importance weights distribution
        ax2 = plt.subplot(2, 3, 2)
        if show_weights and self.weights is not None:
            ax2.hist(self.weights, bins=50, alpha=0.7, density=True, color='blue', edgecolor='black')
            ax2.axvline(np.mean(self.weights), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(self.weights):.3f}')
            ax2.set_xlabel('Importance Weight')
            ax2.set_ylabel('Density')
            ax2.set_title('Importance Weights Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sample distribution comparison
        ax3 = plt.subplot(2, 3, 3)
        if show_samples and self.samples is not None:
            # Show sample histogram
            ax3.hist(self.samples, bins=50, alpha=0.7, density=True, color='lightblue', 
                    edgecolor='black', label='Importance Samples')
            
            # Overlay theoretical densities if possible
            if self.problem_type in ['exponential_tail', 'rare_event', 'option_pricing']:
                x_range = np.linspace(np.min(self.samples), np.max(self.samples), 100)
                
                if self.problem_type == 'exponential_tail':
                    threshold = self.problem_params.get('threshold', 3.0)
                    importance_density = np.exp(-(x_range - threshold)) * (x_range >= threshold)
                    ax3.plot(x_range, importance_density, 'r-', linewidth=2, label='Importance PDF')
                
                elif self.problem_type == 'rare_event':
                    threshold = self.problem_params.get('threshold', 4.0)
                    mu = threshold + 1.0
                    importance_density = np.exp(-(x_range - mu)**2/2) / np.sqrt(2*np.pi)
                    original_density = np.exp(-x_range**2/2) / np.sqrt(2*np.pi)
                    ax3.plot(x_range, importance_density, 'r-', linewidth=2, label='Importance PDF')
                    ax3.plot(x_range, original_density, 'g--', linewidth=2, label='Original PDF')
                
                elif self.problem_type == 'option_pricing':
                    strike = self.problem_params.get('strike', 110.0)
                    spot = self.problem_params.get('spot', 100.0)
                    rate = self.problem_params.get('rate', 0.05)
                    volatility = self.problem_params.get('volatility', 0.2)
                    maturity = self.problem_params.get('maturity', 1.0)
                    
                    d1 = (np.log(spot/strike) + (rate + 0.5*volatility**2)*maturity) / (volatility*np.sqrt(maturity))
                    mu = d1 + 1.0
                    importance_density = np.exp(-(x_range - mu)**2/2) / np.sqrt(2*np.pi)
                    original_density = np.exp(-x_range**2/2) / np.sqrt(2*np.pi)
                    ax3.plot(x_range, importance_density, 'r-', linewidth=2, label='Importance PDF')
                    ax3.plot(x_range, original_density, 'g--', linewidth=2, label='Original PDF')
            
            ax3.set_xlabel('Sample Value')
            ax3.set_ylabel('Density')
            ax3.set_title('Sample Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency metrics summary
        ax4 = plt.subplot(2, 3, 4)
        metrics = [
            f"Variance Reduction: {result.results.get('variance_reduction_factor', 0):.2f}x",
            f"Effective Sample Size: {result.results.get('effective_sample_size', 0):.0f}",
            f"Weight CV: {result.results.get('weight_coefficient_variation', 0):.3f}",
            f"IS Std Error: {result.results.get('is_standard_error', 0):.6f}",
            f"MC Std Error: {result.results.get('mc_standard_error', 0):.6f}"
        ]
        
        if 'true_value' in result.results:
            metrics.extend([
                f"IS Rel Error: {result.results.get('is_relative_error', 0):.4f}%",
                f"MC Rel Error: {result.results.get('mc_relative_error', 0):.4f}%"
            ])
        
        for i, metric in enumerate(metrics):
            ax4.text(0.05, 0.9 - i*0.12, metric, transform=ax4.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Efficiency Metrics')
        ax4.axis('off')
        
        # Plot 5: Effective sample size evolution
        ax5 = plt.subplot(2, 3, 5)
        if self.show_convergence and len(self.convergence_data) > 0:
            sample_counts = [point[0] for point in self.convergence_data]
            eff_sizes = []
            
            for i, n in enumerate(sample_counts):
                weights_subset = self.weights[:n]
                sum_w = np.sum(weights_subset)
                sum_w2 = np.sum(weights_subset**2)
                eff_size = sum_w**2 / sum_w2 if sum_w2 > 0 else 0
                eff_sizes.append(eff_size)
            
            ax5.plot(sample_counts, eff_sizes, 'purple', linewidth=2)
            ax5.plot(sample_counts, sample_counts, 'k--', alpha=0.5, label='Ideal (n_eff = n)')
            ax5.set_xlabel('Number of Samples')
            ax5.set_ylabel('Effective Sample Size')
            ax5.set_title('Effective Sample Size Evolution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Weight coefficient of variation evolution
        ax6 = plt.subplot(2, 3, 6)
        if self.show_convergence and len(self.convergence_data) > 0:
            sample_counts = [point[0] for point in self.convergence_data]
            weight_cvs = []
            
            for i, n in enumerate(sample_counts):
                weights_subset = self.weights[:n]
                cv = np.std(weights_subset) / np.mean(weights_subset) if np.mean(weights_subset) > 0 else float('inf')
                weight_cvs.append(cv)
            
            ax6.plot(sample_counts, weight_cvs, 'orange', linewidth=2)
            ax6.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='CV = 1 (Warning)')
            ax6.axhline(y=5.0, color='r', linestyle='-', alpha=0.7, label='CV = 5 (Poor)')
            ax6.set_xlabel('Number of Samples')
            ax6.set_ylabel('Weight Coefficient of Variation')
            ax6.set_title('Weight CV Evolution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed summary
        print("\n" + "="*60)
        print("IMPORTANCE SAMPLING SIMULATION RESULTS")
        print("="*60)
        
        print(f"\nProblem Type: {self.problem_type}")
        print(f"Number of Samples: {self.n_samples:,}")
        
        if 'true_value' in result.results:
            print(f"\nTrue Value: {result.results['true_value']:.8f}")
        
        print(f"Importance Sampling Estimate: {result.results['importance_sampling_estimate']:.8f}")
        print(f"Standard Monte Carlo Estimate: {result.results['standard_monte_carlo_estimate']:.8f}")
        
        if 'is_absolute_error' in result.results:
            print(f"\nAbsolute Errors:")
            print(f"  IS Error: {result.results['is_absolute_error']:.8f}")
            print(f"  MC Error: {result.results['mc_absolute_error']:.8f}")
            
            print(f"\nRelative Errors:")
            print(f"  IS Error: {result.results['is_relative_error']:.4f}%")
            print(f"  MC Error: {result.results['mc_relative_error']:.4f}%")
        
        print(f"\nEfficiency Metrics:")
        print(f"  Variance Reduction Factor: {result.results['variance_reduction_factor']:.2f}x")
        print(f"  Effective Sample Size: {result.results['effective_sample_size']:.0f} ({result.results['effective_sample_size']/self.n_samples*100:.1f}%)")
        print(f"  Weight Coefficient of Variation: {result.results['weight_coefficient_variation']:.3f}")
        
        print(f"\nStandard Errors:")
        print(f"  IS Standard Error: {result.results['is_standard_error']:.8f}")
        print(f"  MC Standard Error: {result.results['mc_standard_error']:.8f}")
        print(f"  Error Reduction Factor: {result.results['mc_standard_error']/result.results['is_standard_error']:.2f}x")
        
        print(f"\nWeight Statistics:")
        print(f"  Mean Weight: {result.statistics['mean_importance_weight']:.6f}")
        print(f"  Max Weight: {result.statistics['max_importance_weight']:.6f}")
        print(f"  Min Weight: {result.statistics['min_importance_weight']:.6f}")
        print(f"  Weight Entropy: {result.statistics['weight_entropy']:.6f}")
        
        print(f"\nExecution Time: {result.execution_time:.4f} seconds")
        
        # Efficiency assessment
        print(f"\nEfficiency Assessment:")
        eff_sample_ratio = result.results['effective_sample_size'] / self.n_samples
        weight_cv = result.results['weight_coefficient_variation']
        var_reduction = result.results['variance_reduction_factor']
        
        if eff_sample_ratio > 0.5:
            print("  ✓ Excellent effective sample size ratio")
        elif eff_sample_ratio > 0.1:
            print("  ✓ Good effective sample size ratio")
        else:
            print("  ⚠ Low effective sample size - consider improving proposal")
        
        if weight_cv < 1.0:
            print("  ✓ Excellent weight coefficient of variation")
        elif weight_cv < 5.0:
            print("  ✓ Acceptable weight coefficient of variation")
        else:
            print("  ⚠ High weight CV - proposal may be suboptimal")
        
        if var_reduction > 10:
            print("  ✓ Excellent variance reduction achieved")
        elif var_reduction > 2:
            print("  ✓ Good variance reduction achieved")
        else:
            print("  ⚠ Limited variance reduction - check proposal choice")
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'problem_type': {
                'type': 'choice',
                'default': 'exponential_tail',
                'choices': ['exponential_tail', 'rare_event', 'option_pricing', 'custom'],
                'description': 'Type of integration problem'
            },
            'n_samples': {
                'type': 'int',
                'default': 10000,
                'min': 1000,
                'max': 1000000,
                'description': 'Number of importance samples'
            },
            'comparison_samples': {
                'type': 'int',
                'default': None,
                'min': 1000,
                'max': 1000000,
                'description': 'Number of standard MC samples for comparison'
            },
            'show_convergence': {
                'type': 'bool',
                'default': True,
                'description': 'Show convergence tracking'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            },
            'problem_params': {
                'type': 'dict',
                'default': {},
                'description': 'Problem-specific parameters',
                'sub_params': {
                    'exponential_tail': {
                        'threshold': {
                            'type': 'float',
                            'default': 3.0,
                            'min': 1.0,
                            'max': 6.0,
                            'description': 'Integration threshold'
                        }
                    },
                    'rare_event': {
                        'threshold': {
                            'type': 'float',
                            'default': 4.0,
                            'min': 2.0,
                            'max': 6.0,
                            'description': 'Rare event threshold'
                        }
                    },
                    'option_pricing': {
                        'strike': {
                            'type': 'float',
                            'default': 110.0,
                            'min': 50.0,
                            'max': 200.0,
                            'description': 'Option strike price'
                        },
                        'spot': {
                            'type': 'float',
                            'default': 100.0,
                            'min': 50.0,
                            'max': 200.0,
                            'description': 'Current stock price'
                        },
                        'rate': {
                            'type': 'float',
                            'default': 0.05,
                            'min': 0.0,
                            'max': 0.2,
                            'description': 'Risk-free rate'
                        },
                        'volatility': {
                            'type': 'float',
                            'default': 0.2,
                            'min': 0.1,
                            'max': 0.5,
                            'description': 'Volatility'
                        },
                        'maturity': {
                            'type': 'float',
                            'default': 1.0,
                            'min': 0.1,
                            'max': 5.0,
                            'description': 'Time to maturity (years)'
                        }
                    }
                }
            }
        }
    
    def validate_parameters(self) -> list[str]:
        """Validate simulation parameters"""
        errors = []
        
        if self.n_samples < 1000:
            errors.append("n_samples must be at least 1000")
        if self.n_samples > 1000000:
            errors.append("n_samples should not exceed 1,000,000 for performance reasons")
        
        if self.problem_type not in ['exponential_tail', 'rare_event', 'option_pricing', 'custom']:
            errors.append("problem_type must be one of: exponential_tail, rare_event, option_pricing, custom")
        
        if self.problem_type == 'custom':
            if not all([self.integrand, self.original_pdf, self.importance_pdf, self.importance_sampler]):
                errors.append("Custom problem requires integrand, original_pdf, importance_pdf, and importance_sampler")
        
        # Validate problem-specific parameters
        if self.problem_type == 'exponential_tail':
            threshold = self.problem_params.get('threshold', 3.0)
            if threshold < 1.0 or threshold > 6.0:
                errors.append("exponential_tail threshold must be between 1.0 and 6.0")
        
        elif self.problem_type == 'rare_event':
            threshold = self.problem_params.get('threshold', 4.0)
            if threshold < 2.0 or threshold > 6.0:
                errors.append("rare_event threshold must be between 2.0 and 6.0")
        
        elif self.problem_type == 'option_pricing':
            strike = self.problem_params.get('strike', 110.0)
            spot = self.problem_params.get('spot', 100.0)
            rate = self.problem_params.get('rate', 0.05)
            volatility = self.problem_params.get('volatility', 0.2)
            maturity = self.problem_params.get('maturity', 1.0)
            
            if strike <= 0:
                errors.append("option_pricing strike must be positive")
            if spot <= 0:
                errors.append("option_pricing spot must be positive")
            if rate < 0 or rate > 0.5:
                errors.append("option_pricing rate must be between 0 and 0.5")
            if volatility <= 0 or volatility > 1.0:
                errors.append("option_pricing volatility must be between 0 and 1.0")
            if maturity <= 0 or maturity > 10:
                errors.append("option_pricing maturity must be between 0 and 10 years")
        
        return errors




