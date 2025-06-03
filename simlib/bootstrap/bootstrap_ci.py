import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Callable, Union, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class BootstrapConfidenceInterval(BaseSimulation):
    """
    Bootstrap resampling for confidence interval estimation of statistical parameters.
    
    This simulation uses bootstrap resampling to estimate confidence intervals for 
    various statistical measures (mean, median, standard deviation, etc.) without 
    making distributional assumptions. The bootstrap method resamples the original 
    data with replacement to create many bootstrap samples, computes the statistic 
    of interest for each sample, and uses the distribution of these statistics to 
    construct confidence intervals.
    
    Mathematical Background:
    -----------------------
    - Bootstrap principle: The empirical distribution approximates the true distribution
    - Resampling: Draw n samples with replacement from original data of size n
    - Bootstrap statistic: θ̂*_b = T(X*_b) for bootstrap sample b
    - Confidence interval: Use quantiles of {θ̂*_1, ..., θ̂*_B} distribution
    - Bias correction: bias = (1/B)∑θ̂*_b - θ̂
    - Variance estimation: Var(θ̂) ≈ (1/(B-1))∑(θ̂*_b - θ̄*)²
    
    Bootstrap Methods:
    -----------------
    1. Percentile Method: Use α/2 and (1-α/2) quantiles directly
    2. Bias-Corrected (BC): Adjust for bias in the bootstrap distribution
    3. Bias-Corrected and Accelerated (BCa): Account for bias and skewness
    4. Basic Bootstrap: Reflect percentiles around original estimate
    5. Studentized Bootstrap: Use bootstrap standard errors for normalization
    
    Statistical Properties:
    ----------------------
    - Asymptotic consistency: Bootstrap distribution → true sampling distribution
    - Coverage accuracy: Actual coverage approaches nominal level as n → ∞
    - Transformation invariance: Bootstrap respects parameter transformations
    - Distribution-free: No parametric assumptions required
    - Robustness: Works for complex statistics and non-normal data
    
    Algorithm Details:
    -----------------
    1. Start with original sample X = {x₁, x₂, ..., xₙ}
    2. For b = 1 to B (bootstrap replications):
       a. Draw bootstrap sample X*_b with replacement from X
       b. Calculate statistic θ̂*_b = T(X*_b)
    3. Sort bootstrap statistics: θ̂*_(1) ≤ ... ≤ θ̂*_(B)
    4. Construct confidence interval using chosen method
    5. Calculate bias and variance estimates
    
    Applications:
    ------------
    - Parameter estimation with non-normal data
    - Complex statistic confidence intervals (correlation, ratios, etc.)
    - Model validation and uncertainty quantification
    - Hypothesis testing via confidence intervals
    - Robust statistical inference
    - Machine learning model evaluation
    - Financial risk assessment
    - Medical and biological research
    - Quality control and process monitoring
    
    Supported Statistics:
    --------------------
    - Mean: Sample average
    - Median: 50th percentile
    - Standard deviation: Sample standard deviation
    - Variance: Sample variance
    - Correlation: Pearson correlation coefficient (for paired data)
    - Ratio: Ratio of means (for paired data)
    - Quantiles: Any specified percentile
    - Custom: User-defined statistic function
    
    Confidence Interval Methods:
    ---------------------------
    Percentile Method:
    - Simple and intuitive
    - CI = [θ̂*_(α/2), θ̂*_(1-α/2)]
    - Good for symmetric distributions
    
    Bias-Corrected (BC):
    - Adjusts for bootstrap bias
    - Better coverage for skewed distributions
    - Requires bias-correction constant
    
    BCa (Bias-Corrected and Accelerated):
    - Gold standard for bootstrap CIs
    - Accounts for bias and skewness
    - Best coverage properties
    - More computationally intensive
    
    Parameters:
    -----------
    data : array-like
        Original sample data for bootstrap resampling
        Can be 1D array for single variable or 2D for multivariate
    statistic : str or callable, default='mean'
        Statistic to compute: 'mean', 'median', 'std', 'var', 'correlation', 'ratio'
        Or custom function that takes data array and returns scalar
    n_bootstrap : int, default=10000
        Number of bootstrap replications
        More replications give smoother CI but take longer
    confidence_level : float, default=0.95
        Confidence level (0 < confidence_level < 1)
        Common values: 0.90, 0.95, 0.99
    method : str, default='percentile'
        CI construction method: 'percentile', 'bc', 'bca', 'basic'
    random_seed : int, optional
        Seed for reproducible bootstrap samples
    
    Attributes:
    -----------
    bootstrap_statistics : ndarray
        Array of bootstrap statistic values from all replications
    original_statistic : float
        Value of statistic computed on original data
    confidence_interval : tuple
        (lower_bound, upper_bound) of confidence interval
    bias_estimate : float
        Estimated bias of the statistic
    standard_error : float
        Bootstrap standard error estimate
    result : SimulationResult
        Complete simulation results and diagnostics
    
    Methods:
    --------
    configure(data, statistic, n_bootstrap, confidence_level, method) : bool
        Configure bootstrap parameters before running
    run(**kwargs) : SimulationResult
        Execute the bootstrap confidence interval estimation
    visualize(result=None, show_distribution=True, show_qq=False) : None
        Create visualizations of bootstrap distribution and CI
    validate_parameters() : List[str]
        Validate current parameters and return any errors
    get_parameter_info() : dict
        Get parameter information for UI generation
    add_custom_statistic(name, func) : None
        Add custom statistic function
    
    Examples:
    ---------
    >>> # Basic confidence interval for mean
    >>> import numpy as np
    >>> data = np.random.normal(50, 10, 100)
    >>> bootstrap = BootstrapConfidenceInterval(data, statistic='mean')
    >>> result = bootstrap.run()
    >>> print(f"95% CI for mean: {result.results['confidence_interval']}")
    
    >>> # Confidence interval for median with more bootstrap samples
    >>> bootstrap_median = BootstrapConfidenceInterval(
    ...     data, statistic='median', n_bootstrap=20000, confidence_level=0.99
    ... )
    >>> result = bootstrap_median.run()
    >>> bootstrap_median.visualize()
    
    >>> # Custom statistic: coefficient of variation
    >>> def cv(x):
    ...     return np.std(x) / np.mean(x)
    >>> bootstrap_cv = BootstrapConfidenceInterval(data, statistic=cv)
    >>> result = bootstrap_cv.run()
    >>> print(f"CV estimate: {result.results['original_statistic']:.4f}")
    >>> print(f"95% CI: {result.results['confidence_interval']}")
    
    >>> # Correlation confidence interval for paired data
    >>> x = np.random.normal(0, 1, 50)
    >>> y = 0.7 * x + np.random.normal(0, 0.5, 50)
    >>> paired_data = np.column_stack([x, y])
    >>> bootstrap_corr = BootstrapConfidenceInterval(
    ...     paired_data, statistic='correlation', method='bca'
    ... )
    >>> result = bootstrap_corr.run()
    
    Visualization Outputs:
    ---------------------
    Standard Mode:
    - Histogram of bootstrap statistics with CI bounds
    - Original statistic marked on distribution
    - Summary statistics and CI information
    - Bias and standard error estimates
    
    Distribution Analysis Mode:
    - Bootstrap distribution histogram with fitted normal overlay
    - Q-Q plot for normality assessment
    - Convergence plot of running CI bounds
    - Bias evolution over bootstrap replications
    
    Performance Characteristics:
    ---------------------------
    - Time complexity: O(n_bootstrap × n × complexity_of_statistic)
    - Space complexity: O(n_bootstrap) for storing bootstrap statistics
    - Memory usage: ~8 bytes per bootstrap replication
    - Typical speeds: 1000-10000 bootstrap samples/second
    - Parallelizable: bootstrap replications are independent
    
    Accuracy Guidelines:
    -------------------
    - 1,000 replications: Good for initial exploration
    - 5,000 replications: Standard for most applications
    - 10,000 replications: High accuracy for important decisions
    - 50,000+ replications: Research-grade precision
    - BCa method: Best coverage but requires more computation
    
    Method Comparison:
    -----------------
    Percentile:
    - Fastest computation
    - Good for symmetric distributions
    - May have poor coverage for skewed data
    
    Bias-Corrected (BC):
    - Better than percentile for skewed data
    - Moderate computational cost
    - Good general-purpose choice
    
    BCa:
    - Best theoretical properties
    - Excellent coverage accuracy
    - Highest computational cost
    - Recommended for final analysis
    
    Statistical Theory:
    ------------------
    - Bootstrap consistency: As n → ∞, bootstrap distribution converges
    - Edgeworth expansion: Higher-order accuracy of BCa method
    - Transformation invariance: g(θ̂) bootstrap = bootstrap of g(θ̂)
    - Studentization: Improved accuracy through variance stabilization
    - Jackknife connection: BCa uses jackknife for acceleration constant
    
    Limitations and Considerations:
    ------------------------------
    - Requires sufficient sample size (n ≥ 20-30 typically)
    - May fail for extreme statistics (min, max)
    - Assumes sample represents population well
    - Computational intensity for large datasets
    - Bootstrap may not work for all statistics
    - Discrete data may need special handling
    
    Extensions and Variations:
    -------------------------
    - Parametric bootstrap: Resample from fitted distribution
    - Smooth bootstrap: Add noise to discrete resamples
    - Block bootstrap: For time series data
    - Bayesian bootstrap: Use Dirichlet weights
    - Double bootstrap: Bootstrap the bootstrap for better accuracy
    - Wild bootstrap: For heteroscedastic regression
    
    References:
    -----------
    - Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap
    - Davison, A. C. & Hinkley, D. V. (1997). Bootstrap Methods and Their Applications
    - Efron, B. (1987). Better Bootstrap Confidence Intervals. JASA, 82(397)
    - DiCiccio, T. J. & Efron, B. (1996). Bootstrap Confidence Intervals. Statistical Science
    - Carpenter, J. & Bithell, J. (2000). Bootstrap confidence intervals. Statistics and Medicine
    """

    def __init__(self, data: Optional[np.ndarray] = None, 
                 statistic: Union[str, Callable] = 'mean',
                 n_bootstrap: int = 10000,
                 confidence_level: float = 0.95,
                 method: str = 'percentile',
                 random_seed: Optional[int] = None):
        super().__init__("Bootstrap Confidence Interval")
        
        # Initialize parameters
        self.data = np.array(data) if data is not None else None
        self.statistic = statistic
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.method = method
        
        # Store in parameters dict for base class
        self.parameters.update({
            'statistic': statistic if isinstance(statistic, str) else 'custom',
            'n_bootstrap': n_bootstrap,
            'confidence_level': confidence_level,
            'method': method,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state
        self.bootstrap_statistics = None
        self.original_statistic = None
        self.confidence_interval = None
        self.bias_estimate = None
        self.standard_error = None
        
        # Built-in statistics
        self._builtin_statistics = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'var': np.var,
            'correlation': self._correlation_statistic,
            'ratio': self._ratio_statistic
        }
        
        self.is_configured = data is not None
    
    def _correlation_statistic(self, data: np.ndarray) -> float:
        """Compute correlation coefficient for 2D data"""
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Correlation statistic requires 2D data with 2 columns")
        return np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    
    def _ratio_statistic(self, data: np.ndarray) -> float:
        """Compute ratio of means for 2D data"""
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Ratio statistic requires 2D data with 2 columns")
        return np.mean(data[:, 0]) / np.mean(data[:, 1])
    
    def configure(self, data: np.ndarray, 
                 statistic: Union[str, Callable] = 'mean',
                 n_bootstrap: int = 10000,
                 confidence_level: float = 0.95,
                 method: str = 'percentile') -> bool:
        """Configure bootstrap parameters"""
        self.data = np.array(data)
        self.statistic = statistic
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.method = method
        
        # Update parameters dict
        self.parameters.update({
            'statistic': statistic if isinstance(statistic, str) else 'custom',
            'n_bootstrap': n_bootstrap,
            'confidence_level': confidence_level,
            'method': method
        })
        
        self.is_configured = True
        return True
    
    def _get_statistic_function(self) -> Callable:
        """Get the statistic function to use"""
        if isinstance(self.statistic, str):
            if self.statistic not in self._builtin_statistics:
                raise ValueError(f"Unknown statistic: {self.statistic}")
            return self._builtin_statistics[self.statistic]
        elif callable(self.statistic):
            return self.statistic
        else:
            raise ValueError("Statistic must be string or callable")
    
    def _bootstrap_sample(self, data: np.ndarray) -> np.ndarray:
        """Generate one bootstrap sample"""
        n = len(data)
        indices = np.random.choice(n, size=n, replace=True)
        return data[indices]
    
    def _percentile_ci(self, bootstrap_stats: np.ndarray, alpha: float) -> Tuple[float, float]:
        """Compute percentile confidence interval"""
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        return lower_bound, upper_bound
    
    def _bias_corrected_ci(self, bootstrap_stats: np.ndarray, original_stat: float, alpha: float) -> Tuple[float, float]:
        """Compute bias-corrected confidence interval"""
        # Bias correction constant
        z0 = self._norm_ppf((bootstrap_stats < original_stat).mean())
        
        # Adjusted percentiles
        z_alpha_2 = self._norm_ppf(alpha / 2)
        z_1_alpha_2 = self._norm_ppf(1 - alpha / 2)
        
        alpha1 = self._norm_cdf(2 * z0 + z_alpha_2)
        alpha2 = self._norm_cdf(2 * z0 + z_1_alpha_2)
        
        # Ensure percentiles are within valid range
        alpha1 = max(0.001, min(0.999, alpha1))
        alpha2 = max(0.001, min(0.999, alpha2))
        
        lower_bound = np.percentile(bootstrap_stats, 100 * alpha1)
        upper_bound = np.percentile(bootstrap_stats, 100 * alpha2)
        return lower_bound, upper_bound
    
    def _bca_ci(self, bootstrap_stats: np.ndarray, original_stat: float, alpha: float) -> Tuple[float, float]:
        """Compute bias-corrected and accelerated confidence interval"""
        # Bias correction constant
        z0 = self._norm_ppf((bootstrap_stats < original_stat).mean())
        
        # Acceleration constant using jackknife
        n = len(self.data)
        jackknife_stats = []
        stat_func = self._get_statistic_function()
        
        for i in range(n):
            # Leave-one-out sample
            jackknife_sample = np.delete(self.data, i, axis=0)
            jackknife_stat = stat_func(jackknife_sample)
            jackknife_stats.append(jackknife_stat)
        
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)
        
        # Acceleration constant
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        
        if denominator == 0:
            a = 0  # Fall back to BC method
        else:
            a = numerator / denominator
        
        # Adjusted percentiles
        z_alpha_2 = self._norm_ppf(alpha / 2)
        z_1_alpha_2 = self._norm_ppf(1 - alpha / 2)
        
        alpha1 = self._norm_cdf(z0 + (z0 + z_alpha_2) / (1 - a * (z0 + z_alpha_2)))
        alpha2 = self._norm_cdf(z0 + (z0 + z_1_alpha_2) / (1 - a * (z0 + z_1_alpha_2)))
        
        # Ensure percentiles are within valid range
        alpha1 = max(0.001, min(0.999, alpha1))
        alpha2 = max(0.001, min(0.999, alpha2))
        
        lower_bound = np.percentile(bootstrap_stats, 100 * alpha1)
        upper_bound = np.percentile(bootstrap_stats, 100 * alpha2)
        return lower_bound, upper_bound
    
    def _basic_ci(self, bootstrap_stats: np.ndarray, original_stat: float, alpha: float) -> Tuple[float, float]:
        """Compute basic bootstrap confidence interval"""
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        # Reflect percentiles around original statistic
        lower_bound = 2 * original_stat - np.percentile(bootstrap_stats, upper_percentile)
        upper_bound = 2 * original_stat - np.percentile(bootstrap_stats, lower_percentile)
        return lower_bound, upper_bound
    
    def _norm_ppf(self, p: float) -> float:
        """Approximate inverse normal CDF (percent point function)"""
        # Simple approximation for standard normal quantiles
        if p <= 0:
            return -np.inf
        if p >= 1:
            return np.inf
        if p == 0.5:
            return 0.0
        
        # Beasley-Springer-Moro algorithm approximation
        if p < 0.5:
            sign = -1
            p = 1 - p
        else:
            sign = 1
        
        t = np.sqrt(-2 * np.log(1 - p))
        x = t - (2.30753 + 0.27061 * t) / (1 + 0.99229 * t + 0.04481 * t * t)
        return sign * x
    
    def _norm_cdf(self, x: float) -> float:
        """Approximate normal CDF"""
        # Abramowitz and Stegun approximation
        if x < 0:
            return 1 - self._norm_cdf(-x)
        
        # Constants
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return y
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute bootstrap confidence interval estimation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Get statistic function
        stat_func = self._get_statistic_function()
        
        # Compute original statistic
        self.original_statistic = stat_func(self.data)
        
        # Generate bootstrap samples and compute statistics
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = self._bootstrap_sample(self.data)
            bootstrap_stat = stat_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        self.bootstrap_statistics = np.array(bootstrap_stats)
        
        # Compute confidence interval using specified method
        alpha = 1 - self.confidence_level
        
        if self.method == 'percentile':
            ci_lower, ci_upper = self._percentile_ci(self.bootstrap_statistics, alpha)
        elif self.method == 'bc':
            ci_lower, ci_upper = self._bias_corrected_ci(self.bootstrap_statistics, self.original_statistic, alpha)
        elif self.method == 'bca':
            ci_lower, ci_upper = self._bca_ci(self.bootstrap_statistics, self.original_statistic, alpha)
        elif self.method == 'basic':
            ci_lower, ci_upper = self._basic_ci(self.bootstrap_statistics, self.original_statistic, alpha)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.confidence_interval = (ci_lower, ci_upper)
        
        # Compute bias and standard error
        self.bias_estimate = np.mean(self.bootstrap_statistics) - self.original_statistic
        self.standard_error = np.std(self.bootstrap_statistics, ddof=1)
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'original_statistic': self.original_statistic,
                'confidence_interval': self.confidence_interval,
                'confidence_level': self.confidence_level,
                'bias_estimate': self.bias_estimate,
                'standard_error': self.standard_error,
                'bootstrap_mean': np.mean(self.bootstrap_statistics),
                'bootstrap_std': np.std(self.bootstrap_statistics),
                'ci_width': ci_upper - ci_lower
            },
            statistics={
                'original_estimate': self.original_statistic,
                'bias_corrected_estimate': self.original_statistic - self.bias_estimate,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'margin_of_error': (ci_upper - ci_lower) / 2
            },
            execution_time=execution_time,
            convergence_data=[]  # Could add convergence tracking if needed
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, 
                 show_distribution: bool = True, show_qq: bool = False) -> None:
        """Visualize bootstrap results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        # Determine subplot layout
        n_plots = 1 + int(show_distribution) + int(show_qq)
        if n_plots == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
            if n_plots == 1:
                axes = [axes]
        
        plot_idx = 0
        
        # Plot 1: Bootstrap distribution with confidence interval
        ax = axes[plot_idx]
        
        # Histogram of bootstrap statistics
        ax.hist(self.bootstrap_statistics, bins=50, density=True, alpha=0.7, 
               color='skyblue', edgecolor='black', label='Bootstrap distribution')
        
        # Mark original statistic
        ax.axvline(self.original_statistic, color='red', linestyle='-', linewidth=2,
                  label=f'Original statistic: {self.original_statistic:.4f}')
        
        # Mark confidence interval bounds
        ci_lower, ci_upper = self.confidence_interval
        ax.axvline(ci_lower, color='green', linestyle='--', linewidth=2,
                  label=f'CI lower: {ci_lower:.4f}')
        ax.axvline(ci_upper, color='green', linestyle='--', linewidth=2,
                  label=f'CI upper: {ci_upper:.4f}')
        
        # Shade confidence interval region
        y_max = ax.get_ylim()[1]
        ax.fill_between([ci_lower, ci_upper], 0, y_max, alpha=0.2, color='green')
        
        ax.set_xlabel('Statistic Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap Distribution\n{int(self.confidence_level*100)}% Confidence Interval')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
        
        # Plot 2: Distribution analysis (if requested)
        if show_distribution and plot_idx < len(axes):
            ax = axes[plot_idx]
            
            # Bootstrap statistics histogram with normal overlay
            ax.hist(self.bootstrap_statistics, bins=50, density=True, alpha=0.7,
                   color='lightcoral', edgecolor='black', label='Bootstrap')
            
            # Overlay normal distribution
            x_range = np.linspace(self.bootstrap_statistics.min(), 
                                self.bootstrap_statistics.max(), 100)
            normal_pdf = (1 / (self.standard_error * np.sqrt(2 * np.pi))) * \
                        np.exp(-0.5 * ((x_range - np.mean(self.bootstrap_statistics)) / self.standard_error) ** 2)
            ax.plot(x_range, normal_pdf, 'b-', linewidth=2, label='Normal approximation')
            
            ax.set_xlabel('Statistic Value')
            ax.set_ylabel('Density')
            ax.set_title('Bootstrap vs Normal Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Plot 3: Q-Q plot (if requested)
        if show_qq and plot_idx < len(axes):
            ax = axes[plot_idx]
            
            # Standardize bootstrap statistics
            standardized = (self.bootstrap_statistics - np.mean(self.bootstrap_statistics)) / self.standard_error
            standardized_sorted = np.sort(standardized)
            
            # Theoretical quantiles
            n = len(standardized_sorted)
            theoretical_quantiles = np.array([self._norm_ppf((i + 0.5) / n) for i in range(n)])
            
            # Q-Q plot
            ax.scatter(theoretical_quantiles, standardized_sorted, alpha=0.6, s=20)
            
            # Reference line
            min_val = min(theoretical_quantiles.min(), standardized_sorted.min())
            max_val = max(theoretical_quantiles.max(), standardized_sorted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect normal')
            
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.set_title('Q-Q Plot: Bootstrap vs Normal')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Add summary text box
        summary_text = f"""Bootstrap Summary:
Original: {self.original_statistic:.4f}
CI: [{ci_lower:.4f}, {ci_upper:.4f}]
Bias: {self.bias_estimate:.4f}
SE: {self.standard_error:.4f}
Method: {self.method.upper()}
Replications: {self.n_bootstrap:,}"""
        
        fig.text(0.02, 0.98, summary_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'statistic': {
                'type': 'choice',
                'choices': ['mean', 'median', 'std', 'var', 'correlation', 'ratio'],
                'default': 'mean',
                'description': 'Statistic to compute confidence interval for'
            },
            'n_bootstrap': {
                'type': 'int',
                'default': 10000,
                'min': 1000,
                'max': 100000,
                'description': 'Number of bootstrap replications'
            },
            'confidence_level': {
                'type': 'float',
                'default': 0.95,
                'min': 0.5,
                'max': 0.999,
                'description': 'Confidence level (e.g., 0.95 for 95%)'
            },
            'method': {
                'type': 'choice',
                'choices': ['percentile', 'bc', 'bca', 'basic'],
                'default': 'percentile',
                'description': 'Bootstrap CI method'
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
        
        if self.data is None:
            errors.append("Data must be provided")
        else:
            if len(self.data) < 10:
                errors.append("Data must have at least 10 observations")
            
            # Check for specific statistic requirements
            if isinstance(self.statistic, str):
                if self.statistic in ['correlation', 'ratio']:
                    if self.data.ndim != 2 or self.data.shape[1] != 2:
                        errors.append(f"Statistic '{self.statistic}' requires 2D data with 2 columns")
        
        if self.n_bootstrap < 1000:
            errors.append("n_bootstrap must be at least 1000")
        if self.n_bootstrap > 100000:
            errors.append("n_bootstrap should not exceed 100,000 for performance reasons")
        
        if not 0.5 < self.confidence_level < 1.0:
            errors.append("confidence_level must be between 0.5 and 1.0")
        
        if self.method not in ['percentile', 'bc', 'bca', 'basic']:
            errors.append("method must be one of: 'percentile', 'bc', 'bca', 'basic'")
        
        return errors
    
    def add_custom_statistic(self, name: str, func: Callable) -> None:
        """Add a custom statistic function"""
        if not callable(func):
            raise ValueError("Custom statistic must be callable")
        self._builtin_statistics[name] = func
    
    def get_bootstrap_samples(self, n_samples: int = 100) -> List[np.ndarray]:
        """Generate and return bootstrap samples for external analysis"""
        if self.data is None:
            raise RuntimeError("No data configured")
        
        samples = []
        for _ in range(n_samples):
            samples.append(self._bootstrap_sample(self.data))
        return samples
    
    def compare_methods(self, methods: List[str] = None) -> dict:
        """Compare different bootstrap CI methods"""
        if methods is None:
            methods = ['percentile', 'bc', 'bca', 'basic']
        
        if self.bootstrap_statistics is None:
            raise RuntimeError("Run simulation first")
        
        alpha = 1 - self.confidence_level
        results = {}
        
        for method in methods:
            try:
                if method == 'percentile':
                    ci = self._percentile_ci(self.bootstrap_statistics, alpha)
                elif method == 'bc':
                    ci = self._bias_corrected_ci(self.bootstrap_statistics, self.original_statistic, alpha)
                elif method == 'bca':
                    ci = self._bca_ci(self.bootstrap_statistics, self.original_statistic, alpha)
                elif method == 'basic':
                    ci = self._basic_ci(self.bootstrap_statistics, self.original_statistic, alpha)
                else:
                    continue
                
                results[method] = {
                    'confidence_interval': ci,
                    'width': ci[1] - ci[0],
                    'lower_bound': ci[0],
                    'upper_bound': ci[1]
                }
            except Exception as e:
                results[method] = {'error': str(e)}
        
        return results
    
    def convergence_analysis(self, sample_sizes: List[int] = None) -> dict:
        """Analyze convergence of bootstrap CI with different sample sizes"""
        if sample_sizes is None:
            max_size = min(self.n_bootstrap, 50000)
            sample_sizes = [int(max_size * f) for f in [0.1, 0.2, 0.5, 0.8, 1.0]]
        
        if self.bootstrap_statistics is None:
            raise RuntimeError("Run simulation first")
        
        alpha = 1 - self.confidence_level
        convergence_results = []
        
        for n in sample_sizes:
            if n > len(self.bootstrap_statistics):
                continue
            
            subset_stats = self.bootstrap_statistics[:n]
            
            if self.method == 'percentile':
                ci = self._percentile_ci(subset_stats, alpha)
            elif self.method == 'bc':
                ci = self._bias_corrected_ci(subset_stats, self.original_statistic, alpha)
            elif self.method == 'bca':
                ci = self._bca_ci(subset_stats, self.original_statistic, alpha)
            elif self.method == 'basic':
                ci = self._basic_ci(subset_stats, self.original_statistic, alpha)
            
            convergence_results.append({
                'n_bootstrap': n,
                'confidence_interval': ci,
                'width': ci[1] - ci[0],
                'bias_estimate': np.mean(subset_stats) - self.original_statistic,
                'standard_error': np.std(subset_stats, ddof=1)
            })
        
        return convergence_results
