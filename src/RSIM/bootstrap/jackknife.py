import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Callable, Any, Union, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class JackknifeEstimation(BaseSimulation):
    """
    Jackknife resampling method for bias estimation and variance calculation.
    
    The jackknife is a resampling technique that systematically leaves out each observation 
    from a dataset to create multiple "leave-one-out" samples. It's used to estimate the 
    bias and variance of a statistic, providing insights into the reliability and accuracy 
    of statistical estimates without making strong distributional assumptions.
    
    Mathematical Background:
    -----------------------
    For a dataset X = {x₁, x₂, ..., xₙ} and statistic θ̂(X):
    - Jackknife samples: X₍ᵢ₎ = X \ {xᵢ} (dataset without i-th observation)
    - Jackknife estimates: θ̂₍ᵢ₎ = θ̂(X₍ᵢ₎) for i = 1, ..., n
    - Jackknife mean: θ̂₍·₎ = (1/n) Σᵢ θ̂₍ᵢ₎
    - Bias estimate: bias_jack = (n-1)(θ̂₍·₎ - θ̂)
    - Bias-corrected estimate: θ̂_bc = θ̂ - bias_jack = nθ̂ - (n-1)θ̂₍·₎
    - Variance estimate: var_jack = ((n-1)/n) Σᵢ (θ̂₍ᵢ₎ - θ̂₍·₎)²
    - Standard error: SE_jack = √var_jack
    
    Statistical Properties:
    ----------------------
    - Bias estimation: O(n⁻²) for smooth statistics (vs O(n⁻¹) original bias)
    - Variance estimation: Consistent for most statistics
    - Distribution approximation: Asymptotically normal under regularity conditions
    - Confidence intervals: θ̂_bc ± t_{n-1,α/2} × SE_jack
    - Effective for linear and quasi-linear statistics
    - Less effective for non-smooth statistics (e.g., median, quantiles)
    
    Algorithm Details:
    -----------------
    1. Compute original statistic θ̂ on full dataset
    2. For each observation i = 1, ..., n:
       a. Create jackknife sample X₍ᵢ₎ by removing xᵢ
       b. Compute jackknife estimate θ̂₍ᵢ₎ = θ̂(X₍ᵢ₎)
    3. Calculate jackknife mean θ̂₍·₎
    4. Estimate bias: bias_jack = (n-1)(θ̂₍·₎ - θ̂)
    5. Compute bias-corrected estimate: θ̂_bc = θ̂ - bias_jack
    6. Calculate variance: var_jack = ((n-1)/n) Σᵢ (θ̂₍ᵢ₎ - θ̂₍·₎)²
    7. Construct confidence intervals using t-distribution
    
    Applications:
    ------------
    - Bias correction for statistical estimators
    - Variance estimation without distributional assumptions
    - Model validation and cross-validation
    - Outlier detection and influence analysis
    - Regression diagnostics and leverage analysis
    - Time series analysis (modified jackknife)
    - Machine learning model assessment
    - Robust statistics and non-parametric inference
    
    Advantages:
    -----------
    - Computationally simple and deterministic
    - No random sampling required (unlike bootstrap)
    - Exact bias correction for linear statistics
    - Good variance estimation for smooth statistics
    - Interpretable influence measures
    - Works with small sample sizes
    - Provides individual observation influence
    
    Limitations:
    -----------
    - Less accurate than bootstrap for complex statistics
    - Poor performance with non-smooth statistics
    - Assumes statistic is approximately linear
    - Can be unstable for highly variable statistics
    - Limited to bias correction of order O(n⁻¹)
    - May not work well with dependent data
    
    Supported Statistics:
    --------------------
    Built-in statistics:
    - 'mean': Sample mean
    - 'median': Sample median
    - 'std': Sample standard deviation
    - 'var': Sample variance
    - 'skewness': Sample skewness
    - 'kurtosis': Sample kurtosis
    - 'min': Minimum value
    - 'max': Maximum value
    - 'range': Range (max - min)
    - 'iqr': Interquartile range
    - 'mad': Median absolute deviation
    - 'cv': Coefficient of variation
    
    Custom statistics: User-defined functions accepting numpy arrays
    
    Parameters:
    -----------
    data : array-like
        Input dataset for jackknife analysis
        Can be 1D array, list, or pandas Series
    statistic : str or callable, default='mean'
        Statistic to analyze - built-in name or custom function
        Custom functions should accept numpy array and return scalar
    confidence_level : float, default=0.95
        Confidence level for intervals (between 0 and 1)
    store_estimates : bool, default=True
        Whether to store all jackknife estimates for analysis
    
    Attributes:
    -----------
    jackknife_estimates : numpy.ndarray, optional
        Individual jackknife estimates θ̂₍ᵢ₎ (if store_estimates=True)
    influence_values : numpy.ndarray, optional
        Influence of each observation: (n-1)(θ̂₍·₎ - θ̂₍ᵢ₎)
    original_estimate : float
        Original statistic computed on full dataset
    jackknife_mean : float
        Mean of jackknife estimates
    bias_estimate : float
        Estimated bias of the statistic
    bias_corrected_estimate : float
        Bias-corrected estimate
    variance_estimate : float
        Jackknife variance estimate
    standard_error : float
        Standard error of the statistic
    confidence_interval : tuple
        (lower_bound, upper_bound) confidence interval
    result : SimulationResult
        Complete simulation results
    
    Methods:
    --------
    configure(data, statistic, confidence_level, store_estimates) : bool
        Configure jackknife parameters
    run(**kwargs) : SimulationResult
        Execute jackknife analysis
    visualize(result=None, show_influence=True, show_distribution=True) : None
        Create visualizations of jackknife results
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI generation
    add_custom_statistic(name, function) : None
        Add custom statistic function
    
    Examples:
    ---------
    >>> # Basic jackknife analysis of sample mean
    >>> data = np.random.normal(10, 2, 50)
    >>> jack = JackknifeEstimation(data=data, statistic='mean')
    >>> result = jack.run()
    >>> print(f"Original mean: {result.results['original_estimate']:.4f}")
    >>> print(f"Bias-corrected: {result.results['bias_corrected_estimate']:.4f}")
    >>> print(f"Standard error: {result.results['standard_error']:.4f}")
    
    >>> # Jackknife analysis with custom statistic
    >>> def trimmed_mean(x, trim_prop=0.1):
    ...     n_trim = int(len(x) * trim_prop)
    ...     sorted_x = np.sort(x)
    ...     return np.mean(sorted_x[n_trim:-n_trim]) if n_trim > 0 else np.mean(x)
    >>> 
    >>> jack_custom = JackknifeEstimation(data=data, statistic=trimmed_mean)
    >>> result = jack_custom.run()
    >>> jack_custom.visualize()
    
    >>> # Bias correction for sample variance
    >>> jack_var = JackknifeEstimation(data=data, statistic='var', confidence_level=0.99)
    >>> result = jack_var.run()
    >>> print(f"Bias estimate: {result.results['bias_estimate']:.6f}")
    >>> print(f"99% CI: {result.results['confidence_interval']}")
    
    >>> # Influence analysis
    >>> jack_influence = JackknifeEstimation(data=data, statistic='mean', store_estimates=True)
    >>> result = jack_influence.run()
    >>> jack_influence.visualize(show_influence=True)
    >>> # Identify influential observations
    >>> influences = result.results['influence_values']
    >>> outliers = np.where(np.abs(influences) > 2 * np.std(influences))[0]
    >>> print(f"Potentially influential observations: {outliers}")
    
    Visualization Outputs:
    ---------------------
    Standard Mode:
    - Summary statistics and bias correction results
    - Confidence interval visualization
    - Comparison of original vs bias-corrected estimates
    
    Influence Analysis (show_influence=True):
    - Influence plot showing impact of each observation
    - Identification of high-influence points
    - Leverage analysis for outlier detection
    
    Distribution Analysis (show_distribution=True):
    - Histogram of jackknife estimates
    - Normal Q-Q plot for distribution assessment
    - Convergence of estimates across samples
    
    Performance Characteristics:
    ---------------------------
    - Time complexity: O(n × T) where T is statistic computation time
    - Space complexity: O(n) for estimates storage
    - Deterministic: Same results on repeated runs
    - Parallelizable: Jackknife estimates can be computed independently
    - Memory efficient: Only stores necessary intermediate results
    
    Theoretical Foundations:
    -----------------------
    - Based on influence function theory (Hampel, 1974)
    - Connection to infinitesimal jackknife (Jaeckel, 1972)
    - Relationship to bootstrap (Efron, 1979, 1982)
    - Asymptotic properties (Miller, 1974; Efron & Stein, 1981)
    - Robustness properties (Huber, 1981)
    
    Extensions and Variations:
    -------------------------
    - Delete-d jackknife: Remove d > 1 observations
    - Weighted jackknife: Different weights for observations
    - Cluster jackknife: For clustered/grouped data
    - Block jackknife: For time series data
    - Jackknife-after-bootstrap: Combining both methods
    - Infinitesimal jackknife: Theoretical limiting case
    
    Quality Diagnostics:
    -------------------
    - Stability check: Coefficient of variation of jackknife estimates
    - Linearity assessment: Correlation between estimates and influences
    - Outlier detection: Extreme influence values
    - Convergence monitoring: Estimate stability across sample sizes
    
    References:
    -----------
    - Quenouille, M. H. (1949). Approximate tests of correlation in time-series
    - Tukey, J. W. (1958). Bias and confidence in not-quite large samples
    - Miller, R. G. (1974). The jackknife—a review. Biometrika, 61(1), 1-15
    - Efron, B. (1982). The Jackknife, the Bootstrap and Other Resampling Plans
    - Shao, J. & Tu, D. (1995). The Jackknife and Bootstrap
    - Davison, A. C. & Hinkley, D. V. (1997). Bootstrap Methods and their Application
    """

    # Built-in statistics
    _BUILTIN_STATISTICS = {
        'mean': lambda x: np.mean(x),
        'median': lambda x: np.median(x),
        'std': lambda x: np.std(x, ddof=1),
        'var': lambda x: np.var(x, ddof=1),
        'skewness': lambda x: ((x - np.mean(x)) ** 3).mean() / (np.std(x, ddof=1) ** 3),
        'kurtosis': lambda x: ((x - np.mean(x)) ** 4).mean() / (np.std(x, ddof=1) ** 4) - 3,
        'min': lambda x: np.min(x),
        'max': lambda x: np.max(x),
        'range': lambda x: np.max(x) - np.min(x),
        'iqr': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        'mad': lambda x: np.median(np.abs(x - np.median(x))),
        'cv': lambda x: np.std(x, ddof=1) / np.mean(x) if np.mean(x) != 0 else np.inf
    }

    def __init__(self, data: Union[np.ndarray, List, Any] = None, 
                 statistic: Union[str, Callable] = 'mean',
                 confidence_level: float = 0.95,
                 store_estimates: bool = True):
        super().__init__("Jackknife Estimation")
        
        # Convert data to numpy array
        if data is not None:
            self.data = np.asarray(data).flatten()
        else:
            self.data = None
        
        self.statistic = statistic
        self.confidence_level = confidence_level
        self.store_estimates = store_estimates
        
        # Store in parameters dict for base class
        self.parameters.update({
            'data_size': len(self.data) if self.data is not None else 0,
            'statistic': str(statistic),
            'confidence_level': confidence_level,
            'store_estimates': store_estimates
        })
        
        # Results storage
        self.jackknife_estimates = None
        self.influence_values = None
        self.original_estimate = None
        self.jackknife_mean = None
        self.bias_estimate = None
        self.bias_corrected_estimate = None
        self.variance_estimate = None
        self.standard_error = None
        self.confidence_interval = None
        
        # Custom statistics registry
        self._custom_statistics = {}
        
        self.is_configured = data is not None
    
    def configure(self, data: Union[np.ndarray, List, Any], 
                 statistic: Union[str, Callable] = 'mean',
                 confidence_level: float = 0.95,
                 store_estimates: bool = True) -> bool:
        """Configure jackknife parameters"""
        self.data = np.asarray(data).flatten()
        self.statistic = statistic
        self.confidence_level = confidence_level
        self.store_estimates = store_estimates
        
        # Update parameters dict
        self.parameters.update({
            'data_size': len(self.data),
            'statistic': str(statistic),
            'confidence_level': confidence_level,
            'store_estimates': store_estimates
        })
        
        self.is_configured = True
        return True
    
    def add_custom_statistic(self, name: str, function: Callable) -> None:
        """Add a custom statistic function"""
        self._custom_statistics[name] = function
    
    def _get_statistic_function(self) -> Callable:
        """Get the statistic function to use"""
        if callable(self.statistic):
            return self.statistic
        elif isinstance(self.statistic, str):
            if self.statistic in self._BUILTIN_STATISTICS:
                return self._BUILTIN_STATISTICS[self.statistic]
            elif self.statistic in self._custom_statistics:
                return self._custom_statistics[self.statistic]
            else:
                raise ValueError(f"Unknown statistic: {self.statistic}")
        else:
            raise ValueError("Statistic must be a string or callable")
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute jackknife analysis"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data provided for jackknife analysis")
        
        start_time = time.time()
        n = len(self.data)
        
        # Get statistic function
        stat_func = self._get_statistic_function()
        
        # Compute original estimate on full dataset
        try:
            self.original_estimate = stat_func(self.data)
        except Exception as e:
            raise ValueError(f"Error computing statistic on full dataset: {e}")
        
        # Compute jackknife estimates
        jackknife_estimates = []
        
        for i in range(n):
            # Create jackknife sample (leave-one-out)
            jackknife_sample = np.concatenate([self.data[:i], self.data[i+1:]])
            
            try:
                jackknife_est = stat_func(jackknife_sample)
                jackknife_estimates.append(jackknife_est)
            except Exception as e:
                raise ValueError(f"Error computing statistic on jackknife sample {i}: {e}")
        
        jackknife_estimates = np.array(jackknife_estimates)
        
        # Store estimates if requested
        if self.store_estimates:
            self.jackknife_estimates = jackknife_estimates
        
        # Calculate jackknife statistics
        self.jackknife_mean = np.mean(jackknife_estimates)
        
        # Bias estimation: bias = (n-1) * (jackknife_mean - original_estimate)
        self.bias_estimate = (n - 1) * (self.jackknife_mean - self.original_estimate)
        
        # Bias-corrected estimate: original - bias = n * original - (n-1) * jackknife_mean
        self.bias_corrected_estimate = n * self.original_estimate - (n - 1) * self.jackknife_mean
        
        # Variance estimation: var = ((n-1)/n) * sum((jackknife_est - jackknife_mean)^2)
        self.variance_estimate = ((n - 1) / n) * np.sum((jackknife_estimates - self.jackknife_mean) ** 2)
        
        # Standard error
        self.standard_error = np.sqrt(self.variance_estimate)
        
        # Influence values: (n-1) * (jackknife_mean - jackknife_estimates)
        self.influence_values = (n - 1) * (self.jackknife_mean - jackknife_estimates)
        
        # Confidence interval using t-distribution
        from scipy import stats
        alpha = 1 - self.confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        margin_error = t_critical * self.standard_error
        
        self.confidence_interval = (
            self.bias_corrected_estimate - margin_error,
            self.bias_corrected_estimate + margin_error
        )
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'original_estimate': self.original_estimate,
                'jackknife_mean': self.jackknife_mean,
                'bias_estimate': self.bias_estimate,
                'bias_corrected_estimate': self.bias_corrected_estimate,
                'variance_estimate': self.variance_estimate,
                'standard_error': self.standard_error,
                'confidence_interval': self.confidence_interval,
                'confidence_level': self.confidence_level,
                'influence_values': self.influence_values.tolist() if self.influence_values is not None else None,
                'sample_size': n
            },
            statistics={
                'bias_reduction': abs(self.bias_estimate),
                'relative_bias': abs(self.bias_estimate) / abs(self.original_estimate) * 100 if self.original_estimate != 0 else 0,
                'coefficient_of_variation': self.standard_error / abs(self.bias_corrected_estimate) * 100 if self.bias_corrected_estimate != 0 else 0,
                'influence_range': (np.min(self.influence_values), np.max(self.influence_values)),
                'max_influence_index': np.argmax(np.abs(self.influence_values))
            },
            execution_time=execution_time,
            convergence_data=None
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None,
                 show_influence: bool = True,
                 show_distribution: bool = True) -> None:
        """Visualize jackknife results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        # Determine subplot layout
        n_plots = 1 + int(show_influence) + int(show_distribution)
        if n_plots == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
        elif n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot 1: Summary statistics
        ax = axes[plot_idx]
        plot_idx += 1
        
        # Create summary text
        original = result.results['original_estimate']
        bias_corrected = result.results['bias_corrected_estimate']
        bias = result.results['bias_estimate']
        se = result.results['standard_error']
        ci = result.results['confidence_interval']
        
        summary_text = [
            f"Original Estimate: {original:.6f}",
            f"Bias-Corrected: {bias_corrected:.6f}",
            f"Bias Estimate: {bias:.6f}",
            f"Standard Error: {se:.6f}",
            f"95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]",
            f"Sample Size: {result.results['sample_size']}"
        ]
        
        # Display summary
        for i, text in enumerate(summary_text):
            ax.text(0.05, 0.9 - i*0.12, text, transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Confidence interval visualization
        ax.errorbar([0.7], [bias_corrected], yerr=[[bias_corrected - ci[0]], [ci[1] - bias_corrected]], 
                   fmt='ro', capsize=10, capthick=2, markersize=8, label='Bias-corrected estimate')
        ax.plot([0.65, 0.75], [original, original], 'b-', linewidth=3, label='Original estimate')
        
        ax.set_xlim(0.6, 0.8)
        y_range = max(abs(ci[1] - bias_corrected), abs(bias_corrected - ci[0])) * 1.5
        ax.set_ylim(bias_corrected - y_range, bias_corrected + y_range)
        ax.set_title('Jackknife Results Summary')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Influence analysis
        if show_influence and self.influence_values is not None:
            ax = axes[plot_idx]
            plot_idx += 1
            
            influences = self.influence_values
            indices = np.arange(len(influences))
            
            # Plot influence values
            ax.scatter(indices, influences, alpha=0.6, s=30)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Highlight high-influence points
            influence_threshold = 2 * np.std(influences)
            high_influence = np.abs(influences) > influence_threshold
            if np.any(high_influence):
                ax.scatter(indices[high_influence], influences[high_influence], 
                          color='red', s=60, alpha=0.8, label=f'High influence (>{influence_threshold:.3f})')
                ax.legend()
            
            ax.set_xlabel('Observation Index')
            ax.set_ylabel('Influence Value')
            ax.set_title('Jackknife Influence Analysis')
            ax.grid(True, alpha=0.3)
            
            # Add influence statistics
            max_influence_idx = np.argmax(np.abs(influences))
            ax.text(0.02, 0.98, f'Max influence: {influences[max_influence_idx]:.4f} (obs {max_influence_idx})',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Plot 3: Distribution of jackknife estimates
        if show_distribution and self.jackknife_estimates is not None:
            ax = axes[plot_idx]
            plot_idx += 1
            
            estimates = self.jackknife_estimates
            
            # Histogram
            ax.hist(estimates, bins=min(30, len(estimates)//3), alpha=0.7, density=True, 
                   color='skyblue', edgecolor='black')
            
            # Add vertical lines for key statistics
            ax.axvline(self.original_estimate, color='blue', linestyle='-', linewidth=2, 
                      label=f'Original: {self.original_estimate:.4f}')
            ax.axvline(self.jackknife_mean, color='green', linestyle='--', linewidth=2,
                      label=f'Jackknife mean: {self.jackknife_mean:.4f}')
            ax.axvline(self.bias_corrected_estimate, color='red', linestyle=':', linewidth=2,
                      label=f'Bias-corrected: {self.bias_corrected_estimate:.4f}')
            
            # Overlay normal distribution
            x_range = np.linspace(estimates.min(), estimates.max(), 100)
            normal_overlay = (1/np.sqrt(2*np.pi*self.variance_estimate)) * \
                           np.exp(-0.5 * (x_range - self.jackknife_mean)**2 / self.variance_estimate)
            ax.plot(x_range, normal_overlay, 'orange', linewidth=2, alpha=0.8, label='Normal approximation')
            
            ax.set_xlabel('Jackknife Estimates')
            ax.set_ylabel('Density')
            ax.set_title('Distribution of Jackknife Estimates')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'data': {
                'type': 'array',
                'description': 'Input dataset for jackknife analysis'
            },
            'statistic': {
                'type': 'choice',
                'choices': list(self._BUILTIN_STATISTICS.keys()) + ['custom'],
                'default': 'mean',
                'description': 'Statistic to analyze'
            },
            'confidence_level': {
                'type': 'float',
                'default': 0.95,
                'min': 0.01,
                'max': 0.99,
                'description': 'Confidence level for intervals'
            },
            'store_estimates': {
                'type': 'bool',
                'default': True,
                'description': 'Store individual jackknife estimates'
            }
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        
        if self.data is None:
            errors.append("No data provided")
        elif len(self.data) < 3:
            errors.append("Data must contain at least 3 observations")
        elif len(self.data) > 10000:
            errors.append("Data size should not exceed 10,000 observations for performance reasons")
        
        if not (0 < self.confidence_level < 1):
            errors.append("Confidence level must be between 0 and 1")
        
        # Test statistic function
        if self.data is not None and len(self.data) >= 3:
            try:
                self._get_statistic_function()
                # Test on small sample
                test_sample = self.data[:3]
                stat_func = self._get_statistic_function()
                stat_func(test_sample)
            except Exception as e:
                errors.append(f"Invalid statistic function: {e}")
        
        return errors

    def get_builtin_statistics(self) -> List[str]:
        """Get list of available built-in statistics"""
        return list(self._BUILTIN_STATISTICS.keys())
    
    def get_influence_summary(self) -> dict:
        """Get summary of influence analysis"""
        if self.influence_values is None:
            return {}
        
        influences = self.influence_values
        return {
            'max_positive_influence': np.max(influences),
            'max_negative_influence': np.min(influences),
            'max_absolute_influence': influences[np.argmax(np.abs(influences))],
            'most_influential_index': np.argmax(np.abs(influences)),
            'influence_std': np.std(influences),
            'high_influence_count': np.sum(np.abs(influences) > 2 * np.std(influences)),
            'influence_range': np.max(influences) - np.min(influences)
        }
    
    def detect_outliers(self, threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect potentially influential observations based on jackknife influence.
        
        Parameters:
        -----------
        threshold : float, default=2.0
            Threshold in standard deviations for outlier detection
            
        Returns:
        --------
        outlier_indices : np.ndarray
            Indices of potentially influential observations
        influence_scores : np.ndarray
            Standardized influence scores for outliers
        """
        if self.influence_values is None:
            raise RuntimeError("Run jackknife analysis first")
        
        influences = self.influence_values
        influence_std = np.std(influences)
        standardized_influences = influences / influence_std if influence_std > 0 else influences
        
        outlier_mask = np.abs(standardized_influences) > threshold
        outlier_indices = np.where(outlier_mask)[0]
        influence_scores = standardized_influences[outlier_mask]
        
        return outlier_indices, influence_scores
    
    def compare_estimates(self) -> dict:
        """
        Compare different estimates and provide interpretation.
        
        Returns:
        --------
        comparison : dict
            Dictionary with comparison metrics and interpretation
        """
        if self.result is None:
            raise RuntimeError("Run jackknife analysis first")
        
        original = self.original_estimate
        bias_corrected = self.bias_corrected_estimate
        bias = self.bias_estimate
        se = self.standard_error
        
        # Calculate relative changes
        relative_bias = abs(bias) / abs(original) * 100 if original != 0 else 0
        relative_change = abs(bias_corrected - original) / abs(original) * 100 if original != 0 else 0
        
        # Determine significance of bias correction
        bias_significant = abs(bias) > se / 2  # Rough heuristic
        
        # Stability assessment
        cv = se / abs(bias_corrected) * 100 if bias_corrected != 0 else np.inf
        stability = "High" if cv < 5 else "Medium" if cv < 15 else "Low"
        
        return {
            'original_estimate': original,
            'bias_corrected_estimate': bias_corrected,
            'bias_estimate': bias,
            'standard_error': se,
            'relative_bias_percent': relative_bias,
            'relative_change_percent': relative_change,
            'bias_significant': bias_significant,
            'coefficient_of_variation': cv,
            'stability_assessment': stability,
            'recommendation': self._get_recommendation(relative_bias, bias_significant, stability)
        }
    
    def _get_recommendation(self, relative_bias: float, bias_significant: bool, stability: str) -> str:
        """Generate recommendation based on jackknife results"""
        if relative_bias < 1 and not bias_significant:
            return "Bias is negligible. Original estimate is reliable."
        elif relative_bias < 5 and stability in ["High", "Medium"]:
            return "Small bias detected. Bias-corrected estimate recommended."
        elif relative_bias < 10 and stability == "High":
            return "Moderate bias detected. Use bias-corrected estimate with caution."
        elif stability == "Low":
            return "High variability detected. Consider larger sample or different method."
        else:
            return "Substantial bias detected. Bias-corrected estimate strongly recommended."
    
    def bootstrap_comparison(self, n_bootstrap: int = 1000, random_seed: Optional[int] = None) -> dict:
        """
        Compare jackknife results with bootstrap for validation.
        
        Parameters:
        -----------
        n_bootstrap : int, default=1000
            Number of bootstrap samples
        random_seed : int, optional
            Random seed for bootstrap sampling
            
        Returns:
        --------
        comparison : dict
            Comparison between jackknife and bootstrap results
        """
        if self.data is None or self.result is None:
            raise RuntimeError("Run jackknife analysis first")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Get statistic function
        stat_func = self._get_statistic_function()
        n = len(self.data)
        
        # Bootstrap sampling
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(self.data, size=n, replace=True)
            bootstrap_est = stat_func(bootstrap_sample)
            bootstrap_estimates.append(bootstrap_est)
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_estimates)
        bootstrap_bias = bootstrap_mean - self.original_estimate
        bootstrap_se = np.std(bootstrap_estimates, ddof=1)
        bootstrap_var = np.var(bootstrap_estimates, ddof=1)
        
        # Compare with jackknife
        bias_agreement = abs(self.bias_estimate - bootstrap_bias) / max(abs(self.bias_estimate), abs(bootstrap_bias)) if max(abs(self.bias_estimate), abs(bootstrap_bias)) > 0 else 0
        se_agreement = abs(self.standard_error - bootstrap_se) / max(self.standard_error, bootstrap_se) if max(self.standard_error, bootstrap_se) > 0 else 0
        
        return {
            'jackknife_bias': self.bias_estimate,
            'bootstrap_bias': bootstrap_bias,
            'jackknife_se': self.standard_error,
            'bootstrap_se': bootstrap_se,
            'jackknife_var': self.variance_estimate,
            'bootstrap_var': bootstrap_var,
            'bias_relative_difference': bias_agreement * 100,
            'se_relative_difference': se_agreement * 100,
            'methods_agree': bias_agreement < 0.1 and se_agreement < 0.1,
            'bootstrap_estimates': bootstrap_estimates.tolist()
        }

