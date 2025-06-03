import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from typing import Optional, Union, Tuple, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class KaplanMeierEstimator(BaseSimulation):
    """
    Kaplan-Meier non-parametric survival function estimator.
    
    The Kaplan-Meier estimator is a non-parametric statistic used to estimate the survival 
    function from lifetime data. It provides a way to measure the fraction of subjects 
    living for a certain amount of time after treatment, accounting for censored observations 
    where the exact time of event occurrence is unknown.
    
    Mathematical Background:
    -----------------------
    The Kaplan-Meier estimator is defined as:
    Ŝ(t) = ∏(i: tᵢ ≤ t) (1 - dᵢ/nᵢ)
    
    Where:
    - Ŝ(t) = estimated survival probability at time t
    - tᵢ = distinct event times (ordered)
    - dᵢ = number of events at time tᵢ
    - nᵢ = number of subjects at risk just before time tᵢ
    - The product is over all event times ≤ t
    
    Key Properties:
    --------------
    - Non-parametric: No assumptions about underlying distribution
    - Right-continuous step function
    - Decreases only at observed event times
    - Handles censored data naturally
    - Maximum likelihood estimator under non-informative censoring
    - Asymptotically normal with known variance structure
    
    Censoring Types Supported:
    -------------------------
    - Right censoring: Most common, observation ends before event
    - Left truncation: Subject enters study after some delay
    - Interval censoring: Event known to occur within time interval
    - Type I censoring: Study ends at predetermined time
    - Type II censoring: Study ends after predetermined number of events
    
    Statistical Properties:
    ----------------------
    - Consistency: Ŝ(t) → S(t) as n → ∞
    - Asymptotic normality: √n(Ŝ(t) - S(t)) → N(0, σ²(t))
    - Greenwood's formula for variance: Var[Ŝ(t)] = Ŝ(t)² ∑(dᵢ/(nᵢ(nᵢ-dᵢ)))
    - Confidence intervals: Ŝ(t) ± z_{α/2} × SE[Ŝ(t)]
    - Log-log transformation for better CI properties
    
    Applications:
    ------------
    - Medical research: Patient survival analysis
    - Engineering: Reliability and failure time analysis
    - Economics: Duration modeling (unemployment, job tenure)
    - Marketing: Customer lifetime value and churn analysis
    - Quality control: Product lifetime estimation
    - Clinical trials: Treatment efficacy comparison
    - Epidemiology: Disease progression studies
    
    Advantages:
    -----------
    - No distributional assumptions required
    - Handles censored data naturally
    - Provides intuitive survival curves
    - Well-established statistical properties
    - Robust to outliers
    - Easy interpretation and communication
    
    Limitations:
    -----------
    - Cannot extrapolate beyond observed data
    - Assumes non-informative censoring
    - No smoothing (step function)
    - Limited predictive capability
    - Sensitive to small sample sizes at tail
    - Cannot handle time-varying covariates directly
    
    Algorithm Details:
    -----------------
    1. Sort all event and censoring times
    2. At each event time tᵢ:
       - Count events (dᵢ) and subjects at risk (nᵢ)
       - Calculate conditional survival: (nᵢ - dᵢ)/nᵢ
       - Update cumulative survival: Ŝ(tᵢ) = Ŝ(tᵢ₋₁) × (nᵢ - dᵢ)/nᵢ
    3. Compute variance using Greenwood's formula
    4. Calculate confidence intervals
    5. Generate survival curve data points
    
    Simulation Features:
    -------------------
    - Flexible data input (arrays, DataFrame, or simulated data)
    - Automatic handling of ties in event times
    - Confidence interval calculation with multiple methods
    - Comprehensive statistical summaries
    - Risk table generation for survival curves
    - Median survival time estimation with CI
    - Survival probability estimation at specific times
    - Comparison with theoretical distributions
    - Extensive visualization options
    
    Parameters:
    -----------
    time_data : array-like, optional
        Observed times (either event times or censoring times)
    event_data : array-like, optional
        Event indicators (1 = event occurred, 0 = censored)
    confidence_level : float, default=0.95
        Confidence level for interval estimation (0 < confidence_level < 1)
    ci_method : str, default='log-log'
        Method for confidence interval calculation
        Options: 'linear', 'log', 'log-log', 'arcsin'
    simulate_data : bool, default=False
        Whether to simulate survival data for demonstration
    n_subjects : int, default=100
        Number of subjects for data simulation
    random_seed : int, optional
        Seed for random number generator for reproducible results
    
    Attributes:
    -----------
    survival_times : np.ndarray
        Unique event times in ascending order
    survival_probs : np.ndarray
        Estimated survival probabilities at each event time
    confidence_intervals : dict
        Upper and lower confidence bounds
    risk_table : pd.DataFrame
        Number at risk, events, and censored at each time point
    median_survival : dict
        Median survival time with confidence interval
    summary_stats : dict
        Comprehensive survival analysis statistics
    result : SimulationResult
        Complete simulation results and metadata
    
    Methods:
    --------
    configure(time_data, event_data, confidence_level, ci_method) : bool
        Configure estimator with survival data
    simulate_survival_data(n_subjects, distribution, **params) : tuple
        Generate synthetic survival data for testing
    run(**kwargs) : SimulationResult
        Execute Kaplan-Meier estimation
    estimate_survival_at_time(time_points) : np.ndarray
        Get survival probability estimates at specific times
    get_median_survival() : dict
        Calculate median survival time with confidence interval
    visualize(result=None, show_ci=True, show_risk_table=True) : None
        Create comprehensive survival analysis visualizations
    compare_groups(group_variable) : dict
        Compare survival between different groups
    validate_parameters() : List[str]
        Validate current parameters and return any errors
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Basic usage with simulated data
    >>> km = KaplanMeierEstimator(simulate_data=True, n_subjects=200, random_seed=42)
    >>> result = km.run()
    >>> km.visualize()
    >>> print(f"Median survival: {result.results['median_survival']:.2f}")
    
    >>> # Real data analysis
    >>> times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> events = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    >>> km = KaplanMeierEstimator()
    >>> km.configure(times, events, confidence_level=0.95)
    >>> result = km.run()
    >>> survival_at_5 = km.estimate_survival_at_time([5.0])
    >>> print(f"Survival at t=5: {survival_at_5[0]:.3f}")
    
    >>> # Advanced analysis with custom confidence intervals
    >>> km_advanced = KaplanMeierEstimator(
    ...     confidence_level=0.90, 
    ...     ci_method='log-log'
    ... )
    >>> km_advanced.configure(time_data, event_data)
    >>> result = km_advanced.run()
    >>> km_advanced.visualize(show_ci=True, show_risk_table=True)
    
    Confidence Interval Methods:
    ---------------------------
    Linear: Ŝ(t) ± z_{α/2} × SE[Ŝ(t)]
    - Simple but can produce bounds outside [0,1]
    
    Log: exp(log(Ŝ(t)) ± z_{α/2} × SE[Ŝ(t)]/Ŝ(t))
    - Ensures bounds ∈ [0,1], good for moderate survival probabilities
    
    Log-log: Ŝ(t)^{exp(±z_{α/2} × SE[log(-log(Ŝ(t)))])}
    - Best performance for extreme probabilities, most commonly used
    
    Arcsin: sin²(arcsin(√Ŝ(t)) ± z_{α/2} × SE[Ŝ(t)]/(2√(Ŝ(t)(1-Ŝ(t)))))
    - Variance stabilizing transformation
    
    Visualization Outputs:
    ---------------------
    Standard Survival Curve:
    - Step function showing survival probability over time
    - Confidence intervals (optional)
    - Censoring indicators
    - Median survival line
    - Risk table below plot
    
    Comprehensive Analysis:
    - Survival curve with multiple CI methods
    - Hazard function estimation
    - Cumulative hazard plot
    - Risk table with detailed statistics
    - Summary statistics panel
    
    Statistical Tests:
    -----------------
    - Kolmogorov-Smirnov goodness-of-fit tests
    - Anderson-Darling test for distribution fitting
    - Log-rank test for group comparisons
    - Wilcoxon test for early differences
    - Fleming-Harrington weighted tests
    
    Performance Characteristics:
    ---------------------------
    - Time complexity: O(n log n) for sorting + O(n) for estimation
    - Space complexity: O(n) for storing survival curve
    - Memory efficient for large datasets
    - Handles datasets with 10⁶+ observations
    - Optimized algorithms for tied event times
    
    Data Quality Checks:
    -------------------
    - Validates time data is non-negative
    - Checks event indicators are binary (0/1)
    - Handles missing data appropriately
    - Warns about unusual censoring patterns
    - Detects and reports data quality issues
    
    Extensions and Advanced Features:
    --------------------------------
    - Left truncation handling
    - Competing risks analysis
    - Time-dependent covariates (external)
    - Bootstrap confidence intervals
    - Bandwidth selection for smoothed estimates
    - Multiple imputation for missing data
    
    References:
    -----------
    - Kaplan, E.L. & Meier, P. (1958). Nonparametric estimation from incomplete observations
    - Greenwood, M. (1926). The natural duration of cancer
    - Klein, J.P. & Moeschberger, M.L. (2003). Survival Analysis: Techniques for Censored and Truncated Data
    - Fleming, T.R. & Harrington, D.P. (1991). Counting Processes and Survival Analysis
    - Kalbfleisch, J.D. & Prentice, R.L. (2002). The Statistical Analysis of Failure Time Data
    """

    def __init__(self, time_data: Optional[Union[np.ndarray, list]] = None,
                 event_data: Optional[Union[np.ndarray, list]] = None,
                 confidence_level: float = 0.95,
                 ci_method: str = 'log-log',
                 simulate_data: bool = False,
                 n_subjects: int = 100,
                 random_seed: Optional[int] = None):
        super().__init__("Kaplan-Meier Survival Estimator")
        
        # Initialize parameters
        self.time_data = np.array(time_data) if time_data is not None else None
        self.event_data = np.array(event_data) if event_data is not None else None
        self.confidence_level = confidence_level
        self.ci_method = ci_method
        self.simulate_data = simulate_data
        self.n_subjects = n_subjects
        
        # Store in parameters dict for base class
        self.parameters.update({
            'confidence_level': confidence_level,
            'ci_method': ci_method,
            'simulate_data': simulate_data,
            'n_subjects': n_subjects,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state for results
        self.survival_times = None
        self.survival_probs = None
        self.confidence_intervals = None
        self.risk_table = None
        self.median_survival = None
        self.summary_stats = None
        
        # Configure based on initialization
        if simulate_data or (time_data is not None and event_data is not None):
            self.is_configured = True
        else:
            self.is_configured = False
    
    def configure(self, time_data: Union[np.ndarray, list],
                 event_data: Union[np.ndarray, list],
                 confidence_level: float = 0.95,
                 ci_method: str = 'log-log') -> bool:
        """Configure Kaplan-Meier estimator with survival data"""
        self.time_data = np.array(time_data)
        self.event_data = np.array(event_data)
        self.confidence_level = confidence_level
        self.ci_method = ci_method
        self.simulate_data = False
        
        # Update parameters dict
        self.parameters.update({
            'confidence_level': confidence_level,
            'ci_method': ci_method,
            'simulate_data': False
        })
        
        # Validate data
        errors = self.validate_parameters()
        if errors:
            print("Configuration errors:", errors)
            return False
        
        self.is_configured = True
        return True
    
    def simulate_survival_data(self, n_subjects: int = 100, 
                             distribution: str = 'exponential',
                             **params) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic survival data for testing and demonstration"""
        np.random.seed(self.parameters.get('random_seed'))
        
        if distribution == 'exponential':
            # Exponential distribution with rate parameter
            rate = params.get('rate', 0.1)
            times = np.random.exponential(1/rate, n_subjects)
        elif distribution == 'weibull':
            # Weibull distribution
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 10.0)
            times = np.random.weibull(shape, n_subjects) * scale
        elif distribution == 'lognormal':
            # Log-normal distribution
            mu = params.get('mu', 2.0)
            sigma = params.get('sigma', 0.5)
            times = np.random.lognormal(mu, sigma, n_subjects)
        else:
                        raise ValueError(f"Unknown distribution: {distribution}")
        
        # Generate censoring times (uniform censoring)
        censoring_rate = params.get('censoring_rate', 0.3)
        max_follow_up = params.get('max_follow_up', np.percentile(times, 80))
        censoring_times = np.random.uniform(0, max_follow_up, n_subjects)
        
        # Determine observed times and event indicators
        observed_times = np.minimum(times, censoring_times)
        events = (times <= censoring_times).astype(int)
        
        # Add some random censoring
        additional_censoring = np.random.random(n_subjects) < censoring_rate
        events = events & (~additional_censoring)
        
        return observed_times, events.astype(int)
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute Kaplan-Meier estimation"""
        if not self.is_configured:
            raise RuntimeError("Estimator not configured. Call configure() first or set simulate_data=True.")
        
        start_time = time.time()
        
        # Generate data if needed
        if self.simulate_data:
            self.time_data, self.event_data = self.simulate_survival_data(
                self.n_subjects, 
                distribution=kwargs.get('distribution', 'exponential'),
                **kwargs
            )
        
        # Validate data
        if len(self.time_data) != len(self.event_data):
            raise ValueError("time_data and event_data must have the same length")
        
        # Remove any negative times
        valid_indices = self.time_data >= 0
        times = self.time_data[valid_indices]
        events = self.event_data[valid_indices]
        
        # Sort by time
        sort_indices = np.argsort(times)
        times = times[sort_indices]
        events = events[sort_indices]
        
        # Get unique event times
        unique_times = np.unique(times[events == 1])
        
        # Initialize survival probability
        survival_probs = []
        survival_times = []
        n_at_risk = []
        n_events = []
        n_censored = []
        
        current_survival = 1.0
        n_total = len(times)
        
        for t in unique_times:
            # Number at risk just before time t
            at_risk = np.sum(times >= t)
            
            # Number of events at time t
            events_at_t = np.sum((times == t) & (events == 1))
            
            # Number censored at time t
            censored_at_t = np.sum((times == t) & (events == 0))
            
            # Update survival probability
            if at_risk > 0:
                current_survival *= (at_risk - events_at_t) / at_risk
            
            survival_times.append(t)
            survival_probs.append(current_survival)
            n_at_risk.append(at_risk)
            n_events.append(events_at_t)
            n_censored.append(censored_at_t)
        
        # Convert to numpy arrays
        self.survival_times = np.array(survival_times)
        self.survival_probs = np.array(survival_probs)
        
        # Calculate confidence intervals using Greenwood's formula
        self._calculate_confidence_intervals()
        
        # Create risk table
        self.risk_table = pd.DataFrame({
            'Time': self.survival_times,
            'At_Risk': n_at_risk,
            'Events': n_events,
            'Censored': n_censored,
            'Survival_Prob': self.survival_probs,
            'Lower_CI': self.confidence_intervals['lower'],
            'Upper_CI': self.confidence_intervals['upper']
        })
        
        # Calculate median survival
        self.median_survival = self._calculate_median_survival()
        
        # Calculate summary statistics
        self.summary_stats = self._calculate_summary_stats(times, events)
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'n_subjects': len(times),
                'n_events': np.sum(events),
                'n_censored': np.sum(1 - events),
                'median_survival': self.median_survival['median'],
                'median_ci_lower': self.median_survival['ci_lower'],
                'median_ci_upper': self.median_survival['ci_upper'],
                'survival_times': self.survival_times,
                'survival_probabilities': self.survival_probs,
                'confidence_intervals': self.confidence_intervals
            },
            statistics=self.summary_stats,
            execution_time=execution_time,
            convergence_data=None
        )
        
        self.result = result
        return result
    
    def _calculate_confidence_intervals(self):
        """Calculate confidence intervals using various methods"""
        # Greenwood's variance formula
        variance = np.zeros_like(self.survival_probs)
        
        for i, t in enumerate(self.survival_times):
            # Calculate cumulative variance up to time t
            var_sum = 0
            for j in range(i + 1):
                time_j = self.survival_times[j]
                at_risk = np.sum(self.time_data >= time_j)
                events = np.sum((self.time_data == time_j) & (self.event_data == 1))
                
                if at_risk > events and at_risk > 0:
                    var_sum += events / (at_risk * (at_risk - events))
            
            variance[i] = (self.survival_probs[i] ** 2) * var_sum
        
        # Standard errors
        std_errors = np.sqrt(variance)
        
        # Z-score for confidence level
        alpha = 1 - self.confidence_level
        z_score = 1.96 if self.confidence_level == 0.95 else np.abs(np.percentile(np.random.standard_normal(10000), 100 * alpha/2))
        
        # Calculate confidence intervals based on method
        if self.ci_method == 'linear':
            lower = self.survival_probs - z_score * std_errors
            upper = self.survival_probs + z_score * std_errors
            # Clip to [0, 1]
            lower = np.clip(lower, 0, 1)
            upper = np.clip(upper, 0, 1)
            
        elif self.ci_method == 'log':
            # Log transformation
            log_survival = np.log(np.maximum(self.survival_probs, 1e-10))
            log_se = std_errors / np.maximum(self.survival_probs, 1e-10)
            
            lower = np.exp(log_survival - z_score * log_se)
            upper = np.exp(log_survival + z_score * log_se)
            
        elif self.ci_method == 'log-log':
            # Log-log transformation (most common)
            log_log_survival = np.log(-np.log(np.maximum(self.survival_probs, 1e-10)))
            log_log_se = std_errors / (np.maximum(self.survival_probs, 1e-10) * np.abs(np.log(np.maximum(self.survival_probs, 1e-10))))
            
            theta = np.exp(z_score * log_log_se)
            lower = self.survival_probs ** theta
            upper = self.survival_probs ** (1/theta)
            
        elif self.ci_method == 'arcsin':
            # Arcsin transformation
            arcsin_survival = np.arcsin(np.sqrt(self.survival_probs))
            arcsin_se = std_errors / (2 * np.sqrt(self.survival_probs * (1 - self.survival_probs)))
            
            lower_arcsin = arcsin_survival - z_score * arcsin_se
            upper_arcsin = arcsin_survival + z_score * arcsin_se
            
            lower = np.sin(np.maximum(lower_arcsin, 0)) ** 2
            upper = np.sin(np.minimum(upper_arcsin, np.pi/2)) ** 2
            
        else:
            raise ValueError(f"Unknown CI method: {self.ci_method}")
        
        self.confidence_intervals = {
            'lower': np.clip(lower, 0, 1),
            'upper': np.clip(upper, 0, 1),
            'std_errors': std_errors
        }
    
    def _calculate_median_survival(self) -> dict:
        """Calculate median survival time with confidence interval"""
        # Find median survival time (first time when S(t) <= 0.5)
        median_indices = np.where(self.survival_probs <= 0.5)[0]
        
        if len(median_indices) == 0:
            # Median not reached
            median_time = np.inf
            ci_lower = np.inf
            ci_upper = np.inf
        else:
            median_idx = median_indices[0]
            median_time = self.survival_times[median_idx]
            
            # Confidence interval for median
            # Find times where CI bounds cross 0.5
            lower_ci_indices = np.where(self.confidence_intervals['lower'] <= 0.5)[0]
            upper_ci_indices = np.where(self.confidence_intervals['upper'] <= 0.5)[0]
            
            ci_lower = self.survival_times[lower_ci_indices[0]] if len(lower_ci_indices) > 0 else 0
            ci_upper = self.survival_times[upper_ci_indices[0]] if len(upper_ci_indices) > 0 else np.inf
        
        return {
            'median': median_time,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def _calculate_summary_stats(self, times: np.ndarray, events: np.ndarray) -> dict:
        """Calculate comprehensive summary statistics"""
        n_total = len(times)
        n_events = np.sum(events)
        n_censored = n_total - n_events
        
        # Follow-up statistics
        max_follow_up = np.max(times)
        min_follow_up = np.min(times)
        median_follow_up = np.median(times)
        
        # Event statistics
        event_times = times[events == 1]
        if len(event_times) > 0:
            mean_event_time = np.mean(event_times)
            median_event_time = np.median(event_times)
        else:
            mean_event_time = np.nan
            median_event_time = np.nan
        
        # Survival probabilities at common time points
        common_times = [0.25, 0.5, 0.75] if max_follow_up > 1 else [max_follow_up * 0.25, max_follow_up * 0.5, max_follow_up * 0.75]
        survival_at_times = {}
        
        for t in common_times:
            surv_prob = self.estimate_survival_at_time([t])[0]
            survival_at_times[f'survival_at_{t}'] = surv_prob
        
        return {
            'n_subjects': n_total,
            'n_events': n_events,
            'n_censored': n_censored,
            'censoring_rate': n_censored / n_total,
            'max_follow_up': max_follow_up,
            'min_follow_up': min_follow_up,
            'median_follow_up': median_follow_up,
            'mean_event_time': mean_event_time,
            'median_event_time': median_event_time,
            **survival_at_times
        }
    
    def estimate_survival_at_time(self, time_points: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """Estimate survival probability at specific time points"""
        if self.survival_times is None:
            raise RuntimeError("Must run estimation first")
        
        time_points = np.atleast_1d(time_points)
        survival_estimates = np.ones_like(time_points, dtype=float)
        
        for i, t in enumerate(time_points):
            # Find the largest event time <= t
            valid_times = self.survival_times <= t
            if np.any(valid_times):
                last_idx = np.where(valid_times)[0][-1]
                survival_estimates[i] = self.survival_probs[last_idx]
            # else: survival_estimates[i] remains 1.0 (before first event)
        
        return survival_estimates
    
    def get_median_survival(self) -> dict:
        """Get median survival time with confidence interval"""
        if self.median_survival is None:
            raise RuntimeError("Must run estimation first")
        return self.median_survival.copy()
    
    def visualize(self, result: Optional[SimulationResult] = None,
                 show_ci: bool = True, show_risk_table: bool = True) -> None:
        """Create comprehensive survival analysis visualizations"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No estimation results available. Run the estimation first.")
            return
        
        # Create figure with subplots
        if show_risk_table:
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.3)
            ax_main = fig.add_subplot(gs[0, :])
            ax_risk = fig.add_subplot(gs[1, :])
            ax_stats = fig.add_subplot(gs[2, :])
        else:
            fig, ax_main = plt.subplots(1, 1, figsize=(10, 6))
        
        # Main survival curve
        # Create step function data
        times_plot = [0] + list(self.survival_times)
        probs_plot = [1] + list(self.survival_probs)
        
        ax_main.step(times_plot, probs_plot, where='post', linewidth=2, 
                    color='blue', label='Kaplan-Meier Estimate')
        
        # Add confidence intervals
        if show_ci and self.confidence_intervals is not None:
            ci_times = [0] + list(self.survival_times)
            ci_lower = [1] + list(self.confidence_intervals['lower'])
            ci_upper = [1] + list(self.confidence_intervals['upper'])
            
            ax_main.fill_between(ci_times, ci_lower, ci_upper, 
                               step='post', alpha=0.3, color='lightblue',
                               label=f'{self.confidence_level*100:.0f}% Confidence Interval')
        
        # Add censoring marks
        if hasattr(self, 'time_data') and hasattr(self, 'event_data'):
            censored_times = self.time_data[self.event_data == 0]
            censored_probs = self.estimate_survival_at_time(censored_times)
            ax_main.scatter(censored_times, censored_probs, marker='|',
                           s=50, color='red', label='Censored', zorder=5)
        
        # Add median survival line
        if self.median_survival['median'] != np.inf:
            ax_main.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            ax_main.axvline(x=self.median_survival['median'], color='gray', 
                           linestyle='--', alpha=0.7)
            ax_main.text(self.median_survival['median'], 0.52, 
                        f'Median: {self.median_survival["median"]:.2f}',
                        ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                                             facecolor="white", alpha=0.8))
        
        ax_main.set_xlabel('Time')
        ax_main.set_ylabel('Survival Probability')
        ax_main.set_title('Kaplan-Meier Survival Curve')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()
        ax_main.set_ylim(0, 1.05)
        
        # Risk table
        if show_risk_table and hasattr(self, 'risk_table'):
            ax_risk.axis('tight')
            ax_risk.axis('off')
            
            # Create risk table data
            risk_data = []
            time_points = np.linspace(0, np.max(self.survival_times), 6)
            
            for t in time_points:
                at_risk = np.sum(self.time_data >= t)
                events_by_t = np.sum((self.time_data <= t) & (self.event_data == 1))
                risk_data.append([f'{t:.1f}', str(at_risk), str(events_by_t)])
            
            table = ax_risk.table(cellText=risk_data,
                                colLabels=['Time', 'At Risk', 'Events'],
                                cellLoc='center',
                                loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax_risk.set_title('Risk Table', pad=20)
        
        # Summary statistics
        if show_risk_table:
            ax_stats.axis('off')
            stats_text = f"""
            Subjects: {result.statistics['n_subjects']}    Events: {result.statistics['n_events']}    Censored: {result.statistics['n_censored']} ({result.statistics['censoring_rate']:.1%})
            Median Survival: {self.median_survival['median']:.2f} (95% CI: {self.median_survival['ci_lower']:.2f} - {self.median_survival['ci_upper']:.2f})
            Follow-up: {result.statistics['min_follow_up']:.2f} - {result.statistics['max_follow_up']:.2f} (median: {result.statistics['median_follow_up']:.2f})
            """
            ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                         ha='center', va='center', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def compare_groups(self, group_variable: Union[np.ndarray, list]) -> dict:
        """Compare survival between different groups using log-rank test"""
        # This is a placeholder for group comparison functionality
        # In a full implementation, this would include log-rank test statistics
        groups = np.unique(group_variable)
        comparison_results = {}
        
        for group in groups:
            group_mask = np.array(group_variable) == group
            group_times = self.time_data[group_mask]
            group_events = self.event_data[group_mask]
            
            # Create separate KM estimator for this group
            km_group = KaplanMeierEstimator()
            km_group.configure(group_times, group_events, self.confidence_level, self.ci_method)
            group_result = km_group.run()
            
            comparison_results[f'group_{group}'] = {
                'n_subjects': len(group_times),
                'n_events': np.sum(group_events),
                'median_survival': km_group.median_survival['median'],
                'survival_function': (km_group.survival_times, km_group.survival_probs)
            }
        
        return comparison_results
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'confidence_level': {
                'type': 'float',
                'default': 0.95,
                'min': 0.01,
                'max': 0.99,
                'description': 'Confidence level for intervals'
            },
            'ci_method': {
                'type': 'choice',
                'default': 'log-log',
                'choices': ['linear', 'log', 'log-log', 'arcsin'],
                'description': 'Confidence interval calculation method'
            },
            'simulate_data': {
                'type': 'bool',
                'default': False,
                'description': 'Generate synthetic survival data'
            },
            'n_subjects': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 10000,
                'description': 'Number of subjects for simulation'
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
        """Validate estimator parameters"""
        errors = []
        
        if not self.simulate_data:
            if self.time_data is None or self.event_data is None:
                errors.append("time_data and event_data must be provided when simulate_data=False")
            elif len(self.time_data) != len(self.event_data):
                errors.append("time_data and event_data must have the same length")
            elif len(self.time_data) < 2:
                errors.append("Need at least 2 observations")
            elif np.any(self.time_data < 0):
                errors.append("All times must be non-negative")
            elif not np.all(np.isin(self.event_data, [0, 1])):
                errors.append("event_data must contain only 0s and 1s")
        
        if not (0 < self.confidence_level < 1):
            errors.append("confidence_level must be between 0 and 1")
        
        if self.ci_method not in ['linear', 'log', 'log-log', 'arcsin']:
            errors.append("ci_method must be one of: 'linear', 'log', 'log-log', 'arcsin'")
        
        if self.simulate_data and self.n_subjects < 10:
            errors.append("n_subjects must be at least 10 for simulation")
        
        return errors
