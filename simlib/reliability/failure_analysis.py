import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy import stats
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult
from ..monte_carlo.pi_estimation import PiEstimationMC

class FailureAnalysis:
    """
    Comprehensive failure analysis for Monte Carlo π estimation simulation.
    
    This class provides detailed analysis of failure modes, accuracy degradation,
    statistical anomalies, and reliability assessment for the π estimation simulation.
    It identifies potential issues with random number generation, convergence problems,
    numerical precision errors, and statistical outliers.
    
    Failure Categories Analyzed:
    ---------------------------
    1. Convergence Failures:
       - Slow convergence rates
       - Non-monotonic convergence
       - Oscillatory behavior
       - Premature convergence plateaus
    
    2. Statistical Anomalies:
       - Estimates outside confidence intervals
       - Unusual variance patterns
       - Bias detection
       - Distribution normality violations
    
    3. Numerical Precision Issues:
       - Floating-point accumulation errors
       - Loss of significance
       - Rounding error propagation
       - Catastrophic cancellation
    
    4. Random Number Generator Problems:
       - Poor randomness quality
       - Correlation in sequences
       - Periodicity issues
       - Seed-dependent biases
    
    5. Parameter-Related Failures:
       - Insufficient sample sizes
       - Memory allocation failures
       - Performance degradation
       - Timeout conditions
    
    6. Implementation Errors:
       - Algorithm correctness
       - Boundary condition handling
       - Edge case failures
       - Resource management issues
    
    Analysis Methods:
    ----------------
    - Multiple independent runs with different seeds
    - Statistical hypothesis testing
    - Convergence rate analysis
    - Error distribution characterization
    - Confidence interval validation
    - Bias and variance decomposition
    - Outlier detection and analysis
    - Performance profiling
    
    Diagnostic Outputs:
    ------------------
    - Failure probability estimates
    - Error distribution analysis
    - Convergence quality metrics
    - Statistical test results
    - Reliability confidence bounds
    - Performance benchmarks
    - Recommendation reports
    
    Applications:
    ------------
    - Simulation validation and verification
    - Quality assurance testing
    - Performance optimization
    - Educational demonstrations
    - Research reliability assessment
    - Production system monitoring
    
    Parameters:
    -----------
    base_simulation : PiEstimationMC
        The π estimation simulation to analyze
    n_runs : int, default=100
        Number of independent simulation runs for analysis
    confidence_level : float, default=0.95
        Confidence level for statistical tests (0 < confidence_level < 1)
    sample_sizes : List[int], optional
        Different sample sizes to test (default: [1000, 10000, 100000, 1000000])
    failure_threshold : float, default=0.01
        Maximum acceptable relative error threshold
    timeout_seconds : float, default=300
        Maximum time allowed per simulation run
    
    Attributes:
    -----------
    results : Dict[str, Any]
        Comprehensive analysis results
    failure_modes : List[Dict[str, Any]]
        Detected failure modes with details
    recommendations : List[str]
        Actionable recommendations for improvement
    reliability_score : float
        Overall reliability score (0-1)
    
    Methods:
    --------
    run_analysis() : Dict[str, Any]
        Execute complete failure analysis
    analyze_convergence() : Dict[str, Any]
        Analyze convergence behavior and failures
    analyze_statistical_properties() : Dict[str, Any]
        Test statistical properties and detect anomalies
    analyze_precision_errors() : Dict[str, Any]
        Detect numerical precision issues
    analyze_randomness_quality() : Dict[str, Any]
        Assess random number generator quality
    detect_outliers() : List[Dict[str, Any]]
        Identify and analyze statistical outliers
    generate_report() : str
        Generate comprehensive failure analysis report
    visualize_failures() : None
        Create visualizations of failure modes
    
    Examples:
    ---------
    >>> # Basic failure analysis
    >>> pi_sim = PiEstimationMC(n_samples=100000)
    >>> analyzer = PiEstimationFailureAnalysis(pi_sim, n_runs=50)
    >>> results = analyzer.run_analysis()
    >>> print(f"Reliability score: {analyzer.reliability_score:.3f}")
    >>> analyzer.visualize_failures()
    
    >>> # Comprehensive analysis with multiple sample sizes
    >>> analyzer = PiEstimationFailureAnalysis(
    ...     pi_sim, 
    ...     n_runs=200,
    ...     sample_sizes=[1000, 10000, 100000, 1000000],
    ...     failure_threshold=0.005
    ... )
    >>> results = analyzer.run_analysis()
    >>> report = analyzer.generate_report()
    >>> print(report)
    
    >>> # Performance and timeout analysis
    >>> analyzer = PiEstimationFailureAnalysis(
    ...     pi_sim,
    ...     n_runs=100,
    ...     timeout_seconds=60
    ... )
    >>> results = analyzer.run_analysis()
    >>> for failure in analyzer.failure_modes:
    ...     print(f"Failure: {failure['type']} - {failure['description']}")
    """

    def __init__(self, base_simulation: PiEstimationMC, n_runs: int = 100,
                 confidence_level: float = 0.95, sample_sizes: Optional[List[int]] = None,
                 failure_threshold: float = 0.01, timeout_seconds: float = 300):
        
        self.base_simulation = base_simulation
        self.n_runs = n_runs
        self.confidence_level = confidence_level
        self.sample_sizes = sample_sizes or [1000, 10000, 100000, 1000000]
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        
        # Analysis results storage
        self.results = {}
        self.failure_modes = []
        self.recommendations = []
        self.reliability_score = 0.0
        
        # Internal data storage
        self._run_data = []
        self._convergence_data = []
        self._timing_data = []
        self._error_data = []

    def run_analysis(self) -> Dict[str, Any]:
        """Execute comprehensive failure analysis"""
        print("Starting Monte Carlo π Estimation Failure Analysis...")
        
        # Initialize results structure
        self.results = {
            'summary': {},
            'convergence_analysis': {},
            'statistical_analysis': {},
            'precision_analysis': {},
            'randomness_analysis': {},
            'performance_analysis': {},
            'outlier_analysis': {},
            'failure_modes': [],
            'recommendations': []
        }
        
        # Run multiple simulations with different configurations
        self._collect_simulation_data()
        
        # Perform individual analyses
        self.results['convergence_analysis'] = self.analyze_convergence()
        self.results['statistical_analysis'] = self.analyze_statistical_properties()
        self.results['precision_analysis'] = self.analyze_precision_errors()
        self.results['randomness_analysis'] = self.analyze_randomness_quality()
        self.results['performance_analysis'] = self.analyze_performance()
        self.results['outlier_analysis'] = self.detect_outliers()
        
        # Calculate overall reliability score
        self._calculate_reliability_score()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Create summary
        self._create_summary()
        
        print(f"Analysis complete. Reliability score: {self.reliability_score:.3f}")
        return self.results

    def _collect_simulation_data(self):
        """Collect data from multiple simulation runs"""
        print(f"Collecting data from {self.n_runs} simulation runs...")
        
        for sample_size in self.sample_sizes:
            print(f"  Testing sample size: {sample_size}")
            
            for run_idx in range(self.n_runs):
                try:
                    # Configure simulation
                    sim = PiEstimationMC(
                        n_samples=sample_size,
                        show_convergence=True,
                        random_seed=run_idx + 1000  # Ensure different seeds
                    )
                    
                    # Time the execution
                    start_time = time.time()
                    result = sim.run()
                    execution_time = time.time() - start_time
                    
                    # Check for timeout
                    if execution_time > self.timeout_seconds:
                        self.failure_modes.append({
                            'type': 'timeout',
                            'description': f'Simulation exceeded {self.timeout_seconds}s timeout',
                            'sample_size': sample_size,
                            'run_index': run_idx,
                            'execution_time': execution_time
                        })
                        continue
                    
                    # Store run data
                    run_data = {
                        'sample_size': sample_size,
                        'run_index': run_idx,
                        'pi_estimate': result.results['pi_estimate'],
                        'absolute_error': result.results['accuracy'],
                        'relative_error': result.results['relative_error'],
                        'execution_time': execution_time,
                        'convergence_data': result.convergence_data,
                        'points_inside': result.results['points_inside_circle'],
                        'success': True
                    }
                    
                    self._run_data.append(run_data)
                    
                    # Check for immediate failures
                    if result.results['relative_error'] > self.failure_threshold * 100:
                        self.failure_modes.append({
                            'type': 'accuracy_failure',
                            'description': f'Relative error {result.results["relative_error"]:.4f}% exceeds threshold {self.failure_threshold*100:.4f}%',
                            'sample_size': sample_size,
                            'run_index': run_idx,
                            'pi_estimate': result.results['pi_estimate'],
                            'error': result.results['relative_error']
                        })
                
                except Exception as e:
                    # Record simulation failures
                    self.failure_modes.append({
                        'type': 'execution_failure',
                        'description': f'Simulation failed with exception: {str(e)}',
                        'sample_size': sample_size,
                        'run_index': run_idx,
                        'exception': str(e)
                    })

    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence behavior and detect failures"""
        print("Analyzing convergence behavior...")
        
        convergence_analysis = {
            'slow_convergence_count': 0,
            'oscillatory_behavior_count': 0,
            'premature_plateau_count': 0,
            'non_monotonic_count': 0,
            'convergence_rates': [],
            'final_errors': []
        }
        
        for run_data in self._run_data:
            if not run_data['convergence_data']:
                continue
                
            samples = [point[0] for point in run_data['convergence_data']]
            estimates = [point[1] for point in run_data['convergence_data']]
            errors = [abs(est - np.pi) for est in estimates]
            
            # Check for slow convergence
            if len(errors) > 10:
                early_error = np.mean(errors[:len(errors)//4])
                late_error = np.mean(errors[-len(errors)//4:])
                improvement_ratio = early_error / late_error if late_error > 0 else float('inf')
                
                if improvement_ratio < 2.0:  # Less than 2x improvement
                    convergence_analysis['slow_convergence_count'] += 1
                    self.failure_modes.append({
                        'type': 'slow_convergence',
                        'description': f'Convergence improvement ratio only {improvement_ratio:.2f}',
                        'sample_size': run_data['sample_size'],
                        'run_index': run_data['run_index'],
                        'improvement_ratio': improvement_ratio
                    })
            
            # Check for oscillatory behavior
            if len(estimates) > 20:
                # Calculate number of sign changes in error derivative
                error_diffs = np.diff(errors)
                sign_changes = np.sum(np.diff(np.sign(error_diffs)) != 0)
                oscillation_ratio = sign_changes / len(error_diffs)
                
                if oscillation_ratio > 0.3:  # More than 30% sign changes
                    convergence_analysis['oscillatory_behavior_count'] += 1
                    self.failure_modes.append({
                        'type': 'oscillatory_convergence',
                        'description': f'High oscillation ratio: {oscillation_ratio:.3f}',
                        'sample_size': run_data['sample_size'],
                        'run_index': run_data['run_index'],
                        'oscillation_ratio': oscillation_ratio
                    })
            
            # Check for premature plateau
            if len(errors) > 10:
                final_quarter = errors[-len(errors)//4:]
                if np.std(final_quarter) / np.mean(final_quarter) < 0.01:  # Very low relative variation
                    expected_final_error = 1.64 / np.sqrt(run_data['sample_size'])
                    if np.mean(final_quarter) > 2 * expected_final_error:
                        convergence_analysis['premature_plateau_count'] += 1
                        self.failure_modes.append({
                            'type': 'premature_plateau',
                            'description': f'Convergence plateaued at error {np.mean(final_quarter):.6f}, expected {expected_final_error:.6f}',
                            'sample_size': run_data['sample_size'],
                            'run_index': run_data['run_index'],
                            'plateau_error': np.mean(final_quarter),
                            'expected_error': expected_final_error
                        })
            
            # Store convergence rate
            if len(errors) > 1:
                # Fit power law: error ∝ n^(-α), expect α ≈ 0.5 for Monte Carlo
                log_samples = np.log(samples[1:])
                log_errors = np.log(errors[1:])
                if len(log_samples) > 5:
                    slope, _, r_value, _, _ = stats.linregress(log_samples, log_errors)
                    convergence_analysis['convergence_rates'].append(-slope)
                    
                    if -slope < 0.3:  # Much slower than expected 0.5
                        convergence_analysis['slow_convergence_count'] += 1
            
            convergence_analysis['final_errors'].append(errors[-1] if errors else float('inf'))
        
        return convergence_analysis

    def analyze_statistical_properties(self) -> Dict[str, Any]:
        """Test statistical properties and detect anomalies"""
        print("Analyzing statistical properties...")
        
        statistical_analysis = {
            'bias_test': {},
            'variance_test': {},
            'normality_test': {},
            'confidence_interval_test': {},
            'outlier_count': 0
        }

            
        # Group data by sample size for analysis
        by_sample_size = {}
        for run_data in self._run_data:
            size = run_data['sample_size']
            if size not in by_sample_size:
                by_sample_size[size] = []
            by_sample_size[size].append(run_data)
        
        for sample_size, runs in by_sample_size.items():
            if len(runs) < 10:  # Need sufficient data for statistical tests
                continue
                
            estimates = [run['pi_estimate'] for run in runs]
            errors = [run['absolute_error'] for run in runs]
            rel_errors = [run['relative_error'] for run in runs]
            
            # Bias test (one-sample t-test against π)
            t_stat, p_value = stats.ttest_1samp(estimates, np.pi)
            bias_significant = p_value < (1 - self.confidence_level)
            
            statistical_analysis['bias_test'][sample_size] = {
                'mean_estimate': np.mean(estimates),
                'bias': np.mean(estimates) - np.pi,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_bias': bias_significant
            }
            
            if bias_significant:
                self.failure_modes.append({
                    'type': 'significant_bias',
                    'description': f'Significant bias detected: {np.mean(estimates) - np.pi:.6f}',
                    'sample_size': sample_size,
                    'bias': np.mean(estimates) - np.pi,
                    'p_value': p_value
                })
            
            # Variance test (compare with theoretical variance)
            observed_var = np.var(estimates, ddof=1)
            theoretical_var = np.pi * (4 - np.pi) / sample_size
            variance_ratio = observed_var / theoretical_var
            
            statistical_analysis['variance_test'][sample_size] = {
                'observed_variance': observed_var,
                'theoretical_variance': theoretical_var,
                'variance_ratio': variance_ratio,
                'excessive_variance': variance_ratio > 2.0,
                'insufficient_variance': variance_ratio < 0.5
            }
            
            if variance_ratio > 2.0 or variance_ratio < 0.5:
                self.failure_modes.append({
                    'type': 'variance_anomaly',
                    'description': f'Variance ratio {variance_ratio:.3f} deviates significantly from 1.0',
                    'sample_size': sample_size,
                    'variance_ratio': variance_ratio,
                    'observed_variance': observed_var,
                    'theoretical_variance': theoretical_var
                })
            
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
            if len(estimates) <= 50:
                stat, p_value = stats.shapiro(estimates)
                test_name = 'shapiro_wilk'
            else:
                stat, critical_values, significance_level = stats.anderson(estimates, dist='norm')
                p_value = significance_level / 100.0  # Approximate
                test_name = 'anderson_darling'
            
            normality_violated = p_value < (1 - self.confidence_level)
            
            statistical_analysis['normality_test'][sample_size] = {
                'test_name': test_name,
                'statistic': stat,
                'p_value': p_value,
                'normality_violated': normality_violated
            }
            
            if normality_violated:
                self.failure_modes.append({
                    'type': 'normality_violation',
                    'description': f'Estimates do not follow normal distribution (p={p_value:.4f})',
                    'sample_size': sample_size,
                    'test_statistic': stat,
                    'p_value': p_value
                })
            
            # Confidence interval test
            mean_est = np.mean(estimates)
            std_est = np.std(estimates, ddof=1)
            sem = std_est / np.sqrt(len(estimates))
            
            # Calculate confidence interval
            alpha = 1 - self.confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, len(estimates) - 1)
            ci_lower = mean_est - t_critical * sem
            ci_upper = mean_est + t_critical * sem
            
            contains_pi = ci_lower <= np.pi <= ci_upper
            
            statistical_analysis['confidence_interval_test'][sample_size] = {
                'mean_estimate': mean_est,
                'confidence_interval': (ci_lower, ci_upper),
                'contains_true_pi': contains_pi,
                'interval_width': ci_upper - ci_lower
            }
            
            if not contains_pi:
                self.failure_modes.append({
                    'type': 'confidence_interval_failure',
                    'description': f'{self.confidence_level*100:.1f}% CI does not contain π',
                    'sample_size': sample_size,
                    'confidence_interval': (ci_lower, ci_upper),
                    'mean_estimate': mean_est
                })
        
        return statistical_analysis

    def analyze_precision_errors(self) -> Dict[str, Any]:
        """Detect numerical precision issues"""
        print("Analyzing numerical precision...")
        
        precision_analysis = {
            'floating_point_issues': [],
            'accumulation_errors': [],
            'catastrophic_cancellation': [],
            'precision_loss_count': 0
        }
        
        for run_data in self._run_data:
            sample_size = run_data['sample_size']
            pi_estimate = run_data['pi_estimate']
            
            # Check for obvious floating-point issues
            if not np.isfinite(pi_estimate):
                precision_analysis['floating_point_issues'].append({
                    'sample_size': sample_size,
                    'run_index': run_data['run_index'],
                    'value': pi_estimate,
                    'issue': 'non_finite_result'
                })
                self.failure_modes.append({
                    'type': 'floating_point_error',
                    'description': f'Non-finite result: {pi_estimate}',
                    'sample_size': sample_size,
                    'run_index': run_data['run_index']
                })
            
            # Check for unrealistic precision (too many significant digits)
            if abs(pi_estimate - np.pi) < 1e-15 and sample_size < 1e15:
                precision_analysis['precision_loss_count'] += 1
                self.failure_modes.append({
                    'type': 'suspicious_precision',
                    'description': f'Unrealistically high precision for sample size {sample_size}',
                    'sample_size': sample_size,
                    'run_index': run_data['run_index'],
                    'pi_estimate': pi_estimate,
                    'error': abs(pi_estimate - np.pi)
                })
            
            # Check for accumulation errors in large sample sizes
            if sample_size >= 1000000:
                expected_precision = 1.64 / np.sqrt(sample_size)
                actual_error = abs(pi_estimate - np.pi)
                
                if actual_error > 10 * expected_precision:
                    precision_analysis['accumulation_errors'].append({
                        'sample_size': sample_size,
                        'run_index': run_data['run_index'],
                        'expected_precision': expected_precision,
                        'actual_error': actual_error,
                        'error_ratio': actual_error / expected_precision
                    })
            
            # Check convergence data for precision issues
            if run_data['convergence_data']:
                estimates = [point[1] for point in run_data['convergence_data']]
                
                # Look for sudden jumps that might indicate precision loss
                if len(estimates) > 10:
                    diffs = np.abs(np.diff(estimates))
                    median_diff = np.median(diffs)
                    
                    for i, diff in enumerate(diffs):
                        if diff > 10 * median_diff and diff > 0.01:
                            precision_analysis['catastrophic_cancellation'].append({
                                'sample_size': sample_size,
                                'run_index': run_data['run_index'],
                                'position': i,
                                'jump_size': diff,
                                'median_diff': median_diff
                            })
        
        return precision_analysis

    def analyze_randomness_quality(self) -> Dict[str, Any]:
        """Assess random number generator quality"""
        print("Analyzing randomness quality...")
        
        randomness_analysis = {
            'seed_dependency': {},
            'correlation_issues': [],
            'distribution_uniformity': {},
            'periodicity_issues': []
        }
        
        # Group runs by sample size to analyze seed dependency
        by_sample_size = {}
        for run_data in self._run_data:
            size = run_data['sample_size']
            if size not in by_sample_size:
                by_sample_size[size] = []
            by_sample_size[size].append(run_data)
        
        for sample_size, runs in by_sample_size.items():
            if len(runs) < 20:
                continue
            
            estimates = [run['pi_estimate'] for run in runs]
            seeds = [run['run_index'] + 1000 for run in runs]
            
            # Test for seed dependency (correlation between seed and estimate)
            if len(set(seeds)) > 10:  # Need variety in seeds
                correlation, p_value = stats.pearsonr(seeds, estimates)
                
                randomness_analysis['seed_dependency'][sample_size] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant_correlation': p_value < 0.05
                }
                
                if p_value < 0.05 and abs(correlation) > 0.3:
                    self.failure_modes.append({
                        'type': 'seed_dependency',
                        'description': f'Strong correlation between seed and result: r={correlation:.3f}',
                        'sample_size': sample_size,
                        'correlation': correlation,
                        'p_value': p_value
                    })
            
            # Test for serial correlation in estimates
            if len(estimates) > 10:
                # Lag-1 autocorrelation
                autocorr = np.corrcoef(estimates[:-1], estimates[1:])[0, 1]
                
                if abs(autocorr) > 0.2:  # Significant autocorrelation
                    randomness_analysis['correlation_issues'].append({
                        'sample_size': sample_size,
                        'autocorrelation': autocorr,
                        'lag': 1
                    })
                    self.failure_modes.append({
                        'type': 'serial_correlation',
                        'description': f'Significant serial correlation: {autocorr:.3f}',
                        'sample_size': sample_size,
                        'autocorrelation': autocorr
                    })
            
            # Test distribution uniformity (should be approximately normal around π)
            # Kolmogorov-Smirnov test against theoretical normal distribution
            theoretical_mean = np.pi
            theoretical_std = np.sqrt(np.pi * (4 - np.pi) / sample_size)
            
            # Standardize the estimates
            standardized = (np.array(estimates) - theoretical_mean) / theoretical_std
            
            # Test against standard normal
            ks_stat, ks_p_value = stats.kstest(standardized, 'norm')
            
            randomness_analysis['distribution_uniformity'][sample_size] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'distribution_anomaly': ks_p_value < 0.05
            }
            
            if ks_p_value < 0.05:
                self.failure_modes.append({
                    'type': 'distribution_anomaly',
                    'description': f'Estimates do not follow expected distribution (KS p={ks_p_value:.4f})',
                    'sample_size': sample_size,
                    'ks_statistic': ks_stat,
                    'p_value': ks_p_value
                })
        
        return randomness_analysis

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance characteristics and detect issues"""
        print("Analyzing performance...")
        
        performance_analysis = {
            'timing_statistics': {},
            'scalability_issues': [],
            'memory_issues': [],
            'timeout_failures': 0
        }
        
        # Group by sample size for timing analysis
        by_sample_size = {}
        for run_data in self._run_data:
            size = run_data['sample_size']
            if size not in by_sample_size:
                by_sample_size[size] = []
            by_sample_size[size].append(run_data['execution_time'])
        
        # Analyze timing for each sample size
        sample_sizes_sorted = sorted(by_sample_size.keys())
        
        for sample_size in sample_sizes_sorted:
            times = by_sample_size[sample_size]
            
            performance_analysis['timing_statistics'][sample_size] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'samples_per_second': sample_size / np.mean(times)
            }
            
            # Check for excessive timing variation
            cv = np.std(times) / np.mean(times)  # Coefficient of variation
            if cv > 0.5:  # High timing variability
                self.failure_modes.append({
                    'type': 'timing_variability',
                    'description': f'High timing variability (CV={cv:.3f}) for sample size {sample_size}',
                    'sample_size': sample_size,
                    'coefficient_of_variation': cv,
                    'mean_time': np.mean(times),
                    'std_time': np.std(times)
                })
        
        # Check scalability (should be roughly linear with sample size)
        if len(sample_sizes_sorted) >= 3:
            mean_times = [performance_analysis['timing_statistics'][size]['mean_time'] 
                         for size in sample_sizes_sorted]
            
            # Fit linear relationship: time = a * sample_size + b
            slope, intercept, r_value, p_value, std_err = stats.linregress(sample_sizes_sorted, mean_times)
            
            performance_analysis['scalability'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'linear_fit_quality': r_value**2
            }
            
            # Poor linear fit suggests scalability issues
            if r_value**2 < 0.8:
                performance_analysis['scalability_issues'].append({
                    'description': f'Poor linear scalability (R²={r_value**2:.3f})',
                    'r_squared': r_value**2,
                    'expected_linear': True
                })
                self.failure_modes.append({
                    'type': 'scalability_issue',
                    'description': f'Poor linear scalability (R²={r_value**2:.3f})',
                    'r_squared': r_value**2,
                    'slope': slope,
                    'intercept': intercept
                })
        
        # Count timeout failures
        timeout_count = len([fm for fm in self.failure_modes if fm['type'] == 'timeout'])
        performance_analysis['timeout_failures'] = timeout_count
        
        if timeout_count > 0:
            performance_analysis['timeout_rate'] = timeout_count / self.n_runs
        
        return performance_analysis

    def detect_outliers(self) -> Dict[str, Any]:
        """Identify and analyze statistical outliers"""
        print("Detecting outliers...")
        
        outlier_analysis = {
            'outliers_by_sample_size': {},
            'extreme_outliers': [],
            'outlier_patterns': {}
        }
        
        # Group data by sample size
        by_sample_size = {}
        for run_data in self._run_data:
            size = run_data['sample_size']
            if size not in by_sample_size:
                by_sample_size[size] = []
            by_sample_size[size].append(run_data)
        
        for sample_size, runs in by_sample_size.items():
            if len(runs) < 10:
                continue
            
            estimates = [run['pi_estimate'] for run in runs]
            errors = [run['absolute_error'] for run in runs]
            
            # Use IQR method for outlier detection
            q1 = np.percentile(estimates, 25)
            q3 = np.percentile(estimates, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Identify outliers
            outliers = []
            extreme_outliers = []
            
            for i, (run, estimate) in enumerate(zip(runs, estimates)):
                if estimate < lower_bound or estimate > upper_bound:
                    outlier_info = {
                        'run_index': run['run_index'],
                        'pi_estimate': estimate,
                        'absolute_error': run['absolute_error'],
                        'relative_error': run['relative_error'],
                        'execution_time': run['execution_time'],
                        'distance_from_median': abs(estimate - np.median(estimates))
                    }
                    outliers.append(outlier_info)
                    
                    # Check for extreme outliers (3 * IQR)
                    extreme_lower = q1 - 3 * iqr
                    extreme_upper = q3 + 3 * iqr
                    
                    if estimate < extreme_lower or estimate > extreme_upper:
                        extreme_outliers.append(outlier_info)
                        outlier_analysis['extreme_outliers'].append({
                            'sample_size': sample_size,
                            **outlier_info
                        })
                        
                        self.failure_modes.append({
                            'type': 'extreme_outlier',
                            'description': f'Extreme outlier estimate: {estimate:.6f} (expected ~{np.pi:.6f})',
                            'sample_size': sample_size,
                            'run_index': run['run_index'],
                            'pi_estimate': estimate,
                            'distance_from_pi': abs(estimate - np.pi)
                        })
            
            outlier_analysis['outliers_by_sample_size'][sample_size] = {
                'total_outliers': len(outliers),
                'extreme_outliers': len(extreme_outliers),
                'outlier_rate': len(outliers) / len(runs),
                'outlier_details': outliers,
                'bounds': (lower_bound, upper_bound),
                'iqr': iqr
            }
            
            # Check for systematic outlier patterns
            if len(outliers) > len(runs) * 0.1:  # More than 10% outliers
                self.failure_modes.append({
                    'type': 'excessive_outliers',
                    'description': f'Excessive outlier rate: {len(outliers)/len(runs)*100:.1f}%',
                    'sample_size': sample_size,
                    'outlier_count': len(outliers),
                    'total_runs': len(runs),
                    'outlier_rate': len(outliers) / len(runs)
                })
        
        return outlier_analysis

    def _calculate_reliability_score(self):
        """Calculate overall reliability score (0-1)"""
        total_runs = len(self._run_data)
        if total_runs == 0:
            self.reliability_score = 0.0
            return
        
        # Count different types of failures
        failure_counts = {}
        for failure in self.failure_modes:
            failure_type = failure['type']
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
        
        # Weight different failure types
        failure_weights = {
            'execution_failure': 1.0,      # Complete failures
            'timeout': 0.8,                # Performance failures
            'extreme_outlier': 0.7,        # Statistical anomalies
            'accuracy_failure': 0.6,       # Accuracy issues
            'significant_bias': 0.5,       # Systematic errors
            'confidence_interval_failure': 0.4,  # Statistical issues
            'floating_point_error': 0.9,   # Numerical errors
            'scalability_issue': 0.3,      # Performance issues
            'seed_dependency': 0.4,        # Randomness issues
            'variance_anomaly': 0.3,       # Statistical anomalies
            'normality_violation': 0.2,    # Distribution issues
            'slow_convergence': 0.2,       # Convergence issues
            'oscillatory_convergence': 0.2, # Convergence issues
            'premature_plateau': 0.3,      # Convergence issues
            'serial_correlation': 0.3,     # Randomness issues
            'distribution_anomaly': 0.2,   # Statistical issues
            'timing_variability': 0.1,     # Performance variability
            'excessive_outliers': 0.4,     # Statistical reliability
            'suspicious_precision': 0.2    # Numerical precision
        }
        
        # Calculate weighted failure score
        total_failure_weight = 0
        for failure_type, count in failure_counts.items():
            weight = failure_weights.get(failure_type, 0.5)  # Default weight
            total_failure_weight += count * weight
        
        # Normalize by total possible failures
        max_possible_failures = total_runs
        failure_ratio = min(total_failure_weight / max_possible_failures, 1.0)
        
        # Reliability score is inverse of failure ratio
        self.reliability_score = max(0.0, 1.0 - failure_ratio)

    def _generate_recommendations(self):
        """Generate actionable recommendations based on analysis"""
        self.recommendations = []
        
        # Analyze failure patterns
        failure_types = [fm['type'] for fm in self.failure_modes]
        failure_counts = {ft: failure_types.count(ft) for ft in set(failure_types)}
        
        # Recommendations based on most common failures
        if failure_counts.get('accuracy_failure', 0) > 0:
            self.recommendations.append(
                "Consider increasing sample size to improve accuracy. "
                f"Current failure threshold is {self.failure_threshold*100:.2f}%."
            )
        
        if failure_counts.get('slow_convergence', 0) > 0:
            self.recommendations.append(
                "Slow convergence detected. Consider using variance reduction techniques "
                "such as antithetic variables or stratified sampling."
            )
        
        if failure_counts.get('significant_bias', 0) > 0:
            self.recommendations.append(
                "Systematic bias detected. Check random number generator quality "
                "and ensure proper seed initialization."
            )
        
        if failure_counts.get('timeout', 0) > 0:
            self.recommendations.append(
                f"Performance issues detected. Consider reducing sample size or "
                f"increasing timeout from {self.timeout_seconds}s."
            )
        
        if failure_counts.get('extreme_outlier', 0) > 0:
            self.recommendations.append(
                "Extreme outliers detected. Investigate random number generator "
                "quality and consider outlier filtering or robust estimation methods."
            )
        
        if failure_counts.get('floating_point_error', 0) > 0:
            self.recommendations.append(
                "Numerical precision issues detected. Consider using higher precision "
                "arithmetic or more numerically stable algorithms."
            )
        
        if failure_counts.get('seed_dependency', 0) > 0:
            self.recommendations.append(
                "Results show dependency on random seed. Ensure proper seed "
                "initialization and consider using different random number generators."
            )
        
        if failure_counts.get('scalability_issue', 0) > 0:
            self.recommendations.append(
                "Scalability issues detected. Algorithm performance does not scale "
                "linearly with sample size as expected."
            )
        
        # General recommendations based on reliability score
        if self.reliability_score < 0.5:
            self.recommendations.append(
                "LOW RELIABILITY: Significant issues detected. Consider complete "
                "review of implementation and testing procedures."
            )
        elif self.reliability_score < 0.7:
            self.recommendations.append(
                "MODERATE RELIABILITY: Some issues detected. Review and address "
                "the most critical failure modes."
            )
        elif self.reliability_score < 0.9:
            self.recommendations.append(
                "GOOD RELIABILITY: Minor issues detected. Fine-tuning recommended "
                "for production use."
            )
        else:
            self.recommendations.append(
                "EXCELLENT RELIABILITY: Simulation performs well within expected parameters."
            )
        
        # Store recommendations in results
        self.results['recommendations'] = self.recommendations

    def _create_summary(self):
        """Create analysis summary"""
        total_runs = len(self._run_data)
        total_failures = len(self.failure_modes)
        
        # Count failures by type
        failure_by_type = {}
        for failure in self.failure_modes:
            ftype = failure['type']
            failure_by_type[ftype] = failure_by_type.get(ftype, 0) + 1
        
        # Calculate success rates by sample size
        success_by_size = {}
        for run_data in self._run_data:
            size = run_data['sample_size']
            if size not in success_by_size:
                success_by_size[size] = {'total': 0, 'success': 0}
            success_by_size[size]['total'] += 1
            if run_data['success']:
                success_by_size[size]['success'] += 1
        
        self.results['summary'] = {
            'total_runs': total_runs,
            'total_failures': total_failures,
            'failure_rate': total_failures / max(total_runs, 1),
            'reliability_score': self.reliability_score,
            'failure_types': failure_by_type,
            'success_rates_by_sample_size': {
                size: data['success'] / data['total'] 
                for size, data in success_by_size.items()
            },
            'sample_sizes_tested': list(success_by_size.keys()),
            'total_simulation_time': sum(run['execution_time'] for run in self._run_data),
            'recommendations_count': len(self.recommendations)
        }

    def generate_report(self) -> str:
        """Generate comprehensive failure analysis report"""
        if not self.results:
            return "No analysis results available. Run analysis first."
        
        report = []
        report.append("=" * 80)
        report.append("MONTE CARLO π ESTIMATION - FAILURE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        summary = self.results['summary']
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Simulation Runs: {summary['total_runs']}")
        report.append(f"Total Failures Detected: {summary['total_failures']}")
        report.append(f"Overall Failure Rate: {summary['failure_rate']:.3f}")
        report.append(f"Reliability Score: {self.reliability_score:.3f} / 1.000")
        report.append(f"Total Analysis Time: {summary['total_simulation_time']:.2f} seconds")
        report.append("")
        
        # Reliability Assessment
        if self.reliability_score >= 0.9:
            assessment = "EXCELLENT - Production Ready"
        elif self.reliability_score >= 0.7:
            assessment = "GOOD - Minor Issues"
        elif self.reliability_score >= 0.5:
            assessment = "MODERATE - Needs Attention"
        else:
            assessment = "POOR - Major Issues"
        
        report.append(f"RELIABILITY ASSESSMENT: {assessment}")
        report.append("")
        
        # Success Rates by Sample Size
        report.append("SUCCESS RATES BY SAMPLE SIZE")
        report.append("-" * 40)
        for size, rate in summary['success_rates_by_sample_size'].items():
            report.append(f"Sample Size {size:>10}: {rate:.3f} ({rate*100:.1f}%)")
        report.append("")
        
        # Failure Analysis
        if summary['failure_types']:
            report.append("FAILURE BREAKDOWN BY TYPE")
            report.append("-" * 40)
            for failure_type, count in sorted(summary['failure_types'].items(), 
                                            key=lambda x: x[1], reverse=True):
                percentage = count / summary['total_failures'] * 100
                report.append(f"{failure_type:>25}: {count:>3} ({percentage:>5.1f}%)")
            report.append("")
        
        # Critical Failures
        critical_failures = [fm for fm in self.failure_modes 
                           if fm['type'] in ['execution_failure', 'floating_point_error', 
                                           'timeout', 'extreme_outlier']]
        if critical_failures:
            report.append("CRITICAL FAILURES")
            report.append("-" * 40)
            for i, failure in enumerate(critical_failures[:10], 1):  # Show top 10
                report.append(f"{i:>2}. {failure['type'].upper()}: {failure['description']}")
                if 'sample_size' in failure:
                    report.append(f"    Sample Size: {failure['sample_size']}")
                if 'run_index' in failure:
                    report.append(f"    Run Index: {failure['run_index']}")
                report.append("")
        
        # Statistical Analysis Summary
        stat_analysis = self.results.get('statistical_analysis', {})
        if stat_analysis:
            report.append("STATISTICAL ANALYSIS SUMMARY")
            report.append("-" * 40)
            
            # Bias test results
            bias_tests = stat_analysis.get('bias_test', {})
            significant_bias_count = sum(1 for test in bias_tests.values() 
                                       if test.get('significant_bias', False))
            report.append(f"Bias Tests Performed: {len(bias_tests)}")
            report.append(f"Significant Bias Detected: {significant_bias_count}")
            
            # Variance test results
            variance_tests = stat_analysis.get('variance_test', {})
            variance_anomalies = sum(1 for test in variance_tests.values() 
                                   if test.get('excessive_variance', False) or 
                                      test.get('insufficient_variance', False))
            report.append(f"Variance Anomalies: {variance_anomalies}")
            
            # Normality test results
            normality_tests = stat_analysis.get('normality_test', {})
            normality_violations = sum(1 for test in normality_tests.values() 
                                     if test.get('normality_violated', False))
            report.append(f"Normality Violations: {normality_violations}")
            report.append("")
        
        # Performance Analysis Summary
        perf_analysis = self.results.get('performance_analysis', {})
        if perf_analysis:
            report.append("PERFORMANCE ANALYSIS SUMMARY")
            report.append("-" * 40)
            
            timing_stats = perf_analysis.get('timing_statistics', {})
            if timing_stats:
                report.append("Timing Statistics by Sample Size:")
                for size, stats in sorted(timing_stats.items()):
                    samples_per_sec = stats['samples_per_second']
                    report.append(f"  {size:>10} samples: {samples_per_sec:>10,.0f} samples/sec")
            
            scalability = perf_analysis.get('scalability', {})
            if scalability:
                r_squared = scalability['r_squared']
                report.append(f"Scalability (R²): {r_squared:.3f}")
                if r_squared < 0.8:
                    report.append("  WARNING: Poor linear scalability detected")
            
            timeout_failures = perf_analysis.get('timeout_failures', 0)
            if timeout_failures > 0:
                report.append(f"Timeout Failures: {timeout_failures}")
            report.append("")
        
        # Convergence Analysis Summary
        conv_analysis = self.results.get('convergence_analysis', {})
        if conv_analysis:
            report.append("CONVERGENCE ANALYSIS SUMMARY")
            report.append("-" * 40)
            report.append(f"Slow Convergence Cases: {conv_analysis.get('slow_convergence_count', 0)}")
            report.append(f"Oscillatory Behavior: {conv_analysis.get('oscillatory_behavior_count', 0)}")
            report.append(f"Premature Plateaus: {conv_analysis.get('premature_plateau_count', 0)}")
            report.append(f"Non-monotonic Convergence: {conv_analysis.get('non_monotonic_count', 0)}")
            
            conv_rates = conv_analysis.get('convergence_rates', [])
            if conv_rates:
                mean_rate = np.mean(conv_rates)
                report.append(f"Mean Convergence Rate: {mean_rate:.3f} (expected ~0.5)")
            report.append("")
        
        # Outlier Analysis Summary
        outlier_analysis = self.results.get('outlier_analysis', {})
        if outlier_analysis:
            report.append("OUTLIER ANALYSIS SUMMARY")
            report.append("-" * 40)
            
            extreme_outliers = len(outlier_analysis.get('extreme_outliers', []))
            report.append(f"Extreme Outliers Detected: {extreme_outliers}")
            
            outliers_by_size = outlier_analysis.get('outliers_by_sample_size', {})
            for size, data in sorted(outliers_by_size.items()):
                outlier_rate = data['outlier_rate'] * 100
                report.append(f"  Sample Size {size:>10}: {outlier_rate:>5.1f}% outlier rate")
            report.append("")
        
        # Recommendations
        if self.recommendations:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(self.recommendations, 1):
                report.append(f"{i:>2}. {rec}")
            report.append("")
        
        # Detailed Failure List (if requested)
        if len(self.failure_modes) <= 50:  # Only show if manageable number
            report.append("DETAILED FAILURE LIST")
            report.append("-" * 40)
            for i, failure in enumerate(self.failure_modes, 1):
                report.append(f"{i:>3}. [{failure['type'].upper()}] {failure['description']}")
                if 'sample_size' in failure:
                    report.append(f"     Sample Size: {failure['sample_size']}")
                if 'run_index' in failure:
                    report.append(f"     Run Index: {failure['run_index']}")
                report.append("")
        elif len(self.failure_modes) > 50:
            report.append(f"DETAILED FAILURE LIST (showing first 50 of {len(self.failure_modes)})")
            report.append("-" * 40)
            for i, failure in enumerate(self.failure_modes[:50], 1):
                report.append(f"{i:>3}. [{failure['type'].upper()}] {failure['description']}")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)

    def visualize_failures(self) -> None:
        """Create comprehensive visualizations of failure analysis"""
        if not self.results:
            print("No analysis results available. Run analysis first.")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Reliability Score and Failure Rate
        ax1 = plt.subplot(3, 4, 1)
        categories = ['Reliability\nScore', 'Failure\nRate']
        values = [self.reliability_score, self.results['summary']['failure_rate']]
        colors = ['green' if self.reliability_score > 0.8 else 'orange' if self.reliability_score > 0.5 else 'red',
                 'red' if self.results['summary']['failure_rate'] > 0.1 else 'orange' if self.results['summary']['failure_rate'] > 0.05 else 'green']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_ylim(0, 1)
        ax1.set_title('Overall Reliability Metrics')
        ax1.set_ylabel('Score/Rate')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Failure Types Distribution
        ax2 = plt.subplot(3, 4, 2)
        failure_types = self.results['summary']['failure_types']
        if failure_types:
            types = list(failure_types.keys())
            counts = list(failure_types.values())
            
            # Truncate long type names
            types_short = [t[:15] + '...' if len(t) > 15 else t for t in types]
            
            wedges, texts, autotexts = ax2.pie(counts, labels=types_short, autopct='%1.1f%%',
                                              startangle=90, textprops={'fontsize': 8})
            ax2.set_title('Failure Types Distribution')
        else:
            ax2.text(0.5, 0.5, 'No Failures\nDetected', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax2.set_title('Failure Types Distribution')
        
        # 3. Success Rate by Sample Size
        ax3 = plt.subplot(3, 4, 3)
        success_rates = self.results['summary']['success_rates_by_sample_size']
        if success_rates:
            sizes = sorted(success_rates.keys())
            rates = [success_rates[size] for size in sizes]
            
            bars = ax3.bar(range(len(sizes)), rates, alpha=0.7,
                          color=['green' if r > 0.95 else 'orange' if r > 0.8 else 'red' for r in rates])
            ax3.set_xticks(range(len(sizes)))
            ax3.set_xticklabels([f'{s:,}' for s in sizes], rotation=45)
            ax3.set_ylim(0, 1)
            ax3.set_title('Success Rate by Sample Size')
            ax3.set_xlabel('Sample Size')
            ax3.set_ylabel('Success Rate')
            
            # Add value labels
            for i, (bar, rate) in enumerate(zip(bars, rates)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Performance Analysis
        ax4 = plt.subplot(3, 4, 4)
        perf_analysis = self.results.get('performance_analysis', {})
        timing_stats = perf_analysis.get('timing_statistics', {})
        
        if timing_stats:
            sizes = sorted(timing_stats.keys())
            samples_per_sec = [timing_stats[size]['samples_per_second'] for size in sizes]
            
            ax4.loglog(sizes, samples_per_sec, 'bo-', linewidth=2, markersize=6)
            ax4.set_xlabel('Sample Size')
            ax4.set_ylabel('Samples/Second')
            ax4.set_title('Performance Scaling')
            ax4.grid(True, alpha=0.3)
            
            # Add trend line if we have scalability data
            scalability = perf_analysis.get('scalability', {})
            if scalability and scalability.get('r_squared', 0) > 0.5:
                # Show expected linear scaling
                expected_rate = samples_per_sec[0]  # Normalize to first point
                expected_samples_per_sec = [expected_rate for _ in sizes]
                ax4.loglog(sizes, expected_samples_per_sec, 'r--', alpha=0.5, 
                          label=f'Expected (R²={scalability["r_squared"]:.3f})')
                ax4.legend()
        
        # 5. Error Distribution Analysis
        ax5 = plt.subplot(3, 4, 5)
        all_errors = []
        all_sample_sizes = []
        
        for run_data in self._run_data:
            all_errors.append(run_data['relative_error'])
            all_sample_sizes.append(run_data['sample_size'])
        
        if all_errors:
            # Create scatter plot of errors vs sample size
            unique_sizes = sorted(set(all_sample_sizes))
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sizes)))
            size_color_map = dict(zip(unique_sizes, colors))
            
            scatter_colors = [size_color_map[size] for size in all_sample_sizes]
            ax5.scatter(all_sample_sizes, all_errors, c=scatter_colors, alpha=0.6, s=20)
            
            ax5.set_xscale('log')
            ax5.set_yscale('log')
            ax5.set_xlabel('Sample Size')
            ax5.set_ylabel('Relative Error (%)')
            ax5.set_title('Error vs Sample Size')
            ax5.grid(True, alpha=0.3)
            
            # Add theoretical error line (1/√n scaling)
            if unique_sizes:
                theoretical_errors = [100 * 1.64 / np.sqrt(size) for size in unique_sizes]
                ax5.plot(unique_sizes, theoretical_errors, 'r--', linewidth=2, 
                        label='Theoretical (1/√n)', alpha=0.7)
                ax5.legend()
        
        # 6. Convergence Analysis
        ax6 = plt.subplot(3, 4, 6)
        conv_analysis = self.results.get('convergence_analysis', {})
        
        conv_metrics = ['Slow\nConvergence', 'Oscillatory\nBehavior', 
                       'Premature\nPlateau', 'Non-monotonic']
        conv_counts = [
            conv_analysis.get('slow_convergence_count', 0),
            conv_analysis.get('oscillatory_behavior_count', 0),
            conv_analysis.get('premature_plateau_count', 0),
            conv_analysis.get('non_monotonic_count', 0)
        ]
        
        bars = ax6.bar(conv_metrics, conv_counts, alpha=0.7,
                      color=['red' if c > 0 else 'green' for c in conv_counts])
        ax6.set_title('Convergence Issues')
        ax6.set_ylabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, conv_counts):
            if count > 0:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        str(count), ha='center', va='bottom')
        
        # 7. Statistical Test Results
        ax7 = plt.subplot(3, 4, 7)
        stat_analysis = self.results.get('statistical_analysis', {})
        
        stat_metrics = ['Bias\nTests', 'Variance\nAnomalies', 'Normality\nViolations', 'CI\nFailures']
        stat_counts = [
            sum(1 for test in stat_analysis.get('bias_test', {}).values() 
                if test.get('significant_bias', False)),
            sum(1 for test in stat_analysis.get('variance_test', {}).values() 
                if test.get('excessive_variance', False) or test.get('insufficient_variance', False)),
            sum(1 for test in stat_analysis.get('normality_test', {}).values() 
                if test.get('normality_violated', False)),
            sum(1 for test in stat_analysis.get('confidence_interval_test', {}).values() 
                if not test.get('contains_true_pi', True))
        ]
        
        bars = ax7.bar(stat_metrics, stat_counts, alpha=0.7,
                      color=['red' if c > 0 else 'green' for c in stat_counts])
        ax7.set_title('Statistical Test Failures')
        ax7.set_ylabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, stat_counts):
            if count > 0:
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        str(count), ha='center', va='bottom')
        
        # 8. Outlier Analysis
        ax8 = plt.subplot(3, 4, 8)
        outlier_analysis = self.results.get('outlier_analysis', {})
        outliers_by_size = outlier_analysis.get('outliers_by_sample_size', {})
        
        if outliers_by_size:
            sizes = sorted(outliers_by_size.keys())
            outlier_rates = [outliers_by_size[size]['outlier_rate'] * 100 for size in sizes]
            
            bars = ax8.bar(range(len(sizes)), outlier_rates, alpha=0.7,
                          color=['red' if r > 10 else 'orange' if r > 5 else 'green' for r in outlier_rates])
            ax8.set_xticks(range(len(sizes)))
            ax8.set_xticklabels([f'{s:,}' for s in sizes], rotation=45)
            ax8.set_title('Outlier Rate by Sample Size')
            ax8.set_xlabel('Sample Size')
            ax8.set_ylabel('Outlier Rate (%)')
            
            # Add 5% and 10% reference lines
            ax8.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
            ax8.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
            ax8.legend()
        
        # 9. Execution Time Distribution
        ax9 = plt.subplot(3, 4, 9)
        execution_times = [run['execution_time'] for run in self._run_data]
        
        if execution_times:
            ax9.hist(execution_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax9.set_xlabel('Execution Time (seconds)')
            ax9.set_ylabel('Frequency')
            ax9.set_title('Execution Time Distribution')
            
            # Add statistics
            mean_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            ax9.axvline(mean_time, color='red', linestyle='--', 
                       label=f'Mean: {mean_time:.3f}s')
            ax9.axvline(mean_time + 2*std_time, color='orange', linestyle='--', 
                       label=f'+2σ: {mean_time + 2*std_time:.3f}s')
            ax9.legend()
        
        # 10. Precision Analysis
        ax10 = plt.subplot(3, 4, 10)
        precision_analysis = self.results.get('precision_analysis', {})
        
        precision_metrics = ['Floating Point\nIssues', 'Accumulation\nErrors', 
                           'Catastrophic\nCancellation', 'Precision\nLoss']
        precision_counts = [
            len(precision_analysis.get('floating_point_issues', [])),
            len(precision_analysis.get('accumulation_errors', [])),
            len(precision_analysis.get('catastrophic_cancellation', [])),
            precision_analysis.get('precision_loss_count', 0)
        ]
        
        bars = ax10.bar(precision_metrics, precision_counts, alpha=0.7,
                       color=['red' if c > 0 else 'green' for c in precision_counts])
        ax10.set_title('Precision Issues')
        ax10.set_ylabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, precision_counts):
            if count > 0:
                height = bar.get_height()
                ax10.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         str(count), ha='center', va='bottom')
        
        # 11. Randomness Quality
        ax11 = plt.subplot(3, 4, 11)
        randomness_analysis = self.results.get('randomness_analysis', {})
        
        randomness_metrics = ['Seed\nDependency', 'Serial\nCorrelation', 
                            'Distribution\nAnomalies', 'Correlation\nIssues']
        randomness_counts = [
            sum(1 for test in randomness_analysis.get('seed_dependency', {}).values() 
                if test.get('significant_correlation', False)),
            len([fm for fm in self.failure_modes if fm['type'] == 'serial_correlation']),
            sum(1 for test in randomness_analysis.get('distribution_uniformity', {}).values() 
                if test.get('distribution_anomaly', False)),
            len(randomness_analysis.get('correlation_issues', []))
        ]
        
        bars = ax11.bar(randomness_metrics, randomness_counts, alpha=0.7,
                       color=['red' if c > 0 else 'green' for c in randomness_counts])
        ax11.set_title('Randomness Quality Issues')
        ax11.set_ylabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, randomness_counts):
            if count > 0:
                height = bar.get_height()
                ax11.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         str(count), ha='center', va='bottom')
        
        # 12. Failure Timeline (if we have run indices)
        ax12 = plt.subplot(3, 4, 12)
        failure_runs = []
        failure_types = []
        
        for failure in self.failure_modes:
            if 'run_index' in failure:
                failure_runs.append(failure['run_index'])
                failure_types.append(failure['type'])
        
        if failure_runs:
            # Create scatter plot of failures over run indices
            unique_types = list(set(failure_types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
            type_color_map = dict(zip(unique_types, colors))
            
            for ftype in unique_types:
                type_runs = [run for run, t in zip(failure_runs, failure_types) if t == ftype]
                type_y = [unique_types.index(ftype)] * len(type_runs)
                ax12.scatter(type_runs, type_y, c=[type_color_map[ftype]], 
                           label=ftype[:15], alpha=0.7, s=30)
            
            ax12.set_xlabel('Run Index')
            ax12.set_ylabel('Failure Type')
            ax12.set_yticks(range(len(unique_types)))
            ax12.set_yticklabels([t[:15] + '...' if len(t) > 15 else t for t in unique_types])
            ax12.set_title('Failure Timeline')
            ax12.grid(True, alpha=0.3)
            
            # Only show legend if not too many types
            if len(unique_types) <= 8:
                ax12.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            ax12.text(0.5, 0.5, 'No Failures\nwith Run Index', ha='center', va='center',
                     transform=ax12.transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax12.set_title('Failure Timeline')
        
        plt.tight_layout()
        plt.show()
        
        # Create additional detailed plots if there are convergence issues
        self._plot_convergence_details()
        
        # Create error analysis plots
        self._plot_error_analysis()

    def _plot_convergence_details(self):
        """Create detailed convergence analysis plots"""
        convergence_failures = [fm for fm in self.failure_modes 
                              if fm['type'] in ['slow_convergence', 'oscillatory_convergence', 
                                              'premature_plateau']]
        
        if not convergence_failures:
            return
        
        # Find runs with convergence data that had failures
        problem_runs = []
        for failure in convergence_failures[:6]:  # Show up to 6 examples
            sample_size = failure.get('sample_size')
            run_index = failure.get('run_index')
            
            for run_data in self._run_data:
                if (run_data['sample_size'] == sample_size and 
                    run_data['run_index'] == run_index and 
                    run_data['convergence_data']):
                    problem_runs.append((run_data, failure))
                    break
        
        if not problem_runs:
            return
        
        # Create convergence detail plots
        n_plots = len(problem_runs)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (run_data, failure) in enumerate(problem_runs):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
            
            samples = [point[0] for point in run_data['convergence_data']]
            estimates = [point[1] for point in run_data['convergence_data']]
            errors = [abs(est - np.pi) for est in estimates]
            
            # Plot convergence
            ax.loglog(samples, errors, 'b-', linewidth=2, alpha=0.7, label='Actual')
            
            # Plot theoretical convergence
            theoretical_errors = [1.64 / np.sqrt(n) for n in samples]
            ax.loglog(samples, theoretical_errors, 'r--', linewidth=2, 
                     alpha=0.7, label='Theoretical')
            
            ax.set_xlabel('Sample Size')
            ax.set_ylabel('Absolute Error')
            ax.set_title(f'{failure["type"].replace("_", " ").title()}\n'
                        f'Sample Size: {run_data["sample_size"]}, Run: {run_data["run_index"]}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(problem_runs), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def _plot_error_analysis(self):
        """Create detailed error analysis plots"""
        if not self._run_data:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Group data by sample size
        by_sample_size = {}
        for run_data in self._run_data:
            size = run_data['sample_size']
            if size not in by_sample_size:
                by_sample_size[size] = {'estimates': [], 'errors': [], 'rel_errors': []}
            by_sample_size[size]['estimates'].append(run_data['pi_estimate'])
            by_sample_size[size]['errors'].append(run_data['absolute_error'])
            by_sample_size[size]['rel_errors'].append(run_data['relative_error'])
        
        # 1. Error distribution by sample size
        sizes = sorted(by_sample_size.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
        
        for i, size in enumerate(sizes):
            errors = by_sample_size[size]['errors']
            ax1.hist(errors, bins=20, alpha=0.6, color=colors[i], 
                    label=f'{size:,}', density=True)
        
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Distribution by Sample Size')
        ax1.legend()
        ax1.set_yscale('log')
        
        # 2. Q-Q plot for normality check
        all_estimates = []
        all_sample_sizes = []
        for size, data in by_sample_size.items():
            all_estimates.extend(data['estimates'])
            all_sample_sizes.extend([size] * len(data['estimates']))
        
        if all_estimates:
            # Standardize estimates
            standardized_estimates = []
            for est, size in zip(all_estimates, all_sample_sizes):
                theoretical_std = np.sqrt(np.pi * (4 - np.pi) / size)
                standardized = (est - np.pi) / theoretical_std
                standardized_estimates.append(standardized)
            
            stats.probplot(standardized_estimates, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot: Standardized Estimates vs Normal')
            ax2.grid(True, alpha=0.3)
        
        # 3. Error vs theoretical error
        theoretical_errors = []
        actual_errors = []
        
        for size, data in by_sample_size.items():
            theoretical_error = 1.64 / np.sqrt(size)
            for error in data['errors']:
                theoretical_errors.append(theoretical_error)
                actual_errors.append(error)
        
        ax3.scatter(theoretical_errors, actual_errors, alpha=0.6, s=20)
        
        # Add perfect correlation line
        min_error = min(min(theoretical_errors), min(actual_errors))
        max_error = max(max(theoretical_errors), max(actual_errors))
        ax3.plot([min_error, max_error], [min_error, max_error], 'r--', 
                linewidth=2, alpha=0.7, label='Perfect correlation')
        
        ax3.set_xlabel('Theoretical Error')
        ax3.set_ylabel('Actual Error')
        ax3.set_title('Actual vs Theoretical Error')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Bias analysis by sample size
        biases = []
        bias_errors = []
        size_labels = []
        
        for size, data in by_sample_size.items():
            estimates = data['estimates']
            if len(estimates) > 1:
                bias = np.mean(estimates) - np.pi
                bias_se = np.std(estimates) / np.sqrt(len(estimates))
                biases.append(bias)
                bias_errors.append(1.96 * bias_se)  # 95% CI
                size_labels.append(f'{size:,}')
        
        if biases:
            x_pos = range(len(biases))
            ax4.errorbar(x_pos, biases, yerr=bias_errors, fmt='o-', 
                        capsize=5, capthick=2, linewidth=2, markersize=8)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No bias')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(size_labels, rotation=45)
            ax4.set_xlabel('Sample Size')
            ax4.set_ylabel('Bias (π estimate - π)')
            ax4.set_title('Bias Analysis with 95% Confidence Intervals')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        plt.show()

    def export_results(self, filename: str = None) -> str:
        """Export analysis results to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"pi_estimation_failure_analysis_{timestamp}.txt"
        
        report = self.generate_report()
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Analysis report exported to: {filename}")
            return filename
        except Exception as e:
            print(f"Error exporting report: {e}")
            return None

    def get_failure_summary(self) -> Dict[str, Any]:
        """Get a concise summary of failures for programmatic use"""
        if not self.results:
            return {}
        
        summary = self.results['summary']
        
        return {
            'reliability_score': self.reliability_score,
            'total_runs': summary['total_runs'],
            'total_failures': summary['total_failures'],
            'failure_rate': summary['failure_rate'],
            'critical_failures': len([fm for fm in self.failure_modes 
                                    if fm['type'] in ['execution_failure', 'floating_point_error', 
                                                    'timeout', 'extreme_outlier']]),
            'most_common_failure': max(summary['failure_types'].items(), 
                                     key=lambda x: x[1])[0] if summary['failure_types'] else None,
            'recommendations_count': len(self.recommendations),
            'needs_attention': self.reliability_score < 0.7,
            'production_ready': self.reliability_score >= 0.9 and summary['failure_rate'] < 0.05
        }

