import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Tuple, Dict, Any, Union
import sys
import os
from scipy import stats
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class VaRSimulation(BaseSimulation):
    """
    Value at Risk (VaR) Estimation using Multiple Methods
    
    Value at Risk is a statistical measure that quantifies the potential loss in value
    of a portfolio over a defined period for a given confidence interval. This simulation
    implements multiple VaR estimation methods including parametric, historical simulation,
    and Monte Carlo approaches.
    
    Mathematical Background:
    -----------------------
    VaR answers the question: "What is the maximum loss we can expect with X% confidence
    over the next N days?"
    
    Formally: P(Loss ≤ VaR_α) = α
    Where α is the confidence level (e.g., 95% or 99%)
    
    VaR Methods Implemented:
    -----------------------
    1. Parametric VaR (Variance-Covariance):
       VaR = μ + σ * Φ^(-1)(α) * √t
       Where Φ^(-1) is the inverse normal CDF
    
    2. Historical Simulation:
       VaR = α-th percentile of historical returns
    
    3. Monte Carlo Simulation:
       Generate random scenarios and calculate α-th percentile
    
    4. Cornish-Fisher VaR:
       Adjusts parametric VaR for skewness and kurtosis
       VaR = μ + σ * (z_α + (z_α² - 1)/6 * S + (z_α³ - 3z_α)/24 * K) * √t
    
    Expected Shortfall (Conditional VaR):
    ------------------------------------
    ES_α = E[Loss | Loss > VaR_α]
    The expected loss given that the loss exceeds VaR
    
    Applications:
    ------------
    - Portfolio risk management
    - Regulatory capital requirements (Basel III)
    - Risk budgeting and allocation
    - Stress testing and scenario analysis
    - Performance attribution
    - Risk-adjusted performance measurement
    - Trading limit setting
    - Insurance and actuarial modeling
    
    Parameters:
    -----------
    returns : array-like, optional
        Historical returns data (if None, will be simulated)
    confidence_levels : list, default=[0.95, 0.99]
        Confidence levels for VaR calculation
    time_horizon : int, default=1
        Time horizon in days
    portfolio_value : float, default=1000000
        Portfolio value for absolute VaR calculation
    simulation_method : str, default='normal'
        Distribution for Monte Carlo ('normal', 't', 'skewed_t')
    n_simulations : int, default=100000
        Number of Monte Carlo simulations
    window_size : int, default=252
        Rolling window size for historical simulation
    mean_return : float, default=0.0008
        Expected daily return (if simulating data)
    volatility : float, default=0.02
        Daily volatility (if simulating data)
    skewness : float, default=0.0
        Skewness parameter for distribution
    kurtosis : float, default=3.0
        Kurtosis parameter for distribution
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    returns_data : ndarray
        Returns data used for VaR calculation
    var_results : dict
        VaR estimates from different methods
    es_results : dict
        Expected Shortfall estimates
    backtesting_results : dict
        Backtesting statistics
    risk_metrics : dict
        Additional risk metrics
    
    Methods:
    --------
    configure(**kwargs) : bool
        Configure VaR simulation parameters
    run(**kwargs) : SimulationResult
        Execute VaR estimation using multiple methods
    visualize(result=None) : None
        Create comprehensive risk visualizations
    backtest_var(var_estimates, actual_returns) : dict
        Perform VaR backtesting
    calculate_risk_metrics() : dict
        Calculate additional risk metrics
    generate_stress_scenarios() : dict
        Generate stress test scenarios
    validate_parameters() : List[str]
        Validate simulation parameters
    
    Examples:
    ---------
    >>> # Basic VaR estimation with simulated data
    >>> var_sim = VaRSimulation(mean_return=0.001, volatility=0.025)
    >>> result = var_sim.run()
    >>> print(f"95% VaR: ${result.results['parametric_var_95']:.2f}")
    >>> print(f"99% VaR: ${result.results['parametric_var_99']:.2f}")
    
    >>> # VaR with historical data
    >>> returns = np.random.normal(0.0008, 0.02, 1000)  # Your actual returns
    >>> var_sim = VaRSimulation(returns=returns, portfolio_value=5000000)
    >>> result = var_sim.run()
    >>> var_sim.visualize()
    
    >>> # Multi-method comparison
    >>> var_sim = VaRSimulation(confidence_levels=[0.90, 0.95, 0.99])
    >>> result = var_sim.run()
    >>> for method in ['parametric', 'historical', 'monte_carlo']:
    ...     print(f"{method} 99% VaR: ${result.results[f'{method}_var_99']:.2f}")
    """
    
    def __init__(self, returns: Optional[np.ndarray] = None, 
                 confidence_levels: List[float] = [0.95, 0.99],
                 time_horizon: int = 1, portfolio_value: float = 1000000,
                 simulation_method: str = 'normal', n_simulations: int = 100000,
                 window_size: int = 252, mean_return: float = 0.0008,
                 volatility: float = 0.02, skewness: float = 0.0,
                 kurtosis: float = 3.0, random_seed: Optional[int] = None):
        super().__init__("Value at Risk (VaR) Estimation")
        
        # Data parameters
        self.returns = returns
        self.confidence_levels = confidence_levels
        self.time_horizon = time_horizon
        self.portfolio_value = portfolio_value
        
        # Simulation parameters
        self.simulation_method = simulation_method.lower()
        self.n_simulations = n_simulations
        self.window_size = window_size
        
        # Distribution parameters
        self.mean_return = mean_return
        self.volatility = volatility
        self.skewness = skewness
        self.kurtosis = kurtosis
        
        # Store in parameters dict
        self.parameters.update({
            'confidence_levels': confidence_levels,
            'time_horizon': time_horizon,
            'portfolio_value': portfolio_value,
            'simulation_method': simulation_method,
            'n_simulations': n_simulations,
            'window_size': window_size,
            'mean_return': mean_return,
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state
        self.returns_data = None
        self.var_results = {}
        self.es_results = {}
        self.backtesting_results = {}
        self.risk_metrics = {}
        self.is_configured = True
    
    def configure(self, **kwargs) -> bool:
        """Configure VaR simulation parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute VaR estimation using multiple methods"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Generate or use provided returns data
        if self.returns is None:
            # Generate synthetic returns data
            n_observations = max(1000, self.window_size * 2)
            
            if self.simulation_method == 'normal':
                self.returns_data = np.random.normal(
                    self.mean_return, self.volatility, n_observations
                )
            elif self.simulation_method == 't':
                # Student's t-distribution
                df = max(4, self.kurtosis)  # Degrees of freedom
                self.returns_data = stats.t.rvs(df, loc=self.mean_return, 
                                              scale=self.volatility, size=n_observations)
            elif self.simulation_method == 'skewed_t':
                # Skewed t-distribution (approximation)
                base_returns = stats.t.rvs(self.kurtosis, size=n_observations)
                # Apply skewness transformation
                self.returns_data = (self.mean_return + 
                                   self.volatility * (base_returns + self.skewness * (base_returns**2 - 1)))
            else:
                raise ValueError("simulation_method must be 'normal', 't', or 'skewed_t'")
        else:
            self.returns_data = np.array(self.returns)
        
        # Calculate VaR using different methods
        self.var_results = {}
        self.es_results = {}
        
        for confidence_level in self.confidence_levels:
            alpha = confidence_level
            conf_str = f"{int(alpha*100)}"
            
            # 1. Parametric VaR (Variance-Covariance)
            mean_ret = np.mean(self.returns_data)
            std_ret = np.std(self.returns_data)
            z_score = stats.norm.ppf(1 - alpha)  # Note: 1-alpha for loss percentile
            
            parametric_var = -(mean_ret + z_score * std_ret) * np.sqrt(self.time_horizon)
            self.var_results[f'parametric_var_{conf_str}'] = parametric_var * self.portfolio_value
            
            # 2. Historical Simulation VaR
            sorted_returns = np.sort(self.returns_data)
            hist_index = int((1 - alpha) * len(sorted_returns))
            historical_var = -sorted_returns[hist_index] * np.sqrt(self.time_horizon)
            self.var_results[f'historical_var_{conf_str}'] = historical_var * self.portfolio_value
            
            # 3. Monte Carlo VaR
            mc_returns = self._generate_monte_carlo_returns()
            mc_sorted = np.sort(mc_returns)
            mc_index = int((1 - alpha) * len(mc_returns))
            monte_carlo_var = -mc_sorted[mc_index] * np.sqrt(self.time_horizon)
            self.var_results[f'monte_carlo_var_{conf_str}'] = monte_carlo_var * self.portfolio_value
            
            # 4. Cornish-Fisher VaR (adjusted for skewness and kurtosis)
            sample_skew = stats.skew(self.returns_data)
            sample_kurt = stats.kurtosis(self.returns_data, fisher=True)  # Excess kurtosis
            
            cf_adjustment = (z_score + 
                           (z_score**2 - 1) * sample_skew / 6 +
                           (z_score**3 - 3*z_score) * sample_kurt / 24 -
                           (2*z_score**3 - 5*z_score) * sample_skew**2 / 36)
            
            cf_var = -(mean_ret + cf_adjustment * std_ret) * np.sqrt(self.time_horizon)
            self.var_results[f'cornish_fisher_var_{conf_str}'] = cf_var * self.portfolio_value
            
            # Calculate Expected Shortfall (Conditional VaR)
            # Parametric ES
            es_multiplier = stats.norm.pdf(z_score) / (1 - alpha)
            parametric_es = -(mean_ret + es_multiplier * std_ret) * np.sqrt(self.time_horizon)
            self.es_results[f'parametric_es_{conf_str}'] = parametric_es * self.portfolio_value
            
            # Historical ES
            tail_returns = sorted_returns[:hist_index+1]
            historical_es = -np.mean(tail_returns) * np.sqrt(self.time_horizon)
            self.es_results[f'historical_es_{conf_str}'] = historical_es * self.portfolio_value
            
            # Monte Carlo ES
            tail_mc_returns = mc_sorted[:mc_index+1]
            monte_carlo_es = -np.mean(tail_mc_returns) * np.sqrt(self.time_horizon)
            self.es_results[f'monte_carlo_es_{conf_str}'] = monte_carlo_es * self.portfolio_value
        
        # Perform backtesting if we have enough data
        if len(self.returns_data) > self.window_size:
            self.backtesting_results = self.backtest_var()
        
        # Calculate additional risk metrics
        self.risk_metrics = self.calculate_risk_metrics()
        
        execution_time = time.time() - start_time
        
        # Combine all results
        all_results = {**self.var_results, **self.es_results}
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results=all_results,
            statistics={
                **self.risk_metrics,
                'backtesting_results': self.backtesting_results,
                'data_statistics': {
                    'n_observations': len(self.returns_data),
                    'mean_return': np.mean(self.returns_data),
                    'volatility': np.std(self.returns_data),
                    'skewness': stats.skew(self.returns_data),
                    'kurtosis': stats.kurtosis(self.returns_data),
                    'min_return': np.min(self.returns_data),
                    'max_return': np.max(self.returns_data)
                }
            },
            execution_time=execution_time,
            convergence_data=[]  # VaR doesn't have traditional convergence
        )
        
        self.result = result
        return result
    
    def _generate_monte_carlo_returns(self) -> np.ndarray:
        """Generate Monte Carlo returns based on simulation method"""
        if self.simulation_method == 'normal':
            return np.random.normal(
                np.mean(self.returns_data), 
                np.std(self.returns_data), 
                self.n_simulations
            )
        elif self.simulation_method == 't':
            df = max(4, self.kurtosis)
            return stats.t.rvs(df, loc=np.mean(self.returns_data),                              scale=np.std(self.returns_data), size=self.n_simulations)
        elif self.simulation_method == 'skewed_t':
            df = max(4, self.kurtosis)
            base_returns = stats.t.rvs(df, size=self.n_simulations)
            skew_factor = stats.skew(self.returns_data)
            return (np.mean(self.returns_data) + 
                   np.std(self.returns_data) * (base_returns + skew_factor * (base_returns**2 - 1)))
        else:
            return np.random.normal(
                np.mean(self.returns_data), 
                np.std(self.returns_data), 
                self.n_simulations
            )
    
    def backtest_var(self) -> dict:
        """Perform VaR backtesting using rolling window approach"""
        if len(self.returns_data) <= self.window_size:
            return {}
        
        backtesting_results = {}
        
        # Split data into estimation and testing periods
        n_test_periods = len(self.returns_data) - self.window_size
        
        for confidence_level in self.confidence_levels:
            conf_str = f"{int(confidence_level*100)}"
            
            var_forecasts = []
            actual_returns = []
            violations = []
            
            # Rolling window backtesting
            for i in range(n_test_periods):
                # Estimation window
                estimation_data = self.returns_data[i:i+self.window_size]
                
                # Calculate VaR for next period
                mean_ret = np.mean(estimation_data)
                std_ret = np.std(estimation_data)
                z_score = stats.norm.ppf(1 - confidence_level)
                
                var_forecast = -(mean_ret + z_score * std_ret)
                var_forecasts.append(var_forecast)
                
                # Actual return for next period
                actual_return = self.returns_data[i + self.window_size]
                actual_returns.append(actual_return)
                
                # Check for VaR violation (actual loss > VaR)
                violation = actual_return < -var_forecast
                violations.append(violation)
            
            # Calculate backtesting statistics
            violation_rate = np.mean(violations)
            expected_violation_rate = 1 - confidence_level
            
            # Kupiec POF (Proportion of Failures) Test
            n_violations = np.sum(violations)
            n_observations = len(violations)
            
            if n_violations > 0 and n_violations < n_observations:
                lr_pof = -2 * np.log(
                    (expected_violation_rate**n_violations * 
                     (1-expected_violation_rate)**(n_observations-n_violations)) /
                    ((violation_rate**n_violations * 
                     (1-violation_rate)**(n_observations-n_violations)))
                )
                pof_p_value = 1 - stats.chi2.cdf(lr_pof, df=1)
            else:
                lr_pof = np.inf
                pof_p_value = 0.0
            
            # Christoffersen Independence Test
            # Count violation clusters
            violation_transitions = np.diff(violations.astype(int))
            n_00 = np.sum((np.array(violations[:-1]) == 0) & (np.array(violations[1:]) == 0))
            n_01 = np.sum((np.array(violations[:-1]) == 0) & (np.array(violations[1:]) == 1))
            n_10 = np.sum((np.array(violations[:-1]) == 1) & (np.array(violations[1:]) == 0))
            n_11 = np.sum((np.array(violations[:-1]) == 1) & (np.array(violations[1:]) == 1))
            
            if n_01 > 0 and n_10 > 0 and n_00 > 0 and n_11 > 0:
                pi_01 = n_01 / (n_00 + n_01)
                pi_11 = n_11 / (n_10 + n_11)
                pi = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)
                
                lr_ind = -2 * np.log(
                    (pi**(n_01 + n_11) * (1-pi)**(n_00 + n_10)) /
                    (pi_01**n_01 * (1-pi_01)**n_00 * pi_11**n_11 * (1-pi_11)**n_10)
                )
                ind_p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
            else:
                lr_ind = 0.0
                ind_p_value = 1.0
            
            backtesting_results[f'var_{conf_str}'] = {
                'violation_rate': violation_rate,
                'expected_violation_rate': expected_violation_rate,
                'n_violations': n_violations,
                'n_observations': n_observations,
                'kupiec_lr_stat': lr_pof,
                'kupiec_p_value': pof_p_value,
                'christoffersen_lr_stat': lr_ind,
                'christoffersen_p_value': ind_p_value,
                'model_adequate': (pof_p_value > 0.05) and (ind_p_value > 0.05)
            }
        
        return backtesting_results
    
    def calculate_risk_metrics(self) -> dict:
        """Calculate additional risk metrics"""
        returns = self.returns_data
        
        # Basic statistics
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < mean_return]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        # Tail ratio (95th percentile / 5th percentile)
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        tail_ratio = abs(p95 / p5) if p5 != 0 else np.inf
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'tail_ratio': tail_ratio,
            'downside_deviation': downside_deviation,
            'upside_capture': np.mean(returns[returns > 0]) if np.any(returns > 0) else 0,
            'downside_capture': np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
        }
    
    def generate_stress_scenarios(self) -> dict:
        """Generate stress test scenarios"""
        base_mean = np.mean(self.returns_data)
        base_vol = np.std(self.returns_data)
        
        scenarios = {
            'market_crash': {
                'description': 'Market crash scenario (-3 sigma event)',
                'return': base_mean - 3 * base_vol,
                'probability': stats.norm.cdf(-3),
                'portfolio_impact': (base_mean - 3 * base_vol) * self.portfolio_value
            },
            'high_volatility': {
                'description': 'High volatility scenario (2x normal vol)',
                'return': base_mean - 2 * (2 * base_vol),
                'probability': stats.norm.cdf(-2),
                'portfolio_impact': (base_mean - 2 * (2 * base_vol)) * self.portfolio_value
            },
            'black_swan': {
                'description': 'Black swan event (-5 sigma)',
                'return': base_mean - 5 * base_vol,
                'probability': stats.norm.cdf(-5),
                'portfolio_impact': (base_mean - 5 * base_vol) * self.portfolio_value
            },
            'recession': {
                'description': 'Recession scenario (sustained negative returns)',
                'return': -0.02,  # -2% daily return
                'probability': 0.01,  # Estimated probability
                'portfolio_impact': -0.02 * self.portfolio_value
            }
        }
        
        return scenarios
    
    def visualize(self, result: Optional[SimulationResult] = None) -> None:
        """Visualize VaR estimation results"""
        if result is None:
            result = self.result
        
        if result is None or self.returns_data is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Returns distribution with VaR levels
        axes[0,0].hist(self.returns_data, bins=50, alpha=0.7, color='blue', density=True)
        
        # Add VaR lines for 95% confidence level
        if 'parametric_var_95' in self.var_results:
            var_95_return = -self.var_results['parametric_var_95'] / self.portfolio_value
            axes[0,0].axvline(var_95_return, color='red', linestyle='--', linewidth=2, 
                             label=f'95% VaR ({var_95_return:.4f})')
        
        if 'parametric_var_99' in self.var_results:
            var_99_return = -self.var_results['parametric_var_99'] / self.portfolio_value
            axes[0,0].axvline(var_99_return, color='darkred', linestyle='--', linewidth=2,
                             label=f'99% VaR ({var_99_return:.4f})')
        
        axes[0,0].set_xlabel('Daily Returns')
        axes[0,0].set_ylabel('Probability Density')
        axes[0,0].set_title('Returns Distribution with VaR Levels')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: VaR comparison across methods
        methods = ['parametric', 'historical', 'monte_carlo', 'cornish_fisher']
        var_95_values = []
        var_99_values = []
        
        for method in methods:
            if f'{method}_var_95' in self.var_results:
                var_95_values.append(self.var_results[f'{method}_var_95'])
            else:
                var_95_values.append(0)
            
            if f'{method}_var_99' in self.var_results:
                var_99_values.append(self.var_results[f'{method}_var_99'])
            else:
                var_99_values.append(0)
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[0,1].bar(x - width/2, var_95_values, width, label='95% VaR', alpha=0.7)
        axes[0,1].bar(x + width/2, var_99_values, width, label='99% VaR', alpha=0.7)
        axes[0,1].set_xlabel('VaR Method')
        axes[0,1].set_ylabel('VaR ($)')
        axes[0,1].set_title('VaR Comparison Across Methods')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: VaR vs Expected Shortfall
        if self.es_results:
            es_95_values = []
            es_99_values = []
            
            for method in ['parametric', 'historical', 'monte_carlo']:
                if f'{method}_es_95' in self.es_results:
                    es_95_values.append(self.es_results[f'{method}_es_95'])
                else:
                    es_95_values.append(0)
                
                if f'{method}_es_99' in self.es_results:
                    es_99_values.append(self.es_results[f'{method}_es_99'])
                else:
                    es_99_values.append(0)
            
            x_es = np.arange(len(es_95_values))
            axes[0,2].bar(x_es - width/2, [self.var_results.get(f'{m}_var_95', 0) for m in ['parametric', 'historical', 'monte_carlo']], 
                         width, label='95% VaR', alpha=0.7)
            axes[0,2].bar(x_es + width/2, es_95_values, width, label='95% ES', alpha=0.7)
            axes[0,2].set_xlabel('Method')
            axes[0,2].set_ylabel('Risk Measure ($)')
            axes[0,2].set_title('VaR vs Expected Shortfall (95%)')
            axes[0,2].set_xticks(x_es)
            axes[0,2].set_xticklabels(['Parametric', 'Historical', 'Monte Carlo'])
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Time series of returns with VaR violations
        time_index = np.arange(len(self.returns_data))
        axes[1,0].plot(time_index, self.returns_data, alpha=0.7, color='blue', linewidth=0.5)
        
        if 'parametric_var_95' in self.var_results:
            var_threshold = -self.var_results['parametric_var_95'] / self.portfolio_value
            axes[1,0].axhline(var_threshold, color='red', linestyle='--', alpha=0.8, label='95% VaR')
            
            # Highlight violations
            violations = self.returns_data < var_threshold
            if np.any(violations):
                axes[1,0].scatter(time_index[violations], self.returns_data[violations], 
                                 color='red', s=20, alpha=0.8, label='VaR Violations')
        axes[1,0].set_xlabel('Time Period')
        axes[1,0].set_ylabel('Returns')
        axes[1,0].set_title('Returns Time Series with VaR Violations')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Backtesting results
        if self.backtesting_results:
            confidence_levels = [int(cl*100) for cl in self.confidence_levels]
            violation_rates = []
            expected_rates = []
            
            for conf in confidence_levels:
                if f'var_{conf}' in self.backtesting_results:
                    bt_result = self.backtesting_results[f'var_{conf}']
                    violation_rates.append(bt_result['violation_rate'] * 100)
                    expected_rates.append(bt_result['expected_violation_rate'] * 100)
                else:
                    violation_rates.append(0)
                    expected_rates.append(0)
            
            x_bt = np.arange(len(confidence_levels))
            axes[1,1].bar(x_bt - width/2, expected_rates, width, label='Expected Rate', alpha=0.7)
            axes[1,1].bar(x_bt + width/2, violation_rates, width, label='Actual Rate', alpha=0.7)
            axes[1,1].set_xlabel('Confidence Level (%)')
            axes[1,1].set_ylabel('Violation Rate (%)')
            axes[1,1].set_title('VaR Backtesting Results')
            axes[1,1].set_xticks(x_bt)
            axes[1,1].set_xticklabels([f'{cl}%' for cl in confidence_levels])
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Risk metrics summary
        summary_text = f"""
        Portfolio Parameters:
        • Portfolio Value: ${self.portfolio_value:,.0f}
        • Time Horizon: {self.time_horizon} day(s)
        • Data Points: {len(self.returns_data):,}
        • Simulation Method: {self.simulation_method.title()}
        
        VaR Estimates (95% / 99%):
        • Parametric: ${self.var_results.get('parametric_var_95', 0):,.0f} / ${self.var_results.get('parametric_var_99', 0):,.0f}
        • Historical: ${self.var_results.get('historical_var_95', 0):,.0f} / ${self.var_results.get('historical_var_99', 0):,.0f}
        • Monte Carlo: ${self.var_results.get('monte_carlo_var_95', 0):,.0f} / ${self.var_results.get('monte_carlo_var_99', 0):,.0f}
        • Cornish-Fisher: ${self.var_results.get('cornish_fisher_var_95', 0):,.0f} / ${self.var_results.get('cornish_fisher_var_99', 0):,.0f}
        
        Expected Shortfall (95% / 99%):
        • Parametric: ${self.es_results.get('parametric_es_95', 0):,.0f} / ${self.es_results.get('parametric_es_99', 0):,.0f}
        • Historical: ${self.es_results.get('historical_es_95', 0):,.0f} / ${self.es_results.get('historical_es_99', 0):,.0f}
        • Monte Carlo: ${self.es_results.get('monte_carlo_es_95', 0):,.0f} / ${self.es_results.get('monte_carlo_es_99', 0):,.0f}
        
        Risk Metrics:
        • Sharpe Ratio: {self.risk_metrics.get('sharpe_ratio', 0):.3f}
        • Sortino Ratio: {self.risk_metrics.get('sortino_ratio', 0):.3f}
        • Max Drawdown: {self.risk_metrics.get('max_drawdown', 0):.2%}
        • Tail Ratio: {self.risk_metrics.get('tail_ratio', 0):.2f}
        
        Data Statistics:
        • Mean Return: {result.statistics['data_statistics']['mean_return']:.4f}
        • Volatility: {result.statistics['data_statistics']['volatility']:.4f}
        • Skewness: {result.statistics['data_statistics']['skewness']:.3f}
        • Kurtosis: {result.statistics['data_statistics']['kurtosis']:.3f}
        """
        
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                      fontsize=8, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        axes[1,2].set_title('Risk Summary')
        
        plt.tight_layout()
        plt.show()
    
    def plot_stress_scenarios(self) -> None:
        """Create additional visualization for stress scenarios"""
        scenarios = self.generate_stress_scenarios()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Scenario impacts
        scenario_names = list(scenarios.keys())
        impacts = [scenarios[name]['portfolio_impact'] for name in scenario_names]
        probabilities = [scenarios[name]['probability'] for name in scenario_names]
        
        colors = ['red', 'orange', 'darkred', 'purple']
        bars = ax1.bar(scenario_names, impacts, color=colors, alpha=0.7)
        ax1.set_xlabel('Stress Scenario')
        ax1.set_ylabel('Portfolio Impact ($)')
        ax1.set_title('Stress Test Scenarios - Portfolio Impact')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, impact in zip(bars, impacts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${impact:,.0f}', ha='center', va='bottom' if height < 0 else 'top')
        
        # Plot 2: Probability vs Impact
        ax2.scatter(probabilities, np.abs(impacts), s=100, c=colors, alpha=0.7)
        for i, name in enumerate(scenario_names):
            ax2.annotate(name.replace('_', ' ').title(), 
                        (probabilities[i], abs(impacts[i])),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Absolute Impact ($)')
        ax2.set_title('Risk-Return Profile of Stress Scenarios')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def validate_parameters(self) -> List[str]:
        """Validate VaR simulation parameters"""
        errors = []
        
        if not all(0 < cl < 1 for cl in self.confidence_levels):
            errors.append("Confidence levels must be between 0 and 1")
        if self.time_horizon <= 0:
            errors.append("Time horizon must be positive")
        if self.portfolio_value <= 0:
            errors.append("Portfolio value must be positive")
        if self.n_simulations <= 0:
            errors.append("Number of simulations must be positive")
        if self.window_size <= 0:
            errors.append("Window size must be positive")
        if self.volatility <= 0:
            errors.append("Volatility must be positive")
        if self.simulation_method not in ['normal', 't', 'skewed_t']:
            errors.append("Simulation method must be 'normal', 't', or 'skewed_t'")
        if self.returns is not None and len(self.returns) < 30:
            errors.append("Returns data must have at least 30 observations")
        
        return errors
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'confidence_levels': {
                'type': 'list',
                'default': [0.95, 0.99],
                'description': 'Confidence levels for VaR calculation'
            },
            'time_horizon': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 30,
                'description': 'Time horizon in days'
            },
            'portfolio_value': {
                'type': 'float',
                'default': 1000000,
                'min': 1000,
                'max': 1000000000,
                'description': 'Portfolio value ($)'
            },
            'simulation_method': {
                'type': 'choice',
                'default': 'normal',
                'choices': ['normal', 't', 'skewed_t'],
                'description': 'Distribution for Monte Carlo simulation'
            },
            'n_simulations': {
                'type': 'int',
                'default': 100000,
                'min': 1000,
                'max': 1000000,
                'description': 'Number of Monte Carlo simulations'
            },
            'window_size': {
                'type': 'int',
                'default': 252,
                'min': 30,
                'max': 1000,
                'description': 'Rolling window size for backtesting'
            },
            'mean_return': {
                'type': 'float',
                'default': 0.0008,
                'min': -0.01,
                'max': 0.01,
                'description': 'Expected daily return (if simulating data)'
            },
            'volatility': {
                'type': 'float',
                'default': 0.02,
                'min': 0.001,
                'max': 0.1,
                'description': 'Daily volatility (if simulating data)'
            },
            'skewness': {
                'type': 'float',
                'default': 0.0,
                'min': -2.0,
                'max': 2.0,
                'description': 'Skewness parameter for distribution'
            },
            'kurtosis': {
                'type': 'float',
                'default': 3.0,
                'min': 2.0,
                'max': 10.0,
                'description': 'Kurtosis parameter for distribution'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
    
    def export_results(self, filename: str = 'var_results.csv') -> None:
        """Export VaR results to CSV file"""
        if not hasattr(self, 'result') or self.result is None:
            print("No results to export. Run the simulation first.")
            return
        
        # Prepare data for export
        export_data = []
        
        # VaR results
        for key, value in self.var_results.items():
            method, conf_level = key.rsplit('_', 1)
            export_data.append({
                'Metric': 'VaR',
                'Method': method.replace('_', ' ').title(),
                'Confidence_Level': f"{conf_level}%",
                'Value': value,
                'Relative_to_Portfolio': value / self.portfolio_value * 100
            })
        
        # Expected Shortfall results
        for key, value in self.es_results.items():
            method, conf_level = key.rsplit('_', 1)
            export_data.append({
                'Metric': 'Expected Shortfall',
                'Method': method.replace('_', ' ').title(),
                'Confidence_Level': f"{conf_level}%",
                'Value': value,
                'Relative_to_Portfolio': value / self.portfolio_value * 100
            })
        
        # Risk metrics
        for key, value in self.risk_metrics.items():
            export_data.append({
                'Metric': 'Risk Metric',
                'Method': key.replace('_', ' ').title(),
                'Confidence_Level': 'N/A',
                'Value': value,
                'Relative_to_Portfolio': 'N/A'
            })
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def compare_methods(self) -> dict:
        """Compare VaR methods and provide recommendations"""
        comparison = {}
        
        for conf_level in self.confidence_levels:
            conf_str = f"{int(conf_level*100)}"
            
            methods_comparison = {}
            
            # Get VaR values for comparison
            parametric_var = self.var_results.get(f'parametric_var_{conf_str}', 0)
            historical_var = self.var_results.get(f'historical_var_{conf_str}', 0)
            monte_carlo_var = self.var_results.get(f'monte_carlo_var_{conf_str}', 0)
            cf_var = self.var_results.get(f'cornish_fisher_var_{conf_str}', 0)
            
            # Calculate relative differences
            base_var = parametric_var  # Use parametric as base
            
            methods_comparison['parametric'] = {
                'value': parametric_var,
                'relative_diff': 0.0,
                'pros': ['Fast computation', 'Theoretical foundation', 'Easy to understand'],
                'cons': ['Assumes normality', 'May underestimate tail risk']
            }
            
            methods_comparison['historical'] = {
                'value': historical_var,
                'relative_diff': (historical_var - base_var) / base_var * 100 if base_var != 0 else 0,
                'pros': ['No distributional assumptions', 'Uses actual data', 'Captures fat tails'],
                'cons': ['Limited by historical data', 'May not reflect current conditions']
            }
            
            methods_comparison['monte_carlo'] = {
                'value': monte_carlo_var,
                'relative_diff': (monte_carlo_var - base_var) / base_var * 100 if base_var != 0 else 0,
                'pros': ['Flexible distributions', 'Can model complex scenarios', 'Robust estimates'],
                'cons': ['Computationally intensive', 'Requires parameter estimation']
            }
            
            methods_comparison['cornish_fisher'] = {
                'value': cf_var,
                'relative_diff': (cf_var - base_var) / base_var * 100 if base_var != 0 else 0,
                'pros': ['Adjusts for skewness/kurtosis', 'Better than normal assumption', 'Analytical solution'],
                'cons': ['May be unstable for extreme parameters', 'Still parametric approach']
            }
            
            # Determine recommended method based on data characteristics
            data_skew = abs(stats.skew(self.returns_data))
            data_kurt = stats.kurtosis(self.returns_data)
            
            if data_skew < 0.5 and abs(data_kurt) < 1:
                recommended = 'parametric'
                reason = 'Data appears normally distributed'
            elif len(self.returns_data) > 1000:
                recommended = 'historical'
                reason = 'Large dataset available, no distributional assumptions needed'
            elif data_skew > 1 or abs(data_kurt) > 2:
                recommended = 'cornish_fisher'
                reason = 'Data shows significant skewness or kurtosis'
            else:
                recommended = 'monte_carlo'
                reason = 'Flexible approach for moderate non-normality'
            
            comparison[f'{conf_str}%'] = {
                'methods': methods_comparison,
                'recommended_method': recommended,
                'recommendation_reason': reason,
                'data_characteristics': {
                    'skewness': data_skew,
                    'kurtosis': data_kurt,
                    'sample_size': len(self.returns_data)
                }
            }
        
        return comparison


# Now update the __init__.py file to include VaRSimulation
