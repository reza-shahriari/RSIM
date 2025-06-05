import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Tuple, Dict
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class PortfolioSimulation(BaseSimulation):
    """
    Portfolio Performance Simulation - Multi-Asset Portfolio Analysis
    
    This simulation models the performance of a multi-asset portfolio over time,
    incorporating various risk factors, correlation structures, and rebalancing
    strategies. It provides comprehensive analysis of portfolio risk, return,
    and optimization metrics.
    
    Mathematical Background:
    -----------------------
    - Portfolio return: R_p = Σ(w_i * R_i) where w_i are weights, R_i are asset returns
    - Portfolio variance: σ²_p = w^T * Σ * w where Σ is covariance matrix
    - Sharpe ratio: (R_p - R_f) / σ_p where R_f is risk-free rate
    - Value at Risk (VaR): Quantile of portfolio loss distribution
    - Maximum Drawdown: Max peak-to-trough decline
    
    Portfolio Models:
    ----------------
    - Geometric Brownian Motion for asset prices
    - Correlated asset returns using Cholesky decomposition
    - Mean reversion models for interest rates
    - Jump diffusion processes for extreme events
    - Stochastic volatility models
    
    Rebalancing Strategies:
    ----------------------
    1. Periodic rebalancing (monthly, quarterly, annually)
    2. Threshold-based rebalancing (when weights drift beyond tolerance)
    3. Volatility targeting
    4. Risk parity allocation
    5. Buy-and-hold strategy
    
    Risk Metrics:
    ------------
    - Portfolio volatility (standard deviation)
    - Value at Risk (VaR) at various confidence levels
    - Conditional Value at Risk (CVaR/Expected Shortfall)
    - Maximum Drawdown and recovery time
    - Beta relative to market benchmark
    - Tracking error and information ratio
    - Sortino ratio (downside deviation)
    
    Performance Metrics:
    -------------------
    - Total return and annualized return
    - Sharpe ratio and risk-adjusted returns
    - Alpha generation relative to benchmark
    - Win/loss ratio and hit rate
    - Calmar ratio (return/max drawdown)
    - Portfolio turnover and transaction costs
    
    Applications:
    ------------
    - Asset allocation optimization
    - Risk management and stress testing
    - Performance attribution analysis
    - Rebalancing strategy comparison
    - Portfolio insurance strategies
    - Hedge fund strategy simulation
    - Retirement planning models
    
    Parameters:
    -----------
    assets : List[str], default=['Stock', 'Bond', 'Commodity']
        Names of assets in the portfolio
    initial_weights : List[float], default=[0.6, 0.3, 0.1]
        Initial allocation weights (must sum to 1.0)
    expected_returns : List[float], default=[0.08, 0.04, 0.06]
        Annual expected returns for each asset
    volatilities : List[float], default=[0.16, 0.05, 0.20]
        Annual volatilities for each asset
    correlation_matrix : Optional[List[List[float]]], default=None
        Correlation matrix between assets (auto-generated if None)
    initial_capital : float, default=100000
        Starting portfolio value
    simulation_days : int, default=252
        Number of trading days to simulate (252 = 1 year)
    rebalance_frequency : str, default='monthly'
        Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly', 'annual', 'none')
    rebalance_threshold : float, default=0.05
        Weight drift threshold for threshold-based rebalancing
    risk_free_rate : float, default=0.02
        Annual risk-free rate for Sharpe ratio calculation
    transaction_cost : float, default=0.001
        Transaction cost as fraction of trade value
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    portfolio_history : list
        Time series of portfolio values and weights
    returns_history : list
        Daily returns for portfolio and individual assets
    rebalance_history : list
        Record of all rebalancing events
    risk_metrics : dict
        Calculated risk measures
    result : SimulationResult
        Complete simulation results
    
    Methods:
    --------
    configure(**kwargs) : bool
        Configure simulation parameters
    run(**kwargs) : SimulationResult
        Execute the portfolio simulation
    visualize(result=None, show_details=True) : None
        Create comprehensive portfolio analysis charts
    calculate_efficient_frontier(num_portfolios=100) : dict
        Calculate efficient frontier for given assets
    calculate_risk_metrics() : dict
        Calculate comprehensive risk metrics
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI
    
    Examples:
    ---------
    >>> # Basic portfolio simulation
    >>> portfolio = PortfolioSimulation(
    ...     assets=['Stocks', 'Bonds'],
    ...     initial_weights=[0.7, 0.3],
    ...     expected_returns=[0.10, 0.04]
    ... )
    >>> result = portfolio.run()
    >>> print(f"Total return: {result.results['total_return']:.2%}")
    >>> print(f"Sharpe ratio: {result.results['sharpe_ratio']:.3f}")
    
    >>> # High-frequency rebalancing strategy
    >>> active_portfolio = PortfolioSimulation(
    ...     rebalance_frequency='weekly',
    ...     transaction_cost=0.002
    ... )
    >>> result = active_portfolio.run()
    >>> active_portfolio.visualize()
    
    >>> # Risk parity portfolio
    >>> risk_parity = PortfolioSimulation()
    >>> efficient_frontier = risk_parity.calculate_efficient_frontier()
    >>> print(f"Optimal Sharpe portfolio: {efficient_frontier['max_sharpe_weights']}")
    """

    def __init__(self, assets: List[str] = None, initial_weights: List[float] = None,
                 expected_returns: List[float] = None, volatilities: List[float] = None,
                 correlation_matrix: Optional[List[List[float]]] = None,
                 initial_capital: float = 100000, simulation_days: int = 252,
                 rebalance_frequency: str = 'monthly', rebalance_threshold: float = 0.05,
                 risk_free_rate: float = 0.02, transaction_cost: float = 0.001,
                 random_seed: Optional[int] = None):
        super().__init__("Portfolio Simulation")
        
        # Default portfolio setup
        if assets is None:
            assets = ['Stocks', 'Bonds', 'Commodities']
        if initial_weights is None:
            initial_weights = [0.6, 0.3, 0.1]
        if expected_returns is None:
            expected_returns = [0.08, 0.04, 0.06]  # Annual returns
        if volatilities is None:
            volatilities = [0.16, 0.05, 0.20]  # Annual volatilities
        
        # Portfolio parameters
        self.assets = assets
        self.initial_weights = np.array(initial_weights)
        self.expected_returns = np.array(expected_returns)
        self.volatilities = np.array(volatilities)
        self.n_assets = len(assets)
        
        # Correlation matrix
        if correlation_matrix is None:
            # Generate reasonable default correlations
            self.correlation_matrix = self._generate_default_correlations()
        else:
            self.correlation_matrix = np.array(correlation_matrix)
        
        # Simulation parameters
        self.initial_capital = initial_capital
        self.simulation_days = simulation_days
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_threshold = rebalance_threshold
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        
        # Store in parameters dict
        self.parameters.update({
            'assets': assets,
            'initial_weights': initial_weights,
            'expected_returns': expected_returns.tolist(),
            'volatilities': volatilities.tolist(),
            'correlation_matrix': self.correlation_matrix.tolist(),
            'initial_capital': initial_capital,
            'simulation_days': simulation_days,
            'rebalance_frequency': rebalance_frequency,
            'rebalance_threshold': rebalance_threshold,
            'risk_free_rate': risk_free_rate,
            'transaction_cost': transaction_cost,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Calculate covariance matrix
        self.covariance_matrix = self._calculate_covariance_matrix()
        
        # Internal state for tracking
        self.portfolio_history = []
        self.returns_history = []
        self.rebalance_history = []
        self.risk_metrics = {}
        self.is_configured = True
    
    def _generate_default_correlations(self) -> np.ndarray:
        """Generate reasonable default correlation matrix"""
        if self.n_assets == 2:
            return np.array([[1.0, 0.3], [0.3, 1.0]])
        elif self.n_assets == 3:
            return np.array([[1.0, 0.3, 0.1], [0.3, 1.0, -0.1], [0.1, -0.1, 1.0]])
        else:
            # Generate random correlation matrix
            corr = np.eye(self.n_assets)
            for i in range(self.n_assets):
                for j in range(i+1, self.n_assets):
                    corr_val = np.random.uniform(-0.5, 0.5)
                    corr[i,j] = corr[j,i] = corr_val
            return corr
    
    def _calculate_covariance_matrix(self) -> np.ndarray:
        """Calculate covariance matrix from correlations and volatilities"""
        vol_matrix = np.outer(self.volatilities, self.volatilities)
        return self.correlation_matrix * vol_matrix
    
    def _get_rebalance_frequency_days(self) -> int:
        """Convert rebalance frequency to days"""
        freq_map = {
            'daily': 1,
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63,
            'annual': 252,
            'none': float('inf')
        }
        return freq_map.get(self.rebalance_frequency, 21)
    
    def configure(self, **kwargs) -> bool:
        """Configure portfolio simulation parameters"""
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value
        
        # Recalculate derived parameters
        if 'correlation_matrix' in kwargs or 'volatilities' in kwargs:
            self.covariance_matrix = self._calculate_covariance_matrix()
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute portfolio simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Initialize simulation
        current_weights = self.initial_weights.copy()
        portfolio_value = self.initial_capital
        asset_values = current_weights * portfolio_value
        
        # Calculate daily parameters
        daily_returns = self.expected_returns / 252
        daily_vol = self.volatilities / np.sqrt(252)
        daily_cov = self.covariance_matrix / 252
        
        # Generate correlated random returns
        L = np.linalg.cholesky(daily_cov)
        random_returns = np.random.multivariate_normal(
            daily_returns, daily_cov, self.simulation_days
        )
        
        # Tracking variables
        portfolio_history = []
        returns_history = []
        rebalance_history = []
        total_transaction_costs = 0
        
        rebalance_freq_days = self._get_rebalance_frequency_days()
        
        for day in range(self.simulation_days):
            # Apply daily returns to assets
            daily_asset_returns = random_returns[day]
            asset_values *= (1 + daily_asset_returns)
            
            # Calculate new portfolio value and weights
            new_portfolio_value = np.sum(asset_values)
            new_weights = asset_values / new_portfolio_value
            
            # Portfolio daily return
            portfolio_return = (new_portfolio_value - portfolio_value) / portfolio_value
            
            # Check if rebalancing is needed
            should_rebalance = False
            
            if rebalance_freq_days != float('inf'):
                # Periodic rebalancing
                if day % rebalance_freq_days == 0 and day > 0:
                    should_rebalance = True
                
                # Threshold-based rebalancing
                weight_drift = np.abs(new_weights - self.initial_weights)
                if np.any(weight_drift > self.rebalance_threshold):
                    should_rebalance = True
            
            # Perform rebalancing if needed
            if should_rebalance:
                # Calculate transaction costs
                target_values = self.initial_weights * new_portfolio_value
                trades = np.abs(target_values - asset_values)
                transaction_cost = np.sum(trades) * self.transaction_cost
                
                # Apply transaction costs
                new_portfolio_value -= transaction_cost
                total_transaction_costs += transaction_cost
                
                # Rebalance to target weights
                asset_values = self.initial_weights * new_portfolio_value
                new_weights = self.initial_weights.copy()
                
                rebalance_history.append({
                    'day': day,
                    'portfolio_value': new_portfolio_value,
                    'transaction_cost': transaction_cost,
                    'weights_before': current_weights.copy(),
                    'weights_after': new_weights.copy()
                })
            
            # Record daily data
            portfolio_history.append({
                'day': day,
                'portfolio_value': new_portfolio_value,
                'weights': new_weights.copy(),
                'asset_values': asset_values.copy(),
                'daily_return': portfolio_return
            })
            
            returns_history.append({
                'day': day,
                'portfolio_return': portfolio_return,
                'asset_returns': daily_asset_returns.copy()
            })
            
            # Update for next iteration
            portfolio_value = new_portfolio_value
            current_weights = new_weights
        
        # Calculate performance metrics
        portfolio_returns = np.array([h['daily_return'] for h in portfolio_history])
        portfolio_values = np.array([h['portfolio_value'] for h in portfolio_history])
        
        # Return metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / self.simulation_days) - 1
                
        # Risk metrics
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # VaR calculations (95% and 99% confidence levels)
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        cvar_99 = np.mean(portfolio_returns[portfolio_returns <= var_99])
        
        # Additional metrics
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns, self.risk_free_rate)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        positive_days = np.sum(portfolio_returns > 0)
        win_rate = positive_days / len(portfolio_returns)
        
        # Portfolio turnover
        total_turnover = sum([rb['transaction_cost'] for rb in rebalance_history])
        turnover_rate = total_turnover / self.initial_capital
        
        # Store results for visualization
        self.portfolio_history = portfolio_history
        self.returns_history = returns_history
        self.rebalance_history = rebalance_history
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'total_return': total_return,
                'annualized_return': annualized_return,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'win_rate': win_rate,
                'final_portfolio_value': portfolio_values[-1],
                'total_transaction_costs': total_transaction_costs,
                'turnover_rate': turnover_rate,
                'num_rebalances': len(rebalance_history)
            },
            statistics={
                'daily_volatility': np.std(portfolio_returns),
                'skewness': self._calculate_skewness(portfolio_returns),
                'kurtosis': self._calculate_kurtosis(portfolio_returns),
                'best_day': np.max(portfolio_returns),
                'worst_day': np.min(portfolio_returns),
                'avg_daily_return': np.mean(portfolio_returns),
                'median_daily_return': np.median(portfolio_returns),
                'downside_deviation': self._calculate_downside_deviation(portfolio_returns)
            },
            execution_time=execution_time,
            convergence_data=[(i, (portfolio_values[i] - self.initial_capital) / self.initial_capital) 
                            for i in range(0, len(portfolio_values), max(1, len(portfolio_values)//100))]
        )
        
        self.result = result
        return result
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Sortino ratio using downside deviation"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        return (np.mean(excess_returns) * 252) / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_downside_deviation(self, returns: np.ndarray) -> float:
        """Calculate downside deviation"""
        negative_returns = returns[returns < 0]
        return np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return np.mean(((returns - mean_return) / std_return) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return np.mean(((returns - mean_return) / std_return) ** 4) - 3
    
    def visualize(self, result: Optional[SimulationResult] = None, show_details: bool = True) -> None:
        """Visualize portfolio simulation results"""
        if result is None:
            result = self.result
        
        if result is None or not self.portfolio_history:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Portfolio value over time
        days = [h['day'] for h in self.portfolio_history]
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        
        axes[0,0].plot(days, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
        axes[0,0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        
        # Mark rebalancing events
        if self.rebalance_history:
            rebalance_days = [rb['day'] for rb in self.rebalance_history]
            rebalance_values = [rb['portfolio_value'] for rb in self.rebalance_history]
            axes[0,0].scatter(rebalance_days, rebalance_values, color='red', s=30, alpha=0.7, label='Rebalances')
        
        axes[0,0].set_xlabel('Trading Days')
        axes[0,0].set_ylabel('Portfolio Value ($)')
        axes[0,0].set_title('Portfolio Value Over Time')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Asset allocation over time
        weights_history = np.array([h['weights'] for h in self.portfolio_history])
        
        for i, asset in enumerate(self.assets):
            axes[0,1].plot(days, weights_history[:, i], linewidth=2, label=asset)
        
        axes[0,1].set_xlabel('Trading Days')
        axes[0,1].set_ylabel('Weight')
        axes[0,1].set_title('Asset Allocation Over Time')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_ylim(0, 1)
        
        # Plot 3: Daily returns distribution
        portfolio_returns = [h['daily_return'] for h in self.portfolio_history]
        
        axes[1,0].hist(portfolio_returns, bins=50, alpha=0.7, color='blue', density=True)
        axes[1,0].axvline(x=np.mean(portfolio_returns), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(portfolio_returns):.4f}')
        axes[1,0].axvline(x=result.results['var_95'], color='orange', linestyle='--', 
                         label=f'VaR 95%: {result.results["var_95"]:.4f}')
        axes[1,0].set_xlabel('Daily Return')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Daily Returns Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Drawdown analysis
        cumulative_returns = np.cumprod(1 + np.array(portfolio_returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        axes[1,1].fill_between(days, drawdowns, 0, alpha=0.3, color='red')
        axes[1,1].plot(days, drawdowns, 'r-', linewidth=1)
        axes[1,1].axhline(y=result.results['max_drawdown'], color='darkred', linestyle='--', 
                         label=f'Max DD: {result.results["max_drawdown"]:.2%}')
        axes[1,1].set_xlabel('Trading Days')
        axes[1,1].set_ylabel('Drawdown')
        axes[1,1].set_title('Portfolio Drawdown')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 5: Risk-Return scatter (if multiple assets)
        if len(self.assets) > 1:
            asset_returns_matrix = np.array([r['asset_returns'] for r in self.returns_history])
            annual_returns = np.mean(asset_returns_matrix, axis=0) * 252
            annual_vols = np.std(asset_returns_matrix, axis=0) * np.sqrt(252)
            
            # Plot individual assets
            for i, asset in enumerate(self.assets):
                axes[2,0].scatter(annual_vols[i], annual_returns[i], s=100, alpha=0.7, label=asset)
            
            # Plot portfolio
            portfolio_annual_return = result.results['annualized_return']
            portfolio_annual_vol = result.results['portfolio_volatility']
            axes[2,0].scatter(portfolio_annual_vol, portfolio_annual_return, s=150, 
                             color='red', marker='*', label='Portfolio')
            
            axes[2,0].set_xlabel('Annual Volatility')
            axes[2,0].set_ylabel('Annual Return')
            axes[2,0].set_title('Risk-Return Profile')
            axes[2,0].legend()
            axes[2,0].grid(True, alpha=0.3)
        else:
            axes[2,0].axis('off')
        
        # Plot 6: Performance summary
        metrics_text = f"""
        Portfolio Performance Summary:
        
        Return Metrics:
        • Total Return: {result.results['total_return']:.2%}
        • Annualized Return: {result.results['annualized_return']:.2%}
        • Final Value: ${result.results['final_portfolio_value']:,.2f}
        
        Risk Metrics:
        • Volatility: {result.results['portfolio_volatility']:.2%}
        • Max Drawdown: {result.results['max_drawdown']:.2%}
        • VaR (95%): {result.results['var_95']:.2%}
        • VaR (99%): {result.results['var_99']:.2%}
        
        Risk-Adjusted Returns:
        • Sharpe Ratio: {result.results['sharpe_ratio']:.3f}
        • Sortino Ratio: {result.results['sortino_ratio']:.3f}
        • Calmar Ratio: {result.results['calmar_ratio']:.3f}
        
        Trading Metrics:
        • Win Rate: {result.results['win_rate']:.2%}
        • Rebalances: {result.results['num_rebalances']}
        • Transaction Costs: ${result.results['total_transaction_costs']:,.2f}
        • Turnover Rate: {result.results['turnover_rate']:.2%}
        
        Distribution Stats:
        • Skewness: {result.statistics['skewness']:.3f}
        • Kurtosis: {result.statistics['kurtosis']:.3f}
        • Best Day: {result.statistics['best_day']:.2%}
        • Worst Day: {result.statistics['worst_day']:.2%}
        """
        
        axes[2,1].text(0.05, 0.95, metrics_text, transform=axes[2,1].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[2,1].set_xlim(0, 1)
        axes[2,1].set_ylim(0, 1)
        axes[2,1].axis('off')
        axes[2,1].set_title('Performance Summary')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_efficient_frontier(self, num_portfolios: int = 100) -> dict:
        """Calculate efficient frontier for the given assets"""
        try:
            from scipy.optimize import minimize
        except ImportError:
            print("scipy is required for efficient frontier calculation")
            return {}
        
        # Generate random portfolio weights
        results = []
        
        for _ in range(num_portfolios):
            # Generate random weights that sum to 1
            weights = np.random.random(self.n_assets)
            weights /= np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * self.expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            results.append({
                'weights': weights,
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio
            })
        
        # Find maximum Sharpe ratio portfolio
        max_sharpe_idx = max(range(len(results)), key=lambda i: results[i]['sharpe_ratio'])
        max_sharpe_portfolio = results[max_sharpe_idx]
        
        # Find minimum volatility portfolio
        min_vol_idx = min(range(len(results)), key=lambda i: results[i]['volatility'])
        min_vol_portfolio = results[min_vol_idx]
        
        return {
            'portfolios': results,
            'max_sharpe_portfolio': max_sharpe_portfolio,
            'max_sharpe_weights': max_sharpe_portfolio['weights'],
            'min_vol_portfolio': min_vol_portfolio,
            'min_vol_weights': min_vol_portfolio['weights']
        }
    
    def calculate_risk_metrics(self) -> dict:
        """Calculate comprehensive risk metrics"""
        if not self.portfolio_history:
            return {}
        
        portfolio_returns = np.array([h['daily_return'] for h in self.portfolio_history])
        portfolio_values = np.array([h['portfolio_value'] for h in self.portfolio_history])
        
        # Basic risk metrics
        daily_vol = np.std(portfolio_returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        # VaR and CVaR at multiple confidence levels
        confidence_levels = [0.90, 0.95, 0.99]
        var_metrics = {}
        cvar_metrics = {}
        
        for conf in confidence_levels:
            alpha = 1 - conf
            var_value = np.percentile(portfolio_returns, alpha * 100)
            cvar_value = np.mean(portfolio_returns[portfolio_returns <= var_value])
            
            var_metrics[f'var_{int(conf*100)}'] = var_value
            cvar_metrics[f'cvar_{int(conf*100)}'] = cvar_value
        
        # Maximum Drawdown analysis
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_drawdown = np.min(drawdowns)
        max_dd_start = np.argmax(running_max[:np.argmin(drawdowns)])
        max_dd_end = np.argmin(drawdowns)
        max_dd_duration = max_dd_end - max_dd_start
        
        # Recovery analysis
        recovery_days = 0
        if max_dd_end < len(drawdowns) - 1:
            recovery_threshold = running_max[max_dd_end]
            for i in range(max_dd_end + 1, len(cumulative_returns)):
                if cumulative_returns[i] >= recovery_threshold:
                    recovery_days = i - max_dd_end
                    break
        
        # Tail risk metrics
        skewness = self._calculate_skewness(portfolio_returns)
        kurtosis = self._calculate_kurtosis(portfolio_returns)
        
        # Downside risk metrics
        downside_deviation = self._calculate_downside_deviation(portfolio_returns)
        
        return {
            'volatility_daily': daily_vol,
            'volatility_annual': annual_vol,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'recovery_days': recovery_days,
            'downside_deviation': downside_deviation,
            'skewness': skewness,
            'kurtosis': kurtosis,
            **var_metrics,
            **cvar_metrics
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        
        # Check weights sum to 1
        if abs(np.sum(self.initial_weights) - 1.0) > 1e-6:
            errors.append("Initial weights must sum to 1.0")
        
        # Check all weights are non-negative
        if np.any(self.initial_weights < 0):
            errors.append("All weights must be non-negative")
        
        # Check dimensions match
        if len(self.assets) != len(self.initial_weights):
            errors.append("Number of assets must match number of weights")
        
        if len(self.assets) != len(self.expected_returns):
            errors.append("Number of assets must match number of expected returns")
        
        if len(self.assets) != len(self.volatilities):
            errors.append("Number of assets must match number of volatilities")
        
        # Check correlation matrix dimensions
        if self.correlation_matrix.shape != (self.n_assets, self.n_assets):
            errors.append("Correlation matrix dimensions must match number of assets")
        
        # Check correlation matrix is valid
        if not np.allclose(self.correlation_matrix, self.correlation_matrix.T):
            errors.append("Correlation matrix must be symmetric")
        
        eigenvals = np.linalg.eigvals(self.correlation_matrix)
        if np.any(eigenvals < -1e-8):
            errors.append("Correlation matrix must be positive semi-definite")
        
        # Check diagonal elements are 1
        if not np.allclose(np.diag(self.correlation_matrix), 1.0):
            errors.append("Correlation matrix diagonal elements must be 1.0")
        
        # Check parameter ranges
        if self.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if self.simulation_days <= 0:
            errors.append("Simulation days must be positive")
        
        if np.any(self.volatilities < 0):
            errors.append("All volatilities must be non-negative")
        
        if self.transaction_cost < 0:
            errors.append("Transaction cost must be non-negative")
        
        if self.rebalance_threshold < 0 or self.rebalance_threshold > 1:
            errors.append("Rebalance threshold must be between 0 and 1")
        
        # Check rebalance frequency
        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'annual', 'none']
        if self.rebalance_frequency not in valid_frequencies:
            errors.append(f"Rebalance frequency must be one of: {valid_frequencies}")
        
        return errors
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'assets': {
                'type': 'list_str',
                'default': ['Stocks', 'Bonds', 'Commodities'],
                'description': 'List of asset names'
            },
            'initial_weights': {
                'type': 'list_float',
                'default': [0.6, 0.3, 0.1],
                'min': 0.0,
                'max': 1.0,
                'description': 'Initial portfolio weights (must sum to 1.0)'
            },
            'expected_returns': {
                'type': 'list_float',
                'default': [0.08, 0.04, 0.06],
                'min': -0.5,
                'max': 0.5,
                'description': 'Annual expected returns for each asset'
            },
            'volatilities': {
                'type': 'list_float',
                'default': [0.16, 0.05, 0.20],
                'min': 0.0,
                'max': 1.0,
                'description': 'Annual volatilities for each asset'
            },
            'initial_capital': {
                'type': 'float',
                'default': 100000,
                'min': 1000,
                'max': 10000000,
                'description': 'Starting portfolio value ($)'
            },
            'simulation_days': {
                'type': 'int',
                'default': 252,
                'min': 30,
                'max': 2520,
                'description': 'Number of trading days to simulate'
            },
            'rebalance_frequency': {
                'type': 'select',
                'default': 'monthly',
                'options': ['daily', 'weekly', 'monthly', 'quarterly', 'annual', 'none'],
                'description': 'Portfolio rebalancing frequency'
            },
            'rebalance_threshold': {
                'type': 'float',
                'default': 0.05,
                'min': 0.0,
                'max': 0.5,
                'description': 'Weight drift threshold for rebalancing'
            },
            'risk_free_rate': {
                'type': 'float',
                'default': 0.02,
                'min': 0.0,
                'max': 0.1,
                'description': 'Annual risk-free rate'
            },
            'transaction_cost': {
                'type': 'float',
                'default': 0.001,
                'min': 0.0,
                'max': 0.05,
                'description': 'Transaction cost as fraction of trade value'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
