import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Callable
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class NewsvendorModel(BaseSimulation):
    """
    Newsvendor Model Simulation - Single Period Inventory Decision
    
    The newsvendor model is a classical inventory optimization problem for perishable 
    goods or single-period decisions. A decision maker must choose an order quantity 
    before observing demand, balancing the costs of overstocking (unsold inventory) 
    and understocking (lost sales).
    
    Mathematical Background:
    -----------------------
    - Decision variable: Q (order quantity)
    - Random demand: D ~ F(d) with density f(d)
    - Unit cost: c (cost to purchase/produce one unit)
    - Selling price: p (revenue per unit sold)
    - Salvage value: s (value per unit of leftover inventory)
    - Shortage penalty: π (cost per unit of unmet demand)
    
    Profit Function:
    ---------------
    Profit(Q,D) = p·min(Q,D) + s·max(Q-D,0) - π·max(D-Q,0) - c·Q
    
    Expected Profit:
    ---------------
    E[Profit(Q)] = ∫[0 to Q] [pD + s(Q-D) - cQ] f(D)dD + ∫[Q to ∞] [pQ - π(D-Q) - cQ] f(D)dD
    
    Optimal Solution:
    ----------------
    The optimal order quantity Q* satisfies:
    F(Q*) = (p + π - c) / (p + π - s)
    
    This is the critical fractile, representing the optimal service level.
    
    Special Cases:
    -------------
    - No salvage value (s=0): F(Q*) = (p - c) / (p + π)
    - No shortage penalty (π=0): F(Q*) = (p - c) / (p - s)
    - Classic newsvendor (s=0, π=0): F(Q*) = (p - c) / p
    
    Applications:
    ------------
    - Newspaper/magazine ordering
    - Fashion retail buying
    - Seasonal product planning
    - Perishable goods management
    - Event planning (catering, seating)
    - Overbooking decisions
    - Portfolio optimization
    - Supply chain contracts
    
    Parameters:
    -----------
    unit_cost : float, default=5.0
        Cost to purchase/produce one unit
    selling_price : float, default=10.0
        Revenue per unit sold
    salvage_value : float, default=2.0
        Value per unit of leftover inventory
    shortage_penalty : float, default=3.0
        Cost per unit of unmet demand (goodwill loss)
    demand_distribution : str, default='normal'
        Demand distribution type ('normal', 'uniform', 'exponential', 'poisson')
    demand_mean : float, default=100
        Mean of demand distribution
    demand_std : float, default=20
        Standard deviation of demand (for applicable distributions)
    demand_min : float, default=50
        Minimum demand (for uniform distribution)
    demand_max : float, default=150
        Maximum demand (for uniform distribution)
    order_quantity : float, default=None
        Fixed order quantity (if None, optimal will be calculated)
    n_simulations : int, default=10000
        Number of demand scenarios to simulate
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    optimal_quantity : float
        Theoretically optimal order quantity
    critical_fractile : float
        Optimal service level (probability of not stocking out)
    demand_samples : np.ndarray
        Generated demand scenarios
    profit_samples : np.ndarray
        Profit for each demand scenario
    result : SimulationResult
        Complete simulation results
    
    Methods:
    --------
    configure(**kwargs) : bool
        Configure newsvendor parameters
    run(**kwargs) : SimulationResult
        Execute the newsvendor simulation
    visualize(result=None, show_distribution=True) : None
        Create visualizations of results and analysis
    calculate_optimal_quantity() : float
        Calculate theoretical optimal order quantity
    evaluate_quantity(Q) : dict
        Evaluate performance for a specific order quantity
    sensitivity_analysis(param_name, param_range) : dict
        Perform sensitivity analysis on a parameter
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI
    
    Examples:
    ---------
    >>> # Basic newsvendor problem
    >>> nv = NewsvendorModel(unit_cost=3, selling_price=8, salvage_value=1)
    >>> result = nv.run()
    >>> print(f"Optimal quantity: {result.results['optimal_quantity']:.1f}")
    >>> print(f"Expected profit: ${result.results['expected_profit']:.2f}")
    
    >>> # High-margin perishable goods
    >>> nv_perishable = NewsvendorModel(
    ...     unit_cost=2, selling_price=15, salvage_value=0, shortage_penalty=5,
    ...     demand_distribution='exponential', demand_mean=80
    ... )
    >>> result = nv_perishable.run()
    >>> nv_perishable.visualize()
    
    >>> # Sensitivity analysis
    >>> nv_sens = NewsvendorModel()
    >>> sensitivity = nv_sens.sensitivity_analysis('unit_cost', np.linspace(2, 8, 20))
    >>> print("Cost sensitivity completed")
    
    Theoretical Insights:
    --------------------
    - Higher shortage penalty → Higher optimal quantity
    - Higher salvage value → Higher optimal quantity  
    - Higher unit cost → Lower optimal quantity
    - More variable demand → May require safety stock adjustment
    - Critical fractile represents optimal stockout probability
    
    Risk Analysis:
    -------------
    - Profit variance increases with demand variability
    - Downside risk can be measured using Value-at-Risk (VaR)
    - Expected shortfall quantifies tail risk
    - Robust optimization can handle distribution uncertainty
    
    Extensions:
    ----------
    - Multi-product newsvendor with budget constraints
    - Newsvendor with price-dependent demand
    - Dynamic newsvendor over multiple periods
    - Newsvendor with supply uncertainty
    - Behavioral newsvendor with cognitive biases
    """

    def __init__(self, unit_cost: float = 5.0, selling_price: float = 10.0,
                 salvage_value: float = 2.0, shortage_penalty: float = 3.0,
                 demand_distribution: str = 'normal', demand_mean: float = 100,
                 demand_std: float = 20, demand_min: float = 50, demand_max: float = 150,
                 order_quantity: Optional[float] = None, n_simulations: int = 10000,
                 random_seed: Optional[int] = None):
        super().__init__("Newsvendor Model")
        
        # Cost parameters
        self.unit_cost = unit_cost
        self.selling_price = selling_price
        self.salvage_value = salvage_value
        self.shortage_penalty = shortage_penalty
        
        # Demand parameters
        self.demand_distribution = demand_distribution
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.demand_min = demand_min
        self.demand_max = demand_max
        
        # Decision parameters
        self.order_quantity = order_quantity
        self.n_simulations = n_simulations
        
        # Store in parameters dict
        self.parameters.update({
            'unit_cost': unit_cost, 'selling_price': selling_price,
            'salvage_value': salvage_value, 'shortage_penalty': shortage_penalty,
            'demand_distribution': demand_distribution, 'demand_mean': demand_mean,
            'demand_std': demand_std, 'demand_min': demand_min, 'demand_max': demand_max,
            'order_quantity': order_quantity, 'n_simulations': n_simulations,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state
        self.optimal_quantity = None
        self.critical_fractile = None
        self.demand_samples = None
        self.profit_samples = None
        self.is_configured = True
    
    def configure(self, **kwargs) -> bool:
        """Configure newsvendor parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value
        
        self.is_configured = True
        return True
    
    def calculate_optimal_quantity(self) -> float:
        """Calculate theoretical optimal order quantity"""
        # Critical fractile formula
        numerator = self.selling_price + self.shortage_penalty - self.unit_cost
        denominator = self.selling_price + self.shortage_penalty - self.salvage_value
        
        if denominator <= 0:
            raise ValueError("Invalid cost structure: denominator in critical fractile ≤ 0")
        
        self.critical_fractile = numerator / denominator
        
        # Find quantile based on demand distribution
        if self.demand_distribution == 'normal':
            from scipy import stats
            self.optimal_quantity = stats.norm.ppf(self.critical_fractile, 
                                                  self.demand_mean, self.demand_std)
        elif self.demand_distribution == 'uniform':
            self.optimal_quantity = self.demand_min + self.critical_fractile * (self.demand_max - self.demand_min)
        elif self.demand_distribution == 'exponential':
            self.optimal_quantity = -self.demand_mean * np.log(1 - self.critical_fractile)
        elif self.demand_distribution == 'poisson':
            from scipy import stats
            self.optimal_quantity = stats.poisson.ppf(self.critical_fractile, self.demand_mean)
        else:
            raise ValueError(f"Unsupported demand distribution: {self.demand_distribution}")
        
        return self.optimal_quantity
    
    def generate_demand_samples(self) -> np.ndarray:
        """Generate demand samples based on specified distribution"""
        if self.demand_distribution == 'normal':
            samples = np.random.normal(self.demand_mean, self.demand_std, self.n_simulations)
            samples = np.maximum(samples, 0)  # Ensure non-negative demand
        elif self.demand_distribution == 'uniform':
            samples = np.random.uniform(self.demand_min, self.demand_max, self.n_simulations)
        elif self.demand_distribution == 'exponential':
            samples = np.random.exponential(self.demand_mean, self.n_simulations)
        elif self.demand_distribution == 'poisson':
            samples = np.random.poisson(self.demand_mean, self.n_simulations)
        else:
            raise ValueError(f"Unsupported demand distribution: {self.demand_distribution}")
        
        return samples
    
    def calculate_profit(self, order_quantity: float, demand: np.ndarray) -> np.ndarray:
        """Calculate profit for given order quantity and demand scenarios"""
        # Units sold
        units_sold = np.minimum(order_quantity, demand)
        
        # Leftover inventory
        leftover = np.maximum(order_quantity - demand, 0)
        
        # Shortage
        shortage = np.maximum(demand - order_quantity, 0)
        
        # Profit calculation
        revenue = self.selling_price * units_sold
        salvage_revenue = self.salvage_value * leftover
        shortage_cost = self.shortage_penalty * shortage
        ordering_cost = self.unit_cost * order_quantity
        
        profit = revenue + salvage_revenue - shortage_cost - ordering_cost
        
        return profit
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute newsvendor simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Calculate optimal quantity if not specified
        if self.order_quantity is None:
            order_quantity = self.calculate_optimal_quantity()
        else:
            order_quantity = self.order_quantity
            self.calculate_optimal_quantity()  # Still calculate for comparison
        
        # Generate demand samples
        self.demand_samples = self.generate_demand_samples()
        
        # Calculate profits
        self.profit_samples = self.calculate_profit(order_quantity, self.demand_samples)
        
        # Calculate performance metrics
        expected_profit = np.mean(self.profit_samples)
        profit_std = np.std(self.profit_samples)
        
        # Service level metrics
        stockout_probability = np.mean(self.demand_samples > order_quantity)
        fill_rate = np.mean(np.minimum(order_quantity, self.demand_samples) / self.demand_samples)
        
        # Risk metrics
        profit_5th_percentile = np.percentile(self.profit_samples, 5)  # VaR at 95%
        profit_95th_percentile = np.percentile(self.profit_samples, 95)
        
        # Efficiency metrics
        expected_sales = np.mean(np.minimum(order_quantity, self.demand_samples))
        expected_leftover = np.mean(np.maximum(order_quantity - self.demand_samples, 0))
        expected_shortage = np.mean(np.maximum(self.demand_samples - order_quantity, 0))
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'order_quantity': order_quantity,
                'optimal_quantity': self.optimal_quantity,
                'critical_fractile': self.critical_fractile,
                'expected_profit': expected_profit,
                'profit_std': profit_std,
                'stockout_probability': stockout_probability,
                'fill_rate': fill_rate,
                'expected_sales': expected_sales,
                'expected_leftover': expected_leftover,
                'expected_shortage': expected_shortage,
                'profit_5th_percentile': profit_5th_percentile,
                'profit_95th_percentile': profit_95th_percentile
            },
            statistics={
                'profit_mean': expected_profit,
                'profit_variance': profit_std**2,
                'profit_coefficient_of_variation': profit_std / abs(expected_profit) if expected_profit != 0 else np.inf,
                'demand_mean': np.mean(self.demand_samples),
                'demand_std': np.std(self.demand_samples),
                'service_level': 1 - stockout_probability,
                'inventory_turnover': expected_sales / order_quantity if order_quantity > 0 else 0
            },
            execution_time=execution_time,
            convergence_data=[(i*100, np.mean(self.profit_samples[:i*100])) 
                            for i in range(1, min(101, self.n_simulations//100 + 1))]
        )
        
        self.result = result
        return result
    
    def evaluate_quantity(self, Q: float) -> dict:
        """Evaluate performance metrics for a specific order quantity"""
        if self.demand_samples is None:
            self.demand_samples = self.generate_demand_samples()
        
        profits = self.calculate_profit(Q, self.demand_samples)
        
        return {
            'order_quantity': Q,
            'expected_profit': np.mean(profits),
            'profit_std': np.std(profits),
            'stockout_probability': np.mean(self.demand_samples > Q),
            'fill_rate': np.mean(np.minimum(Q, self.demand_samples) / self.demand_samples),
            'expected_leftover': np.mean(np.maximum(Q - self.demand_samples, 0)),
            'expected_shortage': np.mean(np.maximum(self.demand_samples - Q, 0))
        }
    
    def sensitivity_analysis(self, param_name: str, param_range: np.ndarray) -> dict:
        """Perform sensitivity analysis on a parameter"""
        original_value = getattr(self, param_name)
        results = []
        
        for param_value in param_range:
            setattr(self, param_name, param_value)
            self.parameters[param_name] = param_value
            
            # Recalculate optimal quantity
            try:
                optimal_q = self.calculate_optimal_quantity()
                
                # Generate new demand samples if distribution parameters changed
                if param_name in ['demand_mean', 'demand_std', 'demand_min', 'demand_max', 'demand_distribution']:
                    demand_samples = self.generate_demand_samples()
                else:
                    demand_samples = self.demand_samples if self.demand_samples is not None else self.generate_demand_samples()
                
                profits = self.calculate_profit(optimal_q, demand_samples)
                expected_profit = np.mean(profits)
                
                results.append({
                    'parameter_value': param_value,
                    'optimal_quantity': optimal_q,
                    'expected_profit': expected_profit,
                    'critical_fractile': self.critical_fractile
                })
            except Exception as e:
                results.append({
                    'parameter_value': param_value,
                    'optimal_quantity': np.nan,
                    'expected_profit': np.nan,
                    'critical_fractile': np.nan,
                    'error': str(e)
                })
        
        # Restore original value
        setattr(self, param_name, original_value)
        self.parameters[param_name] = original_value
        
        return {
            'parameter_name': param_name,
            'parameter_range': param_range,
            'results': results
        }
    
    def visualize(self, result: Optional[SimulationResult] = None, show_distribution: bool = True) -> None:
        """Visualize newsvendor simulation results"""
        if result is None:
            result = self.result
        
        if result is None or self.demand_samples is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Demand and order quantity distribution
        axes[0,0].hist(self.demand_samples, bins=50, alpha=0.7, color='skyblue', 
                      density=True, label='Demand Distribution')
        axes[0,0].axvline(result.results['order_quantity'], color='red', linestyle='-', 
                         linewidth=2, label=f'Order Quantity: {result.results["order_quantity"]:.1f}')
        axes[0,0].axvline(result.results['optimal_quantity'], color='green', linestyle='--', 
                         linewidth=2, label=f'Optimal Quantity: {result.results["optimal_quantity"]:.1f}')
        axes[0,0].axvline(np.mean(self.demand_samples), color='orange', linestyle=':', 
                         linewidth=2, label=f'Mean Demand: {np.mean(self.demand_samples):.1f}')
        axes[0,0].set_xlabel('Quantity')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Demand Distribution and Order Quantities')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Profit distribution
        axes[0,1].hist(self.profit_samples, bins=50, alpha=0.7, color='lightgreen', 
                      density=True, label='Profit Distribution')
        axes[0,1].axvline(result.results['expected_profit'], color='red', linestyle='-', 
                         linewidth=2, label=f'Expected Profit: ${result.results["expected_profit"]:.2f}')
        axes[0,1].axvline(result.results['profit_5th_percentile'], color='orange', linestyle='--', 
                         linewidth=2, label=f'5th Percentile: ${result.results["profit_5th_percentile"]:.2f}')
        axes[0,1].axvline(result.results['profit_95th_percentile'], color='orange', linestyle='--', 
                         linewidth=2, label=f'95th Percentile: ${result.results["profit_95th_percentile"]:.2f}')
        axes[0,1].set_xlabel('Profit ($)')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('Profit Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Profit vs Order Quantity
        q_range = np.linspace(max(0, np.min(self.demand_samples) - 20), 
                             np.max(self.demand_samples) + 20, 100)
        expected_profits = []
        
        for q in q_range:
            profits = self.calculate_profit(q, self.demand_samples)
            expected_profits.append(np.mean(profits))
        
        axes[1,0].plot(q_range, expected_profits, 'b-', linewidth=2, label='Expected Profit')
        axes[1,0].axvline(result.results['order_quantity'], color='red', linestyle='-', 
                         linewidth=2, label=f'Current Q: {result.results["order_quantity"]:.1f}')
        axes[1,0].axvline(result.results['optimal_quantity'], color='green', linestyle='--', 
                         linewidth=2, label=f'Optimal Q: {result.results["optimal_quantity"]:.1f}')
        
        # Mark the maximum
        max_idx = np.argmax(expected_profits)
        axes[1,0].plot(q_range[max_idx], expected_profits[max_idx], 'ro', markersize=8, 
                      label=f'Max Profit: ${expected_profits[max_idx]:.2f}')
        
        axes[1,0].set_xlabel('Order Quantity')
        axes[1,0].set_ylabel('Expected Profit ($)')
        axes[1,0].set_title('Expected Profit vs Order Quantity')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Performance summary
        summary_text = f"""
        Problem Parameters:
        • Unit Cost: ${self.unit_cost:.2f}
        • Selling Price: ${self.selling_price:.2f}
        • Salvage Value: ${self.salvage_value:.2f}
        • Shortage Penalty: ${self.shortage_penalty:.2f}
        
        Demand Distribution: {self.demand_distribution.title()}
        • Mean: {self.demand_mean:.1f}
        • Std Dev: {self.demand_std:.1f}
        
        Results:
        • Order Quantity: {result.results['order_quantity']:.1f}
        • Optimal Quantity: {result.results['optimal_quantity']:.1f}
        • Critical Fractile: {result.results['critical_fractile']:.3f}
        • Expected Profit: ${result.results['expected_profit']:.2f}
        • Profit Std Dev: ${result.results['profit_std']:.2f}
        • Stockout Prob: {result.results['stockout_probability']:.2%}
        • Fill Rate: {result.results['fill_rate']:.2%}
        • Expected Sales: {result.results['expected_sales']:.1f}
        • Expected Leftover: {result.results['expected_leftover']:.1f}
        • Expected Shortage: {result.results['expected_shortage']:.1f}
        """
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        axes[1,1].set_title('Performance Summary')
        
        plt.tight_layout()
        plt.show()
    
    def validate_parameters(self) -> List[str]:
        """Validate newsvendor parameters"""
        errors = []
        
        if self.unit_cost < 0:
            errors.append("Unit cost must be non-negative")
        if self.selling_price <= 0:
            errors.append("Selling price must be positive")
        if self.salvage_value < 0:
            errors.append("Salvage value must be non-negative")
        if self.shortage_penalty < 0:
            errors.append("Shortage penalty must be non-negative")
        if self.selling_price <= self.unit_cost:
            errors.append("Selling price must be greater than unit cost")
        if self.salvage_value > self.unit_cost:
            errors.append("Salvage value should not exceed unit cost")
        
        # Check critical fractile validity
        try:
            numerator = self.selling_price + self.shortage_penalty - self.unit_cost
            denominator = self.selling_price + self.shortage_penalty - self.salvage_value
            if denominator <= 0:
                errors.append("Invalid cost structure: critical fractile denominator ≤ 0")
            elif numerator <= 0:
                errors.append("Invalid cost structure: critical fractile numerator ≤ 0")
        except:
            errors.append("Error calculating critical fractile")
        
        # Demand distribution parameters
        if self.demand_mean <= 0:
            errors.append("Demand mean must be positive")
        if self.demand_std < 0:
            errors.append("Demand standard deviation must be non-negative")
        if self.demand_distribution == 'uniform' and self.demand_min >= self.demand_max:
            errors.append("For uniform distribution, demand_min must be less than demand_max")
        
        if self.n_simulations <= 0:
            errors.append("Number of simulations must be positive")
        if self.n_simulations < 1000:
            errors.append("Number of simulations should be at least 1000 for reliable results")
        
        return errors
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'unit_cost': {
                'type': 'float',
                'default': 5.0,
                'min': 0,
                'max': 100,
                'description': 'Cost per unit purchased/produced'
            },
            'selling_price': {
                'type': 'float',
                'default': 10.0,
                'min': 0.1,
                'max': 200,
                'description': 'Revenue per unit sold'
            },
            'salvage_value': {
                'type': 'float',
                'default': 2.0,
                'min': 0,
                'max': 100,
                'description': 'Value per unit of leftover inventory'
            },
            'shortage_penalty': {
                'type': 'float',
                'default': 3.0,
                'min': 0,
                'max': 100,
                'description': 'Cost per unit of unmet demand'
            },
            'demand_distribution': {
                'type': 'choice',
                'default': 'normal',
                'choices': ['normal', 'uniform', 'exponential', 'poisson'],
                'description': 'Demand distribution type'
            },
            'demand_mean': {
                'type': 'float',
                'default': 100,
                'min': 1,
                'max': 1000,
                'description': 'Mean demand'
            },
            'demand_std': {
                'type': 'float',
                'default': 20,
                'min': 0,
                'max': 200,
                'description': 'Demand standard deviation'
            },
            'demand_min': {
                'type': 'float',
                'default': 50,
                'min': 0,
                'max': 500,
                'description': 'Minimum demand (uniform distribution)'
            },
            'demand_max': {
                'type': 'float',
                'default': 150,
                'min': 1,
                'max': 1000,
                'description': 'Maximum demand (uniform distribution)'
            },
            'order_quantity': {
                'type': 'float',
                'default': None,
                'min': 0,
                'max': 1000,
                'description': 'Order quantity (None for optimal)'
            },
            'n_simulations': {
                'type': 'int',
                'default': 10000,
                'min': 1000,
                'max': 100000,
                'description': 'Number of demand scenarios to simulate'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }

