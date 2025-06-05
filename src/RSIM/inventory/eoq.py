import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Dict
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class EOQModel(BaseSimulation):
    """
    Economic Order Quantity (EOQ) Model Simulation
    
    The EOQ model is a classical inventory management model that determines the optimal 
    order quantity that minimizes the total cost of inventory management. It balances 
    ordering costs (fixed cost per order) with holding costs (cost of carrying inventory).
    
    Mathematical Background:
    -----------------------
    The EOQ model assumes:
    - Constant demand rate (D units per period)
    - Fixed ordering cost (K per order)
    - Linear holding cost (h per unit per period)
    - Instantaneous replenishment (zero lead time)
    - No stockouts allowed
    
    Total Cost Function:
    -------------------
    TC(Q) = (D/Q) × K + (Q/2) × h
    Where:
    - D/Q = number of orders per period
    - K = ordering cost per order
    - Q/2 = average inventory level
    - h = holding cost per unit per period
    
    Optimal Solution:
    ----------------
    Q* = √(2DK/h)  (Economic Order Quantity)
    
    At optimum:
    - Ordering cost = Holding cost
    - Total cost = √(2DKh)
    - Cycle time = Q*/D
    - Order frequency = D/Q*
    
    Key Performance Metrics:
    -----------------------
    - Total annual cost
    - Average inventory level
    - Order frequency
    - Cycle time
    - Inventory turnover ratio
    - Cost sensitivity analysis
    
    Model Extensions:
    ----------------
    - EOQ with quantity discounts
    - EOQ with planned shortages
    - EOQ with finite production rate
    - Stochastic EOQ with safety stock
    - Multi-item EOQ with constraints
    
    Applications:
    ------------
    - Manufacturing inventory planning
    - Retail replenishment strategies
    - Raw material procurement
    - Spare parts management
    - Distribution planning
    - Supply chain optimization
    
    Assumptions and Limitations:
    ---------------------------
    - Deterministic demand (constant rate)
    - Known and constant parameters
    - Infinite planning horizon
    - Single product focus
    - No capacity constraints
    - Perfect information
    
    Parameters:
    -----------
    demand_rate : float, default=1000
        Annual demand rate (units per year)
    ordering_cost : float, default=50
        Fixed cost per order placed
    holding_cost_rate : float, default=0.2
        Holding cost as fraction of unit cost per year
    unit_cost : float, default=10
        Cost per unit of inventory
    holding_cost_per_unit : float, default=None
        Direct holding cost per unit per year (overrides rate calculation)
    lead_time : float, default=0
        Lead time for orders (in same units as demand_rate)
    safety_stock : float, default=0
        Additional safety stock to maintain
    simulation_periods : int, default=10
        Number of EOQ cycles to simulate
    include_sensitivity : bool, default=True
        Whether to perform sensitivity analysis
    random_seed : int, optional
        Seed for random number generator (for extensions)
    
    Attributes:
    -----------
    optimal_quantity : float
        Calculated optimal order quantity (EOQ)
    optimal_cost : float
        Minimum total cost at EOQ
    reorder_point : float
        Inventory level triggering new order
    cycle_time : float
        Time between orders
    order_frequency : float
        Number of orders per time period
    inventory_profile : list
        Simulated inventory levels over time
    cost_breakdown : dict
        Detailed cost analysis
    sensitivity_results : dict
        Parameter sensitivity analysis results
    
    Methods:
    --------
    configure(**kwargs) : bool
        Configure EOQ model parameters
    run(**kwargs) : SimulationResult
        Execute EOQ analysis and simulation
    visualize(result=None, show_sensitivity=True) : None
        Create comprehensive EOQ visualizations
    calculate_eoq() : float
        Calculate optimal order quantity
    calculate_total_cost(Q) : float
        Calculate total cost for given order quantity
    sensitivity_analysis() : dict
        Perform sensitivity analysis on key parameters
    simulate_inventory_profile() : list
        Simulate inventory levels over time
    compare_policies(quantities) : dict
        Compare performance of different order quantities
    validate_parameters() : List[str]
        Validate model parameters
    get_parameter_info() : dict
        Get parameter information for UI
    
    Examples:
    ---------
    >>> # Basic EOQ calculation
    >>> eoq = EOQModel(demand_rate=1200, ordering_cost=25, unit_cost=5)
    >>> result = eoq.run()
    >>> print(f"Optimal order quantity: {result.results['optimal_quantity']:.0f}")
    >>> print(f"Minimum total cost: ${result.results['optimal_cost']:.2f}")
    
    >>> # High-volume manufacturing scenario
    >>> eoq_mfg = EOQModel(
    ...     demand_rate=50000, ordering_cost=200, unit_cost=15,
    ...     holding_cost_rate=0.25, lead_time=2
    ... )
    >>> result = eoq_mfg.run()
    >>> eoq_mfg.visualize()
    
    >>> # Compare different order quantities
    >>> eoq_compare = EOQModel(demand_rate=2000, ordering_cost=40, unit_cost=8)
    >>> quantities = [100, 200, 300, 400, 500]
    >>> comparison = eoq_compare.compare_policies(quantities)
    >>> print("Policy comparison completed")
    
    Theoretical Insights:
    --------------------
    - EOQ increases with √(demand) and √(ordering_cost)
    - EOQ decreases with √(holding_cost)
    - Total cost is relatively insensitive near the optimum
    - Square root relationship provides natural robustness
    - Equal ordering and holding costs at optimum
    
    Practical Considerations:
    ------------------------
    - Demand variability requires safety stock
    - Quantity discounts may override EOQ logic
    - Storage capacity constraints limit maximum Q
    - Cash flow considerations affect holding costs
    - Supplier minimum order quantities
    - Transportation economies of scale
    
    Cost Structure Analysis:
    -----------------------
    - Ordering costs: setup, administrative, transportation
    - Holding costs: capital, storage, insurance, obsolescence
    - Total cost curve is convex (single minimum)
    - Sensitivity decreases as parameters increase
    - Robust to moderate parameter estimation errors
    
    Extensions and Variations:
    -------------------------
    - EOQ with quantity discounts (price breaks)
    - EOQ with planned shortages (backorder costs)
    - Production EOQ (finite production rate)
    - Multi-product EOQ (shared constraints)
    - Stochastic EOQ (uncertain demand/lead time)
    - Dynamic EOQ (time-varying parameters)
    """

    def __init__(self, demand_rate: float = 1000, ordering_cost: float = 50,
                 holding_cost_rate: float = 0.2, unit_cost: float = 10,
                 holding_cost_per_unit: Optional[float] = None, lead_time: float = 0,
                 safety_stock: float = 0, simulation_periods: int = 10,
                 include_sensitivity: bool = True, random_seed: Optional[int] = None):
        super().__init__("Economic Order Quantity (EOQ) Model")
        
        # Core EOQ parameters
        self.demand_rate = demand_rate
        self.ordering_cost = ordering_cost
        self.holding_cost_rate = holding_cost_rate
        self.unit_cost = unit_cost
        self.holding_cost_per_unit = holding_cost_per_unit
        
        # Additional parameters
        self.lead_time = lead_time
        self.safety_stock = safety_stock
        self.simulation_periods = simulation_periods
        self.include_sensitivity = include_sensitivity
        
        # Store in parameters dict
        self.parameters.update({
            'demand_rate': demand_rate, 'ordering_cost': ordering_cost,
            'holding_cost_rate': holding_cost_rate, 'unit_cost': unit_cost,
            'holding_cost_per_unit': holding_cost_per_unit, 'lead_time': lead_time,
            'safety_stock': safety_stock, 'simulation_periods': simulation_periods,
            'include_sensitivity': include_sensitivity, 'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Calculated values
        self.optimal_quantity = None
        self.optimal_cost = None
        self.reorder_point = None
        self.cycle_time = None
        self.order_frequency = None
        self.inventory_profile = []
        self.cost_breakdown = {}
        self.sensitivity_results = {}
        self.is_configured = True
    
    def configure(self, **kwargs) -> bool:
        """Configure EOQ model parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value
        
        self.is_configured = True
        return True
    
    def calculate_eoq(self) -> float:
        """Calculate the Economic Order Quantity"""
        # Determine holding cost per unit
        if self.holding_cost_per_unit is not None:
            h = self.holding_cost_per_unit
        else:
            h = self.holding_cost_rate * self.unit_cost
        
        # EOQ formula: Q* = sqrt(2DK/h)
        if h <= 0:
            raise ValueError("Holding cost must be positive")
        
        self.optimal_quantity = np.sqrt(2 * self.demand_rate * self.ordering_cost / h)
        return self.optimal_quantity
    
    def calculate_total_cost(self, Q: float) -> float:
        """Calculate total cost for a given order quantity"""
        if Q <= 0:
            return np.inf
        
        # Determine holding cost per unit
        if self.holding_cost_per_unit is not None:
            h = self.holding_cost_per_unit
        else:
            h = self.holding_cost_rate * self.unit_cost
        
        # Total cost = Ordering cost + Holding cost
        ordering_cost_total = (self.demand_rate / Q) * self.ordering_cost
        holding_cost_total = (Q / 2 + self.safety_stock) * h
        
        return ordering_cost_total + holding_cost_total
    
    def simulate_inventory_profile(self) -> List[Dict]:
        """Simulate inventory levels over multiple EOQ cycles"""
        if self.optimal_quantity is None:
            self.calculate_eoq()
        
        profile = []
        current_inventory = self.optimal_quantity + self.safety_stock
        time = 0
        
        # Calculate daily demand rate
        daily_demand = self.demand_rate / 365  # Assuming annual demand rate
        
        for cycle in range(self.simulation_periods):
            cycle_start_time = time
            cycle_start_inventory = current_inventory
            
            # Simulate inventory depletion during cycle
            cycle_length = self.optimal_quantity / daily_demand
            
            for day in range(int(cycle_length) + 1):
                if current_inventory > self.safety_stock:
                    current_inventory -= daily_demand
                    current_inventory = max(current_inventory, self.safety_stock)
                
                profile.append({
                    'time': time,
                    'inventory_level': current_inventory,
                    'cycle': cycle,
                    'days_in_cycle': day
                })
                
                time += 1
                
                # Check if we need to reorder
                if current_inventory <= self.safety_stock + (self.lead_time * daily_demand):
                    # Order arrives after lead time
                    if day >= self.lead_time:
                        current_inventory += self.optimal_quantity
        
        self.inventory_profile = profile
        return profile
    
    def sensitivity_analysis(self) -> Dict:
        """Perform sensitivity analysis on key parameters"""
        if not self.include_sensitivity:
            return {}
        
        base_eoq = self.calculate_eoq()
        base_cost = self.calculate_total_cost(base_eoq)
        
        # Parameter ranges for sensitivity analysis
        param_ranges = {
            'demand_rate': np.linspace(self.demand_rate * 0.5, self.demand_rate * 1.5, 20),
            'ordering_cost': np.linspace(self.ordering_cost * 0.5, self.ordering_cost * 1.5, 20),
            'holding_cost_rate': np.linspace(max(0.01, self.holding_cost_rate * 0.5), 
                                           self.holding_cost_rate * 1.5, 20)
        }
        
        sensitivity_results = {}
        
        for param_name, param_values in param_ranges.items():
            results = []
            original_value = getattr(self, param_name)
            
            for param_value in param_values:
                setattr(self, param_name, param_value)
                try:
                    eoq = self.calculate_eoq()
                    cost = self.calculate_total_cost(eoq)
                    results.append({
                        'parameter_value': param_value,
                        'eoq': eoq,
                        'total_cost': cost,
                        'cost_change_percent': ((cost - base_cost) / base_cost) * 100
                    })
                except:
                    results.append({
                        'parameter_value': param_value,
                        'eoq': np.nan,
                        'total_cost': np.nan,
                        'cost_change_percent': np.nan
                    })
            
            # Restore original value
            setattr(self, param_name, original_value)
            sensitivity_results[param_name] = results
        
        self.sensitivity_results = sensitivity_results
        return sensitivity_results
    
    def compare_policies(self, quantities: List[float]) -> Dict:
        """Compare performance of different order quantities"""
        comparison_results = []
        
        for Q in quantities:
            total_cost = self.calculate_total_cost(Q)
            
            # Calculate performance metrics
            if self.holding_cost_per_unit is not None:
                h = self.holding_cost_per_unit
            else:
                h = self.holding_cost_rate * self.unit_cost
            
            ordering_cost_annual = (self.demand_rate / Q) * self.ordering_cost
            holding_cost_annual = (Q / 2 + self.safety_stock) * h
            cycle_time = Q / self.demand_rate * 365  # in days
            order_frequency = self.demand_rate / Q
            avg_inventory = Q / 2 + self.safety_stock
            inventory_turnover = self.demand_rate / avg_inventory if avg_inventory > 0 else 0
            
            comparison_results.append({
                'order_quantity': Q,
                'total_cost': total_cost,
                'ordering_cost_annual': ordering_cost_annual,
                'holding_cost_annual': holding_cost_annual,
                'cycle_time_days': cycle_time,
                'order_frequency': order_frequency,
                'avg_inventory': avg_inventory,
                'inventory_turnover': inventory_turnover,
                'cost_penalty_percent': ((total_cost - self.optimal_cost) / self.optimal_cost) * 100 if self.optimal_cost else 0
            })
        
        return {
            'quantities_compared': quantities,
            'results': comparison_results,
            'optimal_quantity': self.optimal_quantity,
            'optimal_cost': self.optimal_cost
        }
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute EOQ analysis and simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Calculate EOQ and related metrics
        self.optimal_quantity = self.calculate_eoq()
        self.optimal_cost = self.calculate_total_cost(self.optimal_quantity)
        
        # Calculate derived metrics
        if self.holding_cost_per_unit is not None:
            h = self.holding_cost_per_unit
        else:
            h = self.holding_cost_rate * self.unit_cost
        
        self.cycle_time = self.optimal_quantity / self.demand_rate
        self.order_frequency = self.demand_rate / self.optimal_quantity
        self.reorder_point = self.lead_time * self.demand_rate + self.safety_stock
        
        # Cost breakdown
        ordering_cost_annual = (self.demand_rate / self.optimal_quantity) * self.ordering_cost
        holding_cost_annual = (self.optimal_quantity / 2 + self.safety_stock) * h
        
        self.cost_breakdown = {
            'ordering_cost_annual': ordering_cost_annual,
            'holding_cost_annual': holding_cost_annual,
            'safety_stock_cost': self.safety_stock * h,
            'total_cost': self.optimal_cost
        }
        
        # Simulate inventory profile
        self.simulate_inventory_profile()
        
        # Perform sensitivity analysis
        if self.include_sensitivity:
            self.sensitivity_analysis()
        
        # Calculate additional performance metrics
        avg_inventory = self.optimal_quantity / 2 + self.safety_stock
        inventory_turnover = self.demand_rate / avg_inventory if avg_inventory > 0 else 0
        cycle_time_days = self.cycle_time * 365  # Convert to days if annual demand
        
        # Calculate cost per unit
        cost_per_unit = self.optimal_cost / self.demand_rate
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'optimal_quantity': self.optimal_quantity,
                'optimal_cost': self.optimal_cost,
                'cycle_time': self.cycle_time,
                'cycle_time_days': cycle_time_days,
                'order_frequency': self.order_frequency,
                'reorder_point': self.reorder_point,
                'avg_inventory': avg_inventory,
                'inventory_turnover': inventory_turnover,
                'cost_per_unit': cost_per_unit,
                'ordering_cost_annual': ordering_cost_annual,
                'holding_cost_annual': holding_cost_annual,
                'holding_cost_per_unit_calculated': h
            },
            statistics={
                'cost_breakdown': self.cost_breakdown,
                'inventory_profile_length': len(self.inventory_profile),
                'max_inventory': max([p['inventory_level'] for p in self.inventory_profile]) if self.inventory_profile else 0,
                'min_inventory': min([p['inventory_level'] for p in self.inventory_profile]) if self.inventory_profile else 0,
                'avg_inventory_simulated': np.mean([p['inventory_level'] for p in self.inventory_profile]) if self.inventory_profile else 0
            },
            execution_time=execution_time,
            convergence_data=[(i, self.calculate_total_cost(i)) for i in range(1, int(self.optimal_quantity * 2), max(1, int(self.optimal_quantity * 2 // 100)))]
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, show_sensitivity: bool = True) -> None:
        """Create comprehensive EOQ visualizations"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        # Determine number of subplots based on available data
        n_plots = 3 if (show_sensitivity and self.sensitivity_results) else 2
        if self.inventory_profile:
            n_plots += 1
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: Total Cost Curve
        Q_range = np.linspace(self.optimal_quantity * 0.2, self.optimal_quantity * 2, 200)
        total_costs = [self.calculate_total_cost(Q) for Q in Q_range]
        ordering_costs = [(self.demand_rate / Q) * self.ordering_cost for Q in Q_range]
        
        if self.holding_cost_per_unit is not None:
            h = self.holding_cost_per_unit
        else:
            h = self.holding_cost_rate * self.unit_cost
        
        holding_costs = [(Q / 2 + self.safety_stock) * h for Q in Q_range]
        
        axes[0].plot(Q_range, total_costs, 'b-', linewidth=2, label='Total Cost')
        axes[0].plot(Q_range, ordering_costs, 'r--', linewidth=1, label='Ordering Cost')
        axes[0].plot(Q_range, holding_costs, 'g--', linewidth=1, label='Holding Cost')
        axes[0].axvline(self.optimal_quantity, color='black', linestyle=':', linewidth=2, 
                       label=f'EOQ = {self.optimal_quantity:.0f}')
        axes[0].plot(self.optimal_quantity, self.optimal_cost, 'ro', markersize=8, 
                    label=f'Min Cost = ${self.optimal_cost:.2f}')
        
        axes[0].set_xlabel('Order Quantity')
        axes[0].set_ylabel('Annual Cost ($)')
        axes[0].set_title('EOQ Cost Analysis')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Cost Breakdown Pie Chart
        cost_labels = ['Ordering Cost', 'Holding Cost (Cycle Stock)', 'Safety Stock Cost']
        cost_values = [
            self.cost_breakdown['ordering_cost_annual'],
            self.cost_breakdown['holding_cost_annual'] - self.cost_breakdown['safety_stock_cost'],
            self.cost_breakdown['safety_stock_cost']
        ]
        
        # Remove zero values
        non_zero_costs = [(label, value) for label, value in zip(cost_labels, cost_values) if value > 0]
        if non_zero_costs:
            labels, values = zip(*non_zero_costs)
            colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(values)]
            
            axes[1].pie(values, labels=[f'{label}\n${value:.2f}' for label, value in zip(labels, values)],
                       colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Annual Cost Breakdown')
        
        # Plot 3: Inventory Profile (if simulated)
        if self.inventory_profile:
            times = [p['time'] for p in self.inventory_profile]
            inventory_levels = [p['inventory_level'] for p in self.inventory_profile]
            
            axes[2].plot(times, inventory_levels, 'b-', linewidth=2, label='Inventory Level')
            axes[2].axhline(self.safety_stock, color='r', linestyle='--', linewidth=1, 
                           label=f'Safety Stock = {self.safety_stock:.0f}')
            axes[2].axhline(self.reorder_point, color='orange', linestyle=':', linewidth=1,
                           label=f'Reorder Point = {self.reorder_point:.0f}')
            
            axes[2].set_xlabel('Time (days)')
            axes[2].set_ylabel('Inventory Level')
            axes[2].set_title('Inventory Profile Over Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].axis('off')
        
        # Plot 4: Performance Summary or Sensitivity Analysis
        if show_sensitivity and self.sensitivity_results:
            # Show sensitivity for demand rate
            if 'demand_rate' in self.sensitivity_results:
                sens_data = self.sensitivity_results['demand_rate']
                param_values = [r['parameter_value'] for r in sens_data if not np.isnan(r['eoq'])]
                eoq_values = [r['eoq'] for r in sens_data if not np.isnan(r['eoq'])]
                
                axes[3].plot(param_values, eoq_values, 'b-', linewidth=2, label='EOQ vs Demand Rate')
                axes[3].axvline(self.demand_rate, color='r', linestyle='--', linewidth=1,
                               label=f'Current Demand = {self.demand_rate:.0f}')
                axes[3].set_xlabel('Demand Rate')
                axes[3].set_ylabel('Optimal Order Quantity')
                axes[3].set_title('Sensitivity: EOQ vs Demand Rate')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
        else:
            # Performance summary
            summary_text = f"""
            EOQ Model Results:
            
            Optimal Order Quantity: {result.results['optimal_quantity']:.0f} units
            Minimum Total Cost: ${result.results['optimal_cost']:.2f}/year
            
            Cycle Metrics:
            • Cycle Time: {result.results['cycle_time_days']:.1f} days
            • Order Frequency: {result.results['order_frequency']:.2f} orders/year
            • Reorder Point: {result.results['reorder_point']:.0f} units
            
            Inventory Metrics:
            • Average Inventory: {result.results['avg_inventory']:.0f} units
            • Inventory Turnover: {result.results['inventory_turnover']:.2f} times/year
            • Cost per Unit: ${result.results['cost_per_unit']:.4f}
            
            Cost Structure:
            • Annual Ordering Cost: ${result.results['ordering_cost_annual']:.2f}
            • Annual Holding Cost: ${result.results['holding_cost_annual']:.2f}
            • Holding Cost Rate: {self.holding_cost_rate:.1%}
            """
            
            axes[3].text(0.05, 0.95, summary_text, transform=axes[3].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            axes[3].set_xlim(0, 1)
            axes[3].set_ylim(0, 1)
            axes[3].axis('off')
            axes[3].set_title('Performance Summary')
        
        plt.tight_layout()
        plt.show()
    
    def validate_parameters(self) -> List[str]:
        """Validate EOQ model parameters"""
        errors = []
        
        if self.demand_rate <= 0:
            errors.append("Demand rate must be positive")
        if self.ordering_cost < 0:
            errors.append("Ordering cost must be non-negative")
        if self.holding_cost_rate < 0:
            errors.append("Holding cost rate must be non-negative")
        if self.unit_cost < 0:
            errors.append("Unit cost must be non-negative")
        if self.holding_cost_per_unit is not None and self.holding_cost_per_unit < 0:
            errors.append("Holding cost per unit must be non-negative")
        if self.lead_time < 0:
            errors.append("Lead time must be non-negative")
        if self.safety_stock < 0:
            errors.append("Safety stock must be non-negative")
        if self.simulation_periods <= 0:
            errors.append("Simulation periods must be positive")
        
        # Check if holding cost can be calculated
        if self.holding_cost_per_unit is None and (self.holding_cost_rate == 0 or self.unit_cost == 0):
            errors.append("Either holding_cost_per_unit must be specified or both holding_cost_rate and unit_cost must be positive")
        
        return errors
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'demand_rate': {
                'type': 'float',
                'default': 1000,
                'min': 1,
                'max': 100000,
                'description': 'Annual demand rate (units/year)'
            },
            'ordering_cost': {
                'type': 'float',
                'default': 50,
                'min': 0,
                'max': 1000,
                'description': 'Fixed cost per order'
            },
            'holding_cost_rate': {
                'type': 'float',
                'default': 0.2,
                'min': 0,
                'max': 1.0,
                'description': 'Holding cost as fraction of unit cost'
            },
            'unit_cost': {
                'type': 'float',
                'default': 10,
                'min': 0,
                'max': 1000,
                'description': 'Cost per unit of inventory'
            },
            'holding_cost_per_unit': {
                'type': 'float',
                'default': None,
                'min': 0,
                'max': 100,
                'description': 'Direct holding cost per unit per year (optional)'
            },
            'lead_time': {
                'type': 'float',
                'default': 0,
                'min': 0,
                'max': 365,
                'description': 'Lead time for orders (days)'
            },
            'safety_stock': {
                'type': 'float',
                'default': 0,
                'min': 0,
                'max': 1000,
                'description': 'Safety stock level'
            },
            'simulation_periods': {
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 100,
                'description': 'Number of EOQ cycles to simulate'
            },
            'include_sensitivity': {
                'type': 'bool',
                'default': True,
                'description': 'Include sensitivity analysis'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }

