import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class SQInventoryPolicy(BaseSimulation):
    """
    (s,Q) Inventory Policy Simulation - Continuous Review with Fixed Order Quantity
    
    This simulation models a continuous review inventory system where inventory is 
    monitored continuously, and when the inventory position drops to or below the 
    reorder point s, a fixed quantity Q is ordered. This is one of the most common 
    inventory control policies in practice.
    
    Mathematical Background:
    -----------------------
    - Inventory position = On-hand inventory + On-order inventory - Backorders
    - Reorder trigger: Inventory position ≤ s (reorder point)
    - Order quantity: Always Q units (fixed)
    - Lead time: L periods (deterministic or stochastic)
    - Demand during lead time: D_L ~ distribution with mean μ_L and variance σ²_L
    
    Policy Logic:
    ------------
    1. Monitor inventory position continuously
    2. When inventory position ≤ s, place order for Q units
    3. Order arrives after lead time L
    4. Satisfy demand from on-hand inventory
    5. Backorder if insufficient inventory
    
    Cost Structure:
    --------------
    - Holding cost: h per unit per period for positive inventory
    - Shortage cost: p per unit per period for backorders
    - Ordering cost: K per order placed
    - Total cost = Holding cost + Shortage cost + Ordering cost
    
    Performance Metrics:
    -------------------
    - Average inventory level
    - Service level (fill rate, cycle service level)
    - Total cost per period
    - Order frequency
    - Stockout frequency and duration
    - Inventory turnover ratio
    
    Applications:
    ------------
    - Retail inventory management
    - Manufacturing raw materials
    - Spare parts inventory
    - Distribution center operations
    - Supply chain optimization
    - Inventory policy comparison
    
    Parameters:
    -----------
    s : float, default=50
        Reorder point - inventory level that triggers new order
    Q : float, default=100
        Order quantity - fixed amount ordered each time
    demand_rate : float, default=10
        Average demand per time period
    demand_std : float, default=3
        Standard deviation of demand (for stochastic demand)
    lead_time : float, default=5
        Lead time for orders (periods)
    lead_time_std : float, default=1
        Standard deviation of lead time (0 for deterministic)
    holding_cost : float, default=1.0
        Holding cost per unit per period
    shortage_cost : float, default=10.0
        Shortage/backorder cost per unit per period
    ordering_cost : float, default=50.0
        Fixed cost per order placed
    initial_inventory : float, default=100
        Starting inventory level
    simulation_periods : int, default=1000
        Number of time periods to simulate
    random_seed : int, optional
        Seed for random number generator
    
    Attributes:
    -----------
    inventory_history : list
        Time series of inventory levels
    order_history : list
        Record of all orders placed (time, quantity)
    cost_history : list
        Breakdown of costs by period
    service_metrics : dict
        Service level calculations
    result : SimulationResult
        Complete simulation results
    
    Methods:
    --------
    configure(**kwargs) : bool
        Configure simulation parameters
    run(**kwargs) : SimulationResult
        Execute the inventory simulation
    visualize(result=None, show_details=True) : None
        Create visualizations of inventory behavior
    calculate_optimal_policy() : dict
        Calculate theoretical optimal (s,Q) values
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI
    
    Examples:
    ---------
    >>> # Basic (s,Q) policy simulation
    >>> sq_sim = SQInventoryPolicy(s=30, Q=80, demand_rate=12)
    >>> result = sq_sim.run()
    >>> print(f"Average inventory: {result.results['avg_inventory']:.2f}")
    >>> print(f"Service level: {result.results['service_level']:.2%}")
    
    >>> # High service level policy
    >>> sq_high = SQInventoryPolicy(s=60, Q=120, shortage_cost=50)
    >>> result = sq_high.run()
    >>> sq_high.visualize()
    
    >>> # Compare with optimal policy
    >>> sq_opt = SQInventoryPolicy()
    >>> optimal = sq_opt.calculate_optimal_policy()
    >>> print(f"Optimal s: {optimal['s_optimal']:.1f}")
    >>> print(f"Optimal Q: {optimal['Q_optimal']:.1f}")
    """

    def __init__(self, s: float = 50, Q: float = 100, demand_rate: float = 10,
                 demand_std: float = 3, lead_time: float = 5, lead_time_std: float = 1,
                 holding_cost: float = 1.0, shortage_cost: float = 10.0,
                 ordering_cost: float = 50.0, initial_inventory: float = 100,
                 simulation_periods: int = 1000, random_seed: Optional[int] = None):
        super().__init__("(s,Q) Inventory Policy")
        
        # Policy parameters
        self.s = s  # reorder point
        self.Q = Q  # order quantity
        
        # Demand parameters
        self.demand_rate = demand_rate
        self.demand_std = demand_std
        
        # Lead time parameters
        self.lead_time = lead_time
        self.lead_time_std = lead_time_std
        
        # Cost parameters
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.ordering_cost = ordering_cost
        
        # Simulation parameters
        self.initial_inventory = initial_inventory
        self.simulation_periods = simulation_periods
        
        # Store in parameters dict
        self.parameters.update({
            's': s, 'Q': Q, 'demand_rate': demand_rate, 'demand_std': demand_std,
            'lead_time': lead_time, 'lead_time_std': lead_time_std,
            'holding_cost': holding_cost, 'shortage_cost': shortage_cost,
            'ordering_cost': ordering_cost, 'initial_inventory': initial_inventory,
            'simulation_periods': simulation_periods, 'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Internal state for tracking
        self.inventory_history = []
        self.order_history = []
        self.cost_history = []
        self.service_metrics = {}
        self.is_configured = True
    
    def configure(self, **kwargs) -> bool:
        """Configure (s,Q) policy parameters"""
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute (s,Q) inventory simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Initialize simulation state
        inventory_level = self.initial_inventory
        inventory_position = self.initial_inventory
        outstanding_orders = []  # [(arrival_time, quantity), ...]
        
        # Tracking variables
        inventory_history = []
        order_history = []
        period_costs = []
        total_demand = 0
        total_shortage = 0
        periods_with_shortage = 0
        total_orders = 0
        
        for period in range(self.simulation_periods):
            # Process arriving orders
            arriving_orders = [order for order in outstanding_orders if order[0] <= period]
            for arrival_time, quantity in arriving_orders:
                inventory_level += quantity
                inventory_position += quantity
                outstanding_orders.remove((arrival_time, quantity))
            
            # Generate demand for this period
            if self.demand_std > 0:
                demand = max(0, np.random.normal(self.demand_rate, self.demand_std))
            else:
                demand = self.demand_rate
            
            total_demand += demand
            
            # Satisfy demand
            if inventory_level >= demand:
                inventory_level -= demand
            else:
                # Shortage occurs
                shortage = demand - inventory_level
                total_shortage += shortage
                periods_with_shortage += 1
                inventory_level = 0  # Can't have negative physical inventory
            
            # Update inventory position
            inventory_position = inventory_level + sum(q for t, q in outstanding_orders)
            
            # Check if we need to reorder
            if inventory_position <= self.s:
                # Place order
                if self.lead_time_std > 0:
                    actual_lead_time = max(0, np.random.normal(self.lead_time, self.lead_time_std))
                else:
                    actual_lead_time = self.lead_time
                
                arrival_time = period + actual_lead_time
                outstanding_orders.append((arrival_time, self.Q))
                order_history.append((period, self.Q, arrival_time))
                total_orders += 1
            
            # Calculate period costs
            holding_cost = self.holding_cost * max(0, inventory_level)
            shortage_cost = self.shortage_cost * max(0, demand - max(inventory_level, 0))
            ordering_cost = self.ordering_cost if inventory_position <= self.s else 0
            
            period_cost = holding_cost + shortage_cost + ordering_cost
            period_costs.append({
                'period': period,
                'holding': holding_cost,
                'shortage': shortage_cost,
                'ordering': ordering_cost,
                'total': period_cost
            })
            
            # Record inventory level
            inventory_history.append({
                'period': period,
                'inventory_level': inventory_level,
                'inventory_position': inventory_position,
                'demand': demand,
                'outstanding_orders': len(outstanding_orders)
            })
        
        # Calculate performance metrics
        avg_inventory = np.mean([h['inventory_level'] for h in inventory_history])
        avg_inventory_position = np.mean([h['inventory_position'] for h in inventory_history])
        
        # Service level metrics
        fill_rate = 1 - (total_shortage / total_demand) if total_demand > 0 else 1.0
        cycle_service_level = 1 - (periods_with_shortage / self.simulation_periods)
        
        # Cost metrics
        total_holding_cost = sum(c['holding'] for c in period_costs)
        total_shortage_cost = sum(c['shortage'] for c in period_costs)
        total_ordering_cost = sum(c['ordering'] for c in period_costs)
        total_cost = total_holding_cost + total_shortage_cost + total_ordering_cost
        avg_cost_per_period = total_cost / self.simulation_periods
        
        # Store results for visualization
        self.inventory_history = inventory_history
        self.order_history = order_history
        self.cost_history = period_costs
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'avg_inventory': avg_inventory,
                'avg_inventory_position': avg_inventory_position,
                'fill_rate': fill_rate,
                'cycle_service_level': cycle_service_level,
                'total_cost': total_cost,
                'avg_cost_per_period': avg_cost_per_period,
                'total_orders': total_orders,
                'order_frequency': total_orders / self.simulation_periods,
                'total_shortage': total_shortage,
                'shortage_periods': periods_with_shortage
            },
            statistics={
                'holding_cost_total': total_holding_cost,
                'shortage_cost_total': total_shortage_cost,
                'ordering_cost_total': total_ordering_cost,
                'inventory_turnover': total_demand / avg_inventory if avg_inventory > 0 else 0,
                'stockout_probability': periods_with_shortage / self.simulation_periods
            },
            execution_time=execution_time,
            convergence_data=[(i, sum(c['total'] for c in period_costs[:i+1])/(i+1)) 
                            for i in range(0, len(period_costs), max(1, len(period_costs)//100))]
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, show_details: bool = True) -> None:
        """Visualize (s,Q) inventory simulation results"""
        if result is None:
            result = self.result
        
        if result is None or not self.inventory_history:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Inventory levels over time
        periods = [h['period'] for h in self.inventory_history]
        inventory_levels = [h['inventory_level'] for h in self.inventory_history]
        inventory_positions = [h['inventory_position'] for h in self.inventory_history]
        
        axes[0,0].plot(periods, inventory_levels, 'b-', linewidth=1, label='Inventory Level')
        axes[0,0].plot(periods, inventory_positions, 'r--', linewidth=1, label='Inventory Position')
        axes[0,0].axhline(y=self.s, color='g', linestyle=':', linewidth=2, label=f'Reorder Point (s={self.s})')
        axes[0,0].set_xlabel('Period')
        axes[0,0].set_ylabel('Inventory')
        axes[0,0].set_title('Inventory Levels Over Time')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Cost breakdown
        cost_categories = ['holding', 'shortage', 'ordering']
        cost_totals = [sum(c[cat] for c in self.cost_history) for cat in cost_categories]
        colors = ['blue', 'red', 'green']
        
        axes[0,1].pie(cost_totals, labels=[f'{cat.title()}\n${total:.0f}' for cat, total in zip(cost_categories, cost_totals)],
                     colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Cost Breakdown')
        
        # Plot 3: Service level analysis
        demands = [h['demand'] for h in self.inventory_history]
        shortages = []
        for i, h in enumerate(self.inventory_history):
            if i > 0:  # Skip first period
                prev_inv = self.inventory_history[i-1]['inventory_level']
                shortage = max(0, h['demand'] - prev_inv)
                shortages.append(shortage)
            else:
                shortages.append(0)
        
        axes[1,0].hist(demands, bins=20, alpha=0.7, color='blue', label='Demand')
        axes[1,0].hist([s for s in shortages if s > 0], bins=10, alpha=0.7, color='red', label='Shortages')
        axes[1,0].set_xlabel('Quantity')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Demand and Shortage Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Performance summary
        metrics_text = f"""
        Policy Parameters:
        • Reorder Point (s): {self.s}
        • Order Quantity (Q): {self.Q}
        
        Performance Metrics:
        • Average Inventory: {result.results['avg_inventory']:.2f}
        • Fill Rate: {result.results['fill_rate']:.2%}
        • Service Level: {result.results['cycle_service_level']:.2%}
        • Total Cost: ${result.results['total_cost']:.2f}
        • Cost per Period: ${result.results['avg_cost_per_period']:.2f}
        • Order Frequency: {result.results['order_frequency']:.3f}
        • Inventory Turnover: {result.statistics['inventory_turnover']:.2f}
        """
        
        axes[1,1].text(0.05, 0.95, metrics_text, transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        axes[1,1].set_title('Performance Summary')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_optimal_policy(self) -> dict:
        """Calculate theoretical optimal (s,Q) policy parameters"""
        # Economic Order Quantity (EOQ)
        if self.demand_rate > 0 and self.holding_cost > 0:
            Q_optimal = np.sqrt(2 * self.ordering_cost * self.demand_rate / self.holding_cost)
        else:
            Q_optimal = self.Q
        
        # Optimal reorder point (assuming normal demand during lead time)
        demand_during_lead_time = self.demand_rate * self.lead_time
        std_demand_during_lead_time = self.demand_std * np.sqrt(self.lead_time)
        
        # Safety stock for desired service level (assuming 95% cycle service level)
        from scipy import stats
        safety_factor = stats.norm.ppf(0.95)  # 95% service level
        safety_stock = safety_factor * std_demand_during_lead_time
        s_optimal = demand_during_lead_time + safety_stock
        
        return {
            'Q_optimal': Q_optimal,
            's_optimal': s_optimal,
            'safety_stock': safety_stock,
            'expected_demand_lead_time': demand_during_lead_time,
            'std_demand_lead_time': std_demand_during_lead_time
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        if self.s < 0:
            errors.append("Reorder point (s) must be non-negative")
        if self.Q <= 0:
            errors.append("Order quantity (Q) must be positive")
        if self.demand_rate < 0:
            errors.append("Demand rate must be non-negative")
        if self.demand_std < 0:
            errors.append("Demand standard deviation must be non-negative")
        if self.lead_time < 0:
            errors.append("Lead time must be non-negative")
        if self.lead_time_std < 0:
            errors.append("Lead time standard deviation must be non-negative")
        if self.holding_cost < 0:
            errors.append("Holding cost must be non-negative")
        if self.shortage_cost < 0:
            errors.append("Shortage cost must be non-negative")
        if self.ordering_cost < 0:
            errors.append("Ordering cost must be non-negative")
        if self.simulation_periods <= 0:
            errors.append("Simulation periods must be positive")
        return errors
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            's': {
                'type': 'float',
                'default': 50,
                'min': 0,
                'max': 1000,
                'description': 'Reorder point'
            },
            'Q': {
                'type': 'float',
                'default': 100,
                'min': 1,
                'max': 1000,
                'description': 'Order quantity'
            },
            'demand_rate': {
                'type': 'float',
                'default': 10,
                'min': 0,
                'max': 100,
                'description': 'Average demand per period'
            },
            'demand_std': {
                'type': 'float',
                'default': 3,
                'min': 0,
                'max': 50,
                'description': 'Demand standard deviation'
            },
            'lead_time': {
                'type': 'float',
                'default': 5,
                'min': 0,
                'max': 50,
                'description': 'Lead time (periods)'
            },
            'lead_time_std': {
                'type': 'float',
                'default': 1,
                'min': 0,
                'max': 10,
                'description': 'Lead time standard deviation'
            },
            'holding_cost': {
                'type': 'float',
                'default': 1.0,
                'min': 0,
                'max': 100,
                'description': 'Holding cost per unit per period'
            },
            'shortage_cost': {
                'type': 'float',
                'default': 10.0,
                'min': 0,
                'max': 1000,
                'description': 'Shortage cost per unit per period'
            },
            'ordering_cost': {
                'type': 'float',
                'default': 50.0,
                'min': 0,
                'max': 1000,
                'description': 'Fixed ordering cost per order'
            },
            'initial_inventory': {
                'type': 'float',
                'default': 100,
                'min': 0,
                'max': 1000,
                'description': 'Initial inventory level'
            },
            'simulation_periods': {
                'type': 'int',
                'default': 1000,
                'min': 100,
                'max': 10000,
                'description': 'Number of periods to simulate'
            },
            'random_seed': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 2147483647,
                'description': 'Random seed for reproducibility (optional)'
            }
        }
