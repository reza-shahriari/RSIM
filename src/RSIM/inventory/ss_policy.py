import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ..core.base import BaseSimulation, SimulationResult

class SSInventoryPolicy(BaseSimulation):
    """
    (s,S) Inventory Policy Simulation - Continuous Review with Variable Order Quantity
    
    This simulation models a continuous review inventory system where inventory is 
    monitored continuously, and when the inventory position drops to or below the 
    reorder point s, an order is placed to bring the inventory position up to the 
    order-up-to level S. This policy is also known as the "min-max" policy.
    
    Mathematical Background:
    -----------------------
    - Inventory position = On-hand inventory + On-order inventory - Backorders
    - Reorder trigger: Inventory position ≤ s (reorder point)
    - Order quantity: S - Inventory position (variable quantity)
    - Lead time: L periods (deterministic or stochastic)
    - Demand during lead time: D_L ~ distribution with mean μ_L and variance σ²_L
    
    Policy Logic:
    ------------
    1. Monitor inventory position continuously
    2. When inventory position ≤ s, place order for (S - current position) units
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
    - Average order quantity
    
    Applications:
    ------------
    - Retail inventory management with space constraints
    - Manufacturing with storage limitations
    - Perishable goods inventory
    - High-value item management
    - Warehouse management systems
    - Supply chain optimization
    
    Parameters:
    -----------
    s : float, default=20
        Reorder point - inventory level that triggers new order
    S : float, default=50
        Order-up-to level - target inventory position after ordering
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
    initial_inventory : float, default=40
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
        Calculate theoretical optimal (s,S) values
    validate_parameters() : List[str]
        Validate current parameters
    get_parameter_info() : dict
        Get parameter information for UI
    
    Examples:
    ---------
    >>> # Basic (s,S) policy simulation
    >>> ss_sim = SSInventoryPolicy(s=20, S=60, demand_rate=12)
    >>> result = ss_sim.run()
    >>> print(f"Average inventory: {result.results['avg_inventory']:.2f}")
    >>> print(f"Service level: {result.results['service_level']:.2%}")
    
    >>> # High service level policy
    >>> ss_high = SSInventoryPolicy(s=30, S=80, shortage_cost=50)
    >>> result = ss_high.run()
    >>> ss_high.visualize()
    
    >>> # Compare with optimal policy
    >>> ss_opt = SSInventoryPolicy()
    >>> optimal = ss_opt.calculate_optimal_policy()
    >>> print(f"Optimal s: {optimal['s_optimal']:.1f}")
    >>> print(f"Optimal S: {optimal['S_optimal']:.1f}")
    """

    def __init__(self, s: float = 20, S: float = 50, demand_rate: float = 10,
                 demand_std: float = 3, lead_time: float = 5, lead_time_std: float = 1,
                 holding_cost: float = 1.0, shortage_cost: float = 10.0,
                 ordering_cost: float = 50.0, initial_inventory: float = 40,
                 simulation_periods: int = 1000, random_seed: Optional[int] = None):
        super().__init__("(s,S) Inventory Policy")
        
        # Policy parameters
        self.s = s  # reorder point
        self.S = S  # order-up-to level
        
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
            's': s, 'S': S, 'demand_rate': demand_rate, 'demand_std': demand_std,
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
        """Configure (s,S) policy parameters"""
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.parameters[key] = value
        
        # Validate that S > s
        if self.S <= self.s:
            raise ValueError("Order-up-to level (S) must be greater than reorder point (s)")
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute (s,S) inventory simulation"""
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
        total_order_quantity = 0
        
        for period in range(self.simulation_periods):
            # Process arriving orders
            arriving_orders = [order for order in outstanding_orders if order[0] <= period]
            for arrival_time, quantity in arriving_orders:
                inventory_level += quantity
                outstanding_orders.remove((arrival_time, quantity))
            
            # Update inventory position
            inventory_position = inventory_level + sum(q for t, q in outstanding_orders)
            
            # Generate demand for this period
            if self.demand_std > 0:
                demand = max(0, np.random.normal(self.demand_rate, self.demand_std))
            else:
                demand = self.demand_rate
            
            total_demand += demand
            
            # Satisfy demand
            shortage = 0
            if inventory_level >= demand:
                inventory_level -= demand
            else:
                # Shortage occurs
                shortage = demand - inventory_level
                total_shortage += shortage
                periods_with_shortage += 1
                inventory_level = 0  # Can't have negative physical inventory
            
            # Update inventory position after demand
            inventory_position = inventory_level + sum(q for t, q in outstanding_orders)
            
            # Check if we need to reorder
            order_placed = False
            order_quantity = 0
            if inventory_position <= self.s:
                # Place order to bring position up to S
                order_quantity = self.S - inventory_position
                
                if self.lead_time_std > 0:
                    actual_lead_time = max(0, np.random.normal(self.lead_time, self.lead_time_std))
                else:
                    actual_lead_time = self.lead_time
                
                arrival_time = period + actual_lead_time
                outstanding_orders.append((arrival_time, order_quantity))
                order_history.append((period, order_quantity, arrival_time))
                total_orders += 1
                total_order_quantity += order_quantity
                order_placed = True
                
                # Update inventory position
                inventory_position += order_quantity
            
            # Calculate period costs
            holding_cost = self.holding_cost * max(0, inventory_level)
            shortage_cost = self.shortage_cost * shortage
            ordering_cost = self.ordering_cost if order_placed else 0
            
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
                'shortage': shortage,
                'order_quantity': order_quantity if order_placed else 0,
                'outstanding_orders': len(outstanding_orders)
            })
        
        # Calculate performance metrics
        avg_inventory = np.mean([h['inventory_level'] for h in inventory_history])
        avg_inventory_position = np.mean([h['inventory_position'] for h in inventory_history])
        avg_order_quantity = total_order_quantity / total_orders if total_orders > 0 else 0
        
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
                'avg_order_quantity': avg_order_quantity,
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
                'stockout_probability': periods_with_shortage / self.simulation_periods,
                'order_quantity_std': np.std([h['order_quantity'] for h in inventory_history if h['order_quantity'] > 0]) if total_orders > 0 else 0
            },
            execution_time=execution_time,
            convergence_data=[(i, sum(c['total'] for c in period_costs[:i+1])/(i+1)) 
                            for i in range(0, len(period_costs), max(1, len(period_costs)//100))]
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, show_details: bool = True) -> None:
        """Visualize (s,S) inventory simulation results"""
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
        axes[0,0].axhline(y=self.s, color='orange', linestyle=':', linewidth=2, label=f'Reorder Point (s={self.s})')
        axes[0,0].axhline(y=self.S, color='green', linestyle='-.', linewidth=2, label=f'Order-up-to Level (S={self.S})')
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
        
        # Plot 3: Order quantity distribution
        order_quantities = [h['order_quantity'] for h in self.inventory_history if h['order_quantity'] > 0]
        if order_quantities:
            axes[1,0].hist(order_quantities, bins=min(20, len(set(order_quantities))), 
                          alpha=0.7, color='purple', edgecolor='black')
            axes[1,0].axvline(x=np.mean(order_quantities), color='red', linestyle='--', 
                             linewidth=2, label=f'Mean: {np.mean(order_quantities):.1f}')
            axes[1,0].set_xlabel('Order Quantity')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Order Quantity Distribution')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5, 0.5, 'No orders placed', ha='center', va='center', 
                          transform=axes[1,0].transAxes, fontsize=14)
            axes[1,0].set_title('Order Quantity Distribution')
        
        # Plot 4: Performance summary
        metrics_text = f"""
        Policy Parameters:
        • Reorder Point (s): {self.s}
        • Order-up-to Level (S): {self.S}
        • Policy Range (S-s): {self.S - self.s}
        
        Performance Metrics:
        • Average Inventory: {result.results['avg_inventory']:.2f}
        • Average Order Qty: {result.results['avg_order_quantity']:.2f}
        • Fill Rate: {result.results['fill_rate']:.2%}
        • Service Level: {result.results['cycle_service_level']:.2%}
        • Total Cost: ${result.results['total_cost']:.2f}
        • Cost per Period: ${result.results['avg_cost_per_period']:.2f}
        • Order Frequency: {result.results['order_frequency']:.3f}
        • Inventory Turnover: {result.statistics['inventory_turnover']:.2f}
        • Stockout Probability: {result.statistics['stockout_probability']:.2%}
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
        
        # Additional detailed plots if requested
        if show_details:
            self._plot_detailed_analysis(result)
    
    def _plot_detailed_analysis(self, result: SimulationResult) -> None:
        """Create additional detailed analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Demand vs Shortage analysis
        demands = [h['demand'] for h in self.inventory_history]
        shortages = [h['shortage'] for h in self.inventory_history]
        
        axes[0,0].scatter(demands, shortages, alpha=0.6, s=20)
        axes[0,0].set_xlabel('Demand')
        axes[0,0].set_ylabel('Shortage')
        axes[0,0].set_title('Demand vs Shortage Analysis')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Inventory position control chart
        periods = [h['period'] for h in self.inventory_history]
        inventory_positions = [h['inventory_position'] for h in self.inventory_history]
        
        axes[0,1].plot(periods, inventory_positions, 'b-', linewidth=1)
        axes[0,1].axhline(y=self.s, color='red', linestyle='--', label=f's = {self.s}')
        axes[0,1].axhline(y=self.S, color='green', linestyle='--', label=f'S = {self.S}')
        axes[0,1].fill_between(periods, self.s, self.S, alpha=0.2, color='yellow', label='Control Band')
        axes[0,1].set_xlabel('Period')
        axes[0,1].set_ylabel('Inventory Position')
        axes[0,1].set_title('Inventory Position Control Chart')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Cost evolution over time
        periods = [c['period'] for c in self.cost_history]
        cumulative_costs = np.cumsum([c['total'] for c in self.cost_history])
        avg_costs = cumulative_costs / (np.array(periods) + 1)
        
        axes[1,0].plot(periods, avg_costs, 'r-', linewidth=2)
        axes[1,0].set_xlabel('Period')
        axes[1,0].set_ylabel('Average Cost per Period')
        axes[1,0].set_title('Cost Convergence')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Service level over time (rolling window)
        window_size = min(100, len(self.inventory_history) // 10)
        if window_size > 10:
            rolling_service_levels = []
            rolling_periods = []
            
            for i in range(window_size, len(self.inventory_history)):
                window_data = self.inventory_history[i-window_size:i]
                shortage_periods = sum(1 for h in window_data if h['shortage'] > 0)
                service_level = 1 - (shortage_periods / window_size)
                rolling_service_levels.append(service_level)
                rolling_periods.append(i)
            
            axes[1,1].plot(rolling_periods, rolling_service_levels, 'g-', linewidth=2)
            axes[1,1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Target')
            axes[1,1].set_xlabel('Period')
            axes[1,1].set_ylabel('Rolling Service Level')
            axes[1,1].set_title(f'Service Level Evolution (Window: {window_size})')
            axes[1,1].set_ylim(0, 1)
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Insufficient data\nfor rolling analysis', 
                          ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
            axes[1,1].set_title('Service Level Evolution')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_optimal_policy(self) -> dict:
        """Calculate theoretical optimal (s,S) policy parameters"""
        # For (s,S) policy, we use approximation methods
        # Based on demand during lead time and desired service level
        
        demand_during_lead_time = self.demand_rate * self.lead_time
        std_demand_during_lead_time = self.demand_std * np.sqrt(self.lead_time)
        
        # Calculate optimal order quantity using EOQ as starting point
        if self.demand_rate > 0 and self.holding_cost > 0:
            EOQ = np.sqrt(2 * self.ordering_cost * self.demand_rate / self.holding_cost)
        else:
            EOQ = 30  # Default fallback
        
        # Safety stock for desired service level (assuming 95% cycle service level)
        try:
            from scipy import stats
            safety_factor = stats.norm.ppf(0.95)  # 95% service level
        except ImportError:
            safety_factor = 1.645  # Approximate value for 95%
        
        safety_stock = safety_factor * std_demand_during_lead_time
        
        # Optimal reorder point
        s_optimal = demand_during_lead_time + safety_stock
        
        # Optimal order-up-to level (heuristic approach)
        # S should be large enough to cover demand during lead time plus review period
        review_period_demand = EOQ / self.demand_rate if self.demand_rate > 0 else 10
        S_optimal = s_optimal + EOQ
        
        # Alternative calculation based on cost minimization
        # This is a simplified approach - more complex optimization could be used
        holding_shortage_ratio = self.shortage_cost / self.holding_cost if self.holding_cost > 0 else 10
        adjustment_factor = np.sqrt(holding_shortage_ratio) / 10  # Heuristic adjustment
        
        S_optimal_alt = demand_during_lead_time + safety_stock + EOQ * (1 + adjustment_factor)
        
        return {
            's_optimal': s_optimal,
            'S_optimal': S_optimal,
            'S_optimal_alternative': S_optimal_alt,
            'EOQ_reference': EOQ,
            'safety_stock': safety_stock,
            'expected_demand_lead_time': demand_during_lead_time,
            'std_demand_lead_time': std_demand_during_lead_time,
            'policy_range': S_optimal - s_optimal,
            'review_period_estimate': review_period_demand
        }
    
    def validate_parameters(self) -> List[str]:
        """Validate simulation parameters"""
        errors = []
        if self.s < 0:
            errors.append("Reorder point (s) must be non-negative")
        if self.S <= 0:
            errors.append("Order-up-to level (S) must be positive")
        if self.S <= self.s:
            errors.append("Order-up-to level (S) must be greater than reorder point (s)")
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
        if self.initial_inventory < 0:
            errors.append("Initial inventory must be non-negative")
        return errors
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            's': {
                'type': 'float',
                'default': 20,
                'min': 0,
                'max': 1000,
                'description': 'Reorder point'
            },
            'S': {
                'type': 'float',
                'default': 50,
                'min': 1,
                'max': 1000,
                'description': 'Order-up-to level (must be > s)'
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
                'default': 40,
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
    
    def compare_with_sq_policy(self, sq_policy_params: dict) -> dict:
        """Compare (s,S) policy performance with equivalent (s,Q) policy"""
        # This method allows comparison between the two inventory policies
        # Run current (s,S) simulation
        ss_result = self.run()
        
        # Import and run equivalent (s,Q) policy
        try:
            from .sq_policy import SQInventoryPolicy
            
            # Create (s,Q) policy with similar parameters
            sq_sim = SQInventoryPolicy(
                s=sq_policy_params.get('s', self.s),
                Q=sq_policy_params.get('Q', self.S - self.s),  # Use policy range as Q
                demand_rate=self.demand_rate,
                demand_std=self.demand_std,
                lead_time=self.lead_time,
                lead_time_std=self.lead_time_std,
                holding_cost=self.holding_cost,
                shortage_cost=self.shortage_cost,
                ordering_cost=self.ordering_cost,
                initial_inventory=self.initial_inventory,
                simulation_periods=self.simulation_periods,
                random_seed=self.parameters.get('random_seed')
            )
            
            sq_result = sq_sim.run()
            
            # Compare key metrics
            comparison = {
                'ss_policy': {
                    'avg_inventory': ss_result.results['avg_inventory'],
                    'avg_cost_per_period': ss_result.results['avg_cost_per_period'],
                    'fill_rate': ss_result.results['fill_rate'],
                    'cycle_service_level': ss_result.results['cycle_service_level'],
                    'order_frequency': ss_result.results['order_frequency'],
                    'avg_order_quantity': ss_result.results['avg_order_quantity']
                },
                'sq_policy': {
                    'avg_inventory': sq_result.results['avg_inventory'],
                    'avg_cost_per_period': sq_result.results['avg_cost_per_period'],
                    'fill_rate': sq_result.results['fill_rate'],
                    'cycle_service_level': sq_result.results['cycle_service_level'],
                    'order_frequency': sq_result.results['order_frequency'],
                    'avg_order_quantity': sq_result.results.get('avg_inventory', 0)  # Approximation
                },
                'differences': {
                    'inventory_diff': ss_result.results['avg_inventory'] - sq_result.results['avg_inventory'],
                    'cost_diff': ss_result.results['avg_cost_per_period'] - sq_result.results['avg_cost_per_period'],
                    'service_diff': ss_result.results['fill_rate'] - sq_result.results['fill_rate']
                }
            }
            
            return comparison
            
        except ImportError:
            return {'error': 'SQ policy module not available for comparison'}
    
    def sensitivity_analysis(self, parameter: str, values: List[float]) -> dict:
        """Perform sensitivity analysis on a specific parameter"""
        if parameter not in self.parameters:
            raise ValueError(f"Parameter '{parameter}' not found in simulation parameters")
        
        original_value = getattr(self, parameter)
        results = []
        
        for value in values:
            # Set new parameter value
            setattr(self, parameter, value)
            self.parameters[parameter] = value
            
            # Run simulation
            try:
                result = self.run()
                results.append({
                    'parameter_value': value,
                    'avg_inventory': result.results['avg_inventory'],
                    'avg_cost_per_period': result.results['avg_cost_per_period'],
                    'fill_rate': result.results['fill_rate'],
                    'cycle_service_level': result.results['cycle_service_level'],
                    'total_cost': result.results['total_cost']
                })
            except Exception as e:
                results.append({
                    'parameter_value': value,
                    'error': str(e)
                })
        
        # Restore original value
        setattr(self, parameter, original_value)
        self.parameters[parameter] = original_value
        
        return {
            'parameter': parameter,
            'original_value': original_value,
            'sensitivity_results': results
        }
    
    def plot_sensitivity_analysis(self, sensitivity_results: dict) -> None:
        """Plot sensitivity analysis results"""
        results = sensitivity_results['sensitivity_results']
        parameter = sensitivity_results['parameter']
        
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        values = [r['parameter_value'] for r in valid_results]
        costs = [r['avg_cost_per_period'] for r in valid_results]
        inventories = [r['avg_inventory'] for r in valid_results]
        service_levels = [r['fill_rate'] for r in valid_results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Cost sensitivity
        axes[0].plot(values, costs, 'bo-', linewidth=2, markersize=6)
        axes[0].set_xlabel(f'{parameter}')
        axes[0].set_ylabel('Average Cost per Period')
        axes[0].set_title(f'Cost Sensitivity to {parameter}')
        axes[0].grid(True, alpha=0.3)
        
        # Inventory sensitivity
        axes[1].plot(values, inventories, 'go-', linewidth=2, markersize=6)
        axes[1].set_xlabel(f'{parameter}')
        axes[1].set_ylabel('Average Inventory')
        axes[1].set_title(f'Inventory Sensitivity to {parameter}')
        axes[1].grid(True, alpha=0.3)
        
        # Service level sensitivity
        axes[2].plot(values, service_levels, 'ro-', linewidth=2, markersize=6)
        axes[2].set_xlabel(f'{parameter}')
        axes[2].set_ylabel('Fill Rate')
        axes[2].set_title(f'Service Level Sensitivity to {parameter}')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, result: Optional[SimulationResult] = None) -> str:
        """Generate a comprehensive text report of simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            return "No simulation results available. Run the simulation first."
        
        report = f"""
{'='*60}
(s,S) INVENTORY POLICY SIMULATION REPORT
{'='*60}

SIMULATION PARAMETERS:
{'-'*30}
Policy Parameters:
  • Reorder Point (s): {self.s}
  • Order-up-to Level (S): {self.S}
  • Policy Range (S-s): {self.S - self.s}

Demand Parameters:
  • Average Demand Rate: {self.demand_rate} units/period
  • Demand Std Deviation: {self.demand_std} units
  • Coefficient of Variation: {self.demand_std/self.demand_rate:.2%}

Lead Time Parameters:
  • Average Lead Time: {self.lead_time} periods
  • Lead Time Std Deviation: {self.lead_time_std} periods

Cost Parameters:
  • Holding Cost: ${self.holding_cost}/unit/period
  • Shortage Cost: ${self.shortage_cost}/unit/period
  • Ordering Cost: ${self.ordering_cost}/order
  • Cost Ratio (Shortage/Holding): {self.shortage_cost/self.holding_cost:.1f}

Simulation Settings:
  • Simulation Periods: {self.simulation_periods}
  • Initial Inventory: {self.initial_inventory} units
  • Execution Time: {result.execution_time:.3f} seconds

PERFORMANCE RESULTS:
{'-'*30}
Inventory Metrics:
  • Average Inventory Level: {result.results['avg_inventory']:.2f} units
  • Average Inventory Position: {result.results['avg_inventory_position']:.2f} units
  • Inventory Turnover Ratio: {result.statistics['inventory_turnover']:.2f}

Service Level Metrics:
  • Fill Rate: {result.results['fill_rate']:.2%}
  • Cycle Service Level: {result.results['cycle_service_level']:.2%}
  • Stockout Probability: {result.statistics['stockout_probability']:.2%}
  • Total Shortage: {result.results['total_shortage']:.2f} units
  • Shortage Periods: {result.results['shortage_periods']} out of {self.simulation_periods}

Ordering Metrics:
  • Total Orders Placed: {result.results['total_orders']}
  • Order Frequency: {result.results['order_frequency']:.3f} orders/period
  • Average Order Quantity: {result.results['avg_order_quantity']:.2f} units
  • Order Quantity Std Dev: {result.statistics['order_quantity_std']:.2f} units

Cost Analysis:
  • Total Cost: ${result.results['total_cost']:.2f}
  • Average Cost per Period: ${result.results['avg_cost_per_period']:.2f}
  • Holding Cost Total: ${result.statistics['holding_cost_total']:.2f} ({result.statistics['holding_cost_total']/result.results['total_cost']:.1%})
  • Shortage Cost Total: ${result.statistics['shortage_cost_total']:.2f} ({result.statistics['shortage_cost_total']/result.results['total_cost']:.1%})
  • Ordering Cost Total: ${result.statistics['ordering_cost_total']:.2f} ({result.statistics['ordering_cost_total']/result.results['total_cost']:.1%})

THEORETICAL BENCHMARKS:
{'-'*30}
"""
        
        # Add optimal policy comparison if available
        try:
            optimal = self.calculate_optimal_policy()
            report += f"""Theoretical Optimal Policy:
  • Optimal Reorder Point (s): {optimal['s_optimal']:.1f}
  • Optimal Order-up-to Level (S): {optimal['S_optimal']:.1f}
  • Current vs Optimal s: {self.s - optimal['s_optimal']:+.1f}
  • Current vs Optimal S: {self.S - optimal['S_optimal']:+.1f}
  • Safety Stock: {optimal['safety_stock']:.1f} units
  • Expected Demand during Lead Time: {optimal['expected_demand_lead_time']:.1f} units

"""
        except Exception as e:
            report += f"Optimal policy calculation failed: {str(e)}\n\n"
        
        report += f"""
SIMULATION QUALITY:
{'-'*30}
  • Parameter Validation: {'PASSED' if not self.validate_parameters() else 'FAILED'}
  • Convergence: {'Good' if len(result.convergence_data) > 10 else 'Limited'}
  • Statistical Reliability: {'High' if self.simulation_periods >= 1000 else 'Medium' if self.simulation_periods >= 500 else 'Low'}

RECOMMENDATIONS:
{'-'*30}
"""
        
        # Add recommendations based on results
        recommendations = []
        
        if result.results['fill_rate'] < 0.90:
            recommendations.append("• Consider increasing reorder point (s) or order-up-to level (S) to improve service level")
        
        if result.statistics['holding_cost_total'] > result.statistics['shortage_cost_total'] * 2:
            recommendations.append("• Holding costs dominate - consider reducing inventory levels")
        
        if result.statistics['shortage_cost_total'] > result.statistics['holding_cost_total'] * 2:
            recommendations.append("• Shortage costs dominate - consider increasing inventory levels")
        
        if result.results['order_frequency'] > 0.5:
            recommendations.append("• High order frequency - consider increasing policy range (S-s)")
        
        if result.results['order_frequency'] < 0.05:
            recommendations.append("• Low order frequency - consider decreasing policy range (S-s)")
        
        if not recommendations:
            recommendations.append("• Current policy appears well-balanced")
        
        for rec in recommendations:
            report += rec + "\n"
        
        report += f"\n{'='*60}\nReport generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}"
        
        return report
