import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Tuple
import heapq
from collections import deque
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult

class MM1Queue(BaseSimulation):
    """
    M/M/1 Queue simulation using discrete event simulation.
    
    An M/M/1 queue is a single-server queueing system where:
    - M: Markovian (Poisson) arrival process
    - M: Markovian (exponential) service times
    - 1: Single server
    
    This is one of the most fundamental queueing models in operations research
    and performance analysis, with well-known analytical solutions that can
    be compared against simulation results.
    
    Mathematical Background:
    -----------------------
    - Arrival rate: λ (customers per unit time)
    - Service rate: μ (customers per unit time)
    - Traffic intensity: ρ = λ/μ (must be < 1 for stability)
    - Inter-arrival times: Exponential(λ)
    - Service times: Exponential(μ)
    
    Steady-State Analytical Results:
    -------------------------------
    - Utilization: ρ = λ/μ
    - Average number in system: L = ρ/(1-ρ)
    - Average number in queue: Lq = ρ²/(1-ρ)
    - Average time in system: W = 1/(μ-λ)
    - Average waiting time: Wq = ρ/(μ-λ)
    - Probability of n customers: P(n) = ρⁿ(1-ρ)
    
    Applications:
    ------------
    - Computer system performance analysis
    - Call center modeling
    - Manufacturing systems
    - Network traffic analysis
    - Hospital emergency departments
    - Bank teller operations
    - Web server performance
    
    Simulation Features:
    -------------------
    - Discrete event simulation engine
    - Customer arrival and departure tracking
    - Queue length monitoring over time
    - Waiting time statistics
    - System utilization analysis
    - Transient and steady-state behavior
    - Comparison with theoretical results
    
    Parameters:
    -----------
    arrival_rate : float, default=0.8
        Average arrival rate (λ) in customers per unit time
    service_rate : float, default=1.0
        Average service rate (μ) in customers per unit time
    simulation_time : float, default=1000.0
        Total simulation time
    warmup_time : float, default=100.0
        Warmup period to reach steady state (statistics not collected)
    random_seed : int, optional
        Seed for random number generator for reproducible results
    
    Attributes:
    -----------
    events : list
        Event queue for discrete event simulation
    current_time : float
        Current simulation time
    queue : deque
        Customer queue
    server_busy : bool
        Server status
    statistics : dict
        Collected statistics during simulation
    
    Methods:
    --------
    configure(arrival_rate, service_rate, simulation_time, warmup_time) : bool
        Configure simulation parameters
    run(**kwargs) : SimulationResult
        Execute the M/M/1 queue simulation
    visualize(result=None, **kwargs) : None
        Create comprehensive visualizations of queue performance
    validate_parameters() : List[str]
        Validate current parameters and return any errors
    get_parameter_info() : dict
        Get parameter information for UI generation
    
    Examples:
    ---------
    >>> # Basic M/M/1 queue simulation
    >>> queue = MM1Queue(arrival_rate=0.8, service_rate=1.0, random_seed=42)
    >>> result = queue.run()
    >>> print(f"Average customers in system: {result.results['avg_customers_in_system']:.2f}")
    >>> print(f"Average waiting time: {result.results['avg_waiting_time']:.2f}")
    
    >>> # High utilization scenario
    >>> busy_queue = MM1Queue(arrival_rate=0.95, service_rate=1.0, simulation_time=5000)
    >>> result = busy_queue.run()
    >>> busy_queue.visualize()
    
    >>> # Compare with theoretical results
    >>> queue = MM1Queue(arrival_rate=0.7, service_rate=1.0)
    >>> result = queue.run()
    >>> theoretical_L = 0.7 / (1 - 0.7)  # ρ/(1-ρ)
    >>> empirical_L = result.results['avg_customers_in_system']
    >>> print(f"Theoretical L: {theoretical_L:.2f}, Empirical L: {empirical_L:.2f}")
    
    Notes:
    ------
    - System is stable only when ρ = λ/μ < 1
    - Warmup period helps eliminate initial transient effects
    - Longer simulation times provide more accurate steady-state estimates
    - Results converge to theoretical values as simulation time increases
    
    References:
    -----------
    - Gross, D., et al. (2008). Fundamentals of Queueing Theory
    - Kleinrock, L. (1975). Queueing Systems Volume 1: Theory
    - Ross, S. M. (2014). Introduction to Probability Models
    """
    
    def __init__(self, arrival_rate: float = 0.8, service_rate: float = 1.0, 
                 simulation_time: float = 1000.0, warmup_time: float = 100.0,
                 random_seed: Optional[int] = None):
        super().__init__("M/M/1 Queue")
        
        # Initialize parameters
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        
        # Store in parameters dict for base class
        self.parameters.update({
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'simulation_time': simulation_time,
            'warmup_time': warmup_time,
            'random_seed': random_seed
        })
        
        # Set random seed if provided
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Simulation state
        self.events = []  # Priority queue for events
        self.current_time = 0.0
        self.queue = deque()  # Customer queue
        self.server_busy = False
        self.next_customer_id = 1
        
        # Statistics collection
        self.statistics = {
            'customer_data': [],  # (arrival_time, service_start, departure_time)
            'queue_length_data': [],  # (time, queue_length)
            'system_size_data': [],  # (time, total_customers)
            'server_utilization_data': []  # (time, utilization)
        }
        
        self.is_configured = True
    
    def configure(self, arrival_rate: float = 0.8, service_rate: float = 1.0,
                 simulation_time: float = 1000.0, warmup_time: float = 100.0) -> bool:
        """Configure M/M/1 queue parameters"""
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        
        # Update parameters dict
        self.parameters.update({
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'simulation_time': simulation_time,
            'warmup_time': warmup_time
        })
        
        self.is_configured = True
        return True
    
    def _schedule_event(self, event_time: float, event_type: str, customer_id: int = None):
        """Schedule an event in the event queue"""
        heapq.heappush(self.events, (event_time, event_type, customer_id))
    
    def _generate_interarrival_time(self) -> float:
        """Generate exponential inter-arrival time"""
        return np.random.exponential(1.0 / self.arrival_rate)
    
    def _generate_service_time(self) -> float:
        """Generate exponential service time"""
        return np.random.exponential(1.0 / self.service_rate)
    
    def _record_statistics(self):
        """Record current system state for statistics"""
        if self.current_time >= self.warmup_time:
            queue_length = len(self.queue)
            system_size = queue_length + (1 if self.server_busy else 0)
            
            self.statistics['queue_length_data'].append((self.current_time, queue_length))
            self.statistics['system_size_data'].append((self.current_time, system_size))
            self.statistics['server_utilization_data'].append((self.current_time, 1 if self.server_busy else 0))
    
    def _process_arrival(self, customer_id: int):
        """Process customer arrival event"""
        arrival_time = self.current_time
        
        if not self.server_busy:
            # Server is free, start service immediately
            self.server_busy = True
            service_time = self._generate_service_time()
            departure_time = self.current_time + service_time
            self._schedule_event(departure_time, 'departure', customer_id)
            
            if self.current_time >= self.warmup_time:
                self.statistics['customer_data'].append((arrival_time, arrival_time, departure_time))
        else:
            # Server is busy, join queue
            self.queue.append((customer_id, arrival_time))
        
        # Schedule next arrival
        next_arrival_time = self.current_time + self._generate_interarrival_time()
        if next_arrival_time <= self.simulation_time:
            self._schedule_event(next_arrival_time, 'arrival', self.next_customer_id)
            self.next_customer_id += 1
        
        self._record_statistics()
    
    def _process_departure(self, customer_id: int):
        """Process customer departure event"""
        departure_time = self.current_time
        
        if self.queue:
            # Start serving next customer in queue
            next_customer_id, arrival_time = self.queue.popleft()
            service_start_time = self.current_time
            service_time = self._generate_service_time()
            next_departure_time = self.current_time + service_time
            self._schedule_event(next_departure_time, 'departure', next_customer_id)
            
            if self.current_time >= self.warmup_time:
                self.statistics['customer_data'].append((arrival_time, service_start_time, next_departure_time))
        else:
            # No customers waiting, server becomes idle
            self.server_busy = False
        
        self._record_statistics()
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute M/M/1 queue simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Reset simulation state
        self.events = []
        self.current_time = 0.0
        self.queue = deque()
        self.server_busy = False
        self.next_customer_id = 1
        self.statistics = {
            'customer_data': [],
            'queue_length_data': [],
            'system_size_data': [],
            'server_utilization_data': []
        }
        
        # Schedule first arrival
        first_arrival_time = self._generate_interarrival_time()
        self._schedule_event(first_arrival_time, 'arrival', self.next_customer_id)
        self.next_customer_id += 1
        
        # Main simulation loop
        while self.events and self.current_time < self.simulation_time:
            # Get next event
            event_time, event_type, customer_id = heapq.heappop(self.events)
            self.current_time = event_time
            
            if event_type == 'arrival':
                self._process_arrival(customer_id)
            elif event_type == 'departure':
                self._process_departure(customer_id)
        
        execution_time = time.time() - start_time
        
        # Calculate statistics
        customer_data = self.statistics['customer_data']
        if customer_data:
            waiting_times = [service_start - arrival for arrival, service_start, departure in customer_data]
            system_times = [departure - arrival for arrival, service_start, departure in customer_data]
            service_times = [departure - service_start for arrival, service_start, departure in customer_data]
            
            avg_waiting_time = np.mean(waiting_times)
            avg_system_time = np.mean(system_times)
            avg_service_time = np.mean(service_times)
        else:
            avg_waiting_time = avg_system_time = avg_service_time = 0
        
        # Time-weighted averages
        if self.statistics['system_size_data']:
            times, sizes = zip(*self.statistics['system_size_data'])
            times = np.array(times)
            sizes = np.array(sizes)
            
            # Calculate time-weighted average
            if len(times) > 1:
                time_diffs = np.diff(times)
                weighted_sum = np.sum(sizes[:-1] * time_diffs)
                total_time = times[-1] - times[0]
                avg_customers_in_system = weighted_sum / total_time if total_time > 0 else 0
            else:
                avg_customers_in_system = sizes[0] if sizes else 0
        else:
            avg_customers_in_system = 0
        
        if self.statistics['queue_length_data']:
            times, lengths = zip(*self.statistics['queue_length_data'])
            times = np.array(times)
            lengths = np.array(lengths)
            
            if len(times) > 1:
                time_diffs = np.diff(times)
                weighted_sum = np.sum(lengths[:-1] * time_diffs)
                total_time = times[-1] - times[0]
                avg_queue_length = weighted_sum / total_time if total_time > 0 else 0
            else:
                avg_queue_length = lengths[0] if lengths else 0
        else:
            avg_queue_length = 0
        
        # Server utilization
        if self.statistics['server_utilization_data']:
            times, utils = zip(*self.statistics['server_utilization_data'])
            times = np.array(times)
            utils = np.array(utils)
            
            if len(times) > 1:
                time_diffs = np.diff(times)
                weighted_sum = np.sum(utils[:-1] * time_diffs)
                total_time = times[-1] - times[0]
                server_utilization = weighted_sum / total_time if total_time > 0 else 0
            else:
                server_utilization = utils[0] if utils else 0
        else:

            server_utilization = 0
        
        # Theoretical calculations
        rho = self.arrival_rate / self.service_rate  # Traffic intensity
        if rho < 1:
            theoretical_L = rho / (1 - rho)  # Average customers in system
            theoretical_Lq = (rho ** 2) / (1 - rho)  # Average customers in queue
            theoretical_W = 1 / (self.service_rate - self.arrival_rate)  # Average time in system
            theoretical_Wq = rho / (self.service_rate - self.arrival_rate)  # Average waiting time
            theoretical_utilization = rho
        else:
            # System is unstable
            theoretical_L = float('inf')
            theoretical_Lq = float('inf')
            theoretical_W = float('inf')
            theoretical_Wq = float('inf')
            theoretical_utilization = 1.0
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'avg_customers_in_system': avg_customers_in_system,
                'avg_queue_length': avg_queue_length,
                'avg_waiting_time': avg_waiting_time,
                'avg_system_time': avg_system_time,
                'avg_service_time': avg_service_time,
                'server_utilization': server_utilization,
                'total_customers_served': len(customer_data),
                'traffic_intensity': rho
            },
            statistics={
                'theoretical_L': theoretical_L,
                'theoretical_Lq': theoretical_Lq,
                'theoretical_W': theoretical_W,
                'theoretical_Wq': theoretical_Wq,
                'theoretical_utilization': theoretical_utilization,
                'empirical_L': avg_customers_in_system,
                'empirical_Lq': avg_queue_length,
                'empirical_W': avg_system_time,
                'empirical_Wq': avg_waiting_time,
                'empirical_utilization': server_utilization
            },
            raw_data={
                'customer_data': customer_data,
                'queue_length_data': self.statistics['queue_length_data'],
                'system_size_data': self.statistics['system_size_data'],
                'server_utilization_data': self.statistics['server_utilization_data']
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize M/M/1 queue simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create 2x3 subplot layout
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        ax6 = plt.subplot(2, 3, 6)
        
        # Plot 1: Queue length over time
        queue_data = result.raw_data['queue_length_data']
        if queue_data:
            times, lengths = zip(*queue_data)
            ax1.plot(times, lengths, 'b-', linewidth=1, alpha=0.8)
            ax1.axhline(y=result.statistics['theoretical_Lq'], color='red', linestyle='--', 
                       label=f'Theoretical Lq: {result.statistics["theoretical_Lq"]:.2f}')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Queue Length')
            ax1.set_title('Queue Length Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: System size over time
        system_data = result.raw_data['system_size_data']
        if system_data:
            times, sizes = zip(*system_data)
            ax2.plot(times, sizes, 'g-', linewidth=1, alpha=0.8)
            ax2.axhline(y=result.statistics['theoretical_L'], color='red', linestyle='--',
                       label=f'Theoretical L: {result.statistics["theoretical_L"]:.2f}')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Customers in System')
            ax2.set_title('Total Customers in System Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Server utilization over time
        util_data = result.raw_data['server_utilization_data']
        if util_data:
            times, utils = zip(*util_data)
            ax3.plot(times, utils, 'purple', linewidth=1, alpha=0.8)
            ax3.axhline(y=result.statistics['theoretical_utilization'], color='red', linestyle='--',
                       label=f'Theoretical ρ: {result.statistics["theoretical_utilization"]:.2f}')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Server Utilization')
            ax3.set_title('Server Utilization Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-0.1, 1.1)
        
        # Plot 4: Waiting time distribution
        customer_data = result.raw_data['customer_data']
        if customer_data:
            waiting_times = [service_start - arrival for arrival, service_start, departure in customer_data]
            if waiting_times:
                ax4.hist(waiting_times, bins=30, alpha=0.7, edgecolor='black', density=True)
                ax4.axvline(x=np.mean(waiting_times), color='green', linestyle='-', linewidth=2,
                           label=f'Empirical Mean: {np.mean(waiting_times):.2f}')
                ax4.axvline(x=result.statistics['theoretical_Wq'], color='red', linestyle='--', linewidth=2,
                           label=f'Theoretical Mean: {result.statistics["theoretical_Wq"]:.2f}')
                ax4.set_xlabel('Waiting Time')
                ax4.set_ylabel('Density')
                ax4.set_title('Distribution of Waiting Times')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # Plot 5: System time distribution
        if customer_data:
            system_times = [departure - arrival for arrival, service_start, departure in customer_data]
            if system_times:
                ax5.hist(system_times, bins=30, alpha=0.7, edgecolor='black', density=True)
                ax5.axvline(x=np.mean(system_times), color='green', linestyle='-', linewidth=2,
                           label=f'Empirical Mean: {np.mean(system_times):.2f}')
                ax5.axvline(x=result.statistics['theoretical_W'], color='red', linestyle='--', linewidth=2,
                           label=f'Theoretical Mean: {result.statistics["theoretical_W"]:.2f}')
                ax5.set_xlabel('System Time')
                ax5.set_ylabel('Density')
                ax5.set_title('Distribution of System Times')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # Plot 6: Theoretical vs Empirical comparison
        metrics = ['L (Customers\nin System)', 'Lq (Customers\nin Queue)', 'W (System\nTime)', 'Wq (Waiting\nTime)', 'ρ (Server\nUtilization)']
        theoretical = [
            result.statistics['theoretical_L'],
            result.statistics['theoretical_Lq'],
            result.statistics['theoretical_W'],
            result.statistics['theoretical_Wq'],
            result.statistics['theoretical_utilization']
        ]
        empirical = [
            result.statistics['empirical_L'],
            result.statistics['empirical_Lq'],
            result.statistics['empirical_W'],
            result.statistics['empirical_Wq'],
            result.statistics['empirical_utilization']
        ]
        
        # Handle infinite theoretical values for unstable system
        theoretical_plot = []
        empirical_plot = []
        metrics_plot = []
        for i, (t, e, m) in enumerate(zip(theoretical, empirical, metrics)):
            if not np.isinf(t):
                theoretical_plot.append(t)
                empirical_plot.append(e)
                metrics_plot.append(m)
        
        if theoretical_plot:
            x = np.arange(len(metrics_plot))
            width = 0.35
            
            ax6.bar(x - width/2, theoretical_plot, width, label='Theoretical', alpha=0.7, color='blue')
            ax6.bar(x + width/2, empirical_plot, width, label='Empirical', alpha=0.7, color='orange')
            
            ax6.set_ylabel('Value')
            ax6.set_title('Theoretical vs Empirical Metrics')
            ax6.set_xticks(x)
            ax6.set_xticklabels(metrics_plot, rotation=45, ha='right')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (theo, emp) in enumerate(zip(theoretical_plot, empirical_plot)):
                ax6.text(i - width/2, theo + abs(theo)*0.01, f'{theo:.2f}', 
                        ha='center', va='bottom', fontsize=8)
                ax6.text(i + width/2, emp + abs(emp)*0.01, f'{emp:.2f}', 
                        ha='center', va='bottom', fontsize=8)
        else:
            ax6.text(0.5, 0.5, 'System is unstable\n(ρ ≥ 1)', 
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
            ax6.set_title('System Stability Check')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("M/M/1 QUEUE SIMULATION RESULTS")
        print("="*60)
        print(f"Simulation Parameters:")
        print(f"  Arrival Rate (λ): {self.arrival_rate:.3f}")
        print(f"  Service Rate (μ): {self.service_rate:.3f}")
        print(f"  Traffic Intensity (ρ): {result.results['traffic_intensity']:.3f}")
        print(f"  Simulation Time: {self.simulation_time:.1f}")
        print(f"  Warmup Time: {self.warmup_time:.1f}")
        print(f"  Total Customers Served: {result.results['total_customers_served']}")
        
        print(f"\nPerformance Metrics:")
        print(f"                          Theoretical    Empirical    Difference")
        print(f"  Customers in System:    {result.statistics['theoretical_L']:8.3f}    {result.statistics['empirical_L']:8.3f}    {abs(result.statistics['theoretical_L'] - result.statistics['empirical_L']):8.3f}")
        print(f"  Customers in Queue:     {result.statistics['theoretical_Lq']:8.3f}    {result.statistics['empirical_Lq']:8.3f}    {abs(result.statistics['theoretical_Lq'] - result.statistics['empirical_Lq']):8.3f}")
        print(f"  System Time:            {result.statistics['theoretical_W']:8.3f}    {result.statistics['empirical_W']:8.3f}    {abs(result.statistics['theoretical_W'] - result.statistics['empirical_W']):8.3f}")
        print(f"  Waiting Time:           {result.statistics['theoretical_Wq']:8.3f}    {result.statistics['empirical_Wq']:8.3f}    {abs(result.statistics['theoretical_Wq'] - result.statistics['empirical_Wq']):8.3f}")
        print(f"  Server Utilization:     {result.statistics['theoretical_utilization']:8.3f}    {result.statistics['empirical_utilization']:8.3f}    {abs(result.statistics['theoretical_utilization'] - result.statistics['empirical_utilization']):8.3f}")
        
        if result.results['traffic_intensity'] >= 1:
            print(f"\n⚠️  WARNING: System is unstable (ρ ≥ 1)!")
            print(f"   Queue will grow without bound in steady state.")
        
        print("="*60)
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'arrival_rate': {
                'type': 'float',
                'default': 0.8,
                'min': 0.1,
                'max': 2.0,
                'description': 'Average arrival rate (λ) in customers per unit time'
            },
            'service_rate': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 5.0,
                'description': 'Average service rate (μ) in customers per unit time'
            },
            'simulation_time': {
                'type': 'float',
                'default': 1000.0,
                'min': 100.0,
                'max': 10000.0,
                'description': 'Total simulation time'
            },
            'warmup_time': {
                'type': 'float',
                'default': 100.0,
                'min': 0.0,
                'max': 1000.0,
                'description': 'Warmup period to reach steady state'
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
        if self.arrival_rate <= 0:
            errors.append("arrival_rate must be positive")
        if self.service_rate <= 0:
            errors.append("service_rate must be positive")
        if self.arrival_rate >= self.service_rate:
            errors.append("arrival_rate should be less than service_rate for system stability")
        if self.simulation_time <= 0:
            errors.append("simulation_time must be positive")
        if self.warmup_time < 0:
            errors.append("warmup_time must be non-negative")
        if self.warmup_time >= self.simulation_time:
            errors.append("warmup_time must be less than simulation_time")
        return errors

