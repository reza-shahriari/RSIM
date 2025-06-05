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

class MMKQueue(BaseSimulation):
    """
    M/M/k Queue simulation - Multi-server queueing system.
    
    An M/M/k queue extends the M/M/1 model to k parallel servers:
    - M: Markovian (Poisson) arrival process
    - M: Markovian (exponential) service times
    - k: k parallel identical servers
    
    This model is widely used for systems with multiple service channels,
    such as call centers, bank branches, and computer systems with multiple processors.
    
    Mathematical Background:
    -----------------------
    - Arrival rate: λ (customers per unit time)
    - Service rate per server: μ (customers per unit time)
    - Number of servers: k
    - Traffic intensity: ρ = λ/(kμ) (must be < 1 for stability)
    - System capacity: unlimited queue + k servers
    
    Steady-State Analytical Results:
    -------------------------------
    - Utilization: ρ = λ/(kμ)
    - P₀ = [Σ(n=0 to k-1) (λ/μ)ⁿ/n! + (λ/μ)ᵏ/(k!(1-ρ))]⁻¹
    - Average customers in queue: Lq = P₀(λ/μ)ᵏρ/(k!(1-ρ)²)
    - Average customers in system: L = Lq + λ/μ
    - Average waiting time: Wq = Lq/λ
    - Average time in system: W = Wq + 1/μ
    
    Applications:
    ------------
    - Call centers with multiple agents
    - Bank branches with multiple tellers
    - Multi-processor computer systems
    - Hospital emergency departments
    - Airport check-in counters
    - Manufacturing systems with parallel machines
    
    Parameters:
    -----------
    arrival_rate : float, default=2.0
        Average arrival rate (λ) in customers per unit time
    service_rate : float, default=1.0
        Average service rate per server (μ) in customers per unit time
    num_servers : int, default=3
        Number of parallel servers (k)
    simulation_time : float, default=1000.0
        Total simulation time
    warmup_time : float, default=100.0
        Warmup period to reach steady state
    random_seed : int, optional
        Seed for random number generator
    """
    
    def __init__(self, arrival_rate: float = 2.0, service_rate: float = 1.0, 
                 num_servers: int = 3, simulation_time: float = 1000.0, 
                 warmup_time: float = 100.0, random_seed: Optional[int] = None):
        super().__init__("M/M/k Queue")
        
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.num_servers = num_servers
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        
        self.parameters.update({
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'num_servers': num_servers,
            'simulation_time': simulation_time,
            'warmup_time': warmup_time,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        # Simulation state
        self.events = []
        self.current_time = 0.0
        self.queue = deque()
        self.servers = [None] * num_servers  # None = idle, customer_id = busy
        self.next_customer_id = 1
        
        self.statistics = {
            'customer_data': [],
            'queue_length_data': [],
            'system_size_data': [],
            'server_utilization_data': []
        }
        
        self.is_configured = True
    
    def configure(self, arrival_rate: float = 2.0, service_rate: float = 1.0,
                 num_servers: int = 3, simulation_time: float = 1000.0, 
                 warmup_time: float = 100.0) -> bool:
        """Configure M/M/k queue parameters"""
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.num_servers = num_servers
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        
        self.parameters.update({
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'num_servers': num_servers,
            'simulation_time': simulation_time,
            'warmup_time': warmup_time
        })
        
        # Reset servers array
        self.servers = [None] * num_servers
        self.is_configured = True
        return True
    
    def _find_idle_server(self) -> Optional[int]:
        """Find an idle server, return server index or None"""
        for i, server in enumerate(self.servers):
            if server is None:
                return i
        return None
    
    def _get_busy_servers(self) -> int:
        """Count number of busy servers"""
        return sum(1 for server in self.servers if server is not None)
    
    def _schedule_event(self, event_time: float, event_type: str, customer_id: int = None, server_id: int = None):
        """Schedule an event in the event queue"""
        heapq.heappush(self.events, (event_time, event_type, customer_id, server_id))
    
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
            busy_servers = self._get_busy_servers()
            system_size = queue_length + busy_servers
            
            self.statistics['queue_length_data'].append((self.current_time, queue_length))
            self.statistics['system_size_data'].append((self.current_time, system_size))
            self.statistics['server_utilization_data'].append((self.current_time, busy_servers / self.num_servers))
    
    def _process_arrival(self, customer_id: int):
        """Process customer arrival event"""
        arrival_time = self.current_time
        
        # Check for idle server
        idle_server = self._find_idle_server()
        
        if idle_server is not None:
            # Server available, start service immediately
            self.servers[idle_server] = customer_id
            service_time = self._generate_service_time()
            departure_time = self.current_time + service_time
            self._schedule_event(departure_time, 'departure', customer_id, idle_server)
            
            if self.current_time >= self.warmup_time:
                self.statistics['customer_data'].append((arrival_time, arrival_time, departure_time))
        else:
            # All servers busy, join queue
            self.queue.append((customer_id, arrival_time))
        
        # Schedule next arrival
        next_arrival_time = self.current_time + self._generate_interarrival_time()
        if next_arrival_time <= self.simulation_time:
            self._schedule_event(next_arrival_time, 'arrival', self.next_customer_id)
            self.next_customer_id += 1
        
        self._record_statistics()
    
    def _process_departure(self, customer_id: int, server_id: int):
        """Process customer departure event"""
        departure_time = self.current_time
        
        # Free the server
        self.servers[server_id] = None
        
        if self.queue:
            # Start serving next customer in queue
            next_customer_id, arrival_time = self.queue.popleft()
            self.servers[server_id] = next_customer_id
            service_start_time = self.current_time
            service_time = self._generate_service_time()
            next_departure_time = self.current_time + service_time
            self._schedule_event(next_departure_time, 'departure', next_customer_id, server_id)
            
            if self.current_time >= self.warmup_time:
                self.statistics['customer_data'].append((arrival_time, service_start_time, next_departure_time))
        
        self._record_statistics()
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute M/M/k queue simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Reset simulation state
        self.events = []
        self.current_time = 0.0
        self.queue = deque()
        self.servers = [None] * self.num_servers
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
            event_time, event_type, customer_id, server_id = heapq.heappop(self.events)
            self.current_time = event_time
            
            if event_type == 'arrival':
                self._process_arrival(customer_id)
            elif event_type == 'departure':
                self._process_departure(customer_id, server_id)
        
        execution_time = time.time() - start_time
        
        # Calculate statistics (similar to MM1 but adapted for multiple servers)
        customer_data = self.statistics['customer_data']
        if customer_data:
            waiting_times = [service_start - arrival for arrival, service_start, departure in customer_data]
            system_times = [departure - arrival for arrival, service_start, departure in customer_data]
            
            avg_waiting_time = np.mean(waiting_times)
            avg_system_time = np.mean(system_times)
        else:
            avg_waiting_time = avg_system_time = 0
        
        # Time-weighted averages
        if self.statistics['system_size_data']:
            times, sizes = zip(*self.statistics['system_size_data'])
            times = np.array(times)
            sizes = np.array(sizes)
            
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
        
        # Theoretical calculations for M/M/k
        rho = self.arrival_rate / (self.num_servers * self.service_rate)
        lambda_over_mu = self.arrival_rate / self.service_rate
        
        if rho < 1:
            # Calculate P0 (probability of 0 customers in system)
            sum_term = sum((lambda_over_mu ** n) / np.math.factorial(n) for n in range(self.num_servers))
            p0_denominator = sum_term + ((lambda_over_mu ** self.num_servers) / 
                                       (np.math.factorial(self.num_servers) * (1 - rho)))
            p0 = 1 / p0_denominator
            
            # Theoretical metrics
            theoretical_Lq = (p0 * (lambda_over_mu ** self.num_servers) * rho) / \
                           (np.math.factorial(self.num_servers) * ((1 - rho) ** 2))
            theoretical_L = theoretical_Lq + lambda_over_mu
            theoretical_Wq = theoretical_Lq / self.arrival_rate
            theoretical_W = theoretical_Wq + (1 / self.service_rate)
            theoretical_utilization = lambda_over_mu / self.num_servers
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
                                'server_utilization': server_utilization,
                'total_customers_served': len(customer_data),
                'traffic_intensity': rho,
                'num_servers': self.num_servers
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
        """Visualize M/M/k queue simulation results"""
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
            if not np.isinf(result.statistics['theoretical_Lq']):
                ax1.axhline(y=result.statistics['theoretical_Lq'], color='red', linestyle='--', 
                           label=f'Theoretical Lq: {result.statistics["theoretical_Lq"]:.2f}')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Queue Length')
            ax1.set_title(f'Queue Length Over Time (k={self.num_servers} servers)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: System size over time
        system_data = result.raw_data['system_size_data']
        if system_data:
            times, sizes = zip(*system_data)
            ax2.plot(times, sizes, 'g-', linewidth=1, alpha=0.8)
            if not np.isinf(result.statistics['theoretical_L']):
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
            ax3.set_title('Overall Server Utilization Over Time')
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
                if not np.isinf(result.statistics['theoretical_Wq']):
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
                if not np.isinf(result.statistics['theoretical_W']):
                    ax5.axvline(x=result.statistics['theoretical_W'], color='red', linestyle='--', linewidth=2,
                               label=f'Theoretical Mean: {result.statistics["theoretical_W"]:.2f}')
                ax5.set_xlabel('System Time')
                ax5.set_ylabel('Density')
                ax5.set_title('Distribution of System Times')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # Plot 6: Theoretical vs Empirical comparison
        metrics = ['L\n(System)', 'Lq\n(Queue)', 'W\n(System)', 'Wq\n(Wait)', 'ρ\n(Util)']
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
        
        # Handle infinite theoretical values
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
            ax6.set_title(f'Theoretical vs Empirical Metrics (k={self.num_servers})')
            ax6.set_xticks(x)
            ax6.set_xticklabels(metrics_plot)
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
        
        # Print summary
        print("\n" + "="*60)
        print("M/M/k QUEUE SIMULATION RESULTS")
        print("="*60)
        print(f"Simulation Parameters:")
        print(f"  Arrival Rate (λ): {self.arrival_rate:.3f}")
        print(f"  Service Rate per server (μ): {self.service_rate:.3f}")
        print(f"  Number of servers (k): {self.num_servers}")
        print(f"  Traffic Intensity (ρ): {result.results['traffic_intensity']:.3f}")
        print(f"  Total service capacity: {self.num_servers * self.service_rate:.3f}")
        print(f"  Simulation Time: {self.simulation_time:.1f}")
        print(f"  Total Customers Served: {result.results['total_customers_served']}")
        
        if not any(np.isinf(v) for v in [result.statistics['theoretical_L'], result.statistics['theoretical_Lq']]):
            print(f"\nPerformance Metrics:")
            print(f"                          Theoretical    Empirical    Difference")
            print(f"  Customers in System:    {result.statistics['theoretical_L']:8.3f}    {result.statistics['empirical_L']:8.3f}    {abs(result.statistics['theoretical_L'] - result.statistics['empirical_L']):8.3f}")
            print(f"  Customers in Queue:     {result.statistics['theoretical_Lq']:8.3f}    {result.statistics['empirical_Lq']:8.3f}    {abs(result.statistics['theoretical_Lq'] - result.statistics['empirical_Lq']):8.3f}")
            print(f"  System Time:            {result.statistics['theoretical_W']:8.3f}    {result.statistics['empirical_W']:8.3f}    {abs(result.statistics['theoretical_W'] - result.statistics['empirical_W']):8.3f}")
            print(f"  Waiting Time:           {result.statistics['theoretical_Wq']:8.3f}    {result.statistics['empirical_Wq']:8.3f}    {abs(result.statistics['theoretical_Wq'] - result.statistics['empirical_Wq']):8.3f}")
            print(f"  Server Utilization:     {result.statistics['theoretical_utilization']:8.3f}    {result.statistics['empirical_utilization']:8.3f}    {abs(result.statistics['theoretical_utilization'] - result.statistics['empirical_utilization']):8.3f}")
        
        if result.results['traffic_intensity'] >= 1:
            print(f"\n⚠️  WARNING: System is unstable (ρ ≥ 1)!")
        
        print("="*60)
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'arrival_rate': {
                'type': 'float',
                'default': 2.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Average arrival rate (λ) in customers per unit time'
            },
            'service_rate': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 5.0,
                'description': 'Average service rate per server (μ) in customers per unit time'
            },
            'num_servers': {
                'type': 'int',
                'default': 3,
                'min': 1,
                'max': 20,
                'description': 'Number of parallel servers (k)'
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
        if self.num_servers < 1:
            errors.append("num_servers must be at least 1")
        if self.arrival_rate >= (self.num_servers * self.service_rate):
            errors.append("arrival_rate should be less than total service capacity for stability")
        if self.simulation_time <= 0:
            errors.append("simulation_time must be positive")
        if self.warmup_time < 0:
            errors.append("warmup_time must be non-negative")
        if self.warmup_time >= self.simulation_time:
            errors.append("warmup_time must be less than simulation_time")
        return errors
