import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Callable
import heapq
from collections import deque
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult

class MG1Queue(BaseSimulation):
    """
    M/G/1 Queue simulation - Single server with general service time distribution.
    
    An M/G/1 queue extends the M/M/1 model to allow general service time distributions:
    - M: Markovian (Poisson) arrival process
    - G: General service time distribution
    - 1: Single server
    
    This model is more realistic for many applications where service times
    don't follow exponential distributions, such as file transfers, manufacturing
    processes, or human-operated services.
    
    Mathematical Background:
    -----------------------
    - Arrival rate: λ (customers per unit time)
    - Service time distribution: General with mean E[S] and variance Var[S]
    - Traffic intensity: ρ = λ × E[S] (must be < 1 for stability)
    - Coefficient of variation: C_s = √(Var[S]) / E[S]
    
    Pollaczek-Khinchine Formula:
    ---------------------------
    - Average customers in queue: Lq = ρ² × (1 + C_s²) / (2(1-ρ))
    - Average customers in system: L = Lq + ρ
    - Average waiting time: Wq = Lq / λ
    - Average time in system: W = Wq + E[S]
    
    Supported Service Distributions:
    -------------------------------
    - Exponential: Traditional M/M/1 case
    - Deterministic: Constant service times (D)
    - Uniform: Service times uniformly distributed
    - Normal: Gaussian service times (truncated at 0)
    - Gamma: Flexible shape parameter
    - Lognormal: Right-skewed distribution
    - Custom: User-defined distribution function
    
    Applications:
    ------------
    - File download/upload systems
    - Manufacturing with variable processing times
    - Human-operated service counters
    - Computer systems with variable job sizes
    - Medical procedures with different complexities
    - Network packet processing
    
    Parameters:
    -----------
    arrival_rate : float, default=0.8
        Average arrival rate (λ) in customers per unit time
    service_mean : float, default=1.0
        Mean service time
    service_distribution : str, default='exponential'
        Service time distribution type
    service_params : dict, default={}
        Additional parameters for service distribution
    simulation_time : float, default=1000.0
        Total simulation time
    warmup_time : float, default=100.0
        Warmup period to reach steady state
    random_seed : int, optional
        Seed for random number generator
    """
    
    def __init__(self, arrival_rate: float = 0.8, service_mean: float = 1.0,
                 service_distribution: str = 'exponential', service_params: dict = {},
                 simulation_time: float = 1000.0, warmup_time: float = 100.0,
                 random_seed: Optional[int] = None):
        super().__init__("M/G/1 Queue")
        
        self.arrival_rate = arrival_rate
        self.service_mean = service_mean
        self.service_distribution = service_distribution
        self.service_params = service_params.copy()
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        
        self.parameters.update({
            'arrival_rate': arrival_rate,
            'service_mean': service_mean,
            'service_distribution': service_distribution,
            'service_params': service_params,
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
        self.server_busy = False
        self.next_customer_id = 1
        
        self.statistics = {
            'customer_data': [],
            'queue_length_data': [],
            'system_size_data': [],
            'server_utilization_data': [],
            'service_times': []
        }
        
        self.is_configured = True
    
    def configure(self, arrival_rate: float = 0.8, service_mean: float = 1.0,
                 service_distribution: str = 'exponential', service_params: dict = {},
                 simulation_time: float = 1000.0, warmup_time: float = 100.0) -> bool:
        """Configure M/G/1 queue parameters"""
        self.arrival_rate = arrival_rate
        self.service_mean = service_mean
        self.service_distribution = service_distribution
        self.service_params = service_params.copy()
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        
        self.parameters.update({
            'arrival_rate': arrival_rate,
            'service_mean': service_mean,
            'service_distribution': service_distribution,
            'service_params': service_params,
            'simulation_time': simulation_time,
            'warmup_time': warmup_time
        })
        
        self.is_configured = True
        return True
    
    def _generate_service_time(self) -> float:
        """Generate service time from specified distribution"""
        if self.service_distribution == 'exponential':
            return np.random.exponential(self.service_mean)
        
        elif self.service_distribution == 'deterministic':
            return self.service_mean
        
        elif self.service_distribution == 'uniform':
            # Uniform distribution with specified mean
            # For uniform(a,b): mean = (a+b)/2, so b = 2*mean - a
            a = self.service_params.get('min', 0.5 * self.service_mean)
            b = 2 * self.service_mean - a
            return np.random.uniform(a, max(a, b))
        
        elif self.service_distribution == 'normal':
            # Normal distribution (truncated at 0)
            std = self.service_params.get('std', 0.3 * self.service_mean)
            service_time = np.random.normal(self.service_mean, std)
            return max(0.01, service_time)  # Ensure positive
        
        elif self.service_distribution == 'gamma':
            # Gamma distribution with specified mean
            cv = self.service_params.get('cv', 1.0)  # Coefficient of variation
            shape = 1 / (cv ** 2)
            scale = self.service_mean / shape
            return np.random.gamma(shape, scale)
        
        elif self.service_distribution == 'lognormal':
            # Lognormal distribution
            cv = self.service_params.get('cv', 0.5)
            sigma = np.sqrt(np.log(1 + cv**2))
            mu = np.log(self.service_mean) - 0.5 * sigma**2
            return np.random.lognormal(mu, sigma)
        
        else:
            # Default to exponential
            return np.random.exponential(self.service_mean)
    
    def _calculate_service_variance(self) -> float:
        """Calculate theoretical variance of service time distribution"""
        if self.service_distribution == 'exponential':
            return self.service_mean ** 2
        
        elif self.service_distribution == 'deterministic':
            return 0.0
        
        elif self.service_distribution == 'uniform':
            a = self.service_params.get('min', 0.5 * self.service_mean)
            b = 2 * self.service_mean - a
            return ((b - a) ** 2) / 12
        
        elif self.service_distribution == 'normal':
            std = self.service_params.get('std', 0.3 * self.service_mean)
            return std ** 2
        
        elif self.service_distribution == 'gamma':
            cv = self.service_params.get('cv', 1.0)
            return (cv * self.service_mean) ** 2
        
        elif self.service_distribution == 'lognormal':
            cv = self.service_params.get('cv', 0.5)
            return (cv * self.service_mean) ** 2
        
        else:
            return self.service_mean ** 2
    
    def _schedule_event(self, event_time: float, event_type: str, customer_id: int = None):
        """Schedule an event in the event queue"""
        heapq.heappush(self.events, (event_time, event_type, customer_id))
    
    def _generate_interarrival_time(self) -> float:
        """Generate exponential inter-arrival time"""
        return np.random.exponential(1.0 / self.arrival_rate)
    
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
                self.statistics['service_times'].append(service_time)
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
                self.statistics['service_times'].append(service_time)
        else:
            # No customers waiting, server becomes idle
            self.server_busy = False
        
        self._record_statistics()
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute M/G/1 queue simulation"""
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
            'server_utilization_data': [],
            'service_times': []
        }
        
        # Schedule first arrival
        first_arrival_time = self._generate_interarrival_time()
        self._schedule_event(first_arrival_time, 'arrival', self.next_customer_id)
        self.next_customer_id += 1
        
        # Main simulation loop
        while self.events and self.current_time < self.simulation_time:
            event_time, event_type, customer_id = heapq.heappop(self.events)
            self.current_time = event_time
            
            if event_type == 'arrival':
                self._process_arrival(customer_id)
            elif event_type == 'departure':
                self._process_departure(customer_id)
        
        execution_time = time.time() - start_time
        
        # Calculate statistics
        customer_data = self.statistics['customer_data']
        service_times = self.statistics['service_times']
        
        if customer_data:
            waiting_times = [service_start - arrival for arrival, service_start, departure in customer_data]
            system_times = [departure - arrival for arrival, service_start, departure in customer_data]
            
            avg_waiting_time = np.mean(waiting_times)
            avg_system_time = np.mean(system_times)
            empirical_service_mean = np.mean(service_times) if service_times else self.service_mean
            empirical_service_var = np.var(service_times) if service_times else 0
        else:
            avg_waiting_time = avg_system_time = 0
            empirical_service_mean = self.service_mean
            empirical_service_var = 0
        
        # Time-weighted averages (same as MM1)
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
        
        # Theoretical calculations using Pollaczek-Khinchine formula
        rho = self.arrival_rate * self.service_mean
        theoretical_service_var = self._calculate_service_variance()
        
        if rho < 1:
            # Coefficient of variation squared
            cs_squared = theoretical_service_var / (self.service_mean ** 2)
            
            # Pollaczek-Khinchine formula
            theoretical_Lq = (rho ** 2) * (1 + cs_squared) / (2 * (1 - rho))
            theoretical_L = theoretical_Lq + rho
            theoretical_Wq = theoretical_Lq / self.arrival_rate
            theoretical_W = theoretical_Wq + self.service_mean
            theoretical_utilization = rho
        else:
            # System is unstable
            theoretical_L = float('inf')
            theoretical_Lq = float('inf')
            theoretical_W = float('inf')
            theoretical_Wq = float('inf')
            theoretical_utilization = 1.0
            cs_squared = theoretical_service_var / (self.service_mean ** 2)
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'avg_customers_in_system': avg_customers_in_system,
                'avg_queue_length': avg_queue_length,
                'avg_waiting_time': avg_waiting_time,
                'avg_system_time': avg_system_time,
                'avg_service_time': empirical_service_mean,
                'server_utilization': server_utilization,
                'total_customers_served': len(customer_data),
                'traffic_intensity': rho,
                'service_distribution': self.service_distribution
            },
            statistics={
                'theoretical_L': theoretical_L,
                'theoretical_Lq': theoretical_Lq,
                'theoretical_W': theoretical_W,
                'theoretical_Wq': theoretical_Wq,
                'theoretical_utilization': theoretical_utilization,
                'theoretical_service_mean': self.service_mean,
                'theoretical_service_var': theoretical_service_var,
                'theoretical_cs_squared': cs_squared,
                'empirical_L': avg_customers_in_system,
                'empirical_Lq': avg_queue_length,
                'empirical_W': avg_system_time,
                'empirical_Wq': avg_waiting_time,
                'empirical_utilization': server_utilization,
                'empirical_service_mean': empirical_service_mean,
                'empirical_service_var': empirical_service_var,
                'empirical_cs_squared': empirical_service_var / (empirical_service_mean ** 2) if empirical_service_mean > 0 else 0
            },
            raw_data={
                'customer_data': customer_data,
                'queue_length_data': self.statistics['queue_length_data'],
                'system_size_data': self.statistics['system_size_data'],
                'server_utilization_data': self.statistics['server_utilization_data'],
                'service_times': service_times
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize M/G/1 queue simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        fig = plt.figure(figsize=(18, 14))
        
        # Create 3x3 subplot layout for more comprehensive visualization
        ax1 = plt.subplot(3, 3, 1)
        ax2 = plt.subplot(3, 3, 2)
        ax3 = plt.subplot(3, 3, 3)
        ax4 = plt.subplot(3, 3, 4)
        ax5 = plt.subplot(3, 3, 5)
        ax6 = plt.subplot(3, 3, 6)
        ax7 = plt.subplot(3, 3, 7)
        ax8 = plt.subplot(3, 3, 8)
        ax9 = plt.subplot(3, 3, 9)
        
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
            ax1.set_title(f'Queue Length Over Time ({self.service_distribution})')
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
            ax3.set_title('Server Utilization Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-0.1, 1.1)
        
        # Plot 4: Service time distribution
        service_times = result.raw_data['service_times']
        if service_times:
            ax4.hist(service_times, bins=30, alpha=0.7, edgecolor='black', density=True)
            ax4.axvline(x=np.mean(service_times), color='green', linestyle='-', linewidth=2,
                       label=f'Empirical Mean: {np.mean(service_times):.2f}')
            ax4.axvline(x=result.statistics['theoretical_service_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Theoretical Mean: {result.statistics["theoretical_service_mean"]:.2f}')
            ax4.set_xlabel('Service Time')
            ax4.set_ylabel('Density')
            ax4.set_title(f'Service Time Distribution ({self.service_distribution})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Waiting time distribution
        customer_data = result.raw_data['customer_data']
        if customer_data:
            waiting_times = [service_start - arrival for arrival, service_start, departure in customer_data]
            if waiting_times:
                ax5.hist(waiting_times, bins=30, alpha=0.7, edgecolor='black', density=True)
                ax5.axvline(x=np.mean(waiting_times), color='green', linestyle='-', linewidth=2,
                           label=f'Empirical Mean: {np.mean(waiting_times):.2f}')
                if not np.isinf(result.statistics['theoretical_Wq']):
                    ax5.axvline(x=result.statistics['theoretical_Wq'], color='red', linestyle='--', linewidth=2,
                               label=f'Theoretical Mean: {result.statistics["theoretical_Wq"]:.2f}')
                ax5.set_xlabel('Waiting Time')
                ax5.set_ylabel('Density')
                ax5.set_title('Distribution of Waiting Times')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # Plot 6: System time distribution
        if customer_data:
            system_times = [departure - arrival for arrival, service_start, departure in customer_data]
            if system_times:
                ax6.hist(system_times, bins=30, alpha=0.7, edgecolor='black', density=True)
                ax6.axvline(x=np.mean(system_times), color='green', linestyle='-', linewidth=2,
                           label=f'Empirical Mean: {np.mean(system_times):.2f}')
                if not np.isinf(result.statistics['theoretical_W']):
                    ax6.axvline(x=result.statistics['theoretical_W'], color='red', linestyle='--', linewidth=2,
                               label=f'Theoretical Mean: {result.statistics["theoretical_W"]:.2f}')
                ax6.set_xlabel('System Time')
                ax6.set_ylabel('Density')
                ax6.set_title('Distribution of System Times')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        # Plot 7: Theoretical vs Empirical comparison
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
            
            ax7.bar(x - width/2, theoretical_plot, width, label='Theoretical', alpha=0.7, color='blue')
            ax7.bar(x + width/2, empirical_plot, width, label='Empirical', alpha=0.7, color='orange')
            
            ax7.set_ylabel('Value')
            ax7.set_title('Theoretical vs Empirical Metrics')
            ax7.set_xticks(x)
            ax7.set_xticklabels(metrics_plot)
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (theo, emp) in enumerate(zip(theoretical_plot, empirical_plot)):
                ax7.text(i - width/2, theo + abs(theo)*0.01, f'{theo:.2f}', 
                        ha='center', va='bottom', fontsize=8)
                ax7.text(i + width/2, emp + abs(emp)*0.01, f'{emp:.2f}', 
                        ha='center', va='bottom', fontsize=8)
        else:
            ax7.text(0.5, 0.5, 'System is unstable\n(ρ ≥ 1)', 
                    ha='center', va='center', transform=ax7.transAxes,
                    fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
            ax7.set_title('System Stability Check')
        
        # Plot 8: Service time statistics comparison
        service_metrics = ['Mean', 'Variance', 'CV²']
        theoretical_service = [
            result.statistics['theoretical_service_mean'],
            result.statistics['theoretical_service_var'],
            result.statistics['theoretical_cs_squared']
        ]
        empirical_service = [
            result.statistics['empirical_service_mean'],
            result.statistics['empirical_service_var'],
            result.statistics['empirical_cs_squared']
        ]
        
        x = np.arange(len(service_metrics))
        width = 0.35
        
        ax8.bar(x - width/2, theoretical_service, width, label='Theoretical', alpha=0.7, color='blue')
        ax8.bar(x + width/2, empirical_service, width, label='Empirical', alpha=0.7, color='orange')
        
        ax8.set_ylabel('Value')
        ax8.set_title('Service Time Statistics')
        ax8.set_xticks(x)
        ax8.set_xticklabels(service_metrics)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (theo, emp) in enumerate(zip(theoretical_service, empirical_service)):
            ax8.text(i - width/2, theo + abs(theo)*0.01, f'{theo:.3f}', 
                    ha='center', va='bottom', fontsize=8)
            ax8.text(i + width/2, emp + abs(emp)*0.01, f'{emp:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # Plot 9: Impact of service distribution variability
        # Show how different CV values affect queue performance
        cv_values = [0.0, 0.5, 1.0, 1.5, 2.0]  # Different coefficients of variation
        lq_values = []
        
        for cv in cv_values:
            if result.results['traffic_intensity'] < 1:
                rho = result.results['traffic_intensity']
                lq = (rho ** 2) * (1 + cv**2) / (2 * (1 - rho))
                lq_values.append(lq)
            else:
                lq_values.append(float('inf'))
        
        # Only plot if system is stable
        if not any(np.isinf(lq) for lq in lq_values):
            ax9.plot(cv_values, lq_values, 'b-o', linewidth=2, markersize=6, label='Theoretical Lq')
            
            # Mark current system
            current_cv = np.sqrt(result.statistics['theoretical_cs_squared'])
            current_lq = result.statistics['theoretical_Lq']
            ax9.plot(current_cv, current_lq, 'ro', markersize=10, label=f'Current System (CV={current_cv:.2f})')
            
            ax9.set_xlabel('Coefficient of Variation (CV)')
            ax9.set_ylabel('Average Queue Length (Lq)')
            ax9.set_title('Impact of Service Variability on Queue Length')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'System Unstable\nCannot analyze\nvariability impact', 
                    ha='center', va='center', transform=ax9.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
            ax9.set_title('Service Variability Analysis')
        
        plt.tight_layout()
        plt.show()
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("M/G/1 QUEUE SIMULATION RESULTS")
        print("="*70)
        print(f"Simulation Parameters:")
        print(f"  Arrival Rate (λ): {self.arrival_rate:.3f}")
        print(f"  Service Distribution: {self.service_distribution}")
        print(f"  Service Mean: {self.service_mean:.3f}")
        print(f"  Service Parameters: {self.service_params}")
        print(f"  Traffic Intensity (ρ): {result.results['traffic_intensity']:.3f}")
        print(f"  Simulation Time: {self.simulation_time:.1f}")
        print(f"  Total Customers Served: {result.results['total_customers_served']}")
        
        print(f"\nService Time Statistics:")
        print(f"                          Theoretical    Empirical    Difference")
        print(f"  Mean Service Time:      {result.statistics['theoretical_service_mean']:8.3f}    {result.statistics['empirical_service_mean']:8.3f}    {abs(result.statistics['theoretical_service_mean'] - result.statistics['empirical_service_mean']):8.3f}")
        print(f"  Service Variance:       {result.statistics['theoretical_service_var']:8.3f}    {result.statistics['empirical_service_var']:8.3f}    {abs(result.statistics['theoretical_service_var'] - result.statistics['empirical_service_var']):8.3f}")
        print(f"  Coeff. of Variation²:   {result.statistics['theoretical_cs_squared']:8.3f}    {result.statistics['empirical_cs_squared']:8.3f}    {abs(result.statistics['theoretical_cs_squared'] - result.statistics['empirical_cs_squared']):8.3f}")
        
        if not any(np.isinf(v) for v in [result.statistics['theoretical_L'], result.statistics['theoretical_Lq']]):
            print(f"\nQueue Performance Metrics (Pollaczek-Khinchine):")
            print(f"                          Theoretical    Empirical    Difference")
            print(f"  Customers in System:    {result.statistics['theoretical_L']:8.3f}    {result.statistics['empirical_L']:8.3f}    {abs(result.statistics['theoretical_L'] - result.statistics['empirical_L']):8.3f}")
            print(f"  Customers in Queue:     {result.statistics['theoretical_Lq']:8.3f}    {result.statistics['empirical_Lq']:8.3f}    {abs(result.statistics['theoretical_Lq'] - result.statistics['empirical_Lq']):8.3f}")
            print(f"  System Time:            {result.statistics['theoretical_W']:8.3f}    {result.statistics['empirical_W']:8.3f}    {abs(result.statistics['theoretical_W'] - result.statistics['empirical_W']):8.3f}")
            print(f"  Waiting Time:           {result.statistics['theoretical_Wq']:8.3f}    {result.statistics['empirical_Wq']:8.3f}    {abs(result.statistics['theoretical_Wq'] - result.statistics['empirical_Wq']):8.3f}")
            print(f"  Server Utilization:     {result.statistics['theoretical_utilization']:8.3f}    {result.statistics['empirical_utilization']:8.3f}    {abs(result.statistics['theoretical_utilization'] - result.statistics['empirical_utilization']):8.3f}")
            
            # Compare with M/M/1 (exponential service)
            if self.service_distribution != 'exponential':
                mm1_lq = (result.results['traffic_intensity'] ** 2) / (1 - result.results['traffic_intensity'])
                improvement = ((mm1_lq - result.statistics['theoretical_Lq']) / mm1_lq) * 100
                print(f"\nComparison with M/M/1:")
                print(f"  M/M/1 Queue Length:     {mm1_lq:8.3f}")
                print(f"  M/G/1 Queue Length:     {result.statistics['theoretical_Lq']:8.3f}")
                if improvement > 0:
                    print(f"  Improvement:            {improvement:8.1f}% better")
                else:
                    print(f"  Performance:            {abs(improvement):8.1f}% worse")
        
        if result.results['traffic_intensity'] >= 1:
            print(f"\n⚠️  WARNING: System is unstable (ρ ≥ 1)!")
        
        print("="*70)
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'arrival_rate': {
                'type': 'float',
                'default': 0.8,
                'min': 0.1,
                'max': 5.0,
                'description': 'Average arrival rate (λ) in customers per unit time'
            },
            'service_mean': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Mean service time'
            },
            'service_distribution': {
                'type': 'choice',
                'default': 'exponential',
                'choices': ['exponential', 'deterministic', 'uniform', 'normal', 'gamma', 'lognormal'],
                'description': 'Service time distribution type'
            },
            'service_params': {
                'type': 'dict',
                'default': {},
                'description': 'Additional parameters for service distribution'
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
        if self.service_mean <= 0:
            errors.append("service_mean must be positive")
        if self.arrival_rate * self.service_mean >= 1:
            errors.append("arrival_rate * service_mean should be less than 1 for stability")
        if self.simulation_time <= 0:
            errors.append("simulation_time must be positive")
        if self.warmup_time < 0:
            errors.append("warmup_time must be non-negative")
        if self.warmup_time >= self.simulation_time:
            errors.append("warmup_time must be less than simulation_time")
        
        # Validate service distribution parameters
        if self.service_distribution == 'normal':
            std = self.service_params.get('std', 0.3 * self.service_mean)
            if std <= 0:
                errors.append("Normal distribution std parameter must be positive")
        elif self.service_distribution == 'uniform':
            min_val = self.service_params.get('min', 0.5 * self.service_mean)
            if min_val < 0:
                errors.append("Uniform distribution min parameter must be non-negative")
            if min_val >= 2 * self.service_mean:
                errors.append("Uniform distribution min parameter must be less than 2*service_mean")
        elif self.service_distribution in ['gamma', 'lognormal']:
            cv = self.service_params.get('cv', 1.0)
            if cv <= 0:
                errors.append("Coefficient of variation must be positive")
        
        return errors

