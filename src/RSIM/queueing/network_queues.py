import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Dict, Tuple
import heapq
from collections import deque
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult

class QueueNetwork(BaseSimulation):
    """
    Simple Queue Network simulation - Multiple interconnected queues.
    
    A queue network consists of multiple service stations (queues) where
    customers can move between stations according to routing probabilities.
    This models complex systems like manufacturing networks, computer networks,
    or service systems with multiple stages.
    
    Network Structure:
    -----------------
    - External arrivals to specified stations
    - Internal routing between stations
    - External departures from any station
    - Feedback loops allowed
    
    Applications:
    ------------
    - Manufacturing systems with multiple workstations
    - Computer networks with packet routing
    - Hospital patient flow systems
    - Call center routing systems
    - Supply chain networks
    
    Parameters:
    -----------
    num_stations : int, default=3
        Number of service stations in the network
    arrival_rates : List[float], default=[1.0, 0.0, 0.0]
        External arrival rates to each station
    service_rates : List[float], default=[2.0, 1.5, 1.8]
        Service rates for each station
    routing_matrix : List[List[float]], default=None
        Routing probabilities between stations (row i to column j)
    simulation_time : float, default=1000.0
        Total simulation time
    warmup_time : float, default=100.0
        Warmup period
    random_seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, num_stations: int = 3, 
                 arrival_rates: List[float] = None,
                 service_rates: List[float] = None,
                 routing_matrix: List[List[float]] = None,
                 simulation_time: float = 1000.0,
                 warmup_time: float = 100.0,
                 random_seed: Optional[int] = None):
        super().__init__("Queue Network")
        
        self.num_stations = num_stations
        self.arrival_rates = arrival_rates or [1.0] + [0.0] * (num_stations - 1)
        self.service_rates = service_rates or [2.0, 1.5, 1.8][:num_stations]
        
        # Default routing matrix (simple tandem queue)
        if routing_matrix is None:
            self.routing_matrix = [[0.0] * num_stations for _ in range(num_stations)]
            for i in range(num_stations - 1):
                self.routing_matrix[i][i + 1] = 0.8  # 80% to next station
            # Last station: 100% external departure
        else:
            self.routing_matrix = routing_matrix
        
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        
        # Ensure lists are correct length
        while len(self.arrival_rates) < num_stations:
            self.arrival_rates.append(0.0)
        while len(self.service_rates) < num_stations:
            self.service_rates.append(1.0)
        
        self.parameters.update({
            'num_stations': num_stations,
            'arrival_rates': self.arrival_rates,
            'service_rates': self.service_rates,
            'routing_matrix': self.routing_matrix,
            'simulation_time': simulation_time,
            'warmup_time': warmup_time,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.is_configured = True
    
    def run(self, **kwargs) -> SimulationResult:
        """Execute queue network simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured.")
        
        start_time = time.time()
        
        # Initialize simulation state
        events = []
        current_time = 0.0
        queues = [deque() for _ in range(self.num_stations)]
        servers_busy = [False] * self.num_stations
        next_customer_id = 1
        
                # Statistics tracking
        statistics = {
            'customer_data': [],  # (customer_id, station, arrival_time, service_start, departure_time)
            'queue_lengths': [[] for _ in range(self.num_stations)],  # (time, length) for each station
            'system_sizes': [[] for _ in range(self.num_stations)],   # (time, size) for each station
            'throughput': [0] * self.num_stations,  # Customers served at each station
            'total_arrivals': [0] * self.num_stations,  # Total arrivals to each station
            'total_departures': [0] * self.num_stations  # Total departures from each station
        }
        
        def schedule_event(event_time: float, event_type: str, station: int, customer_id: int = None):
            heapq.heappush(events, (event_time, event_type, station, customer_id))
        
        def record_statistics():
            if current_time >= self.warmup_time:
                for i in range(self.num_stations):
                    queue_length = len(queues[i])
                    system_size = queue_length + (1 if servers_busy[i] else 0)
                    statistics['queue_lengths'][i].append((current_time, queue_length))
                    statistics['system_sizes'][i].append((current_time, system_size))
        
        def process_external_arrival(station: int, customer_id: int):
            nonlocal next_customer_id
            arrival_time = current_time
            statistics['total_arrivals'][station] += 1
            
            if not servers_busy[station]:
                # Server is free, start service immediately
                servers_busy[station] = True
                service_time = np.random.exponential(1.0 / self.service_rates[station])
                departure_time = current_time + service_time
                schedule_event(departure_time, 'departure', station, customer_id)
                
                if current_time >= self.warmup_time:
                    statistics['customer_data'].append((customer_id, station, arrival_time, arrival_time, departure_time))
            else:
                # Server is busy, join queue
                queues[station].append((customer_id, arrival_time))
            
            # Schedule next external arrival to this station
            if self.arrival_rates[station] > 0:
                next_arrival_time = current_time + np.random.exponential(1.0 / self.arrival_rates[station])
                if next_arrival_time <= self.simulation_time:
                    schedule_event(next_arrival_time, 'external_arrival', station, next_customer_id)
                    next_customer_id += 1
            
            record_statistics()
        
        def process_internal_arrival(station: int, customer_id: int):
            arrival_time = current_time
            statistics['total_arrivals'][station] += 1
            
            if not servers_busy[station]:
                # Server is free, start service immediately
                servers_busy[station] = True
                service_time = np.random.exponential(1.0 / self.service_rates[station])
                departure_time = current_time + service_time
                schedule_event(departure_time, 'departure', station, customer_id)
                
                if current_time >= self.warmup_time:
                    statistics['customer_data'].append((customer_id, station, arrival_time, arrival_time, departure_time))
            else:
                # Server is busy, join queue
                queues[station].append((customer_id, arrival_time))
            
            record_statistics()
        
        def process_departure(station: int, customer_id: int):
            departure_time = current_time
            statistics['total_departures'][station] += 1
            
            if current_time >= self.warmup_time:
                statistics['throughput'][station] += 1
            
            # Determine next destination
            rand = np.random.random()
            cumulative_prob = 0.0
            next_station = None
            
            for j in range(self.num_stations):
                cumulative_prob += self.routing_matrix[station][j]
                if rand <= cumulative_prob:
                    next_station = j
                    break
            
            if next_station is not None:
                # Route to another station
                schedule_event(current_time, 'internal_arrival', next_station, customer_id)
            # else: customer leaves the system
            
            # Start serving next customer in queue if any
            if queues[station]:
                next_customer_id, arrival_time = queues[station].popleft()
                service_start_time = current_time
                service_time = np.random.exponential(1.0 / self.service_rates[station])
                next_departure_time = current_time + service_time
                schedule_event(next_departure_time, 'departure', station, next_customer_id)
                
                if current_time >= self.warmup_time:
                    statistics['customer_data'].append((next_customer_id, station, arrival_time, service_start_time, next_departure_time))
            else:
                # No customers waiting, server becomes idle
                servers_busy[station] = False
            
            record_statistics()
        
        # Schedule initial external arrivals
        for station in range(self.num_stations):
            if self.arrival_rates[station] > 0:
                first_arrival_time = np.random.exponential(1.0 / self.arrival_rates[station])
                schedule_event(first_arrival_time, 'external_arrival', station, next_customer_id)
                next_customer_id += 1
        
        # Main simulation loop
        while events and current_time < self.simulation_time:
            event_time, event_type, station, customer_id = heapq.heappop(events)
            current_time = event_time
            
            if event_type == 'external_arrival':
                process_external_arrival(station, customer_id)
            elif event_type == 'internal_arrival':
                process_internal_arrival(station, customer_id)
            elif event_type == 'departure':
                process_departure(station, customer_id)
        
        execution_time = time.time() - start_time
        
        # Calculate performance metrics
        avg_queue_lengths = []
        avg_system_sizes = []
        utilizations = []
        
        for station in range(self.num_stations):
            # Time-weighted averages for queue lengths
            if statistics['queue_lengths'][station]:
                times, lengths = zip(*statistics['queue_lengths'][station])
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
            avg_queue_lengths.append(avg_queue_length)
            
            # Time-weighted averages for system sizes
            if statistics['system_sizes'][station]:
                times, sizes = zip(*statistics['system_sizes'][station])
                times = np.array(times)
                sizes = np.array(sizes)
                
                if len(times) > 1:
                    time_diffs = np.diff(times)
                    weighted_sum = np.sum(sizes[:-1] * time_diffs)
                    total_time = times[-1] - times[0]
                    avg_system_size = weighted_sum / total_time if total_time > 0 else 0
                else:
                    avg_system_size = sizes[0] if sizes else 0
            else:
                avg_system_size = 0
            avg_system_sizes.append(avg_system_size)
            
            # Server utilization
            utilization = (avg_system_size - avg_queue_length) if avg_system_size >= avg_queue_length else 0
            utilizations.append(utilization)
        
        # Calculate throughput rates
        analysis_time = self.simulation_time - self.warmup_time
        throughput_rates = [tp / analysis_time for tp in statistics['throughput']] if analysis_time > 0 else [0] * self.num_stations
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'avg_queue_lengths': avg_queue_lengths,
                'avg_system_sizes': avg_system_sizes,
                'utilizations': utilizations,
                'throughput_rates': throughput_rates,
                'total_throughput': statistics['throughput'],
                'total_arrivals': statistics['total_arrivals'],
                'total_departures': statistics['total_departures']
            },
            statistics={
                'effective_arrival_rates': [sum(statistics['total_arrivals'][i] for i in range(self.num_stations)) / self.simulation_time * self.arrival_rates[i] / sum(self.arrival_rates) if sum(self.arrival_rates) > 0 else 0 for i in range(self.num_stations)],
                'service_rates': self.service_rates,
                'traffic_intensities': [throughput_rates[i] / self.service_rates[i] if self.service_rates[i] > 0 else 0 for i in range(self.num_stations)]
            },
            raw_data={
                'customer_data': statistics['customer_data'],
                'queue_lengths': statistics['queue_lengths'],
                'system_sizes': statistics['system_sizes'],
                'routing_matrix': self.routing_matrix
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize queue network simulation results"""
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
        
        # Plot 1: Queue lengths over time for all stations
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        for station in range(self.num_stations):
            queue_data = result.raw_data['queue_lengths'][station]
            if queue_data:
                times, lengths = zip(*queue_data)
                color = colors[station % len(colors)]
                ax1.plot(times, lengths, color=color, linewidth=1, alpha=0.8, label=f'Station {station+1}')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Queue Length')
        ax1.set_title('Queue Lengths Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: System sizes over time for all stations
        for station in range(self.num_stations):
            system_data = result.raw_data['system_sizes'][station]
            if system_data:
                times, sizes = zip(*system_data)
                color = colors[station % len(colors)]
                ax2.plot(times, sizes, color=color, linewidth=1, alpha=0.8, label=f'Station {station+1}')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Customers in System')
        ax2.set_title('System Sizes Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average performance metrics by station
        stations = [f'Station {i+1}' for i in range(self.num_stations)]
        x = np.arange(len(stations))
        width = 0.25
        
        ax3.bar(x - width, result.results['avg_queue_lengths'], width, label='Avg Queue Length', alpha=0.7)
        ax3.bar(x, result.results['avg_system_sizes'], width, label='Avg System Size', alpha=0.7)
        ax3.bar(x + width, result.results['utilizations'], width, label='Utilization', alpha=0.7)
        
        ax3.set_ylabel('Value')
        ax3.set_title('Performance Metrics by Station')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stations)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Throughput rates
        ax4.bar(stations, result.results['throughput_rates'], alpha=0.7, color='green')
        ax4.set_ylabel('Throughput Rate (customers/time)')
        ax4.set_title('Throughput Rates by Station')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, rate in enumerate(result.results['throughput_rates']):
            ax4.text(i, rate + rate*0.01, f'{rate:.2f}', ha='center', va='bottom')
        
        # Plot 5: Routing matrix visualization
        routing_matrix = np.array(self.routing_matrix)
        im = ax5.imshow(routing_matrix, cmap='Blues', aspect='auto')
        ax5.set_xlabel('To Station')
        ax5.set_ylabel('From Station')
        ax5.set_title('Routing Probability Matrix')
        
        # Add text annotations
        for i in range(self.num_stations):
            for j in range(self.num_stations):
                text = ax5.text(j, i, f'{routing_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black" if routing_matrix[i, j] < 0.5 else "white")
        
        ax5.set_xticks(range(self.num_stations))
        ax5.set_yticks(range(self.num_stations))
        ax5.set_xticklabels([f'S{i+1}' for i in range(self.num_stations)])
        ax5.set_yticklabels([f'S{i+1}' for i in range(self.num_stations)])
        plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # Plot 6: Traffic intensities and stability
        traffic_intensities = result.statistics['traffic_intensities']
        colors_ti = ['green' if ti < 0.9 else 'orange' if ti < 1.0 else 'red' for ti in traffic_intensities]
        
        bars = ax6.bar(stations, traffic_intensities, alpha=0.7, color=colors_ti)
        ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Stability Limit')
        ax6.set_ylabel('Traffic Intensity (ρ)')
        ax6.set_title('Traffic Intensities by Station')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, ti) in enumerate(zip(bars, traffic_intensities)):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{ti:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("QUEUE NETWORK SIMULATION RESULTS")
        print("="*60)
        print(f"Network Configuration:")
        print(f"  Number of Stations: {self.num_stations}")
        print(f"  External Arrival Rates: {[f'{rate:.3f}' for rate in self.arrival_rates]}")
        print(f"  Service Rates: {[f'{rate:.3f}' for rate in self.service_rates]}")
        print(f"  Simulation Time: {self.simulation_time:.1f}")
        
        print(f"\nPerformance by Station:")
        print(f"{'Station':<8} {'Queue Len':<10} {'System Size':<12} {'Utilization':<12} {'Throughput':<12} {'Traffic ρ':<10}")
        print("-" * 70)
        for i in range(self.num_stations):
            print(f"{i+1:<8} {result.results['avg_queue_lengths'][i]:<10.3f} "
                  f"{result.results['avg_system_sizes'][i]:<12.3f} "
                  f"{result.results['utilizations'][i]:<12.3f} "
                  f"{result.results['throughput_rates'][i]:<12.3f} "
                  f"{result.statistics['traffic_intensities'][i]:<10.3f}")
        
        print(f"\nTotal Customers:")
        print(f"  Arrivals: {result.results['total_arrivals']}")
        print(f"  Departures: {result.results['total_departures']}")
        print(f"  Served: {result.results['total_throughput']}")
        
        # Check stability
        unstable_stations = [i+1 for i, ti in enumerate(result.statistics['traffic_intensities']) if ti >= 1.0]
        if unstable_stations:
            print(f"\n⚠️  WARNING: Stations {unstable_stations} may be unstable (ρ ≥ 1.0)")
        
        print("="*60)
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'num_stations': {
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 10,
                'description': 'Number of service stations in the network'
            },
            'arrival_rates': {
                'type': 'list',
                'default': [1.0, 0.0, 0.0],
                'description': 'External arrival rates to each station'
            },
            'service_rates': {
                'type': 'list',
                'default': [2.0, 1.5, 1.8],
                'description': 'Service rates for each station'
            },
            'routing_matrix': {
                'type': 'matrix',
                'default': None,
                'description': 'Routing probabilities between stations'
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
        
        if self.num_stations < 2:
            errors.append("num_stations must be at least 2")
        if self.num_stations > 10:
            errors.append("num_stations should not exceed 10 for performance reasons")
        
        if len(self.arrival_rates) != self.num_stations:
            errors.append("arrival_rates list length must match num_stations")
        if any(rate < 0 for rate in self.arrival_rates):
            errors.append("All arrival rates must be non-negative")
        if sum(self.arrival_rates) <= 0:
            errors.append("At least one arrival rate must be positive")
        
        if len(self.service_rates) != self.num_stations:
            errors.append("service_rates list length must match num_stations")
        if any(rate <= 0 for rate in self.service_rates):
            errors.append("All service rates must be positive")
        
        if self.simulation_time <= 0:
            errors.append("simulation_time must be positive")
        if self.warmup_time < 0:
            errors.append("warmup_time must be non-negative")
        if self.warmup_time >= self.simulation_time:
            errors.append("warmup_time must be less than simulation_time")
        
        # Validate routing matrix
        if len(self.routing_matrix) != self.num_stations:
            errors.append("routing_matrix must have num_stations rows")
        else:
            for i, row in enumerate(self.routing_matrix):
                if len(row) != self.num_stations:
                    errors.append(f"routing_matrix row {i} must have num_stations columns")
                if any(prob < 0 or prob > 1 for prob in row):
                    errors.append(f"routing_matrix row {i} contains invalid probabilities")
                if sum(row) > 1.0:
                    errors.append(f"routing_matrix row {i} probabilities sum to more than 1.0")
        
        return errors
    
    def configure(self, num_stations: int = 3,
                 arrival_rates: List[float] = None,
                 service_rates: List[float] = None,
                 routing_matrix: List[List[float]] = None,
                 simulation_time: float = 1000.0,
                 warmup_time: float = 100.0) -> bool:
        """Configure queue network parameters"""
        self.num_stations = num_stations
        self.arrival_rates = arrival_rates or [1.0] + [0.0] * (num_stations - 1)
        self.service_rates = service_rates or [2.0, 1.5, 1.8][:num_stations]
        
        # Default routing matrix (simple tandem queue)
        if routing_matrix is None:
            self.routing_matrix = [[0.0] * num_stations for _ in range(num_stations)]
            for i in range(num_stations - 1):
                self.routing_matrix[i][i + 1] = 0.8  # 80% to next station
        else:
            self.routing_matrix = routing_matrix
        
        self.simulation_time = simulation_time
        self.warmup_time = warmup_time
        
        # Ensure lists are correct length
        while len(self.arrival_rates) < num_stations:
            self.arrival_rates.append(0.0)
        while len(self.service_rates) < num_stations:
            self.service_rates.append(1.0)
        
        self.parameters.update({
            'num_stations': num_stations,
            'arrival_rates': self.arrival_rates,
            'service_rates': self.service_rates,
            'routing_matrix': self.routing_matrix,
            'simulation_time': simulation_time,
            'warmup_time': warmup_time
        })
        
        self.is_configured = True
        return True

