#!/usr/bin/env python3
"""
Examples demonstrating queueing theory simulations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simlib.queueing import MM1Queue, MMKQueue, MG1Queue, QueueNetwork

def mm1_example():
    """Demonstrate M/M/1 queue simulation"""
    print("="*60)
    print("M/M/1 QUEUE EXAMPLE")
    print("="*60)
    
    # Create and run M/M/1 queue
    queue = MM1Queue(arrival_rate=0.8, service_rate=1.0, simulation_time=5000)
    result = queue.run()
    queue.visualize()

def mmk_example():
    """Demonstrate M/M/k queue simulation"""
    print("="*60)
    print("M/M/k QUEUE EXAMPLE")
    print("="*60)
    
    # Create and run M/M/3 queue
    queue = MMKQueue(arrival_rate=2.5, service_rate=1.0, num_servers=3, simulation_time=5000)
    result = queue.run()
    queue.visualize()

def mg1_example():
    """Demonstrate M/G/1 queue with different service distributions"""
    print("="*60)
    print("M/G/1 QUEUE EXAMPLES")
    print("="*60)
    
    # Example 1: Deterministic service times
    print("\n1. Deterministic Service Times (M/D/1)")
    queue_det = MG1Queue(arrival_rate=0.8, service_mean=1.0, 
                        service_distribution='deterministic', simulation_time=3000)
    result_det = queue_det.run()
    queue_det.visualize()
    
    # Example 2: High variability service times (Gamma distribution)
    print("\n2. High Variability Service Times (Gamma)")
    queue_gamma = MG1Queue(arrival_rate=0.8, service_mean=1.0,
                          service_distribution='gamma', 
                          service_params={'cv': 2.0},  # High coefficient of variation
                          simulation_time=3000)
    result_gamma = queue_gamma.run()
    queue_gamma.visualize()

def network_example():
    """Demonstrate queue network simulation"""
    print("="*60)
    print("QUEUE NETWORK EXAMPLE")
    print("="*60)
    
    # Create a 3-station tandem network
    arrival_rates = [1.0, 0.0, 0.0]  # External arrivals only to station 1
    service_rates = [1.5, 1.2, 1.8]
    
    # Routing matrix: 80% to next station, 20% exit
    routing_matrix = [
        [0.0, 0.8, 0.0],  # Station 1 -> 80% to Station 2
        [0.0, 0.0, 0.8],  # Station 2 -> 80% to Station 3  
        [0.0, 0.0, 0.0]   # Station 3 -> 100% exit
    ]
    
    network = QueueNetwork(
        num_stations=3,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        routing_matrix=routing_matrix,
        simulation_time=3000
    )
    
    result = network.run()
    network.visualize()

def comparison_example():
    """Compare different queue types with same traffic intensity"""
    print("="*60)
    print("QUEUE COMPARISON EXAMPLE")
    print("="*60)
    
    arrival_rate = 0.8
    service_rate = 1.0
    simulation_time = 3000
    
    print(f"Comparing queues with ρ = {arrival_rate/service_rate}")
    
    # M/M/1
    mm1 = MM1Queue(arrival_rate, service_rate, simulation_time)
    result_mm1 = mm1.run()
    
    # M/D/1 (deterministic service)
    md1 = MG1Queue(arrival_rate, 1.0/service_rate, 'deterministic', simulation_time=simulation_time)
    result_md1 = md1.run()
    
    # M/M/2 (two servers, half the service rate each to maintain same capacity)
    mm2 = MMKQueue(arrival_rate, service_rate/2, 2, simulation_time)
    mm2 = MMKQueue(arrival_rate, service_rate/2, 2, simulation_time)
    result_mm2 = mm2.run()
    
    print("\nComparison Results:")
    print(f"{'Metric':<20} {'M/M/1':<10} {'M/D/1':<10} {'M/M/2':<10}")
    print("-" * 50)
    print(f"{'Avg Queue Length':<20} {result_mm1.results['avg_queue_length']:<10.3f} {result_md1.results['avg_queue_length']:<10.3f} {result_mm2.results['avg_queue_length']:<10.3f}")
    print(f"{'Avg Wait Time':<20} {result_mm1.results['avg_waiting_time']:<10.3f} {result_md1.results['avg_waiting_time']:<10.3f} {result_mm2.results['avg_waiting_time']:<10.3f}")
    print(f"{'Server Utilization':<20} {result_mm1.results['server_utilization']:<10.3f} {result_md1.results['server_utilization']:<10.3f} {result_mm2.results['server_utilization']:<10.3f}")
    
    print("\nKey Insights:")
    print("- M/D/1 has lower queue length than M/M/1 due to deterministic service")
    print("- M/M/2 has much lower queue length due to multiple servers")
    print("- All systems have same traffic intensity but different performance")

def interactive_example():
    """Interactive example allowing user to experiment with parameters"""
    print("="*60)
    print("INTERACTIVE QUEUE SIMULATION")
    print("="*60)
    
    while True:
        print("\nChoose a queue type:")
        print("1. M/M/1 Queue")
        print("2. M/M/k Queue") 
        print("3. M/G/1 Queue")
        print("4. Queue Network")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            try:
                arrival_rate = float(input("Enter arrival rate (λ): "))
                service_rate = float(input("Enter service rate (μ): "))
                sim_time = float(input("Enter simulation time (default 1000): ") or "1000")
                
                queue = MM1Queue(arrival_rate, service_rate, sim_time)
                result = queue.run()
                queue.visualize()
                
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                
        elif choice == '2':
            try:
                arrival_rate = float(input("Enter arrival rate (λ): "))
                service_rate = float(input("Enter service rate per server (μ): "))
                num_servers = int(input("Enter number of servers (k): "))
                sim_time = float(input("Enter simulation time (default 1000): ") or "1000")
                
                queue = MMKQueue(arrival_rate, service_rate, num_servers, sim_time)
                result = queue.run()
                queue.visualize()
                
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                
        elif choice == '3':
            try:
                arrival_rate = float(input("Enter arrival rate (λ): "))
                service_mean = float(input("Enter mean service time: "))
                
                print("Service distributions:")
                print("1. Exponential")
                print("2. Deterministic") 
                print("3. Uniform")
                print("4. Normal")
                print("5. Gamma")
                
                dist_choice = input("Choose distribution (1-5): ").strip()
                distributions = {'1': 'exponential', '2': 'deterministic', 
                               '3': 'uniform', '4': 'normal', '5': 'gamma'}
                
                service_dist = distributions.get(dist_choice, 'exponential')
                sim_time = float(input("Enter simulation time (default 1000): ") or "1000")
                
                queue = MG1Queue(arrival_rate, service_mean, service_dist, simulation_time=sim_time)
                result = queue.run()
                queue.visualize()
                
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                
        elif choice == '4':
            try:
                num_stations = int(input("Enter number of stations (2-5): "))
                if num_stations < 2 or num_stations > 5:
                    print("Number of stations must be between 2 and 5")
                    continue
                    
                print("Enter arrival rates for each station:")
                arrival_rates = []
                for i in range(num_stations):
                    rate = float(input(f"Station {i+1} arrival rate: "))
                    arrival_rates.append(rate)
                
                print("Enter service rates for each station:")
                service_rates = []
                for i in range(num_stations):
                    rate = float(input(f"Station {i+1} service rate: "))
                    service_rates.append(rate)
                
                sim_time = float(input("Enter simulation time (default 1000): ") or "1000")
                
                # Use default tandem routing
                network = QueueNetwork(num_stations, arrival_rates, service_rates, 
                                     simulation_time=sim_time)
                result = network.run()
                network.visualize()
                
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                
        elif choice == '5':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-5.")

def main():
    """Main function to run examples"""
    print("QUEUEING THEORY SIMULATION EXAMPLES")
    print("="*60)
    
    examples = {
        '1': ('M/M/1 Queue Example', mm1_example),
        '2': ('M/M/k Queue Example', mmk_example), 
        '3': ('M/G/1 Queue Examples', mg1_example),
        '4': ('Queue Network Example', network_example),
        '5': ('Queue Comparison', comparison_example),
        '6': ('Interactive Mode', interactive_example),
        '7': ('Run All Examples', lambda: [mm1_example(), mmk_example(), mg1_example(), network_example(), comparison_example()])
    }
    
    while True:
        print("\nAvailable Examples:")
        for key, (name, _) in examples.items():
            print(f"{key}. {name}")
        print("8. Exit")
        
        choice = input("\nSelect an example (1-8): ").strip()
        
        if choice in examples:
            try:
                examples[choice][1]()
            except KeyboardInterrupt:
                print("\nExample interrupted by user.")
            except Exception as e:
                print(f"Error running example: {e}")
        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-8.")

if __name__ == "__main__":
    main()
