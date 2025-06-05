# Queueing Theory Simulations

This module provides comprehensive discrete-event simulations for various queueing systems, implementing classical queueing theory models with detailed statistical analysis and visualization capabilities.

## Overview

Queueing theory studies the behavior of waiting lines (queues) and is fundamental to understanding system performance in many domains including:

- **Computer Systems**: CPU scheduling, network packet processing, database transactions
- **Manufacturing**: Production lines, inventory management, quality control
- **Service Industries**: Call centers, hospitals, banks, retail checkout
- **Transportation**: Traffic flow, airport operations, logistics networks

## Implemented Queue Types

### 1. M/M/1 Queue (`MM1Queue`)

**Single-server queue with Poisson arrivals and exponential service times.**

- **Arrival Process**: Poisson with rate λ
- **Service Process**: Exponential with rate μ  
- **Servers**: 1
- **Capacity**: Unlimited
- **Discipline**: FIFO (First In, First Out)

**Key Metrics** (for ρ = λ/μ < 1):
- Average customers in system: L = ρ/(1-ρ)
- Average queue length: Lq = ρ²/(1-ρ)
- Average time in system: W = 1/(μ-λ)
- Average waiting time: Wq = ρ/(μ-λ)

```python
from simlib.queueing import MM1Queue

# Create M/M/1 queue with arrival rate 0.8, service rate 1.0
queue = MM1Queue(arrival_rate=0.8, service_rate=1.0, simulation_time=5000)
result = queue.run()
queue.visualize()

print(f"Average queue length: {result.results['avg_queue_length']:.3f}")
print(f"Server utilization: {result.results['server_utilization']:.3f}")
```

### 2. M/M/k Queue (`MMKQueue`)

**Multi-server queue with Poisson arrivals and exponential service times.**

- **Arrival Process**: Poisson with rate λ
- **Service Process**: Exponential with rate μ per server
- **Servers**: k parallel servers
- **Capacity**: Unlimited
- **Discipline**: FIFO with server selection

**Applications**: Call centers, multi-core processors, parallel service systems

```python
from simlib.queueing import MMKQueue

# Create M/M/3 queue (3 servers)
queue = MMKQueue(arrival_rate=2.5, service_rate=1.0, num_servers=3, simulation_time=5000)
result = queue.run()
queue.visualize()

print(f"Number of servers: {result.results['num_servers']}")
print(f"Server utilization: {result.results['server_utilization']:.3f}")
```

### 3. M/G/1 Queue (`MG1Queue`)

**Single-server queue with Poisson arrivals and general service time distribution.**

- **Arrival Process**: Poisson with rate λ
- **Service Process**: General distribution with mean 1/μ
- **Servers**: 1
- **Capacity**: Unlimited
- **Discipline**: FIFO

**Supported Service Distributions**:
- **Exponential**: M/M/1 equivalent
- **Deterministic**: M/D/1 (constant service times)
- **Uniform**: Service times uniformly distributed
- **Normal**: Normally distributed service times
- **Gamma**: Gamma distributed service times (configurable shape)

**Key Formula** (Pollaczek-Khinchine):
- Average customers in system: L = ρ + (λ²σ² + ρ²)/(2(1-ρ))
- Where σ² is the variance of service time distribution

```python
from simlib.queueing import MG1Queue

# M/D/1 queue (deterministic service times)
queue_det = MG1Queue(
    arrival_rate=0.8, 
    service_mean=1.0, 
    service_distribution='deterministic',
    simulation_time=5000
)
result = queue_det.run()

# M/G/1 with high variability (Gamma distribution)
queue_gamma = MG1Queue(
    arrival_rate=0.8,
    service_mean=1.0,
    service_distribution='gamma',
    service_params={'cv': 2.0},  # Coefficient of variation = 2
    simulation_time=5000
)
result_gamma = queue_gamma.run()

print(f"M/D/1 queue length: {result.results['avg_queue_length']:.3f}")
print(f"M/G/1 queue length: {result_gamma.results['avg_queue_length']:.3f}")
```

### 4. Queue Networks (`QueueNetwork`)

**Networks of interconnected queues with customer routing.**

- **Topology**: Arbitrary network of service stations
- **Routing**: Probabilistic routing matrix between stations
- **External Arrivals**: Poisson arrivals to any subset of stations
- **Service**: Exponential service at each station
- **Applications**: Manufacturing systems, computer networks, supply chains

**Features**:
- Jackson networks (product-form solutions)
- Tandem queues (series configuration)
- Feedback loops and complex routing
- Multi-class extensions possible

```python
from simlib.queueing import QueueNetwork

# 3-station tandem network
arrival_rates = [1.0, 0.0, 0.0]  # External arrivals only to station 1
service_rates = [1.5, 1.2, 1.8]

# Routing: 80% to next station, 20% exit system
routing_matrix = [
    [0.0, 0.8, 0.0],  # Station 1 → 80% to Station 2
    [0.0, 0.0, 0.8],  # Station 2 → 80% to Station 3
    [0.0, 0.0, 0.0]   # Station 3 → 100% exit
]

network = QueueNetwork(
    num_stations=3,
    arrival_rates=arrival_rates,
    service_rates=service_rates,
    routing_matrix=routing_matrix,
    simulation_time=5000
)

result = network.run()
network.visualize()

print("Station Performance:")
for i in range(3):
    print(f"Station {i+1}: Queue={result.results['avg_queue_lengths'][i]:.3f}, "
          f"Util={result.results['utilizations'][i]:.3f}")
```

## Simulation Features

### Statistical Analysis
- **Theoretical vs Empirical Comparison**: Compare simulation results with analytical predictions
- **Confidence Intervals**: Statistical confidence bounds on performance metrics
- **Steady-State Analysis**: Automatic warmup period detection and removal
- **Convergence Monitoring**: Track metric convergence over simulation time

### Visualization Capabilities
- **Time Series Plots**: Queue lengths, system sizes, and utilizations over time
- **Distribution Analysis**: Histograms of waiting times, service times, and inter-arrival times
- **Performance Dashboards**: Multi-panel displays of key metrics
- **Network Topology**: Visual representation of queue networks and routing

### Advanced Features
- **Reproducible Results**: Random seed control for consistent outputs
- **Parameter Validation**: Comprehensive input validation with helpful error messages
- **Performance Optimization**: Efficient event-driven simulation engine
- **Extensible Design**: Easy to add new queue types and service distributions

## Usage Examples

### Basic Queue Analysis
```python
# Compare different queue configurations
configs = [
    ("M/M/1", MM1Queue(0.8, 1.0, 5000)),
    ("M/D/1", MG1Queue(0.8, 1.0, 'deterministic', simulation_time=5000)),
    ("M/M/2", MMKQueue(0.8, 0.5, 2, 5000))  # Same total capacity
]

for name, queue in configs:
    result = queue.run()
    print(f"{name}: Queue Length = {result.results['avg_queue_length']:.3f}")
```

### Capacity Planning
```python
# Find minimum servers needed for target performance
target_wait_time = 2.0
arrival_rate = 3.0
service_rate = 1.0

for k in range(1, 10):
    if k * service_rate <= arrival_rate:
        continue  # Unstable system
    
    queue = MMKQueue(arrival_rate, service_rate, k, 3000)
    result = queue.run()
    
    if result.results['avg_waiting_time'] <= target_wait_time:
        print(f"Minimum servers needed: {k}")
        break
```

### Network Optimization
```python
# Optimize service rates in a network
def evaluate_network(service_rates):
    network = QueueNetwork(
        num_stations=3,
        arrival_rates=[2.0, 0.0, 0.0],
        service_rates=service_rates,
        simulation_time=2000
    )
    result = network.run()
    return max(result.results['avg_queue_lengths'])  # Minimize bottleneck

# Simple optimization loop
best_rates = [2.5, 2.5, 2.5]
best_performance = evaluate_network(best_rates)

print(f"Optimal service rates: {best_rates}")
print(f"Maximum queue length: {best_performance:.3f}")
```

## Performance Considerations

### Simulation Parameters
- **Simulation Time**: Longer simulations provide more accurate results but take more time
- **Warmup Period**: Allow system to reach steady state before collecting statistics
- **Random Seeds**: Use different seeds for multiple replications

### Computational Complexity
- **M/M/1, M/M/k**: O(n) where n is number of events
- **M/G/1**: O(n log n) due to service time generation
- **Networks**: O(n × k) where k is number of stations

### Memory Usage
- **Event Lists**: Grows with system load and simulation time
- **Statistics Collection**: Configurable detail level
- **Visualization Data**: Can be memory-intensive for long simulations

## Theoretical Background

### Little's Law
**L = λW** - The average number of customers in a system equals the arrival rate times the average time in the system.

### Traffic Intensity
**ρ = λ/μ** - The ratio of arrival rate to service rate. Systems are stable when ρ < 1.

### Kendall's Notation
**A/S/c/K/N/D** describes queueing systems:
- **A**: Arrival process (M=Markovian/Poisson, D=Deterministic, G=General)
- **S**: Service process (M=Markovian/Exponential, D=Deterministic, G=General)  
- **c**: Number of servers
- **K**: System capacity (default: ∞)
- **N**: Population size (default: ∞)
- **D**: Service discipline (default: FIFO)

### Jackson Networks
Networks of M/M/1 queues with Poisson external arrivals and Markovian routing have product-form solutions, meaning the steady-state distribution is the product of individual queue distributions.

## Validation and Testing

All queue implementations are validated against:
- **Analytical Results**: Theoretical formulas for performance metrics
- **Known Benchmarks**: Standard queueing theory examples
- **Limit Cases**: Behavior under extreme parameter values
- **Conservation Laws**: Little's Law, flow balance equations

## References

1. **Gross, D. & Harris, C.M.** (2008). *Fundamentals of Queueing Theory*, 4th Edition
2. **Kleinrock, L.** (1975). *Queueing Systems Volume 1: Theory*
3. **Bolch, G. et al.** (2006). *Queueing Networks and Markov Chains*, 2nd Edition
4. **Stewart, W.J.** (2009). *Probability, Markov Chains, Queues, and Simulation*
5. **Tijms, H.C.** (2003). *A First Course in Stochastic Models*

## Contributing

To add new queue types or extend existing functionality:

1. **Inherit from BaseSimulation**: Follow the established interface
2. **Implement Required Methods**: `run()`, `visualize()`, `validate_parameters()`
3. **Add Theoretical Calculations**: Include analytical results where available
4. **Write Tests**: Comprehensive unit tests with known results
5. **Update Documentation**: Include mathematical background and examples

## License

This module is part of the SimLib package and follows the same licensing terms.