import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from typing import Optional, List, Dict, Any, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..core.base import BaseSimulation, SimulationResult


class NetworkTrafficSimulation(BaseSimulation):
    """
    Network traffic flow simulation using packet routing and congestion modeling.
    
    This simulation models data packet transmission through a network where nodes 
    represent routers/switches and edges represent communication links. It includes 
    congestion effects, packet loss, and routing algorithms to study network 
    performance under various traffic loads.
    
    Mathematical Background:
    -----------------------
    Traffic Model:
    - Packets generated at source nodes with Poisson arrival process
    - Each packet has source, destination, and size
    - Links have capacity (packets per time unit) and delay
    - Congestion occurs when traffic exceeds link capacity
    
    Routing:
    - Shortest path routing (default)
    - Load balancing options
    - Adaptive routing based on congestion
    
    Performance Metrics:
    - Throughput: successful packet delivery rate
    - Latency: average packet transmission time
    - Packet loss rate due to congestion
    - Link utilization
    
    Applications:
    ------------
    - Internet traffic analysis
    - Data center network design
    - Telecommunication network planning
    - Quality of Service (QoS) analysis
    - Network capacity planning
    - Congestion control algorithms
    
    Parameters:
    -----------
    network : networkx.Graph or None
        Network topology (if None, generates random network)
    packet_generation_rate : float, default=0.1
        Average packets generated per node per time step
    link_capacity : float, default=5.0
        Maximum packets per link per time step
    packet_size_range : tuple, default=(1, 3)
        Range of packet sizes (min, max)
    simulation_time : int, default=100
        Number of time steps to simulate
    buffer_size : int, default=10
        Maximum packets that can be queued at each node
    network_params : dict, optional
        Parameters for network generation if network is None
    random_seed : int, optional
        Seed for random number generator
    """
    
    def __init__(self, network: Optional[nx.Graph] = None, 
                 packet_generation_rate: float = 0.1, link_capacity: float = 5.0,
                 packet_size_range: Tuple[int, int] = (1, 3), simulation_time: int = 100,
                 buffer_size: int = 10, network_params: Optional[Dict] = None,
                 random_seed: Optional[int] = None):
        super().__init__("Network Traffic Simulation")
        
        self.network = network
        self.packet_generation_rate = packet_generation_rate
        self.link_capacity = link_capacity
        self.packet_size_range = packet_size_range
        self.simulation_time = simulation_time
        self.buffer_size = buffer_size
        self.network_params = network_params or {'n_nodes': 20, 'edge_probability': 0.3}
        
        self.parameters.update({
            'packet_generation_rate': packet_generation_rate,
            'link_capacity': link_capacity,
            'packet_size_range': packet_size_range,
            'simulation_time': simulation_time,
            'buffer_size': buffer_size,
            'network_params': self.network_params,
            'random_seed': random_seed
        })
        
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        self.traffic_data = None
        self.is_configured = True
    
    def configure(self, network: Optional[nx.Graph] = None,
                 packet_generation_rate: float = 0.1, link_capacity: float = 5.0,
                 packet_size_range: Tuple[int, int] = (1, 3), simulation_time: int = 100,
                 buffer_size: int = 10) -> bool:
        """Configure traffic simulation parameters"""
        self.network = network
        self.packet_generation_rate = packet_generation_rate
        self.link_capacity = link_capacity
        self.packet_size_range = packet_size_range
        self.simulation_time = simulation_time
        self.buffer_size = buffer_size
        
        self.parameters.update({
            'packet_generation_rate': packet_generation_rate,
            'link_capacity': link_capacity,
            'packet_size_range': packet_size_range,
            'simulation_time': simulation_time,
            'buffer_size': buffer_size
        })
        
        self.is_configured = True
        return True
    
    def run(self, **kwargs) -> SimulationResult:
        """Run network traffic simulation"""
        if not self.is_configured:
            raise RuntimeError("Simulation not configured. Call configure() first.")
        
        start_time = time.time()
        
        # Create network if not provided
        if self.network is None:
            self.network = nx.erdos_renyi_graph(
                self.network_params['n_nodes'],
                self.network_params['edge_probability']
            )
            # Ensure connectivity
            if not nx.is_connected(self.network):
                # Add edges to make it connected
                components = list(nx.connected_components(self.network))
                for i in range(len(components) - 1):
                    node1 = list(components[i])[0]
                    node2 = list(components[i + 1])[0]
                    self.network.add_edge(node1, node2)
        
        nodes = list(self.network.nodes())
        n_nodes = len(nodes)
        
        # Initialize network state
        node_buffers = {node: [] for node in nodes}  # Packet queues at each node
        link_loads = {edge: 0 for edge in self.network.edges()}  # Current load on each link
        
        # Statistics tracking
        packets_generated = 0
        packets_delivered = 0
        packets_dropped = 0
        total_latency = 0
        delivered_packets = []
        
        # Time series data
        time_series = {
            'time': [],
            'packets_in_network': [],
            'throughput': [],
            'avg_latency': [],
            'packet_loss_rate': [],
            'network_utilization': []
        }
        
        # Packet class for tracking
        class Packet:
            def __init__(self, packet_id, source, destination, size, creation_time):
                self.id = packet_id
                self.source = source
                self.destination = destination
                self.size = size
                self.creation_time = creation_time
                self.current_node = source
                self.path = [source]
                self.hops = 0
        
        active_packets = []
        packet_id_counter = 0
        
        # Precompute shortest paths for routing
        shortest_paths = dict(nx.all_pairs_shortest_path(self.network))
        
        for t in range(self.simulation_time):
            # Generate new packets
            for node in nodes:
                if np.random.random() < self.packet_generation_rate:
                    # Choose random destination (different from source)
                    possible_destinations = [n for n in nodes if n != node]
                    if possible_destinations:
                        destination = np.random.choice(possible_destinations)
                        size = np.random.randint(self.packet_size_range[0], 
                                               self.packet_size_range[1] + 1)
                        
                        packet = Packet(packet_id_counter, node, destination, size, t)
                        packet_id_counter += 1
                        packets_generated += 1
                        
                        # Add to source node buffer if there's space
                        if len(node_buffers[node]) < self.buffer_size:
                            node_buffers[node].append(packet)
                            active_packets.append(packet)
                        else:
                            packets_dropped += 1
            
            # Process packets at each node
            new_link_loads = {edge: 0 for edge in self.network.edges()}
            
            for node in nodes:
                if node_buffers[node]:
                    # Process packets in buffer (FIFO)
                    packets_to_process = list(node_buffers[node])
                    node_buffers[node] = []
                    
                    for packet in packets_to_process:
                        if packet.current_node == packet.destination:
                            # Packet reached destination
                            packets_delivered += 1
                            latency = t - packet.creation_time
                            total_latency += latency
                            delivered_packets.append({
                                'latency': latency,
                                'hops': packet.hops,
                                'size': packet.size
                            })
                            active_packets.remove(packet)
                        else:
                            # Route packet to next hop
                            try:
                                path = shortest_paths[packet.current_node][packet.destination]
                                if len(path) > 1:
                                    next_hop = path[1]
                                    edge = (min(packet.current_node, next_hop), 
                                           max(packet.current_node, next_hop))
                                    
                                    # Check link capacity
                                    if new_link_loads[edge] + packet.size <= self.link_capacity:
                                        # Forward packet
                                        new_link_loads[edge] += packet.size
                                        packet.current_node = next_hop
                                        packet.path.append(next_hop)
                                        packet.hops += 1
                                        
                                        # Add to next node's buffer
                                        if len(node_buffers[next_hop]) < self.buffer_size:
                                            node_buffers[next_hop].append(packet)
                                        else:
                                            # Buffer overflow, drop packet
                                            packets_dropped += 1
                                            active_packets.remove(packet)
                                    else:
                                        # Link congested, keep in current buffer
                                        if len(node_buffers[packet.current_node]) < self.buffer_size:
                                            node_buffers[packet.current_node].append(packet)
                                        else:
                                            packets_dropped += 1
                                            active_packets.remove(packet)
                                else:
                                    # Already at destination (shouldn't happen)
                                    packets_delivered += 1
                                    active_packets.remove(packet)
                            except KeyError:
                                # No path to destination, drop packet
                                packets_dropped += 1
                                active_packets.remove(packet)
            
            link_loads = new_link_loads
            
            # Calculate statistics for this time step
            packets_in_network = len(active_packets)
            
            if t > 0:
                throughput = packets_delivered / t if t > 0 else 0
                avg_latency = total_latency / packets_delivered if packets_delivered > 0 else 0
                packet_loss_rate = packets_dropped / packets_generated if packets_generated > 0 else 0
            else:
                throughput = 0
                avg_latency = 0
                packet_loss_rate = 0
            
            # Network utilization (average link utilization)
            total_capacity = len(self.network.edges()) * self.link_capacity
            current_load = sum(link_loads.values())
            network_utilization = current_load / total_capacity if total_capacity > 0 else 0
            
            # Store time series data
            time_series['time'].append(t)
            time_series['packets_in_network'].append(packets_in_network)
            time_series['throughput'].append(throughput)
            time_series['avg_latency'].append(avg_latency)
            time_series['packet_loss_rate'].append(packet_loss_rate)
            time_series['network_utilization'].append(network_utilization)
        
        execution_time = time.time() - start_time
        
        # Final statistics
        final_throughput = packets_delivered / self.simulation_time
        final_packet_loss_rate = packets_dropped / packets_generated if packets_generated > 0 else 0
        final_avg_latency = total_latency / packets_delivered if packets_delivered > 0 else 0
        
        # Link utilization statistics
        link_utilizations = {}
        for edge in self.network.edges():
            utilization = link_loads[edge] / self.link_capacity
            link_utilizations[edge] = utilization
        
        avg_link_utilization = np.mean(list(link_utilizations.values()))
        max_link_utilization = np.max(list(link_utilizations.values()))
        
        result = SimulationResult(
            simulation_name=self.name,
            parameters=self.parameters.copy(),
            results={
                'packets_generated': packets_generated,
                'packets_delivered': packets_delivered,
                'packets_dropped': packets_dropped,
                'final_throughput': final_throughput,
                'final_packet_loss_rate': final_packet_loss_rate,
                'final_avg_latency': final_avg_latency,
                'avg_link_utilization': avg_link_utilization,
                'max_link_utilization': max_link_utilization,
                'network_size': n_nodes,
                'network_edges': self.network.number_of_edges()
            },
            statistics={
                'delivery_ratio': packets_delivered / packets_generated if packets_generated > 0 else 0,
                'avg_hops': np.mean([p['hops'] for p in delivered_packets]) if delivered_packets else 0,
                'latency_std': np.std([p['latency'] for p in delivered_packets]) if delivered_packets else 0,
                'time_series': time_series
            },
            raw_data={
                'network': self.network,
                'time_series': time_series,
                'delivered_packets': delivered_packets,
                'link_utilizations': link_utilizations
            },
            execution_time=execution_time
        )
        
        self.result = result
        return result
    
    def visualize(self, result: Optional[SimulationResult] = None, **kwargs) -> None:
        """Visualize network traffic simulation results"""
        if result is None:
            result = self.result
        
        if result is None:
            print("No simulation results available. Run the simulation first.")
            return
        
        network = result.raw_data['network']
        time_series = result.raw_data['time_series']
        link_utilizations = result.raw_data['link_utilizations']
        
        fig = plt.figure(figsize=(16, 12))
                
        # Plot 1: Network topology with link utilization
        ax1 = plt.subplot(2, 3, 1)
        pos = nx.spring_layout(network, seed=42)
        
        # Color edges by utilization
        edge_colors = []
        edge_widths = []
        for edge in network.edges():
            utilization = link_utilizations.get(edge, link_utilizations.get((edge[1], edge[0]), 0))
            edge_colors.append(utilization)
            edge_widths.append(1 + 3 * utilization)  # Width based on utilization
        
        # Draw network
        nx.draw_networkx_nodes(network, pos, node_color='lightblue', 
                              node_size=300, alpha=0.8)
        edges = nx.draw_networkx_edges(network, pos, edge_color=edge_colors, 
                                     width=edge_widths, edge_cmap=plt.cm.Reds,
                                     edge_vmin=0, edge_vmax=1, alpha=0.7)
        nx.draw_networkx_labels(network, pos, font_size=8)
        
        ax1.set_title('Network Topology\n(Edge color/width = utilization)')
        ax1.axis('off')
        
        # Add colorbar for edge utilization
        if edges:
            plt.colorbar(edges, ax=ax1, label='Link Utilization', shrink=0.8)
        
        # Plot 2: Throughput and packet loss over time
        ax2 = plt.subplot(2, 3, 2)
        times = time_series['time']
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(times, time_series['throughput'], 'b-', linewidth=2, label='Throughput')
        line2 = ax2_twin.plot(times, time_series['packet_loss_rate'], 'r-', linewidth=2, label='Packet Loss Rate')
        
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Throughput (packets/time)', color='b')
        ax2_twin.set_ylabel('Packet Loss Rate', color='r')
        ax2.set_title('Network Performance Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        # Plot 3: Latency distribution
        ax3 = plt.subplot(2, 3, 3)
        delivered_packets = result.raw_data['delivered_packets']
        if delivered_packets:
            latencies = [p['latency'] for p in delivered_packets]
            ax3.hist(latencies, bins=20, alpha=0.7, edgecolor='black', density=True)
            ax3.axvline(x=np.mean(latencies), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(latencies):.2f}')
            ax3.set_xlabel('Packet Latency (time steps)')
            ax3.set_ylabel('Density')
            ax3.set_title('Packet Latency Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Network utilization over time
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(times, time_series['network_utilization'], 'purple', linewidth=2)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Full Capacity')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Network Utilization')
        ax4.set_title('Overall Network Utilization')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, max(1.2, max(time_series['network_utilization']) * 1.1))
        
        # Plot 5: Traffic statistics summary
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        stats_text = f"""Traffic Statistics:
        
Network: {result.results['network_size']} nodes, {result.results['network_edges']} edges
Simulation Time: {self.simulation_time} steps
        
Packets Generated: {result.results['packets_generated']}
Packets Delivered: {result.results['packets_delivered']}
Packets Dropped: {result.results['packets_dropped']}
        
Final Throughput: {result.results['final_throughput']:.3f} packets/time
Packet Loss Rate: {result.results['final_packet_loss_rate']:.2%}
Average Latency: {result.results['final_avg_latency']:.2f} steps
Delivery Ratio: {result.statistics['delivery_ratio']:.2%}
        
Avg Hops: {result.statistics['avg_hops']:.2f}
Max Link Utilization: {result.results['max_link_utilization']:.2%}
Avg Link Utilization: {result.results['avg_link_utilization']:.2%}
        """
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Plot 6: Hop count distribution
        ax6 = plt.subplot(2, 3, 6)
        if delivered_packets:
            hop_counts = [p['hops'] for p in delivered_packets]
            unique_hops, counts = np.unique(hop_counts, return_counts=True)
            
            ax6.bar(unique_hops, counts, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Number of Hops')
            ax6.set_ylabel('Number of Packets')
            ax6.set_title('Hop Count Distribution')
            ax6.grid(True, alpha=0.3)
            
            # Add mean line
            mean_hops = np.mean(hop_counts)
            ax6.axvline(x=mean_hops, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_hops:.2f}')
            ax6.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_parameter_info(self) -> dict:
        """Return parameter information for UI generation"""
        return {
            'packet_generation_rate': {
                'type': 'float',
                'default': 0.1,
                'min': 0.01,
                'max': 1.0,
                'description': 'Average packets generated per node per time step'
            },
            'link_capacity': {
                'type': 'float',
                'default': 5.0,
                'min': 1.0,
                'max': 20.0,
                'description': 'Maximum packets per link per time step'
            },
            'simulation_time': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 1000,
                'description': 'Number of time steps to simulate'
            },
            'buffer_size': {
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 50,
                'description': 'Maximum packets that can be queued at each node'
            },
            'packet_size_range': {
                'type': 'tuple',
                'default': (1, 3),
                'description': 'Range of packet sizes (min, max)'
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
        if self.packet_generation_rate <= 0:
            errors.append("packet_generation_rate must be positive")
        if self.link_capacity <= 0:
            errors.append("link_capacity must be positive")
        if self.simulation_time < 1:
            errors.append("simulation_time must be at least 1")
        if self.buffer_size < 1:
            errors.append("buffer_size must be at least 1")
        if len(self.packet_size_range) != 2 or self.packet_size_range[0] > self.packet_size_range[1]:
            errors.append("packet_size_range must be (min, max) with min <= max")
        if self.packet_size_range[0] < 1:
            errors.append("minimum packet size must be at least 1")
        return errors
