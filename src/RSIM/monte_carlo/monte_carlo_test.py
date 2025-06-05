#!/usr/bin/env python3
"""
Simple test script to verify Monte Carlo simulations work
"""
import sys
import os

# Add the parent directory to sys.path to import simlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from .pi_estimation import PiEstimationMC
from .random_walk import RandomWalk1D, RandomWalk2D

def test_pi_estimation():
    """Test π estimation"""
    print("Testing π Estimation...")
    try:
        # Create and run π estimation
        pi_sim = PiEstimationMC(n_samples=100000, random_seed=42)
        result = pi_sim.run()
        
        print(f"π estimate: {result.results['pi_estimate']:.6f}")
        print(f"True π: {3.141593:.6f}")
        print(f"Error: {result.results['accuracy']:.6f}")
        print(f"Relative error: {result.results['relative_error']:.4f}%")
        print(f"Execution time: {result.execution_time:.4f} seconds")
        print("✓ π Estimation works!")
        
        # Test visualization (optional)
        print("Testing visualization...")
        pi_sim.visualize(show_points=False)
        print("✓ π Estimation visualization works!")
        
    except Exception as e:
        print(f"✗ π Estimation failed: {e}")
        raise

def test_random_walk_1d():
    """Test 1D random walk"""
    print("\nTesting 1D Random Walk...")
    try:
        # Single walk
        print("- Testing single walk...")
        rw1d_single = RandomWalk1D(n_steps=1000, n_walks=1, random_seed=42)
        result = rw1d_single.run()
        
        print(f"Final position: {result.results['final_positions'][0]}")
        print(f"Max distance: {result.results['max_distance_reached']}")
        print(f"Execution time: {result.execution_time:.4f} seconds")
        
        # Multiple walks
        print("- Testing multiple walks...")
        rw1d_multi = RandomWalk1D(n_steps=500, n_walks=50, step_probability=0.6, random_seed=42)
        result = rw1d_multi.run()
        
        print(f"Mean final position: {result.results['mean_final_position']:.2f}")
        print(f"Std final position: {result.results['std_final_position']:.2f}")
        print(f"Theoretical mean: {result.statistics['theoretical_mean']:.2f}")
        print(f"Empirical mean: {result.statistics['empirical_mean']:.2f}")
        
        print("✓ 1D Random Walk works!")
        
        # Test visualization
        print("Testing visualization...")
        rw1d_single.visualize()
        rw1d_multi.visualize()
        print("✓ 1D Random Walk visualization works!")
        
    except Exception as e:
        print(f"✗ 1D Random Walk failed: {e}")
        raise

def test_random_walk_2d():
    """Test 2D random walk"""
    print("\nTesting 2D Random Walk...")
    try:
        # Single walk
        print("- Testing single 2D walk...")
        rw2d_single = RandomWalk2D(n_steps=1000, n_walks=1, random_seed=42)
        result = rw2d_single.run()
        
        final_x = result.results['final_x_positions'][0]
        final_y = result.results['final_y_positions'][0]
        final_dist = result.results['final_distances'][0]
        
        print(f"Final position: ({final_x:.2f}, {final_y:.2f})")
        print(f"Final distance: {final_dist:.2f}")
        print(f"Max distance: {result.results['max_distance_reached']:.2f}")
        
        # Multiple walks
        print("- Testing multiple 2D walks...")
        rw2d_multi = RandomWalk2D(n_steps=500, n_walks=30, random_seed=42)
        result = rw2d_multi.run()
        
        print(f"Mean final distance: {result.results['mean_final_distance']:.2f}")
        print(f"Mean X position: {result.results['mean_final_x']:.2f}")
        print(f"Mean Y position: {result.results['mean_final_y']:.2f}")
        print(f"Execution time: {result.execution_time:.4f} seconds")
        
        print("✓ 2D Random Walk works!")
        
        # Test visualization
        print("Testing visualization...")
        rw2d_single.visualize()
        rw2d_multi.visualize()
        print("✓ 2D Random Walk visualization works!")
        
    except Exception as e:
        print(f"✗ 2D Random Walk failed: {e}")
        raise

def test_parameter_validation():
    """Test parameter validation"""
    print("\nTesting parameter validation...")
    try:
        # Test π estimation validation
        pi_sim = PiEstimationMC(n_samples=500)  # Too few samples
        errors = pi_sim.validate_parameters()
        print(f"π estimation validation errors: {errors}")
        
        # Test 1D walk validation
        rw1d = RandomWalk1D(n_steps=-1, step_size=-1, n_walks=0)
        errors = rw1d.validate_parameters()
        print(f"1D walk validation errors: {errors}")
        
        # Test 2D walk validation
        rw2d = RandomWalk2D(n_steps=-1, step_size=-1, n_walks=0)
        errors = rw2d.validate_parameters()
        print(f"2D walk validation errors: {errors}")
        
        print("✓ Parameter validation works!")
        
    except Exception as e:
        print(f"✗ Parameter validation failed: {e}")
        raise

def test_configuration():
    """Test configuration methods"""
    print("\nTesting configuration...")
    try:
        # Test π estimation configuration
        pi_sim = PiEstimationMC()
        pi_sim.configure(n_samples=50000, show_convergence=False)
        print(f"π estimation configured: n_samples={pi_sim.n_samples}")
        
        # Test 1D walk configuration
        rw1d = RandomWalk1D()
        rw1d.configure(n_steps=2000, step_size=2.0, n_walks=10, step_probability=0.7)
        print(f"1D walk configured: n_steps={rw1d.n_steps}, step_size={rw1d.step_size}")
        
        # Test 2D walk configuration
        rw2d = RandomWalk2D()
        rw2d.configure(n_steps=1500, step_size=1.5, n_walks=20)
        print(f"2D walk configured: n_steps={rw2d.n_steps}, step_size={rw2d.step_size}")
        
        print("✓ Configuration works!")
        
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        raise

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING MONTE CARLO SIMULATIONS")
    print("="*60)
    
    try:
        test_pi_estimation()
        test_random_walk_1d()
        test_random_walk_2d()
        test_parameter_validation()
        test_configuration()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n" + "="*60)
        print(f"TEST FAILED! ✗")
        print(f"Error: {e}")
        print("="*60)
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
