import sys
import os
import numpy as np

# Add the parent directory to sys.path to import simlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from simlib.markov_chains.discrete_markov import DiscreteMarkovChain
from simlib.markov_chains.continuous_markov import ContinuousMarkovChain
from simlib.markov_chains.mcmc import MetropolisHastings, GibbsSampler
from simlib.markov_chains.hidden_markov import HiddenMarkovModel

# Test discrete Markov chain
def test_discrete_markov():
    print("Testing Discrete Markov Chain...")
    # Create a discrete Markov chain
    transition_matrix = [[0.2, 0.3, 0.5],
                         [0.4, 0.1, 0.5],
                         [0.6, 0.2, 0.2]]
    markov_chain = DiscreteMarkovChain(
        transition_matrix=transition_matrix,
        n_states=3,
        n_steps=10000,
        n_chains=1,
        initial_state=0,
        random_seed=42,
    )
    # Simulate the Markov chain
    markov_chain.run()
    # Plot the results
    markov_chain.visualize()
    # Print the results
    print("Discrete Markov Chain Results:", markov_chain.result)
    print("✓ Discrete Markov Chain test completed\n")

def test_continuous_markov():
    print("Testing Continuous Markov Chain...")
    # Create a continuous Markov chain with rate matrix
    rate_matrix = [[-2.0, 1.0, 1.0],
                   [0.5, -1.5, 1.0],
                   [0.8, 0.7, -1.5]]
    
    continuous_chain = ContinuousMarkovChain(
        rate_matrix=rate_matrix,
        n_states=3,
        total_time=10.0,
        initial_state=0,        
        death_rate=1,
        random_seed=42
    )
    # Simulate the continuous Markov chain
    continuous_chain.run()
    # Visualize results
    continuous_chain.visualize()
    print("Continuous Markov Chain Results:", continuous_chain.result)
    print("✓ Continuous Markov Chain test completed\n")

def test_metropolis_hastings():
    print("Testing Metropolis-Hastings MCMC...")

    
    mh_sampler = MetropolisHastings(
    )
    
    # Run MCMC sampling
    mh_sampler.run()
    # Visualize results
    mh_sampler.visualize()
    print("Metropolis-Hastings acceptance rate:", mh_sampler.acceptance_rate)
    print("✓ Metropolis-Hastings test completed\n")

def test_gibbs_sampler():
    print("Testing Gibbs Sampler...")
    # Define conditional samplers for a bivariate normal distribution
    def conditional_sampler_1(x2, params=None):
        # Sample x1 | x2
        return np.random.normal(0.5 * x2, np.sqrt(0.75))
    
    def conditional_sampler_2(x1, params=None):
        # Sample x2 | x1
        return np.random.normal(0.5 * x1, np.sqrt(0.75))
    
    conditional_samplers = [conditional_sampler_1, conditional_sampler_2]
    
    gibbs_sampler = GibbsSampler(

    )
    
    # Run Gibbs sampling
    gibbs_sampler.run()
    # Visualize results
    gibbs_sampler.visualize()
    print("Gibbs Sampler Results shape:", gibbs_sampler.samples.shape)
    print("✓ Gibbs Sampler test completed\n")

def test_hidden_markov_model():
    print("Testing Hidden Markov Model...")

    hmm = HiddenMarkovModel(

    )
    
    # Generate some observations
    observations = hmm.run()

    # Visualize results
    hmm.visualize()
    
    print("✓ Hidden Markov Model test completed\n")

def run_all_tests():
    print("Running all Markov Chain tests...\n")
    
    try:
        test_discrete_markov()
    except Exception as e:
        print(f"✗ Discrete Markov Chain test failed: {e}\n")
    
    try:
        test_continuous_markov()
    except Exception as e:
        print(f"✗ Continuous Markov Chain test failed: {e}\n")
    
    try:
        test_metropolis_hastings()
    except Exception as e:
        print(f"✗ Metropolis-Hastings test failed: {e}\n")
    
    try:
        test_gibbs_sampler()
    except Exception as e:
        print(f"✗ Gibbs Sampler test failed: {e}\n")


    try:
        test_hidden_markov_model()
    except Exception as e:
        print(f"✗ Hidden Markov Model test failed: {e}\n")
    
    print("All tests completed!")

if __name__ == "__main__":
    run_all_tests()
