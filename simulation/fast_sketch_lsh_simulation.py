"""
fast_sketch_lsh_simulation.py
-----------------------------
Simulate FastSketch LSH collision probability as a function of true Jaccard similarity.

This script generates random set pairs with known Jaccard similarities and tests
how often they collide in at least one LSH band using FastSimilaritySketch.

Results are saved to simulation/fast_sketch_lsh_results.npy for plotting.

Author: (your name)
Date: (today's date)
"""
import numpy as np
import sys
import os
from typing import Set, Tuple

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.fast_sketch_lsh import FastSketchLSH
from util import generate_interval_sets_with_jaccard

def test_lsh_collision(set_A: Set[int], set_B: Set[int], bands: int, sketch_size: int, random_seed: int = 42) -> bool:
    """
    Test if two sets collide in at least one LSH band using FastSketch.
    
    Args:
        set_A: First set
        set_B: Second set  
        bands: Number of LSH bands
        sketch_size: Size of the sketch (must be divisible by bands)
        random_seed: Random seed for reproducibility
        
    Returns:
        True if sets collide in at least one band, False otherwise
        
    Time Complexity: O(sketch_size * (|set_A| + |set_B|))
    """
    # Create LSH instance
    lsh = FastSketchLSH(threshold=0.5, sketch_size=sketch_size, bands=bands, random_seed=random_seed)
    
    # Insert set_A with key "A"
    lsh.insert("A", set_A)
    
    # Query with set_B and check if "A" is returned (collision)
    candidates = lsh.query(set_B)
    return "A" in candidates

def simulate_lsh_curve(
    jaccard_values: np.ndarray, 
    set_size: int, 
    bands: int, 
    sketch_size: int
) -> (np.ndarray, np.ndarray):
    """
    Simulate LSH collision probability for different Jaccard similarities.
    For Jaccard in [0.5, 0.9], use 500 tests; otherwise, use 100 tests.
    
    Args:
        jaccard_values: Array of Jaccard similarities to test
        set_size: Size of each set in the pair
        bands: Number of LSH bands
        sketch_size: Size of the sketch
        
    Returns:
        (collision_probs, num_tests_per_jaccard)
        collision_probs: Array of collision probabilities corresponding to jaccard_values
        num_tests_per_jaccard: Array of num_tests used for each Jaccard value
        
    Time Complexity: O(sum(num_tests_per_jaccard) * sketch_size * set_size)
    """
    collision_probs = np.zeros(len(jaccard_values))
    num_tests_per_jaccard = np.zeros(len(jaccard_values), dtype=int)
    
    for i, target_jaccard in enumerate(jaccard_values):
        # Use 500 tests for 0.5 <= Jaccard <= 0.9, else 100
        if 0.4 <= target_jaccard <= 0.9:
            num_tests = 2000
        else:
            num_tests = 100
        num_tests_per_jaccard[i] = num_tests
        print(f"Testing Jaccard {target_jaccard:.3f} ({i+1}/{len(jaccard_values)}), num_tests={num_tests}")
        
        collisions = 0
        for test_idx in range(num_tests):
            set_A, set_B, actual_jaccard = generate_interval_sets_with_jaccard(
                target_jaccard, set_size, start_id=test_idx * 10000
            )
            if test_lsh_collision(set_A, set_B, bands, sketch_size, random_seed=42):
                collisions += 1
        collision_probs[i] = collisions / num_tests
        print(f"  Collision rate: {collision_probs[i]:.3f}")
    return collision_probs, num_tests_per_jaccard

def main():
    """
    Main simulation function.
    
    Parameters match the kmins_lsh_curve.py setup:
    - LSH bands: 16, rows per band: 8 (sketch_size = 128)
    - Jaccard values: 0.02 to 0.99
    - Set size: 1000 elements
    """
    # === Parameters ===
    bands = 16
    rows_per_band = 8
    sketch_size = bands * rows_per_band  # 128
    set_size = 1000
    
    # Jaccard values from 0.02 to 0.99 (exclusive bounds required)
    jaccard_values = np.linspace(0.02, 0.99, 50)
    
    print(f"FastSketch LSH Simulation")
    print(f"Bands: {bands}, Rows per band: {rows_per_band}, Sketch size: {sketch_size}")
    print(f"Set size: {set_size}")
    print(f"Jaccard range: {jaccard_values[0]:.3f} to {jaccard_values[-1]:.3f}")
    print("="*60)
    
    # Run simulation
    collision_probs, num_tests_per_jaccard = simulate_lsh_curve(jaccard_values, set_size, bands, sketch_size)
    
    # Save results
    results = {
        'jaccard_values': jaccard_values,
        'collision_probs': collision_probs,
        'bands': bands,
        'rows_per_band': rows_per_band,
        'sketch_size': sketch_size,
        'num_tests_per_jaccard': num_tests_per_jaccard,
        'set_size': set_size
    }
    
    output_file = "simulation/fast_sketch_lsh_results.npy"
    np.save(output_file, results)
    print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Min collision rate: {collision_probs.min():.3f}")
    print(f"Max collision rate: {collision_probs.max():.3f}")
    print(f"Mean collision rate: {collision_probs.mean():.3f}")

if __name__ == "__main__":
    main() 