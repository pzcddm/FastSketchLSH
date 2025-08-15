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
import matplotlib.pyplot as plt
from math import comb as math_comb

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.fast_sketch_lsh import FastSketchLSH
from src.fast_sketch import FastSimilaritySketch
from util import generate_interval_sets_with_jaccard, estimate_jaccard

# Make fonts bigger globally
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

def _ensure_figures_dir() -> str:
    # Save figures in the same directory as this script
    figures_dir = os.path.dirname(__file__)
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir

def plot_lsh_curve(j_values: np.ndarray, collision_probs: np.ndarray, k: int, b: int, r: int) -> None:
    figures_dir = _ensure_figures_dir()
    plt.figure(figsize=(12, 8))
    # Plot theoretical k-mins LSH curve: 1 - (1 - J^r)^b
    x_dense = np.linspace(0.0, 1.0, 1000)
    y_theory = 1.0 - (1.0 - x_dense**r)**b
    plt.plot(x_dense, y_theory, linestyle='-', linewidth=2.5, color='red', label=f'k-mins LSH Theoretical (b={b}, r={r})')

    # Plot FastSketch LSH simulated curve
    plt.plot(j_values, collision_probs, marker='o', linestyle='-', linewidth=2.5, markersize=6, color='C0', label='FastSketch LSH (simulated)')
    # Vertical line at J=0.7
    plt.axvline(0.7, color='grey', linestyle='--', linewidth=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Jaccard(A,B) = J')
    plt.ylabel('P(A and B in the same band)')
    plt.title(f'k = {k}, b = {b}, r = {r}')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=False)
    out_path = os.path.join(figures_dir, 'lsh_curve.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def _kmins_prob_curve(x: np.ndarray, k: int, theta: float) -> np.ndarray:
    imin = int(np.ceil(theta * k))
    # Precompute binomial coefficients
    binom_coeffs = np.array([math_comb(k, i) for i in range(imin, k + 1)], dtype=float)
    y = np.zeros_like(x)
    # Sum_{i=imin..k} C(k,i) x^i (1-x)^{k-i}
    for idx, i in enumerate(range(imin, k + 1)):
        y += binom_coeffs[idx] * (x ** i) * ((1 - x) ** (k - i))
    return y

def plot_kmins_and_fastsketch_distribution(k_for_kmins: int, theta: float, t_fast: int, j_values: np.ndarray, set_size: int = 10000, random_seed: int = 52) -> None:
    figures_dir = _ensure_figures_dir()
    # k-mins theoretical curve over J
    x = np.asarray(j_values)
    y_kmins = _kmins_prob_curve(x, k_for_kmins, theta)

    # FastSketch simulated acceptance probability: P(estimate >= theta)
    sketcher = FastSimilaritySketch(sketch_size=t_fast, random_seed=random_seed)
    y_fast = np.zeros_like(x, dtype=float)
    for idx, J in enumerate(x):
        if 0.37 <= J <= 0.9:
            num_tests = 2000
        else:
            num_tests = 100
        accept = 0
        for test_idx in range(num_tests):
            set_A, set_B, _ = generate_interval_sets_with_jaccard(J, set_size, start_id=test_idx * 10000 + idx * 1000000)
            sketch_A = sketcher.sketch(set_A)
            sketch_B = sketcher.sketch(set_B)
            est = estimate_jaccard(sketch_A, sketch_B)
            if est >= theta:
                accept += 1
        y_fast[idx] = accept / num_tests

    # Plot both curves on the same axes
    plt.figure(figsize=(12, 8))
    plt.plot(x, y_kmins, color='C3', linewidth=2.5, label=f'k-mins Theoretical (k={k_for_kmins}, θ={theta:.2f})')
    plt.plot(x, y_fast, 'o-', color='C0', linewidth=2.0, markersize=5, label=f'FastSketch Simulated (t={t_fast}, θ={theta:.2f})')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Jaccard(A,B) = J')
    plt.ylabel('Probability')
    plt.title('k-mins Theoretical vs FastSketch (Probability Curves)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=False)
    # Vertical line at J=theta
    plt.axvline(theta, color='grey', linestyle='--', linewidth=2)
    out_path = os.path.join(figures_dir, 'kmins_theory_and_fastsketch_distribution.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def test_lsh_collision(set_A: Set[int], set_B: Set[int], bands: int, sketch_size: int, random_seed: int = 1234) -> bool:
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
) -> tuple[np.ndarray, np.ndarray]:
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
        # Use 500 tests for 0.37 <= Jaccard <= 0.9, else 100
        if 0.37 <= target_jaccard <= 0.9:
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
    
    # Save results in the same directory as this script
    output_file = os.path.join(os.path.dirname(__file__), "fast_sketch_lsh_results.npy")
    np.save(output_file, results)
    print(f"\nResults saved to {output_file}")

    # === Plot and save LSH figure ===
    plot_lsh_curve(
        j_values=jaccard_values,
        collision_probs=collision_probs,
        k=sketch_size,
        b=bands,
        r=rows_per_band,
    )
    print("LSH figure saved next to the script (lsh_curve.png)")

    # === Plot and save k-mins theoretical + FastSketch distribution figure ===
    plot_kmins_and_fastsketch_distribution(
        k_for_kmins=sketch_size,
        theta=0.70,
        t_fast=sketch_size,
        j_values=jaccard_values,
        set_size=1000,
        random_seed=52,
    )
    print("k-mins theory + FastSketch probability curves saved next to the script (kmins_theory_and_fastsketch_distribution.png)")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Min collision rate: {collision_probs.min():.3f}")
    print(f"Max collision rate: {collision_probs.max():.3f}")
    print(f"Mean collision rate: {collision_probs.mean():.3f}")

if __name__ == "__main__":
    main() 