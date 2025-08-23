import os
import sys
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import mmh3
import numpy as np
import matplotlib.pyplot as plt
import random
from simulation.util import generate_interval_sets_with_jaccard, estimate_jaccard
from src.fast_sketch import FastSimilaritySketch

# Set fixed random seeds for reproducibility
RANDOM_SEED = 52
NUMPY_SEED = 52

# Add these lines to handle Chinese fonts and minus sign rendering
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as default font
plt.rcParams['axes.unicode_minus'] = False    # Ensure minus '-' renders correctly

# Remove the FastSimilaritySketch class definition from this file

# --- Main simulation ---
if __name__ == '__main__':
    # Set seeds for reproducibility
    np.random.seed(NUMPY_SEED)
    random.seed(RANDOM_SEED)
    
    # --- Simulation parameters ---
    sketch_size_t = 256
    num_simulations = 400  # Number of simulations; more yields smoother curves
    target_jaccard = 0.5    # Target Jaccard similarity

    print(f"Starting simulation for {num_simulations} runs...")
    print(f"Target Jaccard: {target_jaccard:.4f}")

    # Create one sketcher instance with fixed seed for reproducibility
    sketcher = FastSimilaritySketch(sketch_size=sketch_size_t, random_seed=RANDOM_SEED)

    results = []
    actual_jaccards = []
    
    for i in range(num_simulations):
        if (i + 1) % 100 == 0:
            print(f"  ...completed {i+1}/{num_simulations} simulations")
        # Use interval-based set generation for reproducibility and efficiency
        set_A, set_B, actual_j = generate_interval_sets_with_jaccard(
            target_jaccard=target_jaccard,
            set_size=10000,
            start_id=i * 100000  # ensure no overlap between different simulations
        )
        sketch_A = sketcher.sketch(set_A)
        sketch_B = sketcher.sketch(set_B)
        estimated_j = estimate_jaccard(sketch_A, sketch_B)
        results.append(estimated_j)
        actual_jaccards.append(actual_j)
    
    avg_actual_jaccard = np.mean(actual_jaccards)
    print(f"Mean actual Jaccard: {avg_actual_jaccard:.4f}")
    print("mean, var:", np.mean(results), np.var(results, ddof=1))
    print("theoretical var:", avg_actual_jaccard*(1-avg_actual_jaccard)/sketch_size_t)
    print("Simulation complete.")

    # --- Plot probability distribution histogram ---
    plt.figure(figsize=(12, 6))
    # 'density=True' normalizes the Y-axis to probability density
    plt.hist(results, bins=50, density=True, alpha=0.75, label=f'Simulation distribution (t={sketch_size_t})')
    
    # Plot the mean actual Jaccard as a reference
    plt.axvline(avg_actual_jaccard, color='r', linestyle='--', linewidth=2, label=f'Mean actual Jaccard = {avg_actual_jaccard:.3f}')
    
    plt.title('Probability Distribution of FastSketch Estimates (Simulation)')
    plt.xlabel('Estimated Jaccard Similarity')
    plt.xlim(avg_actual_jaccard-0.1, avg_actual_jaccard+0.1)
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Save figure to the 'figures' folder inside simulation, with filename including parameters
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'figures'))
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    avg_set_size = np.mean([len(set_A), len(set_B)])
    figure_path = os.path.join(figures_dir, f'fast_jaccard_histogram_t{sketch_size_t}_jaccard{target_jaccard:.2f}_size{int(avg_set_size)}.png')
    plt.savefig(figure_path)
    print(f"Figure saved to: {figure_path}")
    
    plt.show()

    # Report P(estimate > 0.4)
    results_array = np.array(results)
    prob_greater_than_0_4 = np.sum(results_array > 0.4) / num_simulations
    print(f"\nFrom simulation, P(estimate > 0.4) ≈ {prob_greater_than_0_4:.6f}")