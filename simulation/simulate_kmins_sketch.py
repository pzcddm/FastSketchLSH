import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import random
import mmh3
from util import generate_interval_sets_with_jaccard, estimate_jaccard
from src.kmins_sketch import KMinSketch

# Set fixed random seeds for reproducibility
RANDOM_SEED = 42
NUMPY_SEED = 42


# Add these lines to handle Chinese fonts and minus sign rendering
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as default font
plt.rcParams['axes.unicode_minus'] = False    # Ensure minus '-' renders correctly


def main():
    """
    Simulate the k-mins Jaccard estimator with random test data and fixed hash seeds.
    
    Time Complexity: O(num_simulations * (k * n)) where n is the number of elements in the set.
    Space Complexity: O(num_simulations)
    """
    # Set seeds for reproducibility
    np.random.seed(NUMPY_SEED)
    random.seed(RANDOM_SEED)
    
    # Sketch size (number of independent hash functions)
    k = 256
    num_simulations = 400
    target_jaccard = 0.5  # Target Jaccard similarity

    print(f"Starting simulation for {num_simulations} runs...")
    print(f"Target Jaccard: {target_jaccard:.4f}")

    # Create one sketcher instance with fixed seed for reproducibility
    sketcher = KMinSketch(k, random_seed=RANDOM_SEED)

    # Run simulations
    results = []
    actual_jaccards = []
    
    for i in range(num_simulations):
        # Use interval-based set generation for reproducibility and efficiency
        set_A, set_B, actual_j = generate_interval_sets_with_jaccard(
            target_jaccard=target_jaccard,
            set_size=10000,
            start_id=i * 100000  # ensure no overlap between different simulations
        )
        sketch_A = sketcher.sketch(set_A)
        sketch_B = sketcher.sketch(set_B)
        est_j = estimate_jaccard(sketch_A, sketch_B)
        results.append(est_j)
        actual_jaccards.append(actual_j)
        if (i + 1) % 100 == 0:
            print(f"Completed {i+1}/{num_simulations} simulations")
    
    avg_actual_jaccard = np.mean(actual_jaccards)
    print(f"Mean actual Jaccard: {avg_actual_jaccard:.4f}")
    print("mean, var:", np.mean(results), np.var(results, ddof=1))
    print("theoretical var:", avg_actual_jaccard*(1-avg_actual_jaccard)/k)
    
    print("Simulation complete.")

    # Plot histogram of estimated Jaccard similarities
    plt.figure(figsize=(12, 6))
    plt.hist(results, bins=50, density=True, alpha=0.75, label=f"Simulation Distribution (k={k})")
    plt.axvline(avg_actual_jaccard, color='r', linestyle='--', linewidth=2, label=f"Mean actual Jaccard: {avg_actual_jaccard:.4f}")
    plt.title('Histogram of Estimated Jaccard Similarity using KMinSketch')
    plt.xlabel('Estimated Jaccard Similarity')
    plt.xlim(avg_actual_jaccard-0.1, avg_actual_jaccard+0.1)
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Save figure to the 'figures' folder inside simulation, with filename including parameters
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'figures'))
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    avg_set_size = int(np.mean([len(set_A), len(set_B)]))
    figure_path = os.path.join(figures_dir, f'kmins_jaccard_histogram_k{k}_jaccard{target_jaccard:.2f}_size{avg_set_size}.png')
    plt.savefig(figure_path)
    print(f"Figure saved to: {figure_path}")
    
    plt.show()

    # Compute and report the probability that the estimated Jaccard exceeds a threshold (e.g., 0.4)
    results_array = np.array(results)
    prob_above_threshold = np.sum(results_array > 0.4) / num_simulations
    print(f"Probability that estimated Jaccard > 0.4: {prob_above_threshold:.6f}")


if __name__ == '__main__':
    main() 