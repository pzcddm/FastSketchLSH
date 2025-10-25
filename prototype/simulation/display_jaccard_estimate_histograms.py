"""
display_jaccard_estimate_histograms.py
--------------------------------------
Author: Zhencan Peng
Date: 8/23/2025

Create a single figure with two subplots showing the probability distributions
of Jaccard estimates from FastSimilaritySketch and KMinSketch simulations.

- Left subplot: FastSketch histogram (target Jaccard=0.5)
- Right subplot: KMinSketch histogram (target Jaccard=0.5)

Each subplot title includes the empirical variance.

Figures are saved to `prototype/simulation/figures/combined_fast_and_kmins_hist.png`.

This script reuses dataset generators and estimators from `prototype.simulation.util` and
project `prototype/src/` implementations, without duplicating class definitions.
"""

from __future__ import annotations

import os
import sys
import random
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# Ensure project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prototype.simulation.util import generate_interval_sets_with_jaccard, estimate_jaccard  # type: ignore
from prototype.src.fast_sketch import FastSimilaritySketch  # type: ignore
from prototype.src.kmins_sketch import KMinSketch  # type: ignore


def run_fastsketch_simulation(
    sketch_size: int,
    num_simulations: int,
    target_jaccard: float,
    set_size: int,
    random_seed: int,
) -> Tuple[List[float], List[float], float]:
    """Run FastSketch simulations and return estimates, actual Js, and theoretical var.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    sketcher = FastSimilaritySketch(sketch_size=sketch_size, random_seed=random_seed)
    results: List[float] = []
    actual_jaccards: List[float] = []

    for i in range(num_simulations):
        set_A, set_B, actual_j = generate_interval_sets_with_jaccard(
            target_jaccard=target_jaccard,
            set_size=set_size,
            start_id=i * 100000,
        )
        sketch_A = sketcher.sketch(set_A)
        sketch_B = sketcher.sketch(set_B)
        est = estimate_jaccard(sketch_A, sketch_B)
        results.append(est)
        actual_jaccards.append(actual_j)

    mean_actual = float(np.mean(actual_jaccards))
    theoretical_var = mean_actual * (1 - mean_actual) / float(sketch_size)
    return results, actual_jaccards, theoretical_var


def run_kmins_simulation(
    k: int,
    num_simulations: int,
    target_jaccard: float,
    set_size: int,
    random_seed: int,
) -> Tuple[List[float], List[float], float]:
    """Run KMinSketch simulations and return estimates, actual Js, and theoretical var.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    sketcher = KMinSketch(k=k, random_seed=random_seed)
    results: List[float] = []
    actual_jaccards: List[float] = []

    for i in range(num_simulations):
        set_A, set_B, actual_j = generate_interval_sets_with_jaccard(
            target_jaccard=target_jaccard,
            set_size=set_size,
            start_id=i * 100000,
        )
        sketch_A = sketcher.sketch(set_A)
        sketch_B = sketcher.sketch(set_B)
        est = estimate_jaccard(sketch_A, sketch_B)
        results.append(est)
        actual_jaccards.append(actual_j)

    mean_actual = float(np.mean(actual_jaccards))
    theoretical_var = mean_actual * (1 - mean_actual) / float(k)
    return results, actual_jaccards, theoretical_var


def main() -> None:
    # Parameters (mirroring standalone scripts for comparability)
    RANDOM_SEED_FAST = 1234
    RANDOM_SEED_KMINS = 1234

    target_j = 0.5
    num_simulations = 200
    set_size = 1000

    sketch_size = 128

    # Run both simulations
    print("Running FastSketch simulation...")
    fast_results, fast_actuals, fast_theoretical_var = run_fastsketch_simulation(
        sketch_size=sketch_size,
        num_simulations=num_simulations,
        target_jaccard=target_j,
        set_size=set_size,
        random_seed=RANDOM_SEED_FAST,
    )
    
    print("Running KMinSketch simulation...")
    kmins_results, kmins_actuals, kmins_theoretical_var = run_kmins_simulation(
        k=sketch_size,
        num_simulations=num_simulations,
        target_jaccard=target_j,
        set_size=set_size,
        random_seed=RANDOM_SEED_KMINS,
    )

    # Compute empirical stats
    fast_var = float(np.var(fast_results, ddof=1)) if len(fast_results) > 1 else 0.0
    kmins_var = float(np.var(kmins_results, ddof=1)) if len(kmins_results) > 1 else 0.0
    fast_mean_actual = float(np.mean(fast_actuals))
    kmins_mean_actual = float(np.mean(kmins_actuals))

    # Figure with two subplots
    plt.figure(figsize=(14, 6))

    # Left: FastSketch histogram
    plt.subplot(1, 2, 1)
    plt.hist(fast_results, bins=50, density=True, alpha=0.75,
             label=f'Simulation (k={sketch_size})')
    plt.axvline(fast_mean_actual, color='r', linestyle='--', linewidth=2,
                label=f'Mean actual J = {fast_mean_actual:.3f}')
    plt.title(
        f'FastSketch Distribution (k={sketch_size})\nVar={fast_var:.6f}'
    )
    plt.xlabel('Estimated Jaccard Similarity')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.5)
    plt.legend()

    # Right: KMinSketch histogram
    plt.subplot(1, 2, 2)
    plt.hist(kmins_results, bins=50, density=True, alpha=0.75,
             label=f'Simulation (k={sketch_size})')
    plt.axvline(kmins_mean_actual, color='r', linestyle='--', linewidth=2,
                label=f'Mean actual J = {kmins_mean_actual:.3f}')
    plt.title(
        f'KMinSketch Distribution (k={sketch_size})\nVar={kmins_var:.6f}'
    )
    plt.xlabel('Estimated Jaccard Similarity')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.5)
    plt.legend()

    # Save combined figure
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'figures'))
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, 'combined_fast_and_kmins_hist.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Figure saved to: {out_path}")


if __name__ == '__main__':
    main()


