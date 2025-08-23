"""
display_lsh_probdist.py
-----------------------
Author: Zhencan Peng
Date: 8/23/2025

Generate two figures without intermediate temporary files by combining the logic
from `fast_sketch_lsh_simulation.py` and `kmins_lsh_curve.py`:

- Figure 1: LSH theoretical curve 1 - (1 - J^r)^b overlaid with FastSketch LSH
  simulated collision probabilities (bands b, rows-per-band r).
- Figure 2: k-mins theoretical acceptance probability compared with FastSketch
  simulated acceptance probability Pr[estimate >= theta] as a function of true
  Jaccard similarity.

Outputs:
- Saves figures into `simulation/figures/`:
  - `kmins_vs_fastsketch_in_lsh_probdist.png`
  - `kmins_vs_fastsketch_probdist.png`

Notes:
- This script does NOT save any temporary `.npy` files.
- Requires the project `src/` and this `simulation/` directory to be importable.
"""

from __future__ import annotations

import os
import sys
from math import comb as math_comb
from typing import Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Make fonts bigger globally for readability in saved figures
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# Ensure imports work when running this file directly
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(CURRENT_DIR)

from src.fast_sketch_lsh import FastSketchLSH  # type: ignore
from src.fast_sketch import FastSimilaritySketch  # type: ignore
from util import generate_interval_sets_with_jaccard, estimate_jaccard  # type: ignore


"""
User-configurable parameters
Update the following values to change the simulation and plotting settings.
"""
# Threshold for acceptance in k-mins and FastSketch comparisons
THETA: float = 0.70

# LSH banding parameters
BANDS: int = 16
ROWS_PER_BAND: int = 8

# Sketch size used for both FastSketch and k-mins (defaults to b * r)
SKETCH_SIZE: int = BANDS * ROWS_PER_BAND

# Set sizes for different experiments
SET_SIZE_LSH: int = 1000          # set size used in LSH collision simulation
SET_SIZE_PROBDIST: int = 10000     # set size used in probability distribution comparison

# Jaccard values to evaluate for curves
JACCARD_VALUES: np.ndarray = np.linspace(0.02, 0.99, 50)

# Random seeds for reproducibility
RANDOM_SEED_LSH: int = 42          # for LSH candidate collision simulation
RANDOM_SEED_PROBDIST: int = 52     # for probability distribution comparison


def _ensure_figures_dir() -> str:
    """Create and return the `simulation/figures` directory path."""
    figures_dir = os.path.join(CURRENT_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def test_lsh_collision(
    set_A: Set[int],
    set_B: Set[int],
    bands: int,
    sketch_size: int,
    random_seed: int = 1234,
) -> bool:
    """Return True if two sets collide in at least one LSH band using FastSketch.

    Args:
        set_A: First set.
        set_B: Second set.
        bands: Number of LSH bands.
        sketch_size: Sketch size (must be divisible by bands).
        random_seed: Random seed for reproducibility.

    Returns:
        True if sets collide in at least one band, else False.

    Time Complexity:
        O(sketch_size * (|set_A| + |set_B|)) per call due to sketching.
    """
    lsh = FastSketchLSH(threshold=0.5, sketch_size=sketch_size, bands=bands, random_seed=random_seed)
    lsh.insert("A", set_A)
    candidates = lsh.query(set_B)
    return "A" in candidates


def simulate_fastsketch_lsh_curve(
    jaccard_values: np.ndarray,
    set_size: int,
    bands: int,
    rows_per_band: int,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate FastSketch LSH collision probability for a range of Jaccard values.

    For J in [0.37, 0.9], use 2000 trials; otherwise use 100 trials to save time.

    Args:
        jaccard_values: Array of true Jaccard similarities.
        set_size: Number of elements in each set.
        bands: Number of LSH bands (b).
        rows_per_band: Rows per band (r); sketch_size = b * r.
        random_seed: Random seed for reproducibility of LSH hashing.

    Returns:
        (collision_probs, num_tests_per_jaccard)
        collision_probs: Estimated collision probability for each J.
        num_tests_per_jaccard: Trials used for each J.

    Time Complexity:
        O(sum(trials(J)) * sketch_size * set_size), where sketch_size = b * r.
    """
    sketch_size = bands * rows_per_band
    collision_probs = np.zeros(len(jaccard_values), dtype=float)
    num_tests_per_jaccard = np.zeros(len(jaccard_values), dtype=int)

    for idx, target_j in enumerate(jaccard_values):
        num_tests = 2000 if 0.37 <= target_j <= 0.9 else 100
        num_tests_per_jaccard[idx] = num_tests
        print(f"[LSH] J={target_j:.3f} ({idx+1}/{len(jaccard_values)}), trials={num_tests}")

        collisions = 0
        for test_idx in range(num_tests):
            set_A, set_B, _ = generate_interval_sets_with_jaccard(
                target_j, set_size, start_id=test_idx * 10000
            )
            if test_lsh_collision(set_A, set_B, bands, sketch_size, random_seed=random_seed):
                collisions += 1

        collision_probs[idx] = collisions / num_tests
        print(f"  -> collision rate {collision_probs[idx]:.3f}")

    return collision_probs, num_tests_per_jaccard


def plot_lsh_curve(
    j_values: np.ndarray,
    collision_probs: np.ndarray,
    b: int,
    r: int,
    theta: float,
) -> None:
    """Plot theoretical LSH curve and FastSketch LSH simulated collision probability.

    Saves: `simulation/figures/kmins_vs_fastsketch_in_lsh_probdist.png`.
    """
    figures_dir = _ensure_figures_dir()
    plt.figure(figsize=(12, 8))

    # Theoretical LSH: 1 - (1 - J^r)^b
    x_dense = np.linspace(0.0, 1.0, 1000)
    y_theory = 1.0 - (1.0 - x_dense**r)**b
    plt.plot(x_dense, y_theory, linestyle='-', linewidth=2.5, color='red', label=f'k-mins LSH Theoretical (b={b}, r={r})')

    # Simulated FastSketch LSH
    plt.plot(j_values, collision_probs, marker='o', linestyle='-', linewidth=2.5, markersize=6,
             color='C0', label='FastSketch LSH (simulated)')

    plt.axvline(theta, color='grey', linestyle='--', linewidth=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Jaccard(A,B) = J')
    plt.ylabel('P(A and B in the same band)')
    plt.title(f'LSH Collision Probability (b={b}, r={r})')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=False)
    out_path = os.path.join(figures_dir, 'kmins_vs_fastsketch_in_lsh_probdist.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def _kmins_prob_curve(x: np.ndarray, k: int, theta: float) -> np.ndarray:
    """Return k-mins theoretical acceptance curve Pr[at least ceil(theta*k) of k].

    Time Complexity:
        O(k) terms evaluated for each point in x (vectorized arithmetic within each term).
    """
    imin = int(np.ceil(theta * k))
    coeffs = np.array([math_comb(k, i) for i in range(imin, k + 1)], dtype=float)
    y = np.zeros_like(x, dtype=float)
    for idx, i in enumerate(range(imin, k + 1)):
        y += coeffs[idx] * (x ** i) * ((1 - x) ** (k - i))
    return y


def plot_kmins_and_fastsketch_distribution(
    k_for_kmins: int,
    theta: float,
    sketch_size: int,
    j_values: np.ndarray,
    set_size: int = 10000,
    random_seed: int = 52,
) -> None:
    """Plot k-mins theoretical acceptance vs FastSketch simulated acceptance probability.

    Saves: `simulation/figures/kmins_theory_and_fastsketch_distribution.png`.

    For J in [0.37, 0.9], uses 2000 trials; otherwise 100 trials.

    Time Complexity:
        O(sum(trials(J)) * sketch_size * set_size) due to sketching both sets and
        computing the Jaccard estimate from sketches.
    """
    figures_dir = _ensure_figures_dir()

    # Theoretical k-mins curve
    x = np.asarray(j_values, dtype=float)
    y_kmins = _kmins_prob_curve(x, k_for_kmins, theta)

    # FastSketch simulated acceptance probability: Pr[estimate >= theta]
    sketcher = FastSimilaritySketch(sketch_size=sketch_size, random_seed=random_seed)
    y_fast = np.zeros_like(x, dtype=float)

    for idx, J in enumerate(x):
        num_tests = 2000 if 0.37 <= J <= 0.9 else 100
        accept = 0
        for test_idx in range(num_tests):
            set_A, set_B, _ = generate_interval_sets_with_jaccard(
                J, set_size, start_id=test_idx * 10000 + idx * 1000000
            )
            sketch_A = sketcher.sketch(set_A)
            sketch_B = sketcher.sketch(set_B)
            est = estimate_jaccard(sketch_A, sketch_B)
            if est >= theta:
                accept += 1
        y_fast[idx] = accept / num_tests

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(x, y_kmins, color='C3', linewidth=2.5, label=f'k-mins Theoretical (k={k_for_kmins}, θ={theta:.2f})')
    plt.plot(x, y_fast, 'o-', color='C0', linewidth=2.0, markersize=5,
             label=f'FastSketch Simulated (t={sketch_size}, θ={theta:.2f})')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Jaccard(A,B) = J')
    plt.ylabel('Probability')
    plt.title('k-mins Theoretical vs FastSketch (Probability Curves)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=False)
    plt.axvline(theta, color='grey', linestyle='--', linewidth=2)
    out_path = os.path.join(figures_dir, 'kmins_vs_fastsketch_probdist.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    """Run both simulations and generate both figures without temp files.

    Parameters mirror prior scripts for consistency:
    - bands (b) = 16
    - rows_per_band (r) = 8
    - sketch size = b * r = 128
    - set size per set = 1000 (for LSH) and 10000 (distribution comparison)
    - Jaccard values = linspace(0.02, 0.99, 50)
    """
    # Parameters (from the configuration section above)
    bands = BANDS
    rows_per_band = ROWS_PER_BAND
    sketch_size = SKETCH_SIZE
    set_size_lsh = SET_SIZE_LSH
    j_values = JACCARD_VALUES

    print("FastSketch LSH + k-mins combined plotting")
    print(f"Bands: {bands}, Rows per band: {rows_per_band}, Sketch size: {sketch_size}")
    print(f"Theta: {THETA:.2f}")
    print(f"Set sizes: LSH={SET_SIZE_LSH}, ProbDist={SET_SIZE_PROBDIST}")
    print(f"Jaccard range: {j_values[0]:.3f}..{j_values[-1]:.3f}")
    print("=" * 60)

    # Simulate FastSketch LSH collision curve
    collision_probs, _ = simulate_fastsketch_lsh_curve(
        jaccard_values=j_values,
        set_size=set_size_lsh,
        bands=bands,
        rows_per_band=rows_per_band,
        random_seed=RANDOM_SEED_LSH,
    )

    # Plot LSH figure
    plot_lsh_curve(
        j_values=j_values,
        collision_probs=collision_probs,
        b=bands,
        r=rows_per_band,
        theta=THETA,
    )

    # Plot k-mins theoretical vs FastSketch distribution figure
    plot_kmins_and_fastsketch_distribution(
        k_for_kmins=sketch_size,
        theta=THETA,
        sketch_size=sketch_size,
        j_values=j_values,
        set_size=SET_SIZE_PROBDIST,
        random_seed=RANDOM_SEED_PROBDIST,
    )

    # Summary statistics for the LSH simulation
    print("\nSummary (LSH simulation):")
    print(f"Min collision rate: {collision_probs.min():.3f}")
    print(f"Max collision rate: {collision_probs.max():.3f}")
    print(f"Mean collision rate: {collision_probs.mean():.3f}")


if __name__ == "__main__":
    main()



