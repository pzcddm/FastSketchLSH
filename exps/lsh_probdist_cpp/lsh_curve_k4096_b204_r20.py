"""
lsh_curve_k4096_b204_r20.py
---------------------------
Author: Zhencan Peng
Date: 2026-02-14

This script evaluates LSH collision probability for the specific setting:
    - FastSketch sketch size k = 4096
    - LSH bands B = 204
    - rows per band R = 20

Because the current C++ extension requires `num_perm % num_bands == 0`, we
cannot pass `num_perm=4096` and `num_bands=204` directly. Therefore, this
experiment uses the first `B * R = 4080` coordinates from each k=4096 sketch
for LSH indexing/querying, while the sketching itself remains k=4096.

The script mimics the plotting style in the prototype experiment and saves:
    - exps/lsh_probdist_cpp/figures/lsh_collision_cpp_k4096_b204_r20.png
    - exps/lsh_probdist_cpp/figures/lsh_collision_cpp_k4096_b204_r20.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from FastSketchLSH import FastSimilaritySketch, LSH  # type: ignore


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)

K_SKETCH: int = 4096
BANDS: int = 204
ROWS_PER_BAND: int = 20
NUM_PERM_LSH: int = BANDS * ROWS_PER_BAND  # 4080
SET_SIZE: int = 1000
THETA: float = 0.70
SEED: int = 42
CORE_J_LOW: float = 0.37
CORE_J_HIGH: float = 0.90


def _ensure_figures_dir() -> str:
    figures_dir = os.path.join(CURRENT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def _trial_count_for_jaccard(j: float, trials_core: int, trials_tail: int) -> int:
    if CORE_J_LOW <= j <= CORE_J_HIGH:
        return trials_core
    return trials_tail


def generate_interval_lists_with_jaccard(
    target_jaccard: float, set_size: int, start_id: int
) -> Tuple[List[int], List[int]]:
    """Generate two integer interval lists with approximately target Jaccard.

    Time Complexity: O(set_size)
    Space Complexity: O(set_size)
    """
    overlap = int(target_jaccard * 2 * set_size / (1 + target_jaccard))
    overlap = min(max(overlap, 0), set_size)
    offset = set_size - overlap
    seq_a = list(range(start_id, start_id + set_size))
    seq_b = list(range(start_id + offset, start_id + offset + set_size))
    return seq_a, seq_b


def simulate_collision_curve(
    j_values: np.ndarray, set_size: int, trials_core: int, trials_tail: int
) -> np.ndarray:
    """Estimate collision probability curve for k=4096 sketch and B=204,R=20 LSH.

    Time Complexity:
        O(sum(trials(J)) * (set_size + NUM_PERM_LSH))
    Space Complexity:
        O(NUM_PERM_LSH)
    """
    sketcher = FastSimilaritySketch(size=K_SKETCH, seed=SEED)
    rates = np.zeros_like(j_values, dtype=float)

    for idx, j in enumerate(j_values):
        trials = _trial_count_for_jaccard(float(j), trials_core, trials_tail)
        hits = 0
        print(f"[LSH][k=4096,b=204,r=20] J={j:.3f} ({idx+1}/{len(j_values)}), trials={trials}")
        for trial_id in range(trials):
            start_id = trial_id * 10000 + idx * 1000000
            seq_a, seq_b = generate_interval_lists_with_jaccard(float(j), set_size, start_id)
            digest_a = sketcher(seq_a)[:NUM_PERM_LSH]
            digest_b = sketcher(seq_b)[:NUM_PERM_LSH]

            lsh = LSH(num_perm=NUM_PERM_LSH, num_bands=BANDS)
            lsh.insert([digest_a])
            candidates = lsh.query(digest_b)
            if 0 in candidates:
                hits += 1
        rates[idx] = hits / trials
        print(f"  -> collision rate {rates[idx]:.4f}")
    return rates


def save_outputs(j_values: np.ndarray, collision_probs: np.ndarray) -> Tuple[str, str]:
    """Save one figure and one CSV for the curve."""
    figures_dir = _ensure_figures_dir()
    png_path = os.path.join(figures_dir, "lsh_collision_cpp_k4096_b204_r20.png")
    csv_path = os.path.join(figures_dir, "lsh_collision_cpp_k4096_b204_r20.csv")

    x_dense = np.linspace(0.0, 1.0, 1000)
    theory = 1.0 - (1.0 - x_dense ** ROWS_PER_BAND) ** BANDS

    plt.figure(figsize=(12, 8))
    plt.plot(
        x_dense,
        theory,
        linestyle="--",
        linewidth=2.5,
        label=f"MinHash LSH Theory (B={BANDS}, R={ROWS_PER_BAND}, BR={NUM_PERM_LSH})",
    )
    plt.plot(
        j_values,
        collision_probs,
        "o-",
        linewidth=2.0,
        markersize=5,
        label=f"FastSketch C++ Simulated (k={K_SKETCH}, use first {NUM_PERM_LSH} dims)",
    )
    plt.axvline(THETA, color="grey", linestyle="--", linewidth=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Jaccard(A,B) = J")
    plt.ylabel("P(A and B collide in >=1 band)")
    plt.title("LSH Collision Probability (k=4096, B=204, R=20)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["jaccard", "theory_b204_r20", "sim_k4096_use4080"])
        for j, sim in zip(j_values, collision_probs):
            theory_j = 1.0 - (1.0 - float(j) ** ROWS_PER_BAND) ** BANDS
            writer.writerow([f"{float(j):.8f}", f"{theory_j:.8f}", f"{float(sim):.8f}"])

    return png_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LSH collision curve for k=4096, B=204, R=20."
    )
    parser.add_argument("--trials-core", type=int, default=120)
    parser.add_argument("--trials-tail", type=int, default=30)
    parser.add_argument("--num-j", type=int, default=50)
    args = parser.parse_args()

    j_values = np.linspace(0.02, 0.99, args.num_j)
    print("Experiment: k=4096 sketch, B=204, R=20 LSH (use first 4080 dims)")
    print(f"Set size={SET_SIZE}, trials_core={args.trials_core}, trials_tail={args.trials_tail}")

    collision_probs = simulate_collision_curve(
        j_values=j_values,
        set_size=SET_SIZE,
        trials_core=args.trials_core,
        trials_tail=args.trials_tail,
    )
    png_path, csv_path = save_outputs(j_values, collision_probs)
    print("Saved:")
    print(f"- {png_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()
