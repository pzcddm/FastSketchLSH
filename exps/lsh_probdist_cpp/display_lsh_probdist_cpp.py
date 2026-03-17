"""
display_lsh_probdist_cpp.py
---------------------------
Author: Zhencan Peng
Date: 2026-02-14

This experiment reproduces the probability-curve visualization style used in
`prototype/simulation/display_lsh_probdist.py`, but it uses the C++ extension
module `FastSketchLSH` instead of the pure-Python prototype implementation.

Goals:
1) Plot LSH collision probability curves for FastSketch under two sketch sizes:
   - k = 256
   - k = 4096
   and compare both against the theoretical MinHash LSH curve:
   P(candidate) = 1 - (1 - J^r)^b, where k = b * r.
2) Plot acceptance probability curves:
   Pr[estimate >= theta] for FastSketch simulation vs k-mins theoretical
   acceptance for the same k values.

Outputs:
- exps/lsh_probdist_cpp/figures/lsh_collision_cpp_k256_k4096.png
- exps/lsh_probdist_cpp/figures/kmins_vs_fastsketch_cpp_k256_k4096.png

Design notes:
- This script keeps parameter naming and plotting style close to the prototype
  script for easy side-by-side comparison.
- To keep runtime practical for large k=4096, trial counts are configurable
  with command-line arguments.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is importable when running this script directly.
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from FastSketchLSH import FastSimilaritySketch, LSH  # type: ignore


# Plot style aligned with prototype visualization.
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


# Default experiment parameters (mirroring the prototype style).
THETA: float = 0.70
NUM_BANDS: int = 16
SKETCH_SIZES: Tuple[int, int] = (256, 4096)
SET_SIZE_LSH: int = 1000
SET_SIZE_PROBDIST: int = 10000
JACCARD_VALUES: np.ndarray = np.linspace(0.02, 0.99, 50)
RANDOM_SEED_LSH: int = 42
RANDOM_SEED_PROBDIST: int = 52

# Prototype-like trial policy can be expensive at k=4096; keep configurable.
DEFAULT_TRIALS_CORE: int = 200
DEFAULT_TRIALS_TAIL: int = 40
CORE_J_LOW: float = 0.37
CORE_J_HIGH: float = 0.90

# Collision-curve profiles.
# For k=4096, we intentionally evaluate two LSH layouts and truncate digests to BR.
COLLISION_PROFILES: Tuple[Dict[str, int], ...] = (
    {"sketch_size": 256, "num_bands": 16, "rows_per_band": 16},    # BR=256
    {"sketch_size": 4096, "num_bands": 204, "rows_per_band": 20},  # BR=4080
    {"sketch_size": 4096, "num_bands": 163, "rows_per_band": 25},  # BR=4075
)


@dataclass(frozen=True)
class SimConfig:
    """Configuration bundle for simulation settings."""

    theta: float
    num_bands: int
    sketch_sizes: Tuple[int, int]
    set_size_lsh: int
    set_size_probdist: int
    jaccard_values: np.ndarray
    seed_lsh: int
    seed_probdist: int
    trials_core: int
    trials_tail: int


def _ensure_figures_dir() -> str:
    """Create and return the figures directory path for this experiment."""
    figures_dir = os.path.join(CURRENT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def _trial_count_for_jaccard(j: float, trials_core: int, trials_tail: int) -> int:
    """Return trial count with a prototype-like dense middle region."""
    if CORE_J_LOW <= j <= CORE_J_HIGH:
        return trials_core
    return trials_tail


def generate_interval_lists_with_jaccard(
    target_jaccard: float, set_size: int, start_id: int
) -> Tuple[List[int], List[int], float]:
    """
    Generate two integer lists that correspond to interval sets with target Jaccard.

    The generated lists are contiguous ranges so that set overlap is controlled by
    a single offset.

    Time Complexity:
        O(set_size)
    Space Complexity:
        O(set_size)
    """
    overlap = int(target_jaccard * 2 * set_size / (1 + target_jaccard))
    overlap = min(max(overlap, 0), set_size)
    offset = set_size - overlap

    seq_a = list(range(start_id, start_id + set_size))
    seq_b = list(range(start_id + offset, start_id + offset + set_size))
    actual = overlap / (2 * set_size - overlap) if set_size > 0 else 0.0
    return seq_a, seq_b, actual


def estimate_jaccard_from_sketch(sketch_a: Sequence[int], sketch_b: Sequence[int]) -> float:
    """
    Estimate Jaccard as the coordinate-wise match ratio between two digests.

    Time Complexity:
        O(k)
    Space Complexity:
        O(k) due to temporary arrays.
    """
    arr_a = np.asarray(sketch_a, dtype=np.uint64)
    arr_b = np.asarray(sketch_b, dtype=np.uint64)
    return float(np.mean(arr_a == arr_b))


def simulate_fastsketch_lsh_curve(
    sketch_size: int,
    lsh_num_perm: int,
    num_bands: int,
    rows_per_band: int,
    j_values: np.ndarray,
    set_size: int,
    seed: int,
    trials_core: int, trials_tail: int
) -> np.ndarray:
    """
    Simulate LSH collision probability using C++ extension FastSketch + LSH.

    For each J, this estimates:
        Pr[two sets collide in at least one band]
    """
    sketcher = FastSimilaritySketch(size=sketch_size, seed=seed)
    rates = np.zeros_like(j_values, dtype=float)

    for idx, j in enumerate(j_values):
        trials = _trial_count_for_jaccard(float(j), trials_core, trials_tail)
        hits = 0
        print(
            f"[LSH][k={sketch_size}, b={num_bands}, r={rows_per_band}, "
            f"use_perm={lsh_num_perm}] J={j:.3f} ({idx + 1}/{len(j_values)}), "
            f"trials={trials}"
        )
        for trial_id in range(trials):
            start_id = trial_id * 10000 + idx * 1000000
            seq_a, seq_b, _ = generate_interval_lists_with_jaccard(
                target_jaccard=float(j),
                set_size=set_size,
                start_id=start_id,
            )
            digest_a = sketcher(seq_a)[:lsh_num_perm]
            digest_b = sketcher(seq_b)[:lsh_num_perm]

            lsh = LSH(num_perm=lsh_num_perm, num_bands=num_bands)
            lsh.insert([digest_a])
            candidates = lsh.query(digest_b)
            if 0 in candidates:
                hits += 1
        rates[idx] = hits / trials
        print(f"  -> collision rate {rates[idx]:.4f}")
    return rates


def kmins_accept_prob_curve(j_values: np.ndarray, k: int, theta: float) -> np.ndarray:
    """
    Compute theoretical k-mins acceptance probability:
        Pr[Binomial(k, J) >= ceil(theta * k)].

    Time Complexity:
        O(k * len(j_values))
    Space Complexity:
        O(k)
    """
    imin = int(math.ceil(theta * k))
    y = np.zeros_like(j_values, dtype=float)
    log_k_fact = math.lgamma(k + 1.0)

    for j_idx, p in enumerate(j_values):
        if p <= 0.0:
            y[j_idx] = 0.0
            continue
        if p >= 1.0:
            y[j_idx] = 1.0 if imin <= k else 0.0
            continue

        log_p = math.log(float(p))
        log_one_minus_p = math.log1p(-float(p))
        log_terms = np.empty(k - imin + 1, dtype=float)

        for arr_idx, i in enumerate(range(imin, k + 1)):
            log_choose = log_k_fact - math.lgamma(i + 1.0) - math.lgamma(k - i + 1.0)
            log_terms[arr_idx] = log_choose + i * log_p + (k - i) * log_one_minus_p

        max_log = float(np.max(log_terms))
        y[j_idx] = float(np.exp(max_log) * np.sum(np.exp(log_terms - max_log)))

    return y


def simulate_fastsketch_accept_curve(
    sketch_size: int, j_values: np.ndarray, theta: float, set_size: int, seed: int,
    trials_core: int, trials_tail: int
) -> np.ndarray:
    """
    Simulate FastSketch acceptance probability:
        Pr[estimate_jaccard(sketch(A), sketch(B)) >= theta].
    """
    sketcher = FastSimilaritySketch(size=sketch_size, seed=seed)
    probs = np.zeros_like(j_values, dtype=float)

    for idx, j in enumerate(j_values):
        trials = _trial_count_for_jaccard(float(j), trials_core, trials_tail)
        accepted = 0
        print(
            f"[Prob][k={sketch_size}] J={j:.3f} ({idx + 1}/{len(j_values)}), "
            f"trials={trials}"
        )
        for trial_id in range(trials):
            start_id = trial_id * 10000 + idx * 1000000
            seq_a, seq_b, _ = generate_interval_lists_with_jaccard(
                target_jaccard=float(j),
                set_size=set_size,
                start_id=start_id,
            )
            digest_a = sketcher(seq_a)
            digest_b = sketcher(seq_b)
            est = estimate_jaccard_from_sketch(digest_a, digest_b)
            if est >= theta:
                accepted += 1
        probs[idx] = accepted / trials
        print(f"  -> accept prob {probs[idx]:.4f}")
    return probs


def plot_lsh_collision_curves(
    j_values: np.ndarray,
    simulated_by_profile: Dict[str, np.ndarray],
) -> str:
    """Plot and save theoretical vs simulated LSH collision curves for both k."""
    figures_dir = _ensure_figures_dir()
    out_path = os.path.join(figures_dir, "lsh_collision_cpp_k256_k4096.png")
    x_dense = np.linspace(0.0, 1.0, 1000)

    plt.figure(figsize=(12, 8))
    for profile in COLLISION_PROFILES:
        k = profile["sketch_size"]
        num_bands = profile["num_bands"]
        rows_per_band = profile["rows_per_band"]
        lsh_num_perm = num_bands * rows_per_band
        profile_key = f"k{k}_b{num_bands}_r{rows_per_band}"
        y_sim = simulated_by_profile[profile_key]
        y_theory = 1.0 - (1.0 - x_dense ** rows_per_band) ** num_bands
        plt.plot(
            x_dense,
            y_theory,
            linewidth=2.5,
            linestyle="--",
            label=(
                f"MinHash LSH Theory (k={k}, b={num_bands}, r={rows_per_band}, "
                f"BR={lsh_num_perm})"
            ),
        )
        plt.plot(
            j_values,
            y_sim,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=f"FastSketch C++ Simulated (k={k})",
        )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Jaccard(A,B) = J")
    plt.ylabel("P(A and B collide in >=1 band)")
    plt.title("LSH Collision Probability: Theory vs FastSketch C++")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def save_lsh_collision_csv(
    j_values: np.ndarray,
    simulated_by_profile: Dict[str, np.ndarray],
) -> str:
    """Save LSH collision curves (theory + simulation) into CSV."""
    figures_dir = _ensure_figures_dir()
    out_path = os.path.join(figures_dir, "lsh_collision_cpp_k256_k4096.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["jaccard"]
        for profile in COLLISION_PROFILES:
            k = profile["sketch_size"]
            num_bands = profile["num_bands"]
            rows_per_band = profile["rows_per_band"]
            lsh_num_perm = num_bands * rows_per_band
            header.append(
                f"theory_k{k}_b{num_bands}_r{rows_per_band}_br{lsh_num_perm}"
            )
            header.append(f"sim_k{k}")
        writer.writerow(header)

        for idx, j in enumerate(j_values):
            row = [f"{float(j):.8f}"]
            for profile in COLLISION_PROFILES:
                k = profile["sketch_size"]
                num_bands = profile["num_bands"]
                rows_per_band = profile["rows_per_band"]
                profile_key = f"k{k}_b{num_bands}_r{rows_per_band}"
                theory = 1.0 - (1.0 - float(j) ** rows_per_band) ** num_bands
                row.append(f"{theory:.8f}")
                row.append(f"{float(simulated_by_profile[profile_key][idx]):.8f}")
            writer.writerow(row)

    print(f"Saved: {out_path}")
    return out_path


def plot_acceptance_curves(
    j_values: np.ndarray,
    simulated_by_k: Dict[int, np.ndarray],
    theta: float,
) -> str:
    """Plot and save theoretical vs simulated acceptance probability curves."""
    figures_dir = _ensure_figures_dir()
    out_path = os.path.join(figures_dir, "kmins_vs_fastsketch_cpp_k256_k4096.png")

    plt.figure(figsize=(12, 8))
    for k, y_sim in simulated_by_k.items():
        y_theory = kmins_accept_prob_curve(j_values, k, theta)
        plt.plot(
            j_values,
            y_theory,
            linewidth=2.5,
            linestyle="--",
            label=f"k-mins Theory (k={k}, theta={theta:.2f})",
        )
        plt.plot(
            j_values,
            y_sim,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=f"FastSketch C++ Simulated (k={k}, theta={theta:.2f})",
        )

    plt.axvline(theta, color="grey", linestyle="--", linewidth=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Jaccard(A,B) = J")
    plt.ylabel("Probability")
    plt.title("Acceptance Probability: k-mins Theory vs FastSketch C++")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def save_acceptance_csv(
    j_values: np.ndarray,
    simulated_by_k: Dict[int, np.ndarray],
    theta: float,
) -> str:
    """Save acceptance curves (theory + simulation) into CSV."""
    figures_dir = _ensure_figures_dir()
    out_path = os.path.join(figures_dir, "kmins_vs_fastsketch_cpp_k256_k4096.csv")
    ks = list(simulated_by_k.keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["jaccard"]
        for k in ks:
            header.append(f"kmins_theory_k{k}_theta{theta:.2f}")
            header.append(f"fastsketch_sim_k{k}_theta{theta:.2f}")
        writer.writerow(header)

        theory_by_k = {k: kmins_accept_prob_curve(j_values, k, theta) for k in ks}
        for idx, j in enumerate(j_values):
            row = [f"{float(j):.8f}"]
            for k in ks:
                row.append(f"{float(theory_by_k[k][idx]):.8f}")
                row.append(f"{float(simulated_by_k[k][idx]):.8f}")
            writer.writerow(row)

    print(f"Saved: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot LSH/probability distribution curves using FastSketchLSH C++ "
            "extension for k=256 and k=4096."
        )
    )
    parser.add_argument(
        "--trials-core",
        type=int,
        default=DEFAULT_TRIALS_CORE,
        help="Trials for J in [0.37, 0.90].",
    )
    parser.add_argument(
        "--trials-tail",
        type=int,
        default=DEFAULT_TRIALS_TAIL,
        help="Trials for J outside [0.37, 0.90].",
    )
    parser.add_argument(
        "--num-j",
        type=int,
        default=len(JACCARD_VALUES),
        help="Number of Jaccard points in [0.02, 0.99].",
    )
    return parser.parse_args()


def main() -> None:
    """Run simulations and generate both figures."""
    args = parse_args()
    config = SimConfig(
        theta=THETA,
        num_bands=NUM_BANDS,
        sketch_sizes=SKETCH_SIZES,
        set_size_lsh=SET_SIZE_LSH,
        set_size_probdist=SET_SIZE_PROBDIST,
        jaccard_values=np.linspace(0.02, 0.99, args.num_j),
        seed_lsh=RANDOM_SEED_LSH,
        seed_probdist=RANDOM_SEED_PROBDIST,
        trials_core=args.trials_core,
        trials_tail=args.trials_tail,
    )

    print("FastSketch C++ probability-distribution experiment")
    print(f"Sketch sizes: {config.sketch_sizes}")
    print("LSH collision profiles:")
    for profile in COLLISION_PROFILES:
        k = profile["sketch_size"]
        b = profile["num_bands"]
        r = profile["rows_per_band"]
        print(f"  - k={k}: b={b}, r={r}, BR={b * r}")
    print(f"Theta: {config.theta:.2f}")
    print(
        f"Set sizes: LSH={config.set_size_lsh}, ProbDist={config.set_size_probdist}"
    )
    print(
        "Trials: core="
        f"{config.trials_core} (J in [{CORE_J_LOW}, {CORE_J_HIGH}]), "
        f"tail={config.trials_tail}"
    )
    print("=" * 64)

    collision_by_profile: Dict[str, np.ndarray] = {}
    accept_by_k: Dict[int, np.ndarray] = {}

    for profile in COLLISION_PROFILES:
        k = profile["sketch_size"]
        num_bands = profile["num_bands"]
        rows_per_band = profile["rows_per_band"]
        lsh_num_perm = num_bands * rows_per_band
        if lsh_num_perm > k:
            raise ValueError(
                f"LSH BR={lsh_num_perm} exceeds sketch size k={k}."
            )
        profile_key = f"k{k}_b{num_bands}_r{rows_per_band}"
        collision_by_profile[profile_key] = simulate_fastsketch_lsh_curve(
            sketch_size=k,
            lsh_num_perm=lsh_num_perm,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
            j_values=config.jaccard_values,
            set_size=config.set_size_lsh,
            seed=config.seed_lsh,
            trials_core=config.trials_core,
            trials_tail=config.trials_tail,
        )

    for k in config.sketch_sizes:
        accept_by_k[k] = simulate_fastsketch_accept_curve(
            sketch_size=k,
            j_values=config.jaccard_values,
            theta=config.theta,
            set_size=config.set_size_probdist,
            seed=config.seed_probdist,
            trials_core=config.trials_core,
            trials_tail=config.trials_tail,
        )

    path_lsh = plot_lsh_collision_curves(
        j_values=config.jaccard_values,
        simulated_by_profile=collision_by_profile,
    )
    path_lsh_csv = save_lsh_collision_csv(
        j_values=config.jaccard_values,
        simulated_by_profile=collision_by_profile,
    )
    path_prob = plot_acceptance_curves(
        j_values=config.jaccard_values,
        simulated_by_k=accept_by_k,
        theta=config.theta,
    )
    path_prob_csv = save_acceptance_csv(
        j_values=config.jaccard_values,
        simulated_by_k=accept_by_k,
        theta=config.theta,
    )

    print("\nDone.")
    print(f"- {path_lsh}")
    print(f"- {path_lsh_csv}")
    print(f"- {path_prob}")
    print(f"- {path_prob_csv}")


if __name__ == "__main__":
    main()
