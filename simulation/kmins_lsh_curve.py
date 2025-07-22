"""
kmins_lsh_curve.py
------------------
Plot the probability curve for classic LSH (banding), k-mins+threshold, and FastSketch LSH simulation as a function of true Jaccard similarity.

- LSH: Pr[at least one band matches] = 1 - (1 - x^r)^b
- k-mins: Pr[at least imin out of k matches] = sum_{i=imin}^k binom(k, i) x^i (1-x)^{k-i}
- FastSketch LSH: Simulated collision probability from fast_sketch_lsh_simulation.py

Saves the figure as figures/lsh_curve.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, gamma
import os

# === Parameters ===
k1 = 128  # sketch size for k-mins
threshold = 0.7
imin = int(np.ceil(threshold * k1))  # minimum matches for threshold

# LSH banding parameters: (b, r)
lsh_params = [
    (16, 8),
]

# === Functions ===
def lsh_prob(x, b, r):
    return 1 - (1 - x**r)**b

def kmins_prob(x, k, imin):
    # Pr[at least imin out of k matches]
    # Use scipy.special.comb for binomial coefficient
    # x can be a numpy array
    prob = np.zeros_like(x)
    for i in range(imin, k+1):
        prob += comb(k, i) * x**i * (1-x)**(k-i)
    return prob

def load_fastsketch_results():
    """Load FastSketch LSH simulation results if available."""
    results_file = "simulation/fast_sketch_lsh_results.npy"
    if os.path.exists(results_file):
        try:
            results = np.load(results_file, allow_pickle=True).item()
            return results['jaccard_values'], results['collision_probs']
        except Exception as e:
            print(f"Warning: Could not load FastSketch results: {e}")
            return None, None
    else:
        print(f"Warning: FastSketch results file not found: {results_file}")
        return None, None

# === Plotting ===
x = np.linspace(0, 1, 1000)
plt.figure(figsize=(16, 10), dpi=300)

# Plot LSH curves
colors = ['C0', 'C1', 'C2']
for idx, (b, r) in enumerate(lsh_params):
    y = lsh_prob(x, b, r)
    plt.plot(x, y, label=f"LSH Theoretical b={b}, r={r}", lw=3, color=colors[idx])

# Plot k-mins curve
y_kmins = kmins_prob(x, k1, imin)
plt.plot(x, y_kmins, label=f"k-mins Theoretical k={k1}", lw=3, color='C3')

# Plot FastSketch LSH simulation results
fastsketch_x, fastsketch_y = load_fastsketch_results()
if fastsketch_x is not None and fastsketch_y is not None:
    plt.plot(fastsketch_x, fastsketch_y, 'o-', label="FastSketch LSH Simulated b=16, r=8", 
             lw=2, markersize=4, color='C4', alpha=0.8)

# Vertical dashed line at threshold
t = threshold
plt.axvline(t, color='grey', linestyle='--', lw=2)

# Axis labels and limits
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("True Jaccard Similarity", fontsize=14, labelpad=16)
plt.ylabel("Pr[LSH Collision]", fontsize=14, labelpad=16)
plt.title("LSH Collision Probability: Theoretical vs Simulated", fontsize=16, pad=20)
plt.xticks(np.linspace(0, 1, 6), fontsize=12)
plt.yticks(np.linspace(0, 1, 6), fontsize=12)
plt.grid(True, which='both', axis='both', linestyle=':', linewidth=0.5)
plt.legend(loc='lower right', frameon=False, fontsize=11)
plt.tight_layout()

# Save to figures folder
import os
os.makedirs("simulation/figures", exist_ok=True)
plt.savefig("simulation/figures/lsh_curve.png")
plt.close() 