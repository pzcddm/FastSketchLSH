"""
Test FastSketchLSH: Band-collision similarity ratio for sets with Jaccard=0.8

This test generates many random set pairs with real Jaccard similarity 0.8,
inserts one set of each pair into FastSketchLSH, and queries with the other.
It reports the fraction of pairs considered similar (i.e., at least one band collision).

K = 128, bands = 16, threshold = 0.8
"""
import numpy as np
import sys
import os

# Allow running as script or module
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fast_sketch_lsh import FastSketchLSH

# Parameters
K = 128
BANDS = 16
THRESHOLD = 0.8
N_PAIRS = 1000
UNIVERSE_SIZE = 10000
SET_SIZE = 1000
JACCARD_TARGET = 0.8

rng = np.random.default_rng(42)
lsh = FastSketchLSH(threshold=THRESHOLD, sketch_size=K, bands=BANDS)

similar_count = 0
for i in range(N_PAIRS):
    # Generate two sets with Jaccard similarity ~0.8
    # |A| = |B| = SET_SIZE, |A ∩ B| = intersection_size
    # |A ∪ B| = 2*SET_SIZE - intersection_size
    # J = intersection_size / (2*SET_SIZE - intersection_size)
    intersection_size = int(JACCARD_TARGET * 2 * SET_SIZE / (1 + JACCARD_TARGET))
    unique_size = SET_SIZE - intersection_size
    base = rng.choice(UNIVERSE_SIZE, size=intersection_size, replace=False)
    rest = rng.choice(list(set(range(UNIVERSE_SIZE)) - set(base)), size=2*unique_size, replace=False)
    A = set(base) | set(rest[:unique_size])
    B = set(base) | set(rest[unique_size:])
    # Check real Jaccard
    real_jaccard = len(A & B) / len(A | B)
    assert abs(real_jaccard - JACCARD_TARGET) < 0.02, f"Real Jaccard: {real_jaccard}"
    # Insert A, query with B
    lsh.insert(f"A_{i}", A)
    result = lsh.query(B)
    if f"A_{i}" in result:
        similar_count += 1

print(f"Fraction of pairs considered similar (band collision) at Jaccard=0.8: {similar_count / N_PAIRS:.3f}") 