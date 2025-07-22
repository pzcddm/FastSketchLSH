import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

'''
Test KMinSketch

This file implements the classic k-mins minhash algorithm for estimating the Jaccard similarity between two sets.

The algorithm computes k independent hash functions on all elements of the set and selects the minimum hash value from each.
The resulting signature (of length k) is used to estimate the similarity between two sets by comparing positions where the signatures match.

Time Complexity: O(k * n) where n is the number of elements in the set.
Space Complexity: O(k)

This approach provides an unbiased estimate of the Jaccard similarity.
'''

import random
import mmh3
from src.kmins_sketch import KMinSketch
from simulation.util import estimate_jaccard, actual_jaccard


if __name__ == '__main__':
    k = 256
    A = set(range(0, 1000))
    B = set(range(500, 1500))

    print(f"|A| = {len(A)}, |B| = {len(B)}")
    true_j = actual_jaccard(A, B)
    print(f"True Jaccard: {true_j:.4f}")

    sketcher = KMinSketch(k)
    print("Generating sketches...")
    S_A = sketcher.sketch(A)
    S_B = sketcher.sketch(B)
    print("Done.")

    est_j = estimate_jaccard(S_A, S_B)
    print(f"Estimated Jaccard: {est_j:.4f}")
    print(f"Error: {abs(true_j - est_j):.4f}") 