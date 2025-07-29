"""
fast_sketch.py
--------------
Implementation of the Fast Similarity Sketch algorithm (from 'Fast Similarity Sketching', arXiv:1704.04370v4).

Optimized with NumPy operations for better performance.

Author: (your name)
Date: 2025-07-29
"""
import mmh3
import numpy as np
from typing import Iterable, List


class FastSimilaritySketch:
    """
    NumPy-optimized implementation of Algorithm 1 from "Fast Similarity Sketching" paper.
    Uses vectorized operations where possible and handles integer overflow properly.

    Time Complexity: O(t * |A|) where t is sketch size and |A| is set size
    Space Complexity: O(t)
    """

    def __init__(self, sketch_size: int, random_seed: int = 42):
        if not isinstance(sketch_size, int) or sketch_size <= 0:
            raise ValueError("Sketch size (t) must be a positive integer.")
        self.t = sketch_size
        np.random.seed(random_seed)
        # Use uint64 to avoid overflow in hash operations
        self.hash_seeds = np.random.randint(0, 2 ** 32, size=2 * self.t, dtype=np.uint64)

    def sketch(self, A: Iterable) -> np.ndarray:
        # Initialize sketch with maximum possible values
        S = np.full((self.t, 2), np.inf, dtype=np.float64)
        c = 0
        filled_bins = np.zeros(self.t, dtype=bool)

        for i, seed in enumerate(self.hash_seeds):
            current_seed = int(seed)  # Convert to Python int for mmh3

            for a in A:
                element_str = str(a).encode('utf-8')
                # Use uint64 to handle large hash values
                hash_val = np.uint64(mmh3.hash64(element_str, seed=current_seed, signed=False)[0])

                # Vectorized bin selection and comparison
                b = hash_val % self.t if i < self.t else i - self.t

                # Update sketch if current hash is smaller
                if (i < S[b, 0]) or (i == S[b, 0] and hash_val < S[b, 1]):
                    S[b, 0] = i
                    S[b, 1] = hash_val
                    if not filled_bins[b]:
                        filled_bins[b] = True
                        c += 1
            if c == self.t:
                break

        final_sketch = [val for round_idx, val in S]
        return final_sketch