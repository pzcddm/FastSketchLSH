"""
fast_sketch_old.py
--------------
Implementation of the Fast Similarity Sketch algorithm (from 'Fast Similarity Sketching', arXiv:1704.04370v4).

Provides the FastSimilaritySketch class for generating similarity sketches.

Author: (your name)
Date: (today's date)
"""
import mmh3
import numpy as np
from typing import Iterable, List

class FastSimilaritySketchOld:
    """
    Implementation of Algorithm 1 from "Fast Similarity Sketching" paper (1704.04370v4).
    Modified to use fixed random seeds for reproducibility.
    
    Time Complexity: O(|A| + tlogt) where t is sketch size and |A| is set size
    Space Complexity: O(t)
    """
    def __init__(self, sketch_size: int, random_seed: int = 42):
        if not isinstance(sketch_size, int) or sketch_size <= 0:
            raise ValueError("Sketch size (t) must be a positive integer.")
        self.t = sketch_size
        # Use fixed random seed for reproducible hash seeds
        np.random.seed(random_seed)
        # Use dtype=np.int64 to avoid 32-bit integer overflow
        self.hash_seeds = np.random.randint(0, 2**32, size=2 * self.t, dtype=np.int64)

    def sketch(self, A: Iterable) -> List[int]:
        S = [(float('inf'), float('inf'))] * self.t
        c = 0
        filled_bins = [False] * self.t
        
        for i, seed_np in enumerate(self.hash_seeds):
            # Convert numpy.int64 to Python int for mmh3 compatibility
            current_seed = int(seed_np)

            for a in A:
                # We encode a here since there will be 2t*|A| calls to mmh3
                # However, actually at most time, because we only enumerate one round
                # To have a better space locality, we encode a here
                element_str = str(a).encode('utf-8')
                hash_val = mmh3.hash64(element_str, seed=current_seed, signed=False)[0]
                
                b = hash_val % self.t if i < self.t else i - self.t
                v = (i, hash_val)

                # v is a pair of (round_idx, hash_val) because a better way it to concatenate them to one value
                # and that make the value from later round must be larger than the value from earlier round
                # which is aligned with the paper
                if v < S[b]:
                    S[b] = v
                    if not filled_bins[b]:
                        filled_bins[b] = True
                        c += 1
            if c == self.t:
                break
                
        final_sketch = [val for round_idx, val in S]
        return final_sketch 