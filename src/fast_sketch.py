"""
fast_sketch.py
--------------
Implementation of the Fast Similarity Sketch algorithm (from 'Fast Similarity Sketching', arXiv:1704.04370v4).

Provides the FastSimilaritySketch class for generating similarity sketches.

Author: (your name)
Date: (today's date)
"""
import mmh3
import numpy as np
from typing import Iterable, List

class FastSimilaritySketch:
    """
    Implementation of Algorithm 1 from "Fast Similarity Sketching" paper (1704.04370v4).
    Modified to use fixed random seeds for reproducibility.
    
    Time Complexity: O(|A| + tlogt) where t is sketch size and |A| is set size
    Space Complexity: O(t)
    """
    def __init__(self, sketch_size: int, random_seed: int = 42):
        if not isinstance(sketch_size, int) or sketch_size <= 0:
            raise ValueError("Sketch size (t) must be a positive integer.")
        if sketch_size > 1<<12:
            raise ValueError("Sketch size (t) must be less than or equal to 4096.")
        
        self.t = sketch_size
        # Use fixed random seed for reproducible hash seeds
        np.random.seed(random_seed)
        # Use dtype=np.int64 to avoid 32-bit integer overflow
        self.hash_seeds = np.random.randint(0, 2**32, size=2 * self.t, dtype=np.int64)
        self.masked_value = (1 << 52) - 1
        
    def sketch(self, A: Iterable) -> List[int]:
        S = [float('inf')] * self.t
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
                
                # Combine i (round index) and hash_val into a single 64-bit value
                # Top 12 bits: round index i, Bottom 52 bits: hash value
                # This ensures later rounds always produce larger values than earlier rounds
                v = (i << 52) | (hash_val & self.masked_value)
                if v < S[b]:
                    S[b] = v
                    if not filled_bins[b]:
                        filled_bins[b] = True
                        c += 1
            if c == self.t:
                break
                
        final_sketch = list(S)
        return final_sketch 