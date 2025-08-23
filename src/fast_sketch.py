"""
fast_sketch.py
--------------
Implementation of the Fast Similarity Sketch algorithm (from 'Fast Similarity Sketching', arXiv:1704.04370v4).

Provides the FastSimilaritySketch class for generating similarity sketches in pure python.

Author: Zhencan Peng
Date: 2025-08-23
"""
import mmh3
import numpy as np
from typing import Iterable, List

class FastSimilaritySketch:
    """
    Modified to use fixed random seeds for reproducibility.
    
    Time Complexity: O(|A| + klogk) where k is sketch size and |A| is set size
    Space Complexity: O(k)
    """
    def __init__(self, sketch_size: int, random_seed: int = 42):
        if not isinstance(sketch_size, int) or sketch_size <= 0:
            raise ValueError("Sketch size (k) must be a positive integer.")
        if sketch_size > 1<<12:
            raise ValueError("Sketch size (k) must be less than or equal to 4096.")
        
        self.k = sketch_size
        # Use fixed random seed for reproducible hash seeds
        np.random.seed(random_seed)
        # Use dtype=np.int64 to avoid 32-bit integer overflow
        self.hash_seeds = np.random.randint(0, 2**32, size=2 * self.k, dtype=np.int64)
        self.masked_value = (1 << 52) - 1
        
    def sketch(self, A: Iterable) -> List[int]:
        encoded = [str(a).encode('utf-8') for a in A]
        
        S = [(1 << 64) - 1] * self.k
        cnt = 0
        filled_bins = [False] * self.k
        
        for i, seed_np in enumerate(self.hash_seeds):
            # Convert numpy.int64 to Python int for mmh3 compatibility
            current_seed = int(seed_np)

            for element_str in encoded:
                # We encode a here since there will be 2k*|A| calls to mmh3
                # However, actually at most time, because we only enumerate one round
                # To have a better space locality, we encode a here
                hash_val = mmh3.hash64(element_str, seed=current_seed, signed=False)[0]
                
                bin_idx = hash_val % self.k if i < self.k else i - self.k
                
                # Get 52-bit hash using bit mixing to reduce collision patterns
                # XOR higher and lower bits to distribute entropy better
                hash_52bit = ((hash_val >> 12) ^ hash_val) & self.masked_value
                
                # Combine i (round index) and hash_val into a single 64-bit value
                # Top 12 bits: round index i, Bottom 52 bits: hash value
                # This ensures later rounds always produce larger values than earlier rounds
                v = (i << 52) | hash_52bit
                if v < S[bin_idx]:
                    S[bin_idx] = v
                    if not filled_bins[bin_idx]:
                        filled_bins[bin_idx] = True
                        cnt += 1
            if cnt == self.k:
                break
                
        final_sketch = list(S)
        return final_sketch 