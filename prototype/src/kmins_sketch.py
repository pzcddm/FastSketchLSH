"""
kmins_sketch.py
---------------
Implementation of the classic k-mins minhash algorithm for Jaccard similarity estimation.

Provides the KMinSketch class for generating minhash sketches.

Author: (your name)
Date: (today's date)
"""
import mmh3
import random
from typing import Iterable, List

class KMinSketch:
    """
    Implements the classic k-mins minhash method for estimating Jaccard similarity.
    Modified to use fixed random seeds for reproducibility.
    
    Time Complexity: O(k * |A|) where k is sketch size and |A| is set size
    Space Complexity: O(k)
    """
    def __init__(self, k: int, random_seed: int = 42):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Sketch size k must be a positive integer.")
        self.k = k
        # Use fixed random seed for reproducible hash seeds
        random.seed(random_seed)
        self.hash_seeds = [random.getrandbits(31) for _ in range(k)]
        
    def sketch(self, A: Iterable) -> List[int]:
        signature = []
        for seed in self.hash_seeds:
            try:
                # Compute the minimum hash value for this hash function
                min_hash = min(mmh3.hash64(str(a).encode('utf-8'), seed=seed, signed=False)[0] for a in A)
            except ValueError:
                # In case A is empty, assign infinity as the minimum hash
                min_hash = float('inf')
            signature.append(min_hash)
        return signature 