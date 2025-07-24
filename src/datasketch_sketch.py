"""
datasketch_sketch.py
--------------------
Implementation of MinHash sketching using the datasketch library for Jaccard similarity estimation.

Provides the DatasketchMinHashSketch class for generating minhash signatures using datasketch.MinHash.

Author: (your name)
Date: (today's date)
"""
from datasketch import MinHash
from typing import Iterable, List

class DatasketchMinHashSketch:
    """
    Implements MinHash sketching using the datasketch.MinHash class for estimating Jaccard similarity.
    Uses a fixed random seed for reproducibility.
    
    Time Complexity: O(k * |A|) where k is the number of hash functions and |A| is the set size
    Space Complexity: O(k)
    """
    def __init__(self, num_perm: int = 128, random_seed: int = 42):
        if not isinstance(num_perm, int) or num_perm <= 0:
            raise ValueError("num_perm must be a positive integer.")
        self.num_perm = num_perm
        self.random_seed = random_seed

    def sketch(self, A: Iterable) -> List[int]:
        """
        Generate a MinHash signature for the input set A.
        Args:
            A (Iterable): The input set (any iterable of hashable items).
        Returns:
            List[int]: The MinHash signature (list of integers).
        """
        m = MinHash(num_perm=self.num_perm, seed=self.random_seed)
        for a in A:
            m.update(str(a).encode('utf-8'))
        return list(m.hashvalues) 