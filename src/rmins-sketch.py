"""
rmin_sketch.py
---------------


Author: (your name)
Date: (today's date)
"""

import numpy as np
import mmh3
from typing import List, Iterable


def _permute_hash(h: int, a: int, b: int) -> np.uint32:
    """模拟排列哈希，计算(a * h + b) mod 2^64 的高32位"""
    return np.uint32(((a * h + b) % (2**64)) >> 32)


class RMinHashSketch:
    def __init__(self, num_perm: int = 128, random_seed: int = 42):
        self.num_perm = num_perm
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.perm_pairs = [
            (np.random.randint(1, 2**64) | 1, np.random.randint(0, 2**64))
            for _ in range(num_perm)
        ]
        self.hash_values = np.full(num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)

    def sketch(self, items: Iterable) -> List[int]:
        self.hash_values.fill(np.iinfo(np.uint32).max)
        for item in items:
            element_str = str(item).encode('utf-8')
            hash_val = mmh3.hash64(element_str, signed=False)[0]
            for j in range(self.num_perm):
                a, b = self.perm_pairs[j]
                self.hash_values[j] = np.minimum(self.hash_values[j], _permute_hash(hash_val, a, b))
        return self.hash_values.tolist()
    # Cmin实现，Rmin和CMin区别


