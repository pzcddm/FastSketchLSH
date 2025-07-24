import numpy as np
import mmh3
from typing import List, Iterable


class CMinHashSketch:
    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.sigma_a = rng.randint(1, 2 ** 64, dtype=np.uint64) | 1
        self.sigma_b = rng.randint(0, 2 ** 64, dtype=np.uint64)
        self.pi_c = rng.randint(1, 2 ** 64, dtype=np.uint64) | 1
        self.pi_d = rng.randint(0, 2 ** 64, dtype=np.uint64)
        # π_c * k + π_d (k ∈ 0..num_perm-1)
        k_values = np.arange(num_perm, dtype=np.uint64)
        self.pi_precomputed = self.pi_c * k_values + self.pi_d
        self.hash_values = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)

    def sketch(self, items: Iterable) -> List[int]:
        self.hash_values.fill(np.iinfo(np.uint64).max)
        for item in items:
            element_str = str(item).encode('utf-8')
            # 1. compute σ(h) = a*h + b
            h = mmh3.hash64(element_str, signed=False)[0]
            sigma_h = self.sigma_a * np.uint64(h) + self.sigma_b
            # 2. compute  π(σ(h), k) = c*σ(h) + (c*k + d)
            pi_values = self.pi_c * sigma_h + self.pi_precomputed
            # 3. update signature
            self.hash_values = np.minimum(self.hash_values, pi_values)

        return (self.hash_values >> 32).astype(np.uint32).tolist()
    