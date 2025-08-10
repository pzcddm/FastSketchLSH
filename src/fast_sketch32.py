"""
fast_sketch32.py
----------------
32-bit variant of the Fast Similarity Sketch algorithm (from 'Fast Similarity Sketching', arXiv:1704.04370v4).

This implementation uses 32-bit hash values where:
- Top 8 bits select the bin index (mapped to t if t < 256)
- Lower 24 bits serve as the intra-bin priority (min-heap key)

Default sketch size t is 128. This layout is efficient for t ≤ 256.

Author: (your name)
Date: (today's date)
"""
import mmh3
import numpy as np
from typing import Iterable, List


class FastSimilaritySketch32Bit:
    """
    32-bit sketch where bin index is derived from the top 8 bits and the value
    to minimize within each bin is the lower 24 bits.

    - Time Complexity: O(|A| * 2t) worst-case; typically O(|A| + t) due to early stop
    - Space Complexity: O(t)

    Notes:
    - Only supports t ≤ 256 because 8 bits are used for the bin selection.
    - Default t is 128.
    """

    BIN_BITS: int = 8
    VALUE_BITS: int = 32 - BIN_BITS

    def __init__(self, sketch_size: int = 128, random_seed: int = 42):
        if not isinstance(sketch_size, int) or sketch_size <= 0:
            raise ValueError("Sketch size (t) must be a positive integer.")
        if sketch_size > (1 << self.BIN_BITS):
            raise ValueError("Sketch size (t) must be ≤ 256 when using 8-bit bin indices.")

        self.t = sketch_size

        # Precompute masks
        self.bin_mask: int = (1 << self.BIN_BITS) - 1  # 0xFF
        self.value_mask: int = (1 << self.VALUE_BITS) - 1  # 0xFFFFFF

        # Precompute the initial sentinel value for empty bins (32-bit all ones)
        self.initial_value_32bit: int = (1 << 32) - 1  # 0xFFFFFFFF

        # Use fixed random seed for reproducible hash seeds
        np.random.seed(random_seed)
        # mmh3 expects a 32-bit signed seed; use [0, 2**31) to stay in-range
        self.hash_seeds = np.random.randint(0, 2**31, size=2 * self.t, dtype=np.int32)

        # Pre-allocate working buffers to avoid per-call allocations
        self.S: List[int] = [self.initial_value_32bit] * self.t
        self.filled_bins: List[bool] = [False] * self.t

    def reset(self) -> None:
        """
        Reset internal buffers in-place without reallocating.

        This avoids per-call memory allocation in sketch().
        Time: O(t), Space: O(1) additional.
        """
        for i in range(self.t):
            self.S[i] = self.initial_value_32bit
            self.filled_bins[i] = False

    def sketch(self, A: Iterable) -> List[int]:
        encoded = [str(a).encode('utf-8') for a in A]

        # Reset working buffers in-place
        self.reset()

        cnt = 0

        for seed_np in self.hash_seeds:
            current_seed = int(seed_np)

            for element_str in encoded:
                # 32-bit mmh3 hash (unsigned)
                h32 = mmh3.hash(element_str, seed=current_seed, signed=False)

                # Top 8 bits choose the bin (map to [0, t-1] if t < 256)
                bin_byte = (h32 >> self.VALUE_BITS) & self.bin_mask
                bin_idx = bin_byte % self.t

                # Lower 24 bits are the priority value
                v24 = h32 & self.value_mask

                # Compare by the 24-bit priority only
                if v24 < (self.S[bin_idx] & self.value_mask):
                    # Store the 32-bit composed value (bin in top 8, value in lower 24)
                    self.S[bin_idx] = (bin_byte << self.VALUE_BITS) | v24
                    if not self.filled_bins[bin_idx]:
                        self.filled_bins[bin_idx] = True
                        cnt += 1
            if cnt == self.t:
                break

        # Return a copy to avoid aliasing across successive calls
        return list(self.S)