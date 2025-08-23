"""
RMinHash Sketch Implementation using Rensa Package

This module provides a wrapper around the high-performance Rensa RMinHash implementation
to match the interface of the existing sketch classes. Rensa is a Rust-based MinHash
library with Python bindings that offers significant performance improvements over
traditional implementations.

The wrapper maintains compatibility with the simple `sketch(items)` interface while
leveraging Rensa's optimized R-MinHash algorithm for fast similarity estimation
and deduplication.
"""

from typing import List, Iterable
from rensa import RMinHash


class RMinHashSketch:
    """
    R-MinHash sketch implementation using the high-performance Rensa package.

    This class wraps Rensa's `RMinHash` to keep a minimal, consistent interface:
    initialize with `num_perm` and `seed`, and call `sketch(items)` to obtain the
    signature. Items are converted to strings before hashing.

    Time Complexity: O(n * k) where n is number of items, k is number of permutations
    Space Complexity: O(k) for storing the sketch signature
    """

    def __init__(self, num_perm: int = 128, seed: int = 42):
        """
        Initialize the RMinHashSketch with specified parameters.

        Args:
            num_perm: Number of permutations (hash functions) to use. Higher values
                      improve accuracy at higher computational cost. Default: 128.
            seed: Random seed for reproducible hash functions. Default: 42.
        """
        self.num_perm = num_perm
        self.seed = seed

    def sketch(self, items: Iterable) -> List[int]:
        """
        Generate an R-MinHash sketch from the given items.

        Args:
            items: Iterable of items to sketch. Items will be converted to strings.

        Returns:
            List[int]: The R-MinHash signature as a list of integers of length `num_perm`.

        Time Complexity: O(n * k) where n = len(items), k = num_perm
        Space Complexity: O(k) for the output signature
        """
        hasher = RMinHash(num_perm=self.num_perm, seed=self.seed)

        item_list = [str(item) for item in items]
        if item_list:
            hasher.update(item_list)

        signature = hasher.digest()
        return [int(val) for val in signature]
