"""
CMinHash Sketch Implementation using Rensa Package

This module provides a wrapper around the high-performance Rensa CMinHash implementation
to match the interface of the existing CMinHashSketch class. Rensa is a Rust-based
MinHash library with Python bindings that offers significant performance improvements
over traditional implementations.

The wrapper maintains compatibility with the existing sketch interface while leveraging
Rensa's optimized C-MinHash algorithm for faster similarity estimation and deduplication.
"""

from typing import List, Iterable
from rensa import CMinHash


class CMinHashSketch:
    """
    C-MinHash sketch implementation using the high-performance Rensa package.
    
    This class provides a wrapper around Rensa's CMinHash to maintain interface
    compatibility with existing CMinHashSketch implementations while benefiting
    from Rensa's performance optimizations (reportedly 40x faster than datasketch).
    
    The C-MinHash algorithm combines ideas from traditional MinHash and offers
    improved theoretical guarantees for similarity estimation.
    
    Time Complexity: O(n * k) where n is number of items, k is number of permutations
    Space Complexity: O(k) for storing the sketch
    """
    
    def __init__(self, num_perm: int = 128, seed: int = 42):
        """
        Initialize the CMinHashSketch with specified parameters.
        
        Args:
            num_perm (int): Number of permutations (hash functions) to use.
                           Higher values provide better accuracy but increase computation.
                           Default: 128
            seed (int): Random seed for reproducible hash functions.
                       Default: 42
        """
        self.num_perm = num_perm
        self.seed = seed

    
    def sketch(self, items: Iterable) -> List[int]:
        """
        Generate a C-MinHash sketch from the given items.
        
        Args:
            items (Iterable): Collection of items to sketch. Items will be
                            converted to strings before hashing.
        
        Returns:
            List[int]: The C-MinHash signature as a list of hash values.
                      Length equals num_perm parameter.
        
        Time Complexity: O(n * k) where n = len(items), k = num_perm
        Space Complexity: O(k) for the output sketch
        
        Note:
            - Items are converted to strings using str() before hashing
            - The sketch can be used for Jaccard similarity estimation
            - Empty input returns a sketch of maximum values
        """
        # Create a new CMinHash instance for each sketch to ensure clean state
        # This matches the behavior of the original implementation
        sketch_hasher = CMinHash(num_perm=self.num_perm, seed=self.seed)
        
        # Convert all items to strings and create a list for rensa
        # Rensa's update method expects a list of items, not individual items
        item_list = [str(item) for item in items]
        
        # Update the hasher with all items at once
        if item_list:  # Only update if we have items
            sketch_hasher.update(item_list)
        
        # Get the digest (signature) 
        signature = sketch_hasher.digest()
        
        # Convert to list of integers to match the original interface
        return [int(val) for val in signature]
