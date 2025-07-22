"""
fast_sketch_lsh.py
------------------
Locality Sensitive Hashing (LSH) using FastSimilaritySketch.

This module provides the FastSketchLSH class, which implements LSH for fast set similarity search using the FastSimilaritySketch algorithm (see 'Fast Similarity Sketching', arXiv:1704.04370v4).

- Each set is encoded as a FastSimilaritySketch (a vector of integers).
- The sketch is partitioned into bands; each band is hashed to a bucket.
- Items sharing a bucket in any band are candidate matches.
- Querying returns keys whose Jaccard similarity (estimated from sketches) exceeds a threshold.

API is inspired by datasketch.MinHashLSH: https://ekzhu.com/datasketch/lsh.html

Example usage:
    from fast_sketch import FastSimilaritySketch
    from fast_sketch_lsh import FastSketchLSH

    lsh = FastSketchLSH(threshold=0.5, sketch_size=128, bands=16)
    lsh.insert("doc1", set1)
    lsh.insert("doc2", set2)
    result = lsh.query(set3)
    print("Approximate neighbours with Jaccard similarity > 0.5", result)

Author: (your name)
Date: (today's date)
"""
from typing import Any, Dict, Set, List, Iterable, Optional
from collections import defaultdict
from src.fast_sketch import FastSimilaritySketch
import numpy as np

class FastSketchLSH:
    """
    Locality Sensitive Hashing for FastSimilaritySketch.

    Args:
        threshold (float): Jaccard similarity threshold for candidate filtering (0 < threshold < 1).
        sketch_size (int): Length of the sketch vector (must be divisible by bands).
        bands (int): Number of bands to split the sketch into.
        random_seed (int): Seed for reproducibility (default: 42).

    Methods:
        insert(key, set_): Insert a set with a unique key.
        query(set_): Return keys of sets with estimated Jaccard similarity >= threshold.
        remove(key): Remove a key from the index.
        clear(): Remove all keys.

    Time Complexity:
        Insert: O(sketch_size + bands)
        Query: O(sketch_size + candidate_count)
    """
    def __init__(self, threshold: float, sketch_size: int, bands: int, random_seed: int = 42):
        assert 0 < threshold < 1, "Threshold must be in (0, 1)."
        assert sketch_size % bands == 0, "sketch_size must be divisible by bands."
        self.threshold = threshold
        self.sketch_size = sketch_size
        self.bands = bands
        self.rows_per_band = sketch_size // bands
        self.random_seed = random_seed
        self._sketcher = FastSimilaritySketch(sketch_size, random_seed)
        # Buckets: list of dicts, one per band
        self._buckets: List[Dict[int, Set[Any]]] = [defaultdict(set) for _ in range(bands)]
        # Store key for removal only (no sketch needed for query)
        self._keys: Set[Any] = set()

    def _band_hash(self, band: np.ndarray) -> int:
        # Use numpy's built-in hash for speed
        return hash(band.tobytes())

    def insert(self, key: Any, set_: Iterable) -> None:
        """Insert a set with a unique key into the LSH index."""
        sketch = self._sketcher.sketch(set_)
        arr = np.array(sketch, dtype=np.uint64)
        for b in range(self.bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band = arr[start:end]
            h = self._band_hash(band)
            self._buckets[b][h].add(key)
        self._keys.add(key)

    def query(self, set_: Iterable) -> List[Any]:
        """Return keys that share at least one band with the query (no Jaccard check)."""
        sketch = self._sketcher.sketch(set_)
        arr = np.array(sketch, dtype=np.uint64)
        candidates: Set[Any] = set()
        for b in range(self.bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band = arr[start:end]
            h = self._band_hash(band)
            candidates.update(self._buckets[b].get(h, set()))
        return list(candidates)

    def clear(self) -> None:
        """Remove all keys from the index."""
        for b in range(self.bands):
            self._buckets[b].clear()
        self._keys.clear()

    @staticmethod
    def _estimate_jaccard(sketch1: List[int], sketch2: List[int]) -> float:
        """Estimate Jaccard similarity from two sketches."""
        assert len(sketch1) == len(sketch2)
        matches = sum(1 for a, b in zip(sketch1, sketch2) if a == b)
        return matches / len(sketch1) 