"""
util.py
-------
Utility functions for set generation with specified Jaccard similarity.

Author: (your name)
Date: (today's date)

This module provides a simple, efficient function to generate two sets of integers with a specified Jaccard similarity, using a deterministic interval-based method.
"""
from typing import Set, Tuple

def generate_interval_sets_with_jaccard(
    target_jaccard: float, set_size: int, start_id: int = 0
) -> Tuple[Set[int], Set[int], float]:
    """
    Generate two sets of consecutive integers with a specified Jaccard similarity using interval logic.

    Args:
        target_jaccard: Desired Jaccard similarity (0 < target_jaccard < 1)
        set_size: Size of each set (must be positive integer)
        start_id: Starting integer for the first set (default 0)

    Returns:
        (set_A, set_B, actual_jaccard)
        set_A: Set[int] -- [start_id, start_id + set_size)
        set_B: Set[int] -- [start_id + offset, start_id + offset + set_size)
        actual_jaccard: float -- actual Jaccard similarity achieved

    Time Complexity: O(set_size)
    Space Complexity: O(set_size)
    """
    if not (0 < target_jaccard < 1):
        raise ValueError("target_jaccard must be between 0 and 1 (exclusive)")
    if set_size <= 0:
        raise ValueError("set_size must be positive")

    # Calculate offset so that the intersection size is as close as possible to the target
    # For two intervals of length n, overlap = n - offset
    # J = overlap / (2n - overlap) => overlap = J * 2n / (1 + J)
    overlap = int(target_jaccard * 2 * set_size / (1 + target_jaccard))
    offset = set_size - overlap
    if offset < 0:
        offset = 0
        overlap = set_size
    
    set_A = set(range(start_id, start_id + set_size))
    set_B = set(range(start_id + offset, start_id + offset + set_size))
    actual_jaccard = len(set_A & set_B) / len(set_A | set_B)
    return set_A, set_B, actual_jaccard 

def estimate_jaccard(sketch1, sketch2):
    """
    Estimate Jaccard similarity based on k-mins sketches.
    
    Args:
        sketch1: The minhash signature of the first set (list of integers).
        sketch2: The minhash signature of the second set (list of integers).
    
    Returns:
        The estimated Jaccard similarity as a float.
        
    Time Complexity: O(k)
    """
    if len(sketch1) != len(sketch2):
        raise ValueError("Sketches must have the same length to compare.")
    matches = sum(1 for i in range(len(sketch1)) if sketch1[i] == sketch2[i])
    return matches / len(sketch1)

def actual_jaccard(set1, set2):
    """
    Compute the actual Jaccard similarity between two sets.
    
    Args:
        set1: First set.
        set2: Second set.
    
    Returns:
        The Jaccard similarity as a float.
    """
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union != 0 else 0.0
