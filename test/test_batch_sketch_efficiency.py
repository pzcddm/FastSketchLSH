import os
import sys
import time
import math
import random
from typing import List
import numpy as np
HAS_NUMPY = True

"""
Simple efficiency comparison of FastSimilaritySketch:
- Single (one-by-one)
- Batch (single-thread)
- Batch (multi-thread)

Uses Python lists to avoid external dependencies.
"""

# Prefer local cpp_src build over any installed package to avoid ABI/version mismatch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cpp_src')))
from FastSketchLSH import FastSimilaritySketch, omp_max_threads


def generate_int_batches(num_sets: int, set_size: int, universe: int, seed: int = 1234) -> List[List[int]]:
    """
    Generate a list of NumPy int32 arrays representing sets of integers in [0, universe).

    Time: O(num_sets * set_size)
    Space: O(num_sets * set_size)
    """
    rnd = random.Random(seed)
    batches: List[List[int]] = []
    for _ in range(num_sets):
        arr = [rnd.randrange(universe) for _ in range(set_size)]
        batches.append(arr)
    return batches


def benchmark(sketcher: FastSimilaritySketch, batches: List[List[int]], mode: str, num_threads: int = 0) -> float:
    """
    Run the benchmark for a given mode.
    Returns elapsed time in seconds.

    mode in {"single", "batch"}
    If mode == "batch", uses num_threads (0 means all threads).
    """
    t0 = time.perf_counter()
    if mode == "single":
        sink = 0
        for arr in batches:
            a = np.asarray(arr, dtype=np.int32)
            S = sketcher.sketch(a)
            sink ^= int(S[0] & 0xFFFFFFFFFFFFFFFF)
        # Prevent optimization
        if sink == 0xDEADBEEF:  # never true
            print("sink hit")
    elif mode == "batch":
        results = sketcher.sketch_batch(batches, num_threads=num_threads)
        # Light use of results to avoid optimization
        if len(results) == -1:  # never true
            print("impossible")
    else:
        raise ValueError("Unknown mode")
    t1 = time.perf_counter()
    return t1 - t0



def list_to_csr_int32(batches: List[List[int]]):
    total = sum(len(x) for x in batches)
    data = np.empty(total, dtype=np.uint32)
    indptr = np.empty(len(batches) + 1, dtype=np.uint64)
    pos = 0
    indptr[0] = 0
    for i, arr in enumerate(batches):
        n = len(arr)
        if n:
            # Convert the current Python list 'arr' to a NumPy array of type uint32,
            # and assign it to the corresponding slice in the preallocated 'data' array.
            # This efficiently copies the integer values from 'arr' into the correct position in 'data'.
            data[pos:pos+n] = np.fromiter(arr, dtype=np.uint32, count=n)
        pos += n
        indptr[i+1] = pos
    return data, indptr

def benchmark_csr_flat(sketcher: FastSimilaritySketch, batches: List[List[int]], num_threads: int = 0) -> float:
    if not HAS_NUMPY:
        return float('nan')
    data, indptr = list_to_csr_int32(batches)
    t0 = time.perf_counter()
    _ = sketcher.sketch_batch_flat_csr(data, indptr, num_threads=num_threads)
    t1 = time.perf_counter()
    return t1 - t0


if __name__ == "__main__":
    # Parameters
    num_sets = int(float(os.getenv("FSK_NUM_SETS", "10000")))
    set_size = int(os.getenv("FSK_SET_SIZE", "1000"))
    universe = int(os.getenv("FSK_UNIVERSE", "200000"))
    k = int(os.getenv("FSK_K", "128"))
    seed = int(os.getenv("FSK_SEED", "42"))

    print(f"Generating data: num_sets={num_sets}, set_size={set_size}, universe={universe}")
    batches = generate_int_batches(num_sets, set_size, universe, seed)

    sketcher = FastSimilaritySketch(sketch_size=k, seed=seed)
    cpu_threads = omp_max_threads()

    # Single (one by one)
    t_single = benchmark(sketcher, batches, mode="single")
    qps_single = num_sets / t_single if t_single > 0 else float('inf')

    # Batch (single-thread)
    t_batch_1 = benchmark(sketcher, batches, mode="batch", num_threads=1)
    qps_batch_1 = num_sets / t_batch_1 if t_batch_1 > 0 else float('inf')

    # Batch (multi-thread: all CPU threads)
    t_batch_mt = benchmark(sketcher, batches, mode="batch", num_threads=0)
    qps_batch_mt = num_sets / t_batch_mt if t_batch_mt > 0 else float('inf')

    # Flat batch (list input)
    # Flat (list-based) removed

    # CSR zero-copy (requires NumPy)
    t_csr_1 = benchmark_csr_flat(sketcher, batches, num_threads=1)
    qps_csr_1 = num_sets / t_csr_1 if t_csr_1 > 0 else float('inf')
    t_csr_mt = benchmark_csr_flat(sketcher, batches, num_threads=0)
    qps_csr_mt = num_sets / t_csr_mt if t_csr_mt > 0 else float('inf')

    print("\n=== FastSimilaritySketch Batch Efficiency (int32) ===")
    print(f"k={k}, sets={num_sets}, set_size={set_size}, threads={cpu_threads}")
    print(f"- Single (loop): {t_single:.4f}s, QPS={qps_single:.1f}")
    print(f"- Batch (1 thread): {t_batch_1:.4f}s, QPS={qps_batch_1:.1f}")
    print(f"- Batch (all threads): {t_batch_mt:.4f}s, QPS={qps_batch_mt:.1f}")
    # Flat (list-based) removed
    print(f"- CSR Flat (1 thread): {t_csr_1:.4f}s, QPS={qps_csr_1:.1f}")
    print(f"- CSR Flat (all threads): {t_csr_mt:.4f}s, QPS={qps_csr_mt:.1f}")


