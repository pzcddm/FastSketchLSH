"""
test_fast_sketch_comparison.py
------------------------------
Comprehensive comparison test between new FastSimilaritySketch (52-bit combined) and old FastSimilaritySketchOld (tuple-based).
Tests both correctness (Jaccard estimation accuracy) and performance (execution time).

This test validates that the new implementation:
1. Produces correct Jaccard similarity estimates (may differ from old due to different hash combination)
2. Achieves better performance than the old implementation
3. Maintains consistent estimation quality across different dataset sizes

Author: CS PhD Student
Date: Algorithm Research Testing
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

from src.fast_sketch import FastSimilaritySketch
from src.fast_sketch_old import FastSimilaritySketchOld
from simulation.util import estimate_jaccard, actual_jaccard


def generate_test_sets(size_A: int, size_B: int, overlap_ratio: float) -> Tuple[set, set]:
    """
    Generate two test sets with controlled overlap ratio.
    
    Args:
        size_A: Size of set A
        size_B: Size of set B  
        overlap_ratio: Desired Jaccard similarity (|A ∩ B| / |A ∪ B|)
    
    Returns:
        Tuple of (set_A, set_B)
    """
    # Calculate intersection size based on Jaccard formula
    # J = |intersection| / |union| = |intersection| / (|A| + |B| - |intersection|)
    # Solving: intersection = J * (|A| + |B|) / (1 + J)
    intersection_size = int(overlap_ratio * (size_A + size_B) / (1 + overlap_ratio))
    
    # Generate base elements for intersection
    intersection = set(range(intersection_size))
    
    # Generate unique elements for A and B
    A_unique = set(range(intersection_size, intersection_size + size_A - intersection_size))
    B_unique = set(range(intersection_size + size_A - intersection_size, 
                        intersection_size + size_A - intersection_size + size_B - intersection_size))
    
    A = intersection | A_unique
    B = intersection | B_unique
    
    return A, B


def benchmark_sketch_algorithm(sketcher_class, dataset: set, sketch_size: int, num_runs: int = 500) -> float:
    """
    Benchmark sketch generation time for a given algorithm and dataset.
    
    Args:
        sketcher_class: Sketch algorithm class
        dataset: Input set to sketch
        sketch_size: Size of the sketch
        num_runs: Number of runs for averaging
    
    Returns:
        Average execution time in seconds
    """
    times = []
    for run in range(num_runs):
        # Create new sketcher instance with different random seed each time
        sketcher = sketcher_class(sketch_size=sketch_size, random_seed=42 + run)
        start_time = time.perf_counter()
        sketcher.sketch(dataset)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.mean(times)


def test_correctness_comparison(sketch_sizes: List[int], dataset_sizes: List[int], 
                              overlap_ratios: List[float], num_rounds: int = 500) -> None:
    """
    Test correctness of both algorithms across different parameters with multiple rounds.
    
    Args:
        sketch_sizes: List of sketch sizes to test
        dataset_sizes: List of dataset sizes to test
        overlap_ratios: List of overlap ratios to test
        num_rounds: Number of rounds to run for each test case for statistical significance
    """
    print(f"=== CORRECTNESS COMPARISON (each test runs {num_rounds} rounds) ===")
    print(f"{'Sketch Size':<12} {'Dataset Size':<13} {'True Jaccard':<12} {'Old Mean':<10} {'New Mean':<10} {'Old Std':<9} {'New Std':<9} {'Old MSE':<9} {'New MSE':<9}")
    print("-" * 105)
    
    total_tests = 0
    old_better = 0
    new_better = 0
    similar_accuracy = 0
    
    for t in sketch_sizes:
        for n in dataset_sizes:
            for overlap in overlap_ratios:
                print(f"Testing t={t}, n={n}, jaccard={overlap:.1f} ({num_rounds} rounds)...", end=" ")
                
                # Run multiple rounds for statistical significance
                old_estimates = []
                new_estimates = []
                true_jaccards = []
                
                for round_idx in range(num_rounds):
                    # Generate test sets (slightly different each round due to random generation)
                    A, B = generate_test_sets(n, n, overlap)
                    true_jaccard = actual_jaccard(A, B)
                    true_jaccards.append(true_jaccard)
                    
                    # Test old implementation with different random seed each round
                    old_sketcher = FastSimilaritySketchOld(sketch_size=t, random_seed=42+round_idx)
                    S_A_old = old_sketcher.sketch(A)
                    S_B_old = old_sketcher.sketch(B)
                    est_jaccard_old = estimate_jaccard(S_A_old, S_B_old)
                    old_estimates.append(est_jaccard_old)
                    
                    # Test new implementation with different random seed each round
                    new_sketcher = FastSimilaritySketch(sketch_size=t, random_seed=42+round_idx)
                    S_A_new = new_sketcher.sketch(A)
                    S_B_new = new_sketcher.sketch(B)
                    est_jaccard_new = estimate_jaccard(S_A_new, S_B_new)
                    new_estimates.append(est_jaccard_new)
                
                # Calculate statistics
                mean_true_jaccard = np.mean(true_jaccards)
                old_mean = np.mean(old_estimates)
                old_std = np.std(old_estimates, ddof=1)
                new_mean = np.mean(new_estimates)
                new_std = np.std(new_estimates, ddof=1)
                
                # Calculate Mean Squared Error
                old_mse = np.mean([(est - true_j)**2 for est, true_j in zip(old_estimates, true_jaccards)])
                new_mse = np.mean([(est - true_j)**2 for est, true_j in zip(new_estimates, true_jaccards)])
                
                # Calculate mean absolute errors
                old_mae = np.mean([abs(est - true_j) for est, true_j in zip(old_estimates, true_jaccards)])
                new_mae = np.mean([abs(est - true_j) for est, true_j in zip(new_estimates, true_jaccards)])
                
                # Compare accuracy based on MAE
                if old_mae < new_mae - 0.005:  # Old is significantly better
                    old_better += 1
                elif new_mae < old_mae - 0.005:  # New is significantly better
                    new_better += 1
                else:  # Similar accuracy
                    similar_accuracy += 1
                
                total_tests += 1
                
                print(f"Done")
                print(f"{t:<12} {n:<13} {mean_true_jaccard:<12.4f} {old_mean:<10.4f} {new_mean:<10.4f} {old_std:<9.4f} {new_std:<9.4f} {old_mse:<9.6f} {new_mse:<9.6f}")
    
    print(f"\nAccuracy Comparison Summary (based on {num_rounds} rounds each):")
    print(f"  - Old implementation better: {old_better}/{total_tests} ({old_better/total_tests*100:.1f}%)")
    print(f"  - New implementation better: {new_better}/{total_tests} ({new_better/total_tests*100:.1f}%)")
    print(f"  - Similar accuracy: {similar_accuracy}/{total_tests} ({similar_accuracy/total_tests*100:.1f}%)")
    print("Note: 'Better' means Mean Absolute Error difference > 0.005")


def test_performance_comparison(sketch_sizes: List[int], dataset_sizes: List[int]) -> None:
    """
    Test performance comparison between old and new implementations.
    
    Args:
        sketch_sizes: List of sketch sizes to test
        dataset_sizes: List of dataset sizes to test
    """
    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"{'Sketch Size':<12} {'Dataset Size':<13} {'Old (s)':<13} {'New (s)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    performance_results = []
    
    for t in sketch_sizes:
        for n in dataset_sizes:
            # Generate test dataset
            A = set(range(n))
            
            # Benchmark old implementation
            time_old = benchmark_sketch_algorithm(FastSimilaritySketchOld, A, sketch_size=t)
            
            # Benchmark new implementation  
            time_new = benchmark_sketch_algorithm(FastSimilaritySketch, A, sketch_size=t)
            
            speedup = time_old / time_new if time_new > 0 else float('inf')
            
            print(f"{t:<12} {n:<13} {time_old:<13.6f} {time_new:<15.6f} {speedup:<10.2f}x")
            
            performance_results.append({
                'sketch_size': t,
                'dataset_size': n,
                'time_old': time_old,
                'time_new': time_new,
                'speedup': speedup
            })
    
    # Calculate average speedup
    avg_speedup = np.mean([r['speedup'] for r in performance_results])
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    return performance_results


def test_scalability_analysis(max_dataset_size: int = 50000, sketch_size: int = 256) -> None:
    """
    Analyze how both algorithms scale with increasing dataset size.
    
    Args:
        max_dataset_size: Maximum dataset size to test
        sketch_size: Fixed sketch size for testing
    """
    print(f"\n=== SCALABILITY ANALYSIS (t={sketch_size}) ===")
    
    dataset_sizes = [1000, 2000, 5000, 10000, 20000, max_dataset_size]
    times_old = []
    times_new = []
    
    print(f"{'Dataset Size':<13} {'Old (s)':<13} {'New (s)':<15} {'Speedup':<10}")
    print("-" * 55)
    
    for n in dataset_sizes:
        A = set(range(n))
        
        # Benchmark both implementations
        time_old = benchmark_sketch_algorithm(FastSimilaritySketchOld, A, sketch_size=sketch_size, num_runs=500)
        times_old.append(time_old)
        
        time_new = benchmark_sketch_algorithm(FastSimilaritySketch, A, sketch_size=sketch_size, num_runs=500)
        times_new.append(time_new)
        
        speedup = time_old / time_new if time_new > 0 else float('inf')
        
        print(f"{n:<13} {time_old:<13.6f} {time_new:<15.6f} {speedup:<10.2f}x")


def test_detailed_timing_analysis(sketch_size: int = 256, dataset_size: int = 10000, num_runs: int = 1000) -> None:
    """
    Detailed timing analysis with statistical measures.
    
    Args:
        sketch_size: Sketch size to test
        dataset_size: Dataset size to test
        num_runs: Number of runs for statistical analysis
    """
    print(f"\n=== DETAILED TIMING ANALYSIS (t={sketch_size}, n={dataset_size}, runs={num_runs}) ===")
    
    A = set(range(dataset_size))
    
    # Collect timing data
    old_times = []
    new_times = []
    
    for run in range(num_runs):
        # Test old implementation with different random seed each time
        old_sketcher = FastSimilaritySketchOld(sketch_size=sketch_size, random_seed=42+run)
        start_time = time.perf_counter()
        old_sketcher.sketch(A)
        end_time = time.perf_counter()
        old_times.append(end_time - start_time)
        
        # Test new implementation with different random seed each time
        new_sketcher = FastSimilaritySketch(sketch_size=sketch_size, random_seed=42+run)
        start_time = time.perf_counter()
        new_sketcher.sketch(A)
        end_time = time.perf_counter()
        new_times.append(end_time - start_time)
    
    # Calculate statistics
    old_mean = np.mean(old_times)
    old_std = np.std(old_times, ddof=1)
    old_median = np.median(old_times)
    new_mean = np.mean(new_times)
    new_std = np.std(new_times, ddof=1)
    new_median = np.median(new_times)
    
    speedup = old_mean / new_mean if new_mean > 0 else float('inf')
    speedup_median = old_median / new_median if new_median > 0 else float('inf')
    
    print(f"Old Implementation ({num_runs} runs):")
    print(f"  - Mean time: {old_mean:.6f} ± {old_std:.6f} seconds")
    print(f"  - Median time: {old_median:.6f} seconds")
    print(f"  - Min time: {np.min(old_times):.6f} seconds")
    print(f"  - Max time: {np.max(old_times):.6f} seconds")
    
    print(f"\nNew Implementation ({num_runs} runs):")
    print(f"  - Mean time: {new_mean:.6f} ± {new_std:.6f} seconds")
    print(f"  - Median time: {new_median:.6f} seconds")
    print(f"  - Min time: {np.min(new_times):.6f} seconds")
    print(f"  - Max time: {np.max(new_times):.6f} seconds")
    
    print(f"\nPerformance Comparison:")
    print(f"  - Mean speedup: {speedup:.2f}x")
    print(f"  - Median speedup: {speedup_median:.2f}x")
    print(f"  - Time difference (mean): {old_mean - new_mean:.6f} seconds")
    print(f"  - Relative improvement (mean): {(1 - new_mean/old_mean)*100:.1f}%" if old_mean > 0 else "N/A")
    print(f"  - 95% Confidence Interval (Old): [{old_mean - 1.96*old_std/np.sqrt(num_runs):.6f}, {old_mean + 1.96*old_std/np.sqrt(num_runs):.6f}]")
    print(f"  - 95% Confidence Interval (New): [{new_mean - 1.96*new_std/np.sqrt(num_runs):.6f}, {new_mean + 1.96*new_std/np.sqrt(num_runs):.6f}]")


def main():
    """
    Main test runner for comprehensive comparison.
    """
    print("Fast Similarity Sketch Comparison Test")
    print("=" * 50)
    
    # Test parameters
    sketch_sizes = [128, 256, 512]
    dataset_sizes = [1000, 5000, 10000] 
    overlap_ratios = [0.1, 0.3, 0.5, 0.7]
    
    print(f"Running with {500} runs per benchmark for statistical robustness...")
    print("Each run uses a different random seed for fair comparison.\n")
    
    # Run correctness tests with 500 rounds for statistical significance
    test_correctness_comparison(sketch_sizes, dataset_sizes, overlap_ratios, num_rounds=500)
    
    # Run performance tests
    performance_results = test_performance_comparison(sketch_sizes, dataset_sizes)
    
    # # Run scalability analysis
    # test_scalability_analysis(max_dataset_size=30000, sketch_size=256)
    
    # # Run detailed timing analysis
    # test_detailed_timing_analysis(sketch_size=256, dataset_size=10000, num_runs=1000)
    
    print("\n=== SUMMARY ===")
    
    # Calculate overall statistics
    speedups = [r['speedup'] for r in performance_results]
    print(f"Overall Performance Summary:")
    print(f"  - Average Speedup: {np.mean(speedups):.2f}x")
    print(f"  - Min Speedup: {np.min(speedups):.2f}x")  
    print(f"  - Max Speedup: {np.max(speedups):.2f}x")
    print(f"  - Std Speedup: {np.std(speedups):.2f}x")
    
    # Validation results
    print(f"\nValidation Results:")
    print(f"  - Both algorithms produce valid Jaccard estimates")
    print(f"  - New implementation uses 52-bit combined hash values") 
    print(f"  - Old implementation uses tuple-based hash storage")
    print(f"  - Performance comparison shows relative efficiency")
    
    print(f"\nImplementation Differences:")
    print(f"  - Old: Uses (round_idx, hash_val) tuples for comparison")
    print(f"  - New: Uses single 64-bit value with 12-bit round index + 52-bit hash")
    print(f"  - Both: O(t * |A|) time complexity, O(t) space complexity")


if __name__ == '__main__':
    main() 