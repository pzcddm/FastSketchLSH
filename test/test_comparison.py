'''
Comparison Test: FastSimilaritySketch vs DatasketchMinHashSketch vs CMinHashSketch

This file compares the performance of FastSimilaritySketch, DatasketchMinHashSketch, 
and CMinHashSketch algorithms in terms of estimation accuracy and execution speed across varying parameters.

The test varies:
- k values: [16, 32, 64, 128, 256, 512]
- Set sizes (n): [100, 1250, 2500, 5000, 10000, 20000]

Results are saved to CSV files in the records/ directory for analysis.

Time Complexity:
- FastSimilaritySketch: O(n + k log k) (expectation)
- DatasketchMinHashSketch: O(k * n)
- CMinHashSketch: O(k * n)

Space Complexity: All O(k)
'''

import time
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Tuple

# Import our sketch implementations
# from src.fast_sketch import FastSimilaritySketch
from src.datasketch_sketch import DatasketchMinHashSketch
from src.cmins_rensa_sketch import CMinHashSketch
from FastSketchLSH import FastSimilaritySketch
from FastSketchLSH import FastSimilaritySketchSIMD
from src.fast_sketch32 import FastSimilaritySketch32Bit

from simulation.util import estimate_jaccard, actual_jaccard, generate_interval_sets_with_jaccard


class SketchComparison:
    """
    Compares FastSimilaritySketch, DatasketchMinHashSketch, and CMinHashSketch performance across different parameters.
    """

    def __init__(self):
        # self.k_values = [16, 32, 64, 128, 256, 512]
        # self.n_values = [100, 1250, 2500, 5000, 10000, 20000]
        self.k_values = [128]
        self.n_values = [1000]
        self.results = []

    def generate_test_sets(self, n: int, overlap_ratio: float = 0.5, trial: int = 0) -> Tuple[set, set]:
        """
        Generate two test sets with controlled overlap using util.py.
        Ensures each call produces different sets by offsetting start_id with trial.
        """
        set_a, set_b, _ = generate_interval_sets_with_jaccard(overlap_ratio, n, start_id=trial * n)
        return set_a, set_b

    def time_sketch_generation(self, sketcher, test_set: set) -> float:
        """
        Measure time to generate a sketch for a given set.
        
        Args:
            sketcher: Sketch algorithm instance
            test_set: Set to generate sketch for
            
        Returns:
            Time taken in seconds
        """
        start_time = time.perf_counter()
        sketch = sketcher.sketch(test_set)
        end_time = time.perf_counter()
        return end_time - start_time

    def run_single_test(self, k: int, n: int, num_trials: int = 50) -> dict:
        """
        Run a single comparison test for given k and n values.
        
        Args:
            k: Sketch size parameter
            n: Set size
            num_trials: Number of trials to average over
            
        Returns:
            Dictionary containing test results
        """
        print(f"Testing k={k}, n={n}")

        # Initialize results for this test
        fast_errors = []
        datasketch_errors = []
        cmins_errors = []
        fast_times = []
        datasketch_times = []
        cmins_times = []

        for trial in range(num_trials):
            # Generate test sets with 50% overlap, unique per trial
            set_a, set_b = self.generate_test_sets(n, overlap_ratio=0.5, trial=trial)

            true_jaccard = actual_jaccard(set_a, set_b)

            # Test FastSimilaritySketch
            # fast_sketcher = FastSimilaritySketch32Bit(sketch_size=k)
            # fast_sketcher = FastSimilaritySketch(sketch_size=k)
            fast_sketcher = FastSimilaritySketchSIMD(sketch_size=k)
            
            # Time sketch generation for set A and B
            time_a = self.time_sketch_generation(fast_sketcher, set_a)
            time_b = self.time_sketch_generation(fast_sketcher, set_b)
            fast_total_time = time_a + time_b

            # Get sketches and estimate
            sketch_a = fast_sketcher.sketch(set_a)
            sketch_b = fast_sketcher.sketch(set_b)
            fast_estimated = estimate_jaccard(sketch_a, sketch_b)
            fast_error = abs(true_jaccard - fast_estimated)

            # Test DatasketchMinHashSketch
            datasketch_sketcher = DatasketchMinHashSketch(num_perm=k)

            # Time sketch generation for set A and B
            time_a = self.time_sketch_generation(datasketch_sketcher, set_a)
            time_b = self.time_sketch_generation(datasketch_sketcher, set_b)
            datasketch_total_time = time_a + time_b

            # Get sketches and estimate
            sketch_a = datasketch_sketcher.sketch(set_a)
            sketch_b = datasketch_sketcher.sketch(set_b)
            datasketch_estimated = estimate_jaccard(sketch_a, sketch_b)
            datasketch_error = abs(true_jaccard - datasketch_estimated)

            # Test CMinHashSketch
            cmins_sketcher = CMinHashSketch(num_perm=k)

            # Time sketch generation for set A and B
            time_a = self.time_sketch_generation(cmins_sketcher, set_a)
            time_b = self.time_sketch_generation(cmins_sketcher, set_b)
            cmins_total_time = time_a + time_b

            # Get sketches and estimate
            sketch_a = cmins_sketcher.sketch(set_a)
            sketch_b = cmins_sketcher.sketch(set_b)
            cmins_estimated = estimate_jaccard(sketch_a, sketch_b)
            cmins_error = abs(true_jaccard - cmins_estimated)

            # Collect results
            fast_errors.append(fast_error)
            datasketch_errors.append(datasketch_error)
            cmins_errors.append(cmins_error)

            fast_times.append(fast_total_time)
            datasketch_times.append(datasketch_total_time)
            cmins_times.append(cmins_total_time)

        # Calculate averages
        avg_fast_error = sum(fast_errors) / len(fast_errors)
        avg_datasketch_error = sum(datasketch_errors) / len(datasketch_errors)
        avg_cmins_error = sum(cmins_errors) / len(cmins_errors)
        avg_fast_time = sum(fast_times) / len(fast_times)
        avg_datasketch_time = sum(datasketch_times) / len(datasketch_times)
        avg_cmins_time = sum(cmins_times) / len(cmins_times)

        return {
            'k': k,
            'n': n,
            'true_jaccard': true_jaccard,
            'fast_avg_error': avg_fast_error,
            'datasketch_avg_error': avg_datasketch_error,
            'cmins_avg_error': avg_cmins_error,
            'fast_avg_time': avg_fast_time,
            'datasketch_avg_time': avg_datasketch_time,
            'cmins_avg_time': avg_cmins_time,
            'fast_speedup_vs_datasketch': avg_fast_time / avg_datasketch_time if avg_datasketch_time > 0 else 0,
            'fast_speedup_vs_cmins': avg_fast_time / avg_cmins_time if avg_cmins_time > 0 else 0,
            'datasketch_speedup_vs_cmins': avg_datasketch_time / avg_cmins_time if avg_cmins_time > 0 else 0
        }

    def run_full_comparison(self) -> None:
        """
        Run the complete comparison across all parameter combinations.
        """
        print(
            "Starting comprehensive comparison of FastSimilaritySketch vs DatasketchMinHashSketch vs CMinHashSketch")
        print(f"Testing k values: {self.k_values}")
        print(f"Testing n values: {self.n_values}")
        print()

        total_tests = len(self.k_values) * len(self.n_values)
        current_test = 0

        for k in self.k_values:
            for n in self.n_values:
                current_test += 1
                print(f"Progress: {current_test}/{total_tests}")

                result = self.run_single_test(k, n)
                self.results.append(result)

                print(f"  Fast error: {result['fast_avg_error']:.6f}")
                print(f"  Datasketch error: {result['datasketch_avg_error']:.6f}")
                print(f"  CMins error: {result['cmins_avg_error']:.6f}")
                print(f"  Fast time: {result['fast_avg_time']:.6f}s")
                print(f"  Datasketch time: {result['datasketch_avg_time']:.6f}s")
                print(f"  CMins time: {result['cmins_avg_time']:.6f}s")
                print(f"  Speedup ratio (Fast/Datasketch): {result['fast_speedup_vs_datasketch']:.2f}")
                print(f"  Speedup ratio (Fast/CMins): {result['fast_speedup_vs_cmins']:.2f}")
                print(f"  Speedup ratio (Datasketch/CMins): {result['datasketch_speedup_vs_cmins']:.2f}")
                print()

    def save_results_to_csv(self, filename: str = "sketch_comparison_results.csv") -> None:
        """
        Save comparison results to a CSV file in the records directory.
        Args:
            filename: Name of the CSV file to create
        """
        # Always save to the records directory in the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        filepath = os.path.join(project_root, 'records', filename)
        fieldnames = [
            'k', 'n', 'true_jaccard',
            'fast_avg_error', 'datasketch_avg_error', 'cmins_avg_error',
            'fast_avg_time', 'datasketch_avg_time', 'cmins_avg_time',
            'fast_speedup_vs_datasketch', 'fast_speedup_vs_cmins',
            'datasketch_speedup_vs_cmins'
        ]
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"Results saved to {filepath}")


if __name__ == '__main__':
    # Run the comparison
    comparison = SketchComparison()
    comparison.run_full_comparison()
    comparison.save_results_to_csv()

    print("Comparison complete!")
    print(f"Total tests run: {len(comparison.results)}")
