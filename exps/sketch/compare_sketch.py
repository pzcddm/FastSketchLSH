'''
Comparison Test: FastSimilaritySketch vs DatasketchMinHashSketch vs CMinHashSketch

This file compares the performance of FastSimilaritySketch, DatasketchMinHashSketch, 
and CMinHashSketch algorithms in terms of estimation accuracy and execution speed across varying parameters.

The test varies:
- k values: [64, 128, 256]
- Set sizes (n): [100, 400, 1600, 6400, 25600]

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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from typing import List, Tuple, Iterable

from FastSketchLSH import FastSimilaritySketch
from prototype.src.datasketch_sketch import DatasketchMinHashSketch
from prototype.src.cmins_sketch import CMinHashSketch
from prototype.src.rmins_sketch import RMinHashSketch
from prototype.simulation.util import estimate_jaccard, actual_jaccard, generate_interval_sets_with_jaccard


class SketchComparison:
    """
    Compares FastSimilaritySketch, DatasketchMinHashSketch, and CMinHashSketch performance across different parameters.
    """

    def __init__(self):
        self.k_values = [64, 128, 256]
        self.n_values = [100, 400, 1600, 6400, 25600]
        self.results = []

    def generate_test_sets(self, n: int, overlap_ratio: float = 0.5, trial: int = 0) -> Tuple[set, set]:
        """
        Generate two test sets with controlled overlap using util.py.
        Ensures each call produces different sets by offsetting start_id with trial.
        """
        set_a, set_b, _ = generate_interval_sets_with_jaccard(overlap_ratio, n, start_id=trial * n)
        return set_a, set_b

    def time_sketch_generation(self, sketcher, items: Iterable) -> Tuple[List[int], float]:
        """
        Measure time to generate a sketch and return both sketch and duration.

        Args:
            sketcher: Sketch algorithm instance
            items: Iterable passed directly to sketcher.sketch (preprocessed as needed)

        Returns:
            (sketch, elapsed_seconds)
        """
        start_time = time.perf_counter()
        sketch = sketcher.sketch(items)
        end_time = time.perf_counter()
        return sketch, (end_time - start_time)

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
        rmins_errors = []
        fast_times = []
        datasketch_times = []
        cmins_times = []
        rmins_times = []

        for trial in range(num_trials):
            # Generate test sets with 50% overlap, unique per trial
            set_a, set_b = self.generate_test_sets(n, overlap_ratio=0.5, trial=trial)

            true_jaccard = actual_jaccard(set_a, set_b)

            # Precompute representations per algorithm for fair timing
            int_a = list(set_a)
            int_b = list(set_b)
            str_a = [str(x) for x in int_a]
            str_b = [str(x) for x in int_b]

            # Test FastSimilaritySketch (SIMD expects integers)
            fast_sketcher = FastSimilaritySketch(sketch_size=k)

            # sketch_a, time_a = self.time_sketch_generation(fast_sketcher, int_a)
            # sketch_b, time_b = self.time_sketch_generation(fast_sketcher, int_b)
            sketch_a, time_a = self.time_sketch_generation(fast_sketcher, str_a)
            sketch_b, time_b = self.time_sketch_generation(fast_sketcher, str_b)
            
            fast_total_time = time_a + time_b
            fast_estimated = estimate_jaccard(sketch_a, sketch_b)
            fast_error = abs(true_jaccard - fast_estimated)

            # Test DatasketchMinHashSketch
            datasketch_sketcher = DatasketchMinHashSketch(num_perm=k)

            # Time sketch generation for set A and B using pre-stringified items
            sketch_a, time_a = self.time_sketch_generation(datasketch_sketcher, str_a)
            sketch_b, time_b = self.time_sketch_generation(datasketch_sketcher, str_b)
            datasketch_total_time = time_a + time_b
            datasketch_estimated = estimate_jaccard(sketch_a, sketch_b)
            datasketch_error = abs(true_jaccard - datasketch_estimated)

            # Test CMinHashSketch
            cmins_sketcher = CMinHashSketch(num_perm=k)

            # Time sketch generation for set A and B using pre-stringified items
            sketch_a, time_a = self.time_sketch_generation(cmins_sketcher, str_a)
            sketch_b, time_b = self.time_sketch_generation(cmins_sketcher, str_b)
            cmins_total_time = time_a + time_b
            cmins_estimated = estimate_jaccard(sketch_a, sketch_b)
            cmins_error = abs(true_jaccard - cmins_estimated)

            # Collect results
            fast_errors.append(fast_error)
            datasketch_errors.append(datasketch_error)
            cmins_errors.append(cmins_error)

            fast_times.append(fast_total_time)
            datasketch_times.append(datasketch_total_time)
            cmins_times.append(cmins_total_time)

            # Test RMinHashSketch (Rensa)
            rmins_sketcher = RMinHashSketch(num_perm=k)

            # Time sketch generation for set A and B using pre-stringified items
            sketch_a, time_a = self.time_sketch_generation(rmins_sketcher, str_a)
            sketch_b, time_b = self.time_sketch_generation(rmins_sketcher, str_b)
            rmins_total_time = time_a + time_b
            rmins_estimated = estimate_jaccard(sketch_a, sketch_b)
            rmins_error = abs(true_jaccard - rmins_estimated)

            rmins_errors.append(rmins_error)
            rmins_times.append(rmins_total_time)

        # Calculate averages
        avg_fast_error = sum(fast_errors) / len(fast_errors)
        avg_datasketch_error = sum(datasketch_errors) / len(datasketch_errors)
        avg_cmins_error = sum(cmins_errors) / len(cmins_errors)
        avg_rmins_error = sum(rmins_errors) / len(rmins_errors)
        avg_fast_time = sum(fast_times) / len(fast_times)
        avg_datasketch_time = sum(datasketch_times) / len(datasketch_times)
        avg_cmins_time = sum(cmins_times) / len(cmins_times)
        avg_rmins_time = sum(rmins_times) / len(rmins_times)

        return {
            'k': k,
            'n': n,
            'true_jaccard': true_jaccard,
            'fast_avg_error': avg_fast_error,
            'datasketch_avg_error': avg_datasketch_error,
            'cmins_avg_error': avg_cmins_error,
            'rmins_avg_error': avg_rmins_error,
            'fast_avg_time': avg_fast_time,
            'datasketch_avg_time': avg_datasketch_time,
            'cmins_avg_time': avg_cmins_time,
            'rmins_avg_time': avg_rmins_time,
            'fast_speedup_vs_datasketch': avg_datasketch_time / avg_fast_time if avg_fast_time > 0 else 0,
            'fast_speedup_vs_cmins': avg_cmins_time / avg_fast_time if avg_fast_time > 0 else 0,
            'fast_speedup_vs_rmins': avg_rmins_time / avg_fast_time if avg_fast_time > 0 else 0,
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
                print(f"  RMin error: {result['rmins_avg_error']:.6f}")
                print(f"  Fast time: {result['fast_avg_time']:.6f}s")
                print(f"  Datasketch time: {result['datasketch_avg_time']:.6f}s")
                print(f"  CMins time: {result['cmins_avg_time']:.6f}s")
                print(f"  RMin time: {result['rmins_avg_time']:.6f}s")
                print(f"  Speedup (FastSketch vs Datasketch): {result['fast_speedup_vs_datasketch']:.2f}x")
                print(f"  Speedup (FastSketch vs CMins): {result['fast_speedup_vs_cmins']:.2f}x")
                print(f"  Speedup (FastSketch vs RMin): {result['fast_speedup_vs_rmins']:.2f}x")
                print()

    def save_results_to_csv(self, filename: str = "sketch_comparison_results.csv") -> None:
        """
        Save comparison results to a CSV file in the records directory.
        Args:
            filename: Name of the CSV file to create
        """
        # Always save to the records directory alongside this script
        records_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'records'))
        os.makedirs(records_dir, exist_ok=True)
        filepath = os.path.join(records_dir, filename)
        fieldnames = [
            'k', 'n', 'true_jaccard',
            'fast_avg_error', 'datasketch_avg_error', 'cmins_avg_error', 'rmins_avg_error',
            'fast_avg_time', 'datasketch_avg_time', 'cmins_avg_time', 'rmins_avg_time',
            'fast_speedup_vs_datasketch', 'fast_speedup_vs_cmins', 'fast_speedup_vs_rmins'
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
