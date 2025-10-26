import os
import sys
# To import FastSketchLSH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import time
import argparse
import random
from collections import defaultdict
from typing import List, Set, Dict, Any, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset, Features, Sequence, Value
from tqdm import tqdm

# Your original imports
from datasketch import MinHash, MinHashLSH
from datasketch.lsh import _optimal_param
from rensa import RMinHash, RMinHashLSH
from FastSketchLSH import FastSimilaritySketch, LSH


class Timer:
    """Timer class for recording the execution time of algorithms."""

    def __init__(self):
        self.elapsed_times = {}
        self._start_time = None
        self._current_name = None

    def __call__(self, name):
        self._current_name = name
        return self

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self._start_time
        self.elapsed_times[self._current_name] = elapsed


class UnionFind:
    """Union-Find data structure for managing clusters of duplicate documents."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py


def _recall(row):
    """Compute recall."""
    labelled_dups = set(row["duplicates"])
    LEN_LABELLED_DUPLICATES = len(labelled_dups)
    if LEN_LABELLED_DUPLICATES == 0:
        return 1
    dups = set(row["predictions"])
    return len(dups & labelled_dups) / LEN_LABELLED_DUPLICATES


def _precision(row):
    """Compute precision."""
    labelled_dups = set(row["duplicates"])
    dups = set(row["predictions"])
    LEN_DUPLICATES = len(dups)
    if LEN_DUPLICATES == 0:
        return 0
    return len(dups & labelled_dups) / LEN_DUPLICATES


def classify_in_paper(record):
    """Classify prediction result: TP, TN, FP, FN."""
    duplicates = set(record["duplicates"])
    predictions = set(record["predictions"])

    LEN_PREDICTIONS = len(predictions)
    LEN_DUPLICATES = len(duplicates)

    if LEN_PREDICTIONS == 0:
        if LEN_DUPLICATES == 0:
            return "TN"
        if LEN_DUPLICATES > 0:
            return "FN"

    if LEN_PREDICTIONS > 0:
        if LEN_DUPLICATES > 0 and duplicates.issubset(predictions):
            return "TP"
        if LEN_DUPLICATES == 0 or not duplicates.issubset(predictions):
            return "FP"

    raise ValueError(f"This should not happen {duplicates} {predictions}")


def inverse(label: str) -> str:
    """Invert classification label."""
    return {"TP": "TN", "FN": "FP", "FP": "FN", "TN": "TP"}[label]


def calculate_metrics(uf: UnionFind, truth: list, id2core_id: dict, labels: dict, name: str, elapsed_time: float):
    """Compute and return metrics for an algorithm."""
    # Build clusters from UnionFind
    id2cluster = defaultdict(set)
    for idx in range(len(truth)):
        cluster_id = uf.find(idx)
        id2cluster[cluster_id].add(idx)

    # Generate predictions
    predictions = {}
    for x in truth:
        idx = x["id"]
        core_id = id2core_id[idx]
        cluster_id = uf.find(idx)
        neighbors = id2cluster[cluster_id]
        predictions[core_id] = {
            id2core_id[neighbor] for neighbor in neighbors
            if neighbor != idx and neighbor in id2core_id
        }

    # Create DataFrame for evaluation
    df = (
        pd.Series(labels)
        .to_frame("duplicates")
        .reset_index()
        .merge(pd.Series(predictions).to_frame("predictions").reset_index(), on="index")
    )

    # Compute accuracy
    df["Correct"] = df.apply(
        lambda row: set(row["duplicates"]) == set(row["predictions"]), axis=1
    ).astype(int)

    # Compute recall and precision
    recalls = df.apply(_recall, axis=1)
    precisions = df.apply(_precision, axis=1)

    # Classification results
    df["Class"] = df.apply(classify_in_paper, axis=1)
    df["Class_"] = df.apply(lambda row: inverse(row["Class"]), axis=1)

    # Compute precision and recall for each class
    metrics = {}
    for col in ["Class", "Class_"]:
        label_counts = df[col].value_counts().to_dict()
        tp = label_counts.get("TP", 0)
        fp = label_counts.get("FP", 0)
        fn = label_counts.get("FN", 0)

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        metrics[f"{col}_precision"] = precision
        metrics[f"{col}_recall"] = recall

    return {
        "name": name,
        "precision_duplicates": metrics.get("Class_precision", 0),
        "recall_duplicates": metrics.get("Class_recall", 0),
        "precision_non_duplicates": metrics.get("Class__precision", 0),
        "recall_non_duplicates": metrics.get("Class__recall", 0),
        "macro_f1": (metrics.get("Class_precision", 0) + metrics.get("Class__precision", 0)) / 2,
        "accuracy": df["Correct"].mean(),
        "time": elapsed_time,
        "total_predictions": len(predictions),
        "mean_recall": recalls.mean(),
        "mean_precision": precisions.mean()
    }


def convert_results_to_unionfind(kept_indices: Set[int], all_indices: Set[int],
                                 candidate_sets: List[Set[int]]) -> UnionFind:
    """Convert deduplication results into UnionFind format."""
    uf = UnionFind()

    # Initialize all indices
    for idx in all_indices:
        uf.find(idx)

    # Merge documents within each candidate set
    for i, candidates in enumerate(candidate_sets):
        if len(candidates) > 1:  # Merge only when a candidate set has multiple elements
            candidates_list = list(candidates)
            # Merge all elements in the candidate set with the first element
            for j in range(1, len(candidates_list)):
                uf.union(candidates_list[0], candidates_list[j])

    return uf


def print_performance_comparison(results_table: List[Dict]):
    """Print performance comparison."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    # Find the baseline algorithm (Datasketch)
    baseline = next(r for r in results_table if r["name"] == "Datasketch")

    for result in results_table:
        if result["name"] != "Datasketch":
            speedup = baseline["time"] / result["time"]
            accuracy_diff = result["accuracy"] - baseline["accuracy"]

            print(f"\n{result['name']} vs Datasketch:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Accuracy difference: {accuracy_diff:+.4f}")
            print(f"  Mean precision difference: {result['mean_precision'] - baseline['mean_precision']:+.4f}")
            print(f"  Mean recall difference: {result['mean_recall'] - baseline['mean_recall']:+.4f}")


def create_comparison_table(results_table: List[Dict]) -> pd.DataFrame:
    """Create a formatted comparison table."""
    data = []
    for result in results_table:
        data.append([
            result["name"],
            f"{result['precision_duplicates']:.4f}",
            f"{result['recall_duplicates']:.4f}",
            f"{result['precision_non_duplicates']:.4f}",
            f"{result['recall_non_duplicates']:.4f}",
            f"{result['macro_f1']:.4f}",
            f"{result['accuracy']:.4f}",
            f"{result['time']:.2f}s"
        ])

    df = pd.DataFrame(data, columns=[
        "Algorithm",
        "Precision (Duplicates)",
        "Recall (Duplicates)",
        "Precision (Non Duplicates)",
        "Recall (Non Duplicates)",
        "Macro F1 score",
        "Accuracy",
        "Time"
    ])

    return df

def run_evaluation_benchmark(args):
    """Run the full evaluation benchmark."""
    timer = Timer()

    print("Loading dataset...")
    # Load dataset
    ds = load_dataset("pinecone/core-2020-05-10-deduplication", split="train")

    # Sample data
    if args.ratio < 1.0:
        total_size = len(ds)
        sample_size = int(total_size * args.ratio)
        indices = random.sample(range(total_size), sample_size)
        ds = ds.select(indices)

    # Prepare text data
    texts = []
    truth = []
    for idx, record in enumerate(ds):
        text = " ".join((record.get("processed_title", ""),
                         record.get("processed_abstract", ""))).lower()
        texts.append(text)
        truth.append({
            "core_id": record["core_id"],
            "id": idx,
            "duplicates": record.get("labelled_duplicates", [])
        })

    # Build token sets
    token_sets = [list({tok for tok in text.lower().split() if tok}) for text in texts]

    # Build label mapping
    id2core_id = {x["id"]: int(x["core_id"]) for x in truth}
    labels = {
        int(x["core_id"]): set(map(int, x["duplicates"])) if x["duplicates"] else set()
        for x in truth
    }

    bands = args.bands
    rows = args.rows
    num_perm = bands * rows

    print(f"\nLSH Parameters:")
    print(f"  Threshold: {args.lsh_threshold}")
    print(f"  Bands: {bands}, Rows: {rows}")
    print(f"  Effective num_perm: {num_perm}")

    results_table = []

    # 1. Run Datasketch LSH
    print("\nRunning Datasketch LSH...")
    with timer("Datasketch"):
        ds_res = run_datasketch_lsh_with_candidates(
            token_sets, args.lsh_threshold, num_perm, bands, rows
        )

    # Convert to UnionFind format
    ds_uf = convert_results_to_unionfind(
        ds_res["kept_indices"],
        set(range(len(token_sets))),
        ds_res["candidate_sets"]
    )

    # Compute metrics
    ds_metrics = calculate_metrics(
        ds_uf, truth, id2core_id, labels,
        "Datasketch", timer.elapsed_times["Datasketch"]
    )
    results_table.append(ds_metrics)

    # 2. Run FastSketch LSH
    print("\nRunning FastSketch LSH...")
    with timer("FastSketch"):
        fs_res = run_fastsketch_lsh_with_candidates(
            token_sets, args.lsh_threshold, num_perm, bands, args.seed
        )

    fs_uf = convert_results_to_unionfind(
        fs_res["kept_indices"],
        set(range(len(token_sets))),
        fs_res["candidate_sets"]
    )

    fs_metrics = calculate_metrics(
        fs_uf, truth, id2core_id, labels,
        "FastSketch", timer.elapsed_times["FastSketch"]
    )
    results_table.append(fs_metrics)

    # 3. Run Rensa LSH
    print("\nRunning Rensa LSH...")
    with timer("Rensa"):
        rs_res = run_rensa_lsh_with_candidates(
            token_sets, args.lsh_threshold, num_perm, bands
        )

    rs_uf = convert_results_to_unionfind(
        rs_res["kept_indices"],
        set(range(len(token_sets))),
        rs_res["candidate_sets"]
    )

    rs_metrics = calculate_metrics(
        rs_uf, truth, id2core_id, labels,
        "Rensa", timer.elapsed_times["Rensa"]
    )
    results_table.append(rs_metrics)

    # Print results table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    df_results = pd.DataFrame(results_table)

    # Formatted output
    print("\nDetailed Metrics:")
    for _, row in df_results.iterrows():
        print(f"\n{row['name']}:")
        print(f"  Precision (Duplicates): {row['precision_duplicates']:.4f}")
        print(f"  Recall (Duplicates): {row['recall_duplicates']:.4f}")
        print(f"  Precision (Non-Duplicates): {row['precision_non_duplicates']:.4f}")
        print(f"  Recall (Non-Duplicates): {row['recall_non_duplicates']:.4f}")
        print(f"  Macro F1 Score: {row['macro_f1']:.4f}")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  Mean Precision: {row['mean_precision']:.4f}")
        print(f"  Mean Recall: {row['mean_recall']:.4f}")
        print(f"  Time: {row['time']:.2f}s")

    # Create comparison table
    comparison_df = df_results[[
        'name', 'mean_precision', 'mean_recall', 'accuracy', 'time'
    ]].round(4)

    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # Save results
    if args.save_results:
        output_file = f"evaluation_results_{args.lsh_threshold}_{args.num_perm}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    # After printing result tables
    print_performance_comparison(results_table)
    # Create and print a formatted table
    comparison_table = create_comparison_table(results_table)
    print("\n" + "=" * 80)
    print("FORMATTED RESULTS TABLE (Markdown Format)")
    print("=" * 80)
    print(comparison_table.to_markdown(index=False))

# Modify the original run functions to return candidate sets
def run_datasketch_lsh_with_candidates(
        token_sets: List[List[str]],
        threshold: float,
        num_perm: int,
        bands: int,
        rows: int
) -> Dict[str, Any]:
    """Run Datasketch LSH and return candidate sets."""
    n = len(token_sets)

    # Phase1: Build MinHash
    start1 = time.perf_counter()
    minhashes = []
    for tokens in token_sets:
        m = MinHash(num_perm=num_perm)
        for tok in tokens:
            m.update(tok.encode("utf-8"))
        minhashes.append(m)
    phase1_time = time.perf_counter() - start1

    # Phase2: Insert LSH Index
    start2 = time.perf_counter()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, params=(bands, rows))
    for idx, m in enumerate(minhashes):
        lsh.insert(idx, m)
    phase2_time = time.perf_counter() - start2

    # Phase3: Query
    start3 = time.perf_counter()
    candidate_sets = [set(lsh.query(m)) for m in minhashes]

    # Deduplication process
    to_remove = set()
    for i in range(n):
        candidates = candidate_sets[i]
        other_candidates = [c for c in candidates if c != i]

        if other_candidates:
            min_id = min([i] + other_candidates)
            to_remove.update([c for c in [i] + other_candidates if c != min_id])

    kept_indices = set(range(n)) - to_remove
    phase3_time = time.perf_counter() - start3

    return {
        "total_time": phase1_time + phase2_time + phase3_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": phase3_time,
        "kept_indices": kept_indices,
        "removed_count": len(to_remove),
        "kept_count": len(kept_indices),
        "candidate_sets": candidate_sets
    }


def run_fastsketch_lsh_with_candidates(
        token_sets: List[List[str]],
        threshold: float,
        num_perm: int,
        bands: int,
        random_seed: int = 42
) -> Dict[str, Any]:
    """Run FastSketch LSH and return candidate sets."""
    n = len(token_sets)

    # Phase 1: Sketching
    start1 = time.perf_counter()
    token_sets_bytes = [[tok.encode('utf-8') for tok in tokens] for tokens in token_sets]
    sketcher = FastSimilaritySketch(sketch_size=num_perm, seed=random_seed)
    minhashes = sketcher.sketch_batch(token_sets_bytes, num_threads=os.cpu_count() or 1)
    phase1_time = time.perf_counter() - start1

    # Phase 2: LSH Indexing
    start2 = time.perf_counter()
    lsh = LSH(num_perm=num_perm, num_bands=bands)
    lsh.build_from_batch(minhashes)
    phase2_time = time.perf_counter() - start2

    # Phase 3: Querying and Deduplication
    start3 = time.perf_counter()
    candidate_sets_list = lsh.batch_query(minhashes)
    candidate_sets = [set(candidates) for candidates in candidate_sets_list]

    # Deduplication process
    to_remove = set()
    for i in range(n):
        candidates = candidate_sets[i]
        other_candidates = [c for c in candidates if c != i]

        if other_candidates:
            min_id = min([i] + other_candidates)
            to_remove.update([c for c in [i] + other_candidates if c != min_id])

    kept_indices = set(range(n)) - to_remove
    phase3_time = time.perf_counter() - start3

    return {
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": phase3_time,
        "total_time": phase1_time + phase2_time + phase3_time,
        "kept_indices": kept_indices,
        "kept_count": len(kept_indices),
        "removed_count": len(to_remove),
        "candidate_sets": candidate_sets
    }

def run_rensa_lsh_with_candidates(
        token_sets: List[List[str]],
        threshold: float,
        num_perm: int,
        bands: int,
) -> Dict[str, Any]:
    """Run Rensa LSH and return candidate sets."""
    n = len(token_sets)

    # Phase1: Build MinHash
    start1 = time.perf_counter()
    minhashes = []
    for tokens in token_sets:
        m = RMinHash(num_perm=num_perm, seed=42)
        m.update(tokens)
        minhashes.append(m)
    phase1_time = time.perf_counter() - start1

    # Phase2: Insert LSH Index
    start2 = time.perf_counter()
    lsh = RMinHashLSH(threshold=threshold, num_perm=num_perm, num_bands=bands)
    for idx, m in enumerate(minhashes):
        lsh.insert(idx, m)
    phase2_time = time.perf_counter() - start2

    # Phase3: Query
    start3 = time.perf_counter()
    candidate_sets = [set(lsh.query(m)) for m in minhashes]

    # Deduplication process
    to_remove = set()
    for i in range(n):
        candidates = candidate_sets[i]
        other_candidates = [c for c in candidates if c != i]

        if other_candidates:
            min_id = min([i] + other_candidates)
            to_remove.update([c for c in [i] + other_candidates if c != min_id])

    kept_indices = set(range(n)) - to_remove
    phase3_time = time.perf_counter() - start3

    return {
        "total_time": phase1_time + phase2_time + phase3_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": phase3_time,
        "kept_indices": kept_indices,
        "removed_count": len(to_remove),
        "kept_count": len(kept_indices),
        "candidate_sets": candidate_sets
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark LSH deduplication with evaluation metrics"
    )
    parser.add_argument(
        "--num_perm", type=int, default=128,
        help="Number of permutations for MinHash."
    )
    parser.add_argument(
        "--lsh_threshold", type=float, default=0.8,
        help="LSH threshold parameter for candidate generation."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--ratio", type=float, default=0.1,
        help="Fraction of the dataset to use (0 < ratio <= 1)."
    )
    parser.add_argument(
        "--save_results", action="store_true",
        help="Save results to CSV file."
    )

    args = parser.parse_args()
    run_evaluation_benchmark(args)
