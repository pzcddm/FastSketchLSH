import argparse
import os
import sys
import time
import random
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from datasketch.lsh import _optimal_param
from rensa import RMinHash, RMinHashLSH
from tqdm import tqdm
from collections import defaultdict, deque
from typing import List, Set, Dict, Any, Tuple

from datasketch.lsh import _optimal_param


def create_rensa_minhash(text, num_perm, seed):
    m = RMinHash(num_perm=num_perm, seed=seed)
    m.update(text.split())
    return m


def create_datasketch_minhash(text, num_perm):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode("utf-8"))
    return m


# Calculate optimal number of bands (similar to datasketch's approach)
def calculate_optimal_num_bands(threshold, num_perm):
    """Calculate the optimal number of bands to achieve the desired threshold."""
    # This approximates datasketch's internal calculation
    # For a threshold t, we want to find b (bands) and r (rows per band) such that:
    # - b * r = num_perm
    # - A pair with Jaccard similarity s has probability 1-(1-s^r)^b of being a candidate
    # - We want this probability to be high when s >= threshold

    best_num_bands = 1
    best_error = float("inf")

    for b in range(1, num_perm + 1):
        if num_perm % b != 0:
            continue
        r = num_perm // b
        prob_at_threshold = 1 - (1 - threshold**r) ** b
        error = abs(prob_at_threshold - 0.5)
        if error < best_error:
            best_error = error
            best_num_bands = b

    return best_num_bands


def _compute_deduplication_metrics(candidate_sets: List[Set[int]]) -> Dict[str, Any]:
    n = len(candidate_sets)
    total_candidates_checked = sum(len(cs) - 1 for cs in candidate_sets)  # 减去自身

    # 构建图
    graph = defaultdict(list)
    for i, cs in enumerate(candidate_sets):
        for j in cs:
            if j != i:
                graph[i].append(j)
                graph[j].append(i)

    # 查找连通分量
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)

    # 计算保留的索引（每个分量保留最小值）
    duplicate_flags = [0] * n
    kept_indices = []
    for comp in components:
        rep = min(comp)
        kept_indices.append(rep)
        for node in comp:
            if node != rep:
                duplicate_flags[node] = 1

    kept_count = len(kept_indices)
    removed_count = n - kept_count

    return {
        "kept_indices": kept_indices,
        "removed_count": removed_count,
        "kept_count": kept_count,
        "total_candidates": total_candidates_checked,
        "avg_candidates_per_query": total_candidates_checked / n if n > 0 else 0,
    }

def run_datasketch_lsh(
        token_sets: List[List[str]],
        threshold: float,
        num_perm: int,
        bands: int,
        rows: int
) -> Dict[str, Any]:
    from datasketch import MinHash, MinHashLSH
    n = len(token_sets)

    # 阶段1: 构建MinHash
    start1 = time.perf_counter()
    minhashes = []
    for tokens in token_sets:
        m = MinHash(num_perm=num_perm)
        for tok in tokens:
            m.update(tok.encode("utf-8"))
        minhashes.append(m)
    phase1_time = time.perf_counter() - start1

    # 阶段2: 插入到LSH
    start2 = time.perf_counter()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, params=(bands, rows))
    for idx, m in enumerate(minhashes):
        lsh.insert(idx, m)
    phase2_time = time.perf_counter() - start2

    # 阶段3: 查询并计算指标
    start3 = time.perf_counter()
    candidate_sets = [lsh.query(m) for m in minhashes]
    metrics = _compute_deduplication_metrics(candidate_sets)
    phase3_time = time.perf_counter() - start3

    return {
        "total_time": phase1_time + phase2_time + phase3_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": phase3_time,
        **metrics
    }

def _hamming_diff_rate(a: List[int], b: List[int]) -> Tuple[int, float]:
    assert len(a) == len(b)
    diffs = sum(1 for i, j in zip(a, b) if i != j)
    return diffs, diffs / max(1, len(a))

def run_fastsketch_lsh(
        token_sets: List[List[str]],
        threshold: float,
        sketch_size: int,
        bands: int,
        random_seed: int = 42
) -> Dict[str, Any]:
    from src.fast_sketch_lsh import FastSketchLSH
    n = len(token_sets)

    # 阶段1: 插入（包括构建草图）
    start1 = time.perf_counter()
    lsh = FastSketchLSH(threshold=threshold, sketch_size=sketch_size, bands=bands, random_seed=random_seed)
    for idx, tokens in enumerate(token_sets):
        lsh.insert(idx, tokens)
    phase1_time = time.perf_counter() - start1

    # 阶段2: 查询并计算指标
    start2 = time.perf_counter()
    candidate_sets = [lsh.query(tokens) for tokens in token_sets]
    metrics = _compute_deduplication_metrics(candidate_sets)
    phase2_time = time.perf_counter() - start2

    return {
        "total_time": phase1_time + phase2_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": 0.0,  # 无第三阶段
        **metrics
    }

# def deduplicate_with_rensa_lsh(
#     dataset,
#     num_perm,
#     seed,
#     lsh_threshold,
#     num_bands,
#     final_jaccard_threshold,
#     limit=None,
# ):
#     print(
#         f"\nRensa LSH Deduplication (num_perm={num_perm}, lsh_threshold={lsh_threshold}, "
#         f"num_bands={num_bands}, rows_per_band={num_perm // num_bands}, final_jaccard_threshold={final_jaccard_threshold})"
#     )
#     start_time = time.time()
#
#     if limit:
#         dataset = dataset.select(range(limit))
#         print(f"Processing a limited dataset of {limit} rows.")
#
#     # Phase 1: Generate MinHashes
#     print("Phase 1: Generating Rensa MinHashes...")
#     phase1_start = time.time()
#     minhashes = {}
#     for idx, example in tqdm(
#         enumerate(dataset), total=len(dataset), desc="Rensa MinHashing"
#     ):
#         minhashes[idx] = create_rensa_minhash(example["sql"], num_perm, seed)
#     phase1_time = time.time() - phase1_start
#
#     # Phase 2: Build LSH Index
#     print("Phase 2: Building Rensa LSH index...")
#     phase2_start = time.time()
#     lsh_index = RMinHashLSH(
#         threshold=lsh_threshold, num_perm=num_perm, num_bands=num_bands
#     )
#     for doc_id, rminhash_obj in tqdm(
#         minhashes.items(), desc="Inserting into Rensa LSH"
#     ):
#         lsh_index.insert(doc_id, rminhash_obj)
#     phase2_time = time.time() - phase2_start
#
#     # Phase 3: Query and Deduplicate
#     print("Phase 3: Querying Rensa LSH and deduplicating...")
#     phase3_start = time.time()
#     to_remove = set()
#     sorted_doc_ids = sorted(minhashes.keys())
#     total_candidates_checked = 0
#
#     for doc_id in tqdm(sorted_doc_ids, desc="Rensa LSH Querying"):
#         if doc_id in to_remove:
#             continue
#
#         query_minhash = minhashes[doc_id]
#         candidate_ids = lsh_index.query(query_minhash)
#         total_candidates_checked += len(candidate_ids)
#
#         for candidate_id in candidate_ids:
#             if candidate_id == doc_id or candidate_id in to_remove:
#                 continue
#
#             if candidate_id not in minhashes:
#                 continue
#
#             candidate_minhash = minhashes[candidate_id]
#             actual_jaccard = query_minhash.jaccard(candidate_minhash)
#
#             if actual_jaccard >= final_jaccard_threshold:
#                 if doc_id < candidate_id:
#                     to_remove.add(candidate_id)
#                 else:
#                     to_remove.add(doc_id)
#                     break
#
#     phase3_time = time.time() - phase3_start
#     kept_indices = set(sorted_doc_ids) - to_remove
#     total_time = time.time() - start_time
#
#     return {
#         "total_time": total_time,
#         "phase1_time": phase1_time,
#         "phase2_time": phase2_time,
#         "phase3_time": phase3_time,
#         "kept_indices": kept_indices,
#         "removed_count": len(to_remove),
#         "kept_count": len(kept_indices),
#         "total_candidates": total_candidates_checked,
#         "avg_candidates_per_query": total_candidates_checked / len(sorted_doc_ids)
#         if sorted_doc_ids
#         else 0,
#     }
#
#
# def deduplicate_with_datasketch_lsh(
#     dataset, num_perm, lsh_threshold, final_jaccard_threshold, limit=None
# ):
#     print(
#         f"\nDatasketch LSH Deduplication (num_perm={num_perm}, lsh_threshold={lsh_threshold}, "
#         f"final_jaccard_threshold={final_jaccard_threshold})"
#     )
#     # Note: datasketch automatically calculates num_bands internally
#     start_time = time.time()
#
#     if limit:
#         dataset = dataset.select(range(limit))
#         print(f"Processing a limited dataset of {limit} rows.")
#
#     # Phase 1: Generate MinHashes
#     print("Phase 1: Generating Datasketch MinHashes...")
#     phase1_start = time.time()
#     minhashes = {}
#     for idx, example in tqdm(
#         enumerate(dataset), total=len(dataset), desc="Datasketch MinHashing"
#     ):
#         minhashes[idx] = create_datasketch_minhash(example["sql"], num_perm)
#     phase1_time = time.time() - phase1_start
#
#     # Phase 2: Build LSH Index
#     print("Phase 2: Building Datasketch LSH index...")
#     phase2_start = time.time()
#     lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
#
#     for doc_id, minhash_obj in tqdm(
#         minhashes.items(), desc="Inserting into Datasketch LSH"
#     ):
#         lsh.insert(str(doc_id), minhash_obj)
#     phase2_time = time.time() - phase2_start
#
#     # Phase 3: Query and Deduplicate
#     print("Phase 3: Querying Datasketch LSH and deduplicating...")
#     phase3_start = time.time()
#     to_remove = set()
#     sorted_doc_ids = sorted(minhashes.keys())
#     total_candidates_checked = 0
#
#     for doc_id in tqdm(sorted_doc_ids, desc="Datasketch LSH Querying"):
#         if doc_id in to_remove:
#             continue
#
#         query_minhash = minhashes[doc_id]
#         candidate_keys = lsh.query(query_minhash)
#         total_candidates_checked += len(candidate_keys)
#
#         for candidate_key in candidate_keys:
#             candidate_id = int(candidate_key)
#             if candidate_id == doc_id or candidate_id in to_remove:
#                 continue
#
#             if candidate_id not in minhashes:
#                 continue
#
#             candidate_minhash = minhashes[candidate_id]
#             actual_jaccard = query_minhash.jaccard(candidate_minhash)
#
#             if actual_jaccard >= final_jaccard_threshold:
#                 if doc_id < candidate_id:
#                     to_remove.add(candidate_id)
#                 else:
#                     to_remove.add(doc_id)
#                     break
#
#     phase3_time = time.time() - phase3_start
#     kept_indices = set(sorted_doc_ids) - to_remove
#     total_time = time.time() - start_time
#
#     return {
#         "total_time": total_time,
#         "phase1_time": phase1_time,
#         "phase2_time": phase2_time,
#         "phase3_time": phase3_time,
#         "kept_indices": kept_indices,
#         "removed_count": len(to_remove),
#         "kept_count": len(kept_indices),
#         "total_candidates": total_candidates_checked,
#         "avg_candidates_per_query": total_candidates_checked / len(sorted_doc_ids)
#         if sorted_doc_ids
#         else 0,
#     }

def _extract_text(record: dict) -> str:
    """Best-effort extraction of text field from a dataset record."""
    for key in ("text", "content", "document", "body", "raw"):
        if key in record and isinstance(record[key], str) and record[key].strip():
            return record[key]
    # Fallback to concatenation of string-like fields
    parts: List[str] = []
    for v in record.values():
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return " \n ".join(parts)

def _tokenize_to_set(text: str) -> List[str]:
    """Simple whitespace tokenization to a deduplicated list (set-like)."""
    return list({tok for tok in text.lower().split() if tok})

def _build_token_sets(texts: List[str]) -> List[List[str]]:
    return [_tokenize_to_set(t) for t in texts]

def run_lsh_benchmark(args):
    print("Loading dataset...")
    ds = load_dataset("pinecone/core-2020-05-10-deduplication")
    pinecone_ds = list(ds["train"])
    random.shuffle(pinecone_ds)
    dataset_limit = (
        args.limit if args.limit and args.limit < len(pinecone_ds) else None
    )
    print(
        f"Using {dataset_limit if dataset_limit else len(pinecone_ds)} rows for the benchmark."
    )
    texts = [_extract_text(rec) for rec in pinecone_ds]
    token_build_start = time.perf_counter()
    token_sets = _build_token_sets(texts)
    token_build_time = time.perf_counter() - token_build_start

    # User Parameters
    NUM_PERM = args.num_perm
    SEED = args.seed
    LSH_THRESHOLD = args.lsh_threshold
    FINAL_JACCARD_THRESHOLD = args.final_jaccard_threshold

    # Calculate optimal number of bands for fair comparison
    # if args.num_bands:
    #     NUM_BANDS_RENSA = args.num_bands
    # else:
    #     NUM_BANDS_RENSA = calculate_optimal_num_bands(LSH_THRESHOLD, NUM_PERM)
    #     print(
    #         f"\nCalculated optimal num_bands for threshold {LSH_THRESHOLD}: {NUM_BANDS_RENSA}"
    #     )
    bands, rows = _optimal_param(LSH_THRESHOLD, NUM_PERM, 0.5, 0.5)

    # reset the num_perm to the bands * rows (Cause this is the real num_perm we use)
    num_perm = bands * rows
    print(f"bands: {bands}, rows: {rows}, num_perm: {num_perm}")

    # 运行Datasketch LSH
    ds_res = run_datasketch_lsh(token_sets, LSH_THRESHOLD, NUM_PERM, bands, rows)
    # 运行FastSketch LSH
    fs_res = run_fastsketch_lsh(token_sets, LSH_THRESHOLD, NUM_PERM, bands, SEED)

    # 提取简单标记（用于比较）
    ds_flags_simple = [1 if i not in ds_res["kept_indices"] else 0 for i in range(len(token_sets))]
    fs_flags_simple = [1 if i not in fs_res["kept_indices"] else 0 for i in range(len(token_sets))]

    # 比较结果
    diffs, rate = _hamming_diff_rate(ds_flags_simple, fs_flags_simple)

    # 输出结果
    print(f"Total texts: {len(texts)}")
    print(f"bands: {bands}, rows: {rows}, threshold: {LSH_THRESHOLD}, num_perm: {num_perm}")
    print(f"datasketch duplicate count: {ds_res['removed_count']}")
    print(f"fastsketch duplicate count: {fs_res['removed_count']}")
    print(f"Hamming differences: {diffs}, rate: {rate:.4f}")

    # 输出时间信息
    print("\nDatasketch Timing (seconds):")
    print(f"  phase1 (MinHash): {ds_res['phase1_time']:.3f}")
    print(f"  phase2 (Insert): {ds_res['phase2_time']:.3f}")
    print(f"  phase3 (Query): {ds_res['phase3_time']:.3f}")
    print(f"  Total: {ds_res['total_time']:.3f}")

    print("\nFastSketch Timing (seconds):")
    print(f"  phase1 (Insert): {fs_res['phase1_time']:.3f}")
    print(f"  phase2 (Query): {fs_res['phase2_time']:.3f}")
    print(f"  Total: {fs_res['total_time']:.3f}")

    # 输出统计信息
    print("\nDatasketch Stats:")
    print(f"  Kept: {ds_res['kept_count']}, Removed: {ds_res['removed_count']}")
    print(f"  Total candidates: {ds_res['total_candidates']}")
    print(f"  Avg candidates/query: {ds_res['avg_candidates_per_query']:.2f}")

    print("\nFastSketch Stats:")
    print(f"  Kept: {fs_res['kept_count']}, Removed: {fs_res['removed_count']}")
    print(f"  Total candidates: {fs_res['total_candidates']}")
    print(f"  Avg candidates/query: {fs_res['avg_candidates_per_query']:.2f}")

    # # Run benchmarks
    # rensa_lsh_results = deduplicate_with_rensa_lsh(
    #     pinecone_ds,
    #     NUM_PERM,
    #     SEED,
    #     LSH_THRESHOLD,
    #     NUM_BANDS_RENSA,
    #     FINAL_JACCARD_THRESHOLD,
    #     limit=dataset_limit,
    # )
    #
    # datasketch_lsh_results = deduplicate_with_datasketch_lsh(
    #     pinecone_ds,
    #     NUM_PERM,
    #     LSH_THRESHOLD,
    #     FINAL_JACCARD_THRESHOLD,
    #     limit=dataset_limit,
    # )
    #
    # # Print results
    # print("\n" + "=" * 60)
    # print("LSH BENCHMARK RESULTS")
    # print("=" * 60)
    # original_size = dataset_limit if dataset_limit else len(pinecone_ds)
    # print(f"Original dataset size: {original_size}")
    #
    # print("\nRensa RMinHashLSH:")
    # print(f"  Total Time: {rensa_lsh_results['total_time']:.2f} seconds")
    # print(f"    - MinHash generation: {rensa_lsh_results['phase1_time']:.2f}s")
    # print(f"    - LSH index building: {rensa_lsh_results['phase2_time']:.2f}s")
    # print(f"    - Query & deduplication: {rensa_lsh_results['phase3_time']:.2f}s")
    # print(f"  Rows kept: {rensa_lsh_results['kept_count']}")
    # print(f"  Rows removed: {rensa_lsh_results['removed_count']}")
    # print(
    #     f"  Avg candidates per query: {rensa_lsh_results['avg_candidates_per_query']:.2f}"
    # )
    #
    # print("\nDatasketch MinHashLSH:")
    # print(f"  Total Time: {datasketch_lsh_results['total_time']:.2f} seconds")
    # print(f"    - MinHash generation: {datasketch_lsh_results['phase1_time']:.2f}s")
    # print(f"    - LSH index building: {datasketch_lsh_results['phase2_time']:.2f}s")
    # print(f"    - Query & deduplication: {datasketch_lsh_results['phase3_time']:.2f}s")
    # print(f"  Rows kept: {datasketch_lsh_results['kept_count']}")
    # print(f"  Rows removed: {datasketch_lsh_results['removed_count']}")
    # print(
    #     f"  Avg candidates per query: {datasketch_lsh_results['avg_candidates_per_query']:.2f}"
    # )
    #
    # # Accuracy comparison
    # intersection_kept = len(
    #     rensa_lsh_results["kept_indices"].intersection(
    #         datasketch_lsh_results["kept_indices"]
    #     )
    # )
    # union_kept = len(
    #     rensa_lsh_results["kept_indices"].union(datasketch_lsh_results["kept_indices"])
    # )
    # jaccard_kept_sets = intersection_kept / union_kept if union_kept > 0 else 0.0
    #
    # print("\n" + "=" * 60)
    # print("ACCURACY COMPARISON (Jaccard of Kept Sets)")
    # print("=" * 60)
    # print(
    #     f"Jaccard similarity between Rensa and Datasketch kept sets: {jaccard_kept_sets:.4f}"
    # )
    # print(f"  Intersection size: {intersection_kept}")
    # print(f"  Union size: {union_kept}")
    # print(
    #     f"  Rensa kept: {rensa_lsh_results['kept_count']}, "
    #     f"Datasketch kept: {datasketch_lsh_results['kept_count']}"
    # )
    #
    # # Check if results are identical
    # if jaccard_kept_sets >= 0.99:
    #     print("\n✓ Both algorithms produced NEARLY IDENTICAL deduplication results!")
    # else:
    #     print("\n✗ Algorithms produced DIFFERENT deduplication results.")
    #     # Show some differences
    #     rensa_only = (
    #         rensa_lsh_results["kept_indices"] - datasketch_lsh_results["kept_indices"]
    #     )
    #     datasketch_only = (
    #         datasketch_lsh_results["kept_indices"] - rensa_lsh_results["kept_indices"]
    #     )
    #     print(f"  Documents kept only by Rensa: {len(rensa_only)}")
    #     print(f"  Documents kept only by Datasketch: {len(datasketch_only)}")
    #
    # print("\n" + "=" * 60)
    # print("PERFORMANCE COMPARISON")
    # print("=" * 60)
    #
    # # Overall speedup
    # if datasketch_lsh_results["total_time"] > 0:
    #     overall_speedup = (
    #         datasketch_lsh_results["total_time"] / rensa_lsh_results["total_time"]
    #     )
    #     if overall_speedup > 1:
    #         print(
    #             f"Rensa LSH was {overall_speedup:.2f}x faster overall than Datasketch LSH."
    #         )
    #     else:
    #         print(
    #             f"Datasketch LSH was {1 / overall_speedup:.2f}x faster overall than Rensa LSH."
    #         )
    #
    # # Phase-by-phase comparison
    # print("\nPhase-by-phase speedup (Datasketch time / Rensa time):")
    # phases = ["phase1_time", "phase2_time", "phase3_time"]
    # phase_names = ["MinHash generation", "LSH index building", "Query & deduplication"]
    #
    # for phase, name in zip(phases, phase_names):
    #     if rensa_lsh_results[phase] > 0:
    #         speedup = datasketch_lsh_results[phase] / rensa_lsh_results[phase]
    #         print(f"  {name}: {speedup:.2f}x")
    #
    # # Efficiency comparison
    # print("\nEfficiency metrics:")
    # print(
    #     f"  Rensa average candidates per query: {rensa_lsh_results['avg_candidates_per_query']:.2f}"
    # )
    # print(
    #     f"  Datasketch average candidates per query: {datasketch_lsh_results['avg_candidates_per_query']:.2f}"
    # )
    #
    # if rensa_lsh_results["avg_candidates_per_query"] > 0:
    #     candidate_ratio = (
    #         datasketch_lsh_results["avg_candidates_per_query"]
    #         / rensa_lsh_results["avg_candidates_per_query"]
    #     )
    #     print(f"  Candidate generation ratio: {candidate_ratio:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark LSH deduplication with Rensa and Datasketch."
    )
    # parser.add_argument(
    #     "--limit",
    #     type=int,
    #     default=None,
    #     help="Limit the number of dataset rows to process for faster testing.",
    # )
    parser.add_argument(
        "--num_perm", type=int, default=128, help="Number of permutations for MinHash."
    )
    parser.add_argument(
        "--lsh_threshold",
        type=float,
        default=0.8,
        help="LSH threshold parameter for candidate generation.",
    )
    parser.add_argument(
        "--num_bands",
        type=int,
        default=None,
        help="Number of bands for Rensa LSH (default: calculated optimally based on threshold).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed).",
    )
    parser.add_argument(
        "--final_jaccard_threshold",
        type=float,
        default=0.85,
        help="Final Jaccard similarity threshold for deduplication.",
    )

    cli_args = parser.parse_args()
    run_lsh_benchmark(cli_args)