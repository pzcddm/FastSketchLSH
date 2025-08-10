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


def run_datasketch_lsh(#师兄实现
        token_sets: List[List[str]],
        threshold: float,
        num_perm: int,
        bands: int,
        rows: int
) -> Dict[str, Any]:
    from datasketch import MinHash, MinHashLSH
    n = len(token_sets)

    # Phase1: Build MinHash
    start1 = time.perf_counter()
    minhashes = []#师兄用的是list
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

    # Phase3: 查询并使用简化策略去重
    start3 = time.perf_counter()
    # 查询所有候选集并计算平均大小
    candidate_sets = [lsh.query(m) for m in minhashes]

    # 使用简化策略去重
    to_remove = set()
    for i in range(n):
        candidates = candidate_sets[i]
        # 过滤掉自身
        other_candidates = [c for c in candidates if c != i]

        if other_candidates:
            # 移除策略：保留ID最小的文档
            min_id = min([i] + other_candidates)
            # 将组内除最小ID外的所有文档加入移除集合
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
    }

def _hamming_diff_rate(a: List[int], b: List[int]) -> Tuple[int, float]:
    assert len(a) == len(b)
    diffs = sum(1 for i, j in zip(a, b) if i != j)
    return diffs, diffs / max(1, len(a))


def run_fastsketch_lsh(
        token_sets: List[List[str]],
        threshold: float,
        num_perm: int,
        bands: int,
        random_seed: int = 42,
        final_jaccard_threshold: float = 0.8  # 添加最终Jaccard阈值参数
) -> Dict[str, Any]:
    from src.fast_sketch_lsh import FastSketchLSH
    n = len(token_sets)
    # 阶段1: 插入
    start1 = time.perf_counter()
    lsh = FastSketchLSH(threshold=threshold, sketch_size=num_perm, bands=bands, random_seed=random_seed)
    for idx, tokens in enumerate(token_sets):
        lsh.insert(idx, tokens)
    phase1_time = time.perf_counter() - start1

    # 阶段2: 查询并计算指标
    start2 = time.perf_counter()
    candidate_sets = [lsh.query(tokens) for tokens in token_sets]
    phase2_time = time.perf_counter() - start2

    # 去重处理 (生成kept_indices和to_remove)
    start3 = time.perf_counter()
    to_remove = set()
    for i in range(n):
        candidates = candidate_sets[i]
        # 过滤掉自身
        other_candidates = [c for c in candidates if c != i]

        if other_candidates:
            # 移除策略：保留ID最小的文档
            min_id = min([i] + other_candidates)
            # 将组内除最小ID外的所有文档加入移除集合
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
    }

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
    take = max(1, int(args.ratio * len(pinecone_ds)))
    sample = pinecone_ds[:take]
    texts = [_extract_text(rec) for rec in sample]
    token_build_start = time.perf_counter()
    token_sets = _build_token_sets(texts)
    token_build_time = time.perf_counter() - token_build_start
    print(f"  token_build_time: {token_build_time:.3f}")

    # User Parameters
    NUM_PERM = args.num_perm
    SEED = args.seed
    LSH_THRESHOLD = args.lsh_threshold

    #先写死
    # Calculate optimal number of bands for fair comparison
    # if args.num_bands:
    #     NUM_BANDS_RENSA = args.num_bands
    # else:
    #     NUM_BANDS_RENSA = calculate_optimal_num_bands(LSH_THRESHOLD, NUM_PERM)
    #     print(
    #         f"\nCalculated optimal num_bands for threshold {LSH_THRESHOLD}: {NUM_BANDS_RENSA}"
    #     )
    from datasketch.lsh import _optimal_param
    bands, rows = _optimal_param(LSH_THRESHOLD, NUM_PERM, 0.5, 0.5)

    # reset the num_perm to the bands * rows (Cause this is the real num_perm we use)
    num_perm = bands * rows
    print(f"bands: {bands}, rows: {rows}, num_perm: {num_perm}")

    # 运行Datasketch LSH
    ds_res = run_datasketch_lsh(token_sets, LSH_THRESHOLD, num_perm, bands, rows)
    # 运行FastSketch LSH
    fs_res = run_fastsketch_lsh(token_sets, LSH_THRESHOLD, num_perm, bands, SEED)


    print("\nDatasketch MinHashLSH:")
    print(f"  Total Time: {ds_res['total_time']:.2f} seconds")
    print(f"    - MinHash generation: {ds_res['phase1_time']:.2f}s")
    print(f"    - LSH index building: {ds_res['phase2_time']:.2f}s")
    print(f"    - Query & deduplication: {ds_res['phase3_time']:.2f}s")
    print(f"  Rows kept: {ds_res['kept_count']}")
    print(f"  Rows removed: {ds_res['removed_count']}")

    print("\n" + "=" * 60)
    print("LSH BENCHMARK RESULTS")
    print("\nFastSketchLSH:")
    print(f"  Total Time: {fs_res['total_time']:.2f} seconds")
    print(f"    - MinHash generation: {fs_res['phase1_time']:.2f}s")
    print(f"    - LSH index building: {fs_res['phase2_time']:.2f}s")
    print(f"    - Query & deduplication: {fs_res['phase3_time']:.2f}s")
    print(f"  Rows kept: {fs_res['kept_count']}")
    print(f"  Rows removed: {fs_res['removed_count']}")

    # Accuracy comparison
    intersection_kept = len(
        ds_res["kept_indices"].intersection(
            fs_res["kept_indices"]
        )
    )
    union_kept = len(
        ds_res["kept_indices"].union(fs_res["kept_indices"])
    )
    jaccard_kept_sets = intersection_kept / union_kept if union_kept > 0 else 0.0

    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON (Jaccard of Kept Sets)")
    print(
        f"Jaccard similarity between FastSketchLSH and Datasketch kept sets: {jaccard_kept_sets:.4f}"
    )
    print(f"  Intersection size: {intersection_kept}")
    print(f"  Union size: {union_kept}")
    print(
        f"  FastSketchLSH kept: {fs_res['kept_count']}, "
        f"Datasketch kept: {ds_res['kept_count']}"
    )

    # Check if results are identical
    if jaccard_kept_sets >= 0.99:
        print("\n✓ Both algorithms produced NEARLY IDENTICAL deduplication results!")
    else:
        print("\n✗ Algorithms produced DIFFERENT deduplication results.")
        # Show some differences
        FastSketchLSH_only = (
            fs_res["kept_indices"] - ds_res["kept_indices"]
        )
        datasketch_only = (
            ds_res["kept_indices"] - fs_res["kept_indices"]
        )
        print(f"  Documents kept only by FastSketchLSH: {len(FastSketchLSH_only)}")
        print(f"  Documents kept only by Datasketch: {len(datasketch_only)}")

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    # Overall speedup
    if ds_res["total_time"] > 0:
        overall_speedup = (
            ds_res["total_time"] / fs_res["total_time"]
        )
        if overall_speedup > 1:
            print(
                f"FastSketchLSH was {overall_speedup:.2f}x faster overall than Datasketch LSH."
            )
        else:
            print(
                f"Datasketch LSH was {1 / overall_speedup:.2f}x faster overall than FastSketchLSH."
            )

    #rensa原作额外的统计信息：
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
        description="Benchmark LSH deduplication with FastSketchLSH and Datasketch."
    )
    # parser.add_argument(
    #     "--limit",
    #     type=int,
    #     default=None,
    #     help="Limit the number of dataset rows to process for faster testing.",
    # )
    # 使用的是jaccard，而不是hamming
    # 使用了 LSH_t，暂无使用 FINAL_JACCARD_THRESHOLD
    # 暂不支持用户自定义 NUM_BANDS
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
        help="Number of bands for LSH (default: calculated optimally based on threshold).",
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
        default=0.8,
        help="Final Jaccard similarity threshold for deduplication.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use (0 < ratio <= 1). Default: 1.0 (full dataset).",
    )
    cli_args = parser.parse_args()
    run_lsh_benchmark(cli_args)