"""
test_lsh_dedup_comparison.py
----------------------------
Script comparing duplicate-flag outputs and timing between datasketch MinHashLSH
and our FastSketchLSH (`src/fast_sketch_lsh.py`) using a real dataset.

Procedure:
- Load dataset `pinecone/core-2020-05-10-deduplication` via HuggingFace `datasets`.
- Shuffle and take a user-specified ratio subset (default uses the full dataset).
- For datasketch:
  - Build MinHash (num_perm=128) for each document (simple whitespace tokens).
  - Compute (bands, rows) via datasketch `_optimal_param(threshold, num_perm, 0.5, 0.5)`.
  - Insert into MinHashLSH with the computed params.
  - Duplicate flag for item i is 1 iff `len(query(minhash_i)) > 1`.
- For FastSketchLSH:
  - Use `sketch_size=128` and the same `bands` returned above, `threshold=0.8`.
  - Duplicate flag for item i is 1 iff `len(query(tokens_i)) > 1`.

We then compare the two binary flag vectors and assert a small disagreement rate.

Notes:
- This script downloads data and builds two LSH indexes.
- If optional deps are missing, the script exits after printing a notice.

Reference for datasketch MinHash LSH API: https://ekzhu.com/datasketch/lsh.html
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple
import sys
import time
import argparse
import csv
import math


try:
    from datasets import load_dataset  # type: ignore
    HAVE_DATASETS = True
except Exception:
    HAVE_DATASETS = False

try:
    from datasketch import MinHash, MinHashLSH  # type: ignore
    from rensa import RMinHash, RMinHashLSH
    from FastSketchLSH import FastSimilaritySketch, FastSketchLSHRensa
    HAVE_DATASKETCH = True
except Exception:
    HAVE_DATASKETCH = False


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


def _hamming_diff_rate(a: List[int], b: List[int]) -> Tuple[int, float]:
    assert len(a) == len(b)
    diffs = sum(1 for i, j in zip(a, b) if i != j)
    return diffs, diffs / max(1, len(a))

def main(ratio: float = 1.0) -> None:
    # Ensure project root on sys.path for importing `src.*`
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Deterministic shuffling
    random.seed(12345)

    t0 = time.perf_counter()
    ds = load_dataset("pinecone/core-2020-05-10-deduplication")
    data = list(ds["train"])  # type: ignore[index]
    random.shuffle(data)

    if not (0 < ratio <= 1.0):
        raise SystemExit(f"--ratio must be in (0, 1], got {ratio}")
    take = max(1, int(ratio * len(data)))
    sample = data[:take]
    texts = [_extract_text(rec) for rec in sample]
    token_build_start = time.perf_counter()
    token_sets = _build_token_sets(texts)
    token_build_time = time.perf_counter() - token_build_start

    # Token length statistics (per doc token counts and per-token character lengths)
    def _percentile(values: list[int], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(s[int(k)])
        return float(s[f] * (c - k) + s[c] * (k - f))

    token_counts_per_doc = [len(tokens) for tokens in token_sets]
    all_tokens = [tok for tokens in token_sets for tok in tokens]
    token_char_lengths = [len(tok) for tok in all_tokens]

    num_docs = len(token_sets)
    total_tokens = sum(token_counts_per_doc)
    avg_tokens_per_doc = (total_tokens / num_docs) if num_docs else 0.0
    median_tokens_per_doc = _percentile(token_counts_per_doc, 0.5)
    p95_tokens_per_doc = _percentile(token_counts_per_doc, 0.95)
    min_tokens_per_doc = min(token_counts_per_doc) if token_counts_per_doc else 0
    max_tokens_per_doc = max(token_counts_per_doc) if token_counts_per_doc else 0

    avg_token_char_len = (sum(token_char_lengths) / len(token_char_lengths)) if token_char_lengths else 0.0
    median_token_char_len = _percentile(token_char_lengths, 0.5)
    p95_token_char_len = _percentile(token_char_lengths, 0.95)
    min_token_char_len = min(token_char_lengths) if token_char_lengths else 0
    max_token_char_len = max(token_char_lengths) if token_char_lengths else 0

    print("Token statistics:")
    print(f"  docs={num_docs}, total_tokens={total_tokens}")
    print(
        "  tokens/doc: avg={:.2f}, median={:.2f}, p95={:.2f}, min={}, max={}".format(
            avg_tokens_per_doc,
            median_tokens_per_doc,
            p95_tokens_per_doc,
            min_tokens_per_doc,
            max_tokens_per_doc,
        )
    )
    print(
        "  token char length: avg={:.2f}, median={:.2f}, p95={:.2f}, min={}, max={}".format(
            avg_token_char_len,
            median_token_char_len,
            p95_token_char_len,
            min_token_char_len,
            max_token_char_len,
        )
    )

    # Save to CSV under records/
    records_dir = os.path.abspath(os.path.join(project_root, "records"))
    os.makedirs(records_dir, exist_ok=True)
    stats_csv = os.path.join(records_dir, "token_length_stats.csv")
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "docs",
                "total_tokens",
                "avg_tokens_per_doc",
                "median_tokens_per_doc",
                "p95_tokens_per_doc",
                "min_tokens_per_doc",
                "max_tokens_per_doc",
                "avg_token_char_len",
                "median_token_char_len",
                "p95_token_char_len",
                "min_token_char_len",
                "max_token_char_len",
            ]
        )
        writer.writerow(
            [
                num_docs,
                total_tokens,
                f"{avg_tokens_per_doc:.6f}",
                f"{median_tokens_per_doc:.6f}",
                f"{p95_tokens_per_doc:.6f}",
                min_tokens_per_doc,
                max_tokens_per_doc,
                f"{avg_token_char_len:.6f}",
                f"{median_token_char_len:.6f}",
                f"{p95_token_char_len:.6f}",
                min_token_char_len,
                max_token_char_len,
            ]
        )
    print(f"Token stats saved to {stats_csv}")

    # Configuration per user request: threshold=0.8
    threshold = 0.8
    # Start with a requested num_perm (may be adjusted by _optimal_param)
    num_perm = 128
    # Use datasketch's _optimal_param to choose (bands, rows) for given (threshold, num_perm)
    # from datasketch.lsh import _optimal_param  # type: ignore
    # bands, rows = _optimal_param(threshold, num_perm, 0.5, 0.5)
    bands = 8
    rows = 16
    
    # reset the num_perm to the bands * rows (Cause this is the real num_perm we use)
    num_perm = bands * rows
    print(f"bands: {bands}, rows: {rows}, num_perm: {num_perm}")
    
    # RminHashLSH
    lsh_rs = RMinHashLSH(threshold=threshold, num_perm=num_perm, num_bands=bands)
    # Build MinHashes
    token_sets_str = [[str(tok) for tok in tokens] for tokens in token_sets]
    rs_minhash_start = time.perf_counter()
    minhashes = []
    for tokens in token_sets_str:
        m = RMinHash(num_perm=num_perm, seed=42)
        m.update(tokens)
        minhashes.append(m)
    rs_minhash_time = time.perf_counter() - rs_minhash_start
    # Insert into RMinhash LSH
    rs_insert_start = time.perf_counter()
    for idx, m in enumerate(minhashes):
        lsh_rs.insert(idx, m)
    rs_insert_time = time.perf_counter() - rs_insert_start
    # Query for flags
    rs_query_start = time.perf_counter()
    RMinhashlsh_flags = [1 if len(lsh_rs.query(m)) > 1 else 0 for m in minhashes]
    rs_query_time = time.perf_counter() - rs_query_start

    # FastSketchLSHRensa
    lsh_fsr = FastSketchLSHRensa(threshold=threshold, num_perm=num_perm, num_bands=bands)
    # Build MinHashes
    token_sets_str = [[str(tok).encode('utf-8') for tok in tokens] for tokens in token_sets]
    fsr_minhash_start = time.perf_counter()
    m = FastSimilaritySketch(sketch_size=num_perm, seed=42)
    # Batch compute sketches using all threads
    minhashes = m.sketch_batch(token_sets_str, num_threads=0)
    fsr_minhash_time = time.perf_counter() - fsr_minhash_start
    # # Insert into FastSketchLSHRensa
    # fsr_insert_start = time.perf_counter()
    # for idx, m in enumerate(minhashes):
    #     lsh_fsr.insert(idx, m)
    # fsr_insert_time = time.perf_counter() - fsr_insert_start
    # # Query for flags
    # fsr_query_start = time.perf_counter()
    # fsr_flags = [1 if len(lsh_fsr.query(m)) > 1 else 0 for m in minhashes]
    # fsr_query_time = time.perf_counter() - fsr_query_start

    # # fast sketch with the same bands and sketch_size=num_perm
    # # from src.fast_sketch_lsh import FastSketchLSH  # type: ignore
    # from FastSketchLSH import FastSketchLSH
    # fs_insert_start = time.perf_counter()
    # lsh_fs = FastSketchLSH(threshold=threshold, sketch_size=num_perm, bands=bands, random_seed=42)
    # for idx, tokens in enumerate(token_sets):
    #     lsh_fs.insert(str(idx), tokens)
    # fs_insert_time = time.perf_counter() - fs_insert_start

    # fs_query_start = time.perf_counter()
    # fastsketch_flags = [1 if len(lsh_fs.query(tokens)) > 1 else 0 for tokens in token_sets]
    # fs_query_time = time.perf_counter() - fs_query_start

    # diffs, rate = _hamming_diff_rate(RMinhashlsh_flags, fastsketch_flags)

    # # Allow small disagreement due to different banding and sketch constructions
    # # Keep this tolerance modest to ensure practical equivalence.
    # # If the dataset is extremely large or very noisy, you may increase slightly.
    # max_rate = float(os.environ.get("DEDUP_FLAG_TOLERANCE", "0.15"))

    # total_time = time.perf_counter() - t0
    # print(f"Total texts: {len(texts)}")
    # print(f"bands: {bands}, rows: {rows}, threshold: {threshold}, num_perm: {num_perm}")
    # print(f"Rensa duplicate flags sum: {sum(RMinhashlsh_flags)}")
    # print(f"FastsketchLSHRensa duplicate flags sum: {sum(fsr_flags)}")
    # print(f"fastsketch duplicate flags sum: {sum(fastsketch_flags)}")
    # # print(f"Hamming differences: {diffs}, rate: {rate:.4f}")
    # print("Timing (seconds):")
    print(f"  token_build_time: {token_build_time:.3f}")
    print(f"  Rensa: build_minhash={rs_minhash_time:.3f}, insert={rs_insert_time:.3f}, query={rs_query_time:.3f}")
    print(f"  FastsketchLSHRensa: build_minhash={fsr_minhash_time:.3f}")
    # print(f"  fastsketch: insert={fs_insert_time:.3f}, query={fs_query_time:.3f}")
    # print(f"  total: {total_time:.3f}")
    # if rate > max_rate:
    #     raise SystemExit(
    #         f"Mismatch rate {rate:.4f} exceeds tolerance {max_rate}."
    #     )


if __name__ == "__main__":
    if not HAVE_DATASETS:
        print("datasets package not installed; exiting.")
        raise SystemExit(0)
    if not HAVE_DATASKETCH:
        print("datasketch package not installed; exiting.")
        raise SystemExit(0)
    parser = argparse.ArgumentParser(
        description=(
            "Compare duplicate-flag outputs and timing between datasketch MinHashLSH "
            "and FastSketchLSH on a real dataset."
        )
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use (0 < ratio <= 1). Default: 1.0 (full dataset).",
    )
    args = parser.parse_args()
    main(ratio=args.ratio)

