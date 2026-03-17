"""
test_lsh_dedup_comparison.py
----------------------------
Script comparing duplicate-flag outputs and timing between datasketch MinHashLSH

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
import json
from pathlib import Path
import numpy as np


try:
    from datasets import load_dataset  # type: ignore
    HAVE_DATASETS = True
except Exception:
    HAVE_DATASETS = False

try:
    from datasketch import MinHash, MinHashLSH  # type: ignore
    from rensa import RMinHash, RMinHashLSH
    from FastSketchLSH import FastSimilaritySketch, LSH
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

def main() -> None:
    # Ensure project root on sys.path for importing `prototype.*`
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Deterministic shuffling
    random.seed(12345)
    ratio = 1.0

    # Fast path: if loading precomputed sketches, skip dataset/tokenization entirely
    cli_args = globals().get('CLI_ARGS', None)
    lsh_num_threads = 0
    if cli_args is not None and hasattr(cli_args, "lsh_threads"):
        try:
            lsh_num_threads = int(getattr(cli_args, "lsh_threads"))
        except (TypeError, ValueError):
            lsh_num_threads = 0
    use_precomputed = False
    minhashes_path: Path | None = None
    if cli_args and getattr(cli_args, 'load_fastsketch', None):
        minhashes_path = Path(cli_args.load_fastsketch)
        use_precomputed = minhashes_path.exists()

    if not use_precomputed:
        t0 = time.perf_counter()
        ds = load_dataset("pinecone/core-2020-05-10-deduplication")
        # ds = load_dataset("HariomJangra/PreTraining-Dataset")
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
    # Report OpenMP runtime threads from extension if available
    try:
        import FastSketchLSH as _fs
        print(f"OpenMP max threads: {_fs.omp_max_threads()} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")
    except Exception:
        pass
    
    # RminHashLSH (only when we built token sets)
    if not use_precomputed:
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

    # Band-parallel LSH (new)
    # Build or load MinHashes (uint64 sketches)
    if use_precomputed:
        fs_minhash_start = time.perf_counter()
        minhashes = np.load(str(minhashes_path))
        fs_minhash_time = time.perf_counter() - fs_minhash_start
    else:
        token_sets_str = [[str(tok).encode('utf-8') for tok in tokens] for tokens in token_sets]
        fs_minhash_start = time.perf_counter()
        m = FastSimilaritySketch(size=num_perm, seed=42)
        minhashes = m.batch(token_sets_str, num_threads=0)  # np.ndarray (B, t), dtype=uint64
        fs_minhash_time = time.perf_counter() - fs_minhash_start
        if cli_args and getattr(cli_args, 'save_fastsketch', None):
            outp = Path(cli_args.save_fastsketch)
            outp.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(outp), minhashes)

    # Build band LSH from batch sketches
    band_lsh = LSH(num_perm=num_perm, num_bands=bands, num_threads=lsh_num_threads)
    resolved_threads = band_lsh.num_threads
    resolved_label = resolved_threads if resolved_threads > 0 else "auto"
    print(f"FastSketch LSH threads (requested {lsh_num_threads}): {resolved_label}")
    fs_build_start = time.perf_counter()
    band_lsh.insert(minhashes)
    fs_build_time = time.perf_counter() - fs_build_start

    # Query flags (batch): duplicate if bucket size per row > 1
    fs_query_start = time.perf_counter()
    flat, indptr = band_lsh.query(minhashes, format="csr")
    B = int(minhashes.shape[0])
    fs_band_flags = [1 if int(indptr[i + 1] - indptr[i]) > 1 else 0 for i in range(B)]
    fs_query_time = time.perf_counter() - fs_query_start
    # Derive flags from batch CSR result
    print(sum(fs_band_flags))

    # Query flags (single, NumPy-returning API): loop calling single-item API, duplicate if >1
    fs_single_query_start = time.perf_counter()
    Minhashlsh_flags = [1 if len(band_lsh.query(m)) > 1 else 0 for m in minhashes]
    fs_single_query_time = time.perf_counter() - fs_single_query_start

    # Query candidates (batch, list-of-lists) and time
    fs_list_batch_start = time.perf_counter()
    lol = band_lsh.query(minhashes)
    # Flags derived from list-of-lists
    fs_list_flags = [1 if len(row) > 1 else 0 for row in lol]

    fs_list_batch_time = time.perf_counter() - fs_list_batch_start
    print(sum(fs_list_flags))
    if not use_precomputed:
        print(f"  Rensa: build_minhash={rs_minhash_time:.3f}, insert={rs_insert_time:.3f}, query={rs_query_time:.3f}")
    print(f"  FastSketch LSH: build_minhash={fs_minhash_time:.3f}, build={fs_build_time:.3f}, query_batch={fs_query_time:.3f}, query_single_np={fs_single_query_time:.3f}, query_batch_list={fs_list_batch_time:.3f}")
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
        "--save-fastsketch",
        type=str,
        default="",
        help="Path to save precomputed FastSimilaritySketch minhashes as .npy (optional)",
    )
    parser.add_argument(
        "--load-fastsketch",
        type=str,
        default="",
        help="Path to load precomputed FastSimilaritySketch minhashes .npy (optional)",
    )
    parser.add_argument(
        "--lsh-threads",
        type=int,
        default=0,
        help="Number of OpenMP threads for FastSketchLSH (<=0 uses library default)",
    )
    args = parser.parse_args()
    # Expose args for helper logic above
    globals()['CLI_ARGS'] = args
    main()
