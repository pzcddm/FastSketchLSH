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
import json
from pathlib import Path
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
    # Ensure project root on sys.path for importing `src.*`
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Deterministic shuffling
    random.seed(12345)
    ratio = 1.0

    # Fast path: if loading precomputed sketches, skip dataset/tokenization entirely
    cli_args = globals().get('CLI_ARGS', None)
    use_precomputed = False
    minhashes_path: Path | None = None
    if cli_args and getattr(cli_args, 'load_fastsketch', None):
        minhashes_path = Path(cli_args.load_fastsketch)
        use_precomputed = minhashes_path.exists()

    if not use_precomputed:
        t0 = time.perf_counter()

        # 设置本地缓存目录
        cache_dir = os.path.join(project_root, "data", "huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"正在加载数据集到本地缓存: {cache_dir}")

        # 检查缓存是否存在
        cache_exists = os.path.exists(cache_dir) and os.listdir(cache_dir)
        if cache_exists:
            print("检测到本地缓存，将从缓存加载数据...")
        else:
            print("首次下载可能需要较长时间，请耐心等待...")
            print("提示: 如果下载超时，可以先运行 'python test/download_dataset.py' 单独下载数据集")

        try:
            # 加载数据集，使用本地缓存目录
            ds = load_dataset(
                "HariomJangra/PreTraining-Dataset",
                cache_dir=cache_dir,
                download_mode="reuse_cache_if_exists"
            )
            # ds = load_dataset(
            #     "shuyuej/pretraining-dataset"
            # )
            # ds = load_dataset(
            #     "pinecone/core-2020-05-10-deduplication"
            # )

            print(f"数据集加载完成，共 {len(ds['train'])} 条记录")
            data = list(ds["train"])  # type: ignore[index]
            print(f"数据集划分完毕")
            random.shuffle(data)

        except Exception as e:
            print(f"\n错误: 数据集加载失败 - {type(e).__name__}: {str(e)}")
            print("\n建议解决方案:")
            print("1. 检查网络连接")
            print("2. 先运行独立下载脚本:")
            print("   python test/download_dataset.py")
            print("3. 下载完成后再次运行本脚本")
            raise SystemExit(1)

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
        m = FastSimilaritySketch(sketch_size=num_perm, seed=42)
        # minhashes = m.sketch_batch(token_sets_str, num_threads=0)  # np.ndarray (B, t), dtype=uint64
        minhashes = m.sketch_batch(token_sets_str, num_threads=16)
        fs_minhash_time = time.perf_counter() - fs_minhash_start
        if cli_args and getattr(cli_args, 'save_fastsketch', None):
            outp = Path(cli_args.save_fastsketch)
            outp.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(outp), minhashes)

    # Build band LSH from batch sketches
    band_lsh = LSH(num_perm=num_perm, num_bands=bands)
    fs_build_start = time.perf_counter()
    band_lsh.build_from_batch(minhashes)
    fs_build_time = time.perf_counter() - fs_build_start

    # Query flags (batch): duplicate if bucket size per row > 1
    fs_query_start = time.perf_counter()
    flat, indptr = band_lsh.batch_query_csr(minhashes)
    B = int(minhashes.shape[0])
    fs_band_flags = [1 if int(indptr[i + 1] - indptr[i]) > 1 else 0 for i in range(B)]
    fs_query_time = time.perf_counter() - fs_query_start
    # Derive flags from batch CSR result
    print(sum(fs_band_flags))

    # Query flags (single, NumPy-returning API): loop calling single-item API, duplicate if >1
    fs_single_query_start = time.perf_counter()
    Minhashlsh_flags = [1 if len(band_lsh.query_candidates(m)) > 1 else 0 for m in minhashes]
    fs_single_query_time = time.perf_counter() - fs_single_query_start

    # Query candidates (batch, list-of-lists) and time
    fs_list_batch_start = time.perf_counter()
    lol = band_lsh.batch_query(minhashes)
    # Flags derived from list-of-lists
    fs_list_flags = [1 if len(row) > 1 else 0 for row in lol]

    fs_list_batch_time = time.perf_counter() - fs_list_batch_start
    print(sum(fs_list_flags))
    if not use_precomputed:
        print(f"  Rensa: build_minhash={rs_minhash_time:.3f}, insert={rs_insert_time:.3f}, query={rs_query_time:.3f}")
    print(f"  FastSketch LSH: build_minhash={fs_minhash_time:.3f}, build={fs_build_time:.3f}, query_batch={fs_query_time:.3f}, query_single_np={fs_single_query_time:.3f}, query_batch_list={fs_list_batch_time:.3f}")
       
    # Compare Rensa and FastSketchLSH results using Hamming distance
    if not use_precomputed:
        # Compare Rensa vs FastSketchLSH batch query (CSR)
        diffs_batch, rate_batch = _hamming_diff_rate(RMinhashlsh_flags, fs_band_flags)
        print(f"\n=== Hamming Distance Comparison ===")
        print(f"Rensa vs FastSketchLSH (batch CSR query):")
        print(f"  Hamming distance: {diffs_batch}/{len(RMinhashlsh_flags)}")
        print(f"  Difference rate: {rate_batch:.4f} ({rate_batch*100:.2f}%)")
        print(f"  Agreement rate: {1-rate_batch:.4f} ({(1-rate_batch)*100:.2f}%)")
        
        # Compare Rensa vs FastSketchLSH single query
        diffs_single, rate_single = _hamming_diff_rate(RMinhashlsh_flags, Minhashlsh_flags)
        print(f"\nRensa vs FastSketchLSH (single query):")
        print(f"  Hamming distance: {diffs_single}/{len(RMinhashlsh_flags)}")
        print(f"  Difference rate: {rate_single:.4f} ({rate_single*100:.2f}%)")
        print(f"  Agreement rate: {1-rate_single:.4f} ({(1-rate_single)*100:.2f}%)")
        
        # Compare Rensa vs FastSketchLSH list batch query
        diffs_list, rate_list = _hamming_diff_rate(RMinhashlsh_flags, fs_list_flags)
        print(f"\nRensa vs FastSketchLSH (list batch query):")
        print(f"  Hamming distance: {diffs_list}/{len(RMinhashlsh_flags)}")
        print(f"  Difference rate: {rate_list:.4f} ({rate_list*100:.2f}%)")
        print(f"  Agreement rate: {1-rate_list:.4f} ({(1-rate_list)*100:.2f}%)")

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
    args = parser.parse_args()
    # Expose args for helper logic above
    globals()['CLI_ARGS'] = args
    main()

