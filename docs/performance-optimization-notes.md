# Performance Optimization Notes

## Status

This file is the current source of truth for performance status in this repo.

- Strict fair end-to-end benchmark: `python -m exps.end2end.run` with default settings and without `--use-prehashed-csr`
- Primary catch-up plan: `docs/fair-benchmark-sketch-plan.md`

## Current Fair Benchmark Definition

Use this definition unless a document explicitly says otherwise:

- Input to both engines: the same `token_sets`
- Counted engine time: `sketch`, `build`, `query`
- Excluded time: dataset download, text extraction, tokenization, n-gram generation
- FastSketch default: do not use `--use-prehashed-csr`
- Rensa default: matrix / one-shot APIs when available
- Thread control: `--threads N` must limit BOTH engines (FastSketch via `num_threads`, Rensa via `RAYON_NUM_THREADS`)

Why this is the current fair definition:

- Rensa's own benchmark counts token hashing inside `sketch`
- Therefore FastSketch must also hash tokens during `sketch` in the default comparison
- Rensa uses Rayon for internal parallelism; the benchmark must control this via `RAYON_NUM_THREADS` to prevent unfair multi-threading when `--threads 1` is specified

## Current Fair Baseline

Measured on 2026-03-13 with corrected thread control and thread-aware sketch dispatch, 3-5 run medians:

### Single-thread (`--threads 1`)

| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) |
|---|---:|---:|---:|---:|---:|
| PINECONE | fastsketch | 0.515 | 0.093 | 0.036 | 0.644 |
| PINECONE | rensa | 0.879 | 0.016 | 0.000 | 0.895 |
| SHUYUEJ | fastsketch | 0.363 | 0.032 | 0.012 | 0.407 |
| SHUYUEJ | rensa | 0.906 | 0.005 | 0.000 | 0.911 |

### Auto-threads (default, no `--threads`)

| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) |
|---|---:|---:|---:|---:|---:|
| PINECONE | fastsketch | 0.155 | 0.023 | 0.007 | 0.185 |
| PINECONE | rensa | 0.337 | 0.015 | 0.000 | 0.352 |
| SHUYUEJ | fastsketch | 0.115 | 0.007 | 0.002 | 0.124 |
| SHUYUEJ | rensa | 0.293 | 0.005 | 0.000 | 0.298 |

Current conclusion:

- FastSketch is faster than Rensa in all configurations
- Single-thread: FastSketch is 1.39x faster on PINECONE and 2.24x faster on SHUYUEJ
- Auto-threads: FastSketch is 1.90x faster on PINECONE and 2.40x faster on SHUYUEJ
- The key optimization was thread-aware dispatch in `sketch_batch`: multi-thread now uses the ptrs/lengths path where `sketch_batch_flat_bytes` parallelizes both hashing and sketching via OpenMP

## Analysis of Rensa's "11.92x faster" Benchmark Claim

Rensa's GitHub README (https://github.com/beowolx/rensa) claims "11.92x faster than FastSketch"
based on their `benchmarks/full_benchmark.py` suite. The numbers behind this claim: Rensa total
0.118s vs FastSketch total 1.411s (0.118 * 11.92 = 1.407 ≈ 1.411). Below is a detailed analysis
of why this claim was misleading and what the corrected comparison shows.

### What Rensa's benchmark measures

- 7 datasets: FineFineWeb (200K rows), CodeFeedback (157K), AG News (120K), Pinecone (100K), ShuyueJ (38K), BookCorpusOpen (1K), Books3 (1K)
- 2 thread configurations: `threads=1` and `threads=8`
- 128 permutations, threshold 0.8, 8 bands
- Process-isolated runs with randomized engine order
- Phases timed: `sketch` (via `sketch_batch`), `build` (via `build_from_batch`), `query` (via `batch_query_csr`)
- Thread env vars set for all engines: `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, etc.

### Three issues that inflated Rensa's speedup claim

**Issue 1: `sketch_batch` string path was accidentally single-threaded (the dominant factor)**

The chunked fused hash+sketch path added to `sketch_batch` in `init.cpp` called
`sketch_one_prehashed()` in a serial for-loop after releasing the GIL:

```cpp
// This loop ran serially regardless of num_threads!
py::gil_scoped_release release;
for (size_t i = 0; i < csz; ++i) {
    self.sketch_one_prehashed(chunk_hashes.data() + ci_start, ci_n,
                              chunk_out + i * t);
}
```

The `num_threads` parameter was completely ignored. So even when Rensa's benchmark called
`sketch_batch(token_sets, num_threads=8)`, FastSketch sketched single-threaded while Rensa's
Rayon used all 8 threads. Since sketching dominates end-to-end time, this alone explains most
of the 11.92x gap.

**Fix:** Thread-aware dispatch in `sketch_batch` (`init.cpp`):
- `num_threads == 1`: uses the chunked fused path (best single-thread cache behavior)
- `num_threads != 1`: falls through to the ptrs/lengths extraction path, which calls
  `sketch_batch_flat_bytes`. That function parallelizes BOTH token hashing AND the sketch
  kernel across all OpenMP threads. GIL-held work is limited to cheap pointer extraction.

**Issue 2: Rensa's benchmark uses `batch_query_csr` instead of `batch_query_duplicate_flags`**

Their benchmark queries via:
```python
flat, indptr = lsh.batch_query_csr(sketches)
duplicate_flags = [int(indptr[i+1] - indptr[i]) > 1 for i in range(row_count)]
```

This materializes full candidate lists and builds Python flag lists. FastSketch now has
`batch_query_duplicate_flags` which returns a compact uint8 flag array with early exit
per row, avoiding candidate materialization entirely.

**Issue 3: 73% of single-threaded sketch time is in GIL-held string hashing**

Profiling the single-threaded sketch path showed:
- String hashing (GIL-held): ~0.334s out of 0.46s total (73%)
- Sketch kernel (GIL-released): ~0.126s (27%)

The chunked path hashed tokens under GIL then ran only the sketch kernel in parallel.
Since hashing dominates, multi-threaded speedup was capped at ~1.3x even with 8 threads.
The corrected ptrs/lengths path moves hashing into the OpenMP parallel region, giving
near-linear scaling.

### Corrected comparison (PINECONE, 100K docs, this machine)

| Mode | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) |
|---|---|---:|---:|---:|---:|
| Auto-threads | FastSketch | 0.155 | 0.023 | 0.007 | **0.185** |
| Auto-threads | Rensa | 0.337 | 0.015 | 0.000 | **0.352** |
| Single-thread | FastSketch | 0.515 | 0.093 | 0.036 | **0.644** |
| Single-thread | Rensa | 0.879 | 0.016 | 0.000 | **0.895** |

| Mode | FastSketch speedup |
|---|---|
| Auto-threads | **1.90x faster** than Rensa |
| Single-thread | **1.39x faster** than Rensa |

### Summary

Rensa's "11.92x faster" claim was based on benchmarking a FastSketch build where the
multi-threaded string sketch path was broken (serial loop ignoring `num_threads`). After
fixing the thread-aware dispatch, FastSketch is 1.4x–2.4x faster than Rensa across all
tested configurations.

## Already Implemented

These optimizations already exist. Any new optimization pass must inspect them before redoing work:

1. LSH duplicate-flag fast path
- C++: `fastsketchlsh_ext/include/LSH.h`
- C++: `fastsketchlsh_ext/cpp/LSH.cpp`
- Pybind: `fastsketchlsh_ext/cpp/init.cpp`
- Python use site: `exps/end2end/fastsketch_deduplicator.py`

2. Pre-hashed sketch APIs
- `sketch_prehashed(...)`
- `sketch_batch_prehashed(...)`
- `sketch_batch_flat_csr_prehashed(...)`
- Main bindings: `fastsketchlsh_ext/cpp/init.cpp`
- Kernel implementation: `fastsketchlsh_ext/cpp/fastsketch.cpp`
- These interfaces exist, but they are not part of the current strict fair benchmark target

3. Optional pre-hashed dataset cache
- `exps/end2end/util.py`
- `exps/end2end/run.py`
- This is now opt-in only via `--use-prehashed-csr`

4. Fair Rensa adapter alignment
- `exps/end2end/rensa_deduplicator.py`
- Uses matrix / one-shot APIs when available so the comparison is honest
- Now respects `num_threads` via `RAYON_NUM_THREADS` environment variable

5. Validation tests
- `test/test_lsh_duplicate_flags_fastpath.py`
- `test/test_prehashed_consistency.py`

6. Default hash switched to fxhash64
- fxhash64 is now the default token prehash (faster 8-byte stride)
- FNV1a64 available via `FASTSKETCH_USE_FNV1A=1` build flag for compatibility
- Build: `fastsketchlsh_ext/setup.py`, `fastsketchlsh_ext/CmakeLists.txt`

7. Dedicated `sketch_batch_str_lists` binding
- Optimized `list[list[str]]` path: hashes tokens inline then runs CSR kernel
- Avoids intermediate ptr/len arrays
- Available as a library API; the end-to-end adapter uses `sketch_batch` instead (see item 9)

8. `sketch_kernel_direct` + OpenMP single-thread bypass
- Leaner sketch kernel that reads directly from prehashed buffer
- `sketch_batch_flat_csr_prehashed` bypasses OpenMP team creation when threads=1

9. Thread-aware dispatch in `sketch_batch` for string input
- `num_threads == 1`: uses chunked fused path (hash under GIL + serial sketch kernel, best cache locality)
- `num_threads != 1`: falls through to ptrs/lengths extraction under GIL, then `sketch_batch_flat_bytes` which parallelizes both hashing and sketching via OpenMP
- This is the fix for the bug that caused Rensa's "11.92x faster" claim
- Files: `fastsketchlsh_ext/cpp/init.cpp` (dispatch gate), `fastsketchlsh_ext/cpp/fastsketch.cpp` (`sketch_batch_flat_bytes`)
- Used by `exps/end2end/fastsketch_deduplicator.py`

## Remaining Optimization Focus

The next pass should assume the items above already exist and should not re-implement them.

- Keep the benchmark strict and reproducible
- Profile the current `sketch` path into sub-stages before changing it again
- Focus on remaining native batch overhead such as scratch-buffer reuse and matrix representation
- Treat LSH query as maintenance work unless profiling shows a regression

## Documentation Rules

When updating benchmark numbers or optimization status:

- Update this file first
- If the change affects the strict fair benchmark, also update `docs/fair-benchmark-sketch-plan.md`
- Always specify whether thread counts were controlled for BOTH engines
